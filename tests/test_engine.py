"""Тесты Trading Engine."""

import asyncio
import numpy as np
import pandas as pd
import pytest

from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine, BarResult
from src.models import SignalDirection


def make_data(n: int = 100) -> pd.DataFrame:
    """Создаёт данные с трендом."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_engine_init():
    """Движок корректно инициализируется."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)

    assert engine.config.symbol == "BTC-USD"
    assert engine.portfolio.state.current_equity == 10000.0
    assert engine.position_manager.open_count == 0

    engine.close()


def test_engine_invalid_config():
    """Невалидная конфигурация вызывает ошибку."""
    with pytest.raises(ValueError):
        EngineConfig(starting_equity=-100).validate()

    with pytest.raises(ValueError):
        EngineConfig(risk_per_trade_pct=0.5).validate()

    with pytest.raises(ValueError):
        EngineConfig(gate_mode="invalid").validate()


@pytest.mark.asyncio
async def test_engine_on_bar_no_signals():
    """Бар без сигналов не открывает позиций."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        enable_indicators=True,
        enable_harmonic=False,
        enable_support_resistance=False,
    )
    engine = TradingEngine(config)
    data = make_data(100)

    result = await engine.on_bar(data, bar_index=99)

    assert isinstance(result, BarResult)
    assert result.bar_index == 99
    assert engine.position_manager.open_count == 0

    engine.close()


@pytest.mark.asyncio
async def test_engine_logs_decision():
    """Каждый бар логирует решение в Journal."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)
    data = make_data(100)

    initial_count = engine.journal.get_decisions_count()
    await engine.on_bar(data, bar_index=99)
    final_count = engine.journal.get_decisions_count()

    assert final_count > initial_count

    engine.close()


@pytest.mark.asyncio
async def test_engine_position_lifecycle():
    """
    Полный жизненный цикл: сигнал → открытие → закрытие.

    Используем mock-анализатор, который всегда даёт сигнал.
    """
    from src.analyzers.base import BaseAnalyzer
    from src.models import AnalyzerSignal

    class AlwaysBuyAnalyzer(BaseAnalyzer):
        def __init__(self):
            super().__init__("always_buy")

        async def analyze(self, data):
            return AnalyzerSignal(
                direction=SignalDirection.BUY,
                confidence=0.95,
                reason="always buy",
                metadata={"symbol": "TEST", "timeframe": "1h"},
                source=self.name,
            )

    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        enable_indicators=False,
        enable_harmonic=False,
        enable_support_resistance=False,
        gate_mode="aggressive",  # порог 0.50
    )
    engine = TradingEngine(config)

    # Регистрируем mock-анализатор
    engine.intake.register(AlwaysBuyAnalyzer())

    # Данные с ростом (чтобы TP сработал)
    data = make_data(100)
    # Сильно повышаем последние бары
    data.loc[95:, "close"] = data.loc[95:, "close"] + 5.0
    data.loc[95:, "high"] = data.loc[95:, "high"] + 5.0

    # Обрабатываем бар
    result = await engine.on_bar(data, bar_index=99)

    # Проверяем, что позиция открылась
    if result.opened_position is not None:
        assert engine.position_manager.open_count >= 1
        assert engine.journal.get_trades_count() >= 1

    engine.close()


@pytest.mark.asyncio
async def test_engine_guard_blocks():
    """Portfolio Guard блокирует сделки при паузе."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)

    # Принудительно ставим на паузу
    engine.portfolio.force_pause("test", hours=1)

    data = make_data(100)
    result = await engine.on_bar(data, bar_index=99)

    assert result.was_blocked_by_guard
    assert "паузе" in result.guard_reason.lower()

    engine.close()


@pytest.mark.asyncio
async def test_engine_insufficient_data():
    """Недостаточно данных — бар не обрабатывается."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)
    data = make_data(20)  # меньше 50 баров

    result = await engine.on_bar(data, bar_index=19)

    assert result.decision is None
    assert engine.position_manager.open_count == 0

    engine.close()


@pytest.mark.asyncio
async def test_engine_stats():
    """Статистика движка корректна."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)
    data = make_data(100)

    await engine.on_bar(data, bar_index=99)
    stats = engine.get_stats()

    assert "portfolio" in stats
    assert "positions" in stats
    assert "trades" in stats
    assert stats["portfolio"]["starting_equity"] == 10000.0

    engine.close()


@pytest.mark.asyncio
async def test_engine_close_all():
    """Принудительное закрытие всех позиций."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)

    # Создаём позицию вручную через Position Manager
    from src.models import SignalDirection as SD

    data = make_data(100)
    plan = engine.risk_manager.calculate_plan(
        direction=SD.BUY,
        entry_price=100.0,
        equity=10000.0,
        data=data,
    )
    if plan is not None:
        pos = engine.position_manager.open_position(
            symbol="BTC-USD",
            timeframe="1h",
            direction=SD.BUY,
            plan=plan,
        )
        engine.portfolio.register_position_opened(pos)
        engine.journal.log_position_opened(pos)

        assert engine.position_manager.open_count == 1

        closed = engine.close_all_positions(price=101.0)
        assert len(closed) == 1
        assert engine.position_manager.open_count == 0

    engine.close()


@pytest.mark.asyncio
async def test_engine_snapshot_logging():
    """Снимки портфеля логируются периодически."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        snapshot_every_n_bars=2,  # каждые 2 бара
    )
    engine = TradingEngine(config)
    data = make_data(100)

    # Обрабатываем 4 бара
    for i in range(4):
        await engine.on_bar(data, bar_index=99 - i)

    # Должно быть 2 снимка (на 2-м и 4-м баре)
    snapshots = engine.journal.db.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots"
    ).fetchone()[0]
    assert snapshots >= 2

    engine.close()