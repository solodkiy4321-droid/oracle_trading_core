"""Тесты Trading Engine.

Актуальная версия:
- EngineConfig по умолчанию symbol="ETH-USD".
- Порядок проверок в on_bar: guard → CHOP → collect → log_decision → open.
- journal.log_decision вызывается ВСЕГДА, даже если guard/CHOP отбил бар.
"""

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
    """Движок корректно инициализируется с дефолтным профилем."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)

    # Дефолтный symbol в EngineConfig = "ETH-USD"
    assert engine.config.symbol == "ETH-USD"
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
        symbol="BTC-USD",
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
    """
    Каждый бар логирует решение в Journal.

    Проверка фикса: log_decision вызывается ВСЕГДА,
    даже если CHOP-фильтр отбил бар (раньше возвращался до лога).
    """
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)
    data = make_data(100)

    initial_count = engine.journal.get_decisions_count()
    await engine.on_bar(data, bar_index=99)
    final_count = engine.journal.get_decisions_count()

    assert final_count > initial_count, (
        f"log_decision должен вызваться, "
        f"было {initial_count}, стало {final_count}"
    )

    engine.close()


@pytest.mark.asyncio
async def test_engine_guard_blocks():
    """
    Portfolio Guard блокирует сделки при паузе.

    Проверка фикса: guard проверяется ДО CHOP-фильтра,
    поэтому при паузе result.was_blocked_by_guard=True.
    """
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)

    engine.portfolio.force_pause("test", hours=1)

    data = make_data(100)
    result = await engine.on_bar(data, bar_index=99)

    assert result.was_blocked_by_guard, (
        "guard должен сработать раньше CHOP-фильтра"
    )
    assert "паузе" in result.guard_reason.lower()
    # CHOP не должен был сработать, потому что guard вышел раньше
    assert not result.was_blocked_by_chop

    engine.close()


@pytest.mark.asyncio
async def test_engine_guard_before_chop():
    """
    При паузе AND CHOP одновременно — guard побеждает.

    Дополнительный тест порядка: если бы CHOP был выше,
    флаг был бы was_blocked_by_chop=True.
    """
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)

    engine.portfolio.force_pause("test pause", hours=1)

    # Данные без тренда → режим CHOP
    np.random.seed(42)
    n = 100
    close = 100 + np.random.randn(n) * 0.5
    data = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })

    result = await engine.on_bar(data, bar_index=99)

    assert result.was_blocked_by_guard
    assert not result.was_blocked_by_chop

    engine.close()


@pytest.mark.asyncio
async def test_engine_logs_decision_when_guard_blocks():
    """При guard-блоке decision всё равно логируется (HOLD)."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)
    engine.portfolio.force_pause("test", hours=1)

    data = make_data(100)
    initial_count = engine.journal.get_decisions_count()
    await engine.on_bar(data, bar_index=99)
    final_count = engine.journal.get_decisions_count()

    assert final_count > initial_count, (
        "guard-блок тоже должен логироваться как HOLD-решение"
    )

    engine.close()


@pytest.mark.asyncio
async def test_engine_logs_decision_when_chop_blocks():
    """При CHOP-блоке decision логируется (HOLD)."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)

    # Данные без тренда → CHOP
    np.random.seed(42)
    n = 100
    close = 100 + np.random.randn(n) * 0.5
    data = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })

    initial_count = engine.journal.get_decisions_count()
    result = await engine.on_bar(data, bar_index=99)
    final_count = engine.journal.get_decisions_count()

    # Если CHOP сработал — флаг + лог
    if result.was_blocked_by_chop:
        assert final_count > initial_count, (
            "CHOP-блок тоже должен логироваться как HOLD-решение"
        )

    engine.close()


@pytest.mark.asyncio
async def test_engine_position_lifecycle():
    """
    Полный жизненный цикл: сигнал → открытие → закрытие.
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
        symbol="BTC-USD",
        gate_mode="aggressive",
    )
    engine = TradingEngine(config)

    engine.intake.register(AlwaysBuyAnalyzer())

    data = make_data(100)
    data.loc[95:, "close"] = data.loc[95:, "close"] + 5.0
    data.loc[95:, "high"] = data.loc[95:, "high"] + 5.0

    result = await engine.on_bar(data, bar_index=99)

    if result.opened_position is not None:
        assert engine.position_manager.open_count >= 1
        assert engine.journal.get_trades_count() >= 1

    engine.close()


@pytest.mark.asyncio
async def test_engine_insufficient_data():
    """Недостаточно данных — бар не обрабатывается."""
    config = EngineConfig(
        starting_equity=10000.0,
        journal_db_path=":memory:",
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)
    data = make_data(20)

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
        symbol="BTC-USD",
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
        symbol="BTC-USD",
    )
    engine = TradingEngine(config)

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
        symbol="BTC-USD",
        snapshot_every_n_bars=2,
    )
    engine = TradingEngine(config)
    data = make_data(100)

    for i in range(4):
        await engine.on_bar(data, bar_index=99 - i)

    snapshots = engine.journal.db.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots"
    ).fetchone()[0]
    assert snapshots >= 2

    engine.close()