"""Тесты для Бага B: MeanReversionAnalyzer и SignalCollector.

Проверяют:
- MeanReversionAnalyzer.analyze принимает indicators и bar_index.
- SignalCollector._run_one вызывает MeanReversion без TypeError.
- MeanReversion не попадает в gather-exceptions.
- Сигнатура поддерживает inspect.signature без ошибок.
"""

import asyncio
import inspect

import numpy as np
import pandas as pd
import pytest

from src.analyzers.mean_reversion import MeanReversionAnalyzer
from src.analyzers.trend import TrendAnalyzer
from src.signal_intake.collector import (
    SignalCollector,
    _supports_extended_interface,
)


# ---------- Синтетические данные ----------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Генерирует боковой (пилообразный) OHLCV — MeanReversion должен находить сигналы."""
    rng = np.random.default_rng(seed)
    # Боковик: цена колеблется вокруг 100 без тренда
    close = 100 + np.sin(np.arange(n) / 8.0) * 3 + rng.normal(0, 0.3, n)
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------- Сигнатура ----------

def test_mean_reversion_analyze_signature_has_indicators_and_bar_index():
    """
    Проверяем, что analyze принимает indicators и bar_index.
    Это ключевой фикс бага B.
    """
    sig = inspect.signature(MeanReversionAnalyzer.analyze)
    params = sig.parameters
    assert "indicators" in params, "analyze() должен принимать indicators"
    assert "bar_index" in params, "analyze() должен принимать bar_index"


def test_supports_extended_interface_mean_reversion():
    """Хелпер _supports_extended_interface должен вернуть True."""
    analyzer = MeanReversionAnalyzer()
    assert _supports_extended_interface(analyzer) is True


def test_supports_extended_interface_trend():
    """TrendAnalyzer тоже поддерживает расширенный интерфейс."""
    analyzer = TrendAnalyzer()
    assert _supports_extended_interface(analyzer) is True


# ---------- Прямой вызов analyze ----------

@pytest.mark.asyncio
async def test_mean_reversion_analyze_accepts_extended_kwargs():
    """Прямой вызов с indicators=None, bar_index=None не падает."""
    analyzer = MeanReversionAnalyzer()
    data = _make_ohlcv()
    result = await analyzer.analyze(data, indicators=None, bar_index=None)
    # Результат может быть None или AnalyzerSignal — главное, не TypeError
    assert result is None or hasattr(result, "direction")


@pytest.mark.asyncio
async def test_mean_reversion_analyze_accepts_dummy_cache():
    """Передаём фиктивный cache-объект — не должно быть ошибок."""
    analyzer = MeanReversionAnalyzer()
    data = _make_ohlcv()

    class DummyCache:
        n = 300

        def slice(self, i):
            return {}

        def get_swing_points(self, end_idx):
            return []

    result = await analyzer.analyze(data, indicators=DummyCache(), bar_index=299)
    assert result is None or hasattr(result, "direction")


# ---------- SignalCollector ----------

@pytest.mark.asyncio
async def test_collector_calls_mean_reversion_without_gather_exception():
    """
    Регистрируем MeanReversionAnalyzer в SignalCollector.
    Прогоняем collect_all.
    Убеждаемся, что НЕ было необработанных исключений в gather.
    """
    collector = SignalCollector(timeout=5.0)
    collector.register(MeanReversionAnalyzer())

    data = _make_ohlcv()
    signals, rejected, stats = await collector.collect_all(data)

    # Ключевая проверка: failed=0 — нет исключений
    assert stats["failed"] == 0, (
        f"MeanReversion должен работать без exceptions, "
        f"но failed={stats['failed']}, rejected={rejected}"
    )
    # В rejected не должно быть записей от mean_reversion по причине TypeError
    for r in rejected:
        if r.source == "mean_reversion":
            assert "TypeError" not in r.reason, (
                f"Получен TypeError от MeanReversion: {r.reason}"
            )


@pytest.mark.asyncio
async def test_collector_mixed_analyzers_all_work():
    """
    Регистрируем MeanReversion + Trend.
    Оба должны отработать без исключений.
    """
    collector = SignalCollector(timeout=5.0)
    collector.register(MeanReversionAnalyzer())
    collector.register(TrendAnalyzer())

    data = _make_ohlcv(n=500)
    signals, rejected, stats = await collector.collect_all(data)

    assert stats["total"] == 2
    assert stats["failed"] == 0, (
        f"Оба анализатора должны работать, failed={stats['failed']}, "
        f"rejected={[(r.source, r.reason) for r in rejected]}"
    )


@pytest.mark.asyncio
async def test_collector_real_exception_still_caught():
    """
    Если внутри analyze() реально падает TypeError — коллектор
    должен его поймать как Exception, а не как «старый интерфейс».
    Проверяем это на «сломанном» анализаторе.
    """
    from src.analyzers.base import BaseAnalyzer
    from src.models import AnalyzerSignal

    class BrokenAnalyzer(BaseAnalyzer):
        def __init__(self):
            super().__init__("broken")

        async def analyze(
            self,
            data,
            indicators=None,
            bar_index=None,
        ):
            # Настоящая внутренняя ошибка
            return None + 1  # TypeError внутри

    collector = SignalCollector(timeout=5.0)
    collector.register(BrokenAnalyzer())

    data = _make_ohlcv()
    signals, rejected, stats = await collector.collect_all(data)

    assert stats["failed"] == 1
    assert len(rejected) == 1
    assert rejected[0].source == "broken"
    assert "TypeError" in rejected[0].reason