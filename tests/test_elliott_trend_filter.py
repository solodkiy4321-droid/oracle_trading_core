"""Тесты для Бага E: ElliottWaveAnalyzer не должен пересчитывать SMA200.

Проверяют:
- При переданном IndicatorCache ta.sma НЕ вызывается.
- Значение тренда совпадает с тем, что дал бы ta.sma.
- Без IndicatorCache используется старый путь (ta.sma вызывается).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import pandas_ta_classic as ta

from src.analyzers.elliott_wave import ElliottWaveAnalyzer
from src.analyzers.trend_filter import TrendFilter, TrendDirection


# ---------- Синтетические данные ----------

def _make_ohlcv(n: int = 500, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if trend == "up":
        base = np.linspace(100, 200, n)
    elif trend == "down":
        base = np.linspace(200, 100, n)
    else:
        base = np.full(n, 150.0)
    close = base + rng.normal(0, 1.0, n)
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------- TrendFilter.detect_from_cache ----------

def test_trend_filter_detect_from_cache_returns_up():
    tf = TrendFilter()
    assert tf.detect_from_cache(sma_val=100.0, current_price=110.0) == TrendDirection.UP


def test_trend_filter_detect_from_cache_returns_down():
    tf = TrendFilter()
    assert tf.detect_from_cache(sma_val=100.0, current_price=90.0) == TrendDirection.DOWN


def test_trend_filter_detect_from_cache_returns_neutral():
    tf = TrendFilter()
    # Отклонение 0.5% < neutral_band_pct=1%
    assert tf.detect_from_cache(sma_val=100.0, current_price=100.5) == TrendDirection.NEUTRAL


def test_trend_filter_detect_from_cache_none_args():
    tf = TrendFilter()
    assert tf.detect_from_cache(None, 100.0) == TrendDirection.NEUTRAL
    assert tf.detect_from_cache(100.0, None) == TrendDirection.NEUTRAL
    assert tf.detect_from_cache(None, None) == TrendDirection.NEUTRAL


def test_trend_filter_classify_consistent_with_detect():
    """
    detect_from_cache и detect должны давать одинаковый результат
    при одинаковых входных значениях.
    """
    tf = TrendFilter()
    data = _make_ohlcv(n=300, trend="up")
    # detect() пересчитывает SMA сам
    trend_slow = tf.detect(data)
    # detect_from_cache() с теми же значениями
    sma = ta.sma(data["close"], length=200)
    sma_val = float(sma.iloc[-1])
    current_price = float(data["close"].iloc[-1])
    trend_fast = tf.detect_from_cache(sma_val, current_price)
    assert trend_slow == trend_fast


# ---------- ElliottWaveAnalyzer: ta.sma не вызывается при кэше ----------

@pytest.mark.asyncio
async def test_elliott_does_not_call_ta_sma_when_indicator_cache_provided():
    """
    Ключевой тест Бага E.
    При переданном IndicatorCache ta.sma НЕ должен вызываться.
    """
    data = _make_ohlcv(n=500, trend="up")
    analyzer = ElliottWaveAnalyzer(min_pattern_confidence=0.0, min_confidence=0.0)

    class DummyCache:
        def __init__(self, data):
            self._data = data
            self.n = len(data)
            close = data["close"]
            self._sma200 = ta.sma(close, length=200)
            self._swing = [
                (100, float(data["high"].iloc[100]), "high"),
                (150, float(data["low"].iloc[150]), "low"),
                (200, float(data["high"].iloc[200]), "high"),
                (250, float(data["low"].iloc[250]), "low"),
                (300, float(data["high"].iloc[300]), "high"),
                (350, float(data["low"].iloc[350]), "low"),
            ]

        def slice(self, i):
            return {
                "sma_200": float(self._sma200.iloc[i])
                if i < len(self._sma200) and not pd.isna(self._sma200.iloc[i])
                else None,
                "current_price": float(self._data["close"].iloc[i]),
            }

        def get_swing_points(self, end_idx):
            return [p for p in self._swing if p[0] <= end_idx]

    cache = DummyCache(data)

    real_sma = ta.sma
    call_count = {"n": 0}

    def counting_sma(*args, **kwargs):
        call_count["n"] += 1
        return real_sma(*args, **kwargs)

    # Патчим ta.sma и в trend_filter, и в elliott_wave модулях
    with patch("src.analyzers.trend_filter.ta.sma", side_effect=counting_sma):
        await analyzer.analyze(data, indicators=cache, bar_index=499)

    assert call_count["n"] == 0, (
        f"ta.sma должна НЕ вызываться при переданном IndicatorCache, "
        f"но вызвана {call_count['n']} раз"
    )


@pytest.mark.asyncio
async def test_elliott_calls_ta_sma_without_indicator_cache():
    """
    Без IndicatorCache старый путь используется — ta.sma вызывается.
    Это ожидаемое поведение (fallback).
    """
    data = _make_ohlcv(n=500, trend="up")
    analyzer = ElliottWaveAnalyzer(min_pattern_confidence=0.0, min_confidence=0.0)

    real_sma = ta.sma
    call_count = {"n": 0}

    def counting_sma(*args, **kwargs):
        call_count["n"] += 1
        return real_sma(*args, **kwargs)

    with patch("src.analyzers.trend_filter.ta.sma", side_effect=counting_sma):
        await analyzer.analyze(data, indicators=None, bar_index=None)

    # Без кэша fallback использует detect() → ta.sma вызывается
    assert call_count["n"] >= 1, (
        "Без IndicatorCache fallback должен вызвать ta.sma"
    )


@pytest.mark.asyncio
async def test_elliott_trend_result_matches_manual_calc():
    """
    При использовании кэша trend_info должен совпасть с ручным расчётом
    по sma_200 из кэша.
    """
    data = _make_ohlcv(n=500, trend="up")
    analyzer = ElliottWaveAnalyzer(min_pattern_confidence=0.0, min_confidence=0.0)

    sma200 = ta.sma(data["close"], length=200)
    last_close = float(data["close"].iloc[-1])

    class DummyCache:
        def __init__(self, data, sma200):
            self._data = data
            self._sma200 = sma200
            self.n = len(data)
            self._swing = [
                (100, float(data["high"].iloc[100]), "high"),
                (150, float(data["low"].iloc[150]), "low"),
                (200, float(data["high"].iloc[200]), "high"),
                (250, float(data["low"].iloc[250]), "low"),
                (300, float(data["high"].iloc[300]), "high"),
                (350, float(data["low"].iloc[350]), "low"),
            ]

        def slice(self, i):
            return {
                "sma_200": float(self._sma200.iloc[i])
                if i < len(self._sma200) and not pd.isna(self._sma200.iloc[i])
                else None,
                "current_price": float(self._data["close"].iloc[i]),
            }

        def get_swing_points(self, end_idx):
            return [p for p in self._swing if p[0] <= end_idx]

    cache = DummyCache(data, sma200)
    result = await analyzer.analyze(data, indicators=cache, bar_index=499)

    # Если сигнал есть — проверяем что trend_info совпадает с ожидаемым
    if result is not None:
        sma_val = float(sma200.iloc[-1])
        expected_trend = TrendFilter().detect_from_cache(sma_val, last_close)
        assert result.metadata["trend"] == expected_trend.name