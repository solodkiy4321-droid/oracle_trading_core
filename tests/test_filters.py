"""Тесты фильтров актуальности и тренда."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.trend_filter import TrendFilter, TrendDirection
from src.models import SignalDirection


def make_uptrend(n: int = 250) -> pd.DataFrame:
    """Создаёт восходящий тренд."""
    np.random.seed(42)
    close = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_downtrend(n: int = 250) -> pd.DataFrame:
    """Создаёт нисходящий тренд."""
    np.random.seed(43)
    close = np.linspace(150, 100, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_flat(n: int = 250) -> pd.DataFrame:
    """Создаёт боковик."""
    np.random.seed(44)
    close = 100 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.2),
        "low": close - np.abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_trend_up():
    """Uptrend определяется корректно."""
    filter_ = TrendFilter()
    trend = filter_.detect(make_uptrend())
    assert trend == TrendDirection.UP


def test_trend_down():
    """Downtrend определяется корректно."""
    filter_ = TrendFilter()
    trend = filter_.detect(make_downtrend())
    assert trend == TrendDirection.DOWN


def test_trend_neutral():
    """Боковик определяется как NEUTRAL."""
    filter_ = TrendFilter()
    trend = filter_.detect(make_flat())
    assert trend == TrendDirection.NEUTRAL


def test_penalty_against_trend():
    """Сигнал против тренда получает штраф."""
    filter_ = TrendFilter()
    multiplier = filter_.apply_penalty(
        SignalDirection.BUY, TrendDirection.DOWN, penalty=0.3
    )
    assert multiplier == 0.7


def test_no_penalty_with_trend():
    """Сигнал по тренду не получает штраф."""
    filter_ = TrendFilter()
    multiplier = filter_.apply_penalty(
        SignalDirection.BUY, TrendDirection.UP, penalty=0.3
    )
    assert multiplier == 1.0


def test_no_penalty_neutral():
    """Нейтральный тренд не даёт штрафа."""
    filter_ = TrendFilter()
    multiplier = filter_.apply_penalty(
        SignalDirection.BUY, TrendDirection.NEUTRAL, penalty=0.3
    )
    assert multiplier == 1.0


def test_insufficient_data():
    """Недостаточно данных — NEUTRAL."""
    filter_ = TrendFilter()
    short_data = make_uptrend(n=50)
    trend = filter_.detect(short_data)
    assert trend == TrendDirection.NEUTRAL


@pytest.mark.asyncio
async def test_indicators_analyzer_with_trend_filter():
    """IndicatorsAnalyzer работает с фильтром тренда."""
    from src.analyzers.indicators import IndicatorsAnalyzer
    analyzer = IndicatorsAnalyzer(name="indicators_test", use_trend_filter=True)
    data = make_uptrend(250)
    result = await analyzer.analyze(data)
    # Может вернуть сигнал или None — оба варианта валидны
    if result is not None:
        assert "trend" in result.metadata
        assert result.metadata["trend"] in ("UP", "DOWN", "NEUTRAL", "no_filter")


@pytest.mark.asyncio
async def test_harmonic_analyzer_with_trend_filter():
    """HarmonicAnalyzer работает с фильтром тренда."""
    from src.analyzers.harmonic import HarmonicAnalyzer
    analyzer = HarmonicAnalyzer(name="harmonic_test", use_trend_filter=True)
    data = make_uptrend(250)
    result = await analyzer.analyze(data)
    # Может вернуть сигнал или None — оба варианта валидны
    if result is not None:
        assert "trend" in result.metadata