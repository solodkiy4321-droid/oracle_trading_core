"""Тесты фильтров и анализаторов тренда/импульса."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.trend import TrendAnalyzer
from src.analyzers.momentum import MomentumAnalyzer
from src.models import SignalDirection


def make_uptrend(n: int = 400) -> pd.DataFrame:
    np.random.seed(42)
    close = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_downtrend(n: int = 400) -> pd.DataFrame:
    np.random.seed(43)
    close = np.linspace(150, 100, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_flat(n: int = 400) -> pd.DataFrame:
    np.random.seed(44)
    close = 100 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.2),
        "low": close - np.abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# ============ TrendAnalyzer ============

@pytest.mark.asyncio
async def test_trend_returns_signal_or_none_uptrend():
    analyzer = TrendAnalyzer()
    result = await analyzer.analyze(make_uptrend())
    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 1.0
        assert "adx" in result.metadata
        assert "ema_value" in result.metadata
        assert "trend" in result.metadata


@pytest.mark.asyncio
async def test_trend_downtrend_direction():
    analyzer = TrendAnalyzer(adx_threshold=10.0)
    result = await analyzer.analyze(make_downtrend())
    if result is not None:
        assert result.direction == SignalDirection.SELL


@pytest.mark.asyncio
async def test_trend_insufficient_data():
    analyzer = TrendAnalyzer(ema_period=200)
    data = make_uptrend(n=50)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_trend_metadata_structure():
    analyzer = TrendAnalyzer()
    result = await analyzer.analyze(make_uptrend())
    if result is not None:
        required = [
            "ema_period", "ema_value", "adx", "adx_threshold",
            "adx_strong", "distance_pct", "trend", "base_confidence",
        ]
        for key in required:
            assert key in result.metadata, f"Отсутствует ключ {key}"


# ============ MomentumAnalyzer ============

@pytest.mark.asyncio
async def test_momentum_returns_signal_or_none():
    analyzer = MomentumAnalyzer()
    result = await analyzer.analyze(make_flat())
    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 1.0
        assert "rsi" in result.metadata
        assert "zone" in result.metadata


@pytest.mark.asyncio
async def test_momentum_insufficient_data():
    analyzer = MomentumAnalyzer()
    data = make_flat(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_momentum_metadata_structure():
    analyzer = MomentumAnalyzer()
    result = await analyzer.analyze(make_flat())
    if result is not None:
        required = [
            "rsi", "rsi_period", "zone", "regime",
            "oversold", "overbought", "base_confidence",
        ]
        for key in required:
            assert key in result.metadata, f"Отсутствует ключ {key}"


@pytest.mark.asyncio
async def test_momentum_only_in_chop():
    """В трендовых данных momentum не должен давать сигналов."""
    analyzer = MomentumAnalyzer()
    result = await analyzer.analyze(make_uptrend())
    assert result is None