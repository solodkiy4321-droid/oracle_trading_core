"""Тесты анализатора индикаторов (переработанная версия)."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.indicators import IndicatorsAnalyzer
from src.models import SignalDirection


def make_bullish_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = np.linspace(100, 120, n) + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_bearish_data(n: int = 100) -> pd.DataFrame:
    np.random.seed(43)
    close = np.linspace(120, 100, n) + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.mark.asyncio
async def test_indicators_returns_signal_or_none():
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data()
    result = await analyzer.analyze(data)
    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 0.90


@pytest.mark.asyncio
async def test_indicators_insufficient_data():
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_indicators_metadata_structure():
    """Metadata содержит поля после переработки (без RSI и BB)."""
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        meta = result.metadata
        required = [
            "macd_hist", "ema9", "sma20", "sma50", "atr",
            "base_confidence", "agreement_count", "agreement_bonus",
            "trend", "weights",
        ]
        for key in required:
            assert key in meta, f"Отсутствует ключ {key}"


@pytest.mark.asyncio
async def test_indicators_max_confidence_cap():
    analyzer = IndicatorsAnalyzer(max_confidence=0.80)
    data = make_bearish_data()
    result = await analyzer.analyze(data)
    if result is not None:
        assert result.confidence <= 0.80


@pytest.mark.asyncio
async def test_indicators_agreement_bonus_custom_step():
    analyzer = IndicatorsAnalyzer(
        use_trend_filter=False,
        agreement_bonus_step=0.30,
    )
    data = make_bearish_data()
    result = await analyzer.analyze(data)
    if result is not None:
        agreement = result.metadata["agreement_count"]
        if agreement > 1:
            expected_bonus = 1.0 + 0.30 * (agreement - 1)
            assert abs(result.metadata["agreement_bonus"] - expected_bonus) < 1e-6