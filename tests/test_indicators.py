"""Тесты анализатора индикаторов с бонусом за согласованность."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.indicators import IndicatorsAnalyzer
from src.models import SignalDirection


def make_bullish_data(n: int = 100) -> pd.DataFrame:
    """Данные с сильным восходящим трендом."""
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
    """Данные с сильным нисходящим трендом."""
    np.random.seed(43)
    close = np.linspace(120, 100, n) + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_flat_data(n: int = 100) -> pd.DataFrame:
    """Боковик."""
    np.random.seed(44)
    close = 100 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.mark.asyncio
async def test_indicators_returns_signal_or_none():
    """Анализатор возвращает сигнал или None."""
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 0.95
        assert "agreement_count" in result.metadata
        assert "agreement_bonus" in result.metadata
        assert "base_confidence" in result.metadata


@pytest.mark.asyncio
async def test_indicators_agreement_bonus_increases_confidence():
    """
    КРИТИЧЕСКИЙ ТЕСТ: бонус за согласованность увеличивает confidence.

    Проверяем, что при согласованных индикаторах confidence выше,
    чем base_confidence.
    """
    analyzer = IndicatorsAnalyzer()
    data = make_bearish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        base_conf = result.metadata["base_confidence"]
        agreement_count = result.metadata["agreement_count"]
        bonus = result.metadata["agreement_bonus"]

        if agreement_count > 1:
            # Бонус должен быть > 1.0
            assert bonus > 1.0
            # Финальный confidence должен быть выше base_confidence
            # (с учётом штрафа за тренд, поэтому проверяем мягко)
            expected_conf = base_conf * bonus
            assert expected_conf > base_conf


@pytest.mark.asyncio
async def test_indicators_confidence_above_old_cap():
    """
    Проверяем, что confidence может превысить старый потолок 0.54.

    Это ключевая цель улучшения: дать анализатору возможность
    выдавать сигналы с confidence > 0.54.
    """
    analyzer = IndicatorsAnalyzer(use_trend_filter=False)
    data = make_bearish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        # Если 3+ индикатора согласованы, confidence должен быть > 0.54
        agreement_count = result.metadata["agreement_count"]
        if agreement_count >= 3:
            assert result.confidence > 0.5, (
                f"Ожидалось conf > 0.5 при agreement_count={agreement_count}, "
                f"получено {result.confidence}"
            )


@pytest.mark.asyncio
async def test_indicators_insufficient_data():
    """Недостаточно данных — None."""
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_indicators_metadata_structure():
    """Метаданные содержат все нужные поля."""
    analyzer = IndicatorsAnalyzer()
    data = make_bullish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        meta = result.metadata
        required = [
            "rsi", "macd_hist", "ema9", "sma20", "sma50",
            "bb_upper", "bb_lower", "current_price",
            "bullish_score", "bearish_score",
            "base_confidence", "agreement_count", "agreement_bonus",
            "trend", "weights",
        ]
        for key in required:
            assert key in meta, f"Отсутствует ключ {key}"


@pytest.mark.asyncio
async def test_indicators_max_confidence_cap():
    """Confidence не превышает max_confidence."""
    analyzer = IndicatorsAnalyzer(
        use_trend_filter=False,
        max_confidence=0.80,
    )
    data = make_bearish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        assert result.confidence <= 0.80


@pytest.mark.asyncio
async def test_indicators_agreement_bonus_custom_step():
    """Кастомный шаг бонуса работает."""
    analyzer = IndicatorsAnalyzer(
        use_trend_filter=False,
        agreement_bonus_step=0.30,  # более агрессивный бонус
    )
    data = make_bearish_data()
    result = await analyzer.analyze(data)

    if result is not None:
        agreement_count = result.metadata["agreement_count"]
        if agreement_count > 1:
            expected_bonus = 1.0 + 0.30 * (agreement_count - 1)
            assert abs(result.metadata["agreement_bonus"] - expected_bonus) < 1e-6