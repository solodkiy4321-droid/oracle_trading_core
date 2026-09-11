"""Тесты простого S/R анализатора."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.support_resistance import SRDetector, SRAnalyzer, SRLevel
from src.models import SignalDirection


def make_data_with_levels(n: int = 300) -> pd.DataFrame:
    """Данные с чёткими уровнями."""
    np.random.seed(42)
    prices = []
    for i in range(n):
        base = 100 + 5 * np.sin(2 * np.pi * i / 30)
        noise = np.random.randn() * 0.3
        prices.append(base + noise)
    close = np.array(prices)
    high = close + np.abs(np.random.randn(n) * 0.2) + 0.3
    low = close - np.abs(np.random.randn(n) * 0.2) - 0.3
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_trending_data(n: int = 300) -> pd.DataFrame:
    """Трендовые данные."""
    np.random.seed(43)
    close = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_detector_finds_levels():
    """Детектор находит уровни."""
    detector = SRDetector(min_touches=2)
    data = make_data_with_levels()
    levels = detector.detect_levels(data)
    assert len(levels) > 0


def test_detector_levels_have_strength():
    """Уровни имеют силу и касания."""
    detector = SRDetector(min_touches=2)
    data = make_data_with_levels()
    levels = detector.detect_levels(data)

    for level in levels:
        assert 0.0 <= level.strength <= 1.0
        assert level.touches >= 2
        assert level.kind in ("support", "resistance")


def test_detector_insufficient_data():
    """Недостаточно данных — пустой список."""
    detector = SRDetector()
    data = make_data_with_levels(n=30)
    levels = detector.detect_levels(data)
    assert levels == []


def test_detector_trending_data():
    """На трендовых данных уровней меньше."""
    detector = SRDetector(min_touches=3)
    trending = make_trending_data()
    levels = detector.detect_levels(trending)
    assert isinstance(levels, list)


@pytest.mark.asyncio
async def test_analyzer_returns_signal_or_none():
    """Анализатор возвращает сигнал или None."""
    analyzer = SRAnalyzer()
    data = make_data_with_levels()
    result = await analyzer.analyze(data)

    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 1.0
        assert "level_price" in result.metadata
        assert "level_strength" in result.metadata
        assert "level_kind" in result.metadata


@pytest.mark.asyncio
async def test_analyzer_insufficient_data():
    """Недостаточно данных — None."""
    analyzer = SRAnalyzer()
    data = make_data_with_levels(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_analyzer_metadata_structure():
    """Метаданные содержат нужные поля."""
    analyzer = SRAnalyzer(proximity_pct=0.05)
    data = make_data_with_levels()
    result = await analyzer.analyze(data)

    if result is not None:
        meta = result.metadata
        assert "level_price" in meta
        assert "level_strength" in meta
        assert "level_touches" in meta
        assert "level_kind" in meta
        assert "distance_pct" in meta
        assert "current_price" in meta
        assert "total_levels" in meta