"""Тесты анализатора гармонических паттернов."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.harmonic import HarmonicAnalyzer, HarmonicPatternDetector
from src.models import SignalDirection


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Синтетические OHLCV-данные с трендом."""
    np.random.seed(42)
    n = 200
    # Создаём зигзагообразный тренд для появления swing points
    t = np.linspace(0, 8 * np.pi, n)
    close = 100 + 10 * np.sin(t) + np.cumsum(np.random.randn(n) * 0.2)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_swing_points_found(sample_data):
    """Детектор находит swing points."""
    detector = HarmonicPatternDetector()
    points = detector.find_swing_points(sample_data, window=5)
    assert len(points) >= 5


def test_alternation_check():
    """Проверка чередования high/low."""
    from src.analyzers.harmonic import SwingPoint
    detector = HarmonicPatternDetector()

    # Правильное чередование
    good = [
        SwingPoint(0, 100, "high"),
        SwingPoint(10, 90, "low"),
        SwingPoint(20, 95, "high"),
        SwingPoint(30, 85, "low"),
        SwingPoint(40, 92, "high"),
    ]
    assert detector._alternates(*good)

    # Неправильное чередование
    bad = [
        SwingPoint(0, 100, "high"),
        SwingPoint(10, 105, "high"),
    ]
    assert not detector._alternates(*bad)


@pytest.mark.asyncio
async def test_harmonic_analyzer_returns_signal_or_none(sample_data):
    """Анализатор возвращает сигнал или None."""
    analyzer = HarmonicAnalyzer()
    result = await analyzer.analyze(sample_data)
    # Может вернуть сигнал или None — оба варианта валидны
    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)
        assert 0.0 <= result.confidence <= 1.0
        assert result.reason