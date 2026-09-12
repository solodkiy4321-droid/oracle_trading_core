"""Тесты анализатора волн Эллиотта."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.elliott_wave import (
    ElliottWaveDetector,
    ElliottWaveAnalyzer,
    SwingPoint,
    WavePattern,
)
from src.models import SignalDirection


def make_trending_data(n: int = 300) -> pd.DataFrame:
    """Создаёт данные с трендом."""
    np.random.seed(42)
    close = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3) + 0.5,
        "low": close - np.abs(np.random.randn(n) * 0.3) - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_zigzag_data(n: int = 300) -> pd.DataFrame:
    """Создаёт данные с зигзагами (для волн)."""
    np.random.seed(43)
    prices = []
    for i in range(n):
        base = 100 + 15 * np.sin(2 * np.pi * i / 40)
        noise = np.random.randn() * 0.5
        prices.append(base + noise)
    close = np.array(prices)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3) + 0.5,
        "low": close - np.abs(np.random.randn(n) * 0.3) - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_detector_finds_swing_points():
    """Детектор находит swing points."""
    detector = ElliottWaveDetector(swing_window=5)
    data = make_zigzag_data()
    points = detector.find_swing_points(data)

    assert len(points) > 10
    for p in points:
        assert isinstance(p, SwingPoint)
        assert p.kind in ("high", "low")
        assert p.price > 0


def test_detector_alternation_check():
    """Проверка чередования high/low."""
    detector = ElliottWaveDetector()

    good = [
        SwingPoint(0, 100, "high"),
        SwingPoint(10, 90, "low"),
        SwingPoint(20, 95, "high"),
        SwingPoint(30, 85, "low"),
    ]
    assert detector._alternates(*good)

    bad = [
        SwingPoint(0, 100, "high"),
        SwingPoint(10, 105, "high"),
    ]
    assert not detector._alternates(*bad)


def test_detector_finds_patterns():
    """Детектор находит волновые паттерны."""
    detector = ElliottWaveDetector(swing_window=5)
    data = make_zigzag_data()
    points = detector.find_swing_points(data)

    patterns = detector.detect_patterns(points, total_bars=len(data))

    assert isinstance(patterns, list)
    for p in patterns:
        assert isinstance(p, WavePattern)
        assert p.pattern_type in ("impulse", "correction")
        assert p.direction in ("bullish", "bearish")
        assert 0.0 <= p.confidence <= 1.0
        assert p.age_bars >= 0


def test_detector_insufficient_points():
    """Мало точек — пустой список."""
    detector = ElliottWaveDetector()
    patterns = detector.detect_patterns([], total_bars=100)
    assert patterns == []


@pytest.mark.asyncio
async def test_analyzer_returns_signal_or_none():
    """Анализатор возвращает сигнал или None."""
    analyzer = ElliottWaveAnalyzer(
        min_confidence=0.20,
        use_trend_filter=False,
    )
    data = make_zigzag_data()
    result = await analyzer.analyze(data)

    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 1.0
        assert "pattern_type" in result.metadata
        assert "ratios" in result.metadata


@pytest.mark.asyncio
async def test_analyzer_insufficient_data():
    """Мало данных — None."""
    analyzer = ElliottWaveAnalyzer()
    data = make_zigzag_data(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_analyzer_metadata_structure():
    """Метаданные содержат нужные поля."""
    analyzer = ElliottWaveAnalyzer(
        min_confidence=0.20,
        use_trend_filter=False,
    )
    data = make_zigzag_data()
    result = await analyzer.analyze(data)

    if result is not None:
        meta = result.metadata
        required = [
            "pattern_type", "pattern_direction",
            "pattern_points_count", "pattern_age_bars",
            "pattern_end_price", "ratios", "trend", "patterns_found",
        ]
        for key in required:
            assert key in meta, f"Отсутствует ключ {key}"


def test_detector_impulse_rules_validation():
    """Проверка правил импульсной волны."""
    detector = ElliottWaveDetector(fibonacci_tolerance=0.3)

    # Валидная импульсная волна (bullish)
    wave = [
        SwingPoint(0, 100, "low"),
        SwingPoint(10, 110, "high"),
        SwingPoint(20, 106, "low"),   # w2 откат < 100% w1
        SwingPoint(30, 125, "high"),  # w3 > w1
        SwingPoint(40, 120, "low"),   # w4 > w1.price (110)
        SwingPoint(50, 130, "high"),  # w5 > w3
    ]
    valid, ratios = detector._check_impulse_rules(wave)
    assert valid
    assert "w2_w1" in ratios
    assert "w3_w1" in ratios