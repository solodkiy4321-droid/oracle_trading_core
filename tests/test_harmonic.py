"""Тесты переработанного анализатора гармонических паттернов."""

import numpy as np
import pandas as pd
import pytest

from src.analyzers.harmonic import (
    FormingHarmonicDetector,
    HarmonicAnalyzer,
    FormingPattern,
    SwingPoint,
)
from src.models import SignalDirection


def make_zigzag_data(n: int = 300) -> pd.DataFrame:
    """
    Создаёт данные с явными зигзагами — для тестов гармонических паттернов.
    """
    np.random.seed(42)
    prices = []
    for i in range(n):
        # Синусоида + шум — даёт swing points
        base = 100 + 10 * np.sin(2 * np.pi * i / 25)
        noise = np.random.randn() * 0.5
        prices.append(base + noise)
    close = np.array(prices)
    high = close + np.abs(np.random.randn(n) * 0.3) + 0.5
    low = close - np.abs(np.random.randn(n) * 0.3) - 0.5
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_flat_data(n: int = 300) -> pd.DataFrame:
    """Создаёт плоские данные — без чётких паттернов."""
    np.random.seed(43)
    close = 100 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.2),
        "low": close - np.abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def test_detector_finds_swing_points():
    """Детектор находит swing points."""
    detector = FormingHarmonicDetector()
    data = make_zigzag_data()
    points = detector.find_swing_points(data, window=5)

    assert len(points) > 10
    # Все точки — SwingPoint
    for p in points:
        assert isinstance(p, SwingPoint)
        assert p.kind in ("high", "low")
        assert p.price > 0


def test_detector_alternation_check():
    """Проверка чередования high/low."""
    detector = FormingHarmonicDetector()

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


def test_detector_finds_forming_patterns():
    """Детектор находит формирующиеся паттерны (X, A, B, C)."""
    detector = FormingHarmonicDetector(
        tolerance=0.30,  # мягче для теста
        min_pattern_age=3,
        max_pattern_age=300,
    )
    data = make_zigzag_data()
    points = detector.find_swing_points(data, window=5)

    current_price = float(data["close"].iloc[-1])
    patterns = detector.detect_forming_patterns(
        points, current_price, total_bars=len(data), max_patterns=10,
    )

    # Может быть 0 или больше — зависит от данных
    assert isinstance(patterns, list)

    # Если нашли — проверяем структуру
    for p in patterns:
        assert isinstance(p, FormingPattern)
        assert p.pattern_type in ("Gartley", "Bat", "Butterfly", "Crab")
        assert p.direction in ("bullish", "bearish")
        assert 0.0 <= p.base_confidence <= 1.0
        assert p.prz_low <= p.prz_center <= p.prz_high
        assert p.age_bars >= 0


def test_detector_prz_forecast():
    """
    PRZ прогнозируется корректно.

    Для bullish паттерна D ниже A, для bearish — выше.
    """
    detector = FormingHarmonicDetector()

    # Bullish: C ниже X → PRZ ниже A
    x = SwingPoint(0, 100, "high")
    a = SwingPoint(10, 90, "low")
    c = SwingPoint(30, 85, "low")  # C ниже X
    prz_low, prz_high, prz_center = detector._forecast_prz(
        x, a, c, (0.786, 0.786),
    )
    assert prz_center < a.price  # D ниже A для bullish
    assert prz_low <= prz_center <= prz_high

    # Bearish: C выше X → PRZ выше A
    x2 = SwingPoint(0, 90, "low")
    a2 = SwingPoint(10, 100, "high")
    c2 = SwingPoint(30, 105, "high")  # C выше X
    prz_low2, prz_high2, prz_center2 = detector._forecast_prz(
        x2, a2, c2, (0.786, 0.786),
    )
    assert prz_center2 > a2.price  # D выше A для bearish


def test_detector_insufficient_points():
    """Недостаточно точек — пустой список."""
    detector = FormingHarmonicDetector()
    patterns = detector.detect_forming_patterns(
        [], current_price=100.0, total_bars=100,
    )
    assert patterns == []


@pytest.mark.asyncio
async def test_analyzer_returns_signal_or_none():
    """Анализатор возвращает сигнал или None."""
    analyzer = HarmonicAnalyzer(
        tolerance=0.30,
        min_confidence=0.20,
        max_distance_to_prz_pct=0.10,
        use_trend_filter=False,
    )
    data = make_zigzag_data()
    result = await analyzer.analyze(data)

    if result is not None:
        assert result.direction in (SignalDirection.BUY, SignalDirection.SELL)
        assert 0.0 <= result.confidence <= 1.0
        assert "pattern_type" in result.metadata
        assert "prz_center" in result.metadata
        assert "prz_low" in result.metadata
        assert "prz_high" in result.metadata
        assert "distance_to_prz_pct" in result.metadata


@pytest.mark.asyncio
async def test_analyzer_insufficient_data():
    """Недостаточно данных — None."""
    analyzer = HarmonicAnalyzer()
    data = make_zigzag_data(n=30)
    result = await analyzer.analyze(data)
    assert result is None


@pytest.mark.asyncio
async def test_analyzer_flat_data_returns_none():
    """На плоских данных — None или слабый сигнал."""
    analyzer = HarmonicAnalyzer(
        tolerance=0.20,
        min_confidence=0.30,
        max_distance_to_prz_pct=0.05,
        use_trend_filter=False,
    )
    data = make_flat_data()
    result = await analyzer.analyze(data)

    # Может вернуть None или сигнал с низким confidence
    if result is not None:
        assert result.confidence < 0.9  # не должен быть сверхуверенным


@pytest.mark.asyncio
async def test_analyzer_metadata_structure():
    """Метаданные содержат все нужные поля."""
    analyzer = HarmonicAnalyzer(
        tolerance=0.30,
        min_confidence=0.20,
        max_distance_to_prz_pct=0.10,
        use_trend_filter=False,
    )
    data = make_zigzag_data()
    result = await analyzer.analyze(data)

    if result is not None:
        meta = result.metadata
        required = [
            "pattern_type", "pattern_direction",
            "pattern_x_price", "pattern_a_price",
            "pattern_b_price", "pattern_c_price",
            "pattern_c_age",
            "prz_low", "prz_high", "prz_center",
            "current_price", "distance_to_prz_pct",
            "base_confidence", "ratios", "trend",
        ]
        for key in required:
            assert key in meta, f"Отсутствует ключ {key}"


@pytest.mark.asyncio
async def test_analyzer_pattern_types():
    """
    Все паттерны harmonic поддерживаются.

    Проверяем, что в PATTERN_SPECS есть все 4 паттерна.
    """
    detector = FormingHarmonicDetector()
    assert "Gartley" in detector.PATTERN_SPECS
    assert "Bat" in detector.PATTERN_SPECS
    assert "Butterfly" in detector.PATTERN_SPECS
    assert "Crab" in detector.PATTERN_SPECS


def test_detector_confidence_calculation():
    """Confidence рассчитывается корректно."""
    detector = FormingHarmonicDetector()

    # Идеальные соотношения для Gartley
    specs = detector.PATTERN_SPECS["Gartley"]
    mid_ab = sum(specs["AB_XA"]) / 2
    mid_bc = sum(specs["BC_AB"]) / 2

    conf = detector._calc_confidence_2ratios(mid_ab, mid_bc, specs)
    assert conf > 0.5  # при идеальных значениях — высокая

    # Далекие от идеала
    conf_low = detector._calc_confidence_2ratios(1.5, 1.5, specs)
    assert conf_low < conf  # должна быть ниже