"""Анализатор гармонических паттернов.

ПОЛНАЯ ПЕРЕРАБОТКА:
- Детектирует ФОРМИРУЮЩИЕСЯ паттерны (X, A, B, C)
- Прогнозирует PRZ (Potential Reversal Zone)
- Даёт сигнал, когда цена в PRZ

НОВОЕ: поддержка disabled_patterns — список паттернов,
которые нужно игнорировать (на основе диагностики).
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.analyzers.base import BaseAnalyzer
from src.analyzers.trend_filter import TrendFilter, TrendDirection
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str


@dataclass
class FormingPattern:
    """Формирующийся паттерн."""
    pattern_type: str
    direction: str
    x: SwingPoint
    a: SwingPoint
    b: SwingPoint
    c: SwingPoint
    prz_low: float
    prz_high: float
    prz_center: float
    base_confidence: float
    age_bars: int
    ratios: dict


class FormingHarmonicDetector:
    """Детектор формирующихся гармонических паттернов."""

    PATTERN_SPECS = {
        "Gartley": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (0.786, 0.786),
        },
        "Bat": {
            "AB_XA": (0.382, 0.500),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (0.886, 0.886),
        },
        "Butterfly": {
            "AB_XA": (0.786, 0.786),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (1.270, 1.618),
        },
        "Crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (1.618, 1.618),
        },
    }

    def __init__(
        self,
        tolerance: float = 0.20,
        min_pattern_age: int = 3,
        max_pattern_age: int = 200,
    ):
        self.tolerance = tolerance
        self.min_pattern_age = min_pattern_age
        self.max_pattern_age = max_pattern_age

    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> List[SwingPoint]:
        highs = df["high"].values
        lows = df["low"].values
        high_idx, _ = find_peaks(highs, distance=window)
        low_idx, _ = find_peaks(-lows, distance=window)

        points = []
        for idx in high_idx:
            points.append(SwingPoint(index=int(idx), price=float(highs[idx]), kind="high"))
        for idx in low_idx:
            points.append(SwingPoint(index=int(idx), price=float(lows[idx]), kind="low"))
        points.sort(key=lambda p: p.index)
        return points

    def _in_range(self, value: float, expected: Tuple[float, float]) -> bool:
        low, high = expected
        return (low - self.tolerance) <= value <= (high + self.tolerance)

    def _ratio(self, a: float, b: float, c: float, d: float) -> float:
        denom = abs(b - a)
        if denom < 1e-9:
            return 0.0
        return abs(d - c) / denom

    def _alternates(self, *points: SwingPoint) -> bool:
        for i in range(len(points) - 1):
            if points[i].kind == points[i + 1].kind:
                return False
        return True

    def _calc_confidence_2ratios(self, ab_xa: float, bc_ab: float, specs: dict) -> float:
        def score(value, rng):
            low, high = rng
            mid = (low + high) / 2
            span = (high - low) / 2 + 1e-9
            return max(0.0, 1.0 - abs(value - mid) / span)

        score_ab = score(ab_xa, specs["AB_XA"])
        score_bc = score(bc_ab, specs["BC_AB"])
        base = (score_ab + score_bc) / 2.0

        if score_ab > 0.7 and score_bc > 0.7:
            base += 0.15
        elif score_ab > 0.6 and score_bc > 0.6:
            base += 0.05

        return min(1.0, base)

    def _forecast_prz(
        self, x: SwingPoint, a: SwingPoint, c: SwingPoint,
        ad_xa_range: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        xa_range = abs(x.price - a.price)
        ad_low, ad_high = ad_xa_range
        is_bullish = c.price < x.price

        if is_bullish:
            prz_high = a.price - ad_low * xa_range
            prz_low = a.price - ad_high * xa_range
        else:
            prz_low = a.price + ad_low * xa_range
            prz_high = a.price + ad_high * xa_range

        prz_center = (prz_low + prz_high) / 2.0
        return prz_low, prz_high, prz_center

    def detect_forming_patterns(
        self,
        points: List[SwingPoint],
        current_price: float,
        total_bars: int,
        max_patterns: int = 5,
        disabled_patterns: Optional[List[str]] = None,
    ) -> List[FormingPattern]:
        """Ищет формирующиеся паттерны, исключая disabled_patterns."""
        if len(points) < 4:
            return []

        disabled = set(disabled_patterns or [])
        patterns = []

        for i in range(len(points) - 3):
            x, a, b, c = points[i:i + 4]

            if not self._alternates(x, a, b, c):
                continue

            age = total_bars - 1 - c.index
            if age < self.min_pattern_age or age > self.max_pattern_age:
                continue

            for pattern_name, specs in self.PATTERN_SPECS.items():
                if pattern_name in disabled:
                    continue

                ab_xa = self._ratio(x.price, a.price, a.price, b.price)
                bc_ab = self._ratio(a.price, b.price, b.price, c.price)

                if not self._in_range(ab_xa, specs["AB_XA"]):
                    continue
                if not self._in_range(bc_ab, specs["BC_AB"]):
                    continue

                base_conf = self._calc_confidence_2ratios(ab_xa, bc_ab, specs)
                prz_low, prz_high, prz_center = self._forecast_prz(
                    x, a, c, specs["AD_XA"],
                )
                is_bullish = c.price < x.price
                direction = "bullish" if is_bullish else "bearish"

                patterns.append(FormingPattern(
                    pattern_type=pattern_name,
                    direction=direction,
                    x=x, a=a, b=b, c=c,
                    prz_low=prz_low, prz_high=prz_high, prz_center=prz_center,
                    base_confidence=base_conf,
                    age_bars=age,
                    ratios={"AB_XA": ab_xa, "BC_AB": bc_ab},
                ))
                break

        patterns.sort(key=lambda p: p.base_confidence, reverse=True)
        return patterns[:max_patterns]


class HarmonicAnalyzer(BaseAnalyzer):
    """Анализатор гармонических паттернов с поддержкой disabled_patterns."""

    def __init__(
        self,
        name: str = "harmonic",
        tolerance: float = 0.20,
        min_confidence: float = 0.30,
        max_distance_to_prz_pct: float = 0.03,
        min_pattern_age: int = 3,
        max_pattern_age: int = 200,
        use_trend_filter: bool = True,
        trend_penalty: float = 0.2,
        disabled_patterns: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self._detector = FormingHarmonicDetector(
            tolerance=tolerance,
            min_pattern_age=min_pattern_age,
            max_pattern_age=max_pattern_age,
        )
        self._min_confidence = min_confidence
        self._max_distance_to_prz_pct = max_distance_to_prz_pct
        self._use_trend_filter = use_trend_filter
        self._trend_filter = TrendFilter() if use_trend_filter else None
        self._trend_penalty = trend_penalty
        self._disabled_patterns = disabled_patterns or []

    def set_disabled_patterns(self, patterns: List[str]) -> None:
        """Устанавливает список отключённых паттернов."""
        self._disabled_patterns = patterns
        logger.info("Harmonic disabled patterns: %s", patterns)

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        try:
            current_price = float(data["close"].iloc[-1])
            points = self._detector.find_swing_points(data, window=5)
            if len(points) < 4:
                return None

            patterns = self._detector.detect_forming_patterns(
                points, current_price, total_bars=len(data),
                max_patterns=5,
                disabled_patterns=self._disabled_patterns,
            )

            if not patterns:
                return None

            best_pattern = None
            best_confidence = 0.0

            for p in patterns:
                if p.direction == "bullish":
                    distance = abs(current_price - p.prz_center) / current_price
                    price_in_prz = current_price <= p.prz_high * 1.01
                else:
                    distance = abs(current_price - p.prz_center) / current_price
                    price_in_prz = current_price >= p.prz_low * 0.99

                if not price_in_prz:
                    continue
                if distance > self._max_distance_to_prz_pct:
                    continue

                proximity_bonus = max(0.0, 1.0 - distance / self._max_distance_to_prz_pct) * 0.2
                confidence = min(1.0, p.base_confidence + proximity_bonus)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pattern = p

            if best_pattern is None:
                return None

            direction = (
                SignalDirection.BUY if best_pattern.direction == "bullish"
                else SignalDirection.SELL
            )
            confidence = best_confidence

            trend_info = "no_filter"
            if self._use_trend_filter and self._trend_filter is not None:
                trend = self._trend_filter.detect(data)
                multiplier = self._trend_filter.apply_penalty(
                    direction, trend, self._trend_penalty,
                )
                confidence *= multiplier
                trend_info = trend.name

            if confidence < self._min_confidence:
                return None

            distance_to_prz = abs(current_price - best_pattern.prz_center) / current_price
            reasons = (
                f"{best_pattern.pattern_type} ({best_pattern.direction}) "
                f"PRZ={best_pattern.prz_center:.4f}, "
                f"conf={confidence:.2f}, "
                f"C_age={best_pattern.age_bars}bars, "
                f"trend={trend_info}"
            )

            return AnalyzerSignal(
                direction=direction,
                confidence=confidence,
                reason=reasons,
                metadata={
                    "pattern_type": best_pattern.pattern_type,
                    "pattern_direction": best_pattern.direction,
                    "pattern_x_price": best_pattern.x.price,
                    "pattern_a_price": best_pattern.a.price,
                    "pattern_b_price": best_pattern.b.price,
                    "pattern_c_price": best_pattern.c.price,
                    "pattern_c_age": best_pattern.age_bars,
                    "prz_low": best_pattern.prz_low,
                    "prz_high": best_pattern.prz_high,
                    "prz_center": best_pattern.prz_center,
                    "current_price": current_price,
                    "distance_to_prz_pct": distance_to_prz,
                    "base_confidence": best_pattern.base_confidence,
                    "ratios": best_pattern.ratios,
                    "trend": trend_info,
                    "disabled_patterns": self._disabled_patterns,
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("Ошибка в HarmonicAnalyzer: %s", e)
            return None