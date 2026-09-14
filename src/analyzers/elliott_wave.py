"""ElliottWaveAnalyzer — волны Эллиотта.

Категория: ПАТТЕРНЫ.

Использует IndicatorCache если передан — берёт готовые swing_points
И готовый sma_200 для trend-фильтра (без пересчёта ta.sma).
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.analyzers.base import BaseAnalyzer
from src.analyzers.trend_filter import TrendFilter
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str


@dataclass
class WavePattern:
    pattern_type: str
    direction: str
    points: List[SwingPoint]
    confidence: float
    ratios: dict
    age_bars: int


class ElliottWaveDetector:

    W2_W1_RANGE = (0.20, 0.90)
    W3_W1_RANGE = (1.00, 3.50)
    W4_W3_RANGE = (0.15, 0.70)
    W5_W3_RANGE = (0.30, 2.50)
    B_A_RANGE = (0.20, 0.95)

    def __init__(
        self,
        swing_window: int = 5,
        min_pattern_age: int = 3,
        max_pattern_age: int = 100,
        fibonacci_tolerance: float = 0.15,
    ):
        self.swing_window = swing_window
        self.min_pattern_age = min_pattern_age
        self.max_pattern_age = max_pattern_age
        self.fibonacci_tolerance = fibonacci_tolerance

    def find_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        highs = df["high"].values
        lows = df["low"].values

        high_idx, _ = find_peaks(highs, distance=self.swing_window)
        low_idx, _ = find_peaks(-lows, distance=self.swing_window)

        points = []
        for idx in high_idx:
            points.append(SwingPoint(index=int(idx), price=float(highs[idx]), kind="high"))
        for idx in low_idx:
            points.append(SwingPoint(index=int(idx), price=float(lows[idx]), kind="low"))

        points.sort(key=lambda p: p.index)
        return points

    def build_swing_points_from_cache(
        self, swing_points_tuples: List[tuple],
    ) -> List[SwingPoint]:
        return [
            SwingPoint(index=int(idx), price=float(price), kind=kind)
            for idx, price, kind in swing_points_tuples
        ]

    def _alternates(self, *points: SwingPoint) -> bool:
        for i in range(len(points) - 1):
            if points[i].kind == points[i + 1].kind:
                return False
        return True

    def _ratio(self, numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-9:
            return 0.0
        return abs(numerator) / abs(denominator)

    def _in_range(self, value: float, rng: Tuple[float, float]) -> bool:
        return rng[0] <= value <= rng[1]

    def _check_impulse_rules(self, wave: List[SwingPoint]) -> Tuple[bool, dict]:
        if len(wave) != 6:
            return False, {}

        w0, w1, w2, w3, w4, w5 = wave
        is_bullish = w5.price > w0.price

        if not self._alternates(w0, w1, w2, w3, w4, w5):
            return False, {}

        len_w1 = abs(w1.price - w0.price)
        len_w2 = abs(w2.price - w1.price)
        len_w3 = abs(w3.price - w2.price)
        len_w4 = abs(w4.price - w3.price)
        len_w5 = abs(w5.price - w4.price)

        if len_w1 < 1e-9 or len_w3 < 1e-9 or len_w5 < 1e-9:
            return False, {}

        w2_w1 = len_w2 / len_w1
        w3_w1 = len_w3 / len_w1
        w4_w3 = len_w4 / len_w3
        w5_w3 = len_w5 / len_w3

        if not self._in_range(w2_w1, self.W2_W1_RANGE):
            return False, {}
        if not self._in_range(w3_w1, self.W3_W1_RANGE):
            return False, {}
        if not self._in_range(w4_w3, self.W4_W3_RANGE):
            return False, {}
        if not self._in_range(w5_w3, self.W5_W3_RANGE):
            return False, {}

        if is_bullish:
            if w4.price <= w1.price:
                return False, {}
            if w3.price <= w1.price:
                return False, {}
            if w5.price <= w3.price:
                return False, {}
        else:
            if w4.price >= w1.price:
                return False, {}
            if w3.price >= w1.price:
                return False, {}
            if w5.price >= w3.price:
                return False, {}

        ratios = {
            "w2_w1": w2_w1,
            "w3_w1": w3_w1,
            "w4_w3": w4_w3,
            "w5_w3": w5_w3,
        }

        return True, ratios

    def _calc_impulse_confidence(self, ratios: dict) -> float:
        ideal = {
            "w2_w1": 0.618,
            "w3_w1": 1.618,
            "w4_w3": 0.382,
            "w5_w3": 0.618,
        }
        scores = []
        for key, ideal_val in ideal.items():
            actual = ratios.get(key, 0.0)
            if ideal_val > 0:
                diff = abs(actual - ideal_val) / ideal_val
                score = max(0.0, 1.0 - diff)
                scores.append(score)
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _check_correction_rules(self, wave: List[SwingPoint]) -> Tuple[bool, dict]:
        if len(wave) != 4:
            return False, {}
        wa, wb, wc, _ = wave
        if not self._alternates(wa, wb, wc):
            return False, {}
        len_a = abs(wb.price - wa.price)
        len_b = abs(wc.price - wb.price)
        if len_a < 1e-9:
            return False, {}
        b_a = len_b / len_a
        if not self._in_range(b_a, self.B_A_RANGE):
            return False, {}
        return True, {"b_a": b_a}

    def _calc_correction_confidence(self, ratios: dict) -> float:
        b_a = ratios.get("b_a", 0.0)
        diff = abs(b_a - 0.618) / 0.618
        return max(0.0, 1.0 - diff)

    def detect_patterns(
        self,
        points: List[SwingPoint],
        total_bars: int,
        min_pattern_confidence: float = 0.0,
    ) -> List[WavePattern]:
        if len(points) < 6:
            return []

        patterns = []

        for i in range(len(points) - 5):
            wave = points[i:i + 6]
            age = total_bars - 1 - wave[-1].index
            if age < self.min_pattern_age or age > self.max_pattern_age:
                continue
            valid, ratios = self._check_impulse_rules(wave)
            if not valid:
                continue
            confidence = self._calc_impulse_confidence(ratios)
            if confidence < min_pattern_confidence:
                continue
            is_bullish = wave[-1].price > wave[0].price
            patterns.append(WavePattern(
                pattern_type="impulse",
                direction="bullish" if is_bullish else "bearish",
                points=wave, confidence=confidence, ratios=ratios, age_bars=age,
            ))

        for i in range(len(points) - 3):
            wave = points[i:i + 4]
            age = total_bars - 1 - wave[-1].index
            if age < self.min_pattern_age or age > self.max_pattern_age:
                continue
            valid, ratios = self._check_correction_rules(wave)
            if not valid:
                continue
            confidence = self._calc_correction_confidence(ratios)
            if confidence < min_pattern_confidence:
                continue
            is_bullish = wave[-1].price > wave[0].price
            patterns.append(WavePattern(
                pattern_type="correction",
                direction="bullish" if is_bullish else "bearish",
                points=wave, confidence=confidence, ratios=ratios, age_bars=age,
            ))

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns


class ElliottWaveAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        name: str = "elliott_wave",
        swing_window: int = 5,
        min_pattern_age: int = 3,
        max_pattern_age: int = 100,
        fibonacci_tolerance: float = 0.15,
        min_pattern_confidence: float = 0.55,
        min_confidence: float = 0.50,
        max_confidence: float = 0.90,
        use_trend_filter: bool = True,
        trend_penalty: float = 0.2,
    ):
        super().__init__(name)
        self._detector = ElliottWaveDetector(
            swing_window=swing_window,
            min_pattern_age=min_pattern_age,
            max_pattern_age=max_pattern_age,
            fibonacci_tolerance=fibonacci_tolerance,
        )
        self._min_pattern_confidence = min_pattern_confidence
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence
        self._use_trend_filter = use_trend_filter
        self._trend_filter = TrendFilter() if use_trend_filter else None
        self._trend_penalty = trend_penalty

    async def analyze(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
        bar_index: Optional[int] = None,
    ) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        try:
            ind_slice = None
            if indicators is not None:
                i = bar_index if bar_index is not None else indicators.n - 1
                if i < 50 or i >= indicators.n:
                    return None

                swing_tuples = indicators.get_swing_points(end_idx=i)
                points = self._detector.build_swing_points_from_cache(swing_tuples)
                total_bars = i + 1
                ind_slice = indicators.slice(i)
            else:
                points = self._detector.find_swing_points(data)
                total_bars = len(data)

            if len(points) < 6:
                return None

            patterns = self._detector.detect_patterns(
                points,
                total_bars=total_bars,
                min_pattern_confidence=self._min_pattern_confidence,
            )

            if not patterns:
                return None

            best = patterns[0]

            if best.confidence < self._min_confidence:
                return None

            direction = (
                SignalDirection.BUY if best.direction == "bullish"
                else SignalDirection.SELL
            )

            confidence = min(best.confidence, self._max_confidence)

            trend_info = "no_filter"
            if self._use_trend_filter and self._trend_filter is not None:
                if ind_slice is not None:
                    # Быстрый путь: SMA200 уже посчитана в IndicatorCache
                    trend = self._trend_filter.detect_from_cache(
                        sma_val=ind_slice.get("sma_200"),
                        current_price=ind_slice.get("current_price"),
                    )
                else:
                    # Fallback: пересчитываем SMA200 (медленный путь)
                    trend = self._trend_filter.detect(data)

                mult = self._trend_filter.apply_penalty(
                    direction, trend, self._trend_penalty,
                )
                confidence *= mult
                trend_info = trend.name

            if confidence < self._min_confidence:
                return None

            last_point = best.points[-1]
            reason = (
                f"{best.pattern_type.upper()} ({best.direction}) "
                f"end={last_point.price:.4f}, "
                f"conf={confidence:.2f}, "
                f"age={best.age_bars}bars, "
                f"ratios={best.ratios}, "
                f"trend={trend_info}"
            )

            return AnalyzerSignal(
                direction=direction,
                confidence=confidence,
                reason=reason,
                metadata={
                    "pattern_type": best.pattern_type,
                    "pattern_direction": best.direction,
                    "pattern_points_count": len(best.points),
                    "pattern_age_bars": best.age_bars,
                    "pattern_end_price": last_point.price,
                    "ratios": best.ratios,
                    "trend": trend_info,
                    "patterns_found": len(patterns),
                    "min_pattern_confidence": self._min_pattern_confidence,
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("ElliottWaveAnalyzer error: %s", e)
            return None