"""Анализатор гармонических паттернов с фильтрами актуальности и тренда."""

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
    """Значимый максимум или минимум."""
    index: int
    price: float
    kind: str  # 'high' или 'low'


class HarmonicPatternDetector:
    """
    Детектор гармонических паттернов XABCD.

    Поддерживает: Gartley, Bat, Butterfly, Crab.
    Использует соотношения Фибоначчи по спецификациям Скотта Карни.
    """

    PATTERN_SPECS = {
        "Gartley": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.272, 1.618),
            "AD_XA": (0.786, 0.786),
        },
        "Bat": {
            "AB_XA": (0.382, 0.500),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.618),
            "AD_XA": (0.886, 0.886),
        },
        "Butterfly": {
            "AB_XA": (0.786, 0.786),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.618),
            "AD_XA": (1.270, 1.618),
        },
        "Crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (2.240, 3.618),
            "AD_XA": (1.618, 1.618),
        },
    }

    def __init__(self, tolerance: float = 0.15):
        self.tolerance = tolerance

    def find_swing_points(self, df: pd.DataFrame, window: int = 5) -> List[SwingPoint]:
        """Находит значимые максимумы и минимумы."""
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

    def detect_patterns(
        self,
        points: List[SwingPoint],
        current_price: float,
        max_patterns: int = 5,
        max_age_bars: int = 50,
        max_distance_pct: float = 0.10,
        total_bars: Optional[int] = None,
    ) -> List[dict]:
        """
        Ищет гармонические паттерны XABCD с фильтрами актуальности.

        Args:
            points: Swing points
            current_price: Текущая цена (последний close)
            max_patterns: Максимум паттернов в результате
            max_age_bars: Максимальный возраст паттерна в барах
            max_distance_pct: Максимальное расстояние от D до текущей цены
            total_bars: Общее количество баров (для расчёта возраста)
        """
        if len(points) < 5:
            return []

        patterns = []
        last_index = total_bars if total_bars is not None else points[-1].index

        for i in range(len(points) - 4):
            x, a, b, c, d = points[i:i + 5]

            # Фильтр 1: чередование high/low
            if not self._alternates(x, a, b, c, d):
                continue

            # Фильтр 2: возраст паттерна (D не старше max_age_bars)
            age = last_index - d.index
            if age > max_age_bars:
                continue

            # Фильтр 3: близость D к текущей цене
            distance_pct = abs(d.price - current_price) / current_price
            if distance_pct > max_distance_pct:
                continue

            # Определяем направление
            is_bullish = d.price < x.price

            # Проверяем соотношения
            for pattern_name, specs in self.PATTERN_SPECS.items():
                ab_xa = self._ratio(x.price, a.price, a.price, b.price)
                bc_ab = self._ratio(a.price, b.price, b.price, c.price)
                cd_bc = self._ratio(b.price, c.price, c.price, d.price)
                ad_xa = self._ratio(x.price, a.price, a.price, d.price)

                if (
                    self._in_range(ab_xa, specs["AB_XA"])
                    and self._in_range(bc_ab, specs["BC_AB"])
                    and self._in_range(cd_bc, specs["CD_BC"])
                    and self._in_range(ad_xa, specs["AD_XA"])
                ):
                    confidence = self._calc_confidence(ab_xa, bc_ab, cd_bc, ad_xa, specs)
                    # Бонус за свежесть паттерна
                    freshness_bonus = max(0.0, 1.0 - age / max_age_bars) * 0.1
                    confidence = min(1.0, confidence + freshness_bonus)

                    patterns.append({
                        "type": pattern_name,
                        "direction": "bullish" if is_bullish else "bearish",
                        "confidence": confidence,
                        "points": [x, a, b, c, d],
                        "age_bars": age,
                        "distance_pct": distance_pct,
                        "ratios": {
                            "AB_XA": ab_xa,
                            "BC_AB": bc_ab,
                            "CD_BC": cd_bc,
                            "AD_XA": ad_xa,
                        },
                    })
                    break

        patterns.sort(key=lambda p: p["confidence"], reverse=True)
        return patterns[:max_patterns]

    def _alternates(self, *points: SwingPoint) -> bool:
        for i in range(len(points) - 1):
            if points[i].kind == points[i + 1].kind:
                return False
        return True

    def _calc_confidence(
        self, ab_xa: float, bc_ab: float, cd_bc: float, ad_xa: float, specs: dict
    ) -> float:
        def score(value, rng):
            low, high = rng
            mid = (low + high) / 2
            span = (high - low) / 2 + 1e-9
            return max(0.0, 1.0 - abs(value - mid) / span)

        scores = [
            score(ab_xa, specs["AB_XA"]),
            score(bc_ab, specs["BC_AB"]),
            score(cd_bc, specs["CD_BC"]),
            score(ad_xa, specs["AD_XA"]),
        ]
        return float(np.mean(scores))


class HarmonicAnalyzer(BaseAnalyzer):
    """
    Анализатор гармонических паттернов с фильтрами актуальности и тренда.

    Оптимизированные параметры (средний уровень строгости):
    - min_confidence=0.35
    - max_age_bars=50
    - max_distance_pct=0.10
    """

    def __init__(
        self,
        name: str = "harmonic",
        tolerance: float = 0.15,
        min_confidence: float = 0.35,
        max_age_bars: int = 50,
        max_distance_pct: float = 0.10,
        use_trend_filter: bool = True,
        trend_penalty: float = 0.3,
    ):
        """
        Args:
            name: Имя анализатора
            tolerance: Допустимое отклонение от пропорций Фибоначчи
            min_confidence: Минимальная уверенность для принятия сигнала
            max_age_bars: Максимальный возраст паттерна в барах
            max_distance_pct: Максимальное расстояние от D до текущей цены
            use_trend_filter: Использовать ли фильтр тренда
            trend_penalty: Штраф к уверенности за сигнал против тренда
        """
        super().__init__(name)
        self._detector = HarmonicPatternDetector(tolerance=tolerance)
        self._min_confidence = min_confidence
        self._max_age_bars = max_age_bars
        self._max_distance_pct = max_distance_pct
        self._use_trend_filter = use_trend_filter
        self._trend_filter = TrendFilter() if use_trend_filter else None
        self._trend_penalty = trend_penalty

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        try:
            current_price = float(data["close"].iloc[-1])
            points = self._detector.find_swing_points(data, window=5)
            if len(points) < 5:
                return None

            patterns = self._detector.detect_patterns(
                points,
                current_price=current_price,
                max_patterns=3,
                max_age_bars=self._max_age_bars,
                max_distance_pct=self._max_distance_pct,
                total_bars=len(data) - 1,
            )

            if not patterns:
                logger.debug(
                    "Гармонические паттерны не найдены для %d баров", len(data)
                )
                return None

            # Фильтр по минимальной уверенности
            best = patterns[0]
            if best["confidence"] < self._min_confidence:
                logger.debug(
                    "Паттерн %s отвергнут: confidence=%.2f < %.2f",
                    best["type"], best["confidence"], self._min_confidence,
                )
                return None

            direction = (
                SignalDirection.BUY if best["direction"] == "bullish"
                else SignalDirection.SELL
            )

            # Применяем штраф за сигнал против тренда
            confidence = best["confidence"]
            trend_info = "no_filter"
            if self._use_trend_filter and self._trend_filter is not None:
                trend = self._trend_filter.detect(data)
                multiplier = self._trend_filter.apply_penalty(
                    direction, trend, self._trend_penalty
                )
                confidence *= multiplier
                trend_info = trend.name

                if multiplier < 1.0:
                    logger.debug(
                        "Паттерн %s против тренда %s: confidence %.2f -> %.2f",
                        best["type"], trend.name, best["confidence"], confidence,
                    )

            d_point = best["points"][-1]
            reasons = (
                f"{best['type']} ({best['direction']}) "
                f"D={d_point.price:.4f}, "
                f"conf={confidence:.2f}, "
                f"trend={trend_info}"
            )

            return AnalyzerSignal(
                direction=direction,
                confidence=confidence,
                reason=reasons,
                metadata={
                    "pattern_type": best["type"],
                    "pattern_direction": best["direction"],
                    "pattern_price": d_point.price,
                    "pattern_age_bars": best["age_bars"],
                    "pattern_distance_pct": best["distance_pct"],
                    "current_price": current_price,
                    "trend": trend_info,
                    "ratios": best["ratios"],
                    "patterns_found": len(patterns),
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("Ошибка в HarmonicAnalyzer: %s", e)
            return None