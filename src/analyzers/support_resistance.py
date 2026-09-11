"""Анализатор уровней поддержки и сопротивления на основе KDE."""

import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class SRLevel:
    """Уровень поддержки или сопротивления."""
    price: float
    strength: float
    touches: int
    kind: str


class SRDetector:
    """
    Детектор уровней S/R на основе KDE-кластеризации swing points.

    Простая версия — без возраста, объёма и других усложнений.
    """

    def __init__(
        self,
        swing_window: int = 5,
        kde_bandwidth: float = 0.5,
        min_touches: int = 2,
        max_levels: int = 10,
    ):
        self.swing_window = swing_window
        self.kde_bandwidth = kde_bandwidth
        self.min_touches = min_touches
        self.max_levels = max_levels

    def find_swing_points(self, df: pd.DataFrame) -> List[tuple]:
        """Находит swing points: (index, price, kind)."""
        highs = df["high"].values
        lows = df["low"].values

        high_idx, _ = find_peaks(highs, distance=self.swing_window)
        low_idx, _ = find_peaks(-lows, distance=self.swing_window)

        points = []
        for idx in high_idx:
            points.append((int(idx), float(highs[idx]), "high"))
        for idx in low_idx:
            points.append((int(idx), float(lows[idx]), "low"))

        points.sort(key=lambda p: p[0])
        return points

    def detect_levels(self, df: pd.DataFrame) -> List[SRLevel]:
        """Находит уровни S/R через KDE."""
        if df is None or len(df) < 50:
            return []

        swing_points = self.find_swing_points(df)
        if len(swing_points) < self.min_touches:
            return []

        prices = np.array([p[1] for p in swing_points])

        price_min, price_max = prices.min(), prices.max()
        price_range = price_max - price_min
        if price_range <= 0:
            return []

        bandwidth = self.kde_bandwidth * price_range / 100.0
        if bandwidth <= 0:
            bandwidth = price_range * 0.01

        try:
            kde = gaussian_kde(prices, bw_method=bandwidth / prices.std())
        except Exception as e:
            logger.warning("Ошибка KDE: %s", e)
            return []

        grid = np.linspace(price_min, price_max, 200)
        density = kde(grid)

        peak_idx, properties = find_peaks(
            density,
            height=density.max() * 0.1,
            distance=5,
        )

        if len(peak_idx) == 0:
            return []

        peak_heights = properties["peak_heights"]
        sorted_idx = np.argsort(peak_heights)[::-1]

        current_price = float(df["close"].iloc[-1])
        levels = []

        for i in sorted_idx[: self.max_levels * 2]:
            peak_pos = peak_idx[i]
            level_price = float(grid[peak_pos])
            strength = float(peak_heights[i] / peak_heights.max())

            touches = int(np.sum(np.abs(prices - level_price) <= bandwidth))
            if touches < self.min_touches:
                continue

            kind = "support" if level_price < current_price else "resistance"

            levels.append(SRLevel(
                price=level_price,
                strength=strength,
                touches=touches,
                kind=kind,
            ))

        levels.sort(key=lambda x: x.strength, reverse=True)
        return levels[: self.max_levels]


class SRAnalyzer(BaseAnalyzer):
    """
    Простой анализатор уровней S/R.

    Возвращает confidence на основе силы уровня и близости к нему.
    Без trend penalty, без композиции, без дополнительных фильтров.

    Это версия, которая работала: 28 сделок, win rate 50%, P&L +4.14%.
    """

    def __init__(
        self,
        name: str = "support_resistance",
        swing_window: int = 5,
        kde_bandwidth: float = 0.5,
        min_touches: int = 2,
        max_levels: int = 10,
        proximity_pct: float = 0.02,
        min_strength: float = 0.4,
        min_confidence: float = 0.3,
    ):
        super().__init__(name)
        self._detector = SRDetector(
            swing_window=swing_window,
            kde_bandwidth=kde_bandwidth,
            min_touches=min_touches,
            max_levels=max_levels,
        )
        self._proximity_pct = proximity_pct
        self._min_strength = min_strength
        self._min_confidence = min_confidence

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        try:
            current_price = float(data["close"].iloc[-1])
            levels = self._detector.detect_levels(data)

            if not levels:
                logger.debug("Уровни S/R не найдены")
                return None

            best_level = None
            best_distance_pct = float("inf")

            for level in levels:
                if level.strength < self._min_strength:
                    continue
                distance_pct = abs(level.price - current_price) / current_price
                if distance_pct > self._proximity_pct:
                    continue
                if distance_pct < best_distance_pct:
                    best_distance_pct = distance_pct
                    best_level = level

            if best_level is None:
                logger.debug(
                    "Цена %.4f не рядом ни с одним сильным уровнем",
                    current_price,
                )
                return None

            if best_level.kind == "support":
                direction = SignalDirection.BUY
            else:
                direction = SignalDirection.SELL

            proximity_score = 1.0 - (best_distance_pct / self._proximity_pct)
            confidence = best_level.strength * 0.7 + proximity_score * 0.3
            confidence = min(confidence, 1.0)

            if confidence < self._min_confidence:
                return None

            reason = (
                f"{best_level.kind.upper()} at {best_level.price:.4f} "
                f"(strength={best_level.strength:.2f}, "
                f"touches={best_level.touches}, "
                f"distance={best_distance_pct * 100:.2f}%)"
            )

            return AnalyzerSignal(
                direction=direction,
                confidence=confidence,
                reason=reason,
                metadata={
                    "level_price": best_level.price,
                    "level_strength": best_level.strength,
                    "level_touches": best_level.touches,
                    "level_kind": best_level.kind,
                    "distance_pct": best_distance_pct,
                    "current_price": current_price,
                    "total_levels": len(levels),
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("Ошибка в SRAnalyzer: %s", e)
            return None