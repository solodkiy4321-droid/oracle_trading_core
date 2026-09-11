"""Анализатор гармонических паттернов.

ПОЛНАЯ ПЕРЕРАБОТКА:
Классический harmonic ищет завершённые паттерны XABCD — но точка D
всегда в прошлом. Сигнал приходит "поздно".

Новый подход: детектировать ФОРМИРУЮЩИЕСЯ паттерны.
1. Найти X, A, B, C (4 точки)
2. Проверить 2 из 4 соотношений (AB_XA, BC_AB)
3. Спрогнозировать PRZ (Potential Reversal Zone) — где будет D
4. Дать сигнал, когда цена В PRZ

Это позволяет входить заранее — до формирования D.
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
    """Формирующийся паттерн (X, A, B, C найдены, D прогнозируется)."""
    pattern_type: str
    direction: str                 # 'bullish' / 'bearish'
    x: SwingPoint
    a: SwingPoint
    b: SwingPoint
    c: SwingPoint
    prz_low: float                 # Нижняя граница PRZ
    prz_high: float                # Верхняя граница PRZ
    prz_center: float              # Центр PRZ (прогноз точки D)
    base_confidence: float         # Уверенность в паттерне (0-1)
    age_bars: int                  # Возраст C (свежесть паттерна)
    ratios: dict                   # Соотношения AB_XA, BC_AB


class FormingHarmonicDetector:
    """
    Детектор ФОРМИРУЮЩИХСЯ гармонических паттернов.

    Ищет паттерны, где найдены X, A, B, C, но D ещё не сформирован.
    Прогнозирует PRZ — зону, где ожидается точка D.
    """

    # Спецификации паттернов. AD_XA — ключевое для прогноза D.
    PATTERN_SPECS = {
        "Gartley": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (0.786, 0.786),
            "name": "Gartley",
        },
        "Bat": {
            "AB_XA": (0.382, 0.500),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (0.886, 0.886),
            "name": "Bat",
        },
        "Butterfly": {
            "AB_XA": (0.786, 0.786),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (1.270, 1.618),
            "name": "Butterfly",
        },
        "Crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "AD_XA": (1.618, 1.618),
            "name": "Crab",
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
        """Находит swing points."""
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

    def _calc_confidence_2ratios(
        self, ab_xa: float, bc_ab: float, specs: dict,
    ) -> float:
        """Confidence на основе первых 2 соотношений (AB_XA, BC_AB)."""
        def score(value, rng):
            low, high = rng
            mid = (low + high) / 2
            span = (high - low) / 2 + 1e-9
            return max(0.0, 1.0 - abs(value - mid) / span)

        score_ab = score(ab_xa, specs["AB_XA"])
        score_bc = score(bc_ab, specs["BC_AB"])
        base = (score_ab + score_bc) / 2.0

        # Бонус если оба соотношения идеальны
        if score_ab > 0.7 and score_bc > 0.7:
            base += 0.15
        elif score_ab > 0.6 and score_bc > 0.6:
            base += 0.05

        return min(1.0, base)

    def _forecast_prz(
        self, x: SwingPoint, a: SwingPoint, c: SwingPoint, ad_xa_range: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        """
        Прогнозирует PRZ (зона точки D).

        D прогнозируется как A + AD_XA × (X − A) для bearish,
        или A − AD_XA × (A − X) для bullish.

        Возвращает (prz_low, prz_high, prz_center).
        """
        xa_range = abs(x.price - a.price)
        ad_low, ad_high = ad_xa_range

        # Определяем направление
        is_bullish = c.price < x.price  # C ниже X → bullish (ждём D ниже C)

        if is_bullish:
            # D находится НИЖЕ A на расстояние AD_XA × XA
            prz_high = a.price - ad_low * xa_range
            prz_low = a.price - ad_high * xa_range
        else:
            # D находится ВЫШЕ A
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
    ) -> List[FormingPattern]:
        """
        Ищет ФОРМИРУЮЩИЕСЯ паттерны (X, A, B, C найдены, D прогнозируется).

        Проверяет только 2 из 4 соотношений (AB_XA, BC_AB).
        Прогнозирует PRZ.
        """
        if len(points) < 4:
            return []

        patterns = []

        for i in range(len(points) - 3):
            x, a, b, c = points[i:i + 4]

            # Точки должны чередоваться
            if not self._alternates(x, a, b, c):
                continue

            # Возраст: C должен быть свежим
            age = total_bars - 1 - c.index
            if age < self.min_pattern_age or age > self.max_pattern_age:
                continue

            # Проверяем соотношения AB_XA и BC_AB
            for pattern_name, specs in self.PATTERN_SPECS.items():
                ab_xa = self._ratio(x.price, a.price, a.price, b.price)
                bc_ab = self._ratio(a.price, b.price, b.price, c.price)

                if not self._in_range(ab_xa, specs["AB_XA"]):
                    continue
                if not self._in_range(bc_ab, specs["BC_AB"]):
                    continue

                # Оба соотношения прошли — формируем паттерн
                base_conf = self._calc_confidence_2ratios(ab_xa, bc_ab, specs)

                # Прогнозируем PRZ
                prz_low, prz_high, prz_center = self._forecast_prz(
                    x, a, c, specs["AD_XA"],
                )

                is_bullish = c.price < x.price
                direction = "bullish" if is_bullish else "bearish"

                patterns.append(FormingPattern(
                    pattern_type=pattern_name,
                    direction=direction,
                    x=x, a=a, b=b, c=c,
                    prz_low=prz_low,
                    prz_high=prz_high,
                    prz_center=prz_center,
                    base_confidence=base_conf,
                    age_bars=age,
                    ratios={"AB_XA": ab_xa, "BC_AB": bc_ab},
                ))
                break  # один паттерн на 4 точки

        # Сортируем по confidence
        patterns.sort(key=lambda p: p.base_confidence, reverse=True)
        return patterns[:max_patterns]


class HarmonicAnalyzer(BaseAnalyzer):
    """
    Анализатор гармонических паттернов (ПОЛНАЯ ПЕРЕРАБОТКА).

    Ищет ФОРМИРУЮЩИЕСЯ паттерны и даёт сигнал, когда цена в PRZ.

    Преимущества:
    - Сигнал приходит ЗАРАНЕЕ (до формирования D)
    - Паттерны свежие (age C < 200 баров)
    - Confidence учитывает качество соотношений, близость к PRZ, тренд
    """

    def __init__(
        self,
        name: str = "harmonic",
        tolerance: float = 0.20,
        min_confidence: float = 0.30,
        max_distance_to_prz_pct: float = 0.03,   # цена должна быть в пределах 3% от PRZ
        min_pattern_age: int = 3,                 # минимум 3 бара с C
        max_pattern_age: int = 200,               # максимум 200 баров с C
        use_trend_filter: bool = True,
        trend_penalty: float = 0.2,
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
            )

            if not patterns:
                return None

            # Ищем паттерны, где цена находится в PRZ или рядом
            best_pattern = None
            best_confidence = 0.0

            for p in patterns:
                # Проверяем, рядом ли цена с PRZ
                if p.direction == "bullish":
                    # BUY-сигнал: цена должна быть НИЖЕ PRZ high или в PRZ
                    distance = abs(current_price - p.prz_center) / current_price
                    price_in_prz = current_price <= p.prz_high * 1.01  # небольшой запас
                else:
                    # SELL-сигнал: цена должна быть ВЫШЕ PRZ low или в PRZ
                    distance = abs(current_price - p.prz_center) / current_price
                    price_in_prz = current_price >= p.prz_low * 0.99

                if not price_in_prz:
                    continue

                if distance > self._max_distance_to_prz_pct:
                    continue

                # Confidence с учётом близости к PRZ
                proximity_bonus = max(0.0, 1.0 - distance / self._max_distance_to_prz_pct) * 0.2
                confidence = p.base_confidence + proximity_bonus
                confidence = min(1.0, confidence)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pattern = p

            if best_pattern is None:
                return None

            # Определяем направление
            direction = (
                SignalDirection.BUY if best_pattern.direction == "bullish"
                else SignalDirection.SELL
            )

            confidence = best_confidence

            # Тренд-фильтр
            trend_info = "no_filter"
            if self._use_trend_filter and self._trend_filter is not None:
                trend = self._trend_filter.detect(data)
                multiplier = self._trend_filter.apply_penalty(
                    direction, trend, self._trend_penalty
                )
                confidence *= multiplier
                trend_info = trend.name

            if confidence < self._min_confidence:
                return None

            # Формируем reason
            distance_to_prz = abs(current_price - best_pattern.prz_center) / current_price
            reasons = (
                f"{best_pattern.pattern_type} ({best_pattern.direction}) "
                f"PRZ={best_pattern.prz_center:.4f} "
                f"[{best_pattern.prz_low:.4f}-{best_pattern.prz_high:.4f}], "
                f"price={current_price:.4f}, "
                f"dist={distance_to_prz * 100:.2f}%, "
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
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("Ошибка в HarmonicAnalyzer: %s", e)
            return None