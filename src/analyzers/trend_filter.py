"""Фильтр тренда на основе SMA200.

Это НЕ анализатор. Это фильтр-штраф для других анализаторов.
Не регистрируется в SignalIntake, не даёт сигналов.
Используется в HarmonicAnalyzer и ElliottWaveAnalyzer для штрафа
сигналов против тренда.

Поддерживает два режима:
- detect(data): пересчитывает SMA200 через ta.sma (старый путь).
- detect_from_cache(sma_val, current_price): использует уже посчитанные
  значения из IndicatorCache (быстрый путь, без ta.sma).
"""

import logging
from enum import Enum
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0


class TrendFilter:
    """Определяет направление тренда через SMA200."""

    def __init__(self, sma_period: int = 200, neutral_band_pct: float = 0.01):
        self.sma_period = sma_period
        self.neutral_band_pct = neutral_band_pct

    def detect(self, data: pd.DataFrame) -> TrendDirection:
        """
        Определяет тренд по свежему расчёту SMA.

        Используется, когда IndicatorCache недоступен.
        """
        if data is None or len(data) < self.sma_period:
            return TrendDirection.NEUTRAL

        sma = ta.sma(data["close"], length=self.sma_period)
        if sma is None or len(sma) == 0:
            return TrendDirection.NEUTRAL

        sma_val = float(sma.iloc[-1])
        current_price = float(data["close"].iloc[-1])

        return self._classify(sma_val, current_price)

    def detect_from_cache(
        self,
        sma_val: Optional[float],
        current_price: Optional[float],
    ) -> TrendDirection:
        """
        Определяет тренд по значениям, уже посчитанным в IndicatorCache.

        Не вызывает ta.sma — это ключевое отличие от detect().
        """
        if sma_val is None or current_price is None:
            return TrendDirection.NEUTRAL
        return self._classify(sma_val, current_price)

    def _classify(
        self,
        sma_val: float,
        current_price: float,
    ) -> TrendDirection:
        """Общая логика классификации по SMA и текущей цене."""
        if sma_val <= 0:
            return TrendDirection.NEUTRAL

        distance_pct = (current_price - sma_val) / sma_val

        if abs(distance_pct) < self.neutral_band_pct:
            return TrendDirection.NEUTRAL
        elif distance_pct > 0:
            return TrendDirection.UP
        else:
            return TrendDirection.DOWN

    def apply_penalty(
        self,
        signal_direction,
        trend: TrendDirection,
        penalty: float = 0.3,
    ) -> float:
        from src.models import SignalDirection

        if trend == TrendDirection.NEUTRAL:
            return 1.0

        if signal_direction == SignalDirection.BUY and trend == TrendDirection.UP:
            return 1.0
        if signal_direction == SignalDirection.SELL and trend == TrendDirection.DOWN:
            return 1.0

        return 1.0 - penalty