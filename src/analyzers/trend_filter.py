"""Фильтр тренда на основе SMA200."""

import logging
from enum import Enum
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Направление тренда."""
    UP = 1
    DOWN = -1
    NEUTRAL = 0


class TrendFilter:
    """
    Определяет направление тренда через SMA200.
    
    - Цена выше SMA200 → UP
    - Цена ниже SMA200 → DOWN
    - Цена в пределах 1% от SMA200 → NEUTRAL
    """

    def __init__(self, sma_period: int = 200, neutral_band_pct: float = 0.01):
        """
        Args:
            sma_period: Период SMA
            neutral_band_pct: Ширина нейтральной зоны (1% по умолчанию)
        """
        self.sma_period = sma_period
        self.neutral_band_pct = neutral_band_pct

    def detect(self, data: pd.DataFrame) -> TrendDirection:
        """
        Определяет направление тренда.
        
        Returns:
            TrendDirection.UP / DOWN / NEUTRAL
        """
        if data is None or len(data) < self.sma_period:
            logger.debug("Недостаточно данных для SMA%d", self.sma_period)
            return TrendDirection.NEUTRAL

        sma = ta.sma(data["close"], length=self.sma_period)
        if sma is None or len(sma) == 0:
            return TrendDirection.NEUTRAL

        sma_val = float(sma.iloc[-1])
        current_price = float(data["close"].iloc[-1])

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
        """
        Вычисляет штраф к уверенности за сигнал против тренда.
        
        Args:
            signal_direction: SignalDirection сигнала
            trend: Направление тренда
            penalty: Размер штрафа (0.3 = -30% к уверенности)
            
        Returns:
            Множитель уверенности (1.0 = без штрафа, 0.7 = -30%)
        """
        from src.models import SignalDirection

        if trend == TrendDirection.NEUTRAL:
            return 1.0

        if signal_direction == SignalDirection.BUY and trend == TrendDirection.UP:
            return 1.0  # сигнал по тренду
        if signal_direction == SignalDirection.SELL and trend == TrendDirection.DOWN:
            return 1.0  # сигнал по тренду

        # Сигнал против тренда — штраф
        return 1.0 - penalty