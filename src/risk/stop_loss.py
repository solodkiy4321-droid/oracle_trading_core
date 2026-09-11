"""Расчёт стоп-лосса на основе ATR."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta_classic as ta

from src.models import SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class StopLossResult:
    """Результат расчёта стоп-лосса."""
    price: float              # Цена стоп-лосса
    distance: float           # Расстояние от входа до SL (в цене)
    distance_pct: float       # Расстояние в процентах
    atr_value: float          # Значение ATR
    multiplier: float         # Использованный множитель


class StopLossCalculator:
    """
    Рассчитывает стоп-лосс на основе ATR.

    Формула:
    - Long:  SL = Entry - (ATR × Multiplier)
    - Short: SL = Entry + (ATR × Multiplier)

    Это классический подход, описанный в профессиональных
    индикаторах риск-менеджмента.
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        """
        Args:
            atr_period: Период ATR
            atr_multiplier: Множитель ATR для расчёта расстояния
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calculate_atr(self, data: pd.DataFrame) -> Optional[float]:
        """Рассчитывает текущее значение ATR."""
        if data is None or len(data) < self.atr_period + 1:
            return None

        atr = ta.atr(
            data["high"], data["low"], data["close"],
            length=self.atr_period,
        )
        if atr is None or len(atr) == 0:
            return None

        return float(atr.iloc[-1])

    def calculate(
        self,
        entry_price: float,
        direction: SignalDirection,
        data: pd.DataFrame,
        multiplier: Optional[float] = None,
    ) -> Optional[StopLossResult]:
        """
        Рассчитывает стоп-лосс.

        Args:
            entry_price: Цена входа
            direction: Направление сделки (BUY / SELL)
            data: OHLCV DataFrame для расчёта ATR
            multiplier: Переопределение множителя ATR

        Returns:
            StopLossResult или None при ошибке
        """
        if direction == SignalDirection.HOLD:
            return None

        atr_value = self.calculate_atr(data)
        if atr_value is None or atr_value <= 0:
            logger.warning("Не удалось рассчитать ATR")
            return None

        mult = multiplier if multiplier is not None else self.atr_multiplier
        distance = atr_value * mult

        if direction == SignalDirection.BUY:
            sl_price = entry_price - distance
        else:  # SELL
            sl_price = entry_price + distance

        if entry_price <= 0:
            return None

        distance_pct = distance / entry_price

        logger.debug(
            "SL: entry=%.4f, direction=%s, ATR=%.4f, mult=%.1f, "
            "SL=%.4f, distance=%.4f (%.2f%%)",
            entry_price, direction.name, atr_value, mult,
            sl_price, distance, distance_pct * 100,
        )

        return StopLossResult(
            price=sl_price,
            distance=distance,
            distance_pct=distance_pct,
            atr_value=atr_value,
            multiplier=mult,
        )