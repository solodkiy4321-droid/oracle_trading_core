"""Расчёт размера позиции на основе риска."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Результат расчёта размера позиции."""
    size: float               # Размер позиции (в единицах актива)
    risk_amount: float        # Сумма риска в валюте
    risk_pct: float           # Процент риска
    stop_distance: float      # Расстояние до стопа
    position_value: float     # Стоимость позиции
    leverage: float           # Используемое плечо


class PositionSizer:
    """
    Рассчитывает размер позиции на основе риска.

    Формула (классическая fixed-fractional):
    Position Size = (Equity × Risk%) / Stop Distance

    Это стандартный подход в профессиональном риск-менеджменте.
    Правило 1-2%: никогда не рискуйте более 1-2% капитала на сделку.
    """

    # Предупреждение о высоком плече
    HIGH_LEVERAGE_THRESHOLD = 3.0

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        max_position_pct: float = 1.0,
    ):
        """
        Args:
            risk_per_trade_pct: Процент риска на сделку (0.01 = 1%)
            max_position_pct: Максимальный размер позиции как % от equity
        """
        if not 0 < risk_per_trade_pct <= 0.1:
            raise ValueError(
                f"risk_per_trade_pct должен быть в (0, 0.1], "
                f"получено {risk_per_trade_pct}"
            )
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        equity: float,
        entry_price: float,
        stop_distance: float,
        risk_pct: Optional[float] = None,
    ) -> Optional[PositionSizeResult]:
        """
        Рассчитывает размер позиции.

        Args:
            equity: Текущий баланс счёта
            entry_price: Цена входа
            stop_distance: Расстояние от входа до стоп-лосса (в цене)
            risk_pct: Переопределение процента риска

        Returns:
            PositionSizeResult или None при ошибке
        """
        if equity <= 0:
            logger.warning("Equity должна быть > 0, получено %.2f", equity)
            return None

        if entry_price <= 0:
            logger.warning("Entry price должна быть > 0, получено %.4f", entry_price)
            return None

        if stop_distance <= 0:
            logger.warning(
                "Stop distance должна быть > 0, получено %.4f", stop_distance
            )
            return None

        risk_pct_used = risk_pct if risk_pct is not None else self.risk_per_trade_pct
        risk_amount = equity * risk_pct_used

        # Position Size = Risk Amount / Stop Distance
        position_size = risk_amount / stop_distance

        # Проверяем максимальный размер позиции
        position_value = position_size * entry_price
        max_value = equity * self.max_position_pct

        if position_value > max_value:
            position_size = max_value / entry_price
            position_value = position_size * entry_price
            logger.info(
                "Размер позиции ограничен max_position_pct: %.4f",
                self.max_position_pct,
            )

        # Плечо
        leverage = position_value / equity

        if leverage > self.HIGH_LEVERAGE_THRESHOLD:
            logger.warning(
                "Высокое плечо: %.2fx (порог %.2fx). "
                "Позиция больше счёта!",
                leverage, self.HIGH_LEVERAGE_THRESHOLD,
            )

        logger.debug(
            "Position: equity=%.2f, risk=%.2f (%.2f%%), "
            "size=%.6f, value=%.2f, leverage=%.2fx",
            equity, risk_amount, risk_pct_used * 100,
            position_size, position_value, leverage,
        )

        return PositionSizeResult(
            size=position_size,
            risk_amount=risk_amount,
            risk_pct=risk_pct_used,
            stop_distance=stop_distance,
            position_value=position_value,
            leverage=leverage,
        )