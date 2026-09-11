"""Расчёт размера позиции на основе риска."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Результат расчёта размера позиции."""
    size: float
    risk_amount: float
    risk_pct: float
    stop_distance: float
    position_value: float
    leverage: float


class PositionSizer:
    """
    Рассчитывает размер позиции на основе риска.

    Формула (классическая fixed-fractional):
    Position Size = (Equity × Risk%) / Stop Distance

    max_position_pct ограничивает максимальный размер позиции.
    """

    HIGH_LEVERAGE_THRESHOLD = 3.0

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        max_position_pct: float = 1.0,
    ):
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
        """Рассчитывает размер позиции."""
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

        leverage = position_value / equity

        if leverage > self.HIGH_LEVERAGE_THRESHOLD:
            logger.warning(
                "Высокое плечо: %.2fx (порог %.2fx)",
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