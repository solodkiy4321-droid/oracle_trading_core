"""Расчёт тейк-профита на основе соотношения риск/прибыль.

Один уровень TP с RR = default_ratio.
Без частичных закрытий — для чистоты диагностики.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.models import SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TakeProfitLevel:
    """Уровень тейк-профита."""
    price: float
    distance: float
    distance_pct: float
    ratio: float
    percentage: float


@dataclass
class TakeProfitResult:
    """Результат расчёта тейк-профита."""
    levels: List[TakeProfitLevel]
    primary_price: float


class TakeProfitCalculator:
    """Рассчитывает тейк-профит на основе RR."""

    def __init__(self, default_ratio: float = 2.0):
        if default_ratio <= 0:
            raise ValueError(
                f"default_ratio должен быть > 0, получено {default_ratio}"
            )
        self.default_ratio = default_ratio

    def calculate(
        self,
        entry_price: float,
        sl_distance: float,
        direction: SignalDirection,
        ratios: Optional[List[float]] = None,
    ) -> Optional[TakeProfitResult]:
        if direction == SignalDirection.HOLD:
            return None

        if sl_distance <= 0:
            logger.warning(
                "SL distance должна быть > 0, получено %.4f", sl_distance,
            )
            return None

        if entry_price <= 0:
            return None

        if ratios is not None and len(ratios) > 0:
            ratio = float(ratios[0])
        else:
            ratio = self.default_ratio

        distance = sl_distance * ratio

        if direction == SignalDirection.BUY:
            tp_price = entry_price + distance
        else:
            tp_price = entry_price - distance

        distance_pct = distance / entry_price

        level = TakeProfitLevel(
            price=tp_price,
            distance=distance,
            distance_pct=distance_pct,
            ratio=ratio,
            percentage=1.0,
        )

        logger.debug(
            "TP: entry=%.4f, direction=%s, ratio=%.2f, TP=%.4f",
            entry_price, direction.name, ratio, tp_price,
        )

        return TakeProfitResult(
            levels=[level],
            primary_price=tp_price,
        )