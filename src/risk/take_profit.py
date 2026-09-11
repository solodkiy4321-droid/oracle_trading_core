"""Расчёт тейк-профитов на основе соотношения риск/прибыль."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.models import SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class TakeProfitLevel:
    """Один уровень тейк-профита."""
    price: float
    distance: float
    distance_pct: float
    ratio: float              # Соотношение R:R
    percentage: float         # Какой процент позиции закрыть


@dataclass
class TakeProfitResult:
    """Результат расчёта тейк-профитов."""
    levels: List[TakeProfitLevel]
    primary_price: float      # Основной TP (первый уровень)


class TakeProfitCalculator:
    """
    Рассчитывает тейк-профиты на основе соотношения риск/прибыль.

    Формула:
    - Long:  TP = Entry + (SL_Distance × Ratio)
    - Short: TP = Entry - (SL_Distance × Ratio)

    Поддерживает несколько уровней с частичным закрытием позиции.
    """

    def __init__(
        self,
        default_ratio: float = 2.0,
        levels: Optional[List[tuple]] = None,
    ):
        """
        Args:
            default_ratio: Соотношение риск/прибыль по умолчанию (1:2)
            levels: Список кортежей (ratio, percentage) для частичного закрытия
                    По умолчанию: [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]
        """
        self.default_ratio = default_ratio
        self.levels = levels or [(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)]

    def calculate(
        self,
        entry_price: float,
        sl_distance: float,
        direction: SignalDirection,
        ratios: Optional[List[float]] = None,
    ) -> Optional[TakeProfitResult]:
        """
        Рассчитывает тейк-профиты.

        Args:
            entry_price: Цена входа
            sl_distance: Расстояние от входа до стоп-лосса (в цене)
            direction: Направление сделки (BUY / SELL)
            ratios: Список соотношений R:R (переопределяет self.levels)

        Returns:
            TakeProfitResult или None при ошибке
        """
        if direction == SignalDirection.HOLD:
            return None

        if sl_distance <= 0:
            logger.warning("SL distance должна быть > 0, получено %.4f", sl_distance)
            return None

        if entry_price <= 0:
            return None

        # Определяем уровни
        if ratios is not None:
            level_specs = [(r, 1.0 / len(ratios)) for r in ratios]
        else:
            level_specs = self.levels

        levels = []
        for ratio, percentage in level_specs:
            distance = sl_distance * ratio
            if direction == SignalDirection.BUY:
                tp_price = entry_price + distance
            else:  # SELL
                tp_price = entry_price - distance

            distance_pct = distance / entry_price

            levels.append(TakeProfitLevel(
                price=tp_price,
                distance=distance,
                distance_pct=distance_pct,
                ratio=ratio,
                percentage=percentage,
            ))

        if not levels:
            return None

        logger.debug(
            "TP: entry=%.4f, direction=%s, levels=%d",
            entry_price, direction.name, len(levels),
        )

        return TakeProfitResult(
            levels=levels,
            primary_price=levels[0].price,
        )