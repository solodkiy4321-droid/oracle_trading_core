"""Confluence Gate — фильтр принятия решения."""

import logging
from enum import Enum
from typing import Optional

from src.confluence.scoring import ScoringResult
from src.models import SignalDirection

logger = logging.getLogger(__name__)


class GateMode(Enum):
    """Режим строгости фильтра."""
    AGGRESSIVE = "aggressive"       # порог 0.50
    BALANCED = "balanced"           # порог 0.65 (по умолчанию)
    CONSERVATIVE = "conservative"   # порог 0.75


class ConfluenceGate:
    """
    Фильтр, пропускающий только сигналы с достаточным confluence score.

    Логика:
    - Если score >= threshold и direction != HOLD — сигнал проходит
    - Иначе — сигнал блокируется
    """

    MODE_THRESHOLDS = {
        GateMode.AGGRESSIVE: 0.50,
        GateMode.BALANCED: 0.65,
        GateMode.CONSERVATIVE: 0.75,
    }

    def __init__(
        self,
        mode: GateMode = GateMode.BALANCED,
        threshold: Optional[float] = None,
    ):
        """
        Args:
            mode: Режим строгости
            threshold: Явный порог (переопределяет mode)
        """
        self.mode = mode
        self.threshold = threshold if threshold is not None else self.MODE_THRESHOLDS[mode]

    def is_open(self, result: ScoringResult) -> bool:
        """
        Проверяет, проходит ли сигнал через фильтр.

        Args:
            result: Результат Scoring System

        Returns:
            True если сигнал проходит, False если блокируется
        """
        if result.direction == SignalDirection.HOLD:
            logger.debug(
                "Gate закрыт: direction=HOLD (score=%.3f)",
                result.confluence_score,
            )
            return False

        if result.confluence_score < self.threshold:
            logger.info(
                "Gate закрыт: score=%.3f < threshold=%.3f (%s)",
                result.confluence_score, self.threshold, self.mode.value,
            )
            return False

        logger.info(
            "Gate открыт: %s со score=%.3f (threshold=%.3f, %s)",
            result.direction.name, result.confluence_score,
            self.threshold, self.mode.value,
        )
        return True

    def get_threshold(self) -> float:
        """Возвращает текущий порог."""
        return self.threshold