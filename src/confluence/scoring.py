"""Система вычисления совокупного confluence score.

Вход: список сигналов + словарь весов (сумма = 1.0 после
нормализации в WeightManager).

Выход: ScoringResult, где confluence_score — взвешенное среднее
confidence по направлению, всегда в диапазоне [0, 1]. Это делает
score сравнимым с порогами ConfluenceGate (0.50 / 0.65 / 0.75).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict

from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Результат вычисления confluence score."""
    direction: SignalDirection
    bull_score: float
    bear_score: float
    confluence_score: float
    reasons: List[str] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    total_weight_used: float = 0.0


class ScoringSystem:
    """
    Вычисляет confluence score.

    Сохраняет ПОЛНЫЙ reason от анализатора
    (включая детали — паттерн, PRZ, уровень и т.д.).

    score нормализуется на total_weight, поэтому:
    - 0.0 — сигналов нет или они нулевой уверенности,
    - 1.0 — все активные анализаторы дают confidence=1.0
      в одну сторону.
    """

    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def compute(
        self,
        signals: List[AnalyzerSignal],
        weights: Dict[str, float],
    ) -> ScoringResult:
        if not signals:
            return ScoringResult(
                direction=SignalDirection.HOLD,
                bull_score=0.0, bear_score=0.0,
                confluence_score=0.0,
                reasons=["Нет сигналов"],
                total_weight_used=0.0,
            )

        bull_raw = 0.0
        bear_raw = 0.0
        total_weight = 0.0
        reasons: List[str] = []
        breakdown: Dict[str, float] = {}

        for signal in signals:
            weight = weights.get(signal.source, 0.0)
            if weight <= 0:
                continue

            total_weight += weight
            contribution = signal.confidence * weight

            detailed_reason = (
                f"{signal.source}: {signal.direction.name} "
                f"(conf={signal.confidence:.2f}, weight={weight:.2f}, "
                f"contrib={contribution:.3f}) — {signal.reason}"
            )

            if signal.direction == SignalDirection.BUY:
                bull_raw += contribution
                reasons.append(detailed_reason)
                breakdown[signal.source] = contribution
            elif signal.direction == SignalDirection.SELL:
                bear_raw += contribution
                reasons.append(detailed_reason)
                breakdown[signal.source] = -contribution
            else:
                reasons.append(f"{signal.source}: HOLD")

        # Нормализуем на суммарный вес активных сигналов.
        # Если total_weight == 0 — никто не дал валидный сигнал.
        if total_weight > 0:
            bull_score = bull_raw / total_weight
            bear_score = bear_raw / total_weight
        else:
            bull_score = 0.0
            bear_score = 0.0

        if bull_score > bear_score and bull_score >= self.threshold:
            direction = SignalDirection.BUY
            confluence_score = bull_score
        elif bear_score > bull_score and bear_score >= self.threshold:
            direction = SignalDirection.SELL
            confluence_score = bear_score
        else:
            direction = SignalDirection.HOLD
            confluence_score = max(bull_score, bear_score)

        logger.info(
            "Confluence: direction=%s, bull=%.3f, bear=%.3f, score=%.3f, "
            "total_weight=%.3f",
            direction.name, bull_score, bear_score, confluence_score,
            total_weight,
        )

        return ScoringResult(
            direction=direction,
            bull_score=bull_score,
            bear_score=bear_score,
            confluence_score=confluence_score,
            reasons=reasons,
            breakdown=breakdown,
            total_weight_used=total_weight,
        )