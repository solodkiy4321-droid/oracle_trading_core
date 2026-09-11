"""Система вычисления совокупного confluence score."""

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

    КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: сохраняем ПОЛНЫЙ reason от анализатора
    (включая детали — паттерн, PRZ, уровень и т.д.).
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

        bull_score = 0.0
        bear_score = 0.0
        total_weight = 0.0
        reasons = []
        breakdown = {}

        for signal in signals:
            weight = weights.get(signal.source, 0.0)
            if weight <= 0:
                continue

            total_weight += weight
            contribution = signal.confidence * weight

            # ВАЖНО: используем signal.reason (детальный) вместо
            # синтетического "{source}: {direction} (conf=..., ...)"
            # Это сохраняет паттерны harmonic, уровни S/R и т.д.
            detailed_reason = (
                f"{signal.source}: {signal.direction.name} "
                f"(conf={signal.confidence:.2f}, weight={weight:.2f}, "
                f"contrib={contribution:.3f}) — {signal.reason}"
            )

            if signal.direction == SignalDirection.BUY:
                bull_score += contribution
                reasons.append(detailed_reason)
                breakdown[signal.source] = contribution
            elif signal.direction == SignalDirection.SELL:
                bear_score += contribution
                reasons.append(detailed_reason)
                breakdown[signal.source] = -contribution
            else:
                reasons.append(f"{signal.source}: HOLD")

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
            "Confluence: direction=%s, bull=%.3f, bear=%.3f, score=%.3f",
            direction.name, bull_score, bear_score, confluence_score,
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