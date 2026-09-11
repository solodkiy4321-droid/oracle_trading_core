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
    bull_score: float          # 0.0 - 1.0
    bear_score: float          # 0.0 - 1.0
    confluence_score: float    # 0.0 - 1.0 (итоговый)
    reasons: List[str] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    total_weight_used: float = 0.0


class ScoringSystem:
    """
    Вычисляет confluence score на основе сигналов анализаторов.

    КЛЮЧЕВОЙ ПРИНЦИП:
    Score = Σ (confidence_i × weight_i)

    Score НЕ нормализуется по сумме весов участвующих анализаторов.
    Это означает, что если сигнал пришёл только от одного анализатора
    с весом 0.30, максимальный score = 0.30 × 1.0 = 0.30.

    Внутренний threshold определяет, когда bull_score/bear_score
    превращается в направление (BUY/SELL), а не в HOLD.

    С новыми весами (indicators=0.30, harmonic=0.30, support_resistance=0.40)
    два анализатора могут дать максимум 0.60 (при confidence=1.0 каждый).
    Поэтому внутренний threshold снижен до 0.40, чтобы два сильных
    анализатора могли определить направление.
    """

    def __init__(self, threshold: float = 0.40):
        """
        Args:
            threshold: Минимальный порог для определения направления
                       (если bull_score и bear_score ниже — HOLD)
        """
        self.threshold = threshold

    def compute(
        self,
        signals: List[AnalyzerSignal],
        weights: Dict[str, float],
    ) -> ScoringResult:
        """
        Вычисляет confluence score.

        Args:
            signals: Список валидных сигналов от анализаторов
            weights: Веса анализаторов (сумма <= 1.0)

        Returns:
            ScoringResult с направлением, score и обоснованием
        """
        if not signals:
            return ScoringResult(
                direction=SignalDirection.HOLD,
                bull_score=0.0,
                bear_score=0.0,
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
                logger.debug("Пропущен сигнал от %s: вес = 0", signal.source)
                continue

            total_weight += weight
            contribution = signal.confidence * weight

            if signal.direction == SignalDirection.BUY:
                bull_score += contribution
                reasons.append(
                    f"{signal.source}: BUY (conf={signal.confidence:.2f}, "
                    f"weight={weight:.2f}, contrib={contribution:.3f})"
                )
                breakdown[signal.source] = contribution
            elif signal.direction == SignalDirection.SELL:
                bear_score += contribution
                reasons.append(
                    f"{signal.source}: SELL (conf={signal.confidence:.2f}, "
                    f"weight={weight:.2f}, contrib={contribution:.3f})"
                )
                breakdown[signal.source] = -contribution
            else:
                reasons.append(f"{signal.source}: HOLD")

        # ВАЖНО: не нормализуем по total_weight!
        # bull_score и bear_score остаются в абсолютных значениях,
        # ограниченных суммой весов участвующих анализаторов.

        # Определяем направление и итоговый score
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
            "Confluence: direction=%s, bull=%.3f, bear=%.3f, "
            "score=%.3f, total_weight=%.2f",
            direction.name, bull_score, bear_score,
            confluence_score, total_weight,
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