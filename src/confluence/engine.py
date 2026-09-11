"""Главный класс Confluence-движка."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.confluence.regime_detector import RegimeDetector, MarketRegime
from src.confluence.weight_manager import WeightManager
from src.confluence.scoring import ScoringSystem, ScoringResult
from src.confluence.gate import ConfluenceGate, GateMode
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceDecision:
    """Итоговое решение Confluence-движка."""
    direction: SignalDirection
    confluence_score: float
    regime: MarketRegime
    passed: bool                     # прошёл ли сигнал через Gate
    weights_used: dict               # веса, использованные при расчёте
    reasons: List[str]
    bull_score: float = 0.0
    bear_score: float = 0.0


class ConfluenceEngine:
    """
    Главный класс Confluence-движка.

    Объединяет:
    - RegimeDetector: определение рыночного режима
    - WeightManager: управление весами анализаторов
    - ScoringSystem: вычисление confluence score
    - ConfluenceGate: фильтрация сигналов
    """

    def __init__(
        self,
        weight_manager: Optional[WeightManager] = None,
        regime_detector: Optional[RegimeDetector] = None,
        scoring_system: Optional[ScoringSystem] = None,
        gate: Optional[ConfluenceGate] = None,
    ):
        """
        Args:
            weight_manager: Менеджер весов (по умолчанию — с базовыми весами)
            regime_detector: Детектор режима (по умолчанию — SMA200 + ADX)
            scoring_system: Система scoring (по умолчанию — threshold=0.40)
            gate: Фильтр (по умолчанию — BALANCED)
        """
        self._weight_manager = weight_manager or WeightManager()
        self._regime_detector = regime_detector or RegimeDetector()
        self._scoring_system = scoring_system or ScoringSystem(threshold=0.40)
        self._gate = gate or ConfluenceGate(mode=GateMode.BALANCED)

    def decide(
        self,
        signals: List[AnalyzerSignal],
        data: pd.DataFrame,
    ) -> ConfluenceDecision:
        """
        Принимает итоговое решение на основе сигналов и рыночных данных.

        Args:
            signals: Список валидных сигналов от анализаторов
            data: OHLCV DataFrame (для определения режима)

        Returns:
            ConfluenceDecision с итоговым решением
        """
        if not signals:
            logger.info("Нет сигналов для обработки")
            return ConfluenceDecision(
                direction=SignalDirection.HOLD,
                confluence_score=0.0,
                regime=MarketRegime.CHOP,
                passed=False,
                weights_used={},
                reasons=["Нет сигналов"],
            )

        # 1. Определяем рыночный режим
        regime = self._regime_detector.detect(data)
        logger.info("Рыночный режим: %s", regime.value)

        # 2. Получаем веса с учётом режима
        analyzer_names = list({s.source for s in signals})
        weights = self._weight_manager.get_weights(regime, analyzers=analyzer_names)
        logger.info("Веса: %s", {k: f"{v:.3f}" for k, v in weights.items()})

        # 3. Вычисляем confluence score
        scoring = self._scoring_system.compute(signals, weights)

        # 4. Проверяем через Gate
        passed = self._gate.is_open(scoring)

        # 5. Формируем итоговое решение
        return ConfluenceDecision(
            direction=scoring.direction if passed else SignalDirection.HOLD,
            confluence_score=scoring.confluence_score,
            regime=regime,
            passed=passed,
            weights_used=weights,
            reasons=scoring.reasons,
            bull_score=scoring.bull_score,
            bear_score=scoring.bear_score,
        )

    def set_gate_mode(self, mode: GateMode) -> None:
        """Изменить режим строгости фильтра."""
        self._gate = ConfluenceGate(mode=mode)
        logger.info(
            "Gate режим изменён на %s (threshold=%.2f)",
            mode.value, self._gate.get_threshold(),
        )

    def set_base_weights(self, weights: dict) -> None:
        """Переопределить базовые веса анализаторов."""
        self._weight_manager = WeightManager(base_weights=weights)
        logger.info("Базовые веса обновлены: %s", weights)