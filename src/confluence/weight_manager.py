"""Менеджер весов анализаторов с адаптацией под рыночный режим.

Активные анализаторы (4):
- trend
- elliott_wave
- volatility
- volume

Веса ВСЕГДА нормализуются к сумме 1.0 после применения
regime-множителей и фильтрации по активным анализаторам.
Это гарантирует, что confluence_score из ScoringSystem
находится в диапазоне [0, 1] и сравним с фиксированными
порогами ConfluenceGate.
"""

import logging
from typing import Dict, Optional

from src.confluence.regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class WeightManager:

    DEFAULT_BASE_WEIGHTS = {
        "trend": 0.35,
        "elliott_wave": 0.30,
        "volatility": 0.20,
        "volume": 0.15,
    }

    REGIME_MULTIPLIERS = {
        MarketRegime.BULL: {
            "trend": 1.2,
            "elliott_wave": 0.9,
            "volatility": 1.0,
            "volume": 1.0,
        },
        MarketRegime.BEAR: {
            "trend": 1.2,
            "elliott_wave": 0.9,
            "volatility": 1.0,
            "volume": 1.0,
        },
        MarketRegime.CHOP: {
            "trend": 0.7,
            "elliott_wave": 1.2,
            "volatility": 1.2,
            "volume": 1.1,
        },
    }

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        if base_weights is not None:
            self._base_weights = dict(base_weights)
        else:
            self._base_weights = dict(self.DEFAULT_BASE_WEIGHTS)
        self._validate_weights(self._base_weights)

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """
        Проверяет корректность базовых весов.

        Если сумма весов отличается от 1.0 — НОРМАЛИЗУЕТ, а не падает.
        Это защищает от опечаток в per-symbol профилях и от
        накопления ошибок округления.
        """
        if not weights:
            raise ValueError("Словарь базовых весов пуст")

        total = sum(weights.values())
        if total <= 0:
            raise ValueError(
                f"Сумма базовых весов должна быть > 0, получено {total:.4f}"
            )

        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Сумма базовых весов = %.4f, нормализую к 1.0",
                total,
            )
            for name in weights:
                weights[name] /= total

    def get_base_weights(self) -> Dict[str, float]:
        return dict(self._base_weights)

    def get_weights(
        self,
        regime: MarketRegime,
        analyzers: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Возвращает веса для режима regime, нормализованные к 1.0.

        Args:
            regime: рыночный режим (BULL / BEAR / CHOP)
            analyzers: если задан — оставить только эти анализаторы
                       (обычно — те, что вернули сигнал)

        Returns:
            Словарь {analyzer_name: normalized_weight}, сумма = 1.0
        """
        multipliers = self.REGIME_MULTIPLIERS.get(regime, {})

        adjusted: Dict[str, float] = {}
        for name, weight in self._base_weights.items():
            if analyzers is not None and name not in analyzers:
                continue
            mult = multipliers.get(name, 1.0)
            adjusted[name] = weight * mult

        if not adjusted:
            logger.warning("Нет анализаторов для расчёта весов")
            return {}

        total = sum(adjusted.values())
        if total <= 0:
            logger.warning(
                "Сумма скорректированных весов <= 0 (%.4f), возвращаю пустой словарь",
                total,
            )
            return {}

        normalized = {name: w / total for name, w in adjusted.items()}

        logger.debug(
            "Веса для режима %s: %s (сумма=%.4f)",
            regime.value,
            {k: f"{v:.3f}" for k, v in normalized.items()},
            sum(normalized.values()),
        )
        return normalized

    def get_weight(self, analyzer_name: str, regime: MarketRegime) -> float:
        weights = self.get_weights(regime)
        return weights.get(analyzer_name, 0.0)

    def get_total_possible_weight(self, regime: MarketRegime) -> float:
        weights = self.get_weights(regime)
        return sum(weights.values())