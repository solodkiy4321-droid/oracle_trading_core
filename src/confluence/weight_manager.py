"""Менеджер весов анализаторов с адаптацией под рыночный режим.

Активные анализаторы (4):
- trend
- elliott_wave
- volatility
- volume
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
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Сумма базовых весов должна быть 1.0, получено {total:.4f}"
            )

    def get_base_weights(self) -> Dict[str, float]:
        return dict(self._base_weights)

    def get_weights(
        self, regime: MarketRegime, analyzers: Optional[list] = None
    ) -> Dict[str, float]:
        multipliers = self.REGIME_MULTIPLIERS.get(regime, {})

        adjusted = {}
        for name, weight in self._base_weights.items():
            if analyzers is not None and name not in analyzers:
                continue
            mult = multipliers.get(name, 1.0)
            adjusted[name] = weight * mult

        if not adjusted:
            logger.warning("Нет анализаторов для расчёта весов")
            return {}

        logger.debug(
            "Веса для режима %s (без нормализации): %s",
            regime.value,
            {k: f"{v:.3f}" for k, v in adjusted.items()},
        )
        return adjusted

    def get_weight(self, analyzer_name: str, regime: MarketRegime) -> float:
        weights = self.get_weights(regime)
        return weights.get(analyzer_name, 0.0)

    def get_total_possible_weight(self, regime: MarketRegime) -> float:
        weights = self.get_weights(regime)
        return sum(weights.values())