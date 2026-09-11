"""Менеджер весов анализаторов с адаптацией под рыночный режим."""

import logging
from typing import Dict, Optional

from src.confluence.regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class WeightManager:
    """
    Управляет весами анализаторов.

    Веса определяют вклад каждого анализатора в итоговый confluence score.
    Веса адаптируются под рыночный режим (BULL / BEAR / CHOP).

    Важно: веса НЕ нормализуются до 1.0, если в сигналах участвует
    меньше анализаторов, чем зарегистрировано. Это гарантирует, что
    один анализатор не может единолично открыть сделку.

    Распределение весов (пересмотрено после анализа результатов):
    - support_resistance: 0.40 — самый точный (conf 0.95-1.00), частый
    - indicators: 0.30 — шумный, даёт слабые сигналы (conf 0.31-0.54)
    - harmonic: 0.30 — мощный, но редкий
    """

    # Базовые веса (по умолчанию)
    DEFAULT_BASE_WEIGHTS = {
        "indicators": 0.30,
        "harmonic": 0.30,
        "support_resistance": 0.40,
    }

    # Множители весов под рыночный режим
    REGIME_MULTIPLIERS = {
        MarketRegime.BULL: {
            "indicators": 1.2,
            "harmonic": 0.8,
            "support_resistance": 1.0,
        },
        MarketRegime.BEAR: {
            "indicators": 1.2,
            "harmonic": 0.8,
            "support_resistance": 1.0,
        },
        MarketRegime.CHOP: {
            "indicators": 0.8,
            "harmonic": 1.2,
            "support_resistance": 1.3,
        },
    }

    def __init__(self, base_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            base_weights: Переопределение базовых весов
        """
        self._base_weights = {**self.DEFAULT_BASE_WEIGHTS, **(base_weights or {})}
        self._validate_weights(self._base_weights)

    def _validate_weights(self, weights: Dict[str, float]) -> None:
        """Проверяет, что сумма весов равна 1.0."""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Сумма базовых весов должна быть 1.0, получено {total:.4f}"
            )

    def get_base_weights(self) -> Dict[str, float]:
        """Возвращает базовые веса."""
        return dict(self._base_weights)

    def get_weights(
        self, regime: MarketRegime, analyzers: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Возвращает веса с учётом рыночного режима.

        КЛЮЧЕВОЕ ПРАВИЛО: веса НЕ нормализуются до 1.0. Они остаются
        в своих базовых пропорциях, скорректированных на режим.

        Args:
            regime: Текущий рыночный режим
            analyzers: Список имён анализаторов для фильтрации
                       (если None — возвращаются все)

        Returns:
            Веса (сумма <= 1.0)
        """
        multipliers = self.REGIME_MULTIPLIERS.get(regime, {})

        # Применяем множители
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
        """Возвращает вес конкретного анализатора."""
        weights = self.get_weights(regime)
        return weights.get(analyzer_name, 0.0)

    def get_total_possible_weight(self, regime: MarketRegime) -> float:
        """
        Возвращает максимально возможную сумму весов для режима.
        """
        weights = self.get_weights(regime)
        return sum(weights.values())