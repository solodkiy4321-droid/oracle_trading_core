"""Главный класс модуля приёма сигналов."""

import logging
import pandas as pd

from src.analyzers.base import BaseAnalyzer
from src.models import SignalBatch, RejectedSignal
from src.signal_intake.collector import SignalCollector
from src.signal_intake.normalizer import normalize
from src.signal_intake.validator import validate

logger = logging.getLogger(__name__)


class SignalIntake:
    """
    Фасад модуля приёма сигналов.
    
    Объединяет:
    - SignalCollector: сбор сигналов от анализаторов
    - normalize: приведение к единому формату
    - validate: проверка корректности
    """

    def __init__(self, timeout: float = 5.0):
        self._collector = SignalCollector(timeout=timeout)

    def register(self, analyzer: BaseAnalyzer) -> None:
        """Зарегистрировать анализатор."""
        self._collector.register(analyzer)

    async def process(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> SignalBatch:
        """
        Полный цикл приёма сигналов:
        1. Собрать сырые сигналы от всех анализаторов
        2. Нормализовать каждый
        3. Провалидировать каждый
        4. Вернуть батч с валидными и отвергнутыми
        """
        logger.info("Начало приёма сигналов: %s %s", symbol, timeframe)

        raw_signals, rejected_from_collector, stats = await self._collector.collect_all(data)

        valid_signals = []
        rejected = list(rejected_from_collector)

        for raw in raw_signals:
            # Нормализация
            try:
                normalized = normalize(raw, symbol=symbol, timeframe=timeframe)
            except Exception as e:
                logger.exception("Ошибка нормализации сигнала от %s: %s", raw.source, e)
                rejected.append(RejectedSignal(
                    signal=raw,
                    reason=f"normalize error: {e}",
                    source=raw.source,
                ))
                continue

            # Валидация
            is_valid, error = validate(normalized)
            if not is_valid:
                logger.warning("Сигнал от %s отвергнут: %s", normalized.source, error)
                rejected.append(RejectedSignal(
                    signal=normalized,
                    reason=error,
                    source=normalized.source,
                ))
                continue

            valid_signals.append(normalized)

        batch = SignalBatch(
            signals=valid_signals,
            rejected=rejected,
            symbol=symbol,
            timeframe=timeframe,
            stats={
                **stats,
                "valid": len(valid_signals),
                "rejected": len(rejected),
            },
        )

        logger.info(
            "Приём завершён: валидных=%d, отвергнутых=%d",
            len(valid_signals), len(rejected),
        )
        return batch