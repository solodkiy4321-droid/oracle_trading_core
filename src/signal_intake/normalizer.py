"""Нормализатор сигналов."""

import logging
from datetime import datetime, timezone

from src.models import AnalyzerSignal

logger = logging.getLogger(__name__)


def normalize(
    signal: AnalyzerSignal,
    symbol: str,
    timeframe: str,
) -> AnalyzerSignal:
    """
    Приводит сигнал к единому формату.
    
    - Ограничивает confidence диапазоном [0.0, 1.0]
    - Обогащает метаданными (symbol, timeframe)
    - Приводит timestamp к UTC
    """
    # Ограничиваем confidence
    confidence = max(0.0, min(1.0, float(signal.confidence)))

    # Обогащаем метаданные
    metadata = dict(signal.metadata) if signal.metadata else {}
    metadata["symbol"] = symbol
    metadata["timeframe"] = timeframe

    # Приводим timestamp к UTC
    timestamp = signal.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    normalized = AnalyzerSignal(
        direction=signal.direction,
        confidence=confidence,
        reason=signal.reason,
        metadata=metadata,
        source=signal.source,
        timestamp=timestamp,
    )

    logger.debug("Сигнал нормализован: %s -> %s", signal.source, normalized.direction)
    return normalized