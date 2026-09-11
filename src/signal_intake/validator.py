"""Валидатор сигналов."""

import logging
from typing import Tuple

from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


def validate(signal: AnalyzerSignal) -> Tuple[bool, str]:
    """
    Проверяет корректность сигнала.
    
    Returns:
        (валиден_ли, сообщение_об_ошибке)
    """
    # Проверка обязательных полей
    if signal is None:
        return False, "signal is None"

    if not isinstance(signal.direction, SignalDirection):
        return False, f"invalid direction type: {type(signal.direction)}"

    if not isinstance(signal.confidence, (int, float)):
        return False, f"invalid confidence type: {type(signal.confidence)}"

    if not 0.0 <= signal.confidence <= 1.0:
        return False, f"confidence out of range: {signal.confidence}"

    if not isinstance(signal.reason, str):
        return False, f"invalid reason type: {type(signal.reason)}"

    if not signal.reason.strip():
        return False, "reason is empty"

    # Проверка метаданных
    if not isinstance(signal.metadata, dict):
        return False, f"invalid metadata type: {type(signal.metadata)}"

    required_meta = ["symbol", "timeframe"]
    for key in required_meta:
        if key not in signal.metadata:
            return False, f"missing metadata key: {key}"

    return True, ""