"""Модуль приёма сигналов от анализаторов."""

from .collector import SignalCollector
from .normalizer import normalize
from .validator import validate
from .intake import SignalIntake

__all__ = ["SignalCollector", "normalize", "validate", "SignalIntake"]