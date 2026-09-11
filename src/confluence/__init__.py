"""Confluence-движок: объединение сигналов от анализаторов."""

from .weight_manager import WeightManager
from .regime_detector import RegimeDetector, MarketRegime
from .scoring import ScoringSystem, ScoringResult
from .gate import ConfluenceGate, GateMode
from .engine import ConfluenceEngine, ConfluenceDecision

__all__ = [
    "WeightManager",
    "RegimeDetector",
    "MarketRegime",
    "ScoringSystem",
    "ScoringResult",
    "ConfluenceGate",
    "GateMode",
    "ConfluenceEngine",
    "ConfluenceDecision",
]