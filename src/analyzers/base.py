"""Базовый класс для всех анализаторов."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import pandas as pd

from src.models import AnalyzerSignal


class BaseAnalyzer(ABC):
    """Абстрактный базовый класс анализатора."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def analyze(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
        bar_index: Optional[int] = None,
    ) -> Optional[AnalyzerSignal]:
        """
        Args:
            data: DataFrame с колонками open, high, low, close, volume
            indicators: IndicatorCache или None
            bar_index: абсолютный индекс бара (для indicator cache)

        Returns:
            AnalyzerSignal или None
        """
        ...