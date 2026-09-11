"""Базовый класс для всех анализаторов."""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from src.models import AnalyzerSignal


class BaseAnalyzer(ABC):
    """
    Абстрактный базовый класс анализатора.
    
    Все анализаторы должны наследоваться от этого класса
    и реализовывать метод analyze().
    """

    def __init__(self, name: str):
        """
        Args:
            name: Уникальное имя анализатора (используется для весов и логирования)
        """
        self._name = name

    @property
    def name(self) -> str:
        """Имя анализатора."""
        return self._name

    @abstractmethod
    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        """
        Проанализировать данные и вернуть сигнал.
        
        Args:
            data: DataFrame с колонками open, high, low, close, volume
            
        Returns:
            AnalyzerSignal или None, если сигнал не найден
        """
        ...