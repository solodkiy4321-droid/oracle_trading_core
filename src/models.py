"""Структуры данных для ядра анализа Oracle Trading."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


class SignalDirection(Enum):
    """Направление торгового сигнала."""
    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class AnalyzerSignal:
    """
    Сигнал от одного анализатора.
    
    Attributes:
        direction: Направление сигнала (BUY / SELL / HOLD)
        confidence: Уверенность анализатора от 0.0 до 1.0
        reason: Текстовое обоснование сигнала
        metadata: Дополнительные данные от анализатора
        source: Имя анализатора, сгенерировавшего сигнал
        timestamp: Время генерации сигнала (UTC)
    """
    direction: SignalDirection
    confidence: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Валидация при создании объекта."""
        if not isinstance(self.direction, SignalDirection):
            raise ValueError(f"direction должен быть SignalDirection, получен {type(self.direction)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence должен быть в [0.0, 1.0], получен {self.confidence}")
        if not isinstance(self.reason, str):
            raise ValueError(f"reason должен быть строкой, получен {type(self.reason)}")


@dataclass
class RejectedSignal:
    """Отвергнутый сигнал с указанием причины."""
    signal: Optional[AnalyzerSignal]
    reason: str
    source: str = "unknown"


@dataclass
class SignalBatch:
    """
    Батч сигналов, готовый к передаче в Scoring System.
    
    Attributes:
        signals: Список валидных сигналов
        rejected: Список отвергнутых сигналов с причинами
        symbol: Торговый инструмент
        timeframe: Таймфрейм анализа
        stats: Статистика сбора (сколько анализаторов ответило и т.д.)
        created_at: Время создания батча
    """
    signals: List[AnalyzerSignal] = field(default_factory=list)
    rejected: List[RejectedSignal] = field(default_factory=list)
    symbol: str = ""
    timeframe: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_empty(self) -> bool:
        """Проверка, пустой ли батч."""
        return len(self.signals) == 0

    @property
    def count(self) -> int:
        """Количество валидных сигналов."""
        return len(self.signals)