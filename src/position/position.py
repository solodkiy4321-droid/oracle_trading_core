"""Модель одной торговой позиции."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.models import SignalDirection

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Статус позиции."""
    OPEN = "open"              # Открыта, активна
    PARTIALLY_CLOSED = "partial"  # Частично закрыта (TP1 или TP2)
    CLOSED = "closed"          # Полностью закрыта


class CloseReason(Enum):
    """Причина закрытия позиции."""
    STOP_LOSS = "stop_loss"      # Сработал стоп-лосс
    TAKE_PROFIT = "take_profit"  # Сработал тейк-профит
    TIMEOUT = "timeout"          # Позиция слишком долго открыта
    MANUAL = "manual"            # Ручное закрытие
    TRAILING_STOP = "trailing"   # Сработал трейлинг-стоп


@dataclass
class PartialClose:
    """Информация о частичном закрытии."""
    tp_level: int          # Номер TP (1, 2, 3)
    price: float           # Цена закрытия
    size: float            # Размер закрытой части
    pnl: float             # P&L этой части
    timestamp: datetime    # Время закрытия
    ratio: float           # R:R ratio этого уровня


@dataclass
class Position:
    """
    Модель одной торговой позиции.
    
    Отслеживает:
    - Вход (цена, время, размер)
    - Стоп-лосс (начальный, текущий)
    - Тейк-профиты (уровни, размеры, статус)
    - Частичные закрытия
    - Итоговый P&L
    """
    # Идентификация
    id: str                                 # Уникальный ID позиции
    symbol: str                             # Торговый инструмент
    timeframe: str                          # Таймфрейм
    direction: SignalDirection              # BUY / SELL
    
    # Вход
    entry_price: float                      # Цена входа
    entry_time: datetime                    # Время входа
    initial_size: float                     # Начальный размер позиции
    
    # Стоп-лосс
    initial_stop_loss: float                # Начальный SL
    stop_loss: float                        # Текущий SL (может меняться)
    
    # Тейк-профиты
    take_profits: List[float]               # Цены TP
    tp_ratios: List[float]                  # R:R ratio каждого TP
    tp_percentages: List[float]             # Процент закрытия на каждом TP
    
    # Состояние
    status: PositionStatus = PositionStatus.OPEN
    current_size: float = 0.0               # Текущий размер (уменьшается при частичных закрытиях)
    partial_closes: List[PartialClose] = field(default_factory=list)
    
    # Риск
    risk_amount: float = 0.0                # Сумма риска в валюте
    risk_pct: float = 0.0                   # Процент риска
    
    # Метаданные
    metadata: dict = field(default_factory=dict)
    
    # Результат (заполняется при закрытии)
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[CloseReason] = None
    realized_pnl: float = 0.0               # Реализованный P&L
    
    def __post_init__(self):
        """Инициализация."""
        if self.current_size == 0.0:
            self.current_size = self.initial_size
        if not self.id:
            raise ValueError("Position id обязателен")
    
    @property
    def is_open(self) -> bool:
        """Открыта ли позиция."""
        return self.status != PositionStatus.CLOSED
    
    @property
    def remaining_pct(self) -> float:
        """Оставшийся процент позиции."""
        if self.initial_size <= 0:
            return 0.0
        return self.current_size / self.initial_size
    
    @property
    def stop_distance(self) -> float:
        """Расстояние от входа до текущего стопа."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def initial_stop_distance(self) -> float:
        """Расстояние от входа до начального стопа."""
        return abs(self.entry_price - self.initial_stop_loss)
    
    def check_sl_hit(self, high: float, low: float) -> bool:
        """Проверяет, сработал ли стоп-лосс на баре."""
        if self.direction == SignalDirection.BUY:
            return low <= self.stop_loss
        else:  # SELL
            return high >= self.stop_loss
    
    def check_tp_hit(self, high: float, low: float) -> List[int]:
        """
        Проверяет, какие TP сработали на баре.
        
        Returns:
            Список номеров TP (1, 2, 3), которые сработали
        """
        hit = []
        for i, tp in enumerate(self.take_profits):
            tp_num = i + 1
            # Проверяем, не закрыт ли уже этот уровень
            if any(pc.tp_level == tp_num for pc in self.partial_closes):
                continue
            
            if self.direction == SignalDirection.BUY:
                if high >= tp:
                    hit.append(tp_num)
            else:  # SELL
                if low <= tp:
                    hit.append(tp_num)
        
        return hit
    
    def calculate_pnl(self, close_price: float, size: float) -> float:
        """Рассчитывает P&L для закрытия части позиции."""
        if self.direction == SignalDirection.BUY:
            return (close_price - self.entry_price) * size
        else:  # SELL
            return (self.entry_price - close_price) * size
    
    def close_partial(
        self,
        tp_level: int,
        price: float,
        timestamp: datetime,
    ) -> PartialClose:
        """Закрывает часть позиции на TP."""
        if tp_level < 1 or tp_level > len(self.take_profits):
            raise ValueError(f"Неверный уровень TP: {tp_level}")
        
        # Определяем размер закрытия
        pct = self.tp_percentages[tp_level - 1] if tp_level <= len(self.tp_percentages) else 0.0
        size_to_close = self.initial_size * pct
        
        # Не закрываем больше, чем осталось
        size_to_close = min(size_to_close, self.current_size)
        
        pnl = self.calculate_pnl(price, size_to_close)
        ratio = self.tp_ratios[tp_level - 1] if tp_level <= len(self.tp_ratios) else 0.0
        
        partial = PartialClose(
            tp_level=tp_level,
            price=price,
            size=size_to_close,
            pnl=pnl,
            timestamp=timestamp,
            ratio=ratio,
        )
        
        self.partial_closes.append(partial)
        self.current_size -= size_to_close
        self.realized_pnl += pnl
        
        # Если оставшийся размер очень мал — закрываем полностью
        if self.current_size < self.initial_size * 0.01:
            self.current_size = 0.0
            self.status = PositionStatus.CLOSED
            self.close_time = timestamp
            self.close_price = price
            self.close_reason = CloseReason.TAKE_PROFIT
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
        
        logger.info(
            "Частичное закрытие позиции %s: TP%d, size=%.6f, pnl=%.2f, "
            "осталось=%.6f",
            self.id, tp_level, size_to_close, pnl, self.current_size,
        )
        
        return partial
    
    def close_full(
        self,
        price: float,
        timestamp: datetime,
        reason: CloseReason,
    ) -> float:
        """Полностью закрывает позицию."""
        pnl = self.calculate_pnl(price, self.current_size)
        self.realized_pnl += pnl
        
        self.current_size = 0.0
        self.status = PositionStatus.CLOSED
        self.close_time = timestamp
        self.close_price = price
        self.close_reason = reason
        
        logger.info(
            "Полное закрытие позиции %s: reason=%s, price=%.4f, pnl=%.2f",
            self.id, reason.value, price, pnl,
        )
        
        return pnl
    
    def move_stop_to_breakeven(self, commission_pct: float = 0.001) -> None:
        """
        Переводит стоп-лосс в безубыток (с учётом комиссии).
        
        Args:
            commission_pct: Комиссия (например, 0.001 = 0.1%)
        """
        if self.direction == SignalDirection.BUY:
            new_sl = self.entry_price * (1 + commission_pct)
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
                logger.info(
                    "Позиция %s: SL переведён в безубыток (%.4f)",
                    self.id, new_sl,
                )
        else:  # SELL
            new_sl = self.entry_price * (1 - commission_pct)
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl
                logger.info(
                    "Позиция %s: SL переведён в безубыток (%.4f)",
                    self.id, new_sl,
                )
    
    def update_trailing_stop(self, new_sl: float) -> None:
        """
        Обновляет трейлинг-стоп.
        
        Стоп двигается только в сторону прибыли.
        """
        if self.direction == SignalDirection.BUY:
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
        else:  # SELL
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl