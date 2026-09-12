"""Модель одной торговой позиции (с комиссией)."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.models import SignalDirection

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    PARTIALLY_CLOSED = "partial"
    CLOSED = "closed"


class CloseReason(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIMEOUT = "timeout"
    MANUAL = "manual"
    TRAILING_STOP = "trailing"


@dataclass
class PartialClose:
    tp_level: int
    price: float
    size: float
    pnl: float
    timestamp: datetime
    ratio: float


@dataclass
class Position:
    id: str
    symbol: str
    timeframe: str
    direction: SignalDirection

    entry_price: float
    entry_time: datetime
    initial_size: float

    initial_stop_loss: float
    stop_loss: float

    take_profits: List[float]
    tp_ratios: List[float]
    tp_percentages: List[float]

    commission_pct: float = 0.001

    status: PositionStatus = PositionStatus.OPEN
    current_size: float = 0.0
    partial_closes: List[PartialClose] = field(default_factory=list)

    risk_amount: float = 0.0
    risk_pct: float = 0.0

    metadata: dict = field(default_factory=dict)

    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    close_reason: Optional[CloseReason] = None
    realized_pnl: float = 0.0

    def __post_init__(self):
        if self.current_size == 0.0:
            self.current_size = self.initial_size
        if not self.id:
            raise ValueError("Position id обязателен")

    @property
    def is_open(self) -> bool:
        return self.status != PositionStatus.CLOSED

    @property
    def remaining_pct(self) -> float:
        if self.initial_size <= 0:
            return 0.0
        return self.current_size / self.initial_size

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def initial_stop_distance(self) -> float:
        return abs(self.entry_price - self.initial_stop_loss)

    def check_sl_hit(self, high: float, low: float) -> bool:
        if self.direction == SignalDirection.BUY:
            return low <= self.stop_loss
        else:
            return high >= self.stop_loss

    def check_tp_hit(self, high: float, low: float) -> List[int]:
        hit = []
        for i, tp in enumerate(self.take_profits):
            tp_num = i + 1
            if any(pc.tp_level == tp_num for pc in self.partial_closes):
                continue

            if self.direction == SignalDirection.BUY:
                if high >= tp:
                    hit.append(tp_num)
            else:
                if low <= tp:
                    hit.append(tp_num)

        return hit

    def calculate_pnl(self, close_price: float, size: float) -> float:
        """
        Рассчитывает P&L с учётом комиссии.

        Комиссия берётся дважды:
        - на входе (entry_price * size * commission)
        - на выходе (close_price * size * commission)
        """
        if self.direction == SignalDirection.BUY:
            gross = (close_price - self.entry_price) * size
        else:
            gross = (self.entry_price - close_price) * size

        entry_commission = self.entry_price * size * self.commission_pct
        exit_commission = close_price * size * self.commission_pct

        return gross - entry_commission - exit_commission

    def close_partial(
        self,
        tp_level: int,
        price: float,
        timestamp: datetime,
    ) -> PartialClose:
        if tp_level < 1 or tp_level > len(self.take_profits):
            raise ValueError(f"Неверный уровень TP: {tp_level}")

        pct = self.tp_percentages[tp_level - 1] if tp_level <= len(self.tp_percentages) else 0.0
        size_to_close = self.initial_size * pct
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

        if self.current_size < self.initial_size * 0.01:
            self.current_size = 0.0
            self.status = PositionStatus.CLOSED
            self.close_time = timestamp
            self.close_price = price
            self.close_reason = CloseReason.TAKE_PROFIT
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED

        logger.debug(
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
        pnl = self.calculate_pnl(price, self.current_size)
        self.realized_pnl += pnl

        self.current_size = 0.0
        self.status = PositionStatus.CLOSED
        self.close_time = timestamp
        self.close_price = price
        self.close_reason = reason

        logger.debug(
            "Полное закрытие позиции %s: reason=%s, price=%.4f, pnl=%.2f",
            self.id, reason.value, price, pnl,
        )

        return pnl

    def move_stop_to_breakeven(self, commission_pct: float = 0.001) -> None:
        if self.direction == SignalDirection.BUY:
            new_sl = self.entry_price * (1 + commission_pct)
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
        else:
            new_sl = self.entry_price * (1 - commission_pct)
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl

    def update_trailing_stop(self, new_sl: float) -> None:
        if self.direction == SignalDirection.BUY:
            if new_sl > self.stop_loss:
                self.stop_loss = new_sl
        else:
            if new_sl < self.stop_loss:
                self.stop_loss = new_sl