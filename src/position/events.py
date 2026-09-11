"""События Position Manager."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.position.position import Position, CloseReason

logger = logging.getLogger(__name__)


class PositionEventType(Enum):
    """Тип события позиции."""
    OPENED = "opened"
    PARTIAL_CLOSE = "partial_close"
    CLOSED = "closed"
    STOP_MOVED = "stop_moved"
    BREAKEVEN = "breakeven"


@dataclass
class PositionEvent:
    """Событие позиции."""
    type: PositionEventType
    position_id: str
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    price: Optional[float] = None
    size: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[CloseReason] = None
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [
            f"[{self.type.value}]",
            f"pos={self.position_id}",
            f"symbol={self.symbol}",
        ]
        if self.price is not None:
            parts.append(f"price={self.price:.4f}")
        if self.size is not None:
            parts.append(f"size={self.size:.6f}")
        if self.pnl is not None:
            parts.append(f"pnl={self.pnl:.2f}")
        if self.reason is not None:
            parts.append(f"reason={self.reason.value}")
        return " ".join(parts)