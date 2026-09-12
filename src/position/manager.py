"""Position Manager: управление позициями (с комиссией)."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.models import SignalDirection
from src.position.position import (
    Position, PositionStatus, CloseReason, PartialClose,
)
from src.position.events import PositionEvent, PositionEventType
from src.risk.risk_manager import TradePlan

logger = logging.getLogger(__name__)


@dataclass
class PositionUpdate:
    events: List[PositionEvent] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    open_count: int = 0


class PositionManager:

    def __init__(
        self,
        breakeven_after_tp: int = 1,
        trailing_after_tp: int = 2,
        trailing_atr_multiplier: float = 1.0,
        max_position_age_bars: int = 100,
        commission_pct: float = 0.001,
    ):
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._breakeven_after_tp = breakeven_after_tp
        self._trailing_after_tp = trailing_after_tp
        self._trailing_atr_multiplier = trailing_atr_multiplier
        self._max_position_age_bars = max_position_age_bars
        self._commission_pct = commission_pct
        self._bar_counter = 0

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.is_open]

    @property
    def open_count(self) -> int:
        return len(self.open_positions)

    @property
    def closed_positions(self) -> List[Position]:
        return list(self._closed_positions)

    def get_position(self, position_id: str) -> Optional[Position]:
        return self._positions.get(position_id)

    def open_position(
        self,
        symbol: str,
        timeframe: str,
        direction: SignalDirection,
        plan: TradePlan,
        current_time: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Position]:
        if direction == SignalDirection.HOLD:
            logger.warning("Нельзя открыть позицию с direction=HOLD")
            return None

        if not plan.take_profits:
            logger.warning("Нет тейк-профитов в плане")
            return None

        tp_ratios = [lvl.ratio for lvl in plan.tp_result.levels] if plan.tp_result else []
        tp_percentages = [lvl.percentage for lvl in plan.tp_result.levels] if plan.tp_result else []

        merged_metadata = dict(metadata or {})
        merged_metadata["opened_bar_index"] = self._bar_counter

        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=plan.entry_price,
            entry_time=current_time or datetime.now(timezone.utc),
            initial_size=plan.position_size,
            initial_stop_loss=plan.stop_loss,
            stop_loss=plan.stop_loss,
            take_profits=list(plan.take_profits),
            tp_ratios=tp_ratios,
            tp_percentages=tp_percentages,
            commission_pct=self._commission_pct,
            risk_amount=plan.risk_amount,
            risk_pct=plan.risk_pct,
            metadata=merged_metadata,
        )

        self._positions[position.id] = position

        event = PositionEvent(
            type=PositionEventType.OPENED,
            position_id=position.id,
            symbol=symbol,
            timestamp=position.entry_time,
            price=position.entry_price,
            size=position.initial_size,
            details={
                "direction": direction.name,
                "stop_loss": position.stop_loss,
                "take_profits": position.take_profits,
                "risk_pct": position.risk_pct,
                "opened_bar_index": self._bar_counter,
                "commission_pct": self._commission_pct,
            },
        )

        logger.info(
            "Открыта позиция %s: %s %s, entry=%.4f, SL=%.4f, TP=%.4f, "
            "size=%.6f, commission=%.4f, opened_bar=%d",
            position.id, symbol, direction.name,
            position.entry_price, position.stop_loss,
            position.take_profits[0] if position.take_profits else 0.0,
            position.initial_size, self._commission_pct,
            self._bar_counter,
        )

        self._last_event = event
        return position

    def on_bar(
        self,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_time: datetime,
        atr: Optional[float] = None,
    ) -> PositionUpdate:
        self._bar_counter += 1
        events: List[PositionEvent] = []
        closed: List[Position] = []
        total_pnl = 0.0

        for position in list(self._positions.values()):
            if not position.is_open:
                continue

            age_bars = self._calculate_age_bars(position)
            if age_bars >= self._max_position_age_bars:
                pnl = position.close_full(
                    bar_close, bar_time, CloseReason.TIMEOUT,
                )
                total_pnl += pnl
                closed.append(position)
                events.append(PositionEvent(
                    type=PositionEventType.CLOSED,
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=bar_time,
                    price=bar_close,
                    size=position.initial_size,
                    pnl=pnl,
                    reason=CloseReason.TIMEOUT,
                    details={"age_bars": age_bars},
                ))
                continue

            if position.check_sl_hit(bar_high, bar_low):
                pnl = position.close_full(
                    position.stop_loss, bar_time, CloseReason.STOP_LOSS,
                )
                total_pnl += pnl
                closed.append(position)
                events.append(PositionEvent(
                    type=PositionEventType.CLOSED,
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=bar_time,
                    price=position.stop_loss,
                    size=position.initial_size,
                    pnl=pnl,
                    reason=CloseReason.STOP_LOSS,
                ))
                continue

            hit_tps = position.check_tp_hit(bar_high, bar_low)
            for tp_num in hit_tps:
                tp_price = position.take_profits[tp_num - 1]
                partial = position.close_partial(tp_num, tp_price, bar_time)
                total_pnl += partial.pnl

                events.append(PositionEvent(
                    type=PositionEventType.PARTIAL_CLOSE,
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=bar_time,
                    price=tp_price,
                    size=partial.size,
                    pnl=partial.pnl,
                    details={
                        "tp_level": tp_num,
                        "ratio": partial.ratio,
                        "remaining_size": position.current_size,
                    },
                ))

                if not position.is_open:
                    closed.append(position)
                    events.append(PositionEvent(
                        type=PositionEventType.CLOSED,
                        position_id=position.id,
                        symbol=position.symbol,
                        timestamp=bar_time,
                        price=tp_price,
                        size=position.initial_size,
                        pnl=partial.pnl,
                        reason=CloseReason.TAKE_PROFIT,
                    ))
                    break

        self._closed_positions.extend(closed)

        return PositionUpdate(
            events=events,
            closed_positions=closed,
            total_realized_pnl=total_pnl,
            open_count=self.open_count,
        )

    def _calculate_age_bars(self, position: Position) -> int:
        opened_bar = position.metadata.get("opened_bar_index")
        if opened_bar is None:
            logger.warning(
                "Позиция %s не имеет opened_bar_index в metadata, возраст = 0",
                position.id,
            )
            return 0
        return self._bar_counter - opened_bar

    def close_all(
        self,
        price: float,
        timestamp: datetime,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> List[Position]:
        closed = []
        for position in list(self._positions.values()):
            if position.is_open:
                position.close_full(price, timestamp, reason)
                self._closed_positions.append(position)
                closed.append(position)
        logger.info("Закрыто %d позиций", len(closed))
        return closed

    def get_stats(self) -> dict:
        closed = self._closed_positions
        if not closed:
            return {
                "total": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "open": self.open_count,
            }

        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl < 0]
        total_pnl = sum(p.realized_pnl for p in closed)

        return {
            "total": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0.0,
            "total_pnl": total_pnl,
            "open": self.open_count,
        }