"""Position Manager: управление открытыми позициями."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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
    """Результат обработки бара Position Manager."""
    events: List[PositionEvent] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    total_realized_pnl: float = 0.0
    open_count: int = 0


class PositionManager:
    """
    Управляет открытыми позициями.

    Функции:
    - Открытие позиции на основе TradePlan
    - Обработка каждого нового бара: проверка SL/TP
    - Перевод стопа в безубыток после TP1
    - Трейлинг-стоп после TP2
    - Частичное закрытие на TP
    - Полное закрытие по SL/TP/Timeout
    - Таймаут позиций (если открыта слишком долго)
    """

    def __init__(
        self,
        breakeven_after_tp: int = 1,
        trailing_after_tp: int = 2,
        trailing_atr_multiplier: float = 1.0,
        max_position_age_bars: int = 100,
        commission_pct: float = 0.001,
    ):
        """
        Args:
            breakeven_after_tp: После какого TP переводить стоп в безубыток
            trailing_after_tp: После какого TP включать трейлинг-стоп
            trailing_atr_multiplier: Множитель ATR для трейлинг-стопа
            max_position_age_bars: Максимальный возраст позиции в барах
            commission_pct: Комиссия (0.001 = 0.1%)
        """
        self._positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._breakeven_after_tp = breakeven_after_tp
        self._trailing_after_tp = trailing_after_tp
        self._trailing_atr_multiplier = trailing_atr_multiplier
        self._max_position_age_bars = max_position_age_bars
        self._commission_pct = commission_pct
        self._bar_counter = 0  # Глобальный счётчик обработанных баров

    @property
    def open_positions(self) -> List[Position]:
        """Список открытых позиций."""
        return [p for p in self._positions.values() if p.is_open]

    @property
    def open_count(self) -> int:
        """Количество открытых позиций."""
        return len(self.open_positions)

    @property
    def closed_positions(self) -> List[Position]:
        """Список закрытых позиций."""
        return list(self._closed_positions)

    def get_position(self, position_id: str) -> Optional[Position]:
        """Получить позицию по ID."""
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
        """
        Открывает позицию на основе торгового плана.

        Args:
            symbol: Торговый инструмент
            timeframe: Таймфрейм
            direction: Направление (BUY / SELL)
            plan: Торговый план от Risk Manager
            current_time: Время открытия (по умолчанию — сейчас)
            metadata: Дополнительные метаданные

        Returns:
            Position или None при ошибке
        """
        if direction == SignalDirection.HOLD:
            logger.warning("Нельзя открыть позицию с direction=HOLD")
            return None

        if not plan.take_profits:
            logger.warning("Нет тейк-профитов в плане")
            return None

        # Извлекаем данные из плана
        tp_ratios = [lvl.ratio for lvl in plan.tp_result.levels] if plan.tp_result else []
        tp_percentages = [lvl.percentage for lvl in plan.tp_result.levels] if plan.tp_result else []

        # ВАЖНО: добавляем opened_bar_index в metadata
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
            },
        )

        logger.info(
            "Открыта позиция %s: %s %s, entry=%.4f, SL=%.4f, size=%.6f, "
            "opened_bar=%d",
            position.id, symbol, direction.name,
            position.entry_price, position.stop_loss, position.initial_size,
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
        """
        Обрабатывает новый бар: проверяет SL/TP для всех открытых позиций.

        Args:
            bar_high: Максимум бара
            bar_low: Минимум бара
            bar_close: Цена закрытия бара
            bar_time: Время бара
            atr: Текущий ATR (для трейлинг-стопа)

        Returns:
            PositionUpdate с событиями и закрытыми позициями
        """
        self._bar_counter += 1
        events: List[PositionEvent] = []
        closed: List[Position] = []
        total_pnl = 0.0

        for position in list(self._positions.values()):
            if not position.is_open:
                continue

            # 1. Проверяем таймаут (используем реальный возраст позиции)
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

            # 2. Проверяем SL (приоритет выше TP — консервативный подход)
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

            # 3. Проверяем TP
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

                # Перевод в безубыток после N-го TP
                if tp_num == self._breakeven_after_tp:
                    old_sl = position.stop_loss
                    position.move_stop_to_breakeven(self._commission_pct)
                    if position.stop_loss != old_sl:
                        events.append(PositionEvent(
                            type=PositionEventType.BREAKEVEN,
                            position_id=position.id,
                            symbol=position.symbol,
                            timestamp=bar_time,
                            price=position.stop_loss,
                            details={"old_sl": old_sl, "new_sl": position.stop_loss},
                        ))

                # Включаем трейлинг-стоп после N-го TP
                if tp_num >= self._trailing_after_tp and atr is not None and atr > 0:
                    self._apply_trailing_stop(position, bar_close, atr, bar_time, events)

                # Если позиция закрылась полностью
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

        # Обновляем список закрытых
        self._closed_positions.extend(closed)

        return PositionUpdate(
            events=events,
            closed_positions=closed,
            total_realized_pnl=total_pnl,
            open_count=self.open_count,
        )

    def _calculate_age_bars(self, position: Position) -> int:
        """
        Вычисляет возраст позиции в барах.

        ИСПРАВЛЕНО: использует opened_bar_index из metadata позиции,
        а не глобальный счётчик.

        Если opened_bar_index отсутствует (старые позиции),
        возвращает 0 — позиция считается новой.
        """
        opened_bar = position.metadata.get("opened_bar_index")
        if opened_bar is None:
            logger.warning(
                "Позиция %s не имеет opened_bar_index в metadata, "
                "возраст = 0",
                position.id,
            )
            return 0
        return self._bar_counter - opened_bar

    def _apply_trailing_stop(
        self,
        position: Position,
        current_price: float,
        atr: float,
        bar_time: datetime,
        events: List[PositionEvent],
    ) -> None:
        """Применяет трейлинг-стоп."""
        trailing_distance = atr * self._trailing_atr_multiplier

        if position.direction == SignalDirection.BUY:
            new_sl = current_price - trailing_distance
            if new_sl > position.stop_loss:
                old_sl = position.stop_loss
                position.update_trailing_stop(new_sl)
                events.append(PositionEvent(
                    type=PositionEventType.STOP_MOVED,
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=bar_time,
                    price=new_sl,
                    details={"old_sl": old_sl, "type": "trailing"},
                ))
        else:  # SELL
            new_sl = current_price + trailing_distance
            if new_sl < position.stop_loss:
                old_sl = position.stop_loss
                position.update_trailing_stop(new_sl)
                events.append(PositionEvent(
                    type=PositionEventType.STOP_MOVED,
                    position_id=position.id,
                    symbol=position.symbol,
                    timestamp=bar_time,
                    price=new_sl,
                    details={"old_sl": old_sl, "type": "trailing"},
                ))

    def close_all(
        self,
        price: float,
        timestamp: datetime,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> List[Position]:
        """Закрывает все открытые позиции."""
        closed = []
        for position in list(self._positions.values()):
            if position.is_open:
                position.close_full(price, timestamp, reason)
                self._closed_positions.append(position)
                closed.append(position)
        logger.info("Закрыто %d позиций", len(closed))
        return closed

    def get_stats(self) -> dict:
        """Возвращает статистику по позициям."""
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