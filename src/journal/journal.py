"""Journal: главный класс логирования сделок."""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from src.journal.database import Database
from src.journal.models import (
    TradeRecord, EventRecord, PortfolioSnapshot, DecisionRecord,
)
from src.position.position import Position
from src.position.events import PositionEvent
from src.portfolio.state import PortfolioState
from src.confluence.engine import ConfluenceDecision

logger = logging.getLogger(__name__)


class Journal:
    """
    Логирует сделки, события, решения и снимки портфеля в SQLite.
    """

    def __init__(self, db_path: str = "journal.db"):
        """
        Args:
            db_path: Путь к SQLite. ':memory:' для тестов.
        """
        self.db = Database(db_path)

    # ============ Сделки ============

    def log_position_opened(
        self,
        position: Position,
        regime: str = "",
        confluence_score: float = 0.0,
    ) -> int:
        """Логирует открытие позиции."""
        record = TradeRecord(
            position_id=position.id,
            symbol=position.symbol,
            timeframe=position.timeframe,
            direction=position.direction.name,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            initial_size=position.initial_size,
            initial_stop_loss=position.initial_stop_loss,
            final_stop_loss=position.stop_loss,
            take_profits=json.dumps(position.take_profits),
            risk_amount=position.risk_amount,
            risk_pct=position.risk_pct,
            regime=regime,
            confluence_score=confluence_score,
            metadata=json.dumps(position.metadata or {}),
            created_at=datetime.now(timezone.utc),
        )
        cursor = self.db.execute(
            """
            INSERT INTO trades (
                position_id, symbol, timeframe, direction,
                entry_price, entry_time, initial_size,
                initial_stop_loss, final_stop_loss, take_profits,
                risk_amount, risk_pct, regime, confluence_score,
                metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.position_id, record.symbol, record.timeframe,
                record.direction, record.entry_price,
                record.entry_time.isoformat() if record.entry_time else "",
                record.initial_size, record.initial_stop_loss,
                record.final_stop_loss, record.take_profits,
                record.risk_amount, record.risk_pct,
                record.regime, record.confluence_score,
                record.metadata,
                record.created_at.isoformat() if record.created_at else "",
            ),
        )
        self.db.commit()
        trade_id = cursor.lastrowid
        logger.debug("Journal: позиция %s записана (trade_id=%d)", position.id, trade_id)
        return trade_id

    def log_position_closed(self, position: Position) -> None:
        """Обновляет запись о позиции при закрытии."""
        if position.close_time is None:
            logger.warning("Позиция %s закрыта, но close_time отсутствует", position.id)
            return

        exit_reason = position.close_reason.value if position.close_reason else "unknown"
        pnl_pct = (
            position.realized_pnl / (position.entry_price * position.initial_size)
            if position.initial_size * position.entry_price > 0
            else 0.0
        )

        self.db.execute(
            """
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                exit_reason = ?,
                realized_pnl = ?,
                realized_pnl_pct = ?,
                final_stop_loss = ?
            WHERE position_id = ?
            """,
            (
                position.close_price,
                position.close_time.isoformat(),
                exit_reason,
                position.realized_pnl,
                pnl_pct,
                position.stop_loss,
                position.id,
            ),
        )
        self.db.commit()
        logger.debug(
            "Journal: позиция %s обновлена (pnl=%.2f, reason=%s)",
            position.id, position.realized_pnl, exit_reason,
        )

    # ============ События ============

    def log_event(self, event: PositionEvent) -> int:
        """Логирует событие позиции."""
        record = EventRecord(
            position_id=event.position_id,
            symbol=event.symbol,
            event_type=event.type.value,
            timestamp=event.timestamp,
            price=event.price or 0.0,
            size=event.size or 0.0,
            pnl=event.pnl or 0.0,
            reason=event.reason.value if event.reason else "",
            details=json.dumps(event.details),
        )
        cursor = self.db.execute(
            """
            INSERT INTO events (
                position_id, symbol, event_type, timestamp,
                price, size, pnl, reason, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.position_id, record.symbol, record.event_type,
                record.timestamp.isoformat() if record.timestamp else "",
                record.price, record.size, record.pnl,
                record.reason, record.details,
            ),
        )
        self.db.commit()
        return cursor.lastrowid

    def log_events(self, events: List[PositionEvent]) -> int:
        """Логирует несколько событий."""
        count = 0
        for event in events:
            self.log_event(event)
            count += 1
        return count

    # ============ Снимки портфеля ============

    def log_portfolio_snapshot(self, state: PortfolioState) -> int:
        """Логирует снимок состояния портфеля."""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            starting_equity=state.starting_equity,
            current_equity=state.current_equity,
            peak_equity=state.peak_equity,
            total_pnl=state.total_pnl,
            total_pnl_pct=state.total_pnl_pct,
            drawdown_pct=state.current_drawdown_pct,
            open_positions=state.total_open_positions,
            total_trades=state.total_trades,
            total_wins=state.total_wins,
            total_losses=state.total_losses,
            daily_pnl=state.daily_stats.realized_pnl if state.daily_stats else 0.0,
            daily_pnl_pct=state.daily_stats.pnl_pct if state.daily_stats else 0.0,
            is_paused=1 if state.is_paused else 0,
            pause_reason=state.pause_reason,
        )
        cursor = self.db.execute(
            """
            INSERT INTO portfolio_snapshots (
                timestamp, date, starting_equity, current_equity, peak_equity,
                total_pnl, total_pnl_pct, drawdown_pct,
                open_positions, total_trades, total_wins, total_losses,
                daily_pnl, daily_pnl_pct, is_paused, pause_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.timestamp.isoformat(),
                snapshot.date,
                snapshot.starting_equity, snapshot.current_equity,
                snapshot.peak_equity, snapshot.total_pnl,
                snapshot.total_pnl_pct, snapshot.drawdown_pct,
                snapshot.open_positions, snapshot.total_trades,
                snapshot.total_wins, snapshot.total_losses,
                snapshot.daily_pnl, snapshot.daily_pnl_pct,
                snapshot.is_paused, snapshot.pause_reason,
            ),
        )
        self.db.commit()
        logger.debug("Journal: снимок портфеля записан (equity=%.2f)", state.current_equity)
        return cursor.lastrowid

    # ============ Решения Confluence ============

    def log_decision(
        self,
        decision: ConfluenceDecision,
        symbol: str,
        timeframe: str,
        signals_count: int = 0,
    ) -> int:
        """Логирует решение Confluence-движка."""
        record = DecisionRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            timeframe=timeframe,
            regime=decision.regime.value,
            direction=decision.direction.name,
            confluence_score=decision.confluence_score,
            bull_score=decision.bull_score,
            bear_score=decision.bear_score,
            passed_gate=1 if decision.passed else 0,
            signals_count=signals_count,
            weights=json.dumps(decision.weights_used),
            reasons=json.dumps(decision.reasons),
        )
        cursor = self.db.execute(
            """
            INSERT INTO decisions (
                timestamp, symbol, timeframe, regime, direction,
                confluence_score, bull_score, bear_score,
                passed_gate, signals_count, weights, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp.isoformat() if record.timestamp else "",
                record.symbol, record.timeframe, record.regime,
                record.direction, record.confluence_score,
                record.bull_score, record.bear_score,
                record.passed_gate, record.signals_count,
                record.weights, record.reasons,
            ),
        )
        self.db.commit()
        return cursor.lastrowid

    # ============ Утилиты ============

    def close(self) -> None:
        """Закрывает БД."""
        self.db.close()

    def get_trades_count(self) -> int:
        """Возвращает количество записей о сделках."""
        cursor = self.db.execute("SELECT COUNT(*) FROM trades")
        return cursor.fetchone()[0]

    def get_events_count(self) -> int:
        """Возвращает количество событий."""
        cursor = self.db.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]

    def get_decisions_count(self) -> int:
        """Возвращает количество решений."""
        cursor = self.db.execute("SELECT COUNT(*) FROM decisions")
        return cursor.fetchone()[0]