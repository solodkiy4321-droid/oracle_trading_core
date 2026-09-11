"""Аналитика по данным Journal."""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.journal.database import Database

logger = logging.getLogger(__name__)


@dataclass
class TradeStats:
    """Статистика по сделкам."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0

    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_wins_pnl: float = 0.0
    total_losses_pnl: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_pnl: float = 0.0

    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    by_symbol: dict = field(default_factory=dict)
    by_exit_reason: dict = field(default_factory=dict)
    by_regime: dict = field(default_factory=dict)


class JournalAnalytics:
    """
    Аналитика по данным Journal.

    Извлекает метрики:
    - Win rate
    - Profit factor
    - Средняя прибыль/убыток
    - Разбивка по символам, причинам выхода, режимам
    """

    def __init__(self, db: Database):
        """
        Args:
            db: Подключение к БД Journal
        """
        self.db = db

    def get_trade_stats(self) -> TradeStats:
        """Возвращает общую статистику по сделкам."""
        cursor = self.db.execute(
            """
            SELECT
                symbol, realized_pnl, exit_reason, regime
            FROM trades
            WHERE exit_time IS NOT NULL
            """
        )
        rows = cursor.fetchall()

        stats = TradeStats()
        pnls = []

        for row in rows:
            pnl = row["realized_pnl"] or 0.0
            symbol = row["symbol"] or "unknown"
            exit_reason = row["exit_reason"] or "unknown"
            regime = row["regime"] or "unknown"

            stats.total_trades += 1
            stats.total_pnl += pnl
            pnls.append(pnl)

            if pnl > 0:
                stats.wins += 1
                stats.total_wins_pnl += pnl
            elif pnl < 0:
                stats.losses += 1
                stats.total_losses_pnl += abs(pnl)
            else:
                stats.breakeven += 1

            # По символам
            if symbol not in stats.by_symbol:
                stats.by_symbol[symbol] = {"trades": 0, "pnl": 0.0, "wins": 0}
            stats.by_symbol[symbol]["trades"] += 1
            stats.by_symbol[symbol]["pnl"] += pnl
            if pnl > 0:
                stats.by_symbol[symbol]["wins"] += 1

            # По причинам выхода
            if exit_reason not in stats.by_exit_reason:
                stats.by_exit_reason[exit_reason] = {"trades": 0, "pnl": 0.0}
            stats.by_exit_reason[exit_reason]["trades"] += 1
            stats.by_exit_reason[exit_reason]["pnl"] += pnl

            # По режимам
            if regime not in stats.by_regime:
                stats.by_regime[regime] = {"trades": 0, "pnl": 0.0}
            stats.by_regime[regime]["trades"] += 1
            stats.by_regime[regime]["pnl"] += pnl

        # Метрики
        if stats.total_trades > 0:
            stats.win_rate = stats.wins / stats.total_trades

        if stats.wins > 0:
            stats.avg_win = stats.total_wins_pnl / stats.wins
        if stats.losses > 0:
            stats.avg_loss = stats.total_losses_pnl / stats.losses

        if stats.total_trades > 0:
            stats.avg_pnl = stats.total_pnl / stats.total_trades

        if stats.total_losses_pnl > 0:
            stats.profit_factor = stats.total_wins_pnl / stats.total_losses_pnl

        if pnls:
            stats.largest_win = max(pnls)
            stats.largest_loss = min(pnls)

        # Win rate по символам
        for symbol, data in stats.by_symbol.items():
            if data["trades"] > 0:
                data["win_rate"] = data["wins"] / data["trades"]
            else:
                data["win_rate"] = 0.0

        return stats

    def get_equity_curve(self) -> List[dict]:
        """Возвращает equity curve по снимкам портфеля."""
        cursor = self.db.execute(
            """
            SELECT date, current_equity, drawdown_pct
            FROM portfolio_snapshots
            ORDER BY date
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "date": row["date"],
                "equity": row["current_equity"],
                "drawdown_pct": row["drawdown_pct"],
            }
            for row in rows
        ]

    def get_decisions_summary(self) -> dict:
        """Возвращает сводку по решениям Confluence."""
        cursor = self.db.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(passed_gate) as passed,
                SUM(CASE WHEN direction = 'BUY' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN direction = 'SELL' THEN 1 ELSE 0 END) as sells,
                SUM(CASE WHEN direction = 'HOLD' THEN 1 ELSE 0 END) as holds,
                AVG(confluence_score) as avg_score
            FROM decisions
            """
        )
        row = cursor.fetchone()

        return {
            "total_decisions": row["total"] or 0,
            "passed_gate": row["passed"] or 0,
            "pass_rate": (row["passed"] / row["total"]) if row["total"] else 0.0,
            "buys": row["buys"] or 0,
            "sells": row["sells"] or 0,
            "holds": row["holds"] or 0,
            "avg_confluence_score": row["avg_score"] or 0.0,
        }

    def get_trades_by_symbol(self, symbol: str) -> List[dict]:
        """Возвращает все сделки по символу."""
        cursor = self.db.execute(
            """
            SELECT * FROM trades
            WHERE symbol = ?
            ORDER BY entry_time
            """,
            (symbol,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_trades(self) -> List[dict]:
        """Возвращает все сделки."""
        cursor = self.db.execute("SELECT * FROM trades ORDER BY entry_time")
        return [dict(row) for row in cursor.fetchall()]

    def get_events_for_position(self, position_id: str) -> List[dict]:
        """Возвращает все события позиции."""
        cursor = self.db.execute(
            "SELECT * FROM events WHERE position_id = ? ORDER BY timestamp",
            (position_id,),
        )
        return [dict(row) for row in cursor.fetchall()]