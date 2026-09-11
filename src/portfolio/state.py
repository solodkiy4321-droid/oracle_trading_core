"""Состояние портфеля на текущий момент."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Статистика за день."""
    date: date
    starting_equity: float
    realized_pnl: float = 0.0
    trades_count: int = 0
    wins: int = 0
    losses: int = 0

    @property
    def pnl_pct(self) -> float:
        """P&L в процентах от начального equity."""
        if self.starting_equity <= 0:
            return 0.0
        return self.realized_pnl / self.starting_equity


@dataclass
class PortfolioState:
    """
    Текущее состояние портфеля.

    Отслеживает:
    - Equity (текущий баланс + открытые позиции)
    - Peak equity (для расчёта просадки)
    - Дневную статистику
    - Историю сделок
    """
    # Баланс
    starting_equity: float                    # Начальный капитал
    current_equity: float                     # Текущий баланс (без открытых позиций)
    peak_equity: float                        # Пиковый equity (для просадки)

    # Позиции
    open_positions_by_symbol: Dict[str, int] = field(default_factory=dict)
    open_positions_by_group: Dict[str, int] = field(default_factory=dict)

    # Дневная статистика
    daily_stats: Optional[DailyStats] = None

    # История
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0

    # Пауза
    paused_until: Optional[datetime] = None
    pause_reason: str = ""

    # Время последнего обновления
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Инициализация."""
        if self.peak_equity == 0.0:
            self.peak_equity = self.current_equity

    @property
    def total_open_positions(self) -> int:
        """Общее количество открытых позиций."""
        return sum(self.open_positions_by_symbol.values())

    @property
    def current_drawdown_pct(self) -> float:
        """Текущая просадка от пика."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def total_pnl(self) -> float:
        """Общий P&L с начала."""
        return self.current_equity - self.starting_equity

    @property
    def total_pnl_pct(self) -> float:
        """Общий P&L в процентах."""
        if self.starting_equity <= 0:
            return 0.0
        return self.total_pnl / self.starting_equity

    @property
    def is_paused(self) -> bool:
        """Портфель на паузе."""
        if self.paused_until is None:
            return False
        return datetime.now(timezone.utc) < self.paused_until

    def start_new_day(self, today: date) -> None:
        """Начинает новый торговый день."""
        self.daily_stats = DailyStats(
            date=today,
            starting_equity=self.current_equity,
        )
        logger.info(
            "Новый торговый день: %s, equity=%.2f",
            today, self.current_equity,
        )

    def update_equity(self, new_equity: float) -> None:
        """Обновляет equity и peak."""
        self.current_equity = new_equity
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
        self.last_updated = datetime.now(timezone.utc)

    def register_trade_result(self, pnl: float) -> None:
        """
        Регистрирует результат закрытой сделки.

        Обновляет:
        - equity
        - дневную статистику
        - серии побед/поражений
        """
        # Обновляем equity
        self.current_equity += pnl

        # Обновляем дневную статистику
        if self.daily_stats is not None:
            self.daily_stats.realized_pnl += pnl
            self.daily_stats.trades_count += 1
            if pnl > 0:
                self.daily_stats.wins += 1
            elif pnl < 0:
                self.daily_stats.losses += 1

        # Обновляем серии
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.total_wins += 1
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.total_losses += 1

        self.total_trades += 1
        self.last_updated = datetime.now(timezone.utc)

        logger.debug(
            "Сделка закрыта: pnl=%.2f, equity=%.2f, daily_pnl=%.2f, "
            "consecutive_losses=%d",
            pnl, self.current_equity,
            self.daily_stats.realized_pnl if self.daily_stats else 0.0,
            self.consecutive_losses,
        )

    def pause(self, reason: str, hours: int) -> None:
        """Ставит портфель на паузу."""
        from datetime import timedelta
        self.paused_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.pause_reason = reason
        logger.warning(
            "Портфель на паузе до %s (%.1f часов). Причина: %s",
            self.paused_until, hours, reason,
        )

    def resume(self) -> None:
        """Снимает паузу."""
        self.paused_until = None
        self.pause_reason = ""
        logger.info("Портфель возобновлён")

    def get_snapshot(self) -> dict:
        """Возвращает снимок состояния."""
        return {
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "drawdown_pct": self.current_drawdown_pct,
            "open_positions": self.total_open_positions,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "daily_pnl": self.daily_stats.realized_pnl if self.daily_stats else 0.0,
            "daily_pnl_pct": self.daily_stats.pnl_pct if self.daily_stats else 0.0,
        }