"""Portfolio Risk Manager: главный класс управления риском портфеля."""

import logging
from datetime import datetime, timezone, date
from typing import List, Optional

from src.models import SignalDirection
from src.portfolio.state import PortfolioState, DailyStats
from src.portfolio.limits import RiskLimits
from src.portfolio.guard import PortfolioGuard, GuardResult
from src.position.position import Position, CloseReason

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Управляет риском портфеля в целом.

    Функции:
    - Проверка лимитов перед открытием сделки
    - Обновление состояния при закрытии сделок
    - Автоматическая пауза при критических событиях
    - Дневная статистика
    """

    def __init__(
        self,
        starting_equity: float,
        limits: Optional[RiskLimits] = None,
    ):
        """
        Args:
            starting_equity: Начальный капитал
            limits: Настройки лимитов (по умолчанию — стандартные)
        """
        self.limits = limits or RiskLimits()
        self.guard = PortfolioGuard(self.limits)
        self.state = PortfolioState(
            starting_equity=starting_equity,
            current_equity=starting_equity,
            peak_equity=starting_equity,
        )
        self.state.start_new_day(date.today())
        logger.info(
            "Portfolio Risk Manager инициализирован: equity=%.2f",
            starting_equity,
        )

    def can_open(
        self,
        symbol: str,
        risk_pct: float = 0.01,
    ) -> GuardResult:
        """
        Проверяет, можно ли открыть новую сделку.

        Args:
            symbol: Торговый инструмент
            risk_pct: Риск сделки

        Returns:
            GuardResult с решением
        """
        return self.guard.check(self.state, symbol, risk_pct)

    def register_position_opened(self, position: Position) -> None:
        """Регистрирует открытие позиции."""
        symbol = position.symbol
        self.state.open_positions_by_symbol[symbol] = (
            self.state.open_positions_by_symbol.get(symbol, 0) + 1
        )

        group = self.guard._find_correlation_group(symbol)
        if group is not None:
            self.state.open_positions_by_group[group] = (
                self.state.open_positions_by_group.get(group, 0) + 1
            )

        logger.debug(
            "Позиция %s открыта. Открыто: %d (по %s: %d)",
            position.id, self.state.total_open_positions,
            symbol, self.state.open_positions_by_symbol[symbol],
        )

    def register_position_closed(self, position: Position) -> None:
        """
        Регистрирует закрытие позиции.

        Обновляет:
        - equity
        - дневную статистику
        - счётчики позиций
        - серии побед/поражений
        - проверяет необходимость паузы
        """
        symbol = position.symbol

        # Уменьшаем счётчики
        if symbol in self.state.open_positions_by_symbol:
            self.state.open_positions_by_symbol[symbol] -= 1
            if self.state.open_positions_by_symbol[symbol] <= 0:
                del self.state.open_positions_by_symbol[symbol]

        group = self.guard._find_correlation_group(symbol)
        if group is not None and group in self.state.open_positions_by_group:
            self.state.open_positions_by_group[group] -= 1
            if self.state.open_positions_by_group[group] <= 0:
                del self.state.open_positions_by_group[group]

        # Обновляем equity и статистику
        pnl = position.realized_pnl
        self.state.register_trade_result(pnl)

        # Проверяем серию убытков
        if (
            self.state.consecutive_losses >=
            self.limits.pause_after_consecutive_losses
        ):
            self.state.pause(
                reason=(
                    f"Серия убытков: {self.state.consecutive_losses} подряд"
                ),
                hours=self.limits.pause_duration_hours,
            )

        # Проверяем просадку
        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            self.state.pause(
                reason=(
                    f"Просадка {self.state.current_drawdown_pct * 100:.2f}% "
                    f">= {self.limits.max_drawdown_pct * 100:.2f}%"
                ),
                hours=self.limits.pause_duration_hours * 2,
            )

        logger.info(
            "Позиция %s закрыта: pnl=%.2f, equity=%.2f, "
            "drawdown=%.2f%%, consecutive_losses=%d",
            position.id, pnl, self.state.current_equity,
            self.state.current_drawdown_pct * 100,
            self.state.consecutive_losses,
        )

    def update_equity(self, new_equity: float) -> None:
        """
        Обновляет equity (например, после переоценки открытых позиций).

        Также проверяет просадку.
        """
        self.state.update_equity(new_equity)

        if self.state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            if not self.state.is_paused:
                self.state.pause(
                    reason=(
                        f"Просадка {self.state.current_drawdown_pct * 100:.2f}%"
                    ),
                    hours=self.limits.pause_duration_hours * 2,
                )

    def start_new_day(self, today: Optional[date] = None) -> None:
        """Начинает новый торговый день."""
        self.state.start_new_day(today or date.today())

    def force_pause(self, reason: str, hours: int) -> None:
        """Принудительно ставит на паузу."""
        self.state.pause(reason=reason, hours=hours)

    def resume(self) -> None:
        """Возобновляет торговлю."""
        self.state.resume()

    def get_snapshot(self) -> dict:
        """Возвращает снимок состояния портфеля."""
        return self.state.get_snapshot()