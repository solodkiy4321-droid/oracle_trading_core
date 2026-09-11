"""Portfolio Guard: проверка лимитов перед открытием сделки."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.portfolio.state import PortfolioState
from src.portfolio.limits import RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Результат проверки лимитов."""
    allowed: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)


class PortfolioGuard:
    """
    Проверяет, можно ли открыть новую сделку.

    Проверки:
    1. Пауза (после серии убытков или критической просадки)
    2. Дневной лимит потерь
    3. Дневной лимит прибыли (опционально)
    4. Максимальная просадка
    5. Максимум открытых позиций
    6. Максимум позиций по инструменту
    7. Максимум позиций в корреляционной группе
    """

    def __init__(self, limits: RiskLimits):
        """
        Args:
            limits: Настройки лимитов
        """
        limits.validate()
        self.limits = limits

    def check(
        self,
        state: PortfolioState,
        symbol: str,
        risk_pct: float = 0.01,
    ) -> GuardResult:
        """
        Проверяет, можно ли открыть новую сделку.

        Args:
            state: Текущее состояние портфеля
            symbol: Торговый инструмент
            risk_pct: Риск сделки в процентах

        Returns:
            GuardResult с решением и причиной
        """
        warnings: List[str] = []

        # 1. Пауза
        if state.is_paused:
            return GuardResult(
                allowed=False,
                reason=f"Портфель на паузе: {state.pause_reason}",
            )

        # 2. Дневной лимит потерь
        if state.daily_stats is not None:
            daily_pnl_pct = state.daily_stats.pnl_pct
            if daily_pnl_pct <= -self.limits.daily_loss_limit_pct:
                return GuardResult(
                    allowed=False,
                    reason=(
                        f"Дневной лимит потерь достигнут: "
                        f"{daily_pnl_pct * 100:.2f}% <= "
                        f"-{self.limits.daily_loss_limit_pct * 100:.2f}%"
                    ),
                )
            # Предупреждение при приближении к лимиту
            if daily_pnl_pct <= -self.limits.daily_loss_limit_pct * 0.7:
                warnings.append(
                    f"Приближение к дневному лимиту: "
                    f"{daily_pnl_pct * 100:.2f}%"
                )

            # Дневной лимит прибыли
            if daily_pnl_pct >= self.limits.daily_profit_target_pct:
                return GuardResult(
                    allowed=False,
                    reason=(
                        f"Дневная цель по прибыли достигнута: "
                        f"{daily_pnl_pct * 100:.2f}% >= "
                        f"{self.limits.daily_profit_target_pct * 100:.2f}%"
                    ),
                )

        # 3. Максимальная просадка
        if state.current_drawdown_pct >= self.limits.max_drawdown_pct:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Максимальная просадка достигнута: "
                    f"{state.current_drawdown_pct * 100:.2f}% >= "
                    f"{self.limits.max_drawdown_pct * 100:.2f}%"
                ),
            )

        # 4. Максимум открытых позиций
        if state.total_open_positions >= self.limits.max_open_positions:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Максимум открытых позиций: "
                    f"{state.total_open_positions} >= "
                    f"{self.limits.max_open_positions}"
                ),
            )

        # 5. Максимум позиций по инструменту
        symbol_count = state.open_positions_by_symbol.get(symbol, 0)
        if symbol_count >= self.limits.max_positions_per_symbol:
            return GuardResult(
                allowed=False,
                reason=(
                    f"Максимум позиций по {symbol}: "
                    f"{symbol_count} >= "
                    f"{self.limits.max_positions_per_symbol}"
                ),
            )

        # 6. Корреляционная группа
        group = self._find_correlation_group(symbol)
        if group is not None:
            group_count = state.open_positions_by_group.get(group, 0)
            if group_count >= self.limits.max_positions_per_group:
                return GuardResult(
                    allowed=False,
                    reason=(
                        f"Максимум позиций в группе {group}: "
                        f"{group_count} >= "
                        f"{self.limits.max_positions_per_group}"
                    ),
                )

        # 7. Проверка риска
        if risk_pct > self.limits.max_risk_per_symbol_pct:
            warnings.append(
                f"Риск сделки {risk_pct * 100:.2f}% превышает лимит "
                f"{self.limits.max_risk_per_symbol_pct * 100:.2f}%"
            )

        return GuardResult(allowed=True, warnings=warnings)

    def _find_correlation_group(self, symbol: str) -> Optional[str]:
        """Находит корреляционную группу для символа."""
        for group_name, symbols in self.limits.correlation_groups.items():
            if symbol in symbols:
                return group_name
        return None