"""Конфигурация Trading Engine."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """Настройки Trading Engine."""

    # Капитал
    starting_equity: float = 10000.0
    equity_currency: str = "USD"

    # Symbol/Timeframe
    symbol: str = "BTC-USD"
    timeframe: str = "1h"

    # Анализаторы
    enable_indicators: bool = True
    enable_harmonic: bool = True
    enable_support_resistance: bool = True

    # Confluence
    gate_mode: str = "balanced"           # 'aggressive', 'balanced', 'conservative'

    # Risk Manager
    risk_per_trade_pct: float = 0.01      # 1% на сделку
    atr_period: int = 14
    atr_multiplier: float = 1.5
    default_rr_ratio: float = 2.0

    # Position Manager
    breakeven_after_tp: int = 1
    trailing_after_tp: int = 2
    trailing_atr_multiplier: float = 1.0
    max_position_age_bars: int = 100
    commission_pct: float = 0.001

    # Portfolio Risk
    daily_loss_limit_pct: float = 0.03
    daily_profit_target_pct: float = 0.06
    max_drawdown_pct: float = 0.15
    max_open_positions: int = 5
    max_positions_per_symbol: int = 1
    max_positions_per_group: int = 2
    pause_after_consecutive_losses: int = 3
    pause_duration_hours: int = 24

    # Journal
    journal_db_path: str = "journal.db"
    snapshot_every_n_bars: int = 24       # Снимок портфеля раз в 24 бара

    # Логирование
    log_level: str = "INFO"

    def validate(self) -> None:
        """Проверяет корректность конфигурации."""
        if self.starting_equity <= 0:
            raise ValueError(
                f"starting_equity должен быть > 0, получено {self.starting_equity}"
            )
        if not 0 < self.risk_per_trade_pct <= 0.05:
            raise ValueError(
                f"risk_per_trade_pct должен быть в (0, 0.05], "
                f"получено {self.risk_per_trade_pct}"
            )
        if self.atr_multiplier <= 0:
            raise ValueError(
                f"atr_multiplier должен быть > 0, "
                f"получено {self.atr_multiplier}"
            )
        if self.gate_mode not in ("aggressive", "balanced", "conservative"):
            raise ValueError(
                f"Неверный gate_mode: {self.gate_mode}. "
                f"Допустимые: 'aggressive', 'balanced', 'conservative'"
            )