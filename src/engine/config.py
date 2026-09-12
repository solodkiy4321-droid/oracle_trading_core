"""Конфигурация Trading Engine."""

from dataclasses import dataclass
from typing import Optional

from src.engine.symbol_profiles import (
    SymbolProfile,
    SymbolProfileRegistry,
    create_default_registry,
)


@dataclass
class EngineConfig:
    """Настройки Trading Engine."""

    starting_equity: float = 10000.0
    equity_currency: str = "USD"

    symbol: str = "ETH-USD"
    timeframe: str = "1h"

    enable_trend: bool = True
    enable_elliott_wave: bool = True
    enable_volatility: bool = True
    enable_volume: bool = True

    enable_harmonic: bool = False
    enable_support_resistance: bool = False
    enable_momentum: bool = False
    enable_indicators: bool = False

    gate_mode: str = "aggressive"

    risk_per_trade_pct: float = 0.01
    atr_period: int = 14
    atr_multiplier: float = 2.0
    default_rr_ratio: float = 3.0

    breakeven_after_tp: int = 1
    trailing_after_tp: int = 2
    trailing_atr_multiplier: float = 1.0
    max_position_age_bars: int = 100
    commission_pct: float = 0.001

    daily_loss_limit_pct: float = 0.10
    daily_profit_target_pct: float = 0.50
    max_drawdown_pct: float = 0.50
    max_open_positions: int = 5
    max_positions_per_symbol: int = 1
    max_positions_per_group: int = 2
    pause_after_consecutive_losses: int = 10
    pause_duration_hours: int = 1

    journal_db_path: str = "journal.db"
    snapshot_every_n_bars: int = 24

    log_level: str = "INFO"

    use_symbol_profiles: bool = True
    symbol_profiles: Optional[SymbolProfileRegistry] = None

    def get_symbol_profile(self) -> SymbolProfile:
        if not self.use_symbol_profiles:
            return SymbolProfile(
                risk_per_trade_pct=self.risk_per_trade_pct,
                atr_multiplier=self.atr_multiplier,
                atr_period=self.atr_period,
                default_rr_ratio=self.default_rr_ratio,
                notes="Профили отключены",
            )
        registry = self.symbol_profiles
        if registry is None:
            registry = create_default_registry()
            self.symbol_profiles = registry
        return registry.get(self.symbol)

    def validate(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError("starting_equity > 0")
        if not 0 < self.risk_per_trade_pct <= 0.05:
            raise ValueError("risk_per_trade_pct в (0, 0.05]")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier > 0")
        if self.gate_mode not in ("aggressive", "balanced", "conservative"):
            raise ValueError("gate_mode неверный")
        if self.pause_after_consecutive_losses < 1:
            raise ValueError("pause_after_consecutive_losses >= 1")
        if self.pause_duration_hours < 1:
            raise ValueError("pause_duration_hours >= 1")