"""Тесты синхронизации дефолтов EngineConfig и RiskLimits.

Проверяют Баг H:
- Одноимённые поля в EngineConfig и RiskLimits имеют одинаковые дефолты.
- RiskLimits() напрямую даёт те же значения, что TradingEngine
  при создании через EngineConfig.
"""

from dataclasses import fields

from src.engine.config import EngineConfig
from src.portfolio.limits import RiskLimits


# Поля, которые есть и в EngineConfig, и в RiskLimits
SYNC_FIELDS = [
    "daily_loss_limit_pct",
    "daily_profit_target_pct",
    "max_drawdown_pct",
    "max_open_positions",
    "max_positions_per_symbol",
    "max_positions_per_group",
    "pause_after_consecutive_losses",
    "pause_duration_hours",
]


def test_engineconfig_has_all_sync_fields():
    """Все SYNC_FIELDS есть в EngineConfig."""
    field_names = {f.name for f in fields(EngineConfig)}
    for name in SYNC_FIELDS:
        assert name in field_names, (
            f"EngineConfig не имеет поля {name}"
        )


def test_risklimits_has_all_sync_fields():
    """Все SYNC_FIELDS есть в RiskLimits."""
    field_names = {f.name for f in fields(RiskLimits)}
    for name in SYNC_FIELDS:
        assert name in field_names, (
            f"RiskLimits не имеет поля {name}"
        )


def test_defaults_are_equal():
    """Дефолты одноимённых полей совпадают."""
    config = EngineConfig()
    limits = RiskLimits()

    for name in SYNC_FIELDS:
        config_val = getattr(config, name)
        limits_val = getattr(limits, name)
        assert config_val == limits_val, (
            f"{name}: EngineConfig={config_val}, RiskLimits={limits_val} "
            f"(Bаг H — дефолты не синхронизированы)"
        )


def test_daily_loss_limit_synced():
    """daily_loss_limit_pct совпадает."""
    assert EngineConfig().daily_loss_limit_pct == RiskLimits().daily_loss_limit_pct


def test_daily_profit_target_synced():
    """daily_profit_target_pct совпадает."""
    assert (
        EngineConfig().daily_profit_target_pct
        == RiskLimits().daily_profit_target_pct
    )


def test_max_drawdown_synced():
    """max_drawdown_pct совпадает."""
    assert EngineConfig().max_drawdown_pct == RiskLimits().max_drawdown_pct


def test_pause_after_losses_synced():
    """pause_after_consecutive_losses совпадает."""
    assert (
        EngineConfig().pause_after_consecutive_losses
        == RiskLimits().pause_after_consecutive_losses
    )


def test_pause_duration_synced():
    """pause_duration_hours совпадает."""
    assert (
        EngineConfig().pause_duration_hours
        == RiskLimits().pause_duration_hours
    )


def test_max_positions_synced():
    """max_open_positions, max_positions_per_symbol, max_positions_per_group — совпадают."""
    config = EngineConfig()
    limits = RiskLimits()
    assert config.max_open_positions == limits.max_open_positions
    assert config.max_positions_per_symbol == limits.max_positions_per_symbol
    assert config.max_positions_per_group == limits.max_positions_per_group


def test_risklimits_validate_passes_with_defaults():
    """RiskLimits() с дефолтами проходит валидацию."""
    RiskLimits().validate()