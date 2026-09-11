"""Тесты интеграции профилей с Trading Engine (базовые параметры)."""

import pytest

from src.engine.config import EngineConfig
from src.engine.symbol_profiles import (
    SymbolProfile,
    SymbolProfileRegistry,
    create_default_registry,
)


def test_config_uses_btc_profile():
    """
    Config получает профиль для BTC с БАЗОВЫМИ параметрами.

    После отката: weights_override=None, harmonic_disabled_patterns=None.
    """
    config = EngineConfig(symbol="BTC-USD", journal_db_path=":memory:")
    profile = config.get_symbol_profile()

    assert profile.risk_per_trade_pct == 0.007
    assert profile.atr_multiplier == 2.5
    assert profile.max_position_pct == 0.5
    assert profile.gate_mode_override is None
    assert profile.weights_override is None, (
        "BTC использует базовые веса после отката"
    )
    assert profile.harmonic_disabled_patterns is None


def test_config_uses_eth_profile():
    """
    Config получает профиль для ETH с БАЗОВЫМИ параметрами.

    После отката: Butterfly снова включён.
    """
    config = EngineConfig(symbol="ETH-USD", journal_db_path=":memory:")
    profile = config.get_symbol_profile()

    assert profile.risk_per_trade_pct == 0.01
    assert profile.atr_multiplier == 1.5
    assert profile.gate_mode_override is None
    assert profile.weights_override is None
    assert profile.harmonic_disabled_patterns is None, (
        "ETH не отключает паттерны harmonic после отката"
    )


def test_config_default_profile():
    """Неизвестный символ → default."""
    config = EngineConfig(symbol="UNKNOWN", journal_db_path=":memory:")
    profile = config.get_symbol_profile()

    assert profile.risk_per_trade_pct == 0.01
    assert profile.atr_multiplier == 1.5
    assert profile.gate_mode_override is None
    assert profile.weights_override is None
    assert profile.harmonic_disabled_patterns is None


def test_config_profiles_disabled():
    """Профили отключены — используются базовые параметры."""
    config = EngineConfig(
        symbol="BTC-USD",
        journal_db_path=":memory:",
        use_symbol_profiles=False,
        risk_per_trade_pct=0.02,
        atr_multiplier=3.0,
    )
    profile = config.get_symbol_profile()

    assert profile.risk_per_trade_pct == 0.02
    assert profile.atr_multiplier == 3.0
    assert profile.gate_mode_override is None
    assert profile.weights_override is None
    assert profile.harmonic_disabled_patterns is None
    assert "Профили отключены" in profile.notes


def test_config_custom_registry():
    """Кастомный реестр переопределяет default."""
    registry = SymbolProfileRegistry()
    registry.register(
        "CUSTOM",
        SymbolProfile(
            risk_per_trade_pct=0.02,
            atr_multiplier=3.0,
            gate_mode_override="aggressive",
            harmonic_disabled_patterns=["Crab"],
        ),
    )

    config = EngineConfig(
        symbol="CUSTOM",
        journal_db_path=":memory:",
        symbol_profiles=registry,
    )
    profile = config.get_symbol_profile()

    assert profile.risk_per_trade_pct == 0.02
    assert profile.atr_multiplier == 3.0
    assert profile.gate_mode_override == "aggressive"
    assert profile.harmonic_disabled_patterns == ["Crab"]