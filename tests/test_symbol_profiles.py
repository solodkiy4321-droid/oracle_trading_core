"""Тесты профилей инструментов (базовые параметры)."""

import pytest

from src.engine.symbol_profiles import (
    SymbolProfile,
    SymbolProfileRegistry,
    create_default_registry,
)


def test_profile_defaults():
    """Профиль с дефолтными параметрами."""
    profile = SymbolProfile()
    assert profile.risk_per_trade_pct == 0.01
    assert profile.atr_multiplier == 1.5
    assert profile.atr_period == 14
    assert profile.default_rr_ratio == 2.0
    assert profile.max_position_pct == 1.0
    assert profile.gate_mode_override is None
    assert profile.weights_override is None
    assert profile.harmonic_disabled_patterns is None


def test_profile_custom():
    """Профиль с кастомными параметрами."""
    profile = SymbolProfile(
        risk_per_trade_pct=0.007,
        atr_multiplier=2.5,
        max_position_pct=0.5,
        gate_mode_override="conservative",
        weights_override={
            "indicators": 0.6,
            "harmonic": 0.2,
            "support_resistance": 0.2,
        },
        harmonic_disabled_patterns=["Butterfly"],
    )
    assert profile.risk_per_trade_pct == 0.007
    assert profile.atr_multiplier == 2.5
    assert profile.gate_mode_override == "conservative"
    assert profile.harmonic_disabled_patterns == ["Butterfly"]


def test_profile_validate_valid():
    """Валидный профиль проходит проверку."""
    profile = SymbolProfile(risk_per_trade_pct=0.01, atr_multiplier=2.0)
    profile.validate()


def test_profile_validate_invalid_risk():
    """Неверный risk → ошибка."""
    profile = SymbolProfile(risk_per_trade_pct=0.5)
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_validate_invalid_atr():
    """Неверный ATR multiplier → ошибка."""
    profile = SymbolProfile(atr_multiplier=-1.0)
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_validate_invalid_rr():
    """Неверный R:R → ошибка."""
    profile = SymbolProfile(default_rr_ratio=0.0)
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_validate_invalid_gate():
    """Неверный gate_mode_override → ошибка."""
    profile = SymbolProfile(gate_mode_override="invalid_mode")
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_validate_valid_gate():
    """Валидный gate_mode_override проходит."""
    for mode in ["aggressive", "balanced", "conservative"]:
        profile = SymbolProfile(gate_mode_override=mode)
        profile.validate()


def test_profile_weights_override_validation():
    """Сумма weights_override должна быть 1.0."""
    profile = SymbolProfile(
        weights_override={"indicators": 0.5, "harmonic": 0.3},
    )
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_weights_override_valid():
    """Валидный weights_override проходит."""
    profile = SymbolProfile(
        weights_override={"indicators": 0.5, "harmonic": 0.5},
    )
    profile.validate()


def test_profile_invalid_disabled_pattern():
    """Неверный паттерн в disabled_patterns → ошибка."""
    profile = SymbolProfile(
        harmonic_disabled_patterns=["InvalidPattern"],
    )
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_valid_disabled_patterns():
    """Валидные паттерны проходят."""
    profile = SymbolProfile(
        harmonic_disabled_patterns=["Butterfly", "Crab"],
    )
    profile.validate()


def test_registry_register_get():
    """Реестр сохраняет и возвращает профиль."""
    registry = SymbolProfileRegistry()
    profile = SymbolProfile(risk_per_trade_pct=0.007, atr_multiplier=2.5)
    registry.register("BTC-USD", profile)

    retrieved = registry.get("BTC-USD")
    assert retrieved.risk_per_trade_pct == 0.007
    assert retrieved.atr_multiplier == 2.5


def test_registry_fallback_default():
    """Неизвестный символ → default."""
    registry = SymbolProfileRegistry()
    profile = registry.get("UNKNOWN")
    assert profile.risk_per_trade_pct == 0.01
    assert profile.atr_multiplier == 1.5


def test_registry_prefix_match():
    """Префиксное совпадение работает."""
    registry = SymbolProfileRegistry()
    registry.register("BTC", SymbolProfile(risk_per_trade_pct=0.005))

    profile = registry.get("BTCUSDT")
    assert profile.risk_per_trade_pct == 0.005


def test_registry_case_insensitive():
    """Реестр не зависит от регистра."""
    registry = SymbolProfileRegistry()
    registry.register("btc-usd", SymbolProfile(risk_per_trade_pct=0.005))

    profile = registry.get("BTC-USD")
    assert profile.risk_per_trade_pct == 0.005


def test_default_registry_btc_base_params():
    """
    BTC в default registry — БАЗОВЫЕ параметры после отката.

    Мы откатили weights_override и harmonic_disabled_patterns, потому что
    диагностика на 17-19 сделках дала ненадёжные выводы.
    """
    registry = create_default_registry()
    btc = registry.get("BTC-USD")

    assert btc.risk_per_trade_pct == 0.007
    assert btc.atr_multiplier == 2.5
    assert btc.max_position_pct == 0.5
    assert btc.gate_mode_override is None
    assert btc.weights_override is None, (
        "BTC использует базовые веса после отката"
    )
    assert btc.harmonic_disabled_patterns is None, (
        "BTC не отключает паттерны harmonic"
    )


def test_default_registry_eth_base_params():
    """
    ETH в default registry — БАЗОВЫЕ параметры после отката.

    Butterfly снова включён, потому что его отключение сломало систему.
    """
    registry = create_default_registry()
    eth = registry.get("ETH-USD")

    assert eth.risk_per_trade_pct == 0.01
    assert eth.atr_multiplier == 1.5
    assert eth.gate_mode_override is None
    assert eth.weights_override is None
    assert eth.harmonic_disabled_patterns is None, (
        "ETH не отключает паттерны harmonic после отката"
    )


def test_default_registry_aapl():
    """AAPL в default registry — базовые параметры."""
    registry = create_default_registry()
    aapl = registry.get("AAPL")

    assert aapl.risk_per_trade_pct == 0.015
    assert aapl.atr_multiplier == 2.0
    assert aapl.gate_mode_override is None
    assert aapl.weights_override is None
    assert aapl.harmonic_disabled_patterns is None