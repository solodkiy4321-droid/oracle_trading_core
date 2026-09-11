"""Тесты профилей инструментов."""

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
    )
    assert profile.risk_per_trade_pct == 0.007
    assert profile.atr_multiplier == 2.5
    assert profile.gate_mode_override == "conservative"


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


def test_default_registry_btc_has_base_params():
    """
    BTC в default registry использует БАЗОВЫЕ параметры.

    Мы намеренно убрали gate/weights override — для сбора статистики.
    BTC торгует с базовым Gate 0.65 и базовыми весами.
    """
    registry = create_default_registry()
    btc = registry.get("BTC-USD")

    assert btc.risk_per_trade_pct == 0.007
    assert btc.atr_multiplier == 2.5
    assert btc.max_position_pct == 0.5
    assert btc.gate_mode_override is None, (
        "BTC должен использовать базовый Gate (не override)"
    )
    assert btc.weights_override is None, (
        "BTC должен использовать базовые веса (не override)"
    )


def test_default_registry_eth_has_no_overrides():
    """ETH в default registry использует базовые параметры."""
    registry = create_default_registry()
    eth = registry.get("ETH-USD")

    assert eth.risk_per_trade_pct == 0.01
    assert eth.atr_multiplier == 1.5
    assert eth.gate_mode_override is None
    assert eth.weights_override is None


def test_default_registry_aapl():
    """AAPL в default registry."""
    registry = create_default_registry()
    aapl = registry.get("AAPL")

    assert aapl.risk_per_trade_pct == 0.015
    assert aapl.atr_multiplier == 2.0
    assert aapl.gate_mode_override is None