"""Тесты профилей инструментов (рабочая версия, +$4033).

ОТКАТ ПОДХОДА 3:
- Без regime_profiles
- Без apply_regime_multipliers
- Per-symbol веса и harmonic_min_pattern_confidence
"""

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
    assert profile.harmonic_min_pattern_confidence is None


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
        harmonic_min_pattern_confidence=0.5,
    )
    assert profile.risk_per_trade_pct == 0.007
    assert profile.atr_multiplier == 2.5
    assert profile.gate_mode_override == "conservative"
    assert profile.harmonic_min_pattern_confidence == 0.5


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


def test_profile_invalid_pattern_confidence():
    """Неверный harmonic_min_pattern_confidence → ошибка."""
    profile = SymbolProfile(harmonic_min_pattern_confidence=1.5)
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_valid_pattern_confidence():
    """Валидный harmonic_min_pattern_confidence проходит."""
    profile = SymbolProfile(harmonic_min_pattern_confidence=0.5)
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


def test_default_registry_btc_has_4_analyzers():
    """
    BTC в default registry имеет веса 4 анализаторов.

    Веса: S/R 0.40, harmonic 0.25, indicators 0.20, elliott_wave 0.15.
    """
    registry = create_default_registry()
    btc = registry.get("BTC-USD")

    assert btc.risk_per_trade_pct == 0.007
    assert btc.atr_multiplier == 2.5
    assert btc.max_position_pct == 0.5
    assert btc.gate_mode_override is None
    assert btc.weights_override is not None
    assert btc.weights_override["support_resistance"] == 0.40
    assert btc.weights_override["harmonic"] == 0.25
    assert btc.weights_override["indicators"] == 0.20
    assert btc.weights_override["elliott_wave"] == 0.15
    assert btc.harmonic_min_pattern_confidence == 0.50


def test_default_registry_eth_has_4_analyzers():
    """ETH имеет веса 4 анализаторов."""
    registry = create_default_registry()
    eth = registry.get("ETH-USD")

    assert eth.risk_per_trade_pct == 0.01
    assert eth.atr_multiplier == 1.5
    assert eth.weights_override is not None
    assert "elliott_wave" in eth.weights_override
    assert eth.harmonic_min_pattern_confidence == 0.50


def test_default_registry_aapl():
    """AAPL имеет веса 4 анализаторов."""
    registry = create_default_registry()
    aapl = registry.get("AAPL")

    assert aapl.risk_per_trade_pct == 0.015
    assert aapl.atr_multiplier == 2.0
    assert aapl.weights_override is not None
    assert "elliott_wave" in aapl.weights_override


def test_default_registry_all_pairs_have_weights():
    """Все 7 пар имеют weights_override."""
    registry = create_default_registry()
    for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "AVAX-USD",
                   "DOT-USD", "ATOM-USD", "AAPL"]:
        profile = registry.get(symbol)
        assert profile.weights_override is not None, f"{symbol} без weights"
        total = sum(profile.weights_override.values())
        assert abs(total - 1.0) < 1e-6, f"{symbol} сумма весов != 1.0"