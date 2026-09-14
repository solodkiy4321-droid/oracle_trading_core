"""Тесты профилей инструментов.

Актуальная версия:
- 4 активных анализатора: trend, elliott_wave, volatility, volume.
- harmonic_min_pattern_confidence больше не существует.
- AAPL / AVAX в реестр не входят.
"""

import pytest

from src.engine.symbol_profiles import (
    SymbolProfile,
    SymbolProfileRegistry,
    create_default_registry,
)


# ---------- Базовая валидация ----------

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
        weights_override={"trend": 0.5, "volume": 0.3},
    )
    with pytest.raises(ValueError):
        profile.validate()


def test_profile_weights_override_valid():
    """Валидный weights_override проходит."""
    profile = SymbolProfile(
        weights_override={"trend": 0.5, "volume": 0.5},
    )
    profile.validate()


# ---------- Реестр ----------

def test_registry_register_get():
    """Реестр сохраняет и возвращает профиль."""
    registry = SymbolProfileRegistry()
    profile = SymbolProfile(risk_per_trade_pct=0.007, atr_multiplier=2.5)
    registry.register("BTC-USD", profile)

    retrieved = registry.get("BTC-USD")
    assert retrieved.risk_per_trade_pct == 0.007
    assert retrieved.atr_multiplier == 2.5


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


# ---------- Default registry ----------

def test_default_registry_btc():
    """
    BTC в default registry использует профиль с 4 анализаторами.

    Актуальные веса: elliott_wave 0.7, volume 0.3 (остальные 0).
    """
    registry = create_default_registry()
    btc = registry.get("BTC-USD")

    assert btc.timeframe == "2h"
    assert btc.atr_multiplier == 2.0
    assert btc.default_rr_ratio == 3.0
    assert btc.filter_chop is True
    assert btc.weights_override is not None
    assert btc.weights_override["elliott_wave"] == 0.7
    assert btc.weights_override["volume"] == 0.3
    assert btc.weights_override["trend"] == 0.0
    assert btc.weights_override["volatility"] == 0.0
    total = sum(btc.weights_override.values())
    assert abs(total - 1.0) < 1e-6


def test_default_registry_eth():
    """ETH использует профиль с trend-ориентированными весами."""
    registry = create_default_registry()
    eth = registry.get("ETH-USD")

    assert eth.timeframe == "1h"
    assert eth.atr_multiplier == 2.5
    assert eth.gate_mode_override == "balanced"
    assert eth.weights_override is not None
    assert "trend" in eth.weights_override
    assert "elliott_wave" in eth.weights_override
    total = sum(eth.weights_override.values())
    assert abs(total - 1.0) < 1e-6


def test_default_registry_all_pairs_have_weights():
    """Все 5 пар из default registry имеют weights_override."""
    registry = create_default_registry()
    for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "ATOM-USD"]:
        profile = registry.get(symbol)
        assert profile.weights_override is not None, f"{symbol} без weights"
        total = sum(profile.weights_override.values())
        assert abs(total - 1.0) < 1e-6, (
            f"{symbol} сумма весов = {total}, ожидалось 1.0"
        )


def test_default_registry_weights_use_active_analyzers():
    """Все ключи в weights_override — это активные анализаторы."""
    active = {"trend", "elliott_wave", "volatility", "volume"}
    registry = create_default_registry()
    for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "ATOM-USD"]:
        profile = registry.get(symbol)
        if profile.weights_override is None:
            continue
        for key in profile.weights_override:
            assert key in active, (
                f"{symbol}: неактивный анализатор в весах — {key}"
            )