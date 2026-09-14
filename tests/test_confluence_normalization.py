"""Тесты нормализации весов и confluence score.

Проверяют фикс Бага A:
- WeightManager.get_weights всегда даёт сумму 1.0.
- WeightManager._validate_weights нормализует, а не падает.
- ScoringSystem.compute возвращает взвешенное среднее confidence в [0,1].
- Интеграция: сигнал с 2 анализаторами проходит Gate BALANCED.
"""

import math

import pytest

from src.confluence.weight_manager import WeightManager
from src.confluence.regime_detector import MarketRegime
from src.confluence.scoring import ScoringSystem
from src.confluence.gate import ConfluenceGate, GateMode
from src.models import AnalyzerSignal, SignalDirection


# ---------- WeightManager ----------

def test_get_weights_normalized_all_analyzers_bull():
    wm = WeightManager()
    weights = wm.get_weights(MarketRegime.BULL)
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)
    assert set(weights.keys()) == {"trend", "elliott_wave", "volatility", "volume"}


def test_get_weights_normalized_subset_of_analyzers():
    """Раньше сумма падала до ~0.57 — теперь должна быть 1.0."""
    wm = WeightManager()
    weights = wm.get_weights(
        MarketRegime.BULL,
        analyzers=["trend", "volume"],
    )
    assert set(weights.keys()) == {"trend", "volume"}
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)

    # Пропорции должны сохраниться относительно regime-множителей:
    # trend: 0.35 * 1.2 = 0.42; volume: 0.15 * 1.0 = 0.15
    # после нормализации: trend ≈ 0.7368, volume ≈ 0.2632
    assert weights["trend"] > weights["volume"]
    assert math.isclose(
        weights["trend"] / weights["volume"],
        0.42 / 0.15,
        rel_tol=1e-9,
    )


def test_get_weights_normalized_chop_subset():
    wm = WeightManager()
    weights = wm.get_weights(
        MarketRegime.CHOP,
        analyzers=["elliott_wave", "volatility"],
    )
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_get_weights_empty_analyzers_returns_empty():
    wm = WeightManager()
    weights = wm.get_weights(MarketRegime.BULL, analyzers=["nonexistent"])
    assert weights == {}


def test_validate_weights_normalizes_instead_of_raising():
    """Было: ValueError. Стало: нормализация + warning."""
    wm = WeightManager(
        base_weights={"trend": 0.7, "volume": 0.7}  # сумма = 1.4
    )
    base = wm.get_base_weights()
    assert math.isclose(sum(base.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(base["trend"], 0.5, abs_tol=1e-9)
    assert math.isclose(base["volume"], 0.5, abs_tol=1e-9)


def test_validate_weights_zero_sum_raises():
    """Граничный случай: сумма 0 — это уже не нормализуемо."""
    with pytest.raises(ValueError):
        WeightManager(base_weights={"trend": 0.0, "volume": 0.0})


def test_validate_weights_negative_sum_raises():
    with pytest.raises(ValueError):
        WeightManager(base_weights={"trend": -0.5, "volume": 0.5})


def test_validate_weights_empty_dict_raises():
    with pytest.raises(ValueError):
        WeightManager(base_weights={})


# ---------- ScoringSystem ----------

def _sig(source: str, direction: SignalDirection, conf: float) -> AnalyzerSignal:
    return AnalyzerSignal(
        direction=direction,
        confidence=conf,
        reason=f"test {source}",
        metadata={"symbol": "TEST", "timeframe": "1h"},
        source=source,
    )


def test_scoring_perfect_bull_confidence_equals_one():
    """
    Все анализаторы дают BUY с confidence=1.0, веса нормированы.
    Взвешенное среднее = 1.0. Раньше было бы 1.04 или 0.57 —
    в зависимости от активных анализаторов.
    """
    ss = ScoringSystem(threshold=0.40)
    signals = [
        _sig("trend", SignalDirection.BUY, 1.0),
        _sig("elliott_wave", SignalDirection.BUY, 1.0),
        _sig("volatility", SignalDirection.BUY, 1.0),
        _sig("volume", SignalDirection.BUY, 1.0),
    ]
    weights = {
        "trend": 0.42, "elliott_wave": 0.27,
        "volatility": 0.20, "volume": 0.15,
    }  # сумма = 1.04 — ненормированные старые веса

    result = ss.compute(signals, weights)
    assert result.direction == SignalDirection.BUY
    assert math.isclose(result.bull_score, 1.0, abs_tol=1e-9)
    assert math.isclose(result.confluence_score, 1.0, abs_tol=1e-9)
    assert math.isclose(result.total_weight_used, 1.04, abs_tol=1e-9)


def test_scoring_weighted_average_two_analyzers():
    """
    2 активных анализатора с весами после нормализации.
    Ожидаемый score = 0.8*0.737 + 0.5*0.263 ≈ 0.721.
    """
    ss = ScoringSystem(threshold=0.40)
    signals = [
        _sig("trend", SignalDirection.BUY, 0.8),
        _sig("volume", SignalDirection.BUY, 0.5),
    ]
    weights = {"trend": 0.42 / 0.57, "volume": 0.15 / 0.57}

    result = ss.compute(signals, weights)
    expected = 0.8 * (0.42 / 0.57) + 0.5 * (0.15 / 0.57)
    assert math.isclose(result.bull_score, expected, rel_tol=1e-9)
    assert math.isclose(result.confluence_score, expected, rel_tol=1e-9)


def test_scoring_empty_signals_hold():
    ss = ScoringSystem()
    result = ss.compute([], weights={})
    assert result.direction == SignalDirection.HOLD
    assert result.confluence_score == 0.0


def test_scoring_no_matching_weights_hold():
    """Ни один сигнал не имеет веса — total_weight=0, score=0, HOLD."""
    ss = ScoringSystem(threshold=0.40)
    signals = [_sig("unknown", SignalDirection.BUY, 0.9)]
    result = ss.compute(signals, weights={"trend": 0.5})
    assert result.direction == SignalDirection.HOLD
    assert result.confluence_score == 0.0
    assert result.total_weight_used == 0.0


def test_scoring_conflicting_directions():
    """BUY 0.9, SELL 0.6, веса 0.5/0.5 → bull=0.45, bear=0.30 → BUY."""
    ss = ScoringSystem(threshold=0.40)
    signals = [
        _sig("trend", SignalDirection.BUY, 0.9),
        _sig("volume", SignalDirection.SELL, 0.6),
    ]
    weights = {"trend": 0.5, "volume": 0.5}
    result = ss.compute(signals, weights)
    assert result.direction == SignalDirection.BUY
    assert math.isclose(result.bull_score, 0.45, abs_tol=1e-9)
    assert math.isclose(result.bear_score, 0.30, abs_tol=1e-9)


# ---------- Интеграция: WeightManager + ScoringSystem + Gate ----------

def test_integration_two_analyzers_pass_balanced_gate():
    """
    Главный сценарий бага A:
    BULL, 2 активных анализатора (trend+volume), conf=0.8/0.5.
    До фикса score=0.411 → Gate BALANCED (0.65) закрыт.
    После фикса score≈0.721 → Gate открыт.
    """
    wm = WeightManager()
    ss = ScoringSystem(threshold=0.40)
    gate = ConfluenceGate(mode=GateMode.BALANCED)  # threshold=0.65

    signals = [
        _sig("trend", SignalDirection.BUY, 0.8),
        _sig("volume", SignalDirection.BUY, 0.5),
    ]

    weights = wm.get_weights(
        MarketRegime.BULL,
        analyzers=["trend", "volume"],
    )
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)

    scoring = ss.compute(signals, weights)
    assert scoring.direction == SignalDirection.BUY
    assert scoring.confluence_score > 0.65, (
        f"score={scoring.confluence_score:.3f} должен пройти Gate 0.65"
    )
    assert gate.is_open(scoring) is True


def test_integration_all_four_analyzers_pass_aggressive_gate():
    """4 анализатора, все BUY, conf=0.7. Score=0.7 → Gate AGGRESSIVE (0.50)."""
    wm = WeightManager()
    ss = ScoringSystem(threshold=0.40)
    gate = ConfluenceGate(mode=GateMode.AGGRESSIVE)

    signals = [
        _sig("trend", SignalDirection.BUY, 0.7),
        _sig("elliott_wave", SignalDirection.BUY, 0.7),
        _sig("volatility", SignalDirection.BUY, 0.7),
        _sig("volume", SignalDirection.BUY, 0.7),
    ]
    weights = wm.get_weights(MarketRegime.BULL)

    scoring = ss.compute(signals, weights)
    assert math.isclose(scoring.confluence_score, 0.7, abs_tol=1e-9)
    assert gate.is_open(scoring) is True


def test_integration_weak_signal_blocked_by_conservative_gate():
    """4 анализатора conf=0.5 → score=0.5 → Gate CONSERVATIVE (0.75) закрыт."""
    wm = WeightManager()
    ss = ScoringSystem(threshold=0.40)
    gate = ConfluenceGate(mode=GateMode.CONSERVATIVE)

    signals = [
        _sig("trend", SignalDirection.BUY, 0.5),
        _sig("elliott_wave", SignalDirection.BUY, 0.5),
        _sig("volatility", SignalDirection.BUY, 0.5),
        _sig("volume", SignalDirection.BUY, 0.5),
    ]
    weights = wm.get_weights(MarketRegime.BULL)

    scoring = ss.compute(signals, weights)
    assert math.isclose(scoring.confluence_score, 0.5, abs_tol=1e-9)
    assert gate.is_open(scoring) is False