"""Тесты Confluence-движка."""

import numpy as np
import pandas as pd
import pytest

from src.confluence.regime_detector import RegimeDetector, MarketRegime
from src.confluence.weight_manager import WeightManager
from src.confluence.scoring import ScoringSystem
from src.confluence.gate import ConfluenceGate, GateMode
from src.confluence.engine import ConfluenceEngine
from src.models import AnalyzerSignal, SignalDirection


# ============ Фикстуры ============

def make_bull_data(n: int = 300) -> pd.DataFrame:
    """Восходящий тренд с сильным ADX."""
    np.random.seed(42)
    close = np.linspace(100, 200, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_bear_data(n: int = 300) -> pd.DataFrame:
    """Нисходящий тренд с сильным ADX."""
    np.random.seed(43)
    close = np.linspace(200, 100, n) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_chop_data(n: int = 300) -> pd.DataFrame:
    """Боковик."""
    np.random.seed(44)
    close = 100 + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


def make_signal(
    source: str, direction: SignalDirection, confidence: float
) -> AnalyzerSignal:
    """Создаёт тестовый сигнал."""
    return AnalyzerSignal(
        direction=direction,
        confidence=confidence,
        reason=f"test {source}",
        metadata={"symbol": "TEST", "timeframe": "1h"},
        source=source,
    )


# ============ RegimeDetector ============

def test_regime_bull():
    """BULL определяется корректно."""
    detector = RegimeDetector()
    regime = detector.detect(make_bull_data())
    assert regime == MarketRegime.BULL


def test_regime_bear():
    """BEAR определяется корректно."""
    detector = RegimeDetector()
    regime = detector.detect(make_bear_data())
    assert regime == MarketRegime.BEAR


def test_regime_chop():
    """CHOP определяется корректно."""
    detector = RegimeDetector()
    regime = detector.detect(make_chop_data())
    assert regime == MarketRegime.CHOP


def test_regime_insufficient_data():
    """Недостаточно данных — CHOP."""
    detector = RegimeDetector()
    regime = detector.detect(make_bull_data(n=50))
    assert regime == MarketRegime.CHOP


# ============ WeightManager ============

def test_weight_manager_base():
    """
    Базовые веса корректны и суммируются в 1.0.
    """
    wm = WeightManager()
    base = wm.get_base_weights()
    assert abs(base["indicators"] - 0.30) < 1e-6
    assert abs(base["harmonic"] - 0.30) < 1e-6
    assert abs(base["support_resistance"] - 0.40) < 1e-6
    assert abs(sum(base.values()) - 1.0) < 1e-6


def test_weight_manager_invalid_base():
    """Неверная сумма весов вызывает ошибку."""
    with pytest.raises(ValueError):
        WeightManager(base_weights={"indicators": 0.5, "harmonic": 0.4})


def test_weight_manager_regime_adaptation():
    """Веса адаптируются под режим."""
    wm = WeightManager()
    bull_weights = wm.get_weights(MarketRegime.BULL)
    chop_weights = wm.get_weights(MarketRegime.CHOP)

    assert bull_weights["indicators"] > chop_weights["indicators"]
    assert chop_weights["harmonic"] > bull_weights["harmonic"]
    assert chop_weights["support_resistance"] > bull_weights["support_resistance"]

    assert 0.9 < sum(bull_weights.values()) < 1.3
    assert 0.9 < sum(chop_weights.values()) < 1.5

    assert abs(sum(wm.get_base_weights().values()) - 1.0) < 1e-6


def test_weight_manager_single_analyzer_not_max():
    """
    Один анализатор не получает вес 1.0.
    Для indicators в BEAR: 0.30 × 1.2 = 0.36
    """
    wm = WeightManager()
    weights = wm.get_weights(MarketRegime.BEAR, analyzers=["indicators"])

    assert "indicators" in weights
    assert weights["indicators"] < 1.0
    assert abs(weights["indicators"] - 0.36) < 1e-6

    assert "harmonic" not in weights
    assert "support_resistance" not in weights


def test_weight_manager_empty_analyzers():
    """Пустой список анализаторов — пустые веса."""
    wm = WeightManager()
    weights = wm.get_weights(MarketRegime.BULL, analyzers=[])
    assert weights == {}


# ============ ScoringSystem ============

def test_scoring_empty():
    """Пустой список сигналов — HOLD."""
    system = ScoringSystem()
    result = system.compute([], {"indicators": 1.0})
    assert result.direction == SignalDirection.HOLD
    assert result.confluence_score == 0.0


def test_scoring_bull_aligned():
    """
    Согласованные BUY-сигналы дают направление BUY.

    С внутренним threshold=0.40:
    - Два сигнала с conf 0.8/0.7 и весами 0.30/0.30 → score 0.45 > 0.40 → BUY
    """
    system = ScoringSystem(threshold=0.40)
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.8),
        make_signal("harmonic", SignalDirection.BUY, 0.7),
    ]
    weights = {"indicators": 0.30, "harmonic": 0.30}
    result = system.compute(signals, weights)

    assert result.direction == SignalDirection.BUY
    # 0.8×0.30 + 0.7×0.30 = 0.24 + 0.21 = 0.45
    assert result.confluence_score > 0.4
    assert result.bull_score > result.bear_score


def test_scoring_three_analyzers_high_score():
    """
    Три согласованных сигнала дают высокий score.
    """
    system = ScoringSystem(threshold=0.40)
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.8),
        make_signal("harmonic", SignalDirection.BUY, 0.7),
        make_signal("support_resistance", SignalDirection.BUY, 0.9),
    ]
    weights = {"indicators": 0.30, "harmonic": 0.30, "support_resistance": 0.40}
    result = system.compute(signals, weights)

    assert result.direction == SignalDirection.BUY
    # 0.8×0.30 + 0.7×0.30 + 0.9×0.40 = 0.24 + 0.21 + 0.36 = 0.81
    assert result.confluence_score > 0.7


def test_scoring_below_threshold_hold():
    """
    Сигналы ниже внутреннего threshold дают HOLD.
    """
    system = ScoringSystem(threshold=0.40)
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.5),
        make_signal("harmonic", SignalDirection.BUY, 0.5),
    ]
    weights = {"indicators": 0.30, "harmonic": 0.30}
    result = system.compute(signals, weights)

    # 0.5×0.30 + 0.5×0.30 = 0.15 + 0.15 = 0.30 < 0.40
    assert result.direction == SignalDirection.HOLD


def test_scoring_conflicting():
    """Противоречащие сигналы — низкий score."""
    system = ScoringSystem(threshold=0.40)
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.7),
        make_signal("harmonic", SignalDirection.SELL, 0.7),
    ]
    weights = {"indicators": 0.30, "harmonic": 0.30}
    result = system.compute(signals, weights)

    assert result.confluence_score < 0.5


def test_scoring_single_analyzer_low_score():
    """Один анализатор даёт ограниченный score."""
    system = ScoringSystem(threshold=0.40)
    signals = [make_signal("indicators", SignalDirection.SELL, 1.0)]
    weights = {"indicators": 0.36}  # BEAR режим
    result = system.compute(signals, weights)

    # Score = 1.0 × 0.36 = 0.36 < 0.40 → HOLD
    assert result.direction == SignalDirection.HOLD
    assert result.confluence_score <= 0.36 + 1e-6


# ============ ConfluenceGate ============

def test_gate_aggressive_mode():
    """Aggressive режим пропускает сигналы от 0.50."""
    gate = ConfluenceGate(mode=GateMode.AGGRESSIVE)
    assert gate.get_threshold() == 0.50


def test_gate_conservative_mode():
    """Conservative режим требует 0.75."""
    gate = ConfluenceGate(mode=GateMode.CONSERVATIVE)
    assert gate.get_threshold() == 0.75


def test_gate_blocks_hold():
    """HOLD всегда блокируется."""
    from src.confluence.scoring import ScoringResult
    gate = ConfluenceGate()
    result = ScoringResult(
        direction=SignalDirection.HOLD,
        bull_score=0.0,
        bear_score=0.0,
        confluence_score=0.0,
    )
    assert not gate.is_open(result)


def test_gate_blocks_low_score():
    """Низкий score блокируется."""
    from src.confluence.scoring import ScoringResult
    gate = ConfluenceGate(mode=GateMode.BALANCED)
    result = ScoringResult(
        direction=SignalDirection.BUY,
        bull_score=0.5,
        bear_score=0.0,
        confluence_score=0.5,
    )
    assert not gate.is_open(result)


def test_gate_opens_high_score():
    """Высокий score пропускается."""
    from src.confluence.scoring import ScoringResult
    gate = ConfluenceGate(mode=GateMode.BALANCED)
    result = ScoringResult(
        direction=SignalDirection.BUY,
        bull_score=0.8,
        bear_score=0.0,
        confluence_score=0.8,
    )
    assert gate.is_open(result)


# ============ ConfluenceEngine ============

def test_engine_full_bull_flow():
    """
    Полный поток: BULL режим, три согласованных BUY.

    В BULL режиме веса:
    - indicators: 0.30 × 1.2 = 0.36
    - harmonic: 0.30 × 0.8 = 0.24
    - support_resistance: 0.40 × 1.0 = 0.40

    При confidence 0.8 / 0.8 / 0.8:
    score = 0.8×0.36 + 0.8×0.24 + 0.8×0.40 = 0.288 + 0.192 + 0.32 = 0.80
    """
    engine = ConfluenceEngine()
    data = make_bull_data()
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.8),
        make_signal("harmonic", SignalDirection.BUY, 0.8),
        make_signal("support_resistance", SignalDirection.BUY, 0.8),
    ]
    decision = engine.decide(signals, data)

    assert decision.regime == MarketRegime.BULL
    assert decision.direction == SignalDirection.BUY
    assert decision.passed
    assert decision.confluence_score >= 0.65


def test_engine_two_analyzers_strong_signals_pass():
    """
    Два анализатора с сильными сигналами могут пройти Gate.
    """
    engine = ConfluenceEngine()
    data = make_bear_data()
    signals = [
        make_signal("indicators", SignalDirection.SELL, 0.8),
        make_signal("support_resistance", SignalDirection.SELL, 0.95),
    ]
    decision = engine.decide(signals, data)

    # Score = 0.8×0.36 + 0.95×0.40 = 0.288 + 0.38 = 0.668 > 0.65
    assert decision.passed
    assert decision.direction == SignalDirection.SELL


def test_engine_single_analyzer_blocked():
    """
    Один анализатор НЕ может открыть сделку.
    """
    engine = ConfluenceEngine()
    data = make_bull_data()
    signals = [
        make_signal("indicators", SignalDirection.BUY, 1.0),
    ]
    decision = engine.decide(signals, data)

    # Score = 1.0 × 0.36 = 0.36 < 0.40 (внутренний threshold) → HOLD
    assert not decision.passed
    assert decision.direction == SignalDirection.HOLD


def test_engine_no_signals():
    """Нет сигналов — HOLD."""
    engine = ConfluenceEngine()
    decision = engine.decide([], make_bull_data())
    assert decision.direction == SignalDirection.HOLD
    assert not decision.passed


def test_engine_conflict_blocked():
    """Конфликтующие сигналы не проходят Gate."""
    engine = ConfluenceEngine()
    data = make_chop_data()
    signals = [
        make_signal("indicators", SignalDirection.BUY, 0.6),
        make_signal("harmonic", SignalDirection.SELL, 0.6),
    ]
    decision = engine.decide(signals, data)

    assert not decision.passed
    assert decision.direction == SignalDirection.HOLD