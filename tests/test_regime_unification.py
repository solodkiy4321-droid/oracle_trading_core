"""Тесты для Бага F: единый источник правды для рыночного режима.

Проверяют:
- RegimeDetector.detect_from_cache даёт тот же результат, что detect.
- TradingEngine._detect_regime и RegimeDetector согласованы.
- ta.sma/ta.adx не вызываются при переданном IndicatorCache.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.confluence.regime_detector import RegimeDetector, MarketRegime
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine


# ---------- Синтетические данные ----------

def _make_bull(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = np.linspace(100, 200, n) + rng.normal(0, 0.5, n)
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({
        "open": close - 0.5, "high": high, "low": low,
        "close": close, "volume": rng.integers(1000, 10000, n),
    })


def _make_bear(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    close = np.linspace(200, 100, n) + rng.normal(0, 0.5, n)
    high = close + 1.0
    low = close - 1.0
    return pd.DataFrame({
        "open": close + 0.5, "high": high, "low": low,
        "close": close, "volume": rng.integers(1000, 10000, n),
    })


def _make_chop(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(44)
    close = 100 + rng.normal(0, 0.5, n)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({
        "open": close - 0.2, "high": high, "low": low,
        "close": close, "volume": rng.integers(1000, 10000, n),
    })


# ---------- RegimeDetector: detect vs detect_from_cache ----------

def test_regime_bull_detect():
    assert RegimeDetector().detect(_make_bull()) == MarketRegime.BULL


def test_regime_bear_detect():
    assert RegimeDetector().detect(_make_bear()) == MarketRegime.BEAR


def test_regime_chop_detect():
    assert RegimeDetector().detect(_make_chop()) == MarketRegime.CHOP


def test_regime_detect_from_cache_matches_detect_bull():
    """detect_from_cache даёт тот же режим, что detect, на тех же значениях."""
    import pandas_ta_classic as ta
    data = _make_bull()
    detector = RegimeDetector()
    slow = detector.detect(data)

    sma = ta.sma(data["close"], length=200)
    adx_df = ta.adx(data["high"], data["low"], data["close"], length=14)
    sma_val = float(sma.iloc[-1])
    adx_val = float(adx_df[[c for c in adx_df.columns if c.startswith("ADX_")][0]].iloc[-1])
    price = float(data["close"].iloc[-1])

    fast = detector.detect_from_cache(sma_val, adx_val, price)
    assert slow == fast


def test_regime_detect_from_cache_matches_detect_bear():
    import pandas_ta_classic as ta
    data = _make_bear()
    detector = RegimeDetector()
    slow = detector.detect(data)

    sma = ta.sma(data["close"], length=200)
    adx_df = ta.adx(data["high"], data["low"], data["close"], length=14)
    sma_val = float(sma.iloc[-1])
    adx_val = float(adx_df[[c for c in adx_df.columns if c.startswith("ADX_")][0]].iloc[-1])
    price = float(data["close"].iloc[-1])

    fast = detector.detect_from_cache(sma_val, adx_val, price)
    assert slow == fast


def test_regime_detect_from_cache_matches_detect_chop():
    import pandas_ta_classic as ta
    data = _make_chop()
    detector = RegimeDetector()
    slow = detector.detect(data)

    sma = ta.sma(data["close"], length=200)
    adx_df = ta.adx(data["high"], data["low"], data["close"], length=14)
    sma_val = float(sma.iloc[-1])
    adx_val = float(adx_df[[c for c in adx_df.columns if c.startswith("ADX_")][0]].iloc[-1])
    price = float(data["close"].iloc[-1])

    fast = detector.detect_from_cache(sma_val, adx_val, price)
    assert slow == fast


def test_regime_detect_from_cache_none_returns_chop():
    d = RegimeDetector()
    assert d.detect_from_cache(None, 25.0, 100.0) == MarketRegime.CHOP
    assert d.detect_from_cache(100.0, None, 100.0) == MarketRegime.CHOP
    assert d.detect_from_cache(100.0, 25.0, None) == MarketRegime.CHOP


def test_regime_detect_from_cache_zero_sma_returns_chop():
    d = RegimeDetector()
    assert d.detect_from_cache(0.0, 25.0, 100.0) == MarketRegime.CHOP


def test_regime_insufficient_data_returns_chop():
    d = RegimeDetector()
    assert d.detect(_make_bull(n=50)) == MarketRegime.CHOP


# ---------- TradingEngine: _detect_regime не пересчитывает ta.sma при кэше ----------

@pytest.mark.asyncio
async def test_engine_detect_regime_does_not_call_ta_sma_with_cache():
    """
    При наличии IndicatorCache ta.sma НЕ вызывается в _detect_regime.
    """
    from src.analyzers.indicator_cache import IndicatorCache

    data = _make_bull(n=500)
    cache = IndicatorCache(data)

    config = EngineConfig(
        symbol="BTC-USD",
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config, full_data=data)

    real_sma = __import__("pandas_ta_classic").sma
    call_count = {"n": 0}

    def counting_sma(*args, **kwargs):
        call_count["n"] += 1
        return real_sma(*args, **kwargs)

    # Патчим ta.sma в модуле regime_detector — _detect_regime туда не должен ходить
    with patch(
        "src.confluence.regime_detector.ta.sma",
        side_effect=counting_sma,
    ):
        engine._detect_regime(data, bar_index=499)

    assert call_count["n"] == 0, (
        f"ta.sma не должна вызываться при IndicatorCache, "
        f"но вызвана {call_count['n']} раз"
    )

    engine.close()


@pytest.mark.asyncio
async def test_engine_detect_regime_falls_back_without_cache():
    """
    Без IndicatorCache _detect_regime использует RegimeDetector.detect,
    который вызывает ta.sma.
    """
    data = _make_bull(n=500)

    config = EngineConfig(
        symbol="BTC-USD",
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config)  # без full_data → нет кэша

    real_sma = __import__("pandas_ta_classic").sma
    call_count = {"n": 0}

    def counting_sma(*args, **kwargs):
        call_count["n"] += 1
        return real_sma(*args, **kwargs)

    with patch(
        "src.confluence.regime_detector.ta.sma",
        side_effect=counting_sma,
    ):
        engine._detect_regime(data, bar_index=499)

    assert call_count["n"] >= 1, (
        "Без кэша fallback должен вызвать ta.sma"
    )

    engine.close()


@pytest.mark.asyncio
async def test_engine_detect_regime_consistent_with_detector():
    """
    TradingEngine._detect_regime с кэшем даёт тот же режим, что RegimeDetector.
    """
    from src.analyzers.indicator_cache import IndicatorCache

    data = _make_bull(n=500)
    cache = IndicatorCache(data)

    config = EngineConfig(
        symbol="BTC-USD",
        journal_db_path=":memory:",
    )
    engine = TradingEngine(config, full_data=data)

    engine_regime = engine._detect_regime(data, bar_index=499)

    # То же, что RegimeDetector.detect на срезе тех же данных
    expected = RegimeDetector().detect(data)

    assert engine_regime == expected, (
        f"engine вернул {engine_regime}, а RegimeDetector — {expected}"
    )

    engine.close()