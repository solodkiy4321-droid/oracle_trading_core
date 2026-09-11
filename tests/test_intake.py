"""Тесты модуля приёма сигналов."""

import asyncio
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, SignalDirection
from src.signal_intake.intake import SignalIntake


# ============ Mock-анализаторы ============

class MockBuyAnalyzer(BaseAnalyzer):
    """Mock-анализатор, всегда возвращающий BUY."""
    def __init__(self):
        super().__init__("mock_buy")

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        return AnalyzerSignal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            reason="mock buy",
            source=self.name,
        )


class MockSellAnalyzer(BaseAnalyzer):
    """Mock-анализатор, всегда возвращающий SELL."""
    def __init__(self):
        super().__init__("mock_sell")

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        return AnalyzerSignal(
            direction=SignalDirection.SELL,
            confidence=0.7,
            reason="mock sell",
            source=self.name,
        )


class MockCrashAnalyzer(BaseAnalyzer):
    """Mock-анализатор, который падает с исключением."""
    def __init__(self):
        super().__init__("mock_crash")

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        raise RuntimeError("Искусственное падение")


class MockTimeoutAnalyzer(BaseAnalyzer):
    """Mock-анализатор, который зависает."""
    def __init__(self):
        super().__init__("mock_timeout")

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        await asyncio.sleep(10)  # дольше таймаута
        return None


class MockNoneAnalyzer(BaseAnalyzer):
    """Mock-анализатор, возвращающий None (нет сигнала)."""
    def __init__(self):
        super().__init__("mock_none")

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        return None


# ============ Фикстуры ============

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Синтетические OHLCV-данные."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# ============ Тесты ============

@pytest.mark.asyncio
async def test_single_analyzer(sample_data):
    """Один анализатор — один сигнал."""
    intake = SignalIntake()
    intake.register(MockBuyAnalyzer())

    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    assert batch.count == 1
    assert batch.signals[0].direction == SignalDirection.BUY
    assert batch.signals[0].confidence == 0.8
    assert batch.symbol == "BTCUSDT"
    assert batch.timeframe == "1h"


@pytest.mark.asyncio
async def test_multiple_analyzers(sample_data):
    """Несколько анализаторов — несколько сигналов."""
    intake = SignalIntake()
    intake.register(MockBuyAnalyzer())
    intake.register(MockSellAnalyzer())

    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    assert batch.count == 2
    directions = {s.direction for s in batch.signals}
    assert SignalDirection.BUY in directions
    assert SignalDirection.SELL in directions


@pytest.mark.asyncio
async def test_crash_isolation(sample_data):
    """Падение одного анализатора не ломает сбор."""
    intake = SignalIntake()
    intake.register(MockBuyAnalyzer())
    intake.register(MockCrashAnalyzer())

    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    # Buy-сигнал должен пройти
    assert batch.count == 1
    assert batch.signals[0].direction == SignalDirection.BUY

    # Crash должен быть отвергнут
    assert len(batch.rejected) == 1
    assert batch.rejected[0].source == "mock_crash"
    assert "Искусственное падение" in batch.rejected[0].reason


@pytest.mark.asyncio
async def test_timeout_isolation(sample_data):
    """Зависший анализатор не блокирует остальные."""
    intake = SignalIntake(timeout=0.5)
    intake.register(MockBuyAnalyzer())
    intake.register(MockTimeoutAnalyzer())

    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    # Buy-сигнал должен пройти
    assert batch.count == 1
    assert batch.signals[0].direction == SignalDirection.BUY

    # Timeout должен быть отвергнут
    assert len(batch.rejected) == 1
    assert batch.rejected[0].source == "mock_timeout"
    assert "Таймаут" in batch.rejected[0].reason


@pytest.mark.asyncio
async def test_none_analyzer_ignored(sample_data):
    """Анализатор, вернувший None, не считается отвергнутым."""
    intake = SignalIntake()
    intake.register(MockBuyAnalyzer())
    intake.register(MockNoneAnalyzer())

    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    # Только один валидный сигнал
    assert batch.count == 1
    # None-анализатор не попал ни в валидные, ни в отвергнутые
    assert len(batch.rejected) == 0


@pytest.mark.asyncio
async def test_empty_registry(sample_data):
    """Без анализаторов батч пустой."""
    intake = SignalIntake()
    batch = await intake.process(sample_data, "BTCUSDT", "1h")

    assert batch.is_empty
    assert batch.count == 0
    assert batch.stats["total"] == 0


@pytest.mark.asyncio
async def test_metadata_enrichment(sample_data):
    """Метаданные обогащаются symbol и timeframe."""
    intake = SignalIntake()
    intake.register(MockBuyAnalyzer())

    batch = await intake.process(sample_data, "ETHUSDT", "4h")

    assert batch.count == 1
    meta = batch.signals[0].metadata
    assert meta["symbol"] == "ETHUSDT"
    assert meta["timeframe"] == "4h"