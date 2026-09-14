"""VolatilityAnalyzer — волатильность через ATR и squeeze/expansion.

Категория: ВОЛАТИЛЬНОСТЬ.

Использует IndicatorCache если передан — иначе считает сам.
"""

import logging
from typing import Optional, Any

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


class VolatilityAnalyzer(BaseAnalyzer):
    """Анализатор волатильности: ATR squeeze/expansion."""

    def __init__(
        self,
        name: str = "volatility",
        atr_period: int = 14,
        atr_ma_period: int = 50,
        squeeze_ratio: float = 0.7,
        expansion_ratio: float = 1.5,
        breakout_lookback: int = 20,
        min_confidence: float = 0.30,
        max_confidence: float = 0.85,
    ):
        super().__init__(name)
        self._atr_period = atr_period
        self._atr_ma_period = atr_ma_period
        self._squeeze_ratio = squeeze_ratio
        self._expansion_ratio = expansion_ratio
        self._breakout_lookback = breakout_lookback
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence

    def _squeeze_score(self, ratio: float) -> float:
        if ratio >= self._squeeze_ratio:
            return 0.0
        if ratio <= 0.3:
            return 1.0
        span = self._squeeze_ratio - 0.3
        if span <= 0:
            return 0.0
        return (self._squeeze_ratio - ratio) / span

    def _expansion_score(self, ratio: float) -> float:
        if ratio <= self._expansion_ratio:
            return 0.0
        if ratio >= 2.5:
            return 1.0
        span = 2.5 - self._expansion_ratio
        if span <= 0:
            return 0.0
        return (ratio - self._expansion_ratio) / span

    def _breakout_direction(
        self, data: pd.DataFrame, current_price: float,
    ) -> Optional[tuple]:
        lookback = self._breakout_lookback
        if len(data) < lookback + 1:
            return None

        prior = data.iloc[-(lookback + 1):-1]
        hi = float(prior["high"].max())
        lo = float(prior["low"].min())

        if hi <= 0 or lo <= 0 or hi <= lo:
            return None

        if current_price > hi:
            strength = (current_price - hi) / hi
            return SignalDirection.BUY, strength, hi, lo
        if current_price < lo:
            strength = (lo - current_price) / lo
            return SignalDirection.SELL, strength, hi, lo

        return None

    def _compute_fallback(self, data: pd.DataFrame) -> Optional[tuple]:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        atr = ta.atr(high, low, close, length=self._atr_period)
        if atr is None or len(atr) < self._atr_ma_period:
            return None

        atr_series = atr.dropna()
        if len(atr_series) < self._atr_ma_period:
            return None

        atr_val = float(atr_series.iloc[-1])
        atr_ma = float(atr_series.iloc[-self._atr_ma_period:].mean())
        return atr_val, atr_ma

    async def analyze(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
    ) -> Optional[AnalyzerSignal]:
        min_bars = self._atr_ma_period + self._atr_period + 5

        if indicators is not None:
            if indicators.n < min_bars:
                return None

            i = indicators.n - 1
            ind = indicators.slice(i)
            atr_val = ind.get("atr_14")
            atr_ma = ind.get("atr_14_ma_50")

            if atr_val is None or atr_ma is None:
                return None
            if np.isnan(atr_val) or np.isnan(atr_ma):
                return None

            current_price = ind.get("current_price")
            if current_price is None:
                return None
        else:
            if data is None or len(data) < min_bars:
                return None
            fallback = self._compute_fallback(data)
            if fallback is None:
                return None
            atr_val, atr_ma = fallback
            current_price = float(data["close"].iloc[-1])

        if atr_ma <= 0 or atr_val <= 0 or current_price <= 0:
            return None

        ratio = atr_val / atr_ma
        squeeze = self._squeeze_score(ratio)
        expansion = self._expansion_score(ratio)

        direction: Optional[SignalDirection] = None
        zone = ""
        base_conf = 0.0

        if squeeze > 0:
            breakout = self._breakout_direction(data, current_price)
            if breakout is None:
                return None
            bdir, bstrength, bhi, blo = breakout
            direction = bdir
            zone = "squeeze_breakout"
            base_conf = squeeze * min(1.0, 0.5 + bstrength * 50.0)
        elif expansion > 0:
            closes = data["close"]
            direction = (
                SignalDirection.BUY
                if closes.iloc[-1] >= closes.iloc[-2]
                else SignalDirection.SELL
            )
            zone = "expansion"
            base_conf = expansion * 0.5
        else:
            return None

        if direction is None:
            return None

        confidence = 0.3 + 0.5 * base_conf
        confidence = max(
            self._min_confidence,
            min(confidence, self._max_confidence),
        )

        if confidence < self._min_confidence:
            return None

        reason = (
            f"VOLATILITY {zone}: "
            f"ATR={atr_val:.4f}, "
            f"ATR_MA={atr_ma:.4f}, "
            f"ratio={ratio:.2f}, "
            f"conf={confidence:.2f}"
        )

        return AnalyzerSignal(
            direction=direction,
            confidence=confidence,
            reason=reason,
            metadata={
                "atr": atr_val,
                "atr_ma": atr_ma,
                "atr_ratio": ratio,
                "squeeze_score": squeeze,
                "expansion_score": expansion,
                "zone": zone,
                "atr_period": self._atr_period,
                "atr_ma_period": self._atr_ma_period,
                "base_confidence": base_conf,
            },
            source=self.name,
        )