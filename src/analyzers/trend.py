"""TrendAnalyzer — тренд через EMA200 + ADX.

Категория: ТРЕНД.

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


class TrendAnalyzer(BaseAnalyzer):
    """Анализатор тренда: EMA200 + ADX."""

    def __init__(
        self,
        name: str = "trend",
        ema_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        adx_strong: float = 35.0,
        neutral_band_pct: float = 0.01,
        max_distance_pct: float = 0.15,
        min_confidence: float = 0.30,
        max_confidence: float = 0.90,
    ):
        super().__init__(name)
        self._ema_period = ema_period
        self._adx_period = adx_period
        self._adx_threshold = adx_threshold
        self._adx_strong = adx_strong
        self._neutral_band_pct = neutral_band_pct
        self._max_distance_pct = max_distance_pct
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence

    def _compute_fallback(self, data: pd.DataFrame) -> Optional[tuple]:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ema = ta.ema(close, length=self._ema_period)
        if ema is None or len(ema) == 0:
            return None
        ema_val = float(ema.iloc[-1])

        adx_df = ta.adx(high, low, close, length=self._adx_period)
        if adx_df is None or len(adx_df) == 0:
            return None
        adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
        if not adx_cols:
            return None
        adx_val = float(adx_df[adx_cols[0]].iloc[-1])

        current_price = float(close.iloc[-1])
        return ema_val, adx_val, current_price

    async def analyze(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
        bar_index: Optional[int] = None,
    ) -> Optional[AnalyzerSignal]:
        min_bars = max(self._ema_period, self._adx_period) + 10

        if indicators is not None:
            i = bar_index if bar_index is not None else indicators.n - 1
            if i < min_bars or i >= indicators.n:
                return None

            ind = indicators.slice(i)

            ema_key = f"ema_{self._ema_period}"
            ema_val = ind.get(ema_key)
            adx_val = ind.get("adx")
            current_price = ind.get("current_price")

            if ema_val is None or adx_val is None or current_price is None:
                return None
            if np.isnan(ema_val) or np.isnan(adx_val):
                return None
        else:
            if data is None or len(data) < min_bars:
                return None
            fallback = self._compute_fallback(data)
            if fallback is None:
                return None
            ema_val, adx_val, current_price = fallback

        if ema_val <= 0 or current_price <= 0:
            return None

        distance_pct = (current_price - ema_val) / ema_val

        if abs(distance_pct) < self._neutral_band_pct:
            return None

        if adx_val < self._adx_threshold:
            return None

        if distance_pct > 0:
            direction = SignalDirection.BUY
            trend_name = "UP"
        else:
            direction = SignalDirection.SELL
            trend_name = "DOWN"

        adx_norm = min(adx_val / self._adx_strong, 1.0)
        base_conf = 0.3 + 0.5 * adx_norm

        dist_norm = min(abs(distance_pct) / self._max_distance_pct, 1.0)
        conf = base_conf * (1.0 + 0.3 * dist_norm)

        confidence = max(self._min_confidence, min(conf, self._max_confidence))

        if confidence < self._min_confidence:
            return None

        reason = (
            f"TREND {trend_name}: "
            f"price={current_price:.4f}, "
            f"EMA{self._ema_period}={ema_val:.4f}, "
            f"dist={distance_pct * 100:+.2f}%, "
            f"ADX={adx_val:.1f}, "
            f"conf={confidence:.2f}"
        )

        return AnalyzerSignal(
            direction=direction,
            confidence=confidence,
            reason=reason,
            metadata={
                "ema_period": self._ema_period,
                "ema_value": ema_val,
                "adx": adx_val,
                "adx_threshold": self._adx_threshold,
                "adx_strong": self._adx_strong,
                "distance_pct": distance_pct,
                "trend": trend_name,
                "base_confidence": base_conf,
            },
            source=self.name,
        )