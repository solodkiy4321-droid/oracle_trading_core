"""VolumeAnalyzer — подтверждение движения объёмом через OBV.

Категория: ОБЪЁМ.

Использует IndicatorCache если передан — иначе считает сам.
"""

import logging
from typing import Optional, Any, List

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


class VolumeAnalyzer(BaseAnalyzer):
    """Анализатор объёма: OBV + дивергенция."""

    def __init__(
        self,
        name: str = "volume",
        obv_slope_period: int = 20,
        divergence_lookback: int = 30,
        min_confidence: float = 0.30,
        max_confidence: float = 0.85,
    ):
        super().__init__(name)
        self._obv_slope_period = obv_slope_period
        self._divergence_lookback = divergence_lookback
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence

    def _detect_bearish_divergence(self, price: pd.Series, obv: pd.Series) -> float:
        lookback = self._divergence_lookback
        if len(price) < lookback or len(obv) < lookback:
            return 0.0

        p = price.iloc[-lookback:].values
        o = obv.iloc[-lookback:].values

        half = lookback // 2
        if half < 3:
            return 0.0

        p_recent_high = float(np.max(p[half:]))
        p_older_high = float(np.max(p[:half]))
        o_recent_high = float(np.max(o[half:]))
        o_older_high = float(np.max(o[:half]))

        if p_older_high <= 0 or o_older_high == 0:
            return 0.0

        price_made_high = p_recent_high > p_older_high
        obv_made_high = o_recent_high > o_older_high

        if not price_made_high:
            return 0.0
        if obv_made_high:
            return 0.0

        price_gain = (p_recent_high - p_older_high) / p_older_high
        obv_loss = abs(o_recent_high - o_older_high) / (abs(o_older_high) + 1e-9)

        score = min(1.0, 0.5 * price_gain * 100 + 0.5 * min(obv_loss, 1.0))
        return max(0.0, min(score, 1.0))

    def _detect_bullish_divergence(self, price: pd.Series, obv: pd.Series) -> float:
        lookback = self._divergence_lookback
        if len(price) < lookback or len(obv) < lookback:
            return 0.0

        p = price.iloc[-lookback:].values
        o = obv.iloc[-lookback:].values

        half = lookback // 2
        if half < 3:
            return 0.0

        p_recent_low = float(np.min(p[half:]))
        p_older_low = float(np.min(p[:half]))
        o_recent_low = float(np.min(o[half:]))
        o_older_low = float(np.min(o[:half]))

        if p_older_low <= 0:
            return 0.0

        price_made_low = p_recent_low < p_older_low
        obv_made_low = o_recent_low < o_older_low

        if not price_made_low:
            return 0.0
        if obv_made_low:
            return 0.0

        price_drop = (p_older_low - p_recent_low) / p_older_low
        obv_gain = abs(o_recent_low - o_older_low) / (abs(o_older_low) + 1e-9)

        score = min(1.0, 0.5 * price_drop * 100 + 0.5 * min(obv_gain, 1.0))
        return max(0.0, min(score, 1.0))

    def _obv_slope_score(self, obv: pd.Series) -> float:
        period = self._obv_slope_period
        if len(obv) < period:
            return 0.0

        segment = obv.iloc[-period:].values
        n = len(segment)
        if n < 3:
            return 0.0

        x = np.arange(n, dtype=float)
        y = segment.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        num = float(np.sum((x - x_mean) * (y - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))
        if den == 0:
            return 0.0
        slope = num / den

        scale = abs(y_mean) + 1e-9
        slope_norm = slope / scale

        return max(-1.0, min(1.0, slope_norm * 10.0))

    async def analyze(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
    ) -> Optional[AnalyzerSignal]:
        min_bars = self._divergence_lookback + 10

        if indicators is not None:
            if indicators.n < min_bars:
                return None

            obv_series = indicators.obv
            price_series = indicators.close_series

            if obv_series is None or len(obv_series) < min_bars:
                return None
        else:
            if data is None or len(data) < min_bars:
                return None
            if "volume" not in data.columns:
                return None

            close = data["close"]
            volume = data["volume"]

            if volume.abs().sum() == 0:
                return None

            obv_series = ta.obv(close, volume)
            if obv_series is None or len(obv_series) == 0:
                return None
            price_series = close

        bearish = self._detect_bearish_divergence(price_series, obv_series)
        bullish = self._detect_bullish_divergence(price_series, obv_series)
        slope = self._obv_slope_score(obv_series)

        direction: Optional[SignalDirection] = None
        zone = ""
        base_conf = 0.0

        if bearish >= bullish and bearish >= 0.3:
            direction = SignalDirection.SELL
            zone = "bearish_divergence"
            base_conf = bearish
        elif bullish > bearish and bullish >= 0.3:
            direction = SignalDirection.BUY
            zone = "bullish_divergence"
            base_conf = bullish
        elif slope > 0.3:
            direction = SignalDirection.BUY
            zone = "obv_uptrend"
            base_conf = min(slope, 1.0) * 0.6
        elif slope < -0.3:
            direction = SignalDirection.SELL
            zone = "obv_downtrend"
            base_conf = min(abs(slope), 1.0) * 0.6

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
            f"VOLUME {zone}: "
            f"bearish_div={bearish:.2f}, "
            f"bullish_div={bullish:.2f}, "
            f"obv_slope={slope:+.2f}, "
            f"conf={confidence:.2f}"
        )

        return AnalyzerSignal(
            direction=direction,
            confidence=confidence,
            reason=reason,
            metadata={
                "bearish_divergence": bearish,
                "bullish_divergence": bullish,
                "obv_slope": slope,
                "zone": zone,
                "divergence_lookback": self._divergence_lookback,
                "obv_slope_period": self._obv_slope_period,
                "base_confidence": base_conf,
            },
            source=self.name,
        )