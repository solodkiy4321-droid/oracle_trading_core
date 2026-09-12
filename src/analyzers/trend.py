"""TrendAnalyzer — тренд через EMA200 + ADX.

Категория: ТРЕНД.
Вопрос: «Куда идёт рынок и насколько это сильный тренд?»

Принципы:
- EMA200 — направление (цена выше/ниже)
- ADX — сила тренда (порог 20-25)
- Если ADX < порог — сигнала нет (боковик, не наша категория)

Не дублирует:
- momentum (RSI) — тот работает в CHOP
- volatility (ATR) — тот про размах, не про направление
- volume (OBV) — тот про подтверждение объёмом

Confidence — не «сумма всего», а сила тренда:
- ADX даёт базу (чем сильнее тренд, тем выше confidence)
- EMA200 подтверждает направление
- Расстояние цены от EMA200 даёт бонус (но с потолком — не перегреваться)
"""

import logging
from typing import Optional

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

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < self._ema_period + 10:
            return None

        try:
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

        except Exception as e:
            logger.exception("TrendAnalyzer error: %s", e)
            return None