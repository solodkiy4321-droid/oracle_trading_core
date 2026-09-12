"""VolatilityAnalyzer — волатильность через ATR и squeeze/expansion.

Категория: ВОЛАТИЛЬНОСТЬ.
Вопрос: «Насколько широко ходит цена и не сжалась ли она перед пробоем?»

Принципы:
- ATR — текущий размах.
- Сравниваем ATR с его скользящей средней (SMA по ATR).
- Squeeze: ATR < 0.7 × среднего ATR → сжатие, скоро пробой.
- Expansion: ATR > 1.5 × среднего ATR → расширение, тренд или паника.

Сигнал:
- Squeeze сам по себе не даёт направления. Нужен пробой границы.
- Определяем направление по последнему close относительно диапазона
  за N баров: close > верхней границы → BUY, close < нижней → SELL.
- Expansion: если ATR растёт и цена идёт в одну сторону — подтверждение.

Не дублирует:
- trend (EMA200 + ADX) — тот про направление и силу
- momentum (RSI) — тот про перегрев в боковике
- volume (OBV) — тот про подтверждение объёмом

Confidence — сила сжатия/расширения + чёткость пробоя.
"""

import logging
from typing import Optional

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
        self, data: pd.DataFrame, current_price: float
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

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < self._atr_ma_period + self._atr_period + 5:
            return None

        try:
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

            if atr_ma <= 0 or atr_val <= 0:
                return None

            ratio = atr_val / atr_ma
            current_price = float(close.iloc[-1])

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
                base_conf = squeeze * (1.0 - 0.5 * min(bstrength * 50.0, 1.0) * -1.0)
                base_conf = squeeze * min(1.0, 0.5 + bstrength * 50.0)
            elif expansion > 0:
                direction = SignalDirection.BUY if close.iloc[-1] >= close.iloc[-2] else SignalDirection.SELL
                zone = "expansion"
                base_conf = expansion * 0.5
            else:
                return None

            if direction is None:
                return None

            confidence = 0.3 + 0.5 * base_conf
            confidence = max(self._min_confidence, min(confidence, self._max_confidence))

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

        except Exception as e:
            logger.exception("VolatilityAnalyzer error: %s", e)
            return None