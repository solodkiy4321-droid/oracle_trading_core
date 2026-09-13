"""Mean-reversion анализатор для пилообразных рынков.

Категория: MEAN-REVERSION (возврат к среднему).
Вопрос: «Не перегрето ли движение и не пора ли вернуться к среднему?»

Принципы:
- Работает ТОЛЬКО в пилообразных/боковых рынках.
- Использует Bollinger Bands + RSI + distance от SMA.
- BUY когда цена сильно ниже средней + RSI oversold.
- SELL когда цена сильно выше средней + RSI overbought.

Не дублирует:
- trend (EMA200 + ADX) — работает в тренде
- volatility (ATR squeeze) — работает на пробоях
- volume (OBV) — работает с объёмом

Confidence — сила перегрева:
- Чем дальше цена от SMA и чем сильнее RSI — тем выше confidence.
- Чем дольше в боковике — тем выше confidence.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


class MeanReversionAnalyzer(BaseAnalyzer):
    """Mean-reversion анализатор для пилообразных рынков."""

    def __init__(
        self,
        name: str = "mean_reversion",
        rsi_period: int = 14,
        sma_period: int = 20,
        atr_period: int = 14,
        adx_period: int = 14,
        adx_max: float = 20.0,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        min_distance_atr: float = 1.5,
        max_distance_atr: float = 5.0,
        min_confidence: float = 0.30,
        max_confidence: float = 0.85,
    ):
        super().__init__(name)
        self._rsi_period = rsi_period
        self._sma_period = sma_period
        self._atr_period = atr_period
        self._adx_period = adx_period
        self._adx_max = adx_max
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._min_distance_atr = min_distance_atr
        self._max_distance_atr = max_distance_atr
        self._min_confidence = min_confidence
        self._max_confidence = max_confidence

    def _is_choppy(self, data: pd.DataFrame) -> tuple:
        """
        Определяет, является ли рынок пилообразным.

        Критерии:
        - ADX < adx_max (нет тренда)
        - Цена пересекает SMA несколько раз за период

        Returns:
            (is_choppy, adx_value, sma_crosses)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        if len(close) < max(self._adx_period, self._sma_period, 50):
            return False, 0.0, 0

        adx_df = ta.adx(high, low, close, length=self._adx_period)
        if adx_df is None or len(adx_df) == 0:
            return False, 0.0, 0

        adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
        if not adx_cols:
            return False, 0.0, 0

        adx_val = float(adx_df[adx_cols[0]].iloc[-1])
        if np.isnan(adx_val):
            return False, 0.0, 0

        if adx_val >= self._adx_max:
            return False, adx_val, 0

        sma = ta.sma(close, length=self._sma_period)
        if sma is None or len(sma) < 50:
            return False, adx_val, 0

        lookback = min(50, len(sma))
        price_segment = close.iloc[-lookback:].values
        sma_segment = sma.iloc[-lookback:].values

        crosses = 0
        prev_side = None
        for p, s in zip(price_segment, sma_segment):
            if np.isnan(s):
                continue
            side = 1 if p > s else -1
            if prev_side is not None and side != prev_side:
                crosses += 1
            prev_side = side

        return True, adx_val, crosses

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        try:
            is_choppy, adx_val, crosses = self._is_choppy(data)
            if not is_choppy:
                return None

            if crosses < 4:
                return None

            close = data["close"]
            high = data["high"]
            low = data["low"]

            current_price = float(close.iloc[-1])

            sma = ta.sma(close, length=self._sma_period)
            rsi = ta.rsi(close, length=self._rsi_period)
            atr = ta.atr(high, low, close, length=self._atr_period)

            if sma is None or rsi is None or atr is None:
                return None
            if len(sma) == 0 or len(rsi) == 0 or len(atr) == 0:
                return None

            sma_val = float(sma.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            atr_val = float(atr.iloc[-1])

            if sma_val <= 0 or atr_val <= 0 or np.isnan(rsi_val):
                return None

            distance = current_price - sma_val
            distance_atr = abs(distance) / atr_val

            if distance_atr < self._min_distance_atr:
                return None
            if distance_atr > self._max_distance_atr:
                return None

            direction = None
            zone = ""

            if distance < 0 and rsi_val < self._rsi_oversold:
                direction = SignalDirection.BUY
                zone = "oversold"
            elif distance > 0 and rsi_val > self._rsi_overbought:
                direction = SignalDirection.SELL
                zone = "overbought"
            else:
                return None

            distance_score = min(
                1.0, (distance_atr - self._min_distance_atr)
                / (self._max_distance_atr - self._min_distance_atr),
            )

            if zone == "oversold":
                rsi_score = max(
                    0.0, (self._rsi_oversold - rsi_val) / self._rsi_oversold,
                )
            else:
                rsi_score = max(
                    0.0, (rsi_val - self._rsi_overbought)
                    / (100.0 - self._rsi_overbought),
                )

            chop_score = min(1.0, crosses / 10.0)

            base_conf = (
                0.4 * distance_score
                + 0.4 * rsi_score
                + 0.2 * chop_score
            )

            confidence = 0.3 + 0.55 * base_conf
            confidence = max(
                self._min_confidence,
                min(confidence, self._max_confidence),
            )

            if confidence < self._min_confidence:
                return None

            reason = (
                f"MEAN-REVERSION {zone}: "
                f"price={current_price:.4f}, "
                f"SMA={sma_val:.4f}, "
                f"dist={distance_atr:.2f}ATR, "
                f"RSI={rsi_val:.1f}, "
                f"ADX={adx_val:.1f}, "
                f"crosses={crosses}, "
                f"conf={confidence:.2f}"
            )

            return AnalyzerSignal(
                direction=direction,
                confidence=confidence,
                reason=reason,
                metadata={
                    "zone": zone,
                    "sma": sma_val,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "atr": atr_val,
                    "distance_atr": distance_atr,
                    "crosses": crosses,
                    "distance_score": distance_score,
                    "rsi_score": rsi_score,
                    "chop_score": chop_score,
                    "base_confidence": base_conf,
                },
                source=self.name,
            )

        except Exception as e:
            logger.exception("MeanReversionAnalyzer error: %s", e)
            return None