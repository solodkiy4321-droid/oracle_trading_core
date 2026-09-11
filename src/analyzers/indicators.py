"""Анализатор индикаторов с расширенными градациями confidence.

УЛУЧШЕНИЯ:
- RSI пороги 30/70 → 40/60 (больше сигналов)
- Bollinger Bands — близость к границе (не только касание)
- Больше свечных паттернов (Doji, Shooting Star, etc.)
- Градации confidence — учитываем частичные сигналы
"""

import logging
from typing import Optional, Dict

import pandas as pd
import pandas_ta_classic as ta

from src.analyzers.base import BaseAnalyzer
from src.analyzers.trend_filter import TrendFilter, TrendDirection
from src.models import AnalyzerSignal, SignalDirection

logger = logging.getLogger(__name__)


class IndicatorsAnalyzer(BaseAnalyzer):
    """
    Анализатор индикаторов с расширенными градациями confidence.
    """

    DEFAULT_WEIGHTS = {
        "rsi": 1.0,
        "macd": 1.5,
        "ema_alignment": 2.0,
        "bollinger": 1.0,
        "candle_pattern": 1.0,
    }

    def __init__(
        self,
        name: str = "indicators",
        weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.3,
        use_trend_filter: bool = True,
        trend_penalty: float = 0.2,
        agreement_bonus_step: float = 0.15,
        max_confidence: float = 0.95,
    ):
        super().__init__(name)
        self._weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self._min_confidence = min_confidence
        self._use_trend_filter = use_trend_filter
        self._trend_filter = TrendFilter() if use_trend_filter else None
        self._trend_penalty = trend_penalty
        self._agreement_bonus_step = agreement_bonus_step
        self._max_confidence = max_confidence

    async def analyze(self, data: pd.DataFrame) -> Optional[AnalyzerSignal]:
        if data is None or len(data) < 50:
            return None

        rsi = ta.rsi(data["close"], length=14)
        macd = ta.macd(data["close"])
        ema9 = ta.ema(data["close"], length=9)
        sma20 = ta.sma(data["close"], length=20)
        sma50 = ta.sma(data["close"], length=50)
        bb = ta.bbands(data["close"], length=20)

        if any(x is None for x in [rsi, macd, ema9, sma20, sma50, bb]):
            return None

        rsi_val = float(rsi.iloc[-1])
        macd_hist = float(macd["MACDh_12_26_9"].iloc[-1])
        ema9_val = float(ema9.iloc[-1])
        sma20_val = float(sma20.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        bb_upper = float(bb["BBU_20_2.0"].iloc[-1])
        bb_middle = float(bb["BBM_20_2.0"].iloc[-1])
        bb_lower = float(bb["BBL_20_2.0"].iloc[-1])
        current_price = float(data["close"].iloc[-1])

        bullish_score = 0.0
        bearish_score = 0.0
        total_weight = 0.0
        reasons = []
        bullish_count = 0
        bearish_count = 0

        # RSI — ОСЛАБЛЕННЫЕ пороги 40/60
        w = self._weights["rsi"]
        total_weight += w
        if rsi_val < 30:
            bullish_score += w * 1.0
            bullish_count += 1
            reasons.append(f"RSI={rsi_val:.1f} (strong oversold)")
        elif rsi_val < 40:
            bullish_score += w * 0.5
            bullish_count += 1
            reasons.append(f"RSI={rsi_val:.1f} (mild oversold)")
        elif rsi_val > 70:
            bearish_score += w * 1.0
            bearish_count += 1
            reasons.append(f"RSI={rsi_val:.1f} (strong overbought)")
        elif rsi_val > 60:
            bearish_score += w * 0.5
            bearish_count += 1
            reasons.append(f"RSI={rsi_val:.1f} (mild overbought)")

        # MACD
        w = self._weights["macd"]
        total_weight += w
        if macd_hist > 0:
            bullish_score += w
            bullish_count += 1
            reasons.append(f"MACD hist={macd_hist:.4f} (bullish)")
        elif macd_hist < 0:
            bearish_score += w
            bearish_count += 1
            reasons.append(f"MACD hist={macd_hist:.4f} (bearish)")

        # EMA alignment
        w = self._weights["ema_alignment"]
        total_weight += w
        if ema9_val > sma20_val > sma50_val:
            bullish_score += w
            bullish_count += 1
            reasons.append("EMA alignment (bullish)")
        elif ema9_val < sma20_val < sma50_val:
            bearish_score += w
            bearish_count += 1
            reasons.append("EMA alignment (bearish)")
        elif ema9_val > sma20_val:
            bullish_score += w * 0.4
            bullish_count += 1
            reasons.append("EMA partial bullish")
        elif ema9_val < sma20_val:
            bearish_score += w * 0.4
            bearish_count += 1
            reasons.append("EMA partial bearish")

        # Bollinger Bands — ГРАДАЦИИ
        w = self._weights["bollinger"]
        total_weight += w
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            distance_to_lower = (current_price - bb_lower) / bb_range
            distance_to_upper = (bb_upper - current_price) / bb_range

            if current_price <= bb_lower:
                bullish_score += w
                bullish_count += 1
                reasons.append("BB lower touch")
            elif distance_to_lower < 0.2:
                bullish_score += w * 0.5
                bullish_count += 1
                reasons.append("BB near lower")
            elif current_price >= bb_upper:
                bearish_score += w
                bearish_count += 1
                reasons.append("BB upper touch")
            elif distance_to_upper < 0.2:
                bearish_score += w * 0.5
                bearish_count += 1
                reasons.append("BB near upper")

        # Свечные паттерны — БОЛЬШЕ ПАТТЕРНОВ
        w = self._weights["candle_pattern"]
        total_weight += w
        open_ = data["open"].values
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        hammer = ta.cdl_pattern(name="hammer", open_=open_, high=high, low=low, close=close)
        engulfing = ta.cdl_pattern(name="engulfing", open_=open_, high=high, low=low, close=close)
        doji = ta.cdl_pattern(name="doji", open_=open_, high=high, low=low, close=close)
        shooting_star = ta.cdl_pattern(name="shootingstar", open_=open_, high=high, low=low, close=close)

        if hammer is not None and len(hammer) > 0 and float(hammer.iloc[-1]) > 0:
            bullish_score += w
            bullish_count += 1
            reasons.append("Hammer")
        if engulfing is not None and len(engulfing) > 0:
            eng_val = float(engulfing.iloc[-1])
            if eng_val > 0:
                bullish_score += w
                bullish_count += 1
                reasons.append("Bullish engulfing")
            elif eng_val < 0:
                bearish_score += w
                bearish_count += 1
                reasons.append("Bearish engulfing")
        if shooting_star is not None and len(shooting_star) > 0 and float(shooting_star.iloc[-1]) < 0:
            bearish_score += w
            bearish_count += 1
            reasons.append("Shooting star")
        if doji is not None and len(doji) > 0 and float(doji.iloc[-1]) != 0:
            reasons.append("Doji (neutral)")

        if total_weight == 0:
            return None

        bullish_norm = bullish_score / total_weight
        bearish_norm = bearish_score / total_weight

        if bullish_norm > bearish_norm and bullish_norm > 0:
            direction = SignalDirection.BUY
            base_confidence = min(bullish_norm, 1.0)
            agreement_count = bullish_count
        elif bearish_norm > bullish_norm and bearish_norm > 0:
            direction = SignalDirection.SELL
            base_confidence = min(bearish_norm, 1.0)
            agreement_count = bearish_count
        else:
            return None

        if agreement_count > 1:
            bonus = 1.0 + self._agreement_bonus_step * (agreement_count - 1)
        else:
            bonus = 1.0

        confidence = min(base_confidence * bonus, self._max_confidence)

        trend_info = "no_filter"
        if self._use_trend_filter and self._trend_filter is not None:
            trend = self._trend_filter.detect(data)
            multiplier = self._trend_filter.apply_penalty(direction, trend, self._trend_penalty)
            confidence *= multiplier
            trend_info = trend.name

        if confidence < self._min_confidence:
            return None

        return AnalyzerSignal(
            direction=direction,
            confidence=confidence,
            reason=" | ".join(reasons) if reasons else "no signal",
            metadata={
                "rsi": rsi_val,
                "macd_hist": macd_hist,
                "ema9": ema9_val,
                "sma20": sma20_val,
                "sma50": sma50_val,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "current_price": current_price,
                "bullish_score": bullish_norm,
                "bearish_score": bearish_norm,
                "base_confidence": base_confidence,
                "agreement_count": agreement_count,
                "agreement_bonus": bonus,
                "trend": trend_info,
                "weights": dict(self._weights),
            },
            source=self.name,
        )