"""Детектор рыночного режима — единый источник правды.

Определяет BULL / BEAR / CHOP по трём признакам:
- цена относительно SMA200,
- ADX (сила тренда),
- нейтральная зона вокруг SMA.

Два публичных пути:
- detect(data) — пересчитывает SMA200 и ADX (fallback).
- detect_from_cache(sma_val, adx_val, current_price) — использует
  готовые значения из IndicatorCache (быстрый путь).

Оба пути вызывают один и тот же _classify, поэтому результат
детерминирован и не зависит от того, откуда взялись индикаторы.
"""

import logging
from enum import Enum

import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Рыночный режим."""
    BULL = "BULL"      # Восходящий тренд
    BEAR = "BEAR"      # Нисходящий тренд
    CHOP = "CHOP"      # Боковик / неопределённость


class RegimeDetector:
    """
    Определяет рыночный режим.

    Логика:
    - BULL: цена > SMA200 (вне нейтральной зоны) и ADX >= threshold
    - BEAR: цена < SMA200 (вне нейтральной зоны) и ADX >= threshold
    - CHOP: всё остальное (боковик, слабый тренд, недостаток данных)
    """

    def __init__(
        self,
        sma_period: int = 200,
        atr_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        neutral_band_pct: float = 0.02,
    ):
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.neutral_band_pct = neutral_band_pct

    # ---------- Публичные интерфейсы ----------

    def detect(self, data: pd.DataFrame) -> MarketRegime:
        """
        Определяет режим по свежему расчёту SMA200 и ADX.

        Fallback, когда IndicatorCache недоступен.
        """
        if data is None or len(data) < self.sma_period:
            logger.debug(
                "Недостаточно данных для режима: %d баров (нужно %d)",
                len(data) if data is not None else 0,
                self.sma_period,
            )
            return MarketRegime.CHOP

        try:
            sma = ta.sma(data["close"], length=self.sma_period)
            if sma is None or len(sma) == 0:
                return MarketRegime.CHOP

            sma_val = float(sma.iloc[-1])
            current_price = float(data["close"].iloc[-1])

            adx_df = ta.adx(
                data["high"], data["low"], data["close"],
                length=self.adx_period,
            )
            adx_val = 0.0
            if adx_df is not None and len(adx_df) > 0:
                adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
                if adx_cols:
                    adx_val = float(adx_df[adx_cols[0]].iloc[-1])

            return self._classify(sma_val, adx_val, current_price)

        except Exception as e:
            logger.exception("Ошибка в RegimeDetector.detect: %s", e)
            return MarketRegime.CHOP

    def detect_from_cache(
        self,
        sma_val: float,
        adx_val: float,
        current_price: float,
    ) -> MarketRegime:
        """
        Определяет режим по уже посчитанным значениям IndicatorCache.

        Не вызывает ta.sma и ta.adx — это ключевое отличие от detect().
        """
        if sma_val is None or adx_val is None or current_price is None:
            return MarketRegime.CHOP
        if sma_val <= 0:
            return MarketRegime.CHOP
        return self._classify(sma_val, adx_val, current_price)

    # ---------- Общая логика ----------

    def _classify(
        self,
        sma_val: float,
        adx_val: float,
        current_price: float,
    ) -> MarketRegime:
        """Общая классификация — единственная точка правды."""
        if sma_val <= 0:
            return MarketRegime.CHOP

        distance_pct = (current_price - sma_val) / sma_val
        is_strong_trend = adx_val >= self.adx_threshold

        if abs(distance_pct) < self.neutral_band_pct:
            logger.debug(
                "Режим CHOP: цена %.2f близко к SMA%.0f %.2f (%.2f%%)",
                current_price, self.sma_period, sma_val, distance_pct * 100,
            )
            return MarketRegime.CHOP

        if not is_strong_trend:
            logger.debug(
                "Режим CHOP: ADX=%.1f < %.1f (слабый тренд)",
                adx_val, self.adx_threshold,
            )
            return MarketRegime.CHOP

        if distance_pct > 0:
            logger.debug(
                "Режим BULL: цена %.2f > SMA%.0f %.2f (%.2f%%), ADX=%.1f",
                current_price, self.sma_period, sma_val,
                distance_pct * 100, adx_val,
            )
            return MarketRegime.BULL

        logger.debug(
            "Режим BEAR: цена %.2f < SMA%.0f %.2f (%.2f%%), ADX=%.1f",
            current_price, self.sma_period, sma_val,
            distance_pct * 100, adx_val,
        )
        return MarketRegime.BEAR