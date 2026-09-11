"""Детектор рыночного режима на основе SMA200 и ATR."""

import logging
from enum import Enum
from typing import Optional

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
    Определяет рыночный режим по трём признакам:

    1. Цена относительно SMA200 — направление тренда
    2. ATR относительно среднего ATR — волатильность
    3. ADX — сила тренда

    Логика:
    - BULL: цена > SMA200 и ADX > 20
    - BEAR: цена < SMA200 и ADX > 20
    - CHOP: всё остальное (боковик, слабый тренд)
    """

    def __init__(
        self,
        sma_period: int = 200,
        atr_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        neutral_band_pct: float = 0.02,
    ):
        """
        Args:
            sma_period: Период SMA для определения тренда
            atr_period: Период ATR для волатильности
            adx_period: Период ADX для силы тренда
            adx_threshold: Порог ADX для сильного тренда
            neutral_band_pct: Ширина нейтральной зоны вокруг SMA (2%)
        """
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.neutral_band_pct = neutral_band_pct

    def detect(self, data: pd.DataFrame) -> MarketRegime:
        """
        Определяет текущий рыночный режим.

        Args:
            data: OHLCV DataFrame

        Returns:
            MarketRegime.BULL / BEAR / CHOP
        """
        if data is None or len(data) < self.sma_period:
            logger.debug(
                "Недостаточно данных для определения режима: %d баров (нужно %d)",
                len(data) if data is not None else 0,
                self.sma_period,
            )
            return MarketRegime.CHOP

        try:
            # SMA200
            sma = ta.sma(data["close"], length=self.sma_period)
            if sma is None or len(sma) == 0:
                return MarketRegime.CHOP

            sma_val = float(sma.iloc[-1])
            current_price = float(data["close"].iloc[-1])

            if sma_val <= 0:
                return MarketRegime.CHOP

            # ADX
            adx_df = ta.adx(
                data["high"], data["low"], data["close"],
                length=self.adx_period,
            )
            adx_val = 0.0
            if adx_df is not None and len(adx_df) > 0:
                # Колонка ADX называется ADX_14
                adx_cols = [c for c in adx_df.columns if c.startswith("ADX_")]
                if adx_cols:
                    adx_val = float(adx_df[adx_cols[0]].iloc[-1])

            # Определяем режим
            distance_pct = (current_price - sma_val) / sma_val
            is_strong_trend = adx_val >= self.adx_threshold

            if abs(distance_pct) < self.neutral_band_pct:
                # Цена близко к SMA200 — неопределённость
                logger.debug(
                    "Режим CHOP: цена %.2f близка к SMA200 %.2f (%.2f%%)",
                    current_price, sma_val, distance_pct * 100,
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
                    "Режим BULL: цена %.2f > SMA200 %.2f (%.2f%%), ADX=%.1f",
                    current_price, sma_val, distance_pct * 100, adx_val,
                )
                return MarketRegime.BULL
            else:
                logger.debug(
                    "Режим BEAR: цена %.2f < SMA200 %.2f (%.2f%%), ADX=%.1f",
                    current_price, sma_val, distance_pct * 100, adx_val,
                )
                return MarketRegime.BEAR

        except Exception as e:
            logger.exception("Ошибка в RegimeDetector: %s", e)
            return MarketRegime.CHOP