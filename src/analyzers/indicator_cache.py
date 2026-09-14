"""Кэш индикаторов.

Считает все индикаторы ОДИН РАЗ на полном DataFrame.
Анализаторы получают срез через .slice(i).
"""

import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)


class IndicatorCache:
    """
    Предвычисляет индикаторы один раз на полном DataFrame.

    Все индикаторы хранятся как Series/DataFrame с абсолютными индексами.
    Метод .slice(i) возвращает словарь скаляров для бара i.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        swing_window: int = 5,
    ):
        self._data = data
        self._swing_window = swing_window
        self._n = len(data)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Тренд
        self.ema_9 = ta.ema(close, length=9)
        self.ema_50 = ta.ema(close, length=50)
        self.ema_200 = ta.ema(close, length=200)
        self.sma_20 = ta.sma(close, length=20)
        self.sma_50 = ta.sma(close, length=50)
        self.sma_200 = ta.sma(close, length=200)

        # MACD
        self.macd_df = ta.macd(close)

        # ADX
        self.adx_df = ta.adx(high, low, close, length=14)

        # ATR
        self.atr_14 = ta.atr(high, low, close, length=14)

        # ATR MA (rolling) — предвычислено
        if self.atr_14 is not None:
            self.atr_14_ma_50 = self.atr_14.rolling(window=50).mean()
        else:
            self.atr_14_ma_50 = None

        # OBV
        self.obv = ta.obv(close, volume)

        # RSI
        self.rsi_14 = ta.rsi(close, length=14)

        # Swing points (find_peaks)
        from scipy.signal import find_peaks
        highs_arr = high.values
        lows_arr = low.values

        high_idx, _ = find_peaks(highs_arr, distance=swing_window)
        low_idx, _ = find_peaks(-lows_arr, distance=swing_window)

        self._swing_high_indices = set(int(i) for i in high_idx)
        self._swing_low_indices = set(int(i) for i in low_idx)

        # Сортированный список (для get_swing_points)
        self._all_swing_indices = sorted(
            self._swing_high_indices | self._swing_low_indices
        )

        self._highs_arr = highs_arr
        self._lows_arr = lows_arr
        self._closes_arr = close.values

        # Ссылка на close для анализаторов
        self.close_series = close

    @property
    def n(self) -> int:
        return self._n

    def _safe_val(self, series, i: int) -> Optional[float]:
        """Безопасное извлечение значения."""
        if series is None or i < 0 or i >= len(series):
            return None
        v = series.iloc[i]
        if pd.isna(v):
            return None
        return float(v)

    def slice(self, i: int) -> Dict[str, Optional[float]]:
        """
        Возвращает индикаторы на баре i (абсолютный индекс).
        """
        if i < 0 or i >= self._n:
            return {}

        result = {
            "bar_index": i,
            "current_price": float(self._closes_arr[i]),
            "ema_9": self._safe_val(self.ema_9, i),
            "ema_50": self._safe_val(self.ema_50, i),
            "ema_200": self._safe_val(self.ema_200, i),
            "sma_20": self._safe_val(self.sma_20, i),
            "sma_50": self._safe_val(self.sma_50, i),
            "sma_200": self._safe_val(self.sma_200, i),
            "atr_14": self._safe_val(self.atr_14, i),
            "atr_14_ma_50": self._safe_val(self.atr_14_ma_50, i),
            "obv": self._safe_val(self.obv, i),
            "rsi_14": self._safe_val(self.rsi_14, i),
        }

        # MACD
        if self.macd_df is not None and i < len(self.macd_df):
            row = self.macd_df.iloc[i]
            macd_cols = [c for c in self.macd_df.columns if c.startswith("MACD_")]
            hist_cols = [c for c in self.macd_df.columns if c.startswith("MACDh_")]
            signal_cols = [c for c in self.macd_df.columns if c.startswith("MACDs_")]

            if macd_cols:
                v = row[macd_cols[0]]
                result["macd"] = float(v) if not pd.isna(v) else None
            if hist_cols:
                v = row[hist_cols[0]]
                result["macd_hist"] = float(v) if not pd.isna(v) else None
            if signal_cols:
                v = row[signal_cols[0]]
                result["macd_signal"] = float(v) if not pd.isna(v) else None

        # ADX
        if self.adx_df is not None and i < len(self.adx_df):
            row = self.adx_df.iloc[i]
            adx_cols = [c for c in self.adx_df.columns if c.startswith("ADX_")]
            dmp_cols = [c for c in self.adx_df.columns if c.startswith("DMP_")]
            dmn_cols = [c for c in self.adx_df.columns if c.startswith("DMN_")]

            if adx_cols:
                v = row[adx_cols[0]]
                result["adx"] = float(v) if not pd.isna(v) else None
            if dmp_cols:
                v = row[dmp_cols[0]]
                result["dmp"] = float(v) if not pd.isna(v) else None
            if dmn_cols:
                v = row[dmn_cols[0]]
                result["dmn"] = float(v) if not pd.isna(v) else None

        return result

    def is_swing_high(self, i: int) -> bool:
        return i in self._swing_high_indices

    def is_swing_low(self, i: int) -> bool:
        return i in self._swing_low_indices

    def get_swing_points(self, end_idx: int) -> list:
        """Возвращает список swing points до end_idx (включительно)."""
        points = []
        for i in self._swing_high_indices:
            if i <= end_idx:
                points.append((i, float(self._highs_arr[i]), "high"))
        for i in self._swing_low_indices:
            if i <= end_idx:
                points.append((i, float(self._lows_arr[i]), "low"))
        points.sort(key=lambda p: p[0])
        return points