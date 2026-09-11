"""Загрузка рыночных данных из внешних источников."""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Загружает OHLCV-данные из Yahoo Finance через yfinance.
    
    Поддерживает:
    - Криптовалюты: BTC-USD, ETH-USD
    - Акции: AAPL, TSLA
    - Форекс: EURUSD=X
    - Индексы: ^GSPC
    """

    # Маппинг внутренних таймфреймов на интервалы yfinance
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "1h",  # yfinance не поддерживает 4h, берём 1h
        "1d": "1d",
        "1w": "1wk",
    }

    # Маппинг таймфреймов на период загрузки
    PERIOD_MAP = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "1h": "730d",
        "4h": "730d",
        "1d": "5y",
        "1w": "10y",
    }

    def __init__(self, cache_dir: str = "data_cache"):
        """
        Args:
            cache_dir: Папка для кэша загруженных данных
        """
        self.cache_dir = cache_dir
        from pathlib import Path
        Path(cache_dir).mkdir(exist_ok=True)

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: Optional[int] = 500,
    ) -> pd.DataFrame:
        """
        Загружает OHLCV-данные.
        
        Args:
            symbol: Тикер ('BTC-USD', 'AAPL', 'EURUSD=X')
            timeframe: Таймфрейм ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Максимальное количество баров
            
        Returns:
            DataFrame с колонками open, high, low, close, volume
            
        Raises:
            ValueError: если timeframe не поддерживается
            RuntimeError: если данные не загружены
        """
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Неподдерживаемый таймфрейм: {timeframe}. "
                f"Доступные: {list(self.TIMEFRAME_MAP.keys())}"
            )

        interval = self.TIMEFRAME_MAP[timeframe]
        period = self.PERIOD_MAP[timeframe]

        logger.info("Загрузка %s %s (interval=%s, period=%s)", symbol, timeframe, interval, period)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
        except Exception as e:
            logger.exception("Ошибка загрузки %s: %s", symbol, e)
            raise RuntimeError(f"Не удалось загрузить {symbol}: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(f"Пустой ответ для {symbol} {timeframe}")

        # Приводим колонки к нижнему регистру
        df.columns = [c.lower() for c in df.columns]

        # Оставляем только нужные колонки
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Отсутствуют колонки: {missing}")

        df = df[required].copy()

        # Убираем строки с NaN
        df = df.dropna()

        # Ограничиваем количество баров
        if limit is not None and len(df) > limit:
            df = df.tail(limit)

        # Сортируем по времени
        df = df.sort_index()

        logger.info("Загружено %d баров для %s %s", len(df), symbol, timeframe)
        return df

    def fetch_multiple(
        self,
        symbols: list,
        timeframe: str = "1h",
        limit: Optional[int] = 500,
    ) -> dict:
        """
        Загружает данные для нескольких символов.
        
        Returns:
            Словарь {symbol: DataFrame}
        """
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch(symbol, timeframe, limit)
            except Exception as e:
                logger.error("Пропущен %s: %s", symbol, e)
        return result