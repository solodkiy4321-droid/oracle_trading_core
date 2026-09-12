import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_TTL_BY_TIMEFRAME = {
    "1m": 3600,
    "5m": 3600,
    "15m": 3600,
    "30m": 3600,
    "1h": 3600,
    "4h": 3600,
    "1d": 86400,
    "1w": 86400,
}

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_FORMAT = "parquet"


@dataclass
class CacheMeta:
    symbol: str = ""
    timeframe: str = ""
    fetched_at: float = 0.0
    rows: int = 0
    source: str = "yfinance"
    period: str = ""
    interval: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "fetched_at": self.fetched_at,
            "fetched_at_iso": self._iso(),
            "rows": self.rows,
            "source": self.source,
            "period": self.period,
            "interval": self.interval,
        }

    def _iso(self) -> str:
        try:
            return datetime.fromtimestamp(self.fetched_at, tz=timezone.utc).isoformat()
        except Exception:
            return ""

    @classmethod
    def from_dict(cls, d: dict) -> "CacheMeta":
        return cls(
            symbol=d.get("symbol", ""),
            timeframe=d.get("timeframe", ""),
            fetched_at=float(d.get("fetched_at", 0.0)),
            rows=int(d.get("rows", 0)),
            source=d.get("source", "yfinance"),
            period=d.get("period", ""),
            interval=d.get("interval", ""),
        )


class MarketDataFetcher:

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "1h",
        "1d": "1d",
        "1w": "1wk",
    }

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

    def __init__(
        self,
        cache_dir: str = "data_cache",
        cache_enabled: bool = True,
        cache_ttl_override: Optional[int] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.cache_ttl_override = cache_ttl_override

    def _safe_name(self, symbol: str) -> str:
        return symbol.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_dir / f"{self._safe_name(symbol)}_{timeframe}.{CACHE_FORMAT}"

    def _meta_path(self, symbol: str, timeframe: str) -> Path:
        return self.cache_dir / f"{self._safe_name(symbol)}_{timeframe}.meta.json"

    def _get_ttl(self, timeframe: str) -> int:
        if self.cache_ttl_override is not None:
            return self.cache_ttl_override
        return CACHE_TTL_BY_TIMEFRAME.get(timeframe, 3600)

    def _is_cache_valid(self, symbol: str, timeframe: str) -> bool:
        if not self.cache_enabled:
            return False

        cache_path = self._cache_path(symbol, timeframe)
        if not cache_path.exists():
            return False

        try:
            mtime = cache_path.stat().st_mtime
        except OSError:
            return False

        ttl = self._get_ttl(timeframe)
        age = time.time() - mtime
        if age > ttl:
            logger.debug(
                "Cache stale: %s %s (age=%.1fs > ttl=%ds)",
                symbol, timeframe, age, ttl,
            )
            return False

        return True

    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cache_path = self._cache_path(symbol, timeframe)
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning("Broken cache %s: %s. Refetching.", cache_path, e)
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

        if df is None or df.empty:
            logger.warning("Empty cache %s. Refetching.", cache_path)
            return None

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Cache %s missing columns %s. Refetching.", cache_path, missing)
            return None

        logger.info("Loaded from cache: %s %s (%d bars)", symbol, timeframe, len(df))
        return df

    def _save_to_cache(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        period: str,
        interval: str,
    ) -> None:
        if not self.cache_enabled:
            return

        cache_path = self._cache_path(symbol, timeframe)
        meta_path = self._meta_path(symbol, timeframe)

        tmp_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")

        try:
            df.to_parquet(tmp_cache, engine="pyarrow", compression="snappy")

            meta = CacheMeta(
                symbol=symbol,
                timeframe=timeframe,
                fetched_at=time.time(),
                rows=len(df),
                source="yfinance",
                period=period,
                interval=interval,
            )
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

            os.replace(tmp_cache, cache_path)
            os.replace(tmp_meta, meta_path)

            logger.debug(
                "Saved to cache: %s %s (%d bars, %.1f KB)",
                symbol, timeframe, len(df),
                cache_path.stat().st_size / 1024,
            )
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", cache_path, e)
            for p in (tmp_cache, tmp_meta):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    def _read_meta(self, symbol: str, timeframe: str) -> Optional[CacheMeta]:
        meta_path = self._meta_path(symbol, timeframe)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return CacheMeta.from_dict(json.load(f))
        except Exception:
            return None

    def fetch(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: Optional[int] = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Available: {list(self.TIMEFRAME_MAP.keys())}"
            )

        interval = self.TIMEFRAME_MAP[timeframe]
        period = self.PERIOD_MAP[timeframe]

        if not force_refresh and self._is_cache_valid(symbol, timeframe):
            df = self._load_from_cache(symbol, timeframe)
            if df is not None:
                if limit is not None and len(df) > limit:
                    df = df.tail(limit).copy()
                return df

        logger.info(
            "Fetching %s %s (interval=%s, period=%s)",
            symbol, timeframe, interval, period,
        )

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
        except Exception as e:
            logger.exception("Fetch error %s: %s", symbol, e)
            if self.cache_enabled:
                stale = self._load_from_cache(symbol, timeframe)
                if stale is not None:
                    logger.warning(
                        "yfinance unavailable, using stale cache %s %s",
                        symbol, timeframe,
                    )
                    if limit is not None and len(stale) > limit:
                        stale = stale.tail(limit).copy()
                    return stale
            raise RuntimeError(f"Failed to fetch {symbol}: {e}") from e

        if df is None or df.empty:
            raise RuntimeError(f"Empty response for {symbol} {timeframe}")

        df.columns = [c.lower() for c in df.columns]

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns: {missing}")

        df = df[REQUIRED_COLUMNS].copy()
        df = df.dropna()

        if limit is not None and len(df) > limit:
            df = df.tail(limit).copy()

        df = df.sort_index()

        logger.info("Fetched %d bars for %s %s", len(df), symbol, timeframe)

        try:
            if not force_refresh or True:
                full = ticker.history(period=period, interval=interval, auto_adjust=False)
                if full is not None and not full.empty:
                    full.columns = [c.lower() for c in full.columns]
                    if all(c in full.columns for c in REQUIRED_COLUMNS):
                        full = full[REQUIRED_COLUMNS].dropna().sort_index()
                        self._save_to_cache(symbol, timeframe, full, period, interval)
        except Exception as e:
            logger.debug("Failed to save full cache: %s", e)

        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: Optional[int] = 500,
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                result[symbol] = self.fetch(
                    symbol, timeframe, limit, force_refresh=force_refresh,
                )
            except Exception as e:
                logger.error("Skipped %s: %s", symbol, e)
        return result

    def get_cache_info(self) -> List[dict]:
        info = []
        for meta_path in self.cache_dir.glob("*.meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = CacheMeta.from_dict(json.load(f))
                cache_path = self.cache_dir / (
                    f"{self._safe_name(meta.symbol)}_{meta.timeframe}.{CACHE_FORMAT}"
                )
                size_kb = cache_path.stat().st_size / 1024 if cache_path.exists() else 0
                age_sec = time.time() - meta.fetched_at
                ttl = self._get_ttl(meta.timeframe)
                info.append({
                    "symbol": meta.symbol,
                    "timeframe": meta.timeframe,
                    "rows": meta.rows,
                    "age_sec": age_sec,
                    "ttl_sec": ttl,
                    "is_fresh": age_sec <= ttl,
                    "size_kb": size_kb,
                })
            except Exception:
                continue
        return info

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        removed = 0
        pattern = f"{self._safe_name(symbol)}_*" if symbol else "*"
        for p in self.cache_dir.glob(pattern):
            if p.suffix in (f".{CACHE_FORMAT}", ".json"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        logger.info("Cleared cache files: %d", removed)
        return removed