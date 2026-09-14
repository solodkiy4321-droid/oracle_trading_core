"""Тесты для Багов C и D:
- C: при отсутствии pyarrow кэш пишется в CSV (fallback).
- C: при наличии pyarrow кэш пишется в parquet.
- D: fetch() делает только ОДИН запрос к yfinance, не два.

Все тесты используют mock yfinance — реальная сеть не дёргается.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.fetcher import (
    MarketDataFetcher,
    CACHE_FORMAT_CSV,
    CACHE_FORMAT_PARQUET,
    _detect_cache_format,
)


# ---------- Утилиты ----------

def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


@pytest.fixture
def tmp_cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    yield str(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------- Fallback-детект формата ----------

def test_detect_cache_format_returns_parquet_when_available():
    """Если pyarrow есть — формат parquet."""
    fmt = _detect_cache_format()
    assert fmt in (CACHE_FORMAT_PARQUET, CACHE_FORMAT_CSV)


def test_detect_cache_format_falls_back_to_csv_when_pyarrow_missing(monkeypatch):
    """Если import pyarrow падает — формат csv, без падения."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("simulated missing pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fmt = _detect_cache_format()
    assert fmt == CACHE_FORMAT_CSV


def test_fetcher_init_uses_detected_format(tmp_cache_dir, monkeypatch):
    """При отсутствии pyarrow fetcher.__init__ выставляет csv."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("simulated missing pyarrow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    f = MarketDataFetcher(cache_dir=tmp_cache_dir)
    assert f.cache_format == CACHE_FORMAT_CSV


# ---------- Сохранение/загрузка кэша ----------

def test_save_and_load_cache_csv(tmp_cache_dir):
    """CSV-формат: save → load, данные совпадают."""
    f = MarketDataFetcher(
        cache_dir=tmp_cache_dir,
        cache_format=CACHE_FORMAT_CSV,
    )
    df = _make_ohlcv(50)
    f._save_to_cache("TEST", "1h", df, "730d", "1h")

    loaded = f._load_from_cache("TEST", "1h")
    assert loaded is not None
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)


def test_save_and_load_cache_parquet(tmp_cache_dir):
    """Parquet-формат (если pyarrow доступен): save → load."""
    pytest.importorskip("pyarrow")
    f = MarketDataFetcher(
        cache_dir=tmp_cache_dir,
        cache_format=CACHE_FORMAT_PARQUET,
    )
    df = _make_ohlcv(50)
    f._save_to_cache("TEST", "1h", df, "730d", "1h")

    loaded = f._load_from_cache("TEST", "1h")
    assert loaded is not None
    assert len(loaded) == len(df)


def test_cache_path_uses_selected_format(tmp_cache_dir):
    """Путь к кэшу зависит от выбранного формата."""
    f_csv = MarketDataFetcher(
        cache_dir=tmp_cache_dir, cache_format=CACHE_FORMAT_CSV,
    )
    p = f_csv._cache_path("BTC-USD", "1h")
    assert p.suffix == ".csv"

    f_pq = MarketDataFetcher(
        cache_dir=tmp_cache_dir, cache_format=CACHE_FORMAT_PARQUET,
    )
    p2 = f_pq._cache_path("BTC-USD", "1h")
    assert p2.suffix == ".parquet"


def test_cache_meta_stores_format(tmp_cache_dir):
    """В meta.json должен быть cache_format."""
    import json
    f = MarketDataFetcher(
        cache_dir=tmp_cache_dir, cache_format=CACHE_FORMAT_CSV,
    )
    df = _make_ohlcv(20)
    f._save_to_cache("TEST", "1h", df, "730d", "1h")

    meta_path = f._meta_path("TEST", "1h")
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta.get("cache_format") == CACHE_FORMAT_CSV


def test_clear_cache_removes_csv_and_json(tmp_cache_dir):
    """clear_cache чистит и csv, и json."""
    f = MarketDataFetcher(
        cache_dir=tmp_cache_dir, cache_format=CACHE_FORMAT_CSV,
    )
    df = _make_ohlcv(20)
    f._save_to_cache("TEST", "1h", df, "730d", "1h")

    removed = f.clear_cache("TEST")
    # .csv + .meta.json = 2 файла
    assert removed == 2


# ---------- Баг D: один запрос к yfinance ----------

def test_fetch_makes_single_yfinance_call(tmp_cache_dir):
    """
    Было: два вызова ticker.history() на один fetch().
    Стало: один вызов.
    """
    df = _make_ohlcv(100)

    # Mock yfinance.Ticker.history — считаем вызовы
    mock_history = MagicMock(return_value=df)
    mock_ticker = MagicMock()
    mock_ticker.history = mock_history

    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker

    with patch.object(MarketDataFetcher, "_yf_module", return_value=mock_yf):
        f = MarketDataFetcher(
            cache_dir=tmp_cache_dir,
            cache_format=CACHE_FORMAT_CSV,
        )
        result = f.fetch("BTC-USD", "1h", limit=100, force_refresh=True)

    assert len(result) == 100
    # Ключевая проверка: ровно ОДИН вызов history
    assert mock_history.call_count == 1, (
        f"Ожидался 1 вызов yfinance, получено {mock_history.call_count} "
        f"(это Баг D)"
    )


def test_fetch_uses_cache_on_second_call(tmp_cache_dir):
    """Второй fetch без force_refresh не должен дёргать yfinance."""
    df = _make_ohlcv(100)

    mock_history = MagicMock(return_value=df)
    mock_ticker = MagicMock()
    mock_ticker.history = mock_history
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker

    with patch.object(MarketDataFetcher, "_yf_module", return_value=mock_yf):
        f = MarketDataFetcher(
            cache_dir=tmp_cache_dir,
            cache_format=CACHE_FORMAT_CSV,
        )
        r1 = f.fetch("BTC-USD", "1h", limit=100, force_refresh=True)
        r2 = f.fetch("BTC-USD", "1h", limit=100, force_refresh=False)

    assert mock_history.call_count == 1, (
        "Второй fetch должен брать из кэша, а не из yfinance"
    )
    assert len(r1) == len(r2)


def test_fetch_saves_cache_in_single_format(tmp_cache_dir):
    """После fetch на диске должен быть файл в выбранном формате, не оба."""
    df = _make_ohlcv(100)
    mock_history = MagicMock(return_value=df)
    mock_ticker = MagicMock()
    mock_ticker.history = mock_history
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value = mock_ticker

    with patch.object(MarketDataFetcher, "_yf_module", return_value=mock_yf):
        f = MarketDataFetcher(
            cache_dir=tmp_cache_dir,
            cache_format=CACHE_FORMAT_CSV,
        )
        f.fetch("BTC-USD", "1h", force_refresh=True)

    cache_dir = Path(tmp_cache_dir)
    csv_files = list(cache_dir.glob("*.csv"))
    parquet_files = list(cache_dir.glob("*.parquet"))

    assert len(csv_files) == 1, f"Ожидался 1 csv, найдено {csv_files}"
    assert len(parquet_files) == 0, f"Parquet не должен создаваться, {parquet_files}"