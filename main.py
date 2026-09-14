"""Точка входа: бэктест 5 символов с IndicatorCache.

Передаёт full_data в TradingEngine для предвычисления индикаторов.
Замеряет время.
"""

import asyncio
import csv
import logging
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine


setup_logging(level="WARNING")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return resampled.dropna()


def calculate_sharpe(returns: List[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean - risk_free) / std * math.sqrt(252)


def calculate_sortino(returns: List[float], risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    downside = [r for r in returns if r < 0]
    if len(downside) < 2:
        return 0.0
    downside_var = sum(r ** 2 for r in downside) / len(downside)
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return (mean - risk_free) / downside_std * math.sqrt(252)


def calculate_max_drawdown(equity_curve: List[float]) -> tuple:
    if len(equity_curve) < 2:
        return 0.0, 0.0
    peak = equity_curve[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd_dollar = peak - equity
        dd_pct = dd_dollar / peak if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollar = dd_dollar
    return max_dd_pct, max_dd_dollar


@dataclass
class BacktestResult:
    symbol: str = ""
    timeframe: str = ""
    bars: int = 0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    max_dd_dollar: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    starting_equity: float = 0.0
    final_equity: float = 0.0
    duration_sec: float = 0.0
    load_time_sec: float = 0.0
    cache_time_sec: float = 0.0
    loop_time_sec: float = 0.0
    bars_per_sec: float = 0.0
    gate_mode: str = ""
    atr_multiplier: float = 0.0
    rr_ratio: float = 0.0
    filter_chop: bool = False
    regime_1d_filter_mode: str = "none"
    long_only: bool = False
    blocked_by_1d: int = 0
    error: str = ""

    @property
    def is_success(self) -> bool:
        return not self.error


async def run_backtest_async(
    symbol: str,
    data: pd.DataFrame,
    df_1d: pd.DataFrame,
    timeframe: str,
    full_data: pd.DataFrame = None,
    export_dir: str = "backtest_results",
    starting_equity: float = 10000.0,
    gate_mode: str = "aggressive",
) -> BacktestResult:
    result = BacktestResult(
        symbol=symbol, timeframe=timeframe,
        starting_equity=starting_equity, gate_mode=gate_mode,
    )
    t_start = time.time()

    print(f"\n{'=' * 70}")
    print(f"BACKTEST: {symbol} {timeframe} | {len(data)} bars")
    print("=" * 70)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{symbol.replace('-', '_')}_{timeframe}"
    journal_path = f"{export_dir}/{prefix}_journal.db"

    if data is None or len(data) < 500:
        msg = f"Insufficient data: {len(data) if data is not None else 0}"
        print(f"ERROR: {msg}")
        result.error = msg
        result.duration_sec = time.time() - t_start
        return result

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=journal_path,
        gate_mode=gate_mode,
        snapshot_every_n_bars=999999,
    )

    t_cache_start = time.time()
    engine = TradingEngine(config, df_1d=df_1d, full_data=full_data)
    t_cache = time.time() - t_cache_start

    profile = engine.profile
    result.atr_multiplier = profile.atr_multiplier
    result.rr_ratio = profile.default_rr_ratio
    result.filter_chop = profile.filter_chop
    result.regime_1d_filter_mode = profile.regime_1d_filter_mode
    result.long_only = profile.long_only
    result.cache_time_sec = t_cache

    print(
        f"Profile: atr={profile.atr_multiplier}, rr={profile.default_rr_ratio}, "
        f"chop={profile.filter_chop}, 1d={profile.regime_1d_filter_mode}, "
        f"long_only={profile.long_only}, cache={engine.has_indicator_cache}"
    )
    print(f"Cache init: {t_cache:.2f}s")

    start_idx = 250
    processed = 0
    equity_curve: List[float] = []

    t_loop_start = time.time()
    try:
        for i in range(start_idx, len(data)):
            win_start = max(0, i - 500)
            slice_data = data.iloc[win_start: i + 1]
            bar_time = (
                data.index[i].to_pydatetime()
                if hasattr(data.index[i], "to_pydatetime")
                else datetime.now(timezone.utc)
            )
            await engine.on_bar(
                data=slice_data, bar_index=i, bar_time=bar_time,
            )
            processed += 1

            snapshot = engine.portfolio.get_snapshot()
            equity_curve.append(snapshot["current_equity"])

            if processed % 4000 == 0:
                elapsed = time.time() - t_loop_start
                rate = processed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{processed}] equity=${snapshot['current_equity']:.2f}, "
                    f"trades={snapshot['total_trades']}, "
                    f"rate={rate:.0f} bars/s, elapsed={elapsed:.1f}s"
                )
    except Exception as e:
        msg = f"Loop error {symbol}: {e}"
        print(f"ERROR: {msg}")
        logging.exception(msg)
        result.error = msg
        result.loop_time_sec = time.time() - t_loop_start
        result.duration_sec = time.time() - t_start
        engine.close()
        return result

    t_loop = time.time() - t_loop_start

    try:
        stats = engine.get_stats()
        portfolio = stats["portfolio"]
        trades = stats["trades"]
        result.blocked_by_1d = stats.get("blocked_by_1d", 0)

        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                returns.append(
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                )

        sharpe = calculate_sharpe(returns)
        sortino = calculate_sortino(returns)
        max_dd_pct, max_dd_dollar = calculate_max_drawdown(equity_curve)

        expectancy = 0.0
        if trades["total"] > 0:
            expectancy = (
                trades["win_rate"] * trades["avg_win"]
                - (1 - trades["win_rate"]) * trades["avg_loss"]
            )

        result.bars = processed
        result.trades = trades["total"]
        result.wins = trades["wins"]
        result.losses = trades["losses"]
        result.win_rate = trades["win_rate"]
        result.profit_factor = trades["profit_factor"]
        result.total_pnl = portfolio["total_pnl"]
        result.total_pnl_pct = portfolio["total_pnl_pct"]
        result.sharpe = sharpe
        result.sortino = sortino
        result.max_dd_pct = max_dd_pct
        result.max_dd_dollar = max_dd_dollar
        result.expectancy = expectancy
        result.avg_win = trades["avg_win"]
        result.avg_loss = trades["avg_loss"]
        result.final_equity = portfolio["current_equity"]

        print(
            f"  Result: {result.trades} trades, "
            f"WR {result.win_rate * 100:.1f}%, "
            f"PnL ${result.total_pnl:+.2f}, "
            f"blocked_1d={result.blocked_by_1d}"
        )
    except Exception as e:
        msg = f"Stats error {symbol}: {e}"
        print(f"ERROR: {msg}")
        logging.exception(msg)
        result.error = msg
    finally:
        engine.close()

    result.loop_time_sec = t_loop
    result.duration_sec = time.time() - t_start
    result.bars_per_sec = processed / t_loop if t_loop > 0 else 0.0

    print(
        f"  Timing: loop={t_loop:.2f}s, "
        f"cache={t_cache:.2f}s, "
        f"total={result.duration_sec:.2f}s, "
        f"rate={result.bars_per_sec:.0f} bars/s"
    )

    return result


def run_backtest_worker(args: Dict[str, Any]) -> BacktestResult:
    try:
        return asyncio.run(run_backtest_async(**args))
    except Exception as e:
        symbol = args.get("symbol", "unknown")
        logging.exception("Worker crash for %s: %s", symbol, e)
        return BacktestResult(
            symbol=symbol,
            timeframe=args.get("timeframe", ""),
            error=f"Worker crash: {type(e).__name__}: {e}",
        )


def save_summary_csv(results, export_dir="backtest_results", tag=""):
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    summary_path = export_path / f"summary_{timestamp}{suffix}.csv"
    latest_path = export_path / f"summary_latest{suffix}.csv"

    fieldnames = [
        "symbol", "timeframe", "bars", "trades", "wins", "losses", "breakeven",
        "win_rate", "profit_factor", "total_pnl", "total_pnl_pct",
        "sharpe", "sortino", "max_dd_pct", "max_dd_dollar",
        "expectancy", "avg_win", "avg_loss",
        "starting_equity", "final_equity",
        "duration_sec", "load_time_sec", "cache_time_sec",
        "loop_time_sec", "bars_per_sec",
        "gate_mode", "atr_multiplier", "rr_ratio", "filter_chop",
        "regime_1d_filter_mode", "long_only", "blocked_by_1d", "error",
    ]
    rows = []
    for r in results:
        d = asdict(r)
        rows.append({k: d.get(k, "") for k in fieldnames})
    for path in (summary_path, latest_path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSummary: {summary_path}")
    print(f"Latest:  {latest_path}")


def print_summary(results):
    print("\n\n" + "=" * 150)
    print("SUMMARY")
    print("=" * 150)
    header = (
        f"{'Symbol':<12} {'TF':>4} {'Bars':>7} {'Trades':>7} {'Win%':>6} {'PF':>6} "
        f"{'Sharpe':>7} {'MaxDD%':>7} {'PnL$':>10} "
        f"{'Cache':>7} {'Loop':>7} {'Rate':>9} {'Total':>7}"
    )
    print(header)
    print("-" * len(header))

    successful = [r for r in results if r.is_success]
    failed = [r for r in results if not r.is_success]

    for r in successful:
        print(
            f"{r.symbol:<12} {r.timeframe:>4} {r.bars:>7} {r.trades:>7} "
            f"{r.win_rate * 100:>6.1f} {r.profit_factor:>6.2f} "
            f"{r.sharpe:>7.2f} {r.max_dd_pct * 100:>7.2f} "
            f"{r.total_pnl:>+10.2f} "
            f"{r.cache_time_sec:>6.2f}s {r.loop_time_sec:>6.1f}s "
            f"{r.bars_per_sec:>6.0f} b/s {r.duration_sec:>6.1f}s"
        )

    if failed:
        print("\nFAILED:")
        for r in failed:
            print(f"  {r.symbol}: {r.error}")

    if successful:
        total_trades = sum(r.trades for r in successful)
        total_wins = sum(r.wins for r in successful)
        total_pnl = sum(r.total_pnl for r in successful)
        total_loop = sum(r.loop_time_sec for r in successful)
        total_bars = sum(r.bars for r in successful)
        print("\n" + "-" * 80)
        print("TOTAL:")
        print(f"  Trades:          {total_trades}")
        print(f"  Wins:            {total_wins}")
        if total_trades > 0:
            print(f"  Overall WR:      {total_wins / total_trades * 100:.1f}%")
        print(f"  Overall PnL:     ${total_pnl:+.2f}")
        print(f"  Total loop time: {total_loop:.1f}s")
        if total_loop > 0:
            print(f"  Overall rate:    {total_bars / total_loop:.0f} bars/s")


def preload_data(symbol):
    t_start = time.time()
    config = EngineConfig(symbol=symbol, journal_db_path=":memory:")
    profile = config.get_symbol_profile()
    timeframe = profile.timeframe

    fetcher = MarketDataFetcher()
    df_1h = fetcher.fetch(symbol, "1h", limit=40000)
    df_1d = resample_ohlcv(df_1h, "1D")

    if timeframe == "1h":
        df = df_1h
    else:
        rule_map = {"2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
        df = resample_ohlcv(df_1h, rule_map[timeframe])

    return symbol, df, df_1d, timeframe, time.time() - t_start


def main():
    symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "ATOM-USD"]
    starting_equity = 10000.0
    export_dir = "backtest_results"
    gate_mode = "aggressive"

    t_total_start = time.time()

    print("\n" + "=" * 70)
    print(f"BACKTEST: {len(symbols)} symbols with IndicatorCache")
    print(f"CPU count: {cpu_count()}")
    print("=" * 70)

    print("\nPreloading data in main process...")
    symbol_data = {}
    for sym in symbols:
        try:
            _, df, df_1d, tf, load_t = preload_data(sym)
            symbol_data[sym] = (df, df_1d, tf)
            print(f"  {sym}: {len(df)} bars ({tf}), load={load_t:.2f}s")
        except Exception as e:
            print(f"  {sym}: FAILED - {e}")

    if not symbol_data:
        print("\nNo data. Exit.")
        return

    tasks = []
    for sym in symbols:
        if sym not in symbol_data:
            continue
        df, df_1d, tf = symbol_data[sym]
        tasks.append({
            "symbol": sym,
            "data": df,
            "df_1d": df_1d,
            "timeframe": tf,
            "full_data": df,
            "export_dir": export_dir,
            "starting_equity": starting_equity,
            "gate_mode": gate_mode,
        })

    n_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"\nRunning {len(tasks)} tasks with {n_workers} workers...\n")

    t_run_start = time.time()
    results = []

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(run_backtest_worker, tasks):
                results.append(res)
                status = "OK" if res.is_success else f"FAIL: {res.error}"
                print(f"\n[worker done] {res.symbol} -> {status}")
    else:
        for task in tasks:
            results.append(run_backtest_worker(task))

    t_run_total = time.time() - t_run_start
    t_total = time.time() - t_total_start

    order = {sym: i for i, sym in enumerate(symbols)}
    results.sort(key=lambda r: order.get(r.symbol, 999))

    print_summary(results)
    save_summary_csv(results, export_dir=export_dir, tag=gate_mode)

    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(f"  Run (workers):    {t_run_total:.2f}s")
    print(f"  Total:            {t_total:.2f}s")
    print(f"  Workers:          {n_workers}")


if __name__ == "__main__":
    main()