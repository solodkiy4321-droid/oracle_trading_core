"""Точка входа: бэктест 5 крипто-символов.

Timeframe берётся из SymbolProfile:
- BTC: 2h (long_only)
- ETH, ADA, DOT, ATOM: 1h
"""

import asyncio
import csv
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine


setup_logging(level="WARNING")


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
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
    gate_mode: str = ""
    atr_multiplier: float = 0.0
    rr_ratio: float = 0.0
    filter_chop: bool = False
    long_only: bool = False
    error: str = ""

    @property
    def is_success(self) -> bool:
        return not self.error


async def run_backtest_async(
    symbol: str,
    bars_to_process: int = 10000,
    export_dir: str = "backtest_results",
    starting_equity: float = 10000.0,
    force_refresh: bool = False,
    gate_mode: str = "aggressive",
) -> BacktestResult:
    result = BacktestResult(
        symbol=symbol,
        starting_equity=starting_equity,
        gate_mode=gate_mode,
    )
    t_start = datetime.now(timezone.utc)

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        journal_db_path="",
        gate_mode=gate_mode,
        snapshot_every_n_bars=999999,
    )

    profile = config.get_symbol_profile()
    timeframe = profile.timeframe

    result.timeframe = timeframe
    result.atr_multiplier = profile.atr_multiplier
    result.rr_ratio = profile.default_rr_ratio
    result.filter_chop = profile.filter_chop
    result.long_only = profile.long_only

    print(f"\n{'=' * 70}")
    print(f"BACKTEST: {symbol} {timeframe} | {bars_to_process} bars")
    print(
        f"Profile: atr={profile.atr_multiplier}, rr={profile.default_rr_ratio}, "
        f"chop={profile.filter_chop}, long_only={profile.long_only}"
    )
    print("=" * 70)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{symbol.replace('-', '_')}_{timeframe}"
    journal_path = f"{export_dir}/{prefix}_journal.db"

    fetcher = MarketDataFetcher()
    try:
        if timeframe == "1h":
            data = fetcher.fetch(
                symbol, "1h",
                limit=bars_to_process + 250,
                force_refresh=force_refresh,
            )
        else:
            base_limit = bars_to_process * (24 if timeframe == "1d" else
                                            12 if timeframe == "2h" else
                                            4 if timeframe == "4h" else 1)
            base_limit = min(base_limit + 500, 40000)
            df_1h = fetcher.fetch(
                symbol, "1h",
                limit=base_limit,
                force_refresh=force_refresh,
            )
            rule_map = {"2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
            data = resample_ohlcv(df_1h, rule_map[timeframe])
            data = data.tail(bars_to_process + 250)
    except Exception as e:
        msg = f"Fetch error {symbol}: {e}"
        print(f"ERROR: {msg}")
        result.error = msg
        result.duration_sec = (datetime.now(timezone.utc) - t_start).total_seconds()
        return result

    print(f"Loaded {len(data)} bars")

    if len(data) < 500:
        msg = f"Insufficient data: {len(data)} < 500"
        print(f"ERROR: {msg}")
        result.error = msg
        result.duration_sec = (datetime.now(timezone.utc) - t_start).total_seconds()
        return result

    if len(data) < bars_to_process:
        bars_to_process = len(data)

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=journal_path,
        gate_mode=gate_mode,
        snapshot_every_n_bars=999999,
    )

    engine = TradingEngine(config)
    profile = engine.profile

    start_idx = max(250, len(data) - bars_to_process)
    processed = 0
    equity_curve: List[float] = []

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

            if processed % 2000 == 0:
                print(
                    f"  [{processed}] equity=${snapshot['current_equity']:.2f}, "
                    f"trades={snapshot['total_trades']}"
                )
    except Exception as e:
        msg = f"Loop error {symbol}: {e}"
        print(f"ERROR: {msg}")
        logging.exception(msg)
        result.error = msg
        result.duration_sec = (datetime.now(timezone.utc) - t_start).total_seconds()
        engine.close()
        return result

    try:
        stats = engine.get_stats()
        portfolio = stats["portfolio"]
        trades = stats["trades"]

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
            f"PnL ${result.total_pnl:+.2f}"
        )
    except Exception as e:
        msg = f"Stats error {symbol}: {e}"
        print(f"ERROR: {msg}")
        logging.exception(msg)
        result.error = msg
    finally:
        engine.close()

    result.duration_sec = (datetime.now(timezone.utc) - t_start).total_seconds()
    return result


def run_backtest_worker(args: Dict[str, Any]) -> BacktestResult:
    try:
        return asyncio.run(run_backtest_async(**args))
    except Exception as e:
        symbol = args.get("symbol", "unknown")
        logging.exception("Worker crash for %s: %s", symbol, e)
        return BacktestResult(
            symbol=symbol,
            error=f"Worker crash: {type(e).__name__}: {e}",
        )


def save_summary_csv(
    results: List[BacktestResult],
    export_dir: str = "backtest_results",
    tag: str = "",
) -> Path:
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
        "starting_equity", "final_equity", "duration_sec",
        "gate_mode", "atr_multiplier", "rr_ratio", "filter_chop",
        "long_only", "error",
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
    return summary_path


def print_summary(results: List[BacktestResult]) -> None:
    print("\n\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    header = (
        f"{'Symbol':<12} {'TF':>4} {'Trades':>7} {'Win%':>6} {'PF':>6} "
        f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} "
        f"{'PnL%':>7} {'PnL$':>10} {'Chop':>6} {'Long':>6} {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    successful = [r for r in results if r.is_success]
    failed = [r for r in results if not r.is_success]

    for r in successful:
        chop_str = "ON" if r.filter_chop else "off"
        long_str = "ON" if r.long_only else "off"
        print(
            f"{r.symbol:<12} "
            f"{r.timeframe:>4} "
            f"{r.trades:>7} "
            f"{r.win_rate * 100:>6.1f} "
            f"{r.profit_factor:>6.2f} "
            f"{r.sharpe:>7.2f} "
            f"{r.sortino:>8.2f} "
            f"{r.max_dd_pct * 100:>7.2f} "
            f"{r.total_pnl_pct * 100:>7.2f} "
            f"{r.total_pnl:>+10.2f} "
            f"{chop_str:>6} "
            f"{long_str:>6} "
            f"{r.duration_sec:>6.1f}s"
        )

    if failed:
        print("\nFAILED:")
        for r in failed:
            print(f"  {r.symbol}: {r.error}")

    if successful:
        total_trades = sum(r.trades for r in successful)
        total_wins = sum(r.wins for r in successful)
        total_losses = sum(r.losses for r in successful)
        total_pnl = sum(r.total_pnl for r in successful)

        print("\n" + "-" * 80)
        print("TOTAL:")
        print(f"  Trades:          {total_trades}")
        print(f"  Wins/Losses:     {total_wins} / {total_losses}")
        if total_trades > 0:
            print(f"  Overall WR:      {total_wins / total_trades * 100:.1f}%")
        print(f"  Overall PnL:     ${total_pnl:+.2f}")


def main():
    symbols = [
        "BTC-USD",
        "ETH-USD",
        "ADA-USD",
        "DOT-USD",
        "ATOM-USD",
    ]

    bars_per_symbol = 10000
    starting_equity = 10000.0
    export_dir = "backtest_results"
    gate_mode = "aggressive"

    n_workers = min(len(symbols), max(1, cpu_count() - 1))

    print("\n" + "=" * 70)
    print(
        f"BACKTEST: {len(symbols)} symbols x {bars_per_symbol} bars "
        f"({n_workers} workers)"
    )
    print("=" * 70)

    tasks = [
        {
            "symbol": sym,
            "bars_to_process": bars_per_symbol,
            "export_dir": export_dir,
            "starting_equity": starting_equity,
            "force_refresh": False,
            "gate_mode": gate_mode,
        }
        for sym in symbols
    ]

    results: List[BacktestResult] = []

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(run_backtest_worker, tasks):
                results.append(res)
                status = "OK" if res.is_success else f"FAIL: {res.error}"
                print(f"\n[worker done] {res.symbol} -> {status}")
    else:
        for task in tasks:
            results.append(run_backtest_worker(task))

    order = {sym: i for i, sym in enumerate(symbols)}
    results.sort(key=lambda r: order.get(r.symbol, 999))

    print_summary(results)
    save_summary_csv(results, export_dir=export_dir, tag=gate_mode)


if __name__ == "__main__":
    main()