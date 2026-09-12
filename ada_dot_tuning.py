"""Точечная настройка ADA и DOT.

Данные загружаются один раз в главном процессе и передаются в воркеры.
Это исключает race condition yfinance.

Sweep: ATR x RR x filter_chop x long_only для ADA и DOT.

Запуск:
    python ada_dot_tuning.py
"""

import asyncio
import csv
import math
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine
from src.engine.symbol_profiles import SymbolProfile, SymbolProfileRegistry


setup_logging(level="ERROR")


SYMBOLS = ["ADA-USD", "DOT-USD"]
BARS = 10000
WARMUP_BARS = 250
STARTING_EQUITY = 10000.0
EXPORT_DIR = "backtest_results/ada_dot_tuning"

ATR_MULTS = [2.0, 2.5, 3.0, 3.5, 4.0]
RR_RATIOS = [2.0, 2.5, 3.0, 3.5]
FILTER_CHOP_OPTIONS = [True, False]
LONG_ONLY_OPTIONS = [False]

BASE_WEIGHTS = {
    "trend": 0.35,
    "elliott_wave": 0.30,
    "volatility": 0.20,
    "volume": 0.15,
}


def calculate_sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(252)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


async def run_single(
    data: pd.DataFrame,
    symbol: str,
    atr_mult: float,
    rr_ratio: float,
    filter_chop: bool,
    long_only: bool,
    export_dir: str,
) -> Dict[str, Any]:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("-", "_")
    db_path = export_path / (
        f"tmp_{safe_symbol}_{atr_mult}_{rr_ratio}_{filter_chop}_{long_only}.db"
    )

    registry = SymbolProfileRegistry()
    profile = SymbolProfile(
        timeframe="1h",
        risk_per_trade_pct=0.01,
        max_position_pct=1.0,
        atr_multiplier=atr_mult,
        atr_period=14,
        default_rr_ratio=rr_ratio,
        filter_chop=filter_chop,
        long_only=long_only,
        weights_override=dict(BASE_WEIGHTS),
        notes=f"{symbol} tune atr={atr_mult} rr={rr_ratio} chop={filter_chop}",
    )
    registry.register(symbol, profile)
    registry.set_default(profile)

    config = EngineConfig(
        starting_equity=STARTING_EQUITY,
        symbol=symbol,
        timeframe="1h",
        journal_db_path=str(db_path),
        gate_mode="aggressive",
        snapshot_every_n_bars=999999,
        use_symbol_profiles=True,
        symbol_profiles=registry,
    )

    engine = TradingEngine(config)

    n = len(data)
    start_idx = max(WARMUP_BARS, n - BARS)
    equity_curve: List[float] = []
    trades_log: List[Dict[str, Any]] = []

    try:
        for i in range(start_idx, n):
            win_start = max(0, i - 500)
            slice_data = data.iloc[win_start: i + 1]
            bar_time = (
                data.index[i].to_pydatetime()
                if hasattr(data.index[i], "to_pydatetime")
                else datetime.now(timezone.utc)
            )
            result = await engine.on_bar(
                data=slice_data, bar_index=i, bar_time=bar_time,
            )
            snap = engine.portfolio.get_snapshot()
            equity_curve.append(snap["current_equity"])

            if result.opened_position is not None and result.decision is not None:
                d = result.decision
                p = result.opened_position
                trades_log.append({
                    "position_id": p.id,
                    "direction": d.direction.name,
                    "pnl": None,
                })

            for closed in result.closed_positions:
                for t in trades_log:
                    if t["position_id"] == closed.id:
                        t["pnl"] = closed.realized_pnl
                        break
    finally:
        stats = engine.get_stats()
        engine.close()

    try:
        db_path.unlink()
    except OSError:
        pass

    portfolio = stats["portfolio"]
    trades = stats["trades"]

    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns.append(
                (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            )

    sharpe = calculate_sharpe(returns)
    max_dd = calculate_max_drawdown(equity_curve)

    completed = [t for t in trades_log if t.get("pnl") is not None]
    buy_pnls = [t["pnl"] for t in completed if t["direction"] == "BUY"]
    sell_pnls = [t["pnl"] for t in completed if t["direction"] == "SELL"]

    return {
        "symbol": symbol,
        "atr_mult": atr_mult,
        "rr_ratio": rr_ratio,
        "filter_chop": filter_chop,
        "long_only": long_only,
        "trades": trades["total"],
        "wins": trades["wins"],
        "losses": trades["losses"],
        "win_rate": trades["win_rate"],
        "profit_factor": trades["profit_factor"],
        "total_pnl": portfolio["total_pnl"],
        "total_pnl_pct": portfolio["total_pnl_pct"],
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "buy_trades": len(buy_pnls),
        "buy_pnl": sum(buy_pnls) if buy_pnls else 0.0,
        "sell_trades": len(sell_pnls),
        "sell_pnl": sum(sell_pnls) if sell_pnls else 0.0,
        "error": "",
    }


def run_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return asyncio.run(run_single(**args))
    except Exception as e:
        return {
            "symbol": args.get("symbol", "?"),
            "atr_mult": args.get("atr_mult", 0.0),
            "rr_ratio": args.get("rr_ratio", 0.0),
            "filter_chop": args.get("filter_chop", False),
            "long_only": args.get("long_only", False),
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "total_pnl_pct": 0.0,
            "sharpe": 0.0, "max_dd_pct": 0.0,
            "buy_trades": 0, "buy_pnl": 0.0,
            "sell_trades": 0, "sell_pnl": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "symbol", "atr_mult", "rr_ratio", "filter_chop", "long_only",
        "trades", "wins", "losses", "win_rate", "profit_factor",
        "total_pnl", "total_pnl_pct", "sharpe", "max_dd_pct",
        "buy_trades", "buy_pnl", "sell_trades", "sell_pnl", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_matrix(
    rows: List[Dict[str, Any]],
    symbol: str,
    filter_chop: bool,
    metric: str,
) -> None:
    chop_str = "ON" if filter_chop else "off"
    print(f"\n{symbol} — {metric} (chop={chop_str})")
    header = f"{'ATR\\RR':>8}"
    for rr in RR_RATIOS:
        header += f" {rr:>8.2f}"
    print(header)
    print("-" * len(header))

    by_key = {
        (r["atr_mult"], r["rr_ratio"]): r
        for r in rows
        if r["symbol"] == symbol
        and r["filter_chop"] == filter_chop
        and not r.get("error")
    }

    for atr in ATR_MULTS:
        line = f"{atr:>8.2f}"
        for rr in RR_RATIOS:
            r = by_key.get((atr, rr))
            if r is None:
                line += f" {'—':>8}"
            else:
                if metric == "total_pnl":
                    line += f" {r['total_pnl']:>+8.0f}"
                elif metric == "sharpe":
                    line += f" {r['sharpe']:>+8.2f}"
                elif metric == "win_rate":
                    line += f" {r['win_rate'] * 100:>8.1f}"
                elif metric == "trades":
                    line += f" {r['trades']:>8d}"
                else:
                    line += f" {'?':>8}"
        print(line)


def main():
    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    print("=" * 130)
    print("ADA & DOT TUNING")
    print(f"Symbols: {SYMBOLS}")
    print(f"ATR:     {ATR_MULTS}")
    print(f"RR:      {RR_RATIOS}")
    print(f"Chop:    {FILTER_CHOP_OPTIONS}")
    print("=" * 130)

    print("\nPreloading data in main process...")
    fetcher = MarketDataFetcher()
    data_cache: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            data_cache[sym] = fetcher.fetch(sym, "1h", limit=BARS + 250)
            print(f"  {sym}: {len(data_cache[sym])} bars")
        except Exception as e:
            print(f"  {sym}: FAILED - {e}")

    if not data_cache:
        print("\nNo data. Exit.")
        return

    tasks = []
    for sym in SYMBOLS:
        if sym not in data_cache:
            continue
        for atr in ATR_MULTS:
            for rr in RR_RATIOS:
                for chop in FILTER_CHOP_OPTIONS:
                    for long_only in LONG_ONLY_OPTIONS:
                        tasks.append({
                            "data": data_cache[sym],
                            "symbol": sym,
                            "atr_mult": atr,
                            "rr_ratio": rr,
                            "filter_chop": chop,
                            "long_only": long_only,
                            "export_dir": EXPORT_DIR,
                        })

    n_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"\nRunning {len(tasks)} tasks with {n_workers} workers...\n")

    results: List[Dict[str, Any]] = []
    with Pool(processes=n_workers) as pool:
        done = 0
        for res in pool.imap_unordered(run_worker, tasks):
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)}")

    valid = [r for r in results if not r.get("error")]

    for sym in SYMBOLS:
        if sym not in data_cache:
            continue
        print("\n" + "#" * 130)
        print(f"# {sym}")
        print("#" * 130)
        for chop in FILTER_CHOP_OPTIONS:
            print_matrix(valid, sym, chop, "total_pnl")
            print_matrix(valid, sym, chop, "sharpe")

        print(f"\n{sym} — BEST BY PnL (top 10)")
        sym_results = [r for r in valid if r["symbol"] == sym]
        for r in sorted(sym_results, key=lambda x: x["total_pnl"], reverse=True)[:10]:
            chop_str = "ON" if r["filter_chop"] else "off"
            print(
                f"  atr={r['atr_mult']:.1f} rr={r['rr_ratio']:.1f} "
                f"chop={chop_str} "
                f"trades={r['trades']:>3} WR={r['win_rate'] * 100:>5.1f}% "
                f"PF={r['profit_factor']:>5.2f} "
                f"PnL=${r['total_pnl']:>+9.2f} "
                f"Sharpe={r['sharpe']:>+6.3f} "
                f"MaxDD={r['max_dd_pct'] * 100:>5.2f}%"
            )

    csv_path = export_path / "ada_dot_tuning.csv"
    save_csv(results, csv_path)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()