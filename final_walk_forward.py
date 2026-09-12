"""Финальная walk-forward валидация системы.

Прогоняет 5 символов с финальными параметрами на 4 фолдах.
Проверяет устойчивость: стабильна ли система во времени.

Использует предзагрузку данных в главном процессе.

Запуск:
    python final_walk_forward.py
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


SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "ATOM-USD"]
BARS_1H = 40000
WARMUP_BARS = 250
TEST_BARS = 1500
N_FOLDS = 4

STARTING_EQUITY = 10000.0
EXPORT_DIR = "backtest_results/final_walk_forward"

# Финальные параметры из tuning
FINAL_PARAMS = {
    "BTC-USD": {
        "timeframe": "2h",
        "atr": 2.0, "rr": 3.0, "chop": True, "long_only": False,
    },
    "ETH-USD": {
        "timeframe": "1h",
        "atr": 3.0, "rr": 2.5, "chop": True, "long_only": False,
    },
    "ADA-USD": {
        "timeframe": "1h",
        "atr": 3.0, "rr": 2.5, "chop": True, "long_only": False,
    },
    "DOT-USD": {
        "timeframe": "1h",
        "atr": 2.5, "rr": 2.5, "chop": True, "long_only": False,
    },
    "ATOM-USD": {
        "timeframe": "1h",
        "atr": 3.0, "rr": 2.0, "chop": True, "long_only": False,
    },
}

BASE_WEIGHTS = {
    "trend": 0.35,
    "elliott_wave": 0.30,
    "volatility": 0.20,
    "volume": 0.15,
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return resampled.dropna()


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


async def run_fold(
    data: pd.DataFrame,
    symbol: str,
    fold_id: int,
    start_idx: int,
    end_idx: int,
    export_dir: str,
) -> Dict[str, Any]:
    params = FINAL_PARAMS[symbol]
    timeframe = params["timeframe"]

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("-", "_")
    db_path = export_path / f"tmp_{safe_symbol}_fold{fold_id}.db"

    registry = SymbolProfileRegistry()
    profile = SymbolProfile(
        timeframe=timeframe,
        risk_per_trade_pct=0.01,
        max_position_pct=1.0,
        atr_multiplier=params["atr"],
        atr_period=14,
        default_rr_ratio=params["rr"],
        filter_chop=params["chop"],
        long_only=params["long_only"],
        weights_override=dict(BASE_WEIGHTS),
        notes=f"{symbol} fold {fold_id}",
    )
    registry.register(symbol, profile)
    registry.set_default(profile)

    config = EngineConfig(
        starting_equity=STARTING_EQUITY,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=str(db_path),
        gate_mode="aggressive",
        snapshot_every_n_bars=999999,
        use_symbol_profiles=True,
        symbol_profiles=registry,
    )

    engine = TradingEngine(config)

    equity_curve: List[float] = []

    try:
        for i in range(start_idx, end_idx):
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
            snap = engine.portfolio.get_snapshot()
            equity_curve.append(snap["current_equity"])
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

    start_date = data.index[start_idx] if start_idx < len(data) else ""
    end_date = data.index[min(end_idx - 1, len(data) - 1)] if end_idx > 0 else ""

    return {
        "symbol": symbol,
        "fold": fold_id,
        "timeframe": timeframe,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trades": trades["total"],
        "wins": trades["wins"],
        "losses": trades["losses"],
        "win_rate": trades["win_rate"],
        "profit_factor": trades["profit_factor"],
        "total_pnl": portfolio["total_pnl"],
        "total_pnl_pct": portfolio["total_pnl_pct"],
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "error": "",
    }


def run_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return asyncio.run(run_fold(**args))
    except Exception as e:
        return {
            "symbol": args.get("symbol", "?"),
            "fold": args.get("fold_id", 0),
            "timeframe": "?",
            "start_date": "", "end_date": "",
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "total_pnl": 0.0, "total_pnl_pct": 0.0,
            "sharpe": 0.0, "max_dd_pct": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "symbol", "fold", "timeframe", "start_date", "end_date",
        "trades", "wins", "losses", "win_rate", "profit_factor",
        "total_pnl", "total_pnl_pct", "sharpe", "max_dd_pct", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    print("=" * 130)
    print("FINAL WALK-FORWARD VALIDATION")
    print(f"Symbols:  {SYMBOLS}")
    print(f"Folds:    {N_FOLDS}, test {TEST_BARS} bars each")
    print("=" * 130)

    print("\nPreloading data in main process...")
    fetcher = MarketDataFetcher()
    data_cache: Dict[str, Tuple[pd.DataFrame, str]] = {}

    for sym in SYMBOLS:
        params = FINAL_PARAMS[sym]
        timeframe = params["timeframe"]
        try:
            if timeframe == "1h":
                df = fetcher.fetch(sym, "1h", limit=BARS_1H)
            else:
                df_1h = fetcher.fetch(sym, "1h", limit=BARS_1H)
                rule_map = {"2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
                df = resample_ohlcv(df_1h, rule_map[timeframe])

            data_cache[sym] = (df, timeframe)
            print(f"  {sym}: {len(df)} bars ({timeframe})")
        except Exception as e:
            print(f"  {sym}: FAILED - {e}")

    if not data_cache:
        print("\nNo data. Exit.")
        return

    tasks = []
    for sym in SYMBOLS:
        if sym not in data_cache:
            continue
        df, tf = data_cache[sym]
        n = len(df)

        for fold_id in range(N_FOLDS):
            test_end = n - (N_FOLDS - 1 - fold_id) * TEST_BARS
            test_start = test_end - TEST_BARS

            if test_start < WARMUP_BARS or test_end > n:
                continue

            tasks.append({
                "data": df,
                "symbol": sym,
                "fold_id": fold_id,
                "start_idx": test_start,
                "end_idx": test_end,
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
            if done % 5 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)}")

    valid = [r for r in results if not r.get("error")]

    print("\n" + "=" * 130)
    print("PER SYMBOL — ALL FOLDS")
    print("=" * 130)

    symbol_order = {sym: i for i, sym in enumerate(SYMBOLS)}

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue
        sym_results.sort(key=lambda r: r["fold"])

        print(f"\n{sym} ({FINAL_PARAMS[sym]['timeframe']})")
        print(
            f"  {'fold':>5} {'period':>24} {'trades':>7} {'WR%':>7} "
            f"{'PF':>6} {'PnL$':>10} {'Sharpe':>8} {'MaxDD%':>8}"
        )

        for r in sym_results:
            period = f"{r['start_date'][:10]}..{r['end_date'][:10]}"
            print(
                f"  {r['fold']:>5} {period:>24} "
                f"{r['trades']:>7} "
                f"{r['win_rate'] * 100:>7.1f} "
                f"{r['profit_factor']:>6.2f} "
                f"{r['total_pnl']:>+10.2f} "
                f"{r['sharpe']:>+8.3f} "
                f"{r['max_dd_pct'] * 100:>8.2f}"
            )

        profitable = sum(1 for r in sym_results if r["total_pnl"] > 0)
        total_pnl = sum(r["total_pnl"] for r in sym_results)
        avg_sharpe = sum(r["sharpe"] for r in sym_results) / len(sym_results)

        print(
            f"  {'':>5} {'TOTAL':>24} "
            f"{'':>7} {'':>7} {'':>6} "
            f"{total_pnl:>+10.2f} {avg_sharpe:>+8.3f}"
        )
        print(f"  Profitable folds: {profitable}/{len(sym_results)}")

    print("\n" + "=" * 130)
    print("SUMMARY BY SYMBOL")
    print("=" * 130)
    print(
        f"{'symbol':<12} {'folds':>7} {'prof':>7} "
        f"{'trades':>7} {'WR%':>7} {'PnL$':>10} "
        f"{'avg_sharpe':>12} {'avg_maxdd':>10}"
    )
    print("-" * 90)

    total_pnl_all = 0
    total_trades_all = 0
    total_wins_all = 0

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue

        profitable = sum(1 for r in sym_results if r["total_pnl"] > 0)
        total_pnl = sum(r["total_pnl"] for r in sym_results)
        total_trades = sum(r["trades"] for r in sym_results)
        total_wins = sum(r["wins"] for r in sym_results)
        avg_sharpe = sum(r["sharpe"] for r in sym_results) / len(sym_results)
        avg_maxdd = sum(r["max_dd_pct"] for r in sym_results) / len(sym_results)

        total_pnl_all += total_pnl
        total_trades_all += total_trades
        total_wins_all += total_wins

        wr = total_wins / total_trades if total_trades > 0 else 0.0

        print(
            f"{sym:<12} {len(sym_results):>7} {profitable:>7} "
            f"{total_trades:>7} {wr * 100:>7.1f} "
            f"{total_pnl:>+10.2f} "
            f"{avg_sharpe:>+12.3f} "
            f"{avg_maxdd * 100:>10.2f}"
        )

    print("\n" + "=" * 130)
    print("TOTAL")
    print("=" * 130)
    print(f"  Total PnL:    ${total_pnl_all:+.2f}")
    print(f"  Total trades: {total_trades_all}")
    if total_trades_all > 0:
        print(f"  Overall WR:   {total_wins_all / total_trades_all * 100:.1f}%")

    total_folds = sum(
        1 for sym in SYMBOLS
        for r in valid if r["symbol"] == sym
    )
    total_profitable = sum(
        1 for r in valid if r["total_pnl"] > 0
    )

    print(f"\n  Profitable folds: {total_profitable}/{total_folds}")

    if total_folds > 0:
        ratio = total_profitable / total_folds
        print("\n  VERDICT:")
        if ratio >= 0.9:
            print(f"    EXCELLENT: {total_profitable}/{total_folds} folds profitable")
        elif ratio >= 0.75:
            print(f"    GOOD: {total_profitable}/{total_folds} folds profitable")
        elif ratio >= 0.5:
            print(f"    WEAK: {total_profitable}/{total_folds} folds profitable")
        else:
            print(f"    BAD: {total_profitable}/{total_folds} folds profitable")

    csv_path = export_path / "final_walk_forward.csv"
    save_csv(results, csv_path)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()