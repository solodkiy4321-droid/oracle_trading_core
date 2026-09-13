"""Per-symbol оптимизация весов 3 анализаторов.

Анализаторы: trend, elliott_wave, volume.
Шаг 0.1 — точная оптимизация.

Volatility и mean_reversion исключены — sweep v2 показал,
что они не улучшают результат.

Запуск:
    python weights_sweep_3analyzer.py
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
STARTING_EQUITY = 10000.0
EXPORT_DIR = "backtest_results/weights_sweep_3analyzer"

BASE_PARAMS = {
    "BTC-USD": {
        "timeframe": "2h", "atr": 2.0, "rr": 3.0,
        "chop": True, "regime_1d": "baseline", "gate": None,
    },
    "ETH-USD": {
        "timeframe": "1h", "atr": 2.5, "rr": 2.5,
        "chop": True, "regime_1d": "baseline", "gate": "balanced",
    },
    "ADA-USD": {
        "timeframe": "1h", "atr": 3.0, "rr": 2.5,
        "chop": False, "regime_1d": "none", "gate": None,
    },
    "DOT-USD": {
        "timeframe": "1h", "atr": 2.5, "rr": 2.5,
        "chop": False, "regime_1d": "none", "gate": None,
    },
    "ATOM-USD": {
        "timeframe": "1h", "atr": 4.0, "rr": 3.5,
        "chop": True, "regime_1d": "drawdown_10", "gate": "aggressive",
    },
}

ANALYZERS = ["trend", "elliott_wave", "volume"]


def generate_weight_combos(step: float = 0.1) -> List[Dict[str, float]]:
    """Генерирует комбинации весов 3 анализаторов, сумма = 1.0."""
    n = int(round(1.0 / step))
    combos = []
    for i in range(0, n + 1):
        for j in range(0, n + 1 - i):
            k = n - i - j
            combo = {
                "trend": round(i * step, 2),
                "elliott_wave": round(j * step, 2),
                "volume": round(k * step, 2),
            }
            if abs(sum(combo.values()) - 1.0) < 1e-6:
                combos.append(combo)
    return combos


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


async def run_single(
    data: pd.DataFrame,
    df_1d: pd.DataFrame,
    symbol: str,
    weights: Dict[str, float],
    export_dir: str,
) -> Dict[str, Any]:
    params = BASE_PARAMS[symbol]
    timeframe = params["timeframe"]

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("-", "_")
    w_key = f"t{weights['trend']:.1f}_e{weights['elliott_wave']:.1f}_v{weights['volume']:.1f}"
    db_path = export_path / f"tmp_{safe_symbol}_{w_key}.db"

    full_weights = {
        "trend": weights["trend"],
        "elliott_wave": weights["elliott_wave"],
        "volatility": 0.0,
        "volume": weights["volume"],
        "mean_reversion": 0.0,
    }

    registry = SymbolProfileRegistry()
    profile = SymbolProfile(
        timeframe=timeframe,
        risk_per_trade_pct=0.01,
        max_position_pct=1.0,
        atr_multiplier=params["atr"],
        atr_period=14,
        default_rr_ratio=params["rr"],
        filter_chop=params["chop"],
        regime_1d_filter_mode=params["regime_1d"],
        long_only=False,
        gate_mode_override=params["gate"],
        weights_override=full_weights,
        notes=f"{symbol} t={weights['trend']} e={weights['elliott_wave']} v={weights['volume']}",
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

    engine = TradingEngine(config, df_1d=df_1d)

    n = len(data)
    start_idx = WARMUP_BARS
    equity_curve: List[float] = []

    try:
        for i in range(start_idx, n):
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

    return {
        "symbol": symbol,
        "w_trend": weights["trend"],
        "w_elliott": weights["elliott_wave"],
        "w_volume": weights["volume"],
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
        return asyncio.run(run_single(**args))
    except Exception as e:
        return {
            "symbol": args.get("symbol", "?"),
            "w_trend": 0.0, "w_elliott": 0.0, "w_volume": 0.0,
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
        "symbol", "w_trend", "w_elliott", "w_volume",
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

    weight_combos = generate_weight_combos(step=0.1)

    print("=" * 130)
    print("WEIGHTS SWEEP — 3 ANALYZERS (step 0.1)")
    print(f"Symbols:   {SYMBOLS}")
    print(f"Analyzers: {ANALYZERS}")
    print(f"Combos:    {len(weight_combos)} per symbol")
    print(f"Total:     {len(SYMBOLS) * len(weight_combos)} tasks")
    print("=" * 130)

    print("\nPreloading data in main process...")
    fetcher = MarketDataFetcher()
    data_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, str]] = {}

    for sym in SYMBOLS:
        params = BASE_PARAMS[sym]
        timeframe = params["timeframe"]
        try:
            df_1h = fetcher.fetch(sym, "1h", limit=BARS_1H)
            df_1d = resample_ohlcv(df_1h, "1D")

            if timeframe == "1h":
                df = df_1h
            else:
                rule_map = {"2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
                df = resample_ohlcv(df_1h, rule_map[timeframe])

            data_cache[sym] = (df, df_1d, timeframe)
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
        df, df_1d, tf = data_cache[sym]
        for w in weight_combos:
            tasks.append({
                "data": df,
                "df_1d": df_1d,
                "symbol": sym,
                "weights": w,
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
            if done % 50 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)}")

    valid = [r for r in results if not r.get("error")]

    print("\n" + "=" * 130)
    print("BEST WEIGHTS PER SYMBOL (by PnL)")
    print("=" * 130)
    print(
        f"{'symbol':<10} {'trend':>7} {'elliott':>8} {'volume':>7} "
        f"{'trades':>7} {'WR%':>7} {'PF':>6} {'PnL$':>10} "
        f"{'Sharpe':>8} {'MaxDD%':>8}"
    )
    print("-" * 100)

    best_per_symbol = {}

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue

        best = max(sym_results, key=lambda x: x["total_pnl"])
        best_per_symbol[sym] = best

        print(
            f"{sym:<10} "
            f"{best['w_trend']:>7.1f} "
            f"{best['w_elliott']:>8.1f} "
            f"{best['w_volume']:>7.1f} "
            f"{best['trades']:>7} "
            f"{best['win_rate'] * 100:>7.1f} "
            f"{best['profit_factor']:>6.2f} "
            f"{best['total_pnl']:>+10.2f} "
            f"{best['sharpe']:>+8.3f} "
            f"{best['max_dd_pct'] * 100:>8.2f}"
        )

    total_pnl = sum(r["total_pnl"] for r in best_per_symbol.values())
    print(f"\n  TOTAL PnL: ${total_pnl:+.2f}")

    print("\n" + "=" * 130)
    print("BEST WEIGHTS PER SYMBOL (by Sharpe, min 100 trades)")
    print("=" * 130)
    print(
        f"{'symbol':<10} {'trend':>7} {'elliott':>8} {'volume':>7} "
        f"{'trades':>7} {'WR%':>7} {'PF':>6} {'PnL$':>10} {'Sharpe':>8}"
    )
    print("-" * 100)

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        candidates = [r for r in sym_results if r["trades"] >= 100]
        if not candidates:
            continue

        best = max(candidates, key=lambda x: x["sharpe"])
        print(
            f"{sym:<10} "
            f"{best['w_trend']:>7.1f} "
            f"{best['w_elliott']:>8.1f} "
            f"{best['w_volume']:>7.1f} "
            f"{best['trades']:>7} "
            f"{best['win_rate'] * 100:>7.1f} "
            f"{best['profit_factor']:>6.2f} "
            f"{best['total_pnl']:>+10.2f} "
            f"{best['sharpe']:>+8.3f}"
        )

    print("\n" + "=" * 130)
    print("TOP-10 WEIGHTS FOR EACH SYMBOL (by PnL)")
    print("=" * 130)

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue

        print(f"\n{sym}")
        print(
            f"  {'trend':>7} {'elliott':>8} {'volume':>7} "
            f"{'trades':>7} {'WR%':>7} {'PF':>6} "
            f"{'PnL$':>10} {'Sharpe':>8}"
        )

        top10 = sorted(sym_results, key=lambda x: x["total_pnl"], reverse=True)[:10]
        for r in top10:
            print(
                f"  {r['w_trend']:>7.1f} "
                f"{r['w_elliott']:>8.1f} "
                f"{r['w_volume']:>7.1f} "
                f"{r['trades']:>7} "
                f"{r['win_rate'] * 100:>7.1f} "
                f"{r['profit_factor']:>6.2f} "
                f"{r['total_pnl']:>+10.2f} "
                f"{r['sharpe']:>+8.3f}"
            )

    print("\n" + "=" * 130)
    print("STATS BY ANALYZER WEIGHT")
    print("=" * 130)

    analyzer_keys = [
        ("trend", "w_trend"),
        ("elliott_wave", "w_elliott"),
        ("volume", "w_volume"),
    ]

    for analyzer_name, w_key in analyzer_keys:
        print(f"\n{analyzer_name}:")
        print(
            f"  {'weight':>8} {'combos':>8} {'avg_pnl':>12} "
            f"{'best_pnl':>12} {'profitable':>12}"
        )

        for w_val in sorted(set(r[w_key] for r in valid)):
            w_results = [r for r in valid if r[w_key] == w_val]
            if not w_results:
                continue

            avg_pnl = sum(r["total_pnl"] for r in w_results) / len(w_results)
            best_pnl = max(r["total_pnl"] for r in w_results)
            profitable = sum(1 for r in w_results if r["total_pnl"] > 0)

            print(
                f"  {w_val:>8.1f} {len(w_results):>8} "
                f"{avg_pnl:>+12.2f} {best_pnl:>+12.2f} "
                f"{profitable:>5}/{len(w_results):<6}"
            )

    csv_path = export_path / "weights_sweep_3analyzer.csv"
    save_csv(results, csv_path)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()