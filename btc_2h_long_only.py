"""BTC 2h: Long-Only (только BUY) walk-forward.

Проверяет гипотезу: BUY работает во всех фолдах, SELL — нет.
Если Long-Only даёт 4/4 — внедряем для BTC.

Запуск:
    python btc_2h_long_only.py
"""

import asyncio
import csv
import math
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine
from src.engine.symbol_profiles import SymbolProfile, SymbolProfileRegistry
from src.models import SignalDirection


setup_logging(level="ERROR")


SYMBOL = "BTC-USD"
BARS_1H = 40000
WARMUP_BARS = 250
TEST_BARS = 1500
N_FOLDS = 4

STARTING_EQUITY = 10000.0
EXPORT_DIR = "backtest_results/btc_2h_long_only"

ATR_MULT = 2.0
RR_RATIO = 3.0
FILTER_CHOP = True

BASE_WEIGHTS = {
    "trend": 0.35,
    "elliott_wave": 0.30,
    "volatility": 0.20,
    "volume": 0.15,
}

DAILY_LOSS_LIMIT = 0.10
DAILY_PROFIT_TARGET = 0.50
MAX_DRAWDOWN = 0.50
PAUSE_AFTER_LOSSES = 10
PAUSE_DURATION = 1


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


class LongOnlyEngine(TradingEngine):
    """TradingEngine с фильтром: только BUY."""

    def __init__(self, config):
        super().__init__(config)
        self._long_only = True

    async def on_bar(self, data, bar_index, bar_time=None):
        result = await super().on_bar(data, bar_index, bar_time)
        return result

    def _try_open_position(self, decision, data, current_price, bar_time):
        if self._long_only and decision.direction == SignalDirection.SELL:
            return None
        return super()._try_open_position(
            decision, data, current_price, bar_time,
        )


async def run_fold(
    data: pd.DataFrame,
    fold_id: int,
    start_idx: int,
    end_idx: int,
    export_dir: str,
    long_only: bool,
) -> Dict[str, Any]:
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    tag = "long" if long_only else "both"
    db_path = export_path / f"tmp_fold_{fold_id}_{tag}.db"

    registry = SymbolProfileRegistry()
    profile = SymbolProfile(
        risk_per_trade_pct=0.01,
        max_position_pct=1.0,
        atr_multiplier=ATR_MULT,
        atr_period=14,
        default_rr_ratio=RR_RATIO,
        filter_chop=FILTER_CHOP,
        weights_override=dict(BASE_WEIGHTS),
        notes=f"BTC 2h fold {fold_id} {tag}",
    )
    registry.register(SYMBOL, profile)
    registry.set_default(profile)

    config = EngineConfig(
        starting_equity=STARTING_EQUITY,
        symbol=SYMBOL,
        timeframe="2h",
        journal_db_path=str(db_path),
        gate_mode="aggressive",
        snapshot_every_n_bars=999999,
        use_symbol_profiles=True,
        symbol_profiles=registry,
        daily_loss_limit_pct=DAILY_LOSS_LIMIT,
        daily_profit_target_pct=DAILY_PROFIT_TARGET,
        max_drawdown_pct=MAX_DRAWDOWN,
        pause_after_consecutive_losses=PAUSE_AFTER_LOSSES,
        pause_duration_hours=PAUSE_DURATION,
    )

    if long_only:
        engine = LongOnlyEngine(config)
    else:
        engine = TradingEngine(config)

    equity_curve: List[float] = []
    trades_log: List[Dict[str, Any]] = []

    try:
        for i in range(start_idx, end_idx):
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

    start_date = data.index[start_idx] if start_idx < len(data) else ""
    end_date = data.index[min(end_idx - 1, len(data) - 1)] if end_idx > 0 else ""

    return {
        "fold": fold_id,
        "mode": tag,
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
        "buy_trades": len(buy_pnls),
        "buy_pnl": sum(buy_pnls) if buy_pnls else 0.0,
        "sell_trades": len(sell_pnls),
        "sell_pnl": sum(sell_pnls) if sell_pnls else 0.0,
        "error": "",
    }


def run_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return asyncio.run(run_fold(**args))
    except Exception as e:
        return {
            "fold": args.get("fold_id", 0),
            "mode": "long" if args.get("long_only") else "both",
            "start_date": "", "end_date": "",
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
        "fold", "mode", "start_date", "end_date",
        "trades", "wins", "losses", "win_rate", "profit_factor",
        "total_pnl", "total_pnl_pct", "sharpe", "max_dd_pct",
        "buy_trades", "buy_pnl", "sell_trades", "sell_pnl", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("BTC 2h LONG-ONLY vs BOTH")
    print(f"Symbol:      {SYMBOL}")
    print(f"Folds:       {N_FOLDS}, test {TEST_BARS} bars each")
    print(f"ATR x {ATR_MULT}, RR {RR_RATIO}, filter_chop={FILTER_CHOP}")
    print("=" * 120)

    print("\nLoading 1h data...")
    fetcher = MarketDataFetcher()
    df_1h = fetcher.fetch(SYMBOL, "1h", limit=BARS_1H)
    df_2h = resample_ohlcv(df_1h, "2h")
    print(f"  2h: {len(df_2h)} bars")
    print(f"  range: {df_2h.index[0]} .. {df_2h.index[-1]}")

    n = len(df_2h)

    folds: List[Dict[str, Any]] = []
    for fold_id in range(N_FOLDS):
        test_end = n - (N_FOLDS - 1 - fold_id) * TEST_BARS
        test_start = test_end - TEST_BARS
        if test_start < WARMUP_BARS or test_end > n:
            continue
        folds.append({
            "fold_id": fold_id,
            "start_idx": test_start,
            "end_idx": test_end,
        })

    tasks = []
    for f in folds:
        for long_only in [True, False]:
            tasks.append({
                "data": df_2h,
                "fold_id": f["fold_id"],
                "start_idx": f["start_idx"],
                "end_idx": f["end_idx"],
                "export_dir": EXPORT_DIR,
                "long_only": long_only,
            })

    n_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"\nRunning {len(tasks)} tasks with {n_workers} workers...\n")

    results: List[Dict[str, Any]] = []
    with Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(run_worker, tasks):
            results.append(res)
            if res.get("error"):
                print(f"  FAIL fold {res['fold']} {res['mode']}: {res['error']}")
            else:
                print(
                    f"  fold {res['fold']} {res['mode']:<5}: "
                    f"trades={res['trades']:>3} "
                    f"WR={res['win_rate'] * 100:>5.1f}% "
                    f"PF={res['profit_factor']:>5.2f} "
                    f"PnL=${res['total_pnl']:>+9.2f} "
                    f"Sharpe={res['sharpe']:>+6.3f}"
                )

    results.sort(key=lambda r: (r["fold"], r["mode"]))

    for mode in ["long", "both"]:
        mode_results = [
            r for r in results if r["mode"] == mode and not r.get("error")
        ]
        if not mode_results:
            continue

        print("\n" + "=" * 120)
        print(f"MODE: {mode.upper()}")
        print("=" * 120)
        print(
            f"{'fold':>5} {'period':>28} {'trades':>7} {'WR%':>7} "
            f"{'PF':>6} {'PnL$':>10} {'Sharpe':>8} {'MaxDD%':>8}"
        )
        print("-" * 100)

        for r in mode_results:
            period = f"{r['start_date'][:10]}..{r['end_date'][:10]}"
            print(
                f"{r['fold']:>5} {period:>28} "
                f"{r['trades']:>7} "
                f"{r['win_rate'] * 100:>7.1f} "
                f"{r['profit_factor']:>6.2f} "
                f"{r['total_pnl']:>+10.2f} "
                f"{r['sharpe']:>+8.3f} "
                f"{r['max_dd_pct'] * 100:>8.2f}"
            )

        profitable = sum(1 for r in mode_results if r["total_pnl"] > 0)
        total_pnl = sum(r["total_pnl"] for r in mode_results)
        avg_pnl = total_pnl / len(mode_results)
        avg_sharpe = sum(r["sharpe"] for r in mode_results) / len(mode_results)
        avg_maxdd = sum(r["max_dd_pct"] for r in mode_results) / len(mode_results)
        total_trades = sum(r["trades"] for r in mode_results)

        print(f"\n  Profitable:      {profitable}/{len(mode_results)}")
        print(f"  Total PnL:       ${total_pnl:+.2f}")
        print(f"  Avg PnL:         ${avg_pnl:+.2f}")
        print(f"  Avg Sharpe:      {avg_sharpe:+.3f}")
        print(f"  Avg MaxDD:       {avg_maxdd * 100:.2f}%")
        print(f"  Total trades:    {total_trades}")

    print("\n" + "=" * 120)
    print("COMPARISON: LONG-ONLY vs BOTH")
    print("=" * 120)

    for fold_id in range(N_FOLDS):
        long_r = next(
            (r for r in results
             if r["fold"] == fold_id and r["mode"] == "long"
             and not r.get("error")),
            None,
        )
        both_r = next(
            (r for r in results
             if r["fold"] == fold_id and r["mode"] == "both"
             and not r.get("error")),
            None,
        )
        if long_r is None or both_r is None:
            continue

        delta = long_r["total_pnl"] - both_r["total_pnl"]
        print(
            f"  fold {fold_id}: "
            f"long=${long_r['total_pnl']:>+8.2f} "
            f"both=${both_r['total_pnl']:>+8.2f} "
            f"delta=${delta:>+8.2f}"
        )

    csv_path = export_path / "long_only_results.csv"
    save_csv(results, csv_path)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()