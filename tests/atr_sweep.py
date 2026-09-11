"""Параметрический тест ATR multiplier для оптимизации SL."""

import asyncio
import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")  # минимум логов


def calculate_sharpe(returns: list, risk_free: float = 0.0) -> float:
    """Sharpe Ratio."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean - risk_free) / std * math.sqrt(252)


def calculate_sortino(returns: list, risk_free: float = 0.0) -> float:
    """Sortino Ratio."""
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


def calculate_max_drawdown(equity_curve: list) -> tuple:
    """Max Drawdown."""
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


async def run_single_test(
    symbol: str,
    timeframe: str,
    bars_to_process: int,
    atr_multiplier: float,
    data,  # заранее загруженные данные
    starting_equity: float = 10000.0,
    export_dir: str = "backtest_results/atr_sweep",
):
    """Запускает один тест с заданным ATR multiplier."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=f"{export_dir}/temp_{atr_multiplier}.db",
        gate_mode="balanced",
        snapshot_every_n_bars=999999,  # отключаем снимки для скорости
        atr_multiplier=atr_multiplier,
    )

    engine = TradingEngine(config)

    start_idx = max(250, len(data) - bars_to_process)
    total_bars = len(data) - start_idx

    processed = 0
    opened_count = 0
    closed_count = 0
    close_reason_counter = Counter()
    equity_curve = []

    for i in range(start_idx, len(data)):
        slice_data = data.iloc[: i + 1].copy()
        bar_time = (
            data.index[i].to_pydatetime()
            if hasattr(data.index[i], "to_pydatetime")
            else datetime.now(timezone.utc)
        )

        result = await engine.on_bar(
            data=slice_data, bar_index=i, bar_time=bar_time,
        )

        processed += 1
        if result.opened_position is not None:
            opened_count += 1
        for closed_pos in result.closed_positions:
            closed_count += 1
            if closed_pos.close_reason is not None:
                close_reason_counter[closed_pos.close_reason.value] += 1

        snapshot = engine.portfolio.get_snapshot()
        equity_curve.append(snapshot["current_equity"])

    # Финальные метрики
    stats = engine.get_stats()
    portfolio = stats["portfolio"]
    trades = stats["trades"]

    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            returns.append(ret)

    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_dd_pct, max_dd_dollar = calculate_max_drawdown(equity_curve)

    expectancy = 0.0
    if trades["total"] > 0:
        expectancy = (
            trades["win_rate"] * trades["avg_win"]
            - (1 - trades["win_rate"]) * trades["avg_loss"]
        )

    engine.close()

    # Удаляем временную БД
    try:
        Path(f"{export_dir}/temp_{atr_multiplier}.db").unlink()
    except Exception:
        pass

    return {
        "atr_multiplier": atr_multiplier,
        "bars": processed,
        "opened": opened_count,
        "closed": closed_count,
        "trades": trades["total"],
        "wins": trades["wins"],
        "losses": trades["losses"],
        "win_rate": trades["win_rate"],
        "profit_factor": trades["profit_factor"],
        "total_pnl": portfolio["total_pnl"],
        "total_pnl_pct": portfolio["total_pnl_pct"],
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd_pct,
        "expectancy": expectancy,
        "avg_win": trades["avg_win"],
        "avg_loss": trades["avg_loss"],
        "close_reasons": dict(close_reason_counter),
    }


async def main():
    """Запускает ATR sweep на BTC-USD."""
    symbol = "BTC-USD"
    timeframe = "1h"
    bars = 5000

    # Загружаем данные один раз
    print(f"Загрузка данных {symbol} {timeframe}...")
    fetcher = MarketDataFetcher()
    data = fetcher.fetch(symbol, timeframe, limit=bars + 250)
    print(f"Загружено {len(data)} баров")

    # Список multipliers для теста
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    results = []
    for mult in multipliers:
        print(f"\n{'=' * 70}")
        print(f"ТЕСТ: atr_multiplier = {mult}")
        print('=' * 70)

        result = await run_single_test(
            symbol=symbol,
            timeframe=timeframe,
            bars_to_process=bars,
            atr_multiplier=mult,
            data=data,
        )
        results.append(result)

        print(f"  Сделок:           {result['trades']}")
        print(f"  Win rate:         {result['win_rate'] * 100:.1f}%")
        print(f"  Profit factor:    {result['profit_factor']:.2f}")
        print(f"  Expectancy:       ${result['expectancy']:.2f}")
        print(f"  P&L:              ${result['total_pnl']:.2f} "
              f"({result['total_pnl_pct'] * 100:.2f}%)")
        print(f"  Sharpe:           {result['sharpe']:.2f}")
        print(f"  Max DD:           {result['max_dd_pct'] * 100:.2f}%")
        print(f"  Причины: {result['close_reasons']}")

    # Сводная таблица
    print("\n" + "=" * 110)
    print("СВОДНАЯ ТАБЛИЦА ATR SWEEP (BTC-USD 1h, 5000 баров)")
    print("=" * 110)
    header = (
        f"{'ATR x':>6} {'Trades':>7} {'Win%':>6} {'PF':>6} "
        f"{'Exp$':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} "
        f"{'PnL%':>7} {'SL':>4} {'TP':>4}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        sl = r["close_reasons"].get("stop_loss", 0)
        tp = r["close_reasons"].get("take_profit", 0)
        print(
            f"{r['atr_multiplier']:>6.1f} "
            f"{r['trades']:>7} "
            f"{r['win_rate'] * 100:>6.1f} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['expectancy']:>8.2f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['sortino']:>8.2f} "
            f"{r['max_dd_pct'] * 100:>7.2f} "
            f"{r['total_pnl_pct'] * 100:>7.2f} "
            f"{sl:>4} "
            f"{tp:>4}"
        )

    # Находим лучший по expectancy
    best = max(results, key=lambda x: x["expectancy"])
    print(f"\n✅ Лучший ATR multiplier по expectancy: {best['atr_multiplier']}")
    print(f"   Expectancy: ${best['expectancy']:.2f}")
    print(f"   Win rate:   {best['win_rate'] * 100:.1f}%")
    print(f"   PF:         {best['profit_factor']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())