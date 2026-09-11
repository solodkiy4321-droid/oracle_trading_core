"""Точка входа: расширенный бэктест, 8 инструментов (только прибыльные)."""

import asyncio
import math
from datetime import datetime, timezone
from pathlib import Path

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="WARNING")


def calculate_sharpe(returns: list, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean - risk_free) / std * math.sqrt(252)


def calculate_max_drawdown(equity_curve: list) -> tuple:
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


async def run_backtest(symbol, timeframe="1h", bars_to_process=10000,
                       export_dir="backtest_results", starting_equity=10000.0):
    print(f"\n{'=' * 70}")
    print(f"БЭКТЕСТИНГ: {symbol} {timeframe} | {bars_to_process} баров")
    print("=" * 70)

    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"{symbol.replace('-', '_')}_{timeframe}_{timestamp}"

    fetcher = MarketDataFetcher()
    try:
        data = fetcher.fetch(symbol, timeframe, limit=bars_to_process + 250)
    except Exception as e:
        print(f"❌ Ошибка загрузки {symbol}: {e}")
        return None

    print(f"Загружено {len(data)} баров")
    if len(data) < 500:
        return None
    if len(data) < bars_to_process:
        bars_to_process = len(data)

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=f"{export_dir}/{prefix}_journal.db",
        gate_mode="balanced",
        snapshot_every_n_bars=999999,
    )

    engine = TradingEngine(config)
    profile = engine.profile
    print(f"Профиль: risk={profile.risk_per_trade_pct*100:.2f}%, "
          f"atr={profile.atr_multiplier:.2f}, notes={profile.notes}")

    start_idx = max(250, len(data) - bars_to_process)
    processed = 0
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

        snapshot = engine.portfolio.get_snapshot()
        equity_curve.append(snapshot["current_equity"])

        if processed % 2000 == 0:
            print(f"  [{processed}] equity=${snapshot['current_equity']:.2f}, "
                  f"сделок={snapshot['total_trades']}")

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
    max_dd_pct, _ = calculate_max_drawdown(equity_curve)

    expectancy = 0.0
    if trades["total"] > 0:
        expectancy = (
            trades["win_rate"] * trades["avg_win"]
            - (1 - trades["win_rate"]) * trades["avg_loss"]
        )

    print(f"  Результат: {trades['total']} сделок, "
          f"WR {trades['win_rate']*100:.1f}%, "
          f"PnL ${portfolio['total_pnl']:+.2f}")

    engine.close()

    return {
        "symbol": symbol,
        "bars": processed,
        "trades": trades["total"],
        "wins": trades["wins"],
        "losses": trades["losses"],
        "win_rate": trades["win_rate"],
        "profit_factor": trades["profit_factor"],
        "total_pnl": portfolio["total_pnl"],
        "total_pnl_pct": portfolio["total_pnl_pct"],
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "expectancy": expectancy,
    }


async def main():
    """Бэктест на 8 инструментах × 10000 баров."""
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
        ("ADA-USD", "1h"),
        ("AVAX-USD", "1h"),
        ("DOT-USD", "1h"),
        ("ATOM-USD", "1h"),
        ("UNI-USD", "1h"),
        ("AAPL", "1d"),
    ]

    bars_per_symbol = 10000
    starting_equity = 10000.0

    print("\n" + "=" * 70)
    print(f"РАСШИРЕННЫЙ БЭКТЕСТ: {len(symbols)} инструментов × {bars_per_symbol} баров")
    print("=" * 70)

    all_results = []
    for symbol, timeframe in symbols:
        result = await run_backtest(
            symbol=symbol, timeframe=timeframe,
            bars_to_process=bars_per_symbol,
            export_dir="backtest_results",
            starting_equity=starting_equity,
        )
        if result is not None:
            all_results.append(result)

    print("\n\n" + "=" * 100)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 100)
    header = (
        f"{'Symbol':<12} {'Trades':>7} {'Win%':>6} {'PF':>6} "
        f"{'Sharpe':>7} {'MaxDD%':>7} {'PnL%':>7} {'Exp$':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['symbol']:<12} "
            f"{r['trades']:>7} "
            f"{r['win_rate'] * 100:>6.1f} "
            f"{r['profit_factor']:>6.2f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_dd_pct'] * 100:>7.2f} "
            f"{r['total_pnl_pct'] * 100:>7.2f} "
            f"{r['expectancy']:>8.2f}"
        )

    if all_results:
        total_trades = sum(r["trades"] for r in all_results)
        total_wins = sum(r["wins"] for r in all_results)
        total_losses = sum(r["losses"] for r in all_results)
        total_pnl = sum(r["total_pnl"] for r in all_results)

        print("\n" + "-" * 70)
        print(f"ИТОГО:")
        print(f"  Сделок:         {total_trades}")
        print(f"  Побед/Убытков:  {total_wins} / {total_losses}")
        if total_trades > 0:
            print(f"  Общий win rate: {total_wins / total_trades * 100:.1f}%")
        print(f"  Общий P&L:      ${total_pnl:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())