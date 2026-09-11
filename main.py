"""Точка входа: расширенный бэктест на 5000 баров."""

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


def calculate_sortino(returns: list, risk_free: float = 0.0) -> float:
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


async def run_backtest(
    symbol: str,
    timeframe: str = "1h",
    bars_to_process: int = 5000,
    export_dir: str = "backtest_results",
    starting_equity: float = 10000.0,
):
    print("\n" + "=" * 70)
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
        print(f"⚠️  Недостаточно данных: {len(data)}")
        return None
    if len(data) < bars_to_process:
        bars_to_process = len(data)

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=f"{export_dir}/{prefix}_journal.db",
        gate_mode="balanced",
        snapshot_every_n_bars=24,
    )

    engine = TradingEngine(config)
    start_idx = max(250, len(data) - bars_to_process)
    total_bars = len(data) - start_idx

    processed = 0
    opened_count = 0
    closed_count = 0
    regime_counter = Counter()
    close_reason_counter = Counter()
    scores_above_065 = 0
    scores_above_060 = 0
    scores_above_050 = 0
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
        if result.decision is not None:
            regime_counter[result.decision.regime.value] += 1
            score = result.decision.confluence_score
            if score >= 0.65:
                scores_above_065 += 1
            elif score >= 0.60:
                scores_above_060 += 1
            if score >= 0.50:
                scores_above_050 += 1

        snapshot = engine.portfolio.get_snapshot()
        equity_curve.append(snapshot["current_equity"])

        if processed % 500 == 0:
            snapshot = engine.portfolio.get_snapshot()
            print(
                f"  [{processed}/{total_bars}] "
                f"equity=${snapshot['current_equity']:.2f}, "
                f"сделок={snapshot['total_trades']}"
            )

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

    print("\n" + "=" * 70)
    print(f"РЕЗУЛЬТАТЫ: {symbol}")
    print("=" * 70)
    print(f"Обработано баров:        {processed}")
    print(f"Открыто/Закрыто:         {opened_count} / {closed_count}")
    print()
    print(f"Начальный equity:        ${portfolio['starting_equity']:.2f}")
    print(f"Текущий equity:          ${portfolio['current_equity']:.2f}")
    print(f"Общий P&L:               ${portfolio['total_pnl']:.2f} "
          f"({portfolio['total_pnl_pct'] * 100:.2f}%)")
    print()
    print(f"Всего сделок:            {trades['total']}")
    print(f"Побед / Убытков:         {trades['wins']} / {trades['losses']}")
    print(f"Win rate:                {trades['win_rate'] * 100:.1f}%")
    print(f"Profit factor:           {trades['profit_factor']:.2f}")
    print(f"Средняя прибыль:         ${trades['avg_win']:.2f}")
    print(f"Средний убыток:          ${trades['avg_loss']:.2f}")
    print(f"Expectancy:              ${expectancy:.2f}")
    print()
    print(f"Sharpe Ratio:            {sharpe:.2f}")
    print(f"Sortino Ratio:           {sortino:.2f}")
    print(f"Max Drawdown:            {max_dd_pct * 100:.2f}% (${max_dd_dollar:.2f})")
    print()
    print(f"Сигналов >= 0.65:        {scores_above_065}")
    print(f"Сигналов >= 0.60:        {scores_above_060}")
    print(f"Сигналов >= 0.50:        {scores_above_050}")
    print()
    print("Причины закрытия:")
    for reason, count in close_reason_counter.most_common():
        print(f"  {reason:15s}: {count}")
    print()
    print("Режимы:")
    for regime, count in regime_counter.most_common():
        pct = count / processed * 100 if processed > 0 else 0
        print(f"  {regime:10s}: {count:5d} ({pct:5.1f}%)")

    equity_file = export_path / f"{prefix}_equity.csv"
    with open(equity_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bar", "equity"])
        for i, eq in enumerate(equity_curve):
            writer.writerow([i, eq])

    trades_file = export_path / f"{prefix}_trades.csv"
    all_trades = engine.analytics.get_all_trades()
    if all_trades:
        with open(trades_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_trades[0].keys())
            writer.writeheader()
            writer.writerows(all_trades)

    engine.close()

    return {
        "symbol": symbol,
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
        "signals_above_065": scores_above_065,
        "signals_above_060": scores_above_060,
        "signals_above_050": scores_above_050,
        "close_reasons": dict(close_reason_counter),
        "regimes": dict(regime_counter),
    }


async def main():
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
        ("AAPL", "1d"),
    ]

    bars_per_symbol = 5000
    starting_equity = 10000.0

    print("\n" + "=" * 70)
    print("РАСШИРЕННЫЙ БЭКТЕСТ: 3 инструмента x 5000 баров")
    print("=" * 70)

    all_results = []
    for symbol, timeframe in symbols:
        result = await run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            bars_to_process=bars_per_symbol,
            export_dir="backtest_results",
            starting_equity=starting_equity,
        )
        if result is not None:
            all_results.append(result)

    print("\n" + "=" * 70)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 70)
    header = (
        f"{'Symbol':<12} {'Trades':>7} {'Win%':>6} {'PF':>6} "
        f"{'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PnL%':>7} {'Exp$':>8}"
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
            f"{r['sortino']:>8.2f} "
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
        print(f"  Общий P&L:      ${total_pnl:.2f}")
        print(f"  Сигналов >= 0.65: {sum(r['signals_above_065'] for r in all_results)}")
        print(f"  Сигналов >= 0.60: {sum(r['signals_above_060'] for r in all_results)}")

    print("\n" + "=" * 70)
    print("Данные сохранены в backtest_results/")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())