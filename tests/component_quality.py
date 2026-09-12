"""Параллельная диагностика по режимам.

Использование multiprocessing.Pool:
- Каждая пара — в отдельном процессе
- Каждый процесс — своё ядро CPU
- Ожидаемое ускорение: 5-7x

Время: ~10-15 минут вместо 60.
"""

import asyncio
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")


def correlation(x, y):
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def verdict_ic(ic):
    if ic > 0.15: return "✅✅ отлично"
    if ic > 0.10: return "✅ работает"
    if ic > 0.05: return "⚠️  слабо"
    if ic > -0.05: return "❌ шум"
    return "🚨 ВРЕДИТ"


def parse_reason(reason):
    if ":" not in reason:
        return None
    name = reason.split(":")[0].strip()
    if name not in ("indicators", "harmonic", "support_resistance", "elliott_wave"):
        return None
    res = {"analyzer": name, "direction": None, "confidence": None, "pattern": None}
    if "BUY" in reason:
        res["direction"] = "BUY"
    elif "SELL" in reason:
        res["direction"] = "SELL"
    if "conf=" in reason:
        try:
            res["confidence"] = float(reason.split("conf=")[1].split(",")[0])
        except (ValueError, IndexError):
            pass
    for p in ["Gartley", "Bat", "Butterfly", "Crab"]:
        if p.lower() in reason.lower():
            res["pattern"] = p
            break
    return res


def extract_signals(reasons):
    result = {"indicators": None, "harmonic": None,
              "support_resistance": None, "elliott_wave": None}
    for r in reasons:
        parsed = parse_reason(r)
        if parsed:
            result[parsed["analyzer"]] = parsed
    return result


async def _collect_async(symbol: str, timeframe: str, bars: int, db_path: str) -> List[dict]:
    """Асинхронная сборка данных по одной паре."""
    fetcher = MarketDataFetcher()
    try:
        data = fetcher.fetch(symbol, timeframe, limit=bars + 250)
    except Exception as e:
        print(f"❌ {symbol}: {e}")
        return []

    if len(data) < 500:
        return []
    if len(data) < bars:
        bars = len(data)

    config = EngineConfig(
        starting_equity=10000.0,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=db_path,
        gate_mode="balanced",
        snapshot_every_n_bars=999999,
    )
    engine = TradingEngine(config)
    start_idx = max(250, len(data) - bars)
    trades = []

    for i in range(start_idx, len(data)):
        slice_data = data.iloc[: i + 1].copy()
        bar_time = (
            data.index[i].to_pydatetime()
            if hasattr(data.index[i], "to_pydatetime")
            else datetime.now(timezone.utc)
        )
        result = await engine.on_bar(data=slice_data, bar_index=i, bar_time=bar_time)
        if result.opened_position is not None and result.decision is not None:
            d = result.decision
            p = result.opened_position
            sig = extract_signals(d.reasons)
            rec = {
                "position_id": p.id, "symbol": symbol,
                "direction": d.direction.name,
                "regime": d.regime.value,
                "score": d.confluence_score,
                "entry_price": p.entry_price,
                "exit_price": None, "exit_reason": None, "pnl": None,
                "indicators_dir": sig["indicators"]["direction"] if sig["indicators"] else None,
                "indicators_conf": sig["indicators"]["confidence"] if sig["indicators"] else None,
                "harmonic_dir": sig["harmonic"]["direction"] if sig["harmonic"] else None,
                "harmonic_conf": sig["harmonic"]["confidence"] if sig["harmonic"] else None,
                "harmonic_pattern": sig["harmonic"]["pattern"] if sig["harmonic"] else None,
                "sr_dir": sig["support_resistance"]["direction"] if sig["support_resistance"] else None,
                "sr_conf": sig["support_resistance"]["confidence"] if sig["support_resistance"] else None,
                "elliott_dir": sig["elliott_wave"]["direction"] if sig["elliott_wave"] else None,
                "elliott_conf": sig["elliott_wave"]["confidence"] if sig["elliott_wave"] else None,
            }
            trades.append(rec)
        for closed in result.closed_positions:
            for t in trades:
                if t["position_id"] == closed.id:
                    t["exit_price"] = closed.close_price
                    t["exit_reason"] = (
                        closed.close_reason.value if closed.close_reason else "unknown"
                    )
                    t["pnl"] = closed.realized_pnl
                    break
    engine.close()
    return trades


def collect_worker(args: Tuple[str, str, int, str]) -> List[dict]:
    """
    Синхронная обёртка для multiprocessing.

    Запускает asyncio.run() внутри процесса.
    """
    symbol, timeframe, bars, db_path = args
    return asyncio.run(_collect_async(symbol, timeframe, bars, db_path))


def analyze_by_regime(trades, symbol):
    """Разбивка win rate по режимам для пары."""
    print(f"\n{'=' * 80}")
    print(f"{symbol}: WIN RATE ПО РЕЖИМАМ")
    print(f"{'=' * 80}")

    by_regime = defaultdict(list)
    for t in trades:
        if t.get("pnl") is not None:
            by_regime[t["regime"]].append(t["pnl"])

    if not by_regime:
        print("  Нет данных")
        return

    for regime in ["BULL", "BEAR", "CHOP"]:
        pnls = by_regime.get(regime, [])
        if not pnls:
            print(f"  {regime}: нет сделок")
            continue
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls)
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls)
        print(
            f"  {regime:6s}: n={len(pnls):3d}, "
            f"WR={wr * 100:5.1f}%, "
            f"P&L=${total_pnl:+8.2f}, "
            f"avg=${avg_pnl:+7.2f}"
        )


def analyze_analyzer_in_regime(trades, symbol, analyzer_name):
    """Разбивка анализатора по режимам."""
    dk = f"{analyzer_name}_dir"
    rel = [t for t in trades if t.get(dk) is not None and t.get("pnl") is not None]
    if len(rel) < 3:
        return

    print(f"\n  {analyzer_name.upper()} по режимам:")
    by_regime = defaultdict(list)
    for t in rel:
        match = (t[dk] == t["direction"])
        by_regime[t["regime"]].append((match, t["pnl"]))

    for regime in ["BULL", "BEAR", "CHOP"]:
        cases = by_regime.get(regime, [])
        if not cases:
            continue
        matches = [p for m, p in cases if m]
        mismatches = [p for m, p in cases if not m]
        n = len(cases)
        match_wr = sum(1 for p in matches if p > 0) / len(matches) if matches else 0.0
        mismatch_wr = sum(1 for p in mismatches if p > 0) / len(mismatches) if mismatches else 0.0
        ic = match_wr - mismatch_wr if (matches and mismatches) else 0.0
        print(
            f"    {regime:6s}: n={n:3d}, "
            f"WR(match)={match_wr * 100:5.1f}%, "
            f"WR(mismatch)={mismatch_wr * 100:5.1f}%, "
            f"IC={ic:+.3f} {verdict_ic(ic)}"
        )


def main():
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
        ("ADA-USD", "1h"),
        ("AVAX-USD", "1h"),
        ("DOT-USD", "1h"),
        ("ATOM-USD", "1h"),
        ("AAPL", "1d"),
    ]

    export_dir = Path("backtest_results/regime_analysis")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Параллельная обработка
    n_workers = min(len(symbols), cpu_count())
    print(f"\n{'=' * 80}")
    print(f"ПАРАЛЛЕЛЬНАЯ ДИАГНОСТИКА: {len(symbols)} пар, {n_workers} процессов")
    print(f"{'=' * 80}")

    tasks = [
        (sym, tf, 10000, str(export_dir / f"reg_{sym.replace('-', '_')}.db"))
        for sym, tf in symbols
    ]

    all_trades = []
    with Pool(processes=n_workers) as pool:
        results = pool.map(collect_worker, tasks)

    for trades in results:
        all_trades.extend(trades)

    print(f"\n\n{'#' * 80}")
    print(f"# АНАЛИЗ ПО РЕЖИМАМ (всего сделок: {len(all_trades)})")
    print(f"{'#' * 80}")

    completed = [t for t in all_trades if t.get("pnl") is not None]
    wins = [t for t in completed if t["pnl"] > 0]
    print(f"\nВсего: {len(completed)}, Побед: {len(wins)}, "
          f"WR: {len(wins) / len(completed) * 100:.1f}%")
    print(f"PnL: ${sum(t['pnl'] for t in completed):+.2f}")

    # Общая разбивка по режимам
    print(f"\n{'=' * 80}")
    print(f"ОБЩАЯ РАЗБИВКА ПО РЕЖИМАМ (все пары)")
    print(f"{'=' * 80}")
    analyze_by_regime(completed, "ALL")

    # Разбивка по каждой паре
    for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "AVAX-USD",
                   "DOT-USD", "ATOM-USD", "AAPL"]:
        symbol_trades = [t for t in completed if t["symbol"] == symbol]
        if symbol_trades:
            analyze_by_regime(symbol_trades, symbol)

    # Анализаторы по режимам
    print(f"\n\n{'#' * 80}")
    print(f"# АНАЛИЗАТОРЫ ПО РЕЖИМАМ")
    print(f"{'#' * 80}")

    for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "AVAX-USD",
                   "DOT-USD", "ATOM-USD", "AAPL"]:
        symbol_trades = [t for t in completed if t["symbol"] == symbol]
        if len(symbol_trades) < 3:
            continue
        print(f"\n{'=' * 80}")
        print(f"{symbol}")
        print(f"{'=' * 80}")
        for analyzer in ["indicators", "harmonic", "support_resistance", "elliott_wave"]:
            analyze_analyzer_in_regime(symbol_trades, symbol, analyzer)

    # Экспорт
    csv_file = export_dir / "all_trades_with_regime.csv"
    if all_trades:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_trades[0].keys())
            writer.writeheader()
            writer.writerows(all_trades)
        print(f"\nДанные: {csv_file}")


if __name__ == "__main__":
    main()