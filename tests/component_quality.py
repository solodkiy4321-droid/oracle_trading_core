"""Диагностика качества отдельных компонентов анализаторов."""

import asyncio
import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")


def correlation(x: List[float], y: List[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def verdict_ic(ic: float) -> str:
    if ic > 0.15:
        return "✅✅ отлично"
    elif ic > 0.10:
        return "✅ работает"
    elif ic > 0.05:
        return "⚠️  слабо работает"
    elif ic > -0.05:
        return "❌ шум"
    else:
        return "🚨 ВРЕДИТ"


# ============ Парсинг reasons ============

def parse_analyzer_from_reason(reason: str) -> Optional[dict]:
    """Парсит reason. Формат: 'indicators: BUY (conf=0.62, ...) — детали'."""
    if ":" not in reason:
        return None

    analyzer_name = reason.split(":")[0].strip()
    if analyzer_name not in ("indicators", "harmonic", "support_resistance"):
        return None

    result = {
        "analyzer": analyzer_name,
        "direction": None,
        "confidence": None,
        "pattern_type": None,
    }

    if "BUY" in reason:
        result["direction"] = "BUY"
    elif "SELL" in reason:
        result["direction"] = "SELL"

    if "conf=" in reason:
        try:
            result["confidence"] = float(reason.split("conf=")[1].split(",")[0])
        except (ValueError, IndexError):
            pass

    # Паттерн для harmonic (если есть)
    for pattern in ["Gartley", "Bat", "Butterfly", "Crab"]:
        if pattern.lower() in reason.lower():
            result["pattern_type"] = pattern
            break

    return result


def extract_all_signals(decision_reasons: List[str]) -> dict:
    """Извлекает сигналы всех анализаторов."""
    result = {
        "indicators": None,
        "harmonic": None,
        "support_resistance": None,
    }
    for reason in decision_reasons:
        parsed = parse_analyzer_from_reason(reason)
        if parsed is None:
            continue
        analyzer = parsed["analyzer"]
        if result[analyzer] is None:
            result[analyzer] = parsed
    return result


# ============ Сбор данных ============

async def collect_detailed_trades(
    symbol: str,
    timeframe: str = "1h",
    bars_to_process: int = 5000,
    export_dir: str = "backtest_results/component_quality",
) -> List[dict]:
    """Собирает данные по каждой сделке. Удаляет старую БД."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    # УДАЛЯЕМ старую БД для гарантии свежести данных
    db_path = export_path / f"comp_{symbol.replace('-', '_')}.db"
    if db_path.exists():
        try:
            db_path.unlink()
            print(f"  Удалена старая БД: {db_path.name}")
        except Exception as e:
            print(f"  ⚠️  Не удалось удалить {db_path}: {e}")

    fetcher = MarketDataFetcher()
    try:
        data = fetcher.fetch(symbol, timeframe, limit=bars_to_process + 250)
    except Exception as e:
        print(f"❌ Ошибка загрузки {symbol}: {e}")
        return []

    if len(data) < 500:
        return []

    if len(data) < bars_to_process:
        bars_to_process = len(data)

    config = EngineConfig(
        starting_equity=10000.0,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=str(db_path),
        gate_mode="balanced",
        snapshot_every_n_bars=999999,
    )

    engine = TradingEngine(config)
    start_idx = max(250, len(data) - bars_to_process)
    trades = []

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

        if result.opened_position is not None and result.decision is not None:
            decision = result.decision
            pos = result.opened_position
            signals = extract_all_signals(decision.reasons)

            record = {
                "position_id": pos.id,
                "symbol": symbol,
                "direction": decision.direction.name,
                "regime": decision.regime.value,
                "score": decision.confluence_score,
                "entry_price": pos.entry_price,
                "exit_price": None,
                "exit_reason": None,
                "pnl": None,
                "indicators_dir": signals["indicators"]["direction"] if signals["indicators"] else None,
                "indicators_conf": signals["indicators"]["confidence"] if signals["indicators"] else None,
                "harmonic_dir": signals["harmonic"]["direction"] if signals["harmonic"] else None,
                "harmonic_conf": signals["harmonic"]["confidence"] if signals["harmonic"] else None,
                "harmonic_pattern": signals["harmonic"]["pattern_type"] if signals["harmonic"] else None,
                "sr_dir": signals["support_resistance"]["direction"] if signals["support_resistance"] else None,
                "sr_conf": signals["support_resistance"]["confidence"] if signals["support_resistance"] else None,
            }
            trades.append(record)

        for closed in result.closed_positions:
            for t in trades:
                if t["position_id"] == closed.id:
                    t["exit_price"] = closed.close_price
                    t["exit_reason"] = (
                        closed.close_reason.value
                        if closed.close_reason else "unknown"
                    )
                    t["pnl"] = closed.realized_pnl
                    break

    engine.close()
    return trades


# ============ Анализ ============

def analyze_analyzer(trades: List[dict], analyzer_name: str):
    dir_key = f"{analyzer_name}_dir"
    conf_key = f"{analyzer_name}_conf"

    relevant = [
        t for t in trades
        if t.get(dir_key) is not None and t.get("pnl") is not None
    ]

    print(f"\n{'=' * 70}")
    print(f"{analyzer_name.upper()}")
    print(f"{'=' * 70}")

    if len(relevant) < 3:
        print(f"  Недостаточно данных: {len(relevant)}")
        return

    print(f"  Сделок: {len(relevant)}")

    match = [t for t in relevant if t[dir_key] == t["direction"]]
    mismatch = [t for t in relevant if t[dir_key] != t["direction"]]

    def stats(cases):
        if not cases:
            return (0, 0.0, 0.0)
        wins = sum(1 for t in cases if t["pnl"] > 0)
        return (len(cases), wins / len(cases), sum(t["pnl"] for t in cases) / len(cases))

    n_match, wr_match, pnl_match = stats(match)
    n_mismatch, wr_mismatch, pnl_mismatch = stats(mismatch)

    if n_match > 0:
        print(f"  Совпал: n={n_match}, WR={wr_match*100:.1f}%, avg=${pnl_match:+.2f}")
    if n_mismatch > 0:
        print(f"  Против: n={n_mismatch}, WR={wr_mismatch*100:.1f}%, avg=${pnl_mismatch:+.2f}")

    ic_proxy = wr_match - wr_mismatch if (n_match > 0 and n_mismatch > 0) else 0.0
    print(f"  IC proxy: {ic_proxy:+.3f} {verdict_ic(ic_proxy)}")

    confs = [t[conf_key] for t in relevant if t.get(conf_key) is not None]
    pnls = [t["pnl"] for t in relevant if t.get(conf_key) is not None]
    if len(confs) >= 3 and len(set(confs)) > 1:
        ic = correlation(confs, pnls)
        print(f"  IC (confidence vs PnL): {ic:+.3f} {verdict_ic(ic)}")

        print(f"\n  Win rate по confidence:")
        quantiles = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        for low, high in quantiles:
            subset = [(t[conf_key], t["pnl"]) for t in relevant
                      if t.get(conf_key) is not None and low <= t[conf_key] < high]
            if not subset:
                continue
            wins = sum(1 for _, p in subset if p > 0)
            win_rate = wins / len(subset)
            avg_pnl = sum(p for _, p in subset) / len(subset)
            print(
                f"    conf {low:.1f}-{high:.1f}: n={len(subset):3d}, "
                f"WR={win_rate*100:5.1f}%, avg=${avg_pnl:+7.2f}"
            )


def analyze_harmonic_patterns(trades: List[dict]):
    print(f"\n{'=' * 70}")
    print("HARMONIC: ПО ПАТТЕРНАМ")
    print(f"{'=' * 70}")

    for pattern_name in ["Gartley", "Bat", "Butterfly", "Crab"]:
        relevant = [
            t for t in trades
            if t.get("harmonic_pattern") == pattern_name
            and t.get("pnl") is not None
        ]
        if len(relevant) < 2:
            print(f"\n  {pattern_name}: недостаточно данных ({len(relevant)})")
            continue
        wins = sum(1 for t in relevant if t["pnl"] > 0)
        win_rate = wins / len(relevant)
        avg_pnl = sum(t["pnl"] for t in relevant) / len(relevant)
        total_pnl = sum(t["pnl"] for t in relevant)
        print(f"\n  {pattern_name}:")
        print(f"    n={len(relevant)}, WR={win_rate*100:.1f}%, "
              f"avg=${avg_pnl:+.2f}, total=${total_pnl:+.2f}")


async def main():
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
    ]

    all_trades = {}
    for symbol, timeframe in symbols:
        print(f"\n{'=' * 70}")
        print(f"СБОР ДАННЫХ: {symbol} {timeframe}")
        print(f"{'=' * 70}")
        trades = await collect_detailed_trades(
            symbol=symbol, timeframe=timeframe, bars_to_process=5000,
        )
        all_trades[symbol] = trades
        print(f"  Собрано сделок: {len(trades)}")

    for symbol, trades in all_trades.items():
        if not trades:
            continue

        print(f"\n\n{'#' * 70}")
        print(f"# АНАЛИЗ: {symbol}")
        print(f"{'#' * 70}")

        completed = [t for t in trades if t.get("pnl") is not None]
        wins = [t for t in completed if t["pnl"] > 0]
        losses = [t for t in completed if t["pnl"] < 0]
        total_pnl = sum(t["pnl"] for t in completed)

        print(f"\nВсего: {len(completed)}, Побед: {len(wins)}, Убытков: {len(losses)}")
        if completed:
            print(f"Win rate: {len(wins) / len(completed) * 100:.1f}%")
        print(f"PnL: ${total_pnl:+.2f}")

        analyze_analyzer(trades, "indicators")
        analyze_analyzer(trades, "harmonic")
        analyze_analyzer(trades, "support_resistance")
        analyze_harmonic_patterns(trades)

    for symbol, trades in all_trades.items():
        if not trades:
            continue
        export_path = Path("backtest_results/component_quality")
        export_path.mkdir(parents=True, exist_ok=True)
        csv_file = export_path / f"components_{symbol.replace('-', '_')}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        print(f"\nДанные {symbol}: {csv_file}")


if __name__ == "__main__":
    asyncio.run(main())