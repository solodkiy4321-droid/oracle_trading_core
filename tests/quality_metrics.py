"""Диагностика качества анализаторов.

Запускает бэктест и собирает детальную статистику по каждому анализатору:
- Information Coefficient (IC) — корреляция между confidence и PnL
- Разбивка по квантилям confidence — win rate и avg PnL на разных уровнях
- Распределение confidence в победных vs убыточных сделках
- Эффективность по режимам рынка
- Эффективность по направлениям (BUY/SELL)
"""

import asyncio
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation между двумя списками."""
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
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


def extract_analyzer_confidence(decision, analyzer_name: str) -> Optional[float]:
    """Извлекает confidence конкретного анализатора из decision.reasons."""
    for reason in decision.reasons:
        if analyzer_name in reason and "conf=" in reason:
            try:
                return float(reason.split("conf=")[1].split(",")[0])
            except (ValueError, IndexError):
                pass
    return None


def extract_analyzer_direction(decision, analyzer_name: str) -> Optional[str]:
    """Извлекает направление конкретного анализатора из decision.reasons."""
    for reason in decision.reasons:
        if analyzer_name in reason:
            if "BUY" in reason:
                return "BUY"
            elif "SELL" in reason:
                return "SELL"
    return None


async def collect_trade_data(
    symbol: str,
    timeframe: str = "1h",
    bars_to_process: int = 5000,
    export_dir: str = "backtest_results/quality",
) -> List[dict]:
    """
    Запускает бэктест и собирает детальные данные по каждой сделке.

    Возвращает список записей с полями:
    - position_id, direction, regime, score
    - indicators_conf, indicators_dir
    - harmonic_conf, harmonic_dir
    - sr_conf, sr_dir
    - entry_price, exit_price, pnl, close_reason
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    fetcher = MarketDataFetcher()
    try:
        data = fetcher.fetch(symbol, timeframe, limit=bars_to_process + 250)
    except Exception as e:
        print(f"❌ Ошибка загрузки {symbol}: {e}")
        return []

    if len(data) < 500:
        print(f"⚠️  Недостаточно данных для {symbol}")
        return []

    if len(data) < bars_to_process:
        bars_to_process = len(data)

    config = EngineConfig(
        starting_equity=10000.0,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=f"{export_dir}/quality_{symbol.replace('-', '_')}.db",
        gate_mode="balanced",
        snapshot_every_n_bars=999999,
    )

    engine = TradingEngine(config)

    start_idx = max(250, len(data) - bars_to_process)
    trades_data = []

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

        # Записываем открытие позиции
        if result.opened_position is not None and result.decision is not None:
            decision = result.decision
            pos = result.opened_position

            trade_record = {
                "position_id": pos.id,
                "symbol": symbol,
                "direction": decision.direction.name,
                "regime": decision.regime.value,
                "score": decision.confluence_score,
                "indicators_conf": extract_analyzer_confidence(decision, "indicators"),
                "indicators_dir": extract_analyzer_direction(decision, "indicators"),
                "harmonic_conf": extract_analyzer_confidence(decision, "harmonic"),
                "harmonic_dir": extract_analyzer_direction(decision, "harmonic"),
                "sr_conf": extract_analyzer_confidence(decision, "support_resistance"),
                "sr_dir": extract_analyzer_direction(decision, "support_resistance"),
                "entry_price": pos.entry_price,
                "entry_time": bar_time.isoformat() if bar_time else "",
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None,
                "pnl": None,
                "pnl_pct": None,
            }
            trades_data.append(trade_record)

        # Записываем закрытие позиции
        for closed_pos in result.closed_positions:
            for t in trades_data:
                if t["position_id"] == closed_pos.id:
                    t["exit_price"] = closed_pos.close_price
                    t["exit_time"] = (
                        closed_pos.close_time.isoformat()
                        if closed_pos.close_time else ""
                    )
                    t["exit_reason"] = (
                        closed_pos.close_reason.value
                        if closed_pos.close_reason else "unknown"
                    )
                    t["pnl"] = closed_pos.realized_pnl
                    if closed_pos.entry_price > 0:
                        t["pnl_pct"] = closed_pos.realized_pnl / (
                            closed_pos.entry_price * closed_pos.initial_size
                        )
                    break

    engine.close()
    return trades_data


def analyze_analyzer_quality(
    trades_data: List[dict],
    analyzer_name: str,
    conf_key: str,
    dir_key: str,
):
    """Анализирует качество одного анализатора."""
    # Фильтруем сделки, где анализатор сработал
    relevant = [
        t for t in trades_data
        if t.get(conf_key) is not None and t.get("pnl") is not None
    ]

    if len(relevant) < 3:
        print(f"\n{analyzer_name.upper()}: недостаточно данных ({len(relevant)} сделок)")
        return

    confs = [t[conf_key] for t in relevant]
    pnls = [t["pnl"] for t in relevant]

    # ============ Information Coefficient ============
    ic = correlation(confs, pnls)

    print(f"\n{'=' * 70}")
    print(f"{analyzer_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Сделок с этим анализатором: {len(relevant)}")
    print(f"  Information Coefficient:    {ic:+.3f}")
    if ic > 0.10:
        verdict = "✅ работает"
    elif ic > 0.05:
        verdict = "⚠️  слабо работает"
    elif ic > -0.05:
        verdict = "❌ шум"
    else:
        verdict = "🚨 ВРЕДИТ (обратная корреляция)"
    print(f"  Вердикт:                    {verdict}")

    # ============ Разбивка по квантилям confidence ============
    print(f"\n  Win rate и avg PnL по уровням confidence:")
    quantiles = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for low, high in quantiles:
        subset = [
            (t[conf_key], t["pnl"]) for t in relevant
            if low <= t[conf_key] < high
        ]
        if not subset:
            continue
        wins = sum(1 for _, p in subset if p > 0)
        win_rate = wins / len(subset)
        avg_pnl = sum(p for _, p in subset) / len(subset)
        total_pnl = sum(p for _, p in subset)
        print(
            f"    conf {low:.1f}-{high:.1f}: "
            f"n={len(subset):3d}, "
            f"win_rate={win_rate * 100:5.1f}%, "
            f"avg=${avg_pnl:+7.2f}, "
            f"total=${total_pnl:+8.2f}"
        )

    # ============ Победные vs Убыточные ============
    wins = [t for t in relevant if t["pnl"] > 0]
    losses = [t for t in relevant if t["pnl"] < 0]

    print(f"\n  Сравнение победных и убыточных:")
    if wins:
        avg_conf_wins = sum(t[conf_key] for t in wins) / len(wins)
        print(f"    Победные:  n={len(wins):3d}, avg_conf={avg_conf_wins:.3f}")
    if losses:
        avg_conf_losses = sum(t[conf_key] for t in losses) / len(losses)
        print(f"    Убыточные: n={len(losses):3d}, avg_conf={avg_conf_losses:.3f}")

    if wins and losses:
        diff = avg_conf_wins - avg_conf_losses
        print(f"    Разница:   {diff:+.3f} "
              f"({'✅ good' if diff > 0.05 else '⚠️  weak' if diff > 0.02 else '❌ bad'})")

    # ============ Согласованность с направлением сделки ============
    agreement_cases = []
    for t in relevant:
        analyzer_dir = t.get(dir_key)
        trade_dir = t.get("direction")
        if analyzer_dir and trade_dir:
            agreement_cases.append(analyzer_dir == trade_dir)

    if agreement_cases:
        agree_pct = sum(agreement_cases) / len(agreement_cases) * 100
        print(f"\n  Согласованность с направлением сделки: {agree_pct:.1f}%")

    # ============ Эффективность по режимам ============
    regimes = defaultdict(list)
    for t in relevant:
        regimes[t["regime"]].append(t["pnl"])

    if len(regimes) > 1:
        print(f"\n  По режимам рынка:")
        for regime, pnls_r in sorted(regimes.items()):
            wins_r = sum(1 for p in pnls_r if p > 0)
            win_rate = wins_r / len(pnls_r)
            avg_pnl = sum(pnls_r) / len(pnls_r)
            print(
                f"    {regime:6s}: n={len(pnls_r):3d}, "
                f"win_rate={win_rate * 100:5.1f}%, "
                f"avg=${avg_pnl:+7.2f}"
            )

    # ============ Эффективность по направлению ============
    directions = defaultdict(list)
    for t in relevant:
        directions[t["direction"]].append(t["pnl"])

    if len(directions) > 1:
        print(f"\n  По направлению:")
        for direction, pnls_d in sorted(directions.items()):
            wins_d = sum(1 for p in pnls_d if p > 0)
            win_rate = wins_d / len(pnls_d)
            avg_pnl = sum(pnls_d) / len(pnls_d)
            print(
                f"    {direction:5s}: n={len(pnls_d):3d}, "
                f"win_rate={win_rate * 100:5.1f}%, "
                f"avg=${avg_pnl:+7.2f}"
            )


async def main():
    """Запускает диагностику качества по BTC и ETH."""
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
    ]

    all_trades = {}

    for symbol, timeframe in symbols:
        print(f"\n{'=' * 70}")
        print(f"СБОР ДАННЫХ: {symbol} {timeframe}")
        print(f"{'=' * 70}")
        trades = await collect_trade_data(
            symbol=symbol,
            timeframe=timeframe,
            bars_to_process=5000,
        )
        all_trades[symbol] = trades
        print(f"  Собрано сделок: {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t.get("pnl", 0) and t["pnl"] > 0)
            losses = sum(1 for t in trades if t.get("pnl", 0) and t["pnl"] < 0)
            print(f"  Побед: {wins}, Убытков: {losses}")

    # ============ Анализ по каждому символу ============
    for symbol, trades in all_trades.items():
        if not trades:
            continue

        print(f"\n\n{'#' * 70}")
        print(f"# АНАЛИЗ КАЧЕСТВА: {symbol}")
        print(f"{'#' * 70}")

        # Общая статистика
        wins = [t for t in trades if t.get("pnl") and t["pnl"] > 0]
        losses = [t for t in trades if t.get("pnl") and t["pnl"] < 0]
        total_pnl = sum(t["pnl"] for t in trades if t.get("pnl"))

        print(f"\nВсего сделок:  {len(trades)}")
        print(f"Побед:         {len(wins)}")
        print(f"Убытков:       {len(losses)}")
        if trades:
            print(f"Win rate:      {len(wins) / len([t for t in trades if t.get('pnl')]) * 100:.1f}%")
        print(f"Общий PnL:     ${total_pnl:+.2f}")

        # Анализ confluence_score как метрики качества
        print(f"\n{'=' * 70}")
        print(f"CONFLUENCE_SCORE КАК ПРЕДИКТОР")
        print(f"{'=' * 70}")
        scored_trades = [t for t in trades if t.get("score") and t.get("pnl") is not None]
        if len(scored_trades) >= 3:
            scores = [t["score"] for t in scored_trades]
            pnls = [t["pnl"] for t in scored_trades]
            ic = correlation(scores, pnls)
            print(f"  Information Coefficient: {ic:+.3f}")

            print(f"\n  Win rate по уровням confluence_score:")
            quantiles = [(0.0, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
            for low, high in quantiles:
                subset = [(t["score"], t["pnl"]) for t in scored_trades if low <= t["score"] < high]
                if not subset:
                    continue
                wins_q = sum(1 for _, p in subset if p > 0)
                win_rate = wins_q / len(subset)
                avg_pnl = sum(p for _, p in subset) / len(subset)
                print(
                    f"    score {low:.2f}-{high:.2f}: "
                    f"n={len(subset):3d}, "
                    f"win_rate={win_rate * 100:5.1f}%, "
                    f"avg=${avg_pnl:+7.2f}"
                )

        # Анализ каждого анализатора
        analyze_analyzer_quality(
            trades, "indicators", "indicators_conf", "indicators_dir",
        )
        analyze_analyzer_quality(
            trades, "harmonic", "harmonic_conf", "harmonic_dir",
        )
        analyze_analyzer_quality(
            trades, "support_resistance", "sr_conf", "sr_dir",
        )

    # ============ Экспорт данных ============
    for symbol, trades in all_trades.items():
        if not trades:
            continue
        export_path = Path("backtest_results/quality")
        export_path.mkdir(parents=True, exist_ok=True)
        csv_file = export_path / f"quality_{symbol.replace('-', '_')}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        print(f"\nДанные {symbol} сохранены: {csv_file}")

    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА ЗАВЕРШЕНА")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())