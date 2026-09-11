"""Диагностика сигналов: анализ прибыльных и убыточных сделок."""

import asyncio
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")


async def run_backtest_with_diagnostics(
    symbol: str,
    timeframe: str = "1h",
    bars_to_process: int = 5000,
    export_dir: str = "backtest_results/diagnostics",
    starting_equity: float = 10000.0,
):
    """Запускает бэктест и собирает детальную статистику по сделкам."""
    print(f"\n{'=' * 70}")
    print(f"ДИАГНОСТИКА: {symbol} {timeframe} | {bars_to_process} баров")
    print('=' * 70)

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    fetcher = MarketDataFetcher()
    data = fetcher.fetch(symbol, timeframe, limit=bars_to_process + 250)
    print(f"Загружено {len(data)} баров")

    config = EngineConfig(
        starting_equity=starting_equity,
        symbol=symbol,
        timeframe=timeframe,
        journal_db_path=f"{export_dir}/diag_{symbol.replace('-', '_')}.db",
        gate_mode="balanced",
        snapshot_every_n_bars=999999,
    )

    engine = TradingEngine(config)

    start_idx = max(250, len(data) - bars_to_process)

    # Собираем детальную информацию по каждой сделке
    trades_detailed = []

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

        # Записываем открытие позиции с контекстом
        if result.opened_position is not None and result.decision is not None:
            decision = result.decision
            pos = result.opened_position

            # Извлекаем сигналы из reasons
            indicators_conf = None
            harmonic_conf = None
            sr_conf = None
            indicators_dir = None
            harmonic_dir = None
            sr_dir = None

            for reason in decision.reasons:
                if "indicators" in reason and "conf=" in reason:
                    try:
                        indicators_conf = float(
                            reason.split("conf=")[1].split(",")[0]
                        )
                        if "BUY" in reason:
                            indicators_dir = "BUY"
                        elif "SELL" in reason:
                            indicators_dir = "SELL"
                    except (ValueError, IndexError):
                        pass
                if "harmonic" in reason and "conf=" in reason:
                    try:
                        harmonic_conf = float(
                            reason.split("conf=")[1].split(",")[0]
                        )
                        if "BUY" in reason:
                            harmonic_dir = "BUY"
                        elif "SELL" in reason:
                            harmonic_dir = "SELL"
                    except (ValueError, IndexError):
                        pass
                if "support_resistance" in reason and "conf=" in reason:
                    try:
                        sr_conf = float(
                            reason.split("conf=")[1].split(",")[0]
                        )
                        if "BUY" in reason:
                            sr_dir = "BUY"
                        elif "SELL" in reason:
                            sr_dir = "SELL"
                    except (ValueError, IndexError):
                        pass

            trades_detailed.append({
                "position_id": pos.id,
                "symbol": symbol,
                "direction": decision.direction.name,
                "entry_price": pos.entry_price,
                "entry_time": bar_time.isoformat() if bar_time else "",
                "regime": decision.regime.value,
                "confluence_score": decision.confluence_score,
                "signals_count": len(decision.reasons),
                "indicators_conf": indicators_conf,
                "indicators_dir": indicators_dir,
                "harmonic_conf": harmonic_conf,
                "harmonic_dir": harmonic_dir,
                "sr_conf": sr_conf,
                "sr_dir": sr_dir,
                "signal_conflict": (
                    indicators_dir is not None
                    and sr_dir is not None
                    and indicators_dir != sr_dir
                ),
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None,
                "pnl": None,
                "pnl_pct": None,
            })

        # Записываем закрытие позиции
        for closed_pos in result.closed_positions:
            # Находим соответствующую запись
            for t in trades_detailed:
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

    # ============ АНАЛИЗ ============
    print(f"\n{'=' * 70}")
    print(f"АНАЛИЗ СДЕЛОК: {symbol}")
    print('=' * 70)

    if not trades_detailed:
        print("Нет сделок для анализа")
        return trades_detailed

    wins = [t for t in trades_detailed if t["pnl"] is not None and t["pnl"] > 0]
    losses = [t for t in trades_detailed if t["pnl"] is not None and t["pnl"] < 0]

    print(f"\nВсего сделок: {len(trades_detailed)}")
    print(f"Побед: {len(wins)}, Убытков: {len(losses)}")

    # Анализ по анализаторам
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ПО АНАЛИЗАТОРАМ")
    print('=' * 70)

    for analyzer_name, conf_key, dir_key in [
        ("indicators", "indicators_conf", "indicators_dir"),
        ("harmonic", "harmonic_conf", "harmonic_dir"),
        ("support_resistance", "sr_conf", "sr_dir"),
    ]:
        win_confs = [t[conf_key] for t in wins if t[conf_key] is not None]
        loss_confs = [t[conf_key] for t in losses if t[conf_key] is not None]

        print(f"\n{analyzer_name.upper()}:")
        if win_confs:
            print(f"  Победные:  avg_conf={sum(win_confs)/len(win_confs):.3f}, "
                  f"count={len(win_confs)}")
        if loss_confs:
            print(f"  Убыточные: avg_conf={sum(loss_confs)/len(loss_confs):.3f}, "
                  f"count={len(loss_confs)}")

    # Анализ по режимам
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ПО РЕЖИМАМ")
    print('=' * 70)

    regimes_win = Counter(t["regime"] for t in wins)
    regimes_loss = Counter(t["regime"] for t in losses)

    for regime in set(list(regimes_win.keys()) + list(regimes_loss.keys())):
        w = regimes_win.get(regime, 0)
        l = regimes_loss.get(regime, 0)
        total = w + l
        if total > 0:
            print(f"  {regime:8s}: {w}W / {l}L = {w/total*100:.1f}% win rate")

    # Анализ по причинам закрытия
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ ПО ПРИЧИНАМ ЗАКРЫТИЯ")
    print('=' * 70)

    reasons = Counter(t["exit_reason"] for t in trades_detailed)
    for reason, count in reasons.most_common():
        pct = count / len(trades_detailed) * 100
        print(f"  {reason:15s}: {count:4d} ({pct:5.1f}%)")

    # Анализ конфликтов
    print(f"\n{'=' * 70}")
    print("АНАЛИЗ КОНФЛИКТОВ")
    print('=' * 70)

    conflicts = [t for t in trades_detailed if t["signal_conflict"]]
    no_conflicts = [t for t in trades_detailed if not t["signal_conflict"]]

    if conflicts:
        conflict_wins = [t for t in conflicts if t["pnl"] and t["pnl"] > 0]
        print(f"Сделок с конфликтом (indicators vs S/R): {len(conflicts)}")
        print(f"  Из них прибыльных: {len(conflict_wins)}")
        print(f"  Win rate: {len(conflict_wins)/len(conflicts)*100:.1f}%")

    if no_conflicts:
        no_conf_wins = [t for t in no_conflicts if t["pnl"] and t["pnl"] > 0]
        print(f"Сделок без конфликта: {len(no_conflicts)}")
        print(f"  Из них прибыльных: {len(no_conf_wins)}")
        print(f"  Win rate: {len(no_conf_wins)/len(no_conflicts)*100:.1f}%")

    # Топ-5 прибыльных и убыточных
    print(f"\n{'=' * 70}")
    print("ТОП-5 ПРИБЫЛЬНЫХ СДЕЛОК")
    print('=' * 70)
    for t in sorted(wins, key=lambda x: x["pnl"], reverse=True)[:5]:
        print(
            f"  {t['direction']:4s} score={t['confluence_score']:.3f} "
            f"regime={t['regime']:5s} pnl=${t['pnl']:+.2f} "
            f"ind={t['indicators_conf']} harm={t['harmonic_conf']} "
            f"sr={t['sr_conf']}"
        )

    print(f"\n{'=' * 70}")
    print("ТОП-5 УБЫТОЧНЫХ СДЕЛОК")
    print('=' * 70)
    for t in sorted(losses, key=lambda x: x["pnl"])[:5]:
        print(
            f"  {t['direction']:4s} score={t['confluence_score']:.3f} "
            f"regime={t['regime']:5s} pnl=${t['pnl']:+.2f} "
            f"ind={t['indicators_conf']} harm={t['harmonic_conf']} "
            f"sr={t['sr_conf']}"
        )

    # Экспорт в CSV
    csv_file = export_path / f"diagnostics_{symbol.replace('-', '_')}.csv"
    if trades_detailed:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=trades_detailed[0].keys(),
            )
            writer.writeheader()
            writer.writerows(trades_detailed)
        print(f"\nДетальные данные: {csv_file}")

    return trades_detailed


async def main():
    symbols = [
        ("BTC-USD", "1h"),
        ("ETH-USD", "1h"),
    ]

    for symbol, timeframe in symbols:
        await run_backtest_with_diagnostics(
            symbol=symbol,
            timeframe=timeframe,
            bars_to_process=5000,
        )


if __name__ == "__main__":
    asyncio.run(main())