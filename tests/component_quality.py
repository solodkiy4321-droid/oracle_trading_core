"""Диагностика качества отдельных компонентов анализаторов.

Разбирает каждый анализатор на компоненты и измеряет IC каждого компонента:
- indicators: RSI, MACD, EMA, Bollinger, свечи
- harmonic: Gartley, Bat, Butterfly, Crab
- support_resistance: сила уровня, близость, тренд

Цель: понять, какие компоненты работают, какие шумят.
"""

import asyncio
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.engine.config import EngineConfig
from src.engine.trading_engine import TradingEngine

setup_logging(level="ERROR")


# ============ Утилиты ============

def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation."""
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
    """Вердикт по IC."""
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


# ============ Компоненты Indicators ============

def extract_indicator_components(data: pd.DataFrame) -> Dict[str, float]:
    """
    Извлекает компоненты indicators из данных.

    Возвращает словарь с направлениями:
    - rsi_signal: +1 (oversold), -1 (overbought), 0 (нейтрально)
    - macd_signal: +1 (bullish), -1 (bearish), 0
    - ema_signal: +1 (bullish alignment), -1 (bearish), 0
    - bb_signal: +1 (lower touch), -1 (upper touch), 0
    - candle_signal: +1 (bullish candle), -1 (bearish), 0
    """
    if len(data) < 50:
        return {}

    components = {}

    # RSI
    rsi = ta.rsi(data["close"], length=14)
    if rsi is not None and len(rsi) > 0:
        rsi_val = float(rsi.iloc[-1])
        if rsi_val < 30:
            components["rsi_signal"] = 1.0
        elif rsi_val > 70:
            components["rsi_signal"] = -1.0
        else:
            components["rsi_signal"] = 0.0

    # MACD
    macd = ta.macd(data["close"])
    if macd is not None and len(macd) > 0:
        macd_hist = float(macd["MACDh_12_26_9"].iloc[-1])
        components["macd_signal"] = 1.0 if macd_hist > 0 else -1.0

    # EMA
    ema9 = ta.ema(data["close"], length=9)
    sma20 = ta.sma(data["close"], length=20)
    sma50 = ta.sma(data["close"], length=50)
    if all(x is not None and len(x) > 0 for x in [ema9, sma20, sma50]):
        e9 = float(ema9.iloc[-1])
        s20 = float(sma20.iloc[-1])
        s50 = float(sma50.iloc[-1])
        if e9 > s20 > s50:
            components["ema_signal"] = 1.0
        elif e9 < s20 < s50:
            components["ema_signal"] = -1.0
        else:
            components["ema_signal"] = 0.0

    # Bollinger Bands
    bb = ta.bbands(data["close"], length=20)
    if bb is not None and len(bb) > 0:
        bb_upper = float(bb["BBU_20_2.0"].iloc[-1])
        bb_lower = float(bb["BBL_20_2.0"].iloc[-1])
        current = float(data["close"].iloc[-1])
        if current <= bb_lower:
            components["bb_signal"] = 1.0
        elif current >= bb_upper:
            components["bb_signal"] = -1.0
        else:
            components["bb_signal"] = 0.0

    # Свечи
    open_ = data["open"].values
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values

    hammer = ta.cdl_pattern(name="hammer", open_=open_, high=high, low=low, close=close)
    engulfing = ta.cdl_pattern(name="engulfing", open_=open_, high=high, low=low, close=close)

    candle = 0.0
    if hammer is not None and len(hammer) > 0 and float(hammer.iloc[-1]) > 0:
        candle = 1.0
    elif engulfing is not None and len(engulfing) > 0:
        eng_val = float(engulfing.iloc[-1])
        candle = 1.0 if eng_val > 0 else (-1.0 if eng_val < 0 else 0.0)
    components["candle_signal"] = candle

    return components


# ============ Компоненты Harmonic ============

def extract_harmonic_components(decision_reasons: List[str]) -> Dict[str, float]:
    """
    Извлекает паттерн harmonic из decision.reasons.

    Возвращает:
    - gartley_signal: +1 (bullish), -1 (bearish), 0 (нет)
    - bat_signal
    - butterfly_signal
    - crab_signal
    """
    components = {
        "gartley_signal": 0.0,
        "bat_signal": 0.0,
        "butterfly_signal": 0.0,
        "crab_signal": 0.0,
    }

    for reason in decision_reasons:
        if "harmonic" not in reason:
            continue
        reason_lower = reason.lower()
        direction = 0.0
        if "bullish" in reason_lower:
            direction = 1.0
        elif "bearish" in reason_lower:
            direction = -1.0

        if "gartley" in reason_lower:
            components["gartley_signal"] = direction
        elif "bat" in reason_lower:
            components["bat_signal"] = direction
        elif "butterfly" in reason_lower:
            components["butterfly_signal"] = direction
        elif "crab" in reason_lower:
            components["crab_signal"] = direction

    return components


# ============ Компоненты S/R ============

def extract_sr_components(decision_reasons: List[str], meta: dict) -> Dict[str, float]:
    """
    Извлекает компоненты S/R из reasons и metadata.
    """
    components = {
        "sr_strength": 0.0,
        "sr_proximity": 0.0,
        "sr_touches": 0.0,
    }

    for reason in decision_reasons:
        if "support_resistance" not in reason:
            continue
        # Извлекаем strength из reason
        if "strength=" in reason:
            try:
                s = float(reason.split("strength=")[1].split(",")[0])
                components["sr_strength"] = s
            except (ValueError, IndexError):
                pass
        if "touches=" in reason:
            try:
                t = float(reason.split("touches=")[1].split(",")[0])
                components["sr_touches"] = t
            except (ValueError, IndexError):
                pass
        if "distance=" in reason:
            try:
                d_str = reason.split("distance=")[1].rstrip("%").split(",")[0]
                d = float(d_str)
                components["sr_proximity"] = 1.0 - (d / 2.0)  # 0% → 1.0, 2% → 0.0
            except (ValueError, IndexError):
                pass

    return components


# ============ Сбор данных ============

async def collect_detailed_trades(
    symbol: str,
    timeframe: str = "1h",
    bars_to_process: int = 5000,
    export_dir: str = "backtest_results/component_quality",
) -> List[dict]:
    """
    Собирает данные по каждой сделке + компоненты анализаторов на момент входа.
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
        journal_db_path=f"{export_dir}/comp_{symbol.replace('-', '_')}.db",
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

        # Открытие позиции
        if result.opened_position is not None and result.decision is not None:
            decision = result.decision
            pos = result.opened_position

            # Извлекаем компоненты
            indicator_comps = extract_indicator_components(slice_data)
            harmonic_comps = extract_harmonic_components(decision.reasons)
            sr_comps = extract_sr_components(decision.reasons, decision.weights_used)

            record = {
                "position_id": pos.id,
                "symbol": symbol,
                "direction": decision.direction.name,
                "regime": decision.regime.value,
                "score": decision.confluence_score,
                "entry_price": pos.entry_price,
                "entry_time": bar_time.isoformat() if bar_time else "",
                "exit_price": None,
                "exit_time": None,
                "exit_reason": None,
                "pnl": None,
                # Indicator components
                **{f"ind_{k}": v for k, v in indicator_comps.items()},
                # Harmonic components
                **{f"harm_{k}": v for k, v in harmonic_comps.items()},
                # S/R components
                **{f"sr_{k}": v for k, v in sr_comps.items()},
            }
            trades.append(record)

        # Закрытие
        for closed in result.closed_positions:
            for t in trades:
                if t["position_id"] == closed.id:
                    t["exit_price"] = closed.close_price
                    t["exit_time"] = (
                        closed.close_time.isoformat()
                        if closed.close_time else ""
                    )
                    t["exit_reason"] = (
                        closed.close_reason.value
                        if closed.close_reason else "unknown"
                    )
                    t["pnl"] = closed.realized_pnl
                    break

    engine.close()
    return trades


# ============ Анализ ============

def analyze_component(
    trades: List[dict],
    component_key: str,
    component_name: str,
    signal_type: str = "continuous",
):
    """
    Анализирует один компонент.

    signal_type:
    - 'continuous': значение компонента — непрерывная переменная (IC)
    - 'directional': сигнал -1/0/+1 — считаем win rate по совпадению с направлением сделки
    """
    # Фильтруем сделки с этим компонентом
    relevant = [
        t for t in trades
        if component_key in t
        and t[component_key] is not None
        and t.get("pnl") is not None
    ]

    if len(relevant) < 2:
        return None

    if signal_type == "continuous":
        # IC между значением компонента и PnL
        values = [t[component_key] for t in relevant]
        pnls = [t["pnl"] for t in relevant]
        ic = correlation(values, pnls)

        # Дополнительно: PnL когда компонент > 0 vs < 0
        pos_cases = [t["pnl"] for t in relevant if t[component_key] > 0]
        neg_cases = [t["pnl"] for t in relevant if t[component_key] < 0]

        return {
            "name": component_name,
            "n": len(relevant),
            "ic": ic,
            "n_pos": len(pos_cases),
            "avg_pnl_pos": sum(pos_cases) / len(pos_cases) if pos_cases else 0.0,
            "n_neg": len(neg_cases),
            "avg_pnl_neg": sum(neg_cases) / len(neg_cases) if neg_cases else 0.0,
        }

    elif signal_type == "directional":
        # Сигнал -1/0/+1. Считаем win rate когда сигнал совпадает с направлением сделки
        match_pnls = []
        mismatch_pnls = []
        neutral_pnls = []

        for t in relevant:
            sig = t[component_key]
            trade_dir = 1.0 if t["direction"] == "BUY" else -1.0

            if sig == 0.0:
                neutral_pnls.append(t["pnl"])
            elif sig == trade_dir:
                match_pnls.append(t["pnl"])
            else:
                mismatch_pnls.append(t["pnl"])

        def stats(pnls_list):
            if not pnls_list:
                return (0, 0.0, 0.0)
            wins = sum(1 for p in pnls_list if p > 0)
            return (
                len(pnls_list),
                wins / len(pnls_list),
                sum(pnls_list) / len(pnls_list),
            )

        n_match, wr_match, pnl_match = stats(match_pnls)
        n_mismatch, wr_mismatch, pnl_mismatch = stats(mismatch_pnls)
        n_neutral, wr_neutral, pnl_neutral = stats(neutral_pnls)

        # IC = разница между match и mismatch win rate
        ic_proxy = wr_match - wr_mismatch if (n_match > 0 and n_mismatch > 0) else 0.0

        return {
            "name": component_name,
            "n": len(relevant),
            "ic": ic_proxy,
            "n_match": n_match,
            "wr_match": wr_match,
            "pnl_match": pnl_match,
            "n_mismatch": n_mismatch,
            "wr_mismatch": wr_mismatch,
            "pnl_mismatch": pnl_mismatch,
            "n_neutral": n_neutral,
        }


def print_component_report(results: List[dict], title: str, signal_type: str = "continuous"):
    """Печатает отчёт по компонентам."""
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")

    results = [r for r in results if r is not None]
    if not results:
        print("  Нет данных")
        return

    if signal_type == "continuous":
        for r in results:
            print(f"\n  {r['name']}:")
            print(f"    n={r['n']}, IC={r['ic']:+.3f} {verdict_ic(r['ic'])}")
            if r["n_pos"] > 0:
                print(f"    Когда > 0 (n={r['n_pos']}): avg_pnl=${r['avg_pnl_pos']:+.2f}")
            if r["n_neg"] > 0:
                print(f"    Когда < 0 (n={r['n_neg']}): avg_pnl=${r['avg_pnl_neg']:+.2f}")

    elif signal_type == "directional":
        for r in results:
            print(f"\n  {r['name']}:")
            print(f"    IC proxy={r['ic']:+.3f} {verdict_ic(r['ic'])}")
            if r["n_match"] > 0:
                print(
                    f"    Совпал с направлением (n={r['n_match']}): "
                    f"WR={r['wr_match'] * 100:.1f}%, avg=${r['pnl_match']:+.2f}"
                )
            if r["n_mismatch"] > 0:
                print(
                    f"    Против направления (n={r['n_mismatch']}): "
                    f"WR={r['wr_mismatch'] * 100:.1f}%, avg=${r['pnl_mismatch']:+.2f}"
                )
            if r["n_neutral"] > 0:
                print(f"    Нейтрально (n={r['n_neutral']})")


async def main():
    """Запускает диагностику компонентов."""
    symbols = [
        ("ETH-USD", "1h"),  # BTC сначала откатить — пока только ETH
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
        print(f"# АНАЛИЗ КОМПОНЕНТОВ: {symbol}")
        print(f"{'#' * 70}")

        # ============ Indicators components ============
        ind_results = []
        for key, name in [
            ("ind_rsi_signal", "RSI"),
            ("ind_macd_signal", "MACD"),
            ("ind_ema_signal", "EMA"),
            ("ind_bb_signal", "Bollinger Bands"),
            ("ind_candle_signal", "Свечи"),
        ]:
            r = analyze_component(trades, key, name, "directional")
            ind_results.append(r)
        print_component_report(ind_results, "INDICATORS: КОМПОНЕНТЫ", "directional")

        # ============ Harmonic components ============
        harm_results = []
        for key, name in [
            ("harm_gartley_signal", "Gartley"),
            ("harm_bat_signal", "Bat"),
            ("harm_butterfly_signal", "Butterfly"),
            ("harm_crab_signal", "Crab"),
        ]:
            r = analyze_component(trades, key, name, "directional")
            harm_results.append(r)
        print_component_report(harm_results, "HARMONIC: КОМПОНЕНТЫ", "directional")

        # ============ S/R components ============
        sr_results = []
        for key, name in [
            ("sr_sr_strength", "Сила уровня"),
            ("sr_sr_proximity", "Близость к уровню"),
            ("sr_sr_touches", "Количество касаний"),
        ]:
            r = analyze_component(trades, key, name, "continuous")
            sr_results.append(r)
        print_component_report(sr_results, "S/R: КОМПОНЕНТЫ", "continuous")

    # Экспорт
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