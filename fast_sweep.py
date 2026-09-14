"""Быстрый per-symbol sweep весов.

Ключевая идея:
1. Один раз прогнать ВСЕ анализаторы на ВСЕХ барах.
2. Сохранить сырые сигналы: (bar_index, source, direction, confidence).
3. Для каждой комбинации весов — пересчитать confluence БЕЗ перезапуска анализаторов.

Эффект: ×100 ускорение.

Также тестирует:
- gate_mode
- 1d-фильтр (нет, т.к. он блокирует открытие, а не сигналы)
- CHOП-фильтр (нет, по той же причине)

Sweep только по весам 4 анализаторов.

Запуск:
    python fast_sweep.py
"""

import asyncio
import csv
import math
import time
from datetime import datetime, timezone
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.analyzers.indicator_cache import IndicatorCache
from src.analyzers.trend import TrendAnalyzer
from src.analyzers.elliott_wave import ElliottWaveAnalyzer
from src.analyzers.volatility import VolatilityAnalyzer
from src.analyzers.volume import VolumeAnalyzer
from src.confluence.regime_detector import RegimeDetector, MarketRegime
from src.models import AnalyzerSignal, SignalDirection


setup_logging(level="ERROR")


SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "ATOM-USD"]
BARS_1H = 40000
WARMUP_BARS = 250
STARTING_EQUITY = 10000.0
EXPORT_DIR = "backtest_results/fast_sweep"

BASE_PARAMS = {
    "BTC-USD": {
        "timeframe": "2h", "atr": 2.0, "rr": 3.0,
        "chop": True, "regime_1d": "baseline",
    },
    "ETH-USD": {
        "timeframe": "1h", "atr": 2.5, "rr": 2.5,
        "chop": True, "regime_1d": "baseline",
    },
    "ADA-USD": {
        "timeframe": "1h", "atr": 3.0, "rr": 2.5,
        "chop": False, "regime_1d": "none",
    },
    "DOT-USD": {
        "timeframe": "1h", "atr": 2.5, "rr": 2.5,
        "chop": False, "regime_1d": "none",
    },
    "ATOM-USD": {
        "timeframe": "1h", "atr": 4.0, "rr": 3.5,
        "chop": True, "regime_1d": "drawdown_10",
    },
}

ANALYZERS = ["trend", "elliott_wave", "volatility", "volume"]

ATR_PERIOD = 14
TIMEOUT_BARS = 100
RISK_PCT = 0.01


def generate_weight_combos(step: float = 0.1) -> List[Dict[str, float]]:
    n = int(round(1.0 / step))
    combos = []
    for i in range(0, n + 1):
        for j in range(0, n + 1 - i):
            for k in range(0, n + 1 - i - j):
                l = n - i - j - k
                combo = {
                    "trend": round(i * step, 2),
                    "elliott_wave": round(j * step, 2),
                    "volatility": round(k * step, 2),
                    "volume": round(l * step, 2),
                }
                if abs(sum(combo.values()) - 1.0) < 1e-6:
                    combos.append(combo)
    return combos


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    })
    return resampled.dropna()


def simulate_trade_r(
    direction: SignalDirection,
    entry_price: float,
    atr_value: float,
    atr_mult: float,
    rr_ratio: float,
    future_highs,
    future_lows,
    future_closes,
) -> float:
    if atr_value <= 0 or entry_price <= 0 or atr_mult <= 0:
        return 0.0

    sl_distance = atr_value * atr_mult
    tp_distance = sl_distance * rr_ratio

    if direction == SignalDirection.BUY:
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance

    for i in range(len(future_closes)):
        hi = future_highs[i]
        lo = future_lows[i]
        if direction == SignalDirection.BUY:
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return +rr_ratio
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return +rr_ratio

    exit_price = future_closes[-1]
    if direction == SignalDirection.BUY:
        price_move = exit_price - entry_price
    else:
        price_move = entry_price - exit_price
    return price_move / sl_distance


async def collect_signals_for_symbol(
    symbol: str,
    data: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Прогоняет ВСЕ анализаторы на ВСЕХ барах.
    Возвращает:
    - signals_by_bar: {bar_index: {source: (direction, confidence)}}
    - bars_meta: {bar_index: {atr, regime_chop, regime_1d}}
    - trade_outcomes: {bar_index: {direction: R}}
    """
    params = BASE_PARAMS[symbol]
    timeframe = params["timeframe"]

    print(f"  Collecting signals for {symbol} ({len(data)} bars)...")

    t0 = time.time()
    indicator_cache = IndicatorCache(data)
    print(f"    IndicatorCache: {time.time() - t0:.2f}s")

    analyzers = {
        "trend": TrendAnalyzer(name="trend"),
        "elliott_wave": ElliottWaveAnalyzer(name="elliott_wave"),
        "volatility": VolatilityAnalyzer(name="volatility"),
        "volume": VolumeAnalyzer(name="volume"),
    }

    regime_detector = RegimeDetector()
    regime_1d_detector = RegimeDetector()

    signals_by_bar: Dict[int, Dict[str, Tuple[str, float]]] = {}
    bars_meta: Dict[int, Dict[str, Any]] = {}
    trade_outcomes: Dict[int, Dict[str, float]] = {}

    closes = data["close"].values
    highs = data["high"].values
    lows = data["low"].values

    n = len(data)

    t0 = time.time()
    for i in range(WARMUP_BARS, n):
        ind = indicator_cache.slice(i)
        atr_val = ind.get("atr_14")
        if atr_val is None or atr_val <= 0:
            continue

        # Режим на текущем баре (через кэш)
        sma200 = ind.get("sma_200")
        adx = ind.get("adx")
        price = ind.get("current_price")
        chop_blocked = False
        if sma200 is not None and adx is not None and price is not None and sma200 > 0:
            distance_pct = (price - sma200) / sma200
            if abs(distance_pct) < 0.02 or adx < 20.0:
                chop_blocked = True

        # 1d-фильтр
        bar_time = data.index[i]
        regime_1d = _get_regime_1d_static(
            df_1d, bar_time, params["regime_1d"], regime_1d_detector,
        )

        bars_meta[i] = {
            "atr": float(atr_val),
            "chop_blocked": chop_blocked,
            "regime_1d": regime_1d.name,
            "current_price": float(closes[i]),
        }

        # Сигналы от анализаторов
        bar_signals = {}
        for name, analyzer in analyzers.items():
            try:
                sig = await analyzer.analyze(
                    data, indicators=indicator_cache, bar_index=i,
                )
            except Exception:
                sig = None

            if sig is not None:
                bar_signals[name] = (sig.direction.name, float(sig.confidence))

        signals_by_bar[i] = bar_signals

        # Заранее посчитаем R для BUY и SELL (без весов)
        future_highs = highs[i + 1: i + 1 + TIMEOUT_BARS]
        future_lows = lows[i + 1: i + 1 + TIMEOUT_BARS]
        future_closes = closes[i + 1: i + 1 + TIMEOUT_BARS]

        if len(future_closes) >= 5:
            r_buy = simulate_trade_r(
                SignalDirection.BUY, float(closes[i]), float(atr_val),
                params["atr"], params["rr"],
                future_highs, future_lows, future_closes,
            )
            r_sell = simulate_trade_r(
                SignalDirection.SELL, float(closes[i]), float(atr_val),
                params["atr"], params["rr"],
                future_highs, future_lows, future_closes,
            )
            trade_outcomes[i] = {"BUY": r_buy, "SELL": r_sell}

        if (i - WARMUP_BARS) % 3000 == 0 and i > WARMUP_BARS:
            elapsed = time.time() - t0
            rate = (i - WARMUP_BARS) / elapsed if elapsed > 0 else 0
            print(f"    [{i - WARMUP_BARS}/{n - WARMUP_BARS}] rate={rate:.0f} bars/s")

    total_time = time.time() - t0
    print(f"    Collected in {total_time:.2f}s")

    return {
        "symbol": symbol,
        "signals_by_bar": signals_by_bar,
        "bars_meta": bars_meta,
        "trade_outcomes": trade_outcomes,
    }


def _get_regime_1d_static(
    df_1d: pd.DataFrame,
    current_time,
    mode: str,
    regime_detector: RegimeDetector,
) -> MarketRegime:
    if mode == "none":
        return MarketRegime.CHOP
    if df_1d is None or current_time is None:
        return MarketRegime.CHOP

    try:
        available = df_1d[df_1d.index <= current_time]
    except Exception:
        return MarketRegime.CHOP

    if len(available) < 200:
        return MarketRegime.CHOP

    if mode == "baseline":
        window = available.iloc[-300:]
        return regime_detector.detect(window)

    current_price = float(available["close"].iloc[-1])

    if mode == "drawdown_10":
        lookback = min(30, len(available))
        if lookback < 5:
            return MarketRegime.CHOP
        past_price = float(available["close"].iloc[-lookback])
        if past_price <= 0:
            return MarketRegime.CHOP
        drawdown = (current_price - past_price) / past_price
        if drawdown < -0.10:
            return MarketRegime.BEAR
        if drawdown > 0.05:
            return MarketRegime.BULL
        return MarketRegime.CHOP

    return MarketRegime.CHOP


def evaluate_combo(
    data: Dict[str, Any],
    weights: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Быстрый пересчёт confluence для одной комбинации весов.

    Сигналы уже собраны — просто взвешиваем.
    """
    signals_by_bar = data["signals_by_bar"]
    bars_meta = data["bars_meta"]
    trade_outcomes = data["trade_outcomes"]

    total_trades = 0
    wins = 0
    losses = 0
    total_r = 0.0
    gross_win = 0.0
    gross_loss = 0.0

    w_trend = weights["trend"]
    w_elliott = weights["elliott_wave"]
    w_volatility = weights["volatility"]
    w_volume = weights["volume"]

    for bar_idx, signals in signals_by_bar.items():
        if not signals:
            continue

        meta = bars_meta.get(bar_idx)
        if meta is None:
            continue

        # CHOP-фильтр
        if meta["chop_blocked"]:
            continue

        regime_1d = meta["regime_1d"]

        bull_score = 0.0
        bear_score = 0.0

        for source, (direction, confidence) in signals.items():
            if source == "trend":
                w = w_trend
            elif source == "elliott_wave":
                w = w_elliott
            elif source == "volatility":
                w = w_volatility
            elif source == "volume":
                w = w_volume
            else:
                continue

            if w <= 0:
                continue

            contribution = confidence * w
            if direction == "BUY":
                bull_score += contribution
            elif direction == "SELL":
                bear_score += contribution

        # Определяем направление
        if bull_score > bear_score and bull_score >= threshold:
            direction = "BUY"
        elif bear_score > bull_score and bear_score >= threshold:
            direction = "SELL"
        else:
            continue

        # 1d-фильтр
        if regime_1d == "BEAR" and direction == "BUY":
            continue
        if regime_1d == "BULL" and direction == "SELL":
            continue

        # Исход
        outcome = trade_outcomes.get(bar_idx)
        if outcome is None:
            continue

        r = outcome.get(direction)
        if r is None:
            continue

        total_trades += 1
        total_r += r
        if r > 0:
            wins += 1
            gross_win += r
        elif r < 0:
            losses += 1
            gross_loss += abs(r)

    if total_trades == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "total_r": 0.0, "avg_r": 0.0,
        }

    return {
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_trades,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else 0.0,
        "total_r": total_r,
        "avg_r": total_r / total_trades,
    }


def run_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Воркер: собирает сигналы для одного символа."""
    symbol = args["symbol"]
    data = args["data"]
    df_1d = args["df_1d"]

    try:
        return asyncio.run(collect_signals_for_symbol(symbol, data, df_1d))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "symbol": symbol,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    export_path = Path(EXPORT_DIR)
    export_path.mkdir(parents=True, exist_ok=True)

    weight_combos = generate_weight_combos(step=0.1)

    print("=" * 130)
    print("FAST SWEEP (pre-collected signals)")
    print(f"Symbols:  {SYMBOLS}")
    print(f"Combos:   {len(weight_combos)} per symbol")
    print("=" * 130)

    print("\nPreloading data in main process...")
    fetcher = MarketDataFetcher()
    data_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

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

            data_cache[sym] = (df, df_1d)
            print(f"  {sym}: {len(df)} bars ({timeframe})")
        except Exception as e:
            print(f"  {sym}: FAILED - {e}")

    if not data_cache:
        print("\nNo data. Exit.")
        return

    # === Фаза 1: Сбор сигналов ===
    print("\n" + "=" * 70)
    print("PHASE 1: Collect signals (parallel by symbol)")
    print("=" * 70)

    t_collect_start = time.time()

    tasks = [
        {"symbol": sym, "data": data_cache[sym][0], "df_1d": data_cache[sym][1]}
        for sym in SYMBOLS if sym in data_cache
    ]

    n_workers = min(len(tasks), max(1, cpu_count() - 1))
    print(f"Running {len(tasks)} tasks with {n_workers} workers...\n")

    collected_data: Dict[str, Dict[str, Any]] = {}

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_worker, tasks):
            symbol = result.get("symbol", "?")
            if "error" in result:
                print(f"  FAIL {symbol}: {result['error']}")
                continue
            collected_data[symbol] = result
            n_signals = sum(1 for s in result["signals_by_bar"].values() if s)
            print(
                f"  {symbol}: {len(result['signals_by_bar'])} bars, "
                f"{n_signals} with signals"
            )

    t_collect = time.time() - t_collect_start
    print(f"\nCollection time: {t_collect:.1f}s")

    # === Фаза 2: Пересчёт весов (быстро, в одном процессе) ===
    print("\n" + "=" * 70)
    print("PHASE 2: Evaluate weights (fast, single process)")
    print("=" * 70)

    t_eval_start = time.time()

    all_results: List[Dict[str, Any]] = []

    for sym in SYMBOLS:
        if sym not in collected_data:
            continue

        symbol_data = collected_data[sym]
        print(f"\n  {sym}: evaluating {len(weight_combos)} combos...")

        best_pnl = -float("inf")
        best_weights = None

        for w in weight_combos:
            metrics = evaluate_combo(symbol_data, w, threshold=0.5)

            row = {
                "symbol": sym,
                "w_trend": w["trend"],
                "w_elliott": w["elliott_wave"],
                "w_volatility": w["volatility"],
                "w_volume": w["volume"],
                **metrics,
            }
            all_results.append(row)

            if metrics["total_r"] > best_pnl:
                best_pnl = metrics["total_r"]
                best_weights = w

        print(f"    best: {best_weights}, totalR={best_pnl:+.2f}")

    t_eval = time.time() - t_eval_start
    print(f"\nEvaluation time: {t_eval:.1f}s")
    print(f"Total time: {t_collect + t_eval:.1f}s")

    # === Вывод ===
    valid = [r for r in all_results if r.get("trades", 0) > 0]

    print("\n" + "=" * 140)
    print("BEST WEIGHTS PER SYMBOL (by totalR)")
    print("=" * 140)
    print(
        f"{'symbol':<10} {'trend':>6} {'elliott':>8} {'volat':>6} "
        f"{'volume':>7} "
        f"{'trades':>7} {'WR%':>7} {'PF':>6} "
        f"{'totalR':>9} {'avgR':>8}"
    )
    print("-" * 100)

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue

        best = max(sym_results, key=lambda x: x["total_r"])
        print(
            f"{sym:<10} "
            f"{best['w_trend']:>6.1f} "
            f"{best['w_elliott']:>8.1f} "
            f"{best['w_volatility']:>6.1f} "
            f"{best['w_volume']:>7.1f} "
            f"{best['trades']:>7} "
            f"{best['win_rate'] * 100:>7.1f} "
            f"{best['profit_factor']:>6.2f} "
            f"{best['total_r']:>+9.2f} "
            f"{best['avg_r']:>+8.3f}"
        )

    print("\n" + "=" * 140)
    print("TOP-10 WEIGHTS FOR EACH SYMBOL (by totalR)")
    print("=" * 140)

    for sym in SYMBOLS:
        sym_results = [r for r in valid if r["symbol"] == sym]
        if not sym_results:
            continue

        print(f"\n{sym}")
        print(
            f"  {'trend':>6} {'elliott':>8} {'volat':>6} {'volume':>7} "
            f"{'trades':>7} {'WR%':>7} {'PF':>6} "
            f"{'totalR':>9} {'avgR':>8}"
        )

        top10 = sorted(sym_results, key=lambda x: x["total_r"], reverse=True)[:10]
        for r in top10:
            print(
                f"  {r['w_trend']:>6.1f} "
                f"{r['w_elliott']:>8.1f} "
                f"{r['w_volatility']:>6.1f} "
                f"{r['w_volume']:>7.1f} "
                f"{r['trades']:>7} "
                f"{r['win_rate'] * 100:>7.1f} "
                f"{r['profit_factor']:>6.2f} "
                f"{r['total_r']:>+9.2f} "
                f"{r['avg_r']:>+8.3f}"
            )

    csv_path = export_path / "fast_sweep_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()