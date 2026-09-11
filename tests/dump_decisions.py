"""Дамп decisions из БД — проверка формата reasons.

Показывает реальные строки reasons, чтобы понять,
как парсить анализаторы.
"""

import sqlite3
import json
from pathlib import Path


def dump_decisions(db_path: str = "backtest_results/component_quality/comp_BTC_USD.db", limit: int = 10):
    """Показывает примеры reasons из БД."""
    if not Path(db_path).exists():
        print(f"❌ Файл не найден: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("""
        SELECT id, symbol, regime, direction, confluence_score, passed_gate,
               signals_count, weights, reasons
        FROM decisions
        WHERE passed_gate = 1
        ORDER BY id
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    print(f"\n{'=' * 70}")
    print(f"DECISIONS С PASSED_GATE=1: {len(rows)} записей")
    print(f"{'=' * 70}")

    for row in rows:
        print(f"\n--- Decision ID={row['id']} ---")
        print(f"Symbol: {row['symbol']}")
        print(f"Direction: {row['direction']}")
        print(f"Score: {row['confluence_score']:.4f}")
        print(f"Signals count: {row['signals_count']}")
        print(f"Weights: {row['weights']}")

        # Парсим reasons
        try:
            reasons = json.loads(row['reasons'])
            print(f"Reasons ({len(reasons)}):")
            for i, r in enumerate(reasons):
                print(f"  {i+1}. {r}")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Reasons RAW: {row['reasons']}")

    conn.close()


def check_all_files():
    """Проверяет все файлы БД."""
    paths = [
        "backtest_results/component_quality/comp_BTC_USD.db",
        "backtest_results/component_quality/comp_ETH_USD.db",
    ]
    for p in paths:
        if Path(p).exists():
            print(f"\n{'#' * 70}")
            print(f"# Файл: {p}")
            print(f"{'#' * 70}")
            dump_decisions(p, limit=5)
        else:
            print(f"⚠️  Не найден: {p}")


if __name__ == "__main__":
    check_all_files()