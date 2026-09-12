"""Дамп decisions из БД — проверка формата S/R."""

import sqlite3
import json
from pathlib import Path


def dump(db_path, limit=10):
    if not Path(db_path).exists():
        print(f"❌ Не найден: {db_path}")
        return
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, symbol, direction, confluence_score, reasons
        FROM decisions
        WHERE passed_gate = 1
        ORDER BY id LIMIT ?
    """, (limit,)).fetchall()
    print(f"\n{'=' * 70}")
    print(f"{db_path}: {len(rows)} записей")
    print("=" * 70)
    for row in rows:
        print(f"\n--- ID={row['id']} {row['symbol']} {row['direction']} ---")
        try:
            reasons = json.loads(row['reasons'])
            for i, r in enumerate(reasons):
                print(f"  {i+1}. {r}")
        except Exception:
            print(f"  RAW: {row['reasons']}")
    conn.close()


if __name__ == "__main__":
    paths = [
        "backtest_results/component_quality/comp_BTC_USD.db",
        "backtest_results/component_quality/comp_ETH_USD.db",
        "backtest_results/component_quality/comp_ADA_USD.db",
    ]
    for p in paths:
        dump(p, limit=5)