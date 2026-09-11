"""Отладка переработанного HarmonicAnalyzer.

Проверяет:
- Находит ли детектор формирующиеся паттерны
- Прогноз PRZ и расстояние от цены
- Срабатывает ли analyzer
- Стабильность на 20 барах
"""

import asyncio

from src.logger import setup_logging
from src.data.fetcher import MarketDataFetcher
from src.analyzers.harmonic import FormingHarmonicDetector, HarmonicAnalyzer
from src.models import SignalDirection

setup_logging(level="WARNING")


async def debug_harmonic(symbol: str = "ETH-USD", timeframe: str = "1h", bars: int = 5000):
    print(f"\n{'=' * 70}")
    print(f"ОТЛАДКА ПЕРЕРАБОТАННОГО HARMONIC: {symbol} {timeframe}")
    print(f"{'=' * 70}")

    fetcher = MarketDataFetcher()
    data = fetcher.fetch(symbol, timeframe, limit=bars + 250)
    print(f"Загружено {len(data)} баров")

    detector = FormingHarmonicDetector(tolerance=0.20, min_pattern_age=3, max_pattern_age=200)
    swing_points = detector.find_swing_points(data, window=5)
    print(f"Swing points: {len(swing_points)}")

    current_price = float(data["close"].iloc[-1])
    print(f"Текущая цена: {current_price:.4f}")

    # 1. Все формирующиеся паттерны
    print(f"\n{'=' * 70}")
    print("1. ФОРМИРУЮЩИЕСЯ ПАТТЕРНЫ (все, без фильтра цены)")
    print(f"{'=' * 70}")
    patterns = detector.detect_forming_patterns(
        swing_points, current_price, total_bars=len(data), max_patterns=20,
    )
    print(f"Найдено формирующихся паттернов: {len(patterns)}")
    for i, p in enumerate(patterns[:10]):
        dist_pct = abs(current_price - p.prz_center) / current_price * 100
        print(
            f"  {i+1:2d}. {p.pattern_type:10s} ({p.direction:8s}) "
            f"PRZ={p.prz_center:.2f} [{p.prz_low:.2f}-{p.prz_high:.2f}], "
            f"conf={p.base_confidence:.3f}, C_age={p.age_bars}, "
            f"dist={dist_pct:.2f}%"
        )

    # 2. Фильтрация по близости к PRZ
    print(f"\n{'=' * 70}")
    print("2. ПАТТЕРНЫ РЯДОМ С PRZ (в пределах 3%)")
    print(f"{'=' * 70}")
    near_prz = [p for p in patterns if abs(current_price - p.prz_center) / current_price <= 0.03]
    print(f"Паттернов в пределах 3%: {len(near_prz)}")
    for p in near_prz:
        dist_pct = abs(current_price - p.prz_center) / current_price * 100
        print(
            f"  {p.pattern_type:10s} ({p.direction:8s}) "
            f"PRZ={p.prz_center:.2f}, dist={dist_pct:.2f}%, "
            f"conf={p.base_confidence:.3f}"
        )

    # 3. Analyzer
    print(f"\n{'=' * 70}")
    print("3. ANALYZER С РАЗНЫМИ max_distance_to_prz_pct")
    print(f"{'=' * 70}")
    for dist in [0.01, 0.02, 0.03, 0.05, 0.10]:
        analyzer = HarmonicAnalyzer(
            tolerance=0.20,
            min_confidence=0.30,
            max_distance_to_prz_pct=dist,
            use_trend_filter=False,
        )
        result = await analyzer.analyze(data)
        if result is not None:
            print(
                f"  dist={dist*100:.0f}%: {result.direction.name}, "
                f"conf={result.confidence:.3f}, "
                f"pattern={result.metadata['pattern_type']}, "
                f"prz_dist={result.metadata['distance_to_prz_pct']*100:.2f}%"
            )
        else:
            print(f"  dist={dist*100:.0f}%: сигнал НЕ найден")

    # 4. Стабильность на 20 барах
    print(f"\n{'=' * 70}")
    print("4. ПРОВЕРКА НА 20 БАРАХ (dist=3%, min_conf=0.30)")
    print(f"{'=' * 70}")
    analyzer = HarmonicAnalyzer(
        tolerance=0.20,
        min_confidence=0.30,
        max_distance_to_prz_pct=0.03,
        use_trend_filter=False,
    )

    signals_count = 0
    for offset in range(20, 0, -1):
        slice_data = data.iloc[:-offset]
        result = await analyzer.analyze(slice_data)
        if result is not None:
            signals_count += 1
            print(
                f"  Бар -{offset:2d}: {result.direction.name}, "
                f"conf={result.confidence:.3f}, "
                f"{result.metadata['pattern_type']}, "
                f"prz_dist={result.metadata['distance_to_prz_pct']*100:.2f}%"
            )

    print(f"\nВсего сигналов: {signals_count} из 20 баров "
          f"({signals_count/20*100:.1f}%)")


async def main():
    await debug_harmonic("ETH-USD", "1h", 5000)


if __name__ == "__main__":
    asyncio.run(main())