"""Анализаторы рыночных данных.

Активные (4):
- ТРЕНД:          TrendAnalyzer (EMA200 + ADX)
- ПАТТЕРНЫ:       ElliottWaveAnalyzer (волны Эллиотта)
- ВОЛАТИЛЬНОСТЬ:  VolatilityAnalyzer (ATR squeeze/expansion)
- ОБЪЁМ:          VolumeAnalyzer (OBV + дивергенции)

Архив (не активны, но модули сохранены):
- HarmonicAnalyzer — adversarial к trend (dir_agree 0.06), PF 1.13
- SRAnalyzer — PF 1.06, убыточен с комиссией
- MomentumAnalyzer — гипотеза провалилась

Фильтр (не анализатор):
- TrendFilter (SMA200) — штраф для elliott_wave
"""

from .base import BaseAnalyzer
from .trend import TrendAnalyzer
from .elliott_wave import ElliottWaveAnalyzer, ElliottWaveDetector, WavePattern
from .volatility import VolatilityAnalyzer
from .volume import VolumeAnalyzer
from .trend_filter import TrendFilter, TrendDirection

__all__ = [
    "BaseAnalyzer",
    "TrendAnalyzer",
    "ElliottWaveAnalyzer",
    "ElliottWaveDetector",
    "WavePattern",
    "VolatilityAnalyzer",
    "VolumeAnalyzer",
    "TrendFilter",
    "TrendDirection",
]