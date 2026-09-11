"""Профили адаптивных параметров для инструментов.

Данные 7 инструментов × 10000 баров:
- ADA: +7.21% (Default) ⭐
- AVAX: +6.45% (Default) ⭐
- BTC: +6.52% (кастомный ATR 2.5)
- ETH: +5.25% (кастомный ATR 1.5)
- AAPL: +2.32% (кастомный ATR 2.0)
- SOL: −0.82% (Default) → НУЖЕН СВОЙ ПРОФИЛЬ
- MATIC: делистинг

ВЫВОД: Default profile работает для большинства инструментов.
Кастомный нужен только для BTC, ETH, AAPL, SOL.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class SymbolProfile:
    """Профиль адаптивных параметров для инструмента."""
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 1.0
    atr_multiplier: float = 1.5
    atr_period: int = 14
    default_rr_ratio: float = 2.0
    gate_mode_override: Optional[str] = None
    weights_override: Optional[Dict[str, float]] = None
    harmonic_disabled_patterns: Optional[List[str]] = None
    notes: str = ""

    def validate(self) -> None:
        if not 0 < self.risk_per_trade_pct <= 0.05:
            raise ValueError(f"risk_per_trade_pct в (0, 0.05], получено {self.risk_per_trade_pct}")
        if not 0 < self.max_position_pct <= 2.0:
            raise ValueError(f"max_position_pct в (0, 2.0], получено {self.max_position_pct}")
        if self.atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier > 0, получено {self.atr_multiplier}")
        if self.atr_period < 2:
            raise ValueError(f"atr_period >= 2, получено {self.atr_period}")
        if self.default_rr_ratio <= 0:
            raise ValueError(f"default_rr_ratio > 0, получено {self.default_rr_ratio}")
        if self.gate_mode_override is not None:
            if self.gate_mode_override not in ("aggressive", "balanced", "conservative"):
                raise ValueError(f"gate_mode_override неверный: {self.gate_mode_override}")
        if self.weights_override is not None:
            total = sum(self.weights_override.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Сумма weights_override = 1.0, получено {total}")
        if self.harmonic_disabled_patterns is not None:
            valid = {"Gartley", "Bat", "Butterfly", "Crab"}
            for p in self.harmonic_disabled_patterns:
                if p not in valid:
                    raise ValueError(f"Неизвестный паттерн: {p}")


class SymbolProfileRegistry:
    def __init__(self):
        self._profiles: Dict[str, SymbolProfile] = {}
        self._default = SymbolProfile(notes="Default profile")

    def register(self, symbol: str, profile: SymbolProfile) -> None:
        profile.validate()
        self._profiles[symbol.upper()] = profile

    def get(self, symbol: str) -> SymbolProfile:
        symbol_upper = symbol.upper()
        if symbol_upper in self._profiles:
            return self._profiles[symbol_upper]
        for key, profile in self._profiles.items():
            base = key.split("-")[0].split("_")[0]
            if symbol_upper.startswith(base):
                return profile
        return self._default

    def get_default(self) -> SymbolProfile:
        return self._default

    def set_default(self, profile: SymbolProfile) -> None:
        profile.validate()
        self._default = profile

    @property
    def all_profiles(self) -> Dict[str, SymbolProfile]:
        return dict(self._profiles)


def create_default_registry() -> SymbolProfileRegistry:
    """
    Реестр с профилями под данные 7 инструментов.

    Оставляем кастомные только для инструментов, где есть проблема:
    - BTC: кастомный ATR 2.5 (работает +6.52%)
    - ETH: кастомный ATR 1.5 (работает +5.25%)
    - SOL: НОВЫЙ кастомный ATR 2.0 (было −0.82%)
    - AAPL: кастомный ATR 2.0 (работает +2.32%)

    Остальные (ADA, AVAX) — Default (работают отлично).
    """
    registry = SymbolProfileRegistry()

    # BTC-USD — проверено
    registry.register(
        "BTC-USD",
        SymbolProfile(
            risk_per_trade_pct=0.007,
            max_position_pct=0.5,
            atr_multiplier=2.5,
            atr_period=14,
            default_rr_ratio=2.0,
            notes="BTC: +6.52%, 63% WR, PF 2.25",
        ),
    )

    # ETH-USD — проверено
    registry.register(
        "ETH-USD",
        SymbolProfile(
            risk_per_trade_pct=0.01,
            max_position_pct=0.75,
            atr_multiplier=1.5,
            atr_period=14,
            default_rr_ratio=2.0,
            notes="ETH: +5.25%, 55% WR, PF 1.30",
        ),
    )

    # SOL-USD — НОВЫЙ профиль (было −0.82% с Default)
    registry.register(
        "SOL-USD",
        SymbolProfile(
            risk_per_trade_pct=0.007,       # ниже — волатильный
            max_position_pct=0.5,           # ограничение
            atr_multiplier=2.0,             # шире SL
            atr_period=14,
            default_rr_ratio=2.0,
            notes="SOL: тест ATR 2.0 (было −0.82%)",
        ),
    )

    # AAPL — проверено
    registry.register(
        "AAPL",
        SymbolProfile(
            risk_per_trade_pct=0.015,
            max_position_pct=1.0,
            atr_multiplier=2.0,
            atr_period=14,
            default_rr_ratio=2.0,
            notes="AAPL: +2.32%, 100% WR",
        ),
    )

    # ADA, AVAX — Default (работают отлично)

    return registry