"""Профили адаптивных параметров для инструментов."""

import logging
from dataclasses import dataclass
from typing import Optional, Dict

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


class SymbolProfileRegistry:
    """Реестр профилей."""

    def __init__(self):
        self._profiles: Dict[str, SymbolProfile] = {}
        self._default = SymbolProfile(notes="Default profile")

    def register(self, symbol: str, profile: SymbolProfile) -> None:
        profile.validate()
        self._profiles[symbol.upper()] = profile
        logger.info("Профиль %s зарегистрирован", symbol)

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
    Реестр с базовыми профилями.

    ВАЖНО: сейчас приоритет — МОЩНОСТЬ системы.
    Все параметры возвращены к базовым, чтобы системы торговали.

    После измерения качества компонентов — настроим веса под Gate 0.75.
    """
    registry = SymbolProfileRegistry()

    # BTC-USD — базовые параметры для работы
    registry.register(
        "BTC-USD",
        SymbolProfile(
            risk_per_trade_pct=0.007,
            max_position_pct=0.5,
            atr_multiplier=2.5,
            atr_period=14,
            default_rr_ratio=2.0,
            gate_mode_override=None,      # ← базовый Gate 0.65
            weights_override=None,        # ← базовые веса
            notes="BTC: базовые параметры для сбора статистики",
        ),
    )

    # ETH-USD — базовые параметры
    registry.register(
        "ETH-USD",
        SymbolProfile(
            risk_per_trade_pct=0.01,
            max_position_pct=0.75,
            atr_multiplier=1.5,
            atr_period=14,
            default_rr_ratio=2.0,
            gate_mode_override=None,
            weights_override=None,
            notes="ETH: базовые параметры",
        ),
    )

    # AAPL — базовые параметры
    registry.register(
        "AAPL",
        SymbolProfile(
            risk_per_trade_pct=0.015,
            max_position_pct=1.0,
            atr_multiplier=2.0,
            atr_period=14,
            default_rr_ratio=2.0,
            gate_mode_override=None,
            weights_override=None,
            notes="AAPL: базовые параметры",
        ),
    )

    return registry