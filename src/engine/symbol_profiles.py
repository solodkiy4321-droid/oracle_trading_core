"""Профили инструментов (5 крипто-символов).

Per-symbol фильтры + per-symbol веса анализаторов.

Веса оптимизированы через weights_sweep_per_symbol:

| Symbol | trend | elliott | volat | volume | PnL$   |
|--------|-------|---------|-------|--------|--------|
| BTC    | 0.0   | 0.7     | 0.0   | 0.3    | +5396  |
| ETH    | 0.6   | 0.2     | 0.1   | 0.1    | +5264  |
| ADA    | 0.8   | 0.0     | 0.1   | 0.1    | +5298  |
| DOT    | 0.5   | 0.2     | 0.1   | 0.2    | +5362  |
| ATOM   | 0.4   | 0.3     | 0.0   | 0.3    | +5230  |
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class SymbolProfile:
    timeframe: str = "1h"
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 1.0
    atr_multiplier: float = 3.0
    atr_period: int = 14
    default_rr_ratio: float = 2.5
    filter_chop: bool = True
    regime_1d_filter_mode: str = "baseline"
    long_only: bool = False
    gate_mode_override: Optional[str] = None
    weights_override: Optional[Dict[str, float]] = None
    notes: str = ""

    def validate(self) -> None:
        if self.timeframe not in (
            "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w",
        ):
            raise ValueError(f"timeframe неверный: {self.timeframe}")
        if not 0 < self.risk_per_trade_pct <= 0.05:
            raise ValueError("risk_per_trade_pct в (0, 0.05]")
        if not 0 < self.max_position_pct <= 2.0:
            raise ValueError("max_position_pct в (0, 2.0]")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier > 0")
        if self.atr_period < 2:
            raise ValueError("atr_period >= 2")
        if self.default_rr_ratio <= 0:
            raise ValueError("default_rr_ratio > 0")
        if not isinstance(self.filter_chop, bool):
            raise ValueError("filter_chop должен быть bool")
        if self.regime_1d_filter_mode not in (
            "none", "baseline", "drawdown_10", "sma200_only",
        ):
            raise ValueError(
                f"regime_1d_filter_mode неверный: "
                f"{self.regime_1d_filter_mode}"
            )
        if not isinstance(self.long_only, bool):
            raise ValueError("long_only должен быть bool")
        if self.gate_mode_override is not None:
            if self.gate_mode_override not in (
                "aggressive", "balanced", "conservative",
            ):
                raise ValueError("gate_mode_override неверный")
        if self.weights_override is not None:
            total = sum(self.weights_override.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Сумма weights_override = 1.0, "
                    f"получено {total:.4f}"
                )


class SymbolProfileRegistry:
    def __init__(self):
        self._profiles: Dict[str, SymbolProfile] = {}
        self._default = SymbolProfile(notes="Default profile")

    def register(self, symbol: str, profile: SymbolProfile) -> None:
        profile.validate()
        self._profiles[symbol.upper()] = profile

    def get(self, symbol: str) -> SymbolProfile:
        s = symbol.upper()
        if s in self._profiles:
            return self._profiles[s]
        for key, profile in self._profiles.items():
            base = key.split("-")[0].split("_")[0]
            if s.startswith(base):
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
    registry = SymbolProfileRegistry()

    registry.register(
        "BTC-USD",
        SymbolProfile(
            timeframe="2h",
            risk_per_trade_pct=0.01,
            max_position_pct=1.0,
            atr_multiplier=2.0,
            atr_period=14,
            default_rr_ratio=3.0,
            filter_chop=True,
            regime_1d_filter_mode="baseline",
            long_only=False,
            gate_mode_override=None,
            weights_override={
                "trend": 0.0,
                "elliott_wave": 0.7,
                "volatility": 0.0,
                "volume": 0.3,
            },
            notes="BTC: 2h, atr=2.0 rr=3.0 chop=ON 1d=baseline weights=e0.7+v0.3 (+$5396)",
        ),
    )

    registry.register(
        "ETH-USD",
        SymbolProfile(
            timeframe="1h",
            risk_per_trade_pct=0.01,
            max_position_pct=1.0,
            atr_multiplier=2.5,
            atr_period=14,
            default_rr_ratio=2.5,
            filter_chop=True,
            regime_1d_filter_mode="baseline",
            long_only=False,
            gate_mode_override="balanced",
            weights_override={
                "trend": 0.6,
                "elliott_wave": 0.2,
                "volatility": 0.1,
                "volume": 0.1,
            },
            notes="ETH: 1h, atr=2.5 rr=2.5 chop=ON 1d=baseline gate=balanced weights=t0.6 (+$5264)",
        ),
    )

    registry.register(
        "ADA-USD",
        SymbolProfile(
            timeframe="1h",
            risk_per_trade_pct=0.01,
            max_position_pct=1.0,
            atr_multiplier=3.0,
            atr_period=14,
            default_rr_ratio=2.5,
            filter_chop=False,
            regime_1d_filter_mode="none",
            long_only=False,
            gate_mode_override=None,
            weights_override={
                "trend": 0.8,
                "elliott_wave": 0.0,
                "volatility": 0.1,
                "volume": 0.1,
            },
            notes="ADA: 1h, atr=3.0 rr=2.5 chop=off 1d=none weights=t0.8 (+$5298)",
        ),
    )

    registry.register(
        "DOT-USD",
        SymbolProfile(
            timeframe="1h",
            risk_per_trade_pct=0.01,
            max_position_pct=1.0,
            atr_multiplier=2.5,
            atr_period=14,
            default_rr_ratio=2.5,
            filter_chop=False,
            regime_1d_filter_mode="none",
            long_only=False,
            gate_mode_override=None,
            weights_override={
                "trend": 0.5,
                "elliott_wave": 0.2,
                "volatility": 0.1,
                "volume": 0.2,
            },
            notes="DOT: 1h, atr=2.5 rr=2.5 chop=off 1d=none weights=t0.5 (+$5362)",
        ),
    )

    registry.register(
        "ATOM-USD",
        SymbolProfile(
            timeframe="1h",
            risk_per_trade_pct=0.01,
            max_position_pct=1.0,
            atr_multiplier=4.0,
            atr_period=14,
            default_rr_ratio=3.5,
            filter_chop=True,
            regime_1d_filter_mode="drawdown_10",
            long_only=False,
            gate_mode_override="aggressive",
            weights_override={
                "trend": 0.4,
                "elliott_wave": 0.3,
                "volatility": 0.0,
                "volume": 0.3,
            },
            notes="ATOM: 1h, atr=4.0 rr=3.5 chop=ON 1d=drawdown_10 gate=aggressive weights=t0.4 (+$5230)",
        ),
    )

    return registry