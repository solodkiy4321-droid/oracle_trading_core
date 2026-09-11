"""Настройки лимитов риска портфеля."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RiskLimits:
    """
    Настройки лимитов риска.

    ВАЖНО: пауза ОСЛАБЛЕНА для сбора статистики:
    - pause_after_consecutive_losses = 5 (было 3)
    - pause_duration_hours = 12 (было 24)

    Это позволяет собирать больше сделок для анализа.
    """
    # Дневные лимиты
    daily_loss_limit_pct: float = 0.03
    daily_profit_target_pct: float = 0.06

    # Общие лимиты
    max_drawdown_pct: float = 0.15
    max_open_positions: int = 5
    max_positions_per_symbol: int = 1

    # Корреляционные лимиты
    max_positions_per_group: int = 2
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "crypto_major": ["BTC-USD", "BTCUSDT", "BTCUSD"],
        "crypto_alt": [
            "ETH-USD", "ETHUSDT", "SOL-USD", "SOLUSDT",
            "BNB-USD", "BNBUSDT", "XRP-USD", "XRPUSDT",
            "ADA-USD", "ADAUSDT", "DOGE-USD", "DOGEUSDT",
            "LINK-USD", "MATIC-USD",
        ],
        "us_tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "forex_major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    })

    # Лимиты на инструмент
    max_risk_per_symbol_pct: float = 0.02

    # Паузы — ОСЛАБЛЕНЫ для сбора статистики
    pause_after_consecutive_losses: int = 5    # было 3
    pause_duration_hours: int = 12             # было 24

    def validate(self) -> None:
        if not 0 < self.daily_loss_limit_pct <= 0.2:
            raise ValueError(
                f"daily_loss_limit_pct должен быть в (0, 0.2], "
                f"получено {self.daily_loss_limit_pct}"
            )
        if not 0 < self.max_drawdown_pct <= 0.5:
            raise ValueError(
                f"max_drawdown_pct должен быть в (0, 0.5], "
                f"получено {self.max_drawdown_pct}"
            )
        if self.max_open_positions < 1:
            raise ValueError(
                f"max_open_positions должен быть >= 1, "
                f"получено {self.max_open_positions}"
            )
        if self.pause_after_consecutive_losses < 1:
            raise ValueError(
                f"pause_after_consecutive_losses должен быть >= 1, "
                f"получено {self.pause_after_consecutive_losses}"
            )