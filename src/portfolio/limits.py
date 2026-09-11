"""Настройки лимитов риска портфеля."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RiskLimits:
    """
    Настройки лимитов риска.

    Все проценты указываются как доли (0.03 = 3%).
    """
    # Дневные лимиты
    daily_loss_limit_pct: float = 0.03        # -3% за день → стоп торговли
    daily_profit_target_pct: float = 0.06     # +6% за день → стоп торговли (опционально)

    # Общие лимиты
    max_drawdown_pct: float = 0.15            # -15% от пика → пауза
    max_open_positions: int = 5               # Максимум одновременных позиций
    max_positions_per_symbol: int = 1         # Максимум позиций по одному инструменту

    # Корреляционные лимиты
    max_positions_per_group: int = 2          # Максимум позиций в коррелированной группе
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "crypto_major": ["BTC-USD", "BTCUSDT", "BTCUSD"],
        "crypto_alt": ["ETH-USD", "ETHUSDT", "SOL-USD", "SOLUSDT", "BNB-USD"],
        "us_tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        "forex_major": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
    })

    # Лимиты на инструмент
    max_risk_per_symbol_pct: float = 0.02     # Максимум 2% риска на один инструмент

    # Паузы
    pause_after_consecutive_losses: int = 3   # Пауза после N убытков подряд
    pause_duration_hours: int = 24            # Длительность паузы в часах

    def validate(self) -> None:
        """Проверяет корректность настроек."""
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