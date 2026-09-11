"""Модели записей для Journal."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TradeRecord:
    """Запись о сделке (открытой или закрытой позиции)."""
    id: Optional[int] = None
    position_id: str = ""
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""              # 'BUY' / 'SELL'

    # Вход
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    initial_size: float = 0.0

    # Стоп-лосс
    initial_stop_loss: float = 0.0
    final_stop_loss: float = 0.0

    # Тейк-профиты
    take_profits: str = ""           # JSON-строка со списком

    # Выход
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""            # 'tp', 'sl', 'timeout', 'manual'

    # Результат
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0

    # Риск
    risk_amount: float = 0.0
    risk_pct: float = 0.0

    # Контекст
    regime: str = ""                 # 'BULL' / 'BEAR' / 'CHOP'
    confluence_score: float = 0.0
    metadata: str = ""               # JSON-строка

    # Системное
    created_at: Optional[datetime] = None


@dataclass
class EventRecord:
    """Запись о событии позиции."""
    id: Optional[int] = None
    position_id: str = ""
    symbol: str = ""
    event_type: str = ""             # 'opened', 'partial_close', 'closed', etc.
    timestamp: Optional[datetime] = None
    price: float = 0.0
    size: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    details: str = ""                # JSON-строка


@dataclass
class PortfolioSnapshot:
    """Снимок состояния портфеля."""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    date: str = ""                   # YYYY-MM-DD

    starting_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0

    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0

    open_positions: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0

    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    is_paused: int = 0
    pause_reason: str = ""


@dataclass
class DecisionRecord:
    """Запись о решении Confluence-движка."""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    symbol: str = ""
    timeframe: str = ""

    regime: str = ""
    direction: str = ""              # 'BUY' / 'SELL' / 'HOLD'
    confluence_score: float = 0.0
    bull_score: float = 0.0
    bear_score: float = 0.0

    passed_gate: int = 0             # 0 или 1
    signals_count: int = 0
    weights: str = ""                # JSON-строка
    reasons: str = ""                # JSON-строка