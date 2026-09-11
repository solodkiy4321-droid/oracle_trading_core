"""Тесты Portfolio Risk Manager."""

from datetime import datetime, timezone, date
from uuid import uuid4

import pytest

from src.models import SignalDirection
from src.portfolio.limits import RiskLimits
from src.portfolio.state import PortfolioState, DailyStats
from src.portfolio.guard import PortfolioGuard
from src.portfolio.manager import PortfolioRiskManager
from src.position.position import Position, CloseReason


def make_position(
    symbol: str = "BTC-USD",
    direction: SignalDirection = SignalDirection.BUY,
    entry: float = 100.0,
    size: float = 10.0,
    pnl: float = 0.0,
) -> Position:
    """Создаёт тестовую позицию."""
    pos = Position(
        id=str(uuid4())[:8],
        symbol=symbol,
        timeframe="1h",
        direction=direction,
        entry_price=entry,
        entry_time=datetime.now(timezone.utc),
        initial_size=size,
        initial_stop_loss=entry * 0.98 if direction == SignalDirection.BUY else entry * 1.02,
        stop_loss=entry * 0.98 if direction == SignalDirection.BUY else entry * 1.02,
        take_profits=[entry * 1.02, entry * 1.04],
        tp_ratios=[1.0, 2.0],
        tp_percentages=[0.5, 0.5],
        risk_amount=100.0,
        risk_pct=0.01,
    )
    # Если pnl задан — закрываем позицию
    if pnl != 0.0:
        pos.realized_pnl = pnl
        pos.status = pos.status.CLOSED
    return pos


# ============ PortfolioState ============

def test_state_initialization():
    """Состояние корректно инициализируется."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    assert state.current_drawdown_pct == 0.0
    assert state.total_pnl == 0.0
    assert not state.is_paused


def test_state_drawdown_calculation():
    """Просадка считается корректно."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=8500.0,
        peak_equity=10000.0,
    )
    assert state.current_drawdown_pct == pytest.approx(0.15, abs=0.001)


def test_state_register_trade_win():
    """Победная сделка обновляет статистику."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())
    state.register_trade_result(pnl=200.0)

    assert state.current_equity == 10200.0
    assert state.consecutive_wins == 1
    assert state.consecutive_losses == 0
    assert state.total_wins == 1
    assert state.daily_stats.realized_pnl == 200.0


def test_state_register_trade_loss():
    """Убыточная сделка обновляет статистику."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())
    state.register_trade_result(pnl=-100.0)

    assert state.current_equity == 9900.0
    assert state.consecutive_losses == 1
    assert state.consecutive_wins == 0
    assert state.total_losses == 1


def test_state_consecutive_losses():
    """Серия убытков отслеживается."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())

    for _ in range(3):
        state.register_trade_result(pnl=-50.0)

    assert state.consecutive_losses == 3


def test_state_pause():
    """Пауза работает."""
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.pause("test", hours=1)
    assert state.is_paused
    assert state.pause_reason == "test"


# ============ PortfolioGuard ============

def test_guard_allows_normal_trade():
    """Guard разрешает сделку в нормальных условиях."""
    limits = RiskLimits()
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())

    result = guard.check(state, "BTC-USD", risk_pct=0.01)
    assert result.allowed
    assert result.reason == ""


def test_guard_blocks_daily_loss():
    """Guard блокирует при достижении дневного лимита."""
    limits = RiskLimits(daily_loss_limit_pct=0.03)
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=9700.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())
    state.daily_stats.realized_pnl = -300.0  # -3%

    result = guard.check(state, "BTC-USD")
    assert not result.allowed
    assert "Дневной лимит потерь" in result.reason


def test_guard_blocks_drawdown():
    """Guard блокирует при критической просадке."""
    limits = RiskLimits(max_drawdown_pct=0.15)
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=8400.0,  # -16% от пика
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())

    result = guard.check(state, "BTC-USD")
    assert not result.allowed
    assert "просадка" in result.reason.lower()


def test_guard_blocks_max_positions():
    """Guard блокирует при максимуме позиций."""
    limits = RiskLimits(max_open_positions=3)
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
        open_positions_by_symbol={"BTC-USD": 1, "ETH-USD": 1, "AAPL": 1},
    )
    state.start_new_day(date.today())

    result = guard.check(state, "SOL-USD")
    assert not result.allowed
    assert "Максимум открытых позиций" in result.reason


def test_guard_blocks_per_symbol():
    """Guard блокирует превышение позиций по инструменту."""
    limits = RiskLimits(max_positions_per_symbol=1)
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
        open_positions_by_symbol={"BTC-USD": 1},
    )
    state.start_new_day(date.today())

    result = guard.check(state, "BTC-USD")
    assert not result.allowed
    assert "по BTC-USD" in result.reason


def test_guard_blocks_correlation_group():
    """Guard блокирует превышение в корреляционной группе."""
    limits = RiskLimits(
        max_positions_per_group=2,
        correlation_groups={"crypto": ["BTC-USD", "ETH-USD", "SOL-USD"]},
    )
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
        open_positions_by_symbol={"BTC-USD": 1, "ETH-USD": 1},
        open_positions_by_group={"crypto": 2},
    )
    state.start_new_day(date.today())

    result = guard.check(state, "SOL-USD")
    assert not result.allowed
    assert "группе crypto" in result.reason


def test_guard_blocks_paused():
    """Guard блокирует при паузе."""
    limits = RiskLimits()
    guard = PortfolioGuard(limits)
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10000.0,
        peak_equity=10000.0,
    )
    state.start_new_day(date.today())
    state.pause("test pause", hours=1)

    result = guard.check(state, "BTC-USD")
    assert not result.allowed
    assert "паузе" in result.reason.lower()


# ============ PortfolioRiskManager ============

def test_manager_initialization():
    """Менеджер корректно инициализируется."""
    manager = PortfolioRiskManager(starting_equity=10000.0)
    assert manager.state.current_equity == 10000.0
    assert not manager.state.is_paused


def test_manager_can_open():
    """Менеджер разрешает сделку."""
    manager = PortfolioRiskManager(starting_equity=10000.0)
    result = manager.can_open("BTC-USD", risk_pct=0.01)
    assert result.allowed


def test_manager_register_position_opened():
    """Регистрация открытия позиции."""
    manager = PortfolioRiskManager(starting_equity=10000.0)
    pos = make_position(symbol="BTC-USD")

    manager.register_position_opened(pos)
    assert manager.state.open_positions_by_symbol["BTC-USD"] == 1
    assert manager.state.total_open_positions == 1


def test_manager_register_position_closed_win():
    """Регистрация закрытия победной позиции."""
    manager = PortfolioRiskManager(starting_equity=10000.0)
    pos = make_position(symbol="BTC-USD", pnl=200.0)

    manager.register_position_opened(pos)
    manager.register_position_closed(pos)

    assert manager.state.current_equity == 10200.0
    assert manager.state.total_wins == 1
    assert "BTC-USD" not in manager.state.open_positions_by_symbol


def test_manager_pause_after_consecutive_losses():
    """Пауза после серии убытков."""
    limits = RiskLimits(pause_after_consecutive_losses=3)
    manager = PortfolioRiskManager(starting_equity=10000.0, limits=limits)

    for _ in range(3):
        pos = make_position(symbol="BTC-USD", pnl=-100.0)
        manager.register_position_opened(pos)
        manager.register_position_closed(pos)

    assert manager.state.is_paused
    assert "Серия убытков" in manager.state.pause_reason


def test_manager_snapshot():
    """Снимок состояния корректный."""
    manager = PortfolioRiskManager(starting_equity=10000.0)
    pos = make_position(symbol="BTC-USD", pnl=100.0)
    manager.register_position_opened(pos)
    manager.register_position_closed(pos)

    snapshot = manager.get_snapshot()
    assert snapshot["starting_equity"] == 10000.0
    assert snapshot["current_equity"] == 10100.0
    assert snapshot["total_trades"] == 1
    assert snapshot["total_wins"] == 1