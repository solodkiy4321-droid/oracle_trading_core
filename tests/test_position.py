"""Тесты Position Manager.

Актуальная версия:
- Position.calculate_pnl учитывает комиссию 0.1% на вход и выход.
- PositionManager переводит SL в безубыток после TP1 (breakeven_after_tp).
- PositionManager подтягивает trailing-стоп после TP (trailing_after_tp).
"""

from datetime import datetime, timezone

import pytest

from src.models import SignalDirection
from src.position.position import Position, PositionStatus, CloseReason
from src.position.manager import PositionManager
from src.risk.risk_manager import TradePlan
from src.risk.stop_loss import StopLossResult
from src.risk.take_profit import TakeProfitResult, TakeProfitLevel
from src.risk.position_sizer import PositionSizeResult


def make_trade_plan(
    direction: SignalDirection = SignalDirection.BUY,
    entry: float = 100.0,
    sl: float = 98.0,
    tps: list = None,
    size: float = 10.0,
    risk: float = 100.0,
) -> TradePlan:
    """Создаёт торговый план для тестов."""
    tps = tps or [102.0, 104.0, 106.0]
    tp_levels = [
        TakeProfitLevel(
            price=tp,
            distance=abs(tp - entry),
            distance_pct=abs(tp - entry) / entry,
            ratio=float(i + 1),
            percentage=[0.5, 0.3, 0.2][i] if i < 3 else 0.0,
        )
        for i, tp in enumerate(tps)
    ]

    return TradePlan(
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profits=tps,
        position_size=size,
        risk_amount=risk,
        risk_pct=0.01,
        risk_reward_ratio=2.0,
        sl_result=StopLossResult(
            price=sl, distance=abs(entry - sl),
            distance_pct=abs(entry - sl) / entry,
            atr_value=1.5, multiplier=1.5,
        ),
        tp_result=TakeProfitResult(
            levels=tp_levels, primary_price=tps[0],
        ),
        size_result=PositionSizeResult(
            size=size, risk_amount=risk, risk_pct=0.01,
            stop_distance=abs(entry - sl),
            position_value=size * entry, leverage=1.0,
        ),
    )


# ---------- Position ----------

def test_position_initialization():
    """Позиция корректно инициализируется."""
    pos = Position(
        id="test1",
        symbol="BTC-USD",
        timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        initial_size=10.0,
        initial_stop_loss=98.0,
        stop_loss=98.0,
        take_profits=[102.0, 104.0, 106.0],
        tp_ratios=[1.0, 2.0, 3.0],
        tp_percentages=[0.5, 0.3, 0.2],
    )
    assert pos.is_open
    assert pos.current_size == 10.0
    assert pos.remaining_pct == 1.0


def test_position_sl_hit_long():
    """SL срабатывает для BUY при падении."""
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=98.0, stop_loss=98.0,
        take_profits=[102.0], tp_ratios=[1.0], tp_percentages=[1.0],
    )
    assert pos.check_sl_hit(high=101.0, low=97.0)
    assert not pos.check_sl_hit(high=101.0, low=99.0)


def test_position_sl_hit_short():
    """SL срабатывает для SELL при росте."""
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.SELL,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=102.0, stop_loss=102.0,
        take_profits=[98.0], tp_ratios=[1.0], tp_percentages=[1.0],
    )
    assert pos.check_sl_hit(high=103.0, low=99.0)
    assert not pos.check_sl_hit(high=101.0, low=99.0)


def test_position_tp_hit_long():
    """TP срабатывает для BUY при росте."""
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=98.0, stop_loss=98.0,
        take_profits=[102.0, 104.0], tp_ratios=[1.0, 2.0],
        tp_percentages=[0.5, 0.5],
    )
    hit = pos.check_tp_hit(high=103.0, low=99.0)
    assert 1 in hit
    assert 2 not in hit


def test_position_partial_close():
    """Частичное закрытие уменьшает размер."""
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=98.0, stop_loss=98.0,
        take_profits=[102.0, 104.0], tp_ratios=[1.0, 2.0],
        tp_percentages=[0.5, 0.5],
    )
    partial = pos.close_partial(1, 102.0, datetime.now(timezone.utc))
    assert partial.size == 5.0
    assert pos.current_size == 5.0
    assert pos.status == PositionStatus.PARTIALLY_CLOSED
    assert pos.realized_pnl > 0


def test_position_full_close_with_commission():
    """
    Полное закрытие.
    PnL = (102 - 100) × 10 = 20 gross.
    Комиссия: 100×10×0.001 + 102×10×0.001 = 1.0 + 1.02 = 2.02.
    Net = 20 - 2.02 = 17.98.
    """
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=98.0, stop_loss=98.0,
        take_profits=[102.0], tp_ratios=[1.0], tp_percentages=[1.0],
        commission_pct=0.001,
    )
    pnl = pos.close_full(
        102.0, datetime.now(timezone.utc), CloseReason.TAKE_PROFIT,
    )
    assert abs(pnl - 17.98) < 0.01
    assert pos.status == PositionStatus.CLOSED
    assert not pos.is_open


def test_position_breakeven():
    """Перевод в безубыток для BUY."""
    pos = Position(
        id="test1", symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY,
        entry_price=100.0, entry_time=datetime.now(timezone.utc),
        initial_size=10.0, initial_stop_loss=98.0, stop_loss=98.0,
        take_profits=[102.0], tp_ratios=[1.0], tp_percentages=[1.0],
    )
    pos.move_stop_to_breakeven(commission_pct=0.001)
    assert pos.stop_loss >= 100.0
    assert pos.stop_loss == pytest.approx(100.1, abs=0.01)


# ---------- PositionManager ----------

def test_manager_open_position():
    """Position Manager открывает позицию."""
    manager = PositionManager()
    plan = make_trade_plan()
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    assert pos is not None
    assert manager.open_count == 1
    assert pos.symbol == "BTC-USD"


def test_manager_open_position_records_opened_bar_index():
    """При открытии позиции записывается opened_bar_index."""
    manager = PositionManager()
    plan = make_trade_plan()

    for _ in range(5):
        manager.on_bar(100.0, 100.0, 100.0, datetime.now(timezone.utc))

    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    assert "opened_bar_index" in pos.metadata
    assert pos.metadata["opened_bar_index"] == manager._bar_counter


def test_manager_position_not_closed_immediately_by_timeout():
    """
    КРИТИЧЕСКИЙ ТЕСТ: позиция НЕ закрывается по timeout сразу после открытия.
    """
    manager = PositionManager(max_position_age_bars=100)
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0)

    for _ in range(500):
        manager.on_bar(100.0, 100.0, 100.0, datetime.now(timezone.utc))

    assert manager._bar_counter == 500

    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    assert pos is not None
    assert manager.open_count == 1

    for _ in range(5):
        manager.on_bar(
            bar_high=100.5, bar_low=99.5, bar_close=100.0,
            bar_time=datetime.now(timezone.utc),
        )

    assert manager.open_count == 1, (
        f"Позиция закрылась слишком рано! "
        f"opened_bar={pos.metadata.get('opened_bar_index')}, "
        f"current_bar={manager._bar_counter}"
    )
    assert pos.is_open


def test_manager_position_closed_by_timeout_after_max_age():
    """Позиция закрывается по timeout только после max_position_age_bars."""
    manager = PositionManager(max_position_age_bars=10)
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0)

    for _ in range(500):
        manager.on_bar(100.0, 100.0, 100.0, datetime.now(timezone.utc))

    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    assert pos is not None

    for _ in range(9):
        manager.on_bar(
            bar_high=100.5, bar_low=99.5, bar_close=100.0,
            bar_time=datetime.now(timezone.utc),
        )
    assert manager.open_count == 1

    manager.on_bar(
        bar_high=100.5, bar_low=99.5, bar_close=100.0,
        bar_time=datetime.now(timezone.utc),
    )
    assert manager.open_count == 0
    assert pos.close_reason == CloseReason.TIMEOUT


def test_manager_hold_returns_none():
    """HOLD не открывает позицию."""
    manager = PositionManager()
    plan = make_trade_plan(direction=SignalDirection.HOLD)
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.HOLD, plan=plan,
    )
    assert pos is None
    assert manager.open_count == 0


def test_manager_sl_hit_closes_position():
    """SL закрывает позицию при падении цены."""
    manager = PositionManager()
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0)
    manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    update = manager.on_bar(
        bar_high=100.5, bar_low=97.5, bar_close=98.0,
        bar_time=datetime.now(timezone.utc),
    )
    assert len(update.closed_positions) == 1
    assert update.closed_positions[0].close_reason == CloseReason.STOP_LOSS
    assert manager.open_count == 0


def test_manager_tp_hit_partial_close():
    """TP1 закрывает часть позиции."""
    manager = PositionManager()
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                          tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )
    assert manager.open_count == 1
    assert pos.status == PositionStatus.PARTIALLY_CLOSED
    assert pos.current_size < pos.initial_size


def test_manager_breakeven_after_tp1():
    """
    После TP1 SL переводится в безубыток.

    Проверка фикса Бага G: PositionManager.on_bar вызывает
    position.move_stop_to_breakeven при tp_num >= breakeven_after_tp.
    """
    manager = PositionManager(breakeven_after_tp=1)
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                          tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    old_sl = pos.stop_loss
    assert old_sl == 98.0

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )

    assert pos.stop_loss > old_sl, (
        f"После TP1 SL должен подняться: было {old_sl}, стало {pos.stop_loss}"
    )
    assert pos.stop_loss >= 100.0


def test_manager_multiple_positions():
    """Можно открыть несколько позиций."""
    manager = PositionManager()
    plan1 = make_trade_plan(direction=SignalDirection.BUY, entry=100.0)
    plan2 = make_trade_plan(direction=SignalDirection.SELL, entry=100.0, sl=102.0,
                           tps=[98.0, 96.0])

    manager.open_position("BTC-USD", "1h", SignalDirection.BUY, plan1)
    manager.open_position("ETH-USD", "1h", SignalDirection.SELL, plan2)
    assert manager.open_count == 2


def test_manager_stats():
    """Статистика корректна."""
    manager = PositionManager()
    plan = make_trade_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0)
    manager.open_position("BTC-USD", "1h", SignalDirection.BUY, plan)

    manager.on_bar(
        bar_high=100.5, bar_low=97.5, bar_close=98.0,
        bar_time=datetime.now(timezone.utc),
    )

    stats = manager.get_stats()
    assert stats["total"] == 1
    assert stats["losses"] == 1
    assert stats["wins"] == 0
    assert stats["win_rate"] == 0.0


def test_manager_close_all():
    """Закрытие всех позиций."""
    manager = PositionManager()
    plan = make_trade_plan()
    manager.open_position("BTC-USD", "1h", SignalDirection.BUY, plan)
    manager.open_position("ETH-USD", "1h", SignalDirection.BUY, plan)
    assert manager.open_count == 2

    closed = manager.close_all(
        100.0, datetime.now(timezone.utc), CloseReason.MANUAL,
    )
    assert len(closed) == 2
    assert manager.open_count == 0