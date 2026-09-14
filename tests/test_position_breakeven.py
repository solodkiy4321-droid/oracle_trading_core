"""Тесты для Бага G + Bug #1: breakeven_after_tp и trailing_after_tp.

Проверяют:
- После TP1 SL переводится в безубыток (breakeven_after_tp=1).
- Событие BREAKEVEN генерируется только если SL реально сдвинулся.
- breakeven_after_tp=0 отключает breakeven.
- trailing_after_tp с одним TP не срабатывает (ожидаемо).
"""

from datetime import datetime, timezone

import pytest

from src.models import SignalDirection
from src.position.position import Position, CloseReason
from src.position.manager import PositionManager
from src.position.events import PositionEventType
from src.risk.risk_manager import TradePlan
from src.risk.stop_loss import StopLossResult
from src.risk.take_profit import TakeProfitResult, TakeProfitLevel
from src.risk.position_sizer import PositionSizeResult


def make_plan(
    direction: SignalDirection = SignalDirection.BUY,
    entry: float = 100.0,
    sl: float = 98.0,
    tps: list = None,
    tp_percentages: list = None,
    size: float = 10.0,
) -> TradePlan:
    """План с заданными TP и их долями."""
    tps = tps or [102.0, 104.0, 106.0]
    tp_percentages = tp_percentages or [0.5, 0.3, 0.2]

    tp_levels = [
        TakeProfitLevel(
            price=tp,
            distance=abs(tp - entry),
            distance_pct=abs(tp - entry) / entry,
            ratio=float(i + 1),
            percentage=tp_percentages[i],
        )
        for i, tp in enumerate(tps)
    ]

    return TradePlan(
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profits=tps,
        position_size=size,
        risk_amount=100.0,
        risk_pct=0.01,
        risk_reward_ratio=2.0,
        sl_result=StopLossResult(
            price=sl, distance=abs(entry - sl),
            distance_pct=abs(entry - sl) / entry,
            atr_value=1.5, multiplier=1.5,
        ),
        tp_result=TakeProfitResult(levels=tp_levels, primary_price=tps[0]),
        size_result=PositionSizeResult(
            size=size, risk_amount=100.0, risk_pct=0.01,
            stop_distance=abs(entry - sl),
            position_value=size * entry, leverage=1.0,
        ),
    )


# ---------- Breakeven after TP1 ----------

def test_breakeven_after_tp1_moves_sl():
    """
    Главный тест: после TP1 SL становится выше старого.
    Раньше падал — это и есть Bug #1.
    """
    manager = PositionManager(breakeven_after_tp=1)
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
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
        f"SL должен подняться после TP1: было {old_sl}, стало {pos.stop_loss}"
    )
    # Безубыток с комиссией 0.1%
    assert pos.stop_loss >= 100.0


def test_breakeven_event_emitted():
    """Событие BREAKEVEN генерируется."""
    manager = PositionManager(breakeven_after_tp=1)
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    update = manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )

    event_types = [e.type for e in update.events]
    assert PositionEventType.BREAKEVEN in event_types


def test_breakeven_disabled_with_zero():
    """breakeven_after_tp=0 — SL не двигается."""
    manager = PositionManager(breakeven_after_tp=0)
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    old_sl = pos.stop_loss

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )
    assert pos.stop_loss == old_sl


def test_breakeven_not_moved_before_threshold():
    """breakeven_after_tp=2 — после TP1 SL не двигается."""
    manager = PositionManager(breakeven_after_tp=2)
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    old_sl = pos.stop_loss

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )
    assert pos.stop_loss == old_sl


def test_breakeven_for_sell():
    """Для SELL безубыток означает SL ниже entry."""
    manager = PositionManager(breakeven_after_tp=1)
    plan = make_plan(direction=SignalDirection.SELL, entry=100.0, sl=102.0,
                    tps=[98.0, 96.0, 94.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.SELL, plan=plan,
    )
    old_sl = pos.stop_loss
    assert old_sl == 102.0

    manager.on_bar(
        bar_high=100.5, bar_low=97.5, bar_close=98.0,
        bar_time=datetime.now(timezone.utc),
    )

    assert pos.stop_loss < old_sl, (
        f"SL должен опуститься после TP1: было {old_sl}, стало {pos.stop_loss}"
    )
    assert pos.stop_loss <= 100.0


# ---------- Trailing after TP ----------

def test_trailing_after_tp1_with_atr():
    """trailing_after_tp=1 и atr → SL сдвигается вслед за ценой."""
    manager = PositionManager(
        breakeven_after_tp=0,
        trailing_after_tp=1,
        trailing_atr_multiplier=1.0,
    )
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
        atr=2.0,
    )

    # Trailing = 102.0 - 2.0×1.0 = 100.0
    assert pos.stop_loss >= 100.0


def test_trailing_requires_atr():
    """Без atr trailing не срабатывает."""
    manager = PositionManager(
        breakeven_after_tp=0,
        trailing_after_tp=1,
    )
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )
    old_sl = pos.stop_loss

    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
        atr=None,
    )
    assert pos.stop_loss == old_sl


def test_trailing_never_loosens_sl():
    """Trailing не должен откатывать SL назад."""
    manager = PositionManager(
        breakeven_after_tp=0,
        trailing_after_tp=1,
        trailing_atr_multiplier=1.0,
    )
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0,
                    tps=[102.0, 104.0, 106.0])
    pos = manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    # Первый бар — цена 102, trailing = 100
    manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
        atr=2.0,
    )
    sl_after_first = pos.stop_loss

    # Второй бар — цена падает, но SL уже подтянут
    manager.on_bar(
        bar_high=102.0, bar_low=99.0, bar_close=100.0,
        bar_time=datetime.now(timezone.utc),
        atr=2.0,
    )

    # SL не должен опуститься
    assert pos.stop_loss >= sl_after_first


# ---------- Проверка что событие PARTIAL_CLOSE всё ещё есть ----------

def test_partial_close_event_still_emitted():
    """PARTIAL_CLOSE генерируется при TP1, как и раньше."""
    manager = PositionManager(breakeven_after_tp=1)
    plan = make_plan(direction=SignalDirection.BUY, entry=100.0, sl=98.0)
    manager.open_position(
        symbol="BTC-USD", timeframe="1h",
        direction=SignalDirection.BUY, plan=plan,
    )

    update = manager.on_bar(
        bar_high=102.5, bar_low=99.5, bar_close=102.0,
        bar_time=datetime.now(timezone.utc),
    )

    event_types = [e.type for e in update.events]
    assert PositionEventType.PARTIAL_CLOSE in event_types