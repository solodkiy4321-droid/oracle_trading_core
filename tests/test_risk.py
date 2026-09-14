"""Тесты Risk Manager.

Актуальная версия TakeProfitCalculator:
- Один уровень TP с множителем default_ratio.
- Никакого параметра levels в __init__.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.models import SignalDirection
from src.risk.stop_loss import StopLossCalculator
from src.risk.take_profit import TakeProfitCalculator
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager


# ---------- Фикстуры ----------

def make_data(n: int = 100, atr_target: float = 2.0) -> pd.DataFrame:
    """Создаёт данные с заданным ATR."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + atr_target / 2 + np.abs(np.random.randn(n) * 0.3)
    low = close - atr_target / 2 - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# ---------- StopLossCalculator ----------

def test_sl_long():
    """SL для BUY ниже entry."""
    calc = StopLossCalculator(atr_period=14, atr_multiplier=1.5)
    data = make_data()
    result = calc.calculate(
        entry_price=100.0, direction=SignalDirection.BUY, data=data,
    )

    assert result is not None
    assert result.price < 100.0
    assert result.distance > 0
    assert result.multiplier == 1.5


def test_sl_short():
    """SL для SELL выше entry."""
    calc = StopLossCalculator(atr_period=14, atr_multiplier=1.5)
    data = make_data()
    result = calc.calculate(
        entry_price=100.0, direction=SignalDirection.SELL, data=data,
    )

    assert result is not None
    assert result.price > 100.0
    assert result.distance > 0


def test_sl_hold_returns_none():
    """HOLD не даёт SL."""
    calc = StopLossCalculator()
    data = make_data()
    result = calc.calculate(
        entry_price=100.0, direction=SignalDirection.HOLD, data=data,
    )
    assert result is None


def test_sl_insufficient_data():
    """Недостаточно данных — None."""
    calc = StopLossCalculator(atr_period=14)
    data = make_data(n=10)
    result = calc.calculate(
        entry_price=100.0, direction=SignalDirection.BUY, data=data,
    )
    assert result is None


# ---------- TakeProfitCalculator (актуальный API) ----------

def test_tp_long_default_ratio():
    """
    TP для BUY выше entry.
    default_ratio=2.0, sl_distance=2.0 → TP = 100 + 2.0×2.0 = 104.0.
    """
    calc = TakeProfitCalculator(default_ratio=2.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.BUY,
    )

    assert result is not None
    assert len(result.levels) == 1
    assert result.primary_price > 100.0
    assert abs(result.levels[0].price - 104.0) < 1e-6
    assert abs(result.levels[0].ratio - 2.0) < 1e-6
    assert abs(result.levels[0].percentage - 1.0) < 1e-6


def test_tp_short_default_ratio():
    """TP для SELL ниже entry."""
    calc = TakeProfitCalculator(default_ratio=2.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.SELL,
    )

    assert result is not None
    assert result.primary_price < 100.0
    assert abs(result.levels[0].price - 96.0) < 1e-6


def test_tp_custom_ratio():
    """Кастомный default_ratio влияет на TP."""
    calc = TakeProfitCalculator(default_ratio=3.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.BUY,
    )

    assert result is not None
    assert abs(result.levels[0].price - 106.0) < 1e-6
    assert abs(result.levels[0].ratio - 3.0) < 1e-6


def test_tp_ratios_override():
    """Передача ratios переопределяет default_ratio (первый элемент)."""
    calc = TakeProfitCalculator(default_ratio=2.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.BUY,
        ratios=[1.5, 3.0],
    )

    assert result is not None
    assert abs(result.levels[0].price - 103.0) < 1e-6
    assert abs(result.levels[0].ratio - 1.5) < 1e-6


def test_tp_zero_distance():
    """Нулевое расстояние — None."""
    calc = TakeProfitCalculator()
    result = calc.calculate(
        entry_price=100.0, sl_distance=0.0, direction=SignalDirection.BUY,
    )
    assert result is None


def test_tp_hold_returns_none():
    """HOLD — None."""
    calc = TakeProfitCalculator()
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.HOLD,
    )
    assert result is None


# ---------- PositionSizer ----------

def test_position_size_basic():
    """Базовый расчёт размера позиции."""
    sizer = PositionSizer(risk_per_trade_pct=0.01)
    result = sizer.calculate(
        equity=10000.0, entry_price=100.0, stop_distance=2.0,
    )

    assert result is not None
    assert abs(result.risk_amount - 100.0) < 1e-6
    assert abs(result.size - 50.0) < 1e-6


def test_position_size_zero_equity():
    """Нулевой equity — None."""
    sizer = PositionSizer()
    result = sizer.calculate(equity=0, entry_price=100.0, stop_distance=2.0)
    assert result is None


def test_position_size_zero_stop():
    """Нулевой stop — None."""
    sizer = PositionSizer()
    result = sizer.calculate(equity=10000.0, entry_price=100.0, stop_distance=0)
    assert result is None


def test_position_size_high_leverage_warning(caplog):
    """Высокое плечо даёт предупреждение."""
    caplog.set_level(logging.WARNING)

    sizer = PositionSizer(risk_per_trade_pct=0.05, max_position_pct=10.0)
    result = sizer.calculate(equity=1000.0, entry_price=100.0, stop_distance=0.5)

    assert result is not None
    assert result.leverage > 3.0
    assert any("Высокое плечо" in r.message for r in caplog.records)


# ---------- RiskManager ----------

def test_risk_manager_full_plan():
    """Полный торговый план."""
    manager = RiskManager(
        risk_per_trade_pct=0.01,
        atr_period=14,
        atr_multiplier=1.5,
        default_rr_ratio=2.0,
    )
    data = make_data()
    plan = manager.calculate_plan(
        direction=SignalDirection.BUY,
        entry_price=100.0,
        equity=10000.0,
        data=data,
    )

    assert plan is not None
    assert plan.direction == SignalDirection.BUY
    assert plan.stop_loss < 100.0
    assert len(plan.take_profits) > 0
    assert plan.take_profits[0] > 100.0
    assert plan.position_size > 0
    assert plan.risk_amount > 0
    assert plan.risk_pct == 0.01


def test_risk_manager_hold_returns_none():
    """HOLD не даёт план."""
    manager = RiskManager()
    data = make_data()
    plan = manager.calculate_plan(
        direction=SignalDirection.HOLD,
        entry_price=100.0,
        equity=10000.0,
        data=data,
    )
    assert plan is None


def test_risk_manager_validate_buy():
    """Валидация плана BUY."""
    manager = RiskManager()
    data = make_data()
    plan = manager.calculate_plan(
        direction=SignalDirection.BUY,
        entry_price=100.0,
        equity=10000.0,
        data=data,
    )
    is_valid, errors = manager.validate_plan(plan)
    assert is_valid
    assert len(errors) == 0


def test_risk_manager_validate_sell():
    """Валидация плана SELL."""
    manager = RiskManager()
    data = make_data()
    plan = manager.calculate_plan(
        direction=SignalDirection.SELL,
        entry_price=100.0,
        equity=10000.0,
        data=data,
    )
    is_valid, errors = manager.validate_plan(plan)
    assert is_valid
    assert len(errors) == 0