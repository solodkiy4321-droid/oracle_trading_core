"""Тесты Risk Manager."""

import numpy as np
import pandas as pd
import pytest

from src.models import SignalDirection
from src.risk.stop_loss import StopLossCalculator
from src.risk.take_profit import TakeProfitCalculator
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager


# ============ Фикстуры ============

def make_data(n: int = 100, atr_target: float = 2.0) -> pd.DataFrame:
    """Создаёт данные с заданным ATR."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    # Добавляем волатильность для ATR
    high = close + atr_target / 2 + np.abs(np.random.randn(n) * 0.3)
    low = close - atr_target / 2 - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


# ============ StopLossCalculator ============

def test_sl_long():
    """SL для BUY ниже entry."""
    calc = StopLossCalculator(atr_period=14, atr_multiplier=1.5)
    data = make_data()
    result = calc.calculate(entry_price=100.0, direction=SignalDirection.BUY, data=data)

    assert result is not None
    assert result.price < 100.0
    assert result.distance > 0
    assert result.multiplier == 1.5


def test_sl_short():
    """SL для SELL выше entry."""
    calc = StopLossCalculator(atr_period=14, atr_multiplier=1.5)
    data = make_data()
    result = calc.calculate(entry_price=100.0, direction=SignalDirection.SELL, data=data)

    assert result is not None
    assert result.price > 100.0
    assert result.distance > 0


def test_sl_hold_returns_none():
    """HOLD не даёт SL."""
    calc = StopLossCalculator()
    data = make_data()
    result = calc.calculate(entry_price=100.0, direction=SignalDirection.HOLD, data=data)
    assert result is None


def test_sl_insufficient_data():
    """Недостаточно данных — None."""
    calc = StopLossCalculator(atr_period=14)
    data = make_data(n=10)
    result = calc.calculate(entry_price=100.0, direction=SignalDirection.BUY, data=data)
    assert result is None


# ============ TakeProfitCalculator ============

def test_tp_long():
    """TP для BUY выше entry."""
    calc = TakeProfitCalculator(default_ratio=2.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.BUY,
    )

    assert result is not None
    assert len(result.levels) > 0
    assert result.primary_price > 100.0
    # Первый уровень: 100 + 2.0 × 1.0 = 102.0
    assert abs(result.levels[0].price - 102.0) < 1e-6


def test_tp_short():
    """TP для SELL ниже entry."""
    calc = TakeProfitCalculator(default_ratio=2.0)
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.SELL,
    )

    assert result is not None
    assert result.primary_price < 100.0


def test_tp_multiple_levels():
    """Несколько уровней TP."""
    calc = TakeProfitCalculator(levels=[(1.0, 0.5), (2.0, 0.3), (3.0, 0.2)])
    result = calc.calculate(
        entry_price=100.0, sl_distance=2.0, direction=SignalDirection.BUY,
    )

    assert result is not None
    assert len(result.levels) == 3
    # Проверяем пропорции
    assert abs(result.levels[0].price - 102.0) < 1e-6  # 1R
    assert abs(result.levels[1].price - 104.0) < 1e-6  # 2R
    assert abs(result.levels[2].price - 106.0) < 1e-6  # 3R


def test_tp_zero_distance():
    """Нулевое расстояние — None."""
    calc = TakeProfitCalculator()
    result = calc.calculate(
        entry_price=100.0, sl_distance=0.0, direction=SignalDirection.BUY,
    )
    assert result is None


# ============ PositionSizer ============

def test_position_size_basic():
    """Базовый расчёт размера позиции."""
    sizer = PositionSizer(risk_per_trade_pct=0.01)
    result = sizer.calculate(
        equity=10000.0, entry_price=100.0, stop_distance=2.0,
    )

    assert result is not None
    # Risk = 10000 × 0.01 = 100
    # Size = 100 / 2.0 = 50
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
    import logging
    caplog.set_level(logging.WARNING)

    sizer = PositionSizer(risk_per_trade_pct=0.05, max_position_pct=10.0)
    result = sizer.calculate(equity=1000.0, entry_price=100.0, stop_distance=0.5)

    assert result is not None
    # Position value = size × entry
    # size = (1000 × 0.05) / 0.5 = 100
    # value = 100 × 100 = 10000
    # leverage = 10000 / 1000 = 10x
    assert result.leverage > 3.0
    assert any("Высокое плечо" in r.message for r in caplog.records)


# ============ RiskManager ============

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