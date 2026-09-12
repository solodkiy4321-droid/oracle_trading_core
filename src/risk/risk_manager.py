"""Risk Manager: расчёт параметров сделки."""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd

from src.models import SignalDirection
from src.risk.stop_loss import StopLossCalculator, StopLossResult
from src.risk.take_profit import TakeProfitCalculator, TakeProfitResult
from src.risk.position_sizer import PositionSizer, PositionSizeResult

logger = logging.getLogger(__name__)


@dataclass
class TradePlan:
    """Полный торговый план."""
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profits: List[float]
    position_size: float
    risk_amount: float
    risk_pct: float
    risk_reward_ratio: float
    sl_result: Optional[StopLossResult] = None
    tp_result: Optional[TakeProfitResult] = None
    size_result: Optional[PositionSizeResult] = None
    warnings: List[str] = field(default_factory=list)


class RiskManager:

    def __init__(
        self,
        risk_per_trade_pct: float = 0.01,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        default_rr_ratio: float = 2.0,
        max_leverage: float = 3.0,
        max_position_pct: float = 1.0,
    ):
        self.stop_loss_calc = StopLossCalculator(
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
        )
        self.take_profit_calc = TakeProfitCalculator(
            default_ratio=default_rr_ratio,
        )
        self.position_sizer = PositionSizer(
            risk_per_trade_pct=risk_per_trade_pct,
            max_position_pct=max_position_pct,
        )
        self.max_leverage = max_leverage
        self.default_rr_ratio = default_rr_ratio

    def calculate_plan(
        self,
        direction: SignalDirection,
        entry_price: float,
        equity: float,
        data: pd.DataFrame,
        atr_multiplier: Optional[float] = None,
        risk_pct: Optional[float] = None,
        tp_ratios: Optional[List[float]] = None,
    ) -> Optional[TradePlan]:
        if direction == SignalDirection.HOLD:
            logger.debug("Направление HOLD — план не рассчитывается")
            return None

        warnings = []

        sl_result = self.stop_loss_calc.calculate(
            entry_price=entry_price,
            direction=direction,
            data=data,
            multiplier=atr_multiplier,
        )
        if sl_result is None:
            logger.warning("Не удалось рассчитать стоп-лосс")
            return None

        size_result = self.position_sizer.calculate(
            equity=equity,
            entry_price=entry_price,
            stop_distance=sl_result.distance,
            risk_pct=risk_pct,
        )
        if size_result is None:
            logger.warning("Не удалось рассчитать размер позиции")
            return None

        tp_result = self.take_profit_calc.calculate(
            entry_price=entry_price,
            sl_distance=sl_result.distance,
            direction=direction,
            ratios=tp_ratios,
        )
        if tp_result is None:
            logger.warning("Не удалось рассчитать тейк-профит")
            return None

        if size_result.leverage > self.max_leverage:
            warnings.append(
                f"Высокое плечо: {size_result.leverage:.2f}x "
                f"(порог {self.max_leverage:.2f}x)"
            )

        if sl_result.distance_pct > 0.1:
            warnings.append(
                f"Стоп-лосс слишком далеко: "
                f"{sl_result.distance_pct * 100:.1f}% от цены входа"
            )

        plan = TradePlan(
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl_result.price,
            take_profits=[tp.price for tp in tp_result.levels],
            position_size=size_result.size,
            risk_amount=size_result.risk_amount,
            risk_pct=size_result.risk_pct,
            risk_reward_ratio=tp_result.levels[0].ratio if tp_result.levels else 0.0,
            sl_result=sl_result,
            tp_result=tp_result,
            size_result=size_result,
            warnings=warnings,
        )

        logger.info(
            "Торговый план: %s, entry=%.4f, SL=%.4f (%.2f%%), "
            "TP=%.4f (1:%.2f), size=%.6f, risk=%.2f (%.2f%%)",
            direction.name, entry_price,
            sl_result.price, sl_result.distance_pct * 100,
            tp_result.levels[0].price if tp_result.levels else 0,
            plan.risk_reward_ratio,
            size_result.size, size_result.risk_amount,
            size_result.risk_pct * 100,
        )

        for w in warnings:
            logger.warning("Предупреждение: %s", w)

        return plan

    def validate_plan(self, plan: TradePlan) -> tuple:
        errors = []

        if plan.entry_price <= 0:
            errors.append("Entry price <= 0")
        if plan.stop_loss <= 0:
            errors.append("Stop loss <= 0")

        if plan.direction == SignalDirection.BUY:
            if plan.stop_loss >= plan.entry_price:
                errors.append("SL должен быть ниже entry для BUY")
            for tp in plan.take_profits:
                if tp <= plan.entry_price:
                    errors.append(f"TP {tp} должен быть выше entry для BUY")
        elif plan.direction == SignalDirection.SELL:
            if plan.stop_loss <= plan.entry_price:
                errors.append("SL должен быть выше entry для SELL")
            for tp in plan.take_profits:
                if tp >= plan.entry_price:
                    errors.append(f"TP {tp} должен быть ниже entry для SELL")

        if plan.position_size <= 0:
            errors.append("Position size <= 0")
        if plan.risk_pct > 0.05:
            errors.append(f"Риск слишком высокий: {plan.risk_pct * 100:.2f}%")

        return len(errors) == 0, errors