"""Trading Engine с поддержкой профилей инструментов."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from src.engine.config import EngineConfig
from src.engine.symbol_profiles import SymbolProfile
from src.models import SignalDirection
from src.signal_intake.intake import SignalIntake
from src.analyzers.indicators import IndicatorsAnalyzer
from src.analyzers.harmonic import HarmonicAnalyzer
from src.analyzers.support_resistance import SRAnalyzer
from src.confluence.engine import ConfluenceEngine, ConfluenceDecision
from src.confluence.gate import GateMode
from src.risk.risk_manager import RiskManager, TradePlan
from src.position.manager import PositionManager, PositionUpdate
from src.position.position import Position, CloseReason
from src.position.events import PositionEvent, PositionEventType
from src.portfolio.manager import PortfolioRiskManager
from src.portfolio.limits import RiskLimits
from src.portfolio.state import PortfolioState
from src.journal.journal import Journal
from src.journal.analytics import JournalAnalytics

logger = logging.getLogger(__name__)


@dataclass
class BarResult:
    bar_index: int = 0
    bar_time: Optional[datetime] = None
    bar_close: float = 0.0
    closed_positions: List[Position] = field(default_factory=list)
    realized_pnl: float = 0.0
    opened_position: Optional[Position] = None
    decision: Optional[ConfluenceDecision] = None
    events: List[PositionEvent] = field(default_factory=list)
    was_blocked_by_guard: bool = False
    guard_reason: str = ""


class TradingEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.config.validate()

        self.profile: SymbolProfile = self.config.get_symbol_profile()

        logger.info(
            "TradingEngine: %s %s, equity=%.2f",
            self.config.symbol, self.config.timeframe,
            self.config.starting_equity,
        )
        logger.info(
            "Профиль: risk=%.2f%%, atr=%.2f, disabled_patterns=%s",
            self.profile.risk_per_trade_pct * 100,
            self.profile.atr_multiplier,
            self.profile.harmonic_disabled_patterns,
        )

        self._init_journal()
        self._init_portfolio()
        self._init_position_manager()
        self._init_risk_manager()
        self._init_intake()
        self._init_confluence()

        self._bar_counter = 0

    def _init_journal(self) -> None:
        self.journal = Journal(self.config.journal_db_path)
        self.analytics = JournalAnalytics(self.journal.db)

    def _init_portfolio(self) -> None:
        limits = RiskLimits(
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            daily_profit_target_pct=self.config.daily_profit_target_pct,
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_open_positions=self.config.max_open_positions,
            max_positions_per_symbol=self.config.max_positions_per_symbol,
            max_positions_per_group=self.config.max_positions_per_group,
            pause_after_consecutive_losses=self.config.pause_after_consecutive_losses,
            pause_duration_hours=self.config.pause_duration_hours,
        )
        self.portfolio = PortfolioRiskManager(
            starting_equity=self.config.starting_equity,
            limits=limits,
        )

    def _init_position_manager(self) -> None:
        self.position_manager = PositionManager(
            breakeven_after_tp=self.config.breakeven_after_tp,
            trailing_after_tp=self.config.trailing_after_tp,
            trailing_atr_multiplier=self.config.trailing_atr_multiplier,
            max_position_age_bars=self.config.max_position_age_bars,
            commission_pct=self.config.commission_pct,
        )

    def _init_risk_manager(self) -> None:
        self.risk_manager = RiskManager(
            risk_per_trade_pct=self.profile.risk_per_trade_pct,
            atr_period=self.profile.atr_period,
            atr_multiplier=self.profile.atr_multiplier,
            default_rr_ratio=self.profile.default_rr_ratio,
            max_position_pct=self.profile.max_position_pct,
        )

    def _init_intake(self) -> None:
        """Инициализация Signal Intake с параметрами из профиля."""
        self.intake = SignalIntake(timeout=5.0)

        if self.config.enable_indicators:
            self.intake.register(IndicatorsAnalyzer(name="indicators"))
        if self.config.enable_harmonic:
            # NEW: передаём disabled_patterns в HarmonicAnalyzer
            self.intake.register(HarmonicAnalyzer(
                name="harmonic",
                disabled_patterns=self.profile.harmonic_disabled_patterns,
            ))
        if self.config.enable_support_resistance:
            self.intake.register(SRAnalyzer(name="support_resistance"))

        logger.info(
            "Анализаторы: %s",
            [a.name for a in self.intake._collector.analyzers],
        )

    def _init_confluence(self) -> None:
        gate_mode_map = {
            "aggressive": GateMode.AGGRESSIVE,
            "balanced": GateMode.BALANCED,
            "conservative": GateMode.CONSERVATIVE,
        }
        gate_mode_str = self.profile.gate_mode_override or self.config.gate_mode
        gate_mode = gate_mode_map.get(gate_mode_str, GateMode.BALANCED)

        self.confluence = ConfluenceEngine()

        if self.profile.weights_override is not None:
            self.confluence.set_base_weights(self.profile.weights_override)
            logger.info("weights_override: %s", self.profile.weights_override)

        self.confluence.set_gate_mode(gate_mode)

    async def on_bar(self, data, bar_index, bar_time=None):
        self._bar_counter += 1
        result = BarResult(bar_index=bar_index, bar_time=bar_time)

        if len(data) < 50:
            return result

        current_bar = data.iloc[-1]
        bar_high = float(current_bar["high"])
        bar_low = float(current_bar["low"])
        bar_close = float(current_bar["close"])
        result.bar_close = bar_close

        atr = self._calculate_atr(data)
        position_update = self.position_manager.on_bar(
            bar_high=bar_high, bar_low=bar_low, bar_close=bar_close,
            bar_time=bar_time or datetime.now(timezone.utc),
            atr=atr,
        )
        result.events = position_update.events

        for closed_pos in position_update.closed_positions:
            self._on_position_closed(closed_pos)
            result.closed_positions.append(closed_pos)
            result.realized_pnl += closed_pos.realized_pnl

        guard_result = self.portfolio.can_open(
            symbol=self.config.symbol,
            risk_pct=self.profile.risk_per_trade_pct,
        )
        if not guard_result.allowed:
            result.was_blocked_by_guard = True
            result.guard_reason = guard_result.reason
            return result

        batch = await self.intake.process(
            data, symbol=self.config.symbol, timeframe=self.config.timeframe,
        )
        decision = self.confluence.decide(batch.signals, data)
        result.decision = decision

        self.journal.log_decision(
            decision=decision, symbol=self.config.symbol,
            timeframe=self.config.timeframe, signals_count=batch.count,
        )

        if decision.passed and decision.direction != SignalDirection.HOLD:
            opened = self._try_open_position(
                decision=decision, data=data,
                current_price=bar_close,
                bar_time=bar_time or datetime.now(timezone.utc),
            )
            if opened is not None:
                result.opened_position = opened

        if self._bar_counter % self.config.snapshot_every_n_bars == 0:
            self.journal.log_portfolio_snapshot(self.portfolio.state)

        return result

    def _try_open_position(self, decision, data, current_price, bar_time):
        plan = self.risk_manager.calculate_plan(
            direction=decision.direction,
            entry_price=current_price,
            equity=self.portfolio.state.current_equity,
            data=data,
        )
        if plan is None:
            return None

        is_valid, errors = self.risk_manager.validate_plan(plan)
        if not is_valid:
            logger.warning("Невалидный план: %s", errors)
            return None

        position = self.position_manager.open_position(
            symbol=self.config.symbol, timeframe=self.config.timeframe,
            direction=decision.direction, plan=plan,
            current_time=bar_time,
            metadata={"regime": decision.regime.value,
                     "confluence_score": decision.confluence_score},
        )
        if position is None:
            return None

        self.portfolio.register_position_opened(position)
        self.journal.log_position_opened(
            position=position, regime=decision.regime.value,
            confluence_score=decision.confluence_score,
        )
        return position

    def _on_position_closed(self, position):
        self.portfolio.register_position_closed(position)
        self.journal.log_position_closed(position)

    def _calculate_atr(self, data):
        import pandas_ta_classic as ta
        if len(data) < self.profile.atr_period + 1:
            return None
        atr = ta.atr(
            data["high"], data["low"], data["close"],
            length=self.profile.atr_period,
        )
        if atr is None or len(atr) == 0:
            return None
        return float(atr.iloc[-1])

    def start_new_day(self):
        self.portfolio.start_new_day()
        self.journal.log_portfolio_snapshot(self.portfolio.state)

    def close_all_positions(self, price):
        closed = self.position_manager.close_all(
            price, datetime.now(timezone.utc), CloseReason.MANUAL,
        )
        for pos in closed:
            self._on_position_closed(pos)
        return closed

    def get_stats(self):
        snapshot = self.portfolio.get_snapshot()
        position_stats = self.position_manager.get_stats()
        trade_stats = self.analytics.get_trade_stats()
        return {
            "portfolio": snapshot,
            "positions": position_stats,
            "trades": {
                "total": trade_stats.total_trades,
                "wins": trade_stats.wins,
                "losses": trade_stats.losses,
                "win_rate": trade_stats.win_rate,
                "profit_factor": trade_stats.profit_factor,
                "total_pnl": trade_stats.total_pnl,
                "avg_win": trade_stats.avg_win,
                "avg_loss": trade_stats.avg_loss,
            },
            "profile": {
                "symbol": self.config.symbol,
                "risk_per_trade_pct": self.profile.risk_per_trade_pct,
                "atr_multiplier": self.profile.atr_multiplier,
                "gate_mode_override": self.profile.gate_mode_override,
                "weights_override": self.profile.weights_override,
                "harmonic_disabled_patterns": self.profile.harmonic_disabled_patterns,
                "notes": self.profile.notes,
            },
        }

    def close(self):
        self.journal.close()
        logger.info("TradingEngine закрыт")