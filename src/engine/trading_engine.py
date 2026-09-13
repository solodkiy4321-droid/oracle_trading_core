"""Trading Engine: 4 анализатора + CHOP + long_only + per-symbol 1d режим."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import pandas_ta_classic as ta

from src.engine.config import EngineConfig
from src.engine.symbol_profiles import SymbolProfile
from src.models import SignalDirection
from src.signal_intake.intake import SignalIntake
from src.analyzers.trend import TrendAnalyzer
from src.analyzers.elliott_wave import ElliottWaveAnalyzer
from src.analyzers.volatility import VolatilityAnalyzer
from src.analyzers.volume import VolumeAnalyzer
from src.confluence.engine import ConfluenceEngine, ConfluenceDecision
from src.confluence.gate import GateMode
from src.confluence.regime_detector import RegimeDetector, MarketRegime
from src.risk.risk_manager import RiskManager
from src.position.manager import PositionManager
from src.position.position import Position, CloseReason
from src.position.events import PositionEvent
from src.portfolio.manager import PortfolioRiskManager
from src.portfolio.limits import RiskLimits
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
    was_blocked_by_chop: bool = False
    was_blocked_by_long_only: bool = False
    was_blocked_by_regime_1d: bool = False
    guard_reason: str = ""


class TradingEngine:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        df_1d: Optional[pd.DataFrame] = None,
    ):
        self.config = config or EngineConfig()
        self.config.validate()
        self.profile: SymbolProfile = self.config.get_symbol_profile()
        self._timeframe = self.profile.timeframe or self.config.timeframe
        self._df_1d = df_1d
        self._regime_1d_detector = RegimeDetector()
        self._blocked_by_1d = 0

        logger.info(
            "TradingEngine: %s %s, equity=%.2f, "
            "filter_chop=%s, long_only=%s, regime_1d_mode=%s, df_1d=%s",
            self.config.symbol, self._timeframe,
            self.config.starting_equity,
            self.profile.filter_chop, self.profile.long_only,
            self.profile.regime_1d_filter_mode,
            len(df_1d) if df_1d is not None else 0,
        )
        self._init_journal()
        self._init_portfolio()
        self._init_position_manager()
        self._init_risk_manager()
        self._init_intake()
        self._init_confluence()
        self._init_regime_detector()
        self._bar_counter = 0

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def blocked_by_1d(self) -> int:
        return self._blocked_by_1d

    def _init_journal(self):
        self.journal = Journal(self.config.journal_db_path)
        self.analytics = JournalAnalytics(self.journal.db)

    def _init_portfolio(self):
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
            starting_equity=self.config.starting_equity, limits=limits,
        )

    def _init_position_manager(self):
        self.position_manager = PositionManager(
            breakeven_after_tp=self.config.breakeven_after_tp,
            trailing_after_tp=self.config.trailing_after_tp,
            trailing_atr_multiplier=self.config.trailing_atr_multiplier,
            max_position_age_bars=self.config.max_position_age_bars,
            commission_pct=self.config.commission_pct,
        )

    def _init_risk_manager(self):
        self.risk_manager = RiskManager(
            risk_per_trade_pct=self.profile.risk_per_trade_pct,
            atr_period=self.profile.atr_period,
            atr_multiplier=self.profile.atr_multiplier,
            default_rr_ratio=self.profile.default_rr_ratio,
            max_position_pct=self.profile.max_position_pct,
        )

    def _init_intake(self):
        self.intake = SignalIntake(timeout=5.0)

        if self.config.enable_trend:
            self.intake.register(TrendAnalyzer(name="trend"))
        if self.config.enable_elliott_wave:
            self.intake.register(ElliottWaveAnalyzer(name="elliott_wave"))
        if self.config.enable_volatility:
            self.intake.register(VolatilityAnalyzer(name="volatility"))
        if self.config.enable_volume:
            self.intake.register(VolumeAnalyzer(name="volume"))

        logger.info(
            "Анализаторы: %s",
            [a.name for a in self.intake.analyzers],
        )

    def _init_confluence(self):
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
        self.confluence.set_gate_mode(gate_mode)

    def _init_regime_detector(self):
        self.regime_detector = RegimeDetector()

    def _get_regime_1d(self, current_time) -> MarketRegime:
        mode = self.profile.regime_1d_filter_mode

        if mode == "none":
            return MarketRegime.CHOP

        if self._df_1d is None or current_time is None:
            return MarketRegime.CHOP

        try:
            available = self._df_1d[self._df_1d.index <= current_time]
        except Exception:
            return MarketRegime.CHOP

        if len(available) < 200:
            return MarketRegime.CHOP

        if mode == "baseline":
            window = available.iloc[-300:]
            return self._regime_1d_detector.detect(window)

        current_price = float(available["close"].iloc[-1])

        if mode == "sma200_only":
            sma = ta.sma(available["close"], length=200)
            if sma is None or len(sma) == 0:
                return MarketRegime.CHOP
            sma_val = float(sma.iloc[-1])
            if sma_val <= 0:
                return MarketRegime.CHOP
            distance = (current_price - sma_val) / sma_val
            if abs(distance) < 0.02:
                return MarketRegime.CHOP
            return MarketRegime.BULL if distance > 0 else MarketRegime.BEAR

        if mode == "drawdown_10":
            lookback = min(30, len(available))
            if lookback < 5:
                return MarketRegime.CHOP
            past_price = float(available["close"].iloc[-lookback])
            if past_price <= 0:
                return MarketRegime.CHOP
            drawdown = (current_price - past_price) / past_price
            if drawdown < -0.10:
                return MarketRegime.BEAR
            if drawdown > 0.05:
                return MarketRegime.BULL
            return MarketRegime.CHOP

        return MarketRegime.CHOP

    async def on_bar(self, data, bar_index, bar_time=None):
        self._bar_counter += 1
        result = BarResult(bar_index=bar_index, bar_time=bar_time)

        if self._bar_counter % self.config.snapshot_every_n_bars == 0:
            self.journal.log_portfolio_snapshot(self.portfolio.state)

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

        if self.profile.filter_chop:
            regime = self.regime_detector.detect(data)
            if regime == MarketRegime.CHOP:
                result.was_blocked_by_chop = True
                return result

        guard_result = self.portfolio.can_open(
            symbol=self.config.symbol,
            risk_pct=self.profile.risk_per_trade_pct,
        )
        if not guard_result.allowed:
            result.was_blocked_by_guard = True
            result.guard_reason = guard_result.reason
            return result

        batch = await self.intake.process(
            data, symbol=self.config.symbol, timeframe=self._timeframe,
        )
        decision = self.confluence.decide(batch.signals, data)
        result.decision = decision

        self.journal.log_decision(
            decision=decision, symbol=self.config.symbol,
            timeframe=self._timeframe, signals_count=batch.count,
        )

        if decision.passed and decision.direction != SignalDirection.HOLD:
            if self.profile.long_only and decision.direction == SignalDirection.SELL:
                result.was_blocked_by_long_only = True
                return result

            if self.profile.regime_1d_filter_mode != "none":
                regime_1d = self._get_regime_1d(bar_time)
                if (regime_1d == MarketRegime.BEAR
                        and decision.direction == SignalDirection.BUY):
                    self._blocked_by_1d += 1
                    result.was_blocked_by_regime_1d = True
                    return result
                if (regime_1d == MarketRegime.BULL
                        and decision.direction == SignalDirection.SELL):
                    self._blocked_by_1d += 1
                    result.was_blocked_by_regime_1d = True
                    return result

            opened = self._try_open_position(
                decision=decision, data=data,
                current_price=bar_close,
                bar_time=bar_time or datetime.now(timezone.utc),
            )
            if opened is not None:
                result.opened_position = opened

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
            logger.debug("Невалидный план: %s", errors)
            return None

        position = self.position_manager.open_position(
            symbol=self.config.symbol, timeframe=self._timeframe,
            direction=decision.direction, plan=plan,
            current_time=bar_time,
            metadata={
                "regime": decision.regime.value,
                "confluence_score": decision.confluence_score,
            },
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
                "timeframe": self._timeframe,
                "risk_per_trade_pct": self.profile.risk_per_trade_pct,
                "atr_multiplier": self.profile.atr_multiplier,
                "default_rr_ratio": self.profile.default_rr_ratio,
                "filter_chop": self.profile.filter_chop,
                "regime_1d_filter_mode": self.profile.regime_1d_filter_mode,
                "long_only": self.profile.long_only,
                "weights_override": self.profile.weights_override,
                "notes": self.profile.notes,
            },
            "blocked_by_1d": self._blocked_by_1d,
        }

    def close(self):
        self.journal.close()
        logger.info("TradingEngine закрыт")