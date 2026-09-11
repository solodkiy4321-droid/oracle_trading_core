"""Trading Engine: главный класс интеграции."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import List, Optional

import pandas as pd

from src.engine.config import EngineConfig
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
    """Результат обработки одного бара."""
    bar_index: int = 0
    bar_time: Optional[datetime] = None
    bar_close: float = 0.0

    # Закрытые позиции на этом баре
    closed_positions: List[Position] = field(default_factory=list)
    realized_pnl: float = 0.0

    # Открытая позиция (если была)
    opened_position: Optional[Position] = None

    # Решение Confluence
    decision: Optional[ConfluenceDecision] = None

    # События
    events: List[PositionEvent] = field(default_factory=list)

    # Флаги
    was_blocked_by_guard: bool = False
    guard_reason: str = ""


class TradingEngine:
    """
    Trading Engine: оркестратор всех компонентов.

    На каждом баре выполняет полный цикл:
    1. Обработка открытых позиций (SL/TP/Timeout)
    2. Обновление Portfolio State
    3. Логирование
    4. Проверка Portfolio Guard
    5. Сбор сигналов
    6. Confluence-движок
    7. Открытие позиции
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Args:
            config: Конфигурация движка
        """
        self.config = config or EngineConfig()
        self.config.validate()

        # Инициализация компонентов
        self._init_journal()
        self._init_portfolio()
        self._init_position_manager()
        self._init_risk_manager()
        self._init_intake()
        self._init_confluence()

        # Счётчик баров
        self._bar_counter = 0

        logger.info(
            "TradingEngine инициализирован: %s %s, equity=%.2f, gate=%s",
            self.config.symbol, self.config.timeframe,
            self.config.starting_equity, self.config.gate_mode,
        )

    # ============ Инициализация ============

    def _init_journal(self) -> None:
        """Инициализация Journal."""
        self.journal = Journal(self.config.journal_db_path)
        self.analytics = JournalAnalytics(self.journal.db)

    def _init_portfolio(self) -> None:
        """Инициализация Portfolio Risk Manager."""
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
        """Инициализация Position Manager."""
        self.position_manager = PositionManager(
            breakeven_after_tp=self.config.breakeven_after_tp,
            trailing_after_tp=self.config.trailing_after_tp,
            trailing_atr_multiplier=self.config.trailing_atr_multiplier,
            max_position_age_bars=self.config.max_position_age_bars,
            commission_pct=self.config.commission_pct,
        )

    def _init_risk_manager(self) -> None:
        """Инициализация Risk Manager."""
        self.risk_manager = RiskManager(
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            atr_period=self.config.atr_period,
            atr_multiplier=self.config.atr_multiplier,
            default_rr_ratio=self.config.default_rr_ratio,
        )

    def _init_intake(self) -> None:
        """Инициализация Signal Intake с анализаторами."""
        self.intake = SignalIntake(timeout=5.0)

        if self.config.enable_indicators:
            self.intake.register(IndicatorsAnalyzer(name="indicators"))
        if self.config.enable_harmonic:
            self.intake.register(HarmonicAnalyzer(name="harmonic"))
        if self.config.enable_support_resistance:
            self.intake.register(SRAnalyzer(name="support_resistance"))

        logger.info(
            "Анализаторы: %s",
            [a.name for a in self.intake._collector.analyzers],
        )

    def _init_confluence(self) -> None:
        """Инициализация Confluence-движка."""
        gate_mode_map = {
            "aggressive": GateMode.AGGRESSIVE,
            "balanced": GateMode.BALANCED,
            "conservative": GateMode.CONSERVATIVE,
        }
        gate_mode = gate_mode_map.get(self.config.gate_mode, GateMode.BALANCED)
        self.confluence = ConfluenceEngine()
        self.confluence.set_gate_mode(gate_mode)

    # ============ Главный цикл ============

    async def on_bar(
        self,
        data: pd.DataFrame,
        bar_index: int,
        bar_time: Optional[datetime] = None,
    ) -> BarResult:
        """
        Обрабатывает один бар.

        Args:
            data: OHLCV DataFrame (только данные до текущего бара включительно)
            bar_index: Индекс текущего бара
            bar_time: Время бара

        Returns:
            BarResult с результатами обработки
        """
        self._bar_counter += 1
        result = BarResult(bar_index=bar_index, bar_time=bar_time)

        # Проверяем, что данных достаточно
        if len(data) < 50:
            logger.debug("Недостаточно данных: %d баров", len(data))
            return result

        # Текущий бар
        current_bar = data.iloc[-1]
        bar_high = float(current_bar["high"])
        bar_low = float(current_bar["low"])
        bar_close = float(current_bar["close"])
        result.bar_close = bar_close

        # 1. Обработка открытых позиций (SL/TP/Timeout)
        atr = self._calculate_atr(data)
        position_update = self.position_manager.on_bar(
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=bar_close,
            bar_time=bar_time or datetime.now(timezone.utc),
            atr=atr,
        )
        result.events = position_update.events

        # 2. Регистрация закрытых позиций
        for closed_pos in position_update.closed_positions:
            self._on_position_closed(closed_pos)
            result.closed_positions.append(closed_pos)
            result.realized_pnl += closed_pos.realized_pnl

        # 3. Проверка Portfolio Guard
        guard_result = self.portfolio.can_open(
            symbol=self.config.symbol,
            risk_pct=self.config.risk_per_trade_pct,
        )
        if not guard_result.allowed:
            result.was_blocked_by_guard = True
            result.guard_reason = guard_result.reason
            logger.debug("Portfolio Guard заблокировал: %s", guard_result.reason)
            return result

        # 4. Сбор сигналов
        batch = await self.intake.process(
            data, symbol=self.config.symbol, timeframe=self.config.timeframe,
        )

        # 5. Confluence-движок
        decision = self.confluence.decide(batch.signals, data)
        result.decision = decision

        # 6. Логирование решения
        self.journal.log_decision(
            decision=decision,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            signals_count=batch.count,
        )

        # 7. Открытие позиции (если решение прошло)
        if decision.passed and decision.direction != SignalDirection.HOLD:
            opened = self._try_open_position(
                decision=decision,
                data=data,
                current_price=bar_close,
                bar_time=bar_time or datetime.now(timezone.utc),
            )
            if opened is not None:
                result.opened_position = opened

        # 8. Периодический снимок портфеля
        if self._bar_counter % self.config.snapshot_every_n_bars == 0:
            self.journal.log_portfolio_snapshot(self.portfolio.state)

        return result

    def _try_open_position(
        self,
        decision: ConfluenceDecision,
        data: pd.DataFrame,
        current_price: float,
        bar_time: datetime,
    ) -> Optional[Position]:
        """Пытается открыть позицию на основе решения."""
        # Рассчитываем торговый план
        plan = self.risk_manager.calculate_plan(
            direction=decision.direction,
            entry_price=current_price,
            equity=self.portfolio.state.current_equity,
            data=data,
        )

        if plan is None:
            logger.warning("Не удалось рассчитать торговый план")
            return None

        # Валидация плана
        is_valid, errors = self.risk_manager.validate_plan(plan)
        if not is_valid:
            logger.warning("Невалидный план: %s", errors)
            return None

        # Открываем позицию
        position = self.position_manager.open_position(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            direction=decision.direction,
            plan=plan,
            current_time=bar_time,
            metadata={
                "regime": decision.regime.value,
                "confluence_score": decision.confluence_score,
            },
        )

        if position is None:
            return None

        # Регистрируем в Portfolio
        self.portfolio.register_position_opened(position)

        # Логируем в Journal
        self.journal.log_position_opened(
            position=position,
            regime=decision.regime.value,
            confluence_score=decision.confluence_score,
        )

        logger.info(
            "Открыта позиция %s: %s %s, entry=%.4f, SL=%.4f, size=%.6f",
            position.id, self.config.symbol, decision.direction.name,
            position.entry_price, position.stop_loss, position.initial_size,
        )

        return position

    def _on_position_closed(self, position: Position) -> None:
        """Обрабатывает закрытие позиции."""
        # Обновляем Portfolio
        self.portfolio.register_position_closed(position)

        # Логируем в Journal
        self.journal.log_position_closed(position)

        logger.info(
            "Позиция %s закрыта: reason=%s, pnl=%.2f, equity=%.2f",
            position.id,
            position.close_reason.value if position.close_reason else "unknown",
            position.realized_pnl,
            self.portfolio.state.current_equity,
        )

    def _calculate_atr(self, data: pd.DataFrame) -> Optional[float]:
        """Рассчитывает ATR для трейлинг-стопа."""
        import pandas_ta_classic as ta

        if len(data) < self.config.atr_period + 1:
            return None

        atr = ta.atr(
            data["high"], data["low"], data["close"],
            length=self.config.atr_period,
        )
        if atr is None or len(atr) == 0:
            return None
        return float(atr.iloc[-1])

    # ============ Управление ============

    def start_new_day(self) -> None:
        """Начинает новый торговый день."""
        self.portfolio.start_new_day()
        self.journal.log_portfolio_snapshot(self.portfolio.state)

    def close_all_positions(self, price: float) -> List[Position]:
        """Закрывает все позиции принудительно."""
        closed = self.position_manager.close_all(
            price, datetime.now(timezone.utc), CloseReason.MANUAL,
        )
        for pos in closed:
            self._on_position_closed(pos)
        return closed

    def get_stats(self) -> dict:
        """Возвращает текущую статистику."""
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
        }

    def close(self) -> None:
        """Закрывает ресурсы."""
        self.journal.close()
        logger.info("TradingEngine закрыт")