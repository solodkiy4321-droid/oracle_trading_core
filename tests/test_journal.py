"""Тесты Journal."""

from datetime import datetime, timezone, date
from uuid import uuid4

import pytest

from src.models import SignalDirection
from src.position.position import Position, CloseReason
from src.position.events import PositionEvent, PositionEventType
from src.portfolio.state import PortfolioState
from src.confluence.engine import ConfluenceDecision
from src.confluence.regime_detector import MarketRegime
from src.journal.journal import Journal
from src.journal.analytics import JournalAnalytics


def make_position(
    symbol: str = "BTC-USD",
    direction: SignalDirection = SignalDirection.BUY,
    entry: float = 100.0,
    size: float = 10.0,
    pnl: float = 0.0,
    closed: bool = False,
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
    if closed:
        pos.realized_pnl = pnl
        pos.close_time = datetime.now(timezone.utc)
        pos.close_price = entry + (pnl / size if size else 0)
        pos.close_reason = CloseReason.TAKE_PROFIT if pnl > 0 else CloseReason.STOP_LOSS
        pos.status = pos.status.CLOSED
    return pos


# ============ Journal: запись сделок ============

def test_journal_init():
    """Journal инициализируется."""
    journal = Journal(":memory:")
    assert journal.get_trades_count() == 0
    assert journal.get_events_count() == 0
    assert journal.get_decisions_count() == 0
    journal.close()


def test_journal_log_position_opened():
    """Открытие позиции логируется."""
    journal = Journal(":memory:")
    pos = make_position(symbol="BTC-USD")

    trade_id = journal.log_position_opened(pos, regime="BULL", confluence_score=0.7)
    assert trade_id > 0
    assert journal.get_trades_count() == 1

    # Проверяем содержимое
    trades = journal.db.execute("SELECT * FROM trades").fetchall()
    assert len(trades) == 1
    assert trades[0]["symbol"] == "BTC-USD"
    assert trades[0]["direction"] == "BUY"
    assert trades[0]["regime"] == "BULL"
    assert trades[0]["confluence_score"] == 0.7

    journal.close()


def test_journal_log_position_closed():
    """Закрытие позиции обновляет запись."""
    journal = Journal(":memory:")
    pos = make_position(symbol="BTC-USD", pnl=200.0, closed=True)

    journal.log_position_opened(pos)
    journal.log_position_closed(pos)

    trades = journal.db.execute("SELECT * FROM trades").fetchall()
    assert len(trades) == 1
    assert trades[0]["realized_pnl"] == 200.0
    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_price"] is not None

    journal.close()


def test_journal_log_event():
    """Событие логируется."""
    journal = Journal(":memory:")
    event = PositionEvent(
        type=PositionEventType.OPENED,
        position_id="pos1",
        symbol="BTC-USD",
        price=100.0,
        size=10.0,
    )

    event_id = journal.log_event(event)
    assert event_id > 0
    assert journal.get_events_count() == 1

    journal.close()


def test_journal_log_events_bulk():
    """Пакетная запись событий."""
    journal = Journal(":memory:")
    events = [
        PositionEvent(
            type=PositionEventType.OPENED,
            position_id="pos1",
            symbol="BTC-USD",
            price=100.0,
        ),
        PositionEvent(
            type=PositionEventType.PARTIAL_CLOSE,
            position_id="pos1",
            symbol="BTC-USD",
            price=102.0,
            pnl=10.0,
        ),
    ]

    count = journal.log_events(events)
    assert count == 2
    assert journal.get_events_count() == 2

    journal.close()


def test_journal_log_portfolio_snapshot():
    """Снимок портфеля логируется."""
    journal = Journal(":memory:")
    state = PortfolioState(
        starting_equity=10000.0,
        current_equity=10200.0,
        peak_equity=10200.0,
    )
    state.start_new_day(date.today())

    snapshot_id = journal.log_portfolio_snapshot(state)
    assert snapshot_id > 0

    snapshots = journal.db.execute("SELECT * FROM portfolio_snapshots").fetchall()
    assert len(snapshots) == 1
    assert snapshots[0]["current_equity"] == 10200.0

    journal.close()


def test_journal_log_decision():
    """Решение Confluence логируется."""
    journal = Journal(":memory:")
    decision = ConfluenceDecision(
        direction=SignalDirection.BUY,
        confluence_score=0.7,
        regime=MarketRegime.BULL,
        passed=True,
        weights_used={"indicators": 0.3, "harmonic": 0.3},
        reasons=["indicators: BUY", "harmonic: BUY"],
        bull_score=0.7,
        bear_score=0.0,
    )

    decision_id = journal.log_decision(
        decision, symbol="BTC-USD", timeframe="1h", signals_count=2,
    )
    assert decision_id > 0
    assert journal.get_decisions_count() == 1

    journal.close()


# ============ JournalAnalytics ============

def test_analytics_empty():
    """Аналитика на пустой БД."""
    journal = Journal(":memory:")
    analytics = JournalAnalytics(journal.db)

    stats = analytics.get_trade_stats()
    assert stats.total_trades == 0
    assert stats.win_rate == 0.0
    assert stats.profit_factor == 0.0

    journal.close()


def test_analytics_basic_stats():
    """Базовая статистика по сделкам."""
    journal = Journal(":memory:")

    # Победная сделка
    win_pos = make_position(symbol="BTC-USD", pnl=200.0, closed=True)
    journal.log_position_opened(win_pos)
    journal.log_position_closed(win_pos)

    # Убыточная сделка
    loss_pos = make_position(symbol="ETH-USD", pnl=-100.0, closed=True)
    journal.log_position_opened(loss_pos)
    journal.log_position_closed(loss_pos)

    analytics = JournalAnalytics(journal.db)
    stats = analytics.get_trade_stats()

    assert stats.total_trades == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.win_rate == 0.5
    assert stats.total_pnl == 100.0
    assert stats.profit_factor == 2.0  # 200 / 100

    journal.close()


def test_analytics_by_symbol():
    """Разбивка по символам."""
    journal = Journal(":memory:")

    for symbol, pnl in [("BTC-USD", 200.0), ("BTC-USD", -50.0), ("ETH-USD", 100.0)]:
        pos = make_position(symbol=symbol, pnl=pnl, closed=True)
        journal.log_position_opened(pos)
        journal.log_position_closed(pos)

    analytics = JournalAnalytics(journal.db)
    stats = analytics.get_trade_stats()

    assert "BTC-USD" in stats.by_symbol
    assert "ETH-USD" in stats.by_symbol
    assert stats.by_symbol["BTC-USD"]["trades"] == 2
    assert stats.by_symbol["BTC-USD"]["pnl"] == 150.0
    assert stats.by_symbol["ETH-USD"]["pnl"] == 100.0

    journal.close()


def test_analytics_decisions_summary():
    """Сводка по решениям Confluence."""
    journal = Journal(":memory:")

    # Принятое решение
    decision1 = ConfluenceDecision(
        direction=SignalDirection.BUY,
        confluence_score=0.7,
        regime=MarketRegime.BULL,
        passed=True,
        weights_used={},
        reasons=[],
    )
    journal.log_decision(decision1, "BTC-USD", "1h", signals_count=2)

    # Отклонённое решение
    decision2 = ConfluenceDecision(
        direction=SignalDirection.HOLD,
        confluence_score=0.3,
        regime=MarketRegime.CHOP,
        passed=False,
        weights_used={},
        reasons=[],
    )
    journal.log_decision(decision2, "ETH-USD", "1h", signals_count=1)

    analytics = JournalAnalytics(journal.db)
    summary = analytics.get_decisions_summary()

    assert summary["total_decisions"] == 2
    assert summary["passed_gate"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["buys"] == 1
    assert summary["holds"] == 1

    journal.close()


def test_analytics_get_all_trades():
    """Получение всех сделок."""
    journal = Journal(":memory:")

    for symbol in ["BTC-USD", "ETH-USD", "AAPL"]:
        pos = make_position(symbol=symbol, pnl=50.0, closed=True)
        journal.log_position_opened(pos)
        journal.log_position_closed(pos)

    analytics = JournalAnalytics(journal.db)
    trades = analytics.get_all_trades()
    assert len(trades) == 3

    journal.close()