"""Подключение к SQLite и создание схемы."""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# SQL-схема
SCHEMA_SQL = """
-- Таблица сделок
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    initial_size REAL NOT NULL,
    initial_stop_loss REAL NOT NULL,
    final_stop_loss REAL NOT NULL,
    take_profits TEXT NOT NULL,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,
    realized_pnl REAL DEFAULT 0.0,
    realized_pnl_pct REAL DEFAULT 0.0,
    risk_amount REAL DEFAULT 0.0,
    risk_pct REAL DEFAULT 0.0,
    regime TEXT DEFAULT '',
    confluence_score REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(realized_pnl);

-- Таблица событий
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL DEFAULT 0.0,
    size REAL DEFAULT 0.0,
    pnl REAL DEFAULT 0.0,
    reason TEXT DEFAULT '',
    details TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_position ON events(position_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

-- Таблица снимков портфеля
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    date TEXT NOT NULL,
    starting_equity REAL NOT NULL,
    current_equity REAL NOT NULL,
    peak_equity REAL NOT NULL,
    total_pnl REAL DEFAULT 0.0,
    total_pnl_pct REAL DEFAULT 0.0,
    drawdown_pct REAL DEFAULT 0.0,
    open_positions INTEGER DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_losses INTEGER DEFAULT 0,
    daily_pnl REAL DEFAULT 0.0,
    daily_pnl_pct REAL DEFAULT 0.0,
    is_paused INTEGER DEFAULT 0,
    pause_reason TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(date);

-- Таблица решений Confluence
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    regime TEXT DEFAULT '',
    direction TEXT DEFAULT '',
    confluence_score REAL DEFAULT 0.0,
    bull_score REAL DEFAULT 0.0,
    bear_score REAL DEFAULT 0.0,
    passed_gate INTEGER DEFAULT 0,
    signals_count INTEGER DEFAULT 0,
    weights TEXT DEFAULT '',
    reasons TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
"""


class Database:
    """
    Обёртка над SQLite для Journal.

    Отвечает за:
    - Подключение к базе
    - Создание схемы
    - Транзакции
    - Закрытие соединения
    """

    def __init__(self, db_path: str = "journal.db"):
        """
        Args:
            db_path: Путь к файлу SQLite. ':memory:' для in-memory БД (тесты).
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self) -> None:
        """Создаёт соединение и схему."""
        # Для :memory: держим одно соединение
        if self.db_path == ":memory:":
            self._conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
            logger.info("Journal: in-memory БД инициализирована")
        else:
            # Для файла создаём директорию
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
            logger.info("Journal: БД инициализирована (%s)", self.db_path)

    @property
    def conn(self) -> sqlite3.Connection:
        """Возвращает активное соединение."""
        if self._conn is None:
            raise RuntimeError("Database не инициализирована")
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Выполняет SQL-запрос."""
        return self.conn.execute(sql, params)

    def commit(self) -> None:
        """Коммитит транзакцию."""
        self.conn.commit()

    def close(self) -> None:
        """Закрывает соединение."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Journal: БД закрыта")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()