"""Настройка логирования для Oracle Trading Core."""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> None:
    """
    Настраивает логирование в консоль и файл.
    
    Args:
        log_dir: Папка для логов
        level: Уровень логирования
        console: Логировать ли в консоль
        file: Логировать ли в файл
    """
    # Создаём папку для логов
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Формат логов
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Удаляем старые хендлеры (чтобы не дублировать при повторном вызове)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Консольный хендлер
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Файловый хендлер (с ротацией)
    if file:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            filename=log_path / "oracle_trading.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # хранить 5 архивных файлов
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Отдельный файл для ошибок
    error_handler = RotatingFileHandler(
        filename=log_path / "errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    root_logger.addHandler(error_handler)


def get_logger(name: str) -> logging.Logger:
    """Получить логгер по имени модуля."""
    return logging.getLogger(name)