"""Сборщик сигналов от анализаторов."""

import asyncio
import logging
from typing import List, Tuple, Optional
import pandas as pd

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, RejectedSignal

logger = logging.getLogger(__name__)


class SignalCollector:
    """
    Собирает сигналы от всех зарегистрированных анализаторов.

    Ключевые особенности:
    - Параллельный запуск через asyncio.gather
    - Изоляция сбоев (return_exceptions=True)
    - Таймауты на каждый анализатор
    """

    def __init__(self, timeout: float = 5.0):
        self._analyzers: List[BaseAnalyzer] = []
        self._timeout = timeout

    def register(self, analyzer: BaseAnalyzer) -> None:
        if not isinstance(analyzer, BaseAnalyzer):
            raise TypeError(f"Ожидается BaseAnalyzer, получен {type(analyzer)}")
        self._analyzers.append(analyzer)
        logger.info("Анализатор зарегистрирован: %s", analyzer.name)

    def unregister(self, analyzer: BaseAnalyzer) -> None:
        if analyzer in self._analyzers:
            self._analyzers.remove(analyzer)
            logger.info("Анализатор удалён: %s", analyzer.name)

    def clear(self) -> None:
        self._analyzers.clear()
        logger.info("Все анализаторы удалены")

    @property
    def analyzers(self) -> List[BaseAnalyzer]:
        return list(self._analyzers)

    async def _run_one(
        self, analyzer: BaseAnalyzer, data: pd.DataFrame
    ) -> Tuple[str, Optional[AnalyzerSignal], Optional[str]]:
        try:
            signal = await asyncio.wait_for(
                analyzer.analyze(data),
                timeout=self._timeout,
            )
            return analyzer.name, signal, None
        except asyncio.TimeoutError:
            msg = f"Таймаут {self._timeout}s"
            logger.warning("Анализатор %s: %s", analyzer.name, msg)
            return analyzer.name, None, msg
        except Exception as e:
            msg = f"Исключение: {type(e).__name__}: {e}"
            logger.exception("Анализатор %s упал: %s", analyzer.name, e)
            return analyzer.name, None, msg

    async def collect_all(
        self, data: pd.DataFrame
    ) -> Tuple[List[AnalyzerSignal], List[RejectedSignal], dict]:
        if not self._analyzers:
            logger.warning("Нет зарегистрированных анализаторов")
            return [], [], {"total": 0, "responded": 0, "failed": 0}

        tasks = [self._run_one(a, data) for a in self._analyzers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[AnalyzerSignal] = []
        rejected: List[RejectedSignal] = []
        failed = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error("Необработанное исключение в gather: %s", result)
                failed += 1
                continue

            name, signal, error = result

            if error is not None:
                failed += 1
                rejected.append(RejectedSignal(signal=None, reason=error, source=name))
                continue

            if signal is None:
                continue

            signals.append(signal)

        stats = {
            "total": len(self._analyzers),
            "responded": len(signals),
            "failed": failed,
        }
        logger.info("Сбор завершён: %s", stats)
        return signals, rejected, stats