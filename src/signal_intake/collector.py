"""Сборщик сигналов от анализаторов.

Поддерживает IndicatorCache + bar_index.

Вызов анализаторов производится через inspect.signature —
это позволяет явно определять, поддерживает ли конкретный
анализатор расширенный интерфейс (indicators, bar_index),
и НЕ ловить TypeError (который мог бы маскировать реальные
ошибки внутри анализатора).
"""

import asyncio
import inspect
import logging
from typing import List, Tuple, Optional, Any
import pandas as pd

from src.analyzers.base import BaseAnalyzer
from src.models import AnalyzerSignal, RejectedSignal

logger = logging.getLogger(__name__)


def _supports_extended_interface(analyzer: BaseAnalyzer) -> bool:
    """
    Проверяет, принимает ли analyzer.analyze параметры
    indicators и bar_index.

    Это нужно, чтобы:
    - не вызывать старый интерфейс с неожиданными keyword-аргументами,
    - не ловить TypeError широким except (маскирует реальные баги).
    """
    try:
        sig = inspect.signature(analyzer.analyze)
    except (TypeError, ValueError):
        # Не удалось получить сигнатуру — считаем, что старый интерфейс
        return False

    params = sig.parameters
    has_indicators = "indicators" in params
    has_bar_index = "bar_index" in params

    # Если у параметров есть VAR_KEYWORD (**kwargs) — считаем, что поддерживает
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    return (has_indicators and has_bar_index) or has_var_kw


class SignalCollector:
    """Собирает сигналы от всех зарегистрированных анализаторов."""

    def __init__(self, timeout: float = 5.0):
        self._analyzers: List[BaseAnalyzer] = []
        self._timeout = timeout
        # Кэш: имя анализатора → поддерживает ли расширенный интерфейс
        self._supports_cache: dict = {}

    def register(self, analyzer: BaseAnalyzer) -> None:
        if not isinstance(analyzer, BaseAnalyzer):
            raise TypeError(f"Ожидается BaseAnalyzer, получен {type(analyzer)}")
        self._analyzers.append(analyzer)
        self._supports_cache[analyzer.name] = _supports_extended_interface(analyzer)
        logger.info(
            "Анализатор зарегистрирован: %s (extended=%s)",
            analyzer.name,
            self._supports_cache[analyzer.name],
        )

    def unregister(self, analyzer: BaseAnalyzer) -> None:
        if analyzer in self._analyzers:
            self._analyzers.remove(analyzer)
            self._supports_cache.pop(analyzer.name, None)
            logger.info("Анализатор удалён: %s", analyzer.name)

    def clear(self) -> None:
        self._analyzers.clear()
        self._supports_cache.clear()
        logger.info("Все анализаторы удалены")

    @property
    def analyzers(self) -> List[BaseAnalyzer]:
        return list(self._analyzers)

    async def _run_one(
        self,
        analyzer: BaseAnalyzer,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
        bar_index: Optional[int] = None,
    ) -> Tuple[str, Optional[AnalyzerSignal], Optional[str]]:
        try:
            if self._supports_cache.get(analyzer.name, False):
                signal = await asyncio.wait_for(
                    analyzer.analyze(
                        data,
                        indicators=indicators,
                        bar_index=bar_index,
                    ),
                    timeout=self._timeout,
                )
            else:
                # Старый интерфейс: только data
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
            # Здесь ловим ВСЁ остальное (включая TypeError от самого
            # анализатора — это уже реальная ошибка, а не «старый интерфейс»).
            msg = f"Исключение: {type(e).__name__}: {e}"
            logger.exception("Анализатор %s упал: %s", analyzer.name, e)
            return analyzer.name, None, msg

    async def collect_all(
        self,
        data: pd.DataFrame,
        indicators: Optional[Any] = None,
        bar_index: Optional[int] = None,
    ) -> Tuple[List[AnalyzerSignal], List[RejectedSignal], dict]:
        if not self._analyzers:
            logger.warning("Нет зарегистрированных анализаторов")
            return [], [], {"total": 0, "responded": 0, "failed": 0}

        tasks = [
            self._run_one(a, data, indicators, bar_index)
            for a in self._analyzers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[AnalyzerSignal] = []
        rejected: List[RejectedSignal] = []
        failed = 0

        for result in results:
            if isinstance(result, Exception):
                # Это исключение, которое не поймал даже _run_one.
                # Такое не должно случаться, но подстрахуемся.
                logger.error("Необработанное исключение в gather: %s", result)
                failed += 1
                rejected.append(RejectedSignal(
                    signal=None,
                    reason=f"gather error: {type(result).__name__}: {result}",
                    source="unknown",
                ))
                continue

            name, signal, error = result
            if error is not None:
                failed += 1
                rejected.append(RejectedSignal(
                    signal=None, reason=error, source=name,
                ))
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