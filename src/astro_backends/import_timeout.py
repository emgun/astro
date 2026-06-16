from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType, ModuleType
from typing import Any

DEFAULT_OPTIONAL_IMPORT_TIMEOUT_S = 15.0


@contextmanager
def optional_import_timeout(module_name: str, timeout_s: float) -> Iterator[None]:
    if (
        timeout_s <= 0.0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return

    def handle_timeout(_signum: int, _frame: FrameType | None) -> None:
        raise TimeoutError(
            f"import timed out for optional backend module {module_name!r} "
            f"after {timeout_s:g} seconds"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_s)
    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def import_optional_module(
    module_name: str,
    import_module: Callable[[str], ModuleType | Any],
    *,
    timeout_s: float = DEFAULT_OPTIONAL_IMPORT_TIMEOUT_S,
) -> ModuleType | Any:
    with optional_import_timeout(module_name, timeout_s):
        return import_module(module_name)
