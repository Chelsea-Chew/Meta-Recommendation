"""Operational kill switch and circuit breaker controls for LLM Gateway."""

from __future__ import annotations

import threading
from typing import Optional

from llm_gateway.constants import DEFAULT_KILL_SWITCH_REASON


class KillSwitchManager:
    """Thread-safe kill switch state controller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: bool = False
        self._reason: Optional[str] = None

    def enable(self, reason: str = DEFAULT_KILL_SWITCH_REASON) -> None:
        with self._lock:
            self._active = True
            self._reason = reason

    def disable(self) -> None:
        with self._lock:
            self._active = False
            self._reason = None

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_reason(self) -> Optional[str]:
        with self._lock:
            return self._reason


_GLOBAL_KILL_SWITCH = KillSwitchManager()


def set_kill_switch(active: bool, reason: str = DEFAULT_KILL_SWITCH_REASON) -> None:
    if active:
        _GLOBAL_KILL_SWITCH.enable(reason)
    else:
        _GLOBAL_KILL_SWITCH.disable()



def is_kill_switch_active() -> bool:
    return _GLOBAL_KILL_SWITCH.is_active()


def get_kill_switch_reason() -> Optional[str]:
    return _GLOBAL_KILL_SWITCH.get_reason()
