"""Keep the computer from idle-sleeping while a long run is in progress.

Real runs were stalled for hours by the PC going to sleep: agents froze mid-attempt, the
network dropped on every wake (seen as DNS "outages"), and sleep time didn't count toward
attempt timeouts. This asks the OS not to idle-sleep *while this process runs*, the way a
video player does. It changes no settings, and ends when the process exits. A closed lid or
an explicit Sleep still sleeps.
"""

from __future__ import annotations

import os

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> bool:
    """Returns True if the request was made (Windows only; elsewhere use e.g. `caffeinate`/`systemd-inhibit`)."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False
