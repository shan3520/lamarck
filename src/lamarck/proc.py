"""Run a subprocess with a timeout that really stops it.

`subprocess.run(..., timeout=)` kills only the direct child. On Windows, grandchildren
(e.g. a `find` started by bash) survive, keep the output pipes open, and the call then
blocks forever. A real run hung this way on 2026-10-03. Here, a timeout kills the whole
process tree.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass


@dataclass
class Result:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run(args, *, shell: bool = False, cwd=None, env=None, timeout: float | None = None) -> Result:
    kwargs: dict = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True  # own process group, so the whole tree can be killed
    proc = subprocess.Popen(args, shell=shell, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **kwargs)
    try:
        out, err = proc.communicate(timeout=timeout)
        return Result(proc.returncode, out, err, False)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return Result(None, out or "", err or "", True)
