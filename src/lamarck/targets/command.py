"""Run a real coding agent CLI (Claude Code, Codex, ...) on a git-mined task.

For each rollout:
  1. create a throwaway git worktree at the task's base commit,
  2. check out the task's test files from the fixing commit,
  3. write the rendered harness to the agent's instruction file (CLAUDE.md / AGENTS.md),
  4. run the agent command with the task prompt,
  5. run the task's verifier (the tests) in the worktree,
  6. delete the worktree.

The agent command is a template. Placeholders: {prompt_file}, {workdir}.
Example: claude -p "$(cat {prompt_file})" --permission-mode acceptEdits
Agents act inside the throwaway worktree, but they can still run arbitrary commands:
use a sandboxed machine or container for untrusted repositories.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from ..harness import Harness
from ..tasks import Task, verify
from .base import Trial

# `git worktree add/remove` mutate shared repo metadata; serialize them across worker threads.
_WORKTREE_LOCK = threading.Lock()


def _git(repo: str | Path, *args: str) -> None:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")


@dataclass
class CommandTarget:
    agent_cmd: str
    harness_file: str = "CLAUDE.md"
    timeout: int = 1800
    name: str = "command"

    def run(self, harness: Harness, task: Task, seed: int) -> Trial:
        meta = task.meta
        for key in ("repo", "base", "commit"):
            if key not in meta:
                raise ValueError(f"task {task.id} lacks meta[{key!r}]; CommandTarget needs git-mined tasks")
        tmp = Path(tempfile.mkdtemp(prefix="lamarck-"))
        wt = tmp / "wt"
        try:
            with _WORKTREE_LOCK:
                _git(meta["repo"], "worktree", "add", "--detach", str(wt), meta["base"])
            if meta.get("test_files"):
                _git(wt, "checkout", meta["commit"], "--", *meta["test_files"])
            (wt / self.harness_file).write_text(harness.render(), encoding="utf-8")
            prompt_file = tmp / "prompt.md"
            prompt_file.write_text(task.prompt, encoding="utf-8")
            cmd = self.agent_cmd.format(prompt_file=prompt_file.as_posix(), workdir=wt.as_posix())
            try:
                proc = subprocess.run(cmd, shell=True, cwd=wt, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
                out, err = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                err = "agent timed out"
            v = verify(task, out, workdir=wt)
            trace = f"AGENT OUTPUT (tail):\n{out[-3000:]}\n\nAGENT STDERR (tail):\n{err[-1000:]}\n\nVERIFIER: {v.detail or 'PASS'}"
            return Trial(task.id, v.passed, output=out[-4000:], detail=v.detail, trace=trace, tokens=0)
        finally:
            try:
                if wt.exists():
                    with _WORKTREE_LOCK:
                        _git(meta["repo"], "worktree", "remove", "--force", str(wt))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
