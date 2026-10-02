"""Run a coding agent CLI (Claude Code, Codex, lamarck's mini agent, ...) on a git-mined task.

For each rollout:
  1. materialize a *snapshot* of the repo at the task's base commit (no `.git` inside),
  2. write in the task's test files from the fixing commit,
  3. write the rendered harness to the agent's instruction file (CLAUDE.md / AGENTS.md / AGENT.md),
  4. run the agent command with the task prompt,
  5. run the task's verifier (the tests) in the snapshot,
  6. delete the snapshot.

The snapshot has no git history on purpose: in a worktree the agent could run
`git show <fix commit>` and copy the answer.

The agent command is a template. Placeholders: {prompt_file}, {workdir}.
Example: claude -p "$(cat {prompt_file})" --permission-mode acceptEdits
Agents can run arbitrary commands: use a sandboxed machine or container for untrusted repositories.

Agent protocol (optional): print `LAMARCK_TOKENS=<n>` to report token use; exit with
code 75 when the model provider's daily quota is exhausted (the run then stops and saves).
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..evaluate import FatalRunError
from ..harness import Harness
from ..tasks import Task, verify
from .base import Trial

QUOTA_EXIT_CODE = 75
_TOKENS = re.compile(r"^LAMARCK_TOKENS=(\d+)\s*$", flags=re.MULTILINE)


def _git_bytes(repo: str | Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def materialize(repo: str | Path, ref: str, dest: str | Path, overlay_commit: str | None = None, overlay_files: list[str] | tuple = ()) -> Path:
    """Extract the tree at `ref` into `dest` without any git metadata, then overwrite
    `overlay_files` with their content at `overlay_commit`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    data = _git_bytes(repo, "archive", "--format=tar", ref)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # Python without extraction filters
            tar.extractall(dest)
    for rel in overlay_files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git_bytes(repo, "show", f"{overlay_commit}:{rel}"))
    return dest


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
        try:
            wd = materialize(meta["repo"], meta["base"], tmp / "wd", meta["commit"], meta.get("test_files", []))
            (wd / self.harness_file).write_text(harness.render(), encoding="utf-8")
            prompt_file = tmp / "prompt.md"
            prompt_file.write_text(task.prompt, encoding="utf-8")
            cmd = self.agent_cmd.format(prompt_file=prompt_file.as_posix(), workdir=wd.as_posix())
            code = None
            try:
                proc = subprocess.run(cmd, shell=True, cwd=wd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
                out, err, code = proc.stdout, proc.stderr, proc.returncode
            except subprocess.TimeoutExpired as exc:
                out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                err = "agent timed out"
            if code == QUOTA_EXIT_CODE:
                raise FatalRunError("daily model quota exhausted inside the agent (exit 75)")
            found = _TOKENS.findall(out)
            tokens = int(found[-1]) if found else 0
            v = verify(task, out, workdir=wd)
            trace = f"AGENT OUTPUT (tail):\n{out[-3000:]}\n\nAGENT STDERR (tail):\n{err[-1000:]}\n\nVERIFIER: {v.detail or 'PASS'}"
            return Trial(task.id, v.passed, output=out[-4000:], detail=v.detail, trace=trace, tokens=tokens, score=v.score)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
