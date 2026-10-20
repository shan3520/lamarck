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

The agent command is a template. Placeholders: {prompt_file}, {workdir}, {python} (the running interpreter, quoted).
Example (works in cmd and sh): claude -p --output-format json --permission-mode acceptEdits --allowedTools Bash < {prompt_file}
Agents can run arbitrary commands: use a sandboxed machine or container for untrusted repositories.

Agent protocol (optional): print `LAMARCK_TOKENS=<n>` to report token use; exit with
code 75 when the model provider's daily quota is exhausted (the run then stops and saves).
Claude Code's `--output-format json` result is understood directly: its token usage is read, and
an API error (bad login, rate limit, outage) pauses the run instead of scoring the task as failed.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .. import proc as _proc
from ..evaluate import FatalRunError
from ..harness import Harness
from ..tasks import Task, expand_python, plain_env, verify
from .base import Trial

QUOTA_EXIT_CODE = 75
AGENT_CRASH_EXIT_CODE = 70

# Variables a *parent* Claude Code session sets for itself. An agent launched from inside one (e.g. Lamarck run by
# an agent, or from the Claude desktop app) must not inherit them: they are that session's private connection,
# and the nested `claude` failed with 401 until they were removed. The user's own ANTHROPIC_API_KEY is kept.
_PARENT_SESSION = re.compile(r"^(CLAUDECODE|CLAUDE_CODE_\w+|CLAUDE_AGENT_SDK_\w+|CLAUDE_PID|CLAUDE_EFFORT|CLAUDE_PREVIEW_\w+|"
                             r"ANTHROPIC_AUTH_TOKEN|ANTHROPIC_BASE_URL|ANTHROPIC_MODEL|ANTHROPIC_SMALL_FAST_MODEL)$")


def agent_env() -> dict:
    env = plain_env()
    if "CLAUDECODE" in os.environ or "CLAUDE_CODE_SESSION_ID" in os.environ:  # we are inside a Claude Code session
        env = {k: v for k, v in env.items() if not _PARENT_SESSION.match(k)}
    return env


def claude_code_result(out: str) -> dict | None:
    """The JSON result object printed by `claude -p --output-format json`, if that's what the agent printed."""
    text = out.strip()
    for candidate in (text, text[text.find('{'):] if '{' in text else ''):
        try:
            d = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict) and d.get("type") == "result" and "usage" in d:
            return d
    return None


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
            prompt_file.write_text(expand_python(task.prompt), encoding="utf-8")
            cmd = expand_python(self.agent_cmd).format(prompt_file=prompt_file.as_posix(), workdir=wd.as_posix())
            res = _proc.run(cmd, shell=True, cwd=wd, timeout=self.timeout, env=agent_env())
            out, err, code = res.stdout, res.stderr + ("\nagent timed out" if res.timed_out else ""), res.returncode
            if code == QUOTA_EXIT_CODE:
                raise FatalRunError("model provider unavailable inside the agent (exit 75): " + " ".join(err.split())[-400:])
            if code == AGENT_CRASH_EXIT_CODE:  # a bug in the agent program, not the model failing the task
                return Trial(task.id, False, detail="FAIL: infrastructure error (agent crashed)", trace=err[-2000:],
                             error="agent crashed: " + " ".join(err.split())[-300:])
            if res.timed_out and "--- step" not in out:
                # killed before the model answered even once: a provider hang (seen on NVIDIA's free tier,
                # where 5 of 7 GLM attempts were scored as task failures), not the model failing the task
                return Trial(task.id, False, detail="FAIL: infrastructure error (no model reply before the timeout)", trace=err[-2000:],
                             error="agent timed out before the model replied once")
            found = _TOKENS.findall(out)
            tokens = int(found[-1]) if found else 0
            models: list[str] = []
            cc = claude_code_result(out)
            if cc is not None:
                # e.g. 401 bad login, "Not logged in" (no status, terminal_reason api_error), 429, 529 overloaded:
                # the model was never reached, so this is never the agent failing the task
                if cc.get("is_error") and (cc.get("api_error_status") or cc.get("terminal_reason") == "api_error"):
                    raise FatalRunError(f"Claude Code API error {cc.get('api_error_status') or ''}: {str(cc.get('result'))[:200]}")
                u = cc.get("usage") or {}
                tokens = sum(int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
                models = sorted((cc.get("modelUsage") or {}).keys())  # what really answered (a gateway may swap models)
            v = verify(task, out, workdir=wd)
            trace = f"AGENT OUTPUT (tail):\n{out[-3000:]}\n\nAGENT STDERR (tail):\n{err[-1000:]}\n\nVERIFIER: {v.detail or 'PASS'}"
            return Trial(task.id, v.passed, output=out[-4000:], detail=v.detail, trace=trace, tokens=tokens, score=v.score, models=models)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
