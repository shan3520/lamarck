"""A minimal, token-frugal coding agent (bash only), in the spirit of mini-swe-agent.

    python -m lamarck.agents.mini --model qwen/qwen3.8-27b --prompt-file task.md

It reads its *harness* from AGENT.md in the working directory. That file is what
Lamarck evolves. The tool protocol below is fixed and is not part of the harness.

Each turn the model answers with exactly one ```bash block (run in the working
directory) or the word SUBMIT. Built for free tiers: few steps, short observations,
only the most recent exchanges kept in full, and a hard token cap per attempt.

Exit codes: 0 normal; 75 the provider's daily quota is exhausted (the caller stops and saves).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..chat_compat import GatewayHTTPError
from ..evaluate import is_fatal
from ..llm import EmptyResponseError, complete, make_client

QUOTA_EXIT_CODE = 75

PROTOCOL = """\
## How to act
You are working in a code repository. You can only act by running shell commands.
Answer each turn with a brief thought followed by exactly ONE fenced block:
```bash
<one shell command>
```
The command runs in the repository root and you will see its (truncated) output.
To edit files, use commands such as python - <<'EOF' ... EOF, sed, or cat > file <<'EOF'.
When the task is complete, answer with the single word SUBMIT and no code block."""

_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", flags=re.DOTALL)


def find_bash() -> str | None:
    """Git Bash on Windows (never WSL's System32 bash), plain bash elsewhere."""
    if os.environ.get("LAMARCK_BASH"):
        return os.environ["LAMARCK_BASH"]
    if os.name == "nt":
        try:
            exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()
            for parent in Path(exec_path).parents:
                cand = parent / "bin" / "bash.exe"
                if cand.is_file():
                    return str(cand)
        except OSError:
            pass
        return None
    return shutil.which("bash")


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars omitted] ...\n" + text[-half:]


def run_command(cmd: str, cwd: Path, timeout: int, bash: str | None) -> tuple[int, str]:
    argv = [bash, "-c", cmd] if bash else cmd
    try:
        p = subprocess.run(argv, shell=bash is None, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, f"(command timed out after {timeout}s)"


def build_user(task: str, history: list[tuple[str, str]], keep: int, step: int, max_steps: int) -> str:
    parts = [f"# Task\n{task.strip()}"]
    older, recent = history[:-keep] if keep else history, history[-keep:] if keep else []
    if older:
        parts.append("# Earlier steps (summarized)\n" + "\n".join(f"- $ {c.splitlines()[0][:120]}  ->  {' | '.join(o.splitlines()[:2])[:120]}" for c, o in older))
    for c, o in recent:
        parts.append(f"# You ran\n```bash\n{c}\n```\n# Output\n{o}")
    parts.append(f"Step {step}/{max_steps}. Reply with one ```bash block, or SUBMIT if the task is done.")
    return "\n\n".join(parts)


def main(argv: list[str] | None = None, client=None) -> int:
    ap = argparse.ArgumentParser(prog="lamarck-mini-agent")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--harness", default="AGENT.md")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-tokens-total", type=int, default=30000)
    ap.add_argument("--obs-chars", type=int, default=1500)
    ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--cmd-timeout", type=int, default=120)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args(argv)

    cwd = Path.cwd()
    task = Path(a.prompt_file).read_text(encoding="utf-8")
    hp = cwd / a.harness
    harness = hp.read_text(encoding="utf-8") if hp.is_file() else ""
    system = (harness.strip() + "\n\n" + PROTOCOL) if harness.strip() else PROTOCOL
    bash = find_bash()
    client = client or make_client(a.provider)
    history: list[tuple[str, str]] = []
    tokens = 0
    end = "step limit"
    for step in range(1, a.max_steps + 1):
        if tokens >= a.max_tokens_total:
            end = "token cap"
            break
        try:
            c = complete(client, model=a.model, system=system, user=build_user(task, history, a.keep, step, a.max_steps), max_tokens=2500, effort=a.effort)
        except EmptyResponseError:
            history.append(("(no command)", "The model returned an empty reply. Reply with one ```bash block or SUBMIT."))
            continue
        except GatewayHTTPError as exc:
            if is_fatal(exc):
                print(f"FATAL: {exc}", file=sys.stderr)
                print(f"LAMARCK_TOKENS={tokens}")
                return QUOTA_EXIT_CODE
            raise
        tokens += c.tokens
        m = _BLOCK.search(c.text)
        if not m:
            if "SUBMIT" in c.text:
                end = "submitted"
                break
            history.append(("(no command)", "No ```bash block found. Reply with exactly one ```bash block, or SUBMIT."))
            continue
        cmd = m.group(1).strip()
        code, out = run_command(cmd, cwd, a.cmd_timeout, bash)
        obs = f"exit code {code}\n{truncate(out.strip(), a.obs_chars)}"
        history.append((cmd, obs))
        print(f"--- step {step}\n$ {cmd}\n{obs}\n", flush=True)
    print(f"END: {end} after {len(history)} commands")
    print(f"LAMARCK_TOKENS={tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
