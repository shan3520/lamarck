"""Mine verified tasks from a repository's own git history.

The blocker DarwinX names for deployment is that "fitness needs a verifier and
production tasks rarely arrive with one". A repo's history is a free supply:
every commit that changed source code *and* tests is a task —

    start from the parent commit, bring in the commit's tests,
    ask the agent to make them pass (the commit message is the issue text),
    verify by running exactly those tests.

These tasks are specific to the team's codebase, conventions and tooling, which
is the distribution their agent actually works on.
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .tasks import Task, TaskSuite

DEFAULT_TEST_GLOBS = ("test_*.py", "*_test.py", "tests/*", "test/*", "*.test.*", "*.spec.*", "__tests__/*", "*Test.java", "*_test.go")


def _git(repo: str | Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def is_test_file(path: str, globs: tuple[str, ...] = DEFAULT_TEST_GLOBS) -> bool:
    p = PurePosixPath(path)
    parts = p.parts
    for g in globs:
        if fnmatch.fnmatch(p.name, g):
            return True
        if "/" in g:  # directory pattern like tests/*
            d = g.split("/")[0]
            if d in parts[:-1]:
                return True
    return False


@dataclass
class MineConfig:
    test_cmd: str = "python -m pytest -q {tests}"
    max_commits: int = 300
    max_tasks: int = 60
    heldout_fraction: float = 0.3
    test_globs: tuple[str, ...] = DEFAULT_TEST_GLOBS
    max_files_changed: int = 15


def mine_git_tasks(repo: str | Path, cfg: MineConfig | None = None) -> TaskSuite:
    cfg = cfg or MineConfig()
    repo = Path(repo).resolve()
    log = _git(repo, "log", "--no-merges", f"-n{cfg.max_commits}", "--format=%H%x1f%P%x1f%s%x1f%b%x1e")
    tasks: list[Task] = []
    for rec in log.split("\x1e"):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        sha, parents, subject, body = (rec.split("\x1f") + ["", "", "", ""])[:4]
        parents = parents.split()
        if len(parents) != 1:
            continue  # root commit (nothing to start from) or merge
        changes = _git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", sha).splitlines()
        added_or_modified, src = [], []
        for line in changes:
            bits = line.split("\t")
            if len(bits) < 2:
                continue
            status, path = bits[0], bits[-1]
            if is_test_file(path, cfg.test_globs):
                if status[0] in "AM":
                    added_or_modified.append(path)
            else:
                src.append(path)
        if not added_or_modified or not src or len(changes) > cfg.max_files_changed:
            continue
        tests = sorted(added_or_modified)
        prompt = (
            f"{subject.strip()}\n\n{body.strip()}\n\n"
            f"Implement this change in the repository. These tests describe the expected behavior and must pass "
            f"(do not modify them): {', '.join(tests)}"
        ).replace("\n\n\n", "\n\n")
        tasks.append(
            Task(
                id=f"git-{sha[:10]}",
                prompt=prompt,
                verifier={"type": "command", "cmd": cfg.test_cmd.format(tests=" ".join(tests))},
                tags=["git"],
                meta={"repo": str(repo), "base": parents[0], "commit": sha, "test_files": tests, "src_files": sorted(src)},
            )
        )
        if len(tasks) >= cfg.max_tasks:
            break
    return TaskSuite(tasks).assign_splits(cfg.heldout_fraction)
