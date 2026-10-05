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
    test_cmd: str = "{python} -m pytest -q {tests}"
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
        test_cmd = cfg.test_cmd.replace("{tests}", " ".join(tests))  # not .format(): {python} must survive
        prompt = (
            f"{subject.strip()}\n\n{body.strip()}\n\n"
            f"Implement this change in the repository. These tests describe the expected behavior and must pass "
            f"(do not modify them): {', '.join(tests)}\n"
            f"Run them with: {test_cmd}"
        ).replace("\n\n\n", "\n\n")
        tasks.append(
            Task(
                id=f"git-{sha[:10]}",
                prompt=prompt,
                verifier={"type": "command", "cmd": test_cmd},
                tags=["git"],
                meta={"repo": str(repo), "base": parents[0], "commit": sha, "test_files": tests, "src_files": sorted(src)},
            )
        )
        if len(tasks) >= cfg.max_tasks:
            break
    return TaskSuite(tasks).assign_splits(cfg.heldout_fraction)


@dataclass
class Validation:
    task_id: str
    ok: bool
    fails_before: bool
    passes_after: bool
    seconds: float
    note: str = ""


def validate_task(task: Task, max_seconds: float = 120.0) -> Validation:
    """SWE-bench-style fail-to-pass check: the task's tests must FAIL on the base snapshot
    (with the new tests written in) and PASS on the fixing commit."""
    import shutil
    import tempfile
    import time

    from .targets.command import materialize
    from .tasks import verify

    meta = task.meta
    tmp = Path(tempfile.mkdtemp(prefix="lamarck-validate-"))
    try:
        before = materialize(meta["repo"], meta["base"], tmp / "before", meta["commit"], meta.get("test_files", []))
        fails_before = not verify(task, "", workdir=before, timeout=int(max_seconds) + 30).passed
        after = materialize(meta["repo"], meta["commit"], tmp / "after")
        t0 = time.monotonic()
        passes_after = verify(task, "", workdir=after, timeout=int(max_seconds) + 30).passed
        secs = time.monotonic() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    note = ""
    if not passes_after:
        note = "tests do not pass even with the real fix (environment or dependencies?)"
    elif not fails_before:
        note = "tests already pass before the fix (nothing to do)"
    elif secs > max_seconds:
        note = f"tests too slow ({secs:.0f}s > {max_seconds:.0f}s)"
    return Validation(task.id, fails_before and passes_after and secs <= max_seconds, fails_before, passes_after, secs, note)


def validate_suite(suite: TaskSuite, max_seconds: float = 120.0, log=print) -> tuple[TaskSuite, list[Validation]]:
    results = []
    for t in suite.tasks:
        v = validate_task(t, max_seconds)
        results.append(v)
        if log:
            log(f"  {'OK  ' if v.ok else 'DROP'} {t.id} ({v.seconds:.1f}s) {v.note}")
    keep = {v.task_id for v in results if v.ok}
    return TaskSuite([t for t in suite.tasks if t.id in keep]), results
