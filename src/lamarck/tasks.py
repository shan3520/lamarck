"""Tasks and verifiers.

Every task carries a verifier: fitness comes only from measured pass/fail,
never from a model's opinion of itself (DarwinX: "no gold solutions, no
hand-picked winners"). Tasks are split into ``train`` (the loop may look at
them) and ``heldout`` (the loop never sees them; used only to decide whether a
change ships).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRAIN = "train"
HELDOUT = "heldout"
VERIFIER_TYPES = ("exact", "contains", "not_contains", "regex", "json", "command", "all")


@dataclass
class Task:
    id: str
    prompt: str
    verifier: dict
    split: str = TRAIN
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=str(d["id"]),
            prompt=d.get("prompt", ""),
            verifier=d.get("verifier", {}),
            split=d.get("split", TRAIN),
            tags=list(d.get("tags", [])),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class TaskSuite:
    tasks: list[Task]

    def __post_init__(self) -> None:
        ids = [t.id for t in self.tasks]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate task ids: {sorted(dupes)}")
        for t in self.tasks:
            if t.split not in (TRAIN, HELDOUT):
                raise ValueError(f"task {t.id}: split must be {TRAIN!r} or {HELDOUT!r}")

    def train(self) -> list[Task]:
        return [t for t in self.tasks if t.split == TRAIN]

    def heldout(self) -> list[Task]:
        return [t for t in self.tasks if t.split == HELDOUT]

    def by_id(self) -> dict[str, Task]:
        return {t.id: t for t in self.tasks}

    def assign_splits(self, heldout_fraction: float = 0.3, salt: str = "lamarck") -> "TaskSuite":
        """Deterministic, hash-based split so re-runs never leak held-out tasks into training."""
        out = []
        for t in self.tasks:
            h = int(hashlib.sha256(f"{salt}:{t.id}".encode()).hexdigest(), 16) % 10_000
            split = HELDOUT if h < heldout_fraction * 10_000 else TRAIN
            out.append(Task(t.id, t.prompt, t.verifier, split, t.tags, t.meta))
        return TaskSuite(out)

    @classmethod
    def load(cls, path: str | Path) -> "TaskSuite":
        """Relative `meta.repo` paths (git-mined tasks) are resolved against the task file's folder,
        so a task suite can be committed and used on any machine."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["tasks"] if isinstance(data, dict) else data
        tasks = [Task.from_dict(d) for d in items]
        for t in tasks:
            repo = t.meta.get("repo")
            if repo and not Path(repo).is_absolute():
                t.meta["repo"] = str((path.parent / repo).resolve())
        return cls(tasks)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"tasks": [t.to_dict() for t in self.tasks]}, indent=2), encoding="utf-8")


@dataclass
class Verdict:
    passed: bool
    detail: str = ""
    score: float | None = None  # partial credit in [0, 1]; defaults to 1.0 / 0.0

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = 1.0 if self.passed else 0.0


def verify(task: Task, output: str, workdir: str | Path | None = None, timeout: int = 600) -> Verdict:
    """Check `output` against the task's verifier. `detail` explains a failure in a
    task-agnostic way so the weakness miner can cluster failures."""
    return _check(task.verifier, output, workdir, timeout)


def _check(v: dict, output: str, workdir, timeout: int) -> Verdict:
    kind = v.get("type")
    if kind == "exact":
        ok = output.strip() == str(v["value"]).strip()
        return Verdict(ok, "" if ok else "FAIL: output does not exactly match the expected answer")
    if kind == "contains":
        missing = [s for s in v["values"] if s not in output]
        if missing:
            label = v.get("label") or f"missing required text {missing[0]!r}"
            return Verdict(False, f"FAIL: {label}")
        return Verdict(True)
    if kind == "not_contains":
        present = [s for s in v["values"] if s in output]
        if present:
            label = v.get("label") or f"contains forbidden text {present[0]!r}"
            return Verdict(False, f"FAIL: {label}")
        return Verdict(True)
    if kind == "regex":
        ok = re.search(v["pattern"], output, flags=re.MULTILINE) is not None
        label = v.get("label") or f"output does not match pattern {v['pattern']!r}"
        return Verdict(ok, "" if ok else f"FAIL: {label}")
    if kind == "json":
        return _check_json(v, output)
    if kind == "command":
        return _check_command(v, workdir, timeout)
    if kind == "all":
        # Partial credit + every failing rule listed: a fix for one of several required
        # conventions must be visible as progress, and the proposer must see all of them.
        results = [_check(sub, output, workdir, timeout) for sub in v["checks"]]
        failed = [r for r in results if not r.passed]
        score = sum(r.score for r in results) / len(results) if results else 1.0
        if not failed:
            return Verdict(True, score=1.0)
        detail = failed[0].detail + "".join(f"\nalso {r.detail}" for r in failed[1:])
        return Verdict(False, detail, score=score)
    raise ValueError(f"unknown verifier type {kind!r}; expected one of {VERIFIER_TYPES}")


def _extract_json(output: str):
    text = output.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _check_json(v: dict, output: str) -> Verdict:
    strict = v.get("strict", False)
    try:
        obj = json.loads(output.strip()) if strict else _extract_json(output)
    except (json.JSONDecodeError, ValueError):
        return Verdict(False, "FAIL: output is not valid JSON" + (" (prose or code fences around JSON are not allowed)" if strict else ""))
    if not isinstance(obj, dict):
        return Verdict(False, "FAIL: JSON output must be an object")
    for key in v.get("required_keys", []):
        if key not in obj:
            return Verdict(False, f"FAIL: JSON is missing required key {key!r}")
    for key, expected in v.get("equals", {}).items():
        if obj.get(key) != expected:
            return Verdict(False, f"FAIL: JSON field {key!r} has the wrong value")
    for key, pattern in v.get("matches", {}).items():
        if not re.fullmatch(pattern, str(obj.get(key, ""))):
            label = v.get("labels", {}).get(key) or f"JSON field {key!r} has the wrong format"
            return Verdict(False, f"FAIL: {label}")
    return Verdict(True)


def expand_python(text: str) -> str:
    """Replace the `{python}` placeholder with the running interpreter (quoted), so committed
    task files work on any machine and always use the environment Lamarck runs in."""
    import sys

    return text.replace("{python}", f'"{Path(sys.executable).as_posix()}"')


def plain_env() -> dict:
    """Environment for test/agent subprocesses: UTF-8 I/O and no ANSI colors (they waste tokens)."""
    import os

    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "NO_COLOR": "1", "PY_COLORS": "0", "FORCE_COLOR": "0", "TERM": "dumb"})
    return env


def _check_command(v: dict, workdir, timeout: int) -> Verdict:
    if workdir is None:
        return Verdict(False, "FAIL: command verifier needs a working directory")
    try:
        proc = subprocess.run(expand_python(v["cmd"]), shell=True, cwd=str(workdir), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=v.get("timeout", timeout), env=plain_env())
    except subprocess.TimeoutExpired:
        return Verdict(False, "FAIL: verification command timed out")
    if proc.returncode == 0:
        return Verdict(True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    return Verdict(False, "FAIL: verification command failed\n" + "\n".join(tail))


def task_specific_literals(suite: TaskSuite) -> set[str]:
    """Strings that identify exactly one task (ids, and verifier literals used by a
    single task). A harness edit containing one of these is memorizing a task,
    not learning a reusable behavior — the leakage RRSI's critic and the
    'Bad Genius' paper warn about."""
    counts: dict[str, int] = {}
    for t in suite.tasks:
        for lit in set(_verifier_literals(t.verifier)):
            counts[lit] = counts.get(lit, 0) + 1
    literals = {lit for lit, n in counts.items() if n == 1 and len(lit) >= 4}
    literals |= {t.id for t in suite.tasks if len(t.id) >= 4}
    # identifiers in prompts (ticket numbers, order ids, hashes) that occur in a single task
    prompt_counts: dict[str, int] = {}
    for t in suite.tasks:
        for tok in set(_ID_TOKEN.findall(t.prompt)):
            prompt_counts[tok] = prompt_counts.get(tok, 0) + 1
    literals |= {tok for tok, n in prompt_counts.items() if n == 1}
    return literals


_ID_TOKEN = re.compile(r"\b(?=[\w-]*\d)[\w-]{5,}\b")


def _verifier_literals(v: dict) -> list[str]:
    kind = v.get("type")
    if kind == "exact":
        return [str(v["value"]).strip()]
    if kind in ("contains", "not_contains"):
        return [str(s) for s in v["values"]]
    if kind == "json":
        return [str(x) for x in v.get("equals", {}).values()]
    if kind == "all":
        return [lit for sub in v["checks"] for lit in _verifier_literals(sub)]
    return []
