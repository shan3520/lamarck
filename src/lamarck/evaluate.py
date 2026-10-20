"""Noise-aware evaluation: avg@k per task.

Agentic outcomes are stochastic (repeated runs diverge even at temperature 0;
2-6 point swings on SWE-bench Verified, per DarwinX §10). Every decision in
Lamarck is therefore made on per-task solve *rates* over k rollouts, with an
explicit noise band.
"""

from __future__ import annotations

import json
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .harness import Harness
from .tasks import Task
from .targets.base import Target, Trial


class FatalRunError(RuntimeError):
    """An error no amount of retrying fixes (out of credits, bad credentials). Aborts the run."""


FATAL_STATUS = {401: "authentication failed (check the API key)", 402: "out of credits / payment required", 403: "permission denied for this model or key",
                404: "model not found for this key (removed or renamed by the provider?)"}  # NVIDIA dropped a model mid-test


def _daily_quota_exhausted(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return getattr(exc, "status_code", None) == 429 and ("per-day" in msg or "per day" in msg)


def _rate_limited(exc: BaseException) -> bool:
    # a 429 that survived the client's retries: the run must pause, never score the task as failed
    return getattr(exc, "status_code", None) == 429


def _network_down(exc: BaseException) -> bool:
    return getattr(exc, "status_code", None) == 0 and "network" in str(exc).lower()


def is_fatal(exc: BaseException) -> bool:
    return (isinstance(exc, FatalRunError) or getattr(exc, "status_code", None) in FATAL_STATUS
            or _rate_limited(exc) or _network_down(exc))


def describe_fatal(exc: BaseException) -> str:
    if isinstance(exc, FatalRunError):
        return str(exc)
    if _network_down(exc):
        return "network unavailable (no internet connection?)"
    if _daily_quota_exhausted(exc):
        return "daily free-model request quota exhausted (resets at 00:00 UTC) (HTTP 429)"
    if _rate_limited(exc):
        return "rate limit still exceeded after retries (HTTP 429); lower the request rate or wait a few minutes"
    return f"{FATAL_STATUS[exc.status_code]} (HTTP {exc.status_code})"


@dataclass
class EvalResult:
    rates: dict[str, float] = field(default_factory=dict)  # task_id -> avg@k solve rate
    trials: dict[str, list[Trial]] = field(default_factory=dict)
    k: int = 0
    tokens: int = 0
    infra_errors: int = 0

    @property
    def mean(self) -> float:
        """Mean partial-credit score."""
        return sum(self.rates.values()) / len(self.rates) if self.rates else 0.0

    @property
    def pass_rate(self) -> float:
        """Share of rollouts that fully passed (the strict number)."""
        ts = [t for v in self.trials.values() for t in v]
        return sum(t.passed and t.error is None for t in ts) / len(ts) if ts else 0.0

    def solved(self, threshold: float = 0.5) -> set[str]:
        return {t for t, r in self.rates.items() if r >= threshold}

    def merge(self, other: "EvalResult") -> "EvalResult":
        """Pool rollouts from two evaluations of the *same* harness (e.g. screen + confirm)."""
        trials = {t: list(v) for t, v in self.trials.items()}
        for t, v in other.trials.items():
            trials.setdefault(t, []).extend(v)
        return _summarize(trials, self.tokens + other.tokens)

    def to_dict(self) -> dict:
        return {"rates": self.rates, "k": self.k, "tokens": self.tokens, "infra_errors": self.infra_errors}


def _summarize(trials: dict[str, list[Trial]], tokens: int) -> EvalResult:
    rates, errors, k = {}, 0, 0
    for tid, ts in trials.items():
        scored = [t for t in ts if t.error is None]
        errors += len(ts) - len(scored)
        # an infra error counts as a failure with the full denominator (RRSI: missing trial = 0)
        rates[tid] = sum(t.score for t in scored) / len(ts) if ts else 0.0  # mean partial-credit score
        k = max(k, len(ts))
    return EvalResult(rates=rates, trials=trials, k=k, tokens=tokens, infra_errors=errors)


class TrialCache:
    """Append-only store of finished rollouts, keyed by (harness fingerprint, task, seed).

    This is what makes multi-day runs resumable: a resumed run replays the whole search,
    and every rollout that already happened is answered from disk at zero cost, so the
    run continues exactly where it stopped (even halfway through an evaluation)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self._needs_newline = False
        if self.path and self.path.is_file():
            text = self.path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a write cut off by a shutdown or power loss: that rollout is simply redone
                    self._data[rec["key"]] = rec["trial"]
            self._needs_newline = bool(text) and not text.endswith("\n")

    @staticmethod
    def key(fingerprint: str, task_id: str, seed: int) -> str:
        return f"{fingerprint}|{task_id}|{seed}"

    def get(self, key: str) -> Trial | None:
        with self._lock:
            d = self._data.get(key)
            if d is not None:
                self.hits += 1
        return Trial.from_dict(d) if d is not None else None

    def put(self, key: str, trial: Trial) -> None:
        d = trial.to_dict()
        with self._lock:
            self._data[key] = d
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    if self._needs_newline:  # never glue a new record onto a cut-off line
                        f.write("\n")
                        self._needs_newline = False
                    f.write(json.dumps({"key": key, "trial": d}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def evaluate(target: Target, harness: Harness, tasks: list[Task], k: int, seed: int = 0, workers: int = 4, cache: TrialCache | None = None) -> EvalResult:
    jobs = [(task, seed + i) for task in tasks for i in range(k)]
    fp = harness.fingerprint()

    def run(job: tuple[Task, int]) -> Trial:
        task, s = job
        key = TrialCache.key(fp, task.id, s)
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit
        trial = _run_one(task, s)
        if cache is not None:
            cache.put(key, trial)
        return trial

    def _run_one(task: Task, s: int) -> Trial:
        try:
            return target.run(harness, task, s)
        except Exception as exc:  # infra failure: record it, never crash the search...
            if is_fatal(exc):  # ...unless retrying can't help (billing, auth): stop the whole run
                raise FatalRunError(describe_fatal(exc)) from exc
            return Trial(task.id, False, detail="FAIL: infrastructure error", error=f"{type(exc).__name__}: {exc}")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, jobs))
    else:
        results = [run(j) for j in jobs]
    trials: dict[str, list[Trial]] = {t.id: [] for t in tasks}
    for tr in results:
        trials[tr.task_id].append(tr)
    return _summarize(trials, sum(t.tokens for t in results))


def noise_band(result: EvalResult, n_boot: int = 400, seed: int = 0) -> float:
    """Bootstrap standard error of the mean solve rate, in rate units (0..1).

    Resamples rollouts within each task, which captures the run-to-run noise a
    re-evaluation of the *same* harness would show."""
    if not result.trials:
        return 0.0
    rng = random.Random(seed)
    if min(len(ts) for ts in result.trials.values()) < 2:
        # One rollout per task carries no within-task noise information (it would read as 0).
        # Fall back to resampling tasks: a conservative standard error of the mean.
        per_task = list(result.rates.values())
        boots = [sum(rng.choice(per_task) for _ in per_task) / len(per_task) for _ in range(n_boot)]
        mu = sum(boots) / len(boots)
        return math.sqrt(sum((m - mu) ** 2 for m in boots) / (len(boots) - 1))
    means = []
    for _ in range(n_boot):
        total = 0.0
        for ts in result.trials.values():
            sample = [rng.choice(ts) for _ in ts]
            total += sum(t.score for t in sample if t.error is None) / len(sample)
        means.append(total / len(result.trials))
    mu = sum(means) / len(means)
    return math.sqrt(sum((m - mu) ** 2 for m in means) / (len(means) - 1))
