"""Weakness mining: turn raw rollouts into evidence a proposer can act on.

Follows DarwinX §2.4's partition of tasks:
  * reliable  - solved in (almost) every rollout: protect these (preservation set)
  * variance  - mixed passes and fails: contrast the agent's own pass vs fail (self-derived signal)
  * walls     - never solved: failure diagnosis, or a teacher trajectory if one exists
and Self-Harness's clustering of failures into recurring, model-specific patterns.
Raw traces are kept (Meta-Harness: compressed feedback loses the signal).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .evaluate import EvalResult
from .targets.base import Trial

RELIABLE_AT = 0.8
_NUM = re.compile(r"\d+(?:\.\d+)?")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")  # some repos force colored test output (e.g. addopts = --color=yes)
_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")


def signature(detail: str) -> str:
    """Normalize a failure message so the same failure on different tasks clusters together."""
    first = _ANSI.sub("", detail or "FAIL: unknown failure").strip().splitlines()[0]
    first = _HEX.sub("<id>", first)
    first = _NUM.sub("<n>", first)
    return first[:200]


@dataclass
class Cluster:
    signature: str
    task_ids: list[str]
    examples: list[Trial] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.task_ids)


@dataclass
class Evidence:
    reliable: list[str]
    variance: list[str]
    walls: list[str]
    clusters: list[Cluster]
    contrasts: list[tuple[Trial, Trial]]  # (pass, fail) pairs on the same task
    rates: dict[str, float]

    def summary(self) -> str:
        return f"{len(self.reliable)} reliable / {len(self.variance)} variance / {len(self.walls)} walls; {len(self.clusters)} failure clusters"

    def render(self, max_clusters: int = 6, max_chars: int = 1500) -> str:
        """Text form for an LLM proposer."""
        lines = [f"Task partition: {self.summary()}.", ""]
        for i, c in enumerate(self.clusters[:max_clusters], 1):
            lines.append(f"### Failure cluster {i}: {c.signature}  (tasks affected: {c.size})")
            for ex in c.examples[:2]:
                lines.append("Trace excerpt:\n" + _clip(ex.trace or ex.detail, max_chars))
            lines.append("")
        for i, (good, bad) in enumerate(self.contrasts[:2], 1):
            lines.append(f"### Contrast {i}: same task, one pass and one fail")
            lines.append("PASS trace:\n" + _clip(good.trace, max_chars // 2))
            lines.append("FAIL trace:\n" + _clip(bad.trace, max_chars // 2))
            lines.append("")
        return "\n".join(lines)


def _clip(text: str, n: int) -> str:
    text = _ANSI.sub("", text or "")
    return text if len(text) <= n else "…" + text[-n:]


def mine(result: EvalResult, examples_per_cluster: int = 3) -> Evidence:
    reliable, variance, walls = [], [], []
    for tid, r in sorted(result.rates.items()):
        if r >= RELIABLE_AT:
            reliable.append(tid)
        elif r == 0:
            walls.append(tid)
        else:
            variance.append(tid)

    groups: dict[str, list[Trial]] = defaultdict(list)
    for tid, trials in result.trials.items():
        if result.rates.get(tid, 0) >= RELIABLE_AT:
            continue
        for t in trials:
            if not t.passed and t.error is None:
                groups[signature(t.detail)].append(t)

    clusters = []
    for sig, trials in groups.items():
        task_ids = sorted({t.task_id for t in trials})
        clusters.append(Cluster(sig, task_ids, trials[:examples_per_cluster]))
    clusters.sort(key=lambda c: (-c.size, c.signature))

    contrasts = []
    for tid in variance:
        ts = result.trials[tid]
        good = next((t for t in ts if t.passed), None)
        bad = min((t for t in ts if not t.passed and t.error is None), key=lambda t: t.score, default=None)
        if good and bad:
            contrasts.append((good, bad))

    return Evidence(reliable, variance, walls, clusters, contrasts, dict(result.rates))
