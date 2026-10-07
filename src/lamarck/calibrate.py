"""Calibrate a task suite to the agent before evolving its harness.

Harness evolution only gets a signal from tasks the agent *sometimes* solves. A task it
never solves (a "wall") or always solves gives every candidate harness the same score,
so it costs rollouts and adds nothing but noise. DarwinX partitions tasks the same way
(reliable / variance / wall); this step applies the partition up front.

Seen in practice: on real repositories, free models solved 0-12% of mined tasks; every
proposed edit then looked useless ("no net gain") because nearly every task was a wall.

Calibration rollouts use their own seeds and are never reused by the run: the tasks are
*selected* on these outcomes, so reusing them would bias the seed harness's measured
score (regression to the mean would flatter every later candidate).
"""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate import EvalResult, TrialCache, evaluate
from .harness import Harness
from .targets.base import Target
from .tasks import TaskSuite

CALIBRATION_SEED = 20_000  # disjoint from the seeds a run uses (0.., 10_000.. for held-out)


@dataclass
class TaskCalibration:
    task_id: str
    rate: float  # mean partial-credit score over k rollouts
    k: int
    zone: str  # "wall" | "learnable" | "solved"
    split: str
    errors: int = 0


def zone(rate: float, low: float = 0.0, high: float = 0.9) -> str:
    if rate <= low:
        return "wall"
    if rate >= high:
        return "solved"
    return "learnable"


def calibrate(target: Target, harness: Harness, suite: TaskSuite, k: int = 2, workers: int = 3,
              cache: TrialCache | None = None, low: float = 0.0, high: float = 0.9) -> tuple[list[TaskCalibration], EvalResult]:
    res = evaluate(target, harness, suite.tasks, k, seed=CALIBRATION_SEED, workers=workers, cache=cache)
    split = {t.id: t.split for t in suite.tasks}
    out = []
    for t in suite.tasks:
        trials = res.trials.get(t.id, [])
        rate = res.rates.get(t.id, 0.0)
        out.append(TaskCalibration(t.id, rate, len(trials), zone(rate, low, high), split[t.id], sum(tr.error is not None for tr in trials)))
    return out, res


def select(suite: TaskSuite, cals: list[TaskCalibration], heldout_fraction: float | None = None) -> TaskSuite:
    """Keep the learnable tasks. Splits are kept as mined unless a fraction is given."""
    keep = {c.task_id for c in cals if c.zone == "learnable" and not c.errors}
    kept = TaskSuite([t for t in suite.tasks if t.id in keep])
    return kept.assign_splits(heldout_fraction) if heldout_fraction is not None else kept


def table(cals: list[TaskCalibration]) -> str:
    lines = [f"{'task':<22} {'split':<8} {'score':>6}  zone"]
    for c in sorted(cals, key=lambda c: (c.zone != "learnable", -c.rate, c.task_id)):
        note = f"  ({c.errors} infra errors)" if c.errors else ""
        lines.append(f"{c.task_id:<22} {c.split:<8} {c.rate:>6.2f}  {c.zone}{note}")
    counts = {z: sum(c.zone == z for c in cals) for z in ("learnable", "wall", "solved")}
    lines.append(f"\n{counts['learnable']} learnable, {counts['wall']} walls (never solved), {counts['solved']} already solved")
    return "\n".join(lines)
