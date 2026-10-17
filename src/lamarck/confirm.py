"""One pre-registered confirmation of a run's held-out verdict.

A run's held-out check can land in the noise band (run 8: +5.0 pts against a 9.3-pt band).
Re-testing until it passes would be p-hacking, so confirmation is a single look, declared in
advance: the seed and best harnesses get `extra_k` more rollouts on the *same* never-seen
tasks, with seeds the run never used; these are pooled with the run's held-out rollouts, and
the run's own gate (gain > noise band) is applied once. A second confirmation is refused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .archive import Archive
from .evaluate import TrialCache, evaluate, noise_band
from .selection import GateConfig, heldout_gate
from .targets.base import Target
from .tasks import Task

HELDOUT_SEED = 10_000  # the evolver's held-out seed offset


@dataclass
class Confirmation:
    best: str
    tasks: int
    k_total: int
    seed_mean: float
    best_mean: float
    seed_pass: float
    best_pass: float
    delta_pts: float
    noise_pts: float
    ship: bool
    reason: str


def confirm(target: Target, archive: Archive, best_id: str, heldout: list[Task], k_done: int, extra_k: int,
            gate: GateConfig, cache: TrialCache | None = None, workers: int = 3) -> Confirmation:
    best = archive.nodes[best_id]
    results = []
    for h in (archive.root_harness, archive.harness(best)):
        done = evaluate(target, h, heldout, k_done, seed=HELDOUT_SEED, workers=workers, cache=cache)  # replayed from the run
        more = evaluate(target, h, heldout, extra_k, seed=HELDOUT_SEED + k_done, workers=workers, cache=cache)
        results.append(done.merge(more))
    root_r, best_r = results
    noise = math.sqrt(noise_band(root_r) ** 2 + noise_band(best_r) ** 2)
    d = heldout_gate(root_r.mean, best_r.mean, noise, gate.noise_mult)
    return Confirmation(best_id, len(heldout), k_done + extra_k, root_r.mean, best_r.mean, root_r.pass_rate, best_r.pass_rate,
                        (best_r.mean - root_r.mean) * 100, noise * 100 * gate.noise_mult, d.accept, d.reason)


def as_dict(c: Confirmation) -> dict:
    return asdict(c)
