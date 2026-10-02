"""The evolution loop.

    for each iteration:
        parent   <- archive.select_parent()            (exploit best lineage, sometimes broaden)
        evidence <- mine(parent's rollouts)             (clusters, contrasts, walls)
        for proposal in proposer(parent, evidence, annealed budget, history):
            critic screen                               (leakage / size / budget / surface)
            screen on train with k_screen rollouts      (permissive: g > 0, R <= δ, cost rule)
            confirm with k_confirm rollouts             (strict: gain above noise, R <= δ)
        every few iterations: merge complementary specialists
    finally: evaluate the pre-registered best variant on HELD-OUT tasks and ship only
    if the held-out gain clears the noise band.

The held-out split is never shown to the proposer and never used for selection.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .archive import CONFIRMED, PROMOTED, REVERTED, Archive, Node
from .critic import Critic, edit_budget
from .evaluate import EvalResult, evaluate, noise_band
from .harness import EditConflict, Harness
from .miner import mine
from .proposers.base import HistoryEntry, Proposer
from .selection import Decision, GateConfig, compare, confirm, heldout_gate, screen
from .targets.base import Target
from .tasks import TaskSuite, task_specific_literals


@dataclass
class EvolveConfig:
    iterations: int = 8
    k_screen: int = 2
    k_confirm: int = 4
    k_heldout: int = 4
    beta: float = 0.3
    proposals_per_iter: int = 3
    merge_every: int = 3
    budget_start: int = 3
    budget_end: int = 1
    workers: int = 4
    seed: int = 0
    max_total_tokens: int | None = None
    max_rollouts: int | None = None  # request cap (e.g. free-model daily quotas); held-out check is reserved
    gate: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        if self.k_confirm <= self.k_screen:
            raise ValueError("k_confirm must be larger than k_screen (confirmation re-tests with more rollouts)")


@dataclass
class RunResult:
    archive: Archive
    best: Node
    root_heldout: EvalResult | None
    best_heldout: EvalResult | None
    decision: Decision
    history: list[HistoryEntry]
    total_tokens: int
    total_rollouts: int
    log: list[str]

    @property
    def best_harness(self) -> Harness:
        return self.archive.harness(self.best)


class Evolver:
    def __init__(
        self,
        target: Target,
        proposer: Proposer,
        suite: TaskSuite,
        seed_harness: Harness,
        config: EvolveConfig | None = None,
        critic: Critic | None = None,
        log: Callable[[str], None] | None = print,
    ):
        self.target = target
        self.proposer = proposer
        self.suite = suite
        self.cfg = config or EvolveConfig()
        self.train = suite.train()
        self.heldout = suite.heldout()
        if not self.train:
            raise ValueError("task suite has no training tasks")
        self.critic = critic or Critic(denylist=task_specific_literals(suite))
        self.archive = Archive(seed_harness)
        self.history: list[HistoryEntry] = []
        self.evals: dict[str, EvalResult] = {}
        self.noise: dict[str, float] = {}
        self.tokens = 0
        self.rollouts = 0
        self.lines: list[str] = []
        self._log = log

    # ---- helpers --------------------------------------------------------
    def log(self, msg: str) -> None:
        self.lines.append(msg)
        if self._log:
            self._log(msg)

    def _eval(self, h: Harness, tasks, k: int, seed_offset: int = 0) -> EvalResult:
        res = evaluate(self.target, h, tasks, k, seed=self.cfg.seed + seed_offset, workers=self.cfg.workers)
        self.tokens += res.tokens
        self.rollouts += sum(len(v) for v in res.trials.values())
        return res

    @staticmethod
    def _tokens_per_rollout(res: EvalResult) -> float:
        n = sum(len(v) for v in res.trials.values())
        return res.tokens / n if n else 0.0

    def _over_budget(self) -> bool:
        if self.cfg.max_total_tokens is not None and self.tokens >= self.cfg.max_total_tokens:
            return True
        if self.cfg.max_rollouts is not None:
            reserve = 2 * len(self.heldout) * self.cfg.k_heldout  # always keep enough for the final held-out check
            next_candidate = len(self.train) * self.cfg.k_confirm
            return self.rollouts + next_candidate + reserve > self.cfg.max_rollouts
        return False

    # ---- main loop ------------------------------------------------------
    def run(self) -> RunResult:
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        n = len(self.train)
        self.log(f"Evaluating seed harness on {n} train tasks (k={cfg.k_confirm})...")
        root_res = self._eval(self.archive.root_harness, self.train, cfg.k_confirm)
        root = self.archive.add_root(root_res.rates, self._tokens_per_rollout(root_res))
        self.evals[root.id] = root_res
        self.noise[root.id] = noise_band(root_res)
        self.log(f"  seed: train mean {root_res.mean:.3f}  (noise +/-{self.noise[root.id]:.3f})")

        for it in range(cfg.iterations):
            if self._over_budget():
                self.log(f"Budget reached ({self.tokens} tokens, {self.rollouts} rollouts); stopping early.")
                break
            parent = self.archive.select_parent(rng, cfg.beta)
            parent_res = self.evals[parent.id]
            evidence = mine(parent_res)
            budget = edit_budget(it, cfg.iterations, cfg.budget_start, cfg.budget_end)
            self.log(f"[iter {it + 1}/{cfg.iterations}] parent {parent.id} (G={parent.lineage_gain:+.2f}) | {evidence.summary()} | edit budget {budget}")
            if not evidence.clusters:
                self.log("  parent solves every train task reliably; nothing to fix here")
                continue
            proposals = self.proposer.propose(self.archive.harness(parent), evidence, budget, self.history, it)[: cfg.proposals_per_iter]
            if not proposals:
                self.log("  proposer returned no proposals")
            for prop in proposals:
                self._consider(parent, prop, budget, it, n)
                if self._over_budget():
                    break
            if cfg.merge_every and (it + 1) % cfg.merge_every == 0:
                self._try_merge(it, n)

        return self._finish()

    def _consider(self, parent: Node, prop, budget: int, it: int, n: int) -> None:
        cfg = self.cfg
        lineage = parent.lineage + list(prop.edits)
        child_h = self.archive.root_harness.apply(lineage)
        fp = child_h.fingerprint()
        node = Node(self.archive.new_id(), parent.id, list(prop.edits), lineage, fp, REVERTED, hypothesis=prop.hypothesis, iteration=it)

        def reject(reason: str, **extra) -> None:
            node.reason = reason
            for k, v in extra.items():
                setattr(node, k, v)
            self.archive.add(node)
            self.history.append(HistoryEntry(prop.hypothesis, "rejected", reason, node.gain))
            self.log(f"  x {prop.hypothesis!r}: {reason}")

        if fp == parent.fingerprint:
            return reject("no-op edit (harness unchanged)")
        if any(m.fingerprint == fp for m in self.archive.nodes.values()):
            self.log(f"  | {prop.hypothesis!r}: already evaluated, skipping")
            return
        why = self.critic.review(list(prop.edits), budget)
        if why:
            return reject(f"critic: {why}")

        parent_res = self.evals[parent.id]
        screen_res = self._eval(child_h, self.train, cfg.k_screen)
        tok_delta = self._tokens_per_rollout(screen_res) - parent.tokens_per_rollout
        cmp1 = compare(parent_res.rates, screen_res.rates, cfg.gate.solved_threshold, cfg.gate.preserve_threshold)
        d1 = screen(cmp1, self.noise[parent.id], n, cfg.gate, tok_delta)
        if not d1.accept:
            return reject(f"screen: {d1.reason}", gain=cmp1.gain, regression=cmp1.regression, klass=cmp1.klass, train_rates=screen_res.rates)

        extra = self._eval(child_h, self.train, cfg.k_confirm - cfg.k_screen, seed_offset=cfg.k_screen)
        full = screen_res.merge(extra)
        cmp2 = compare(parent_res.rates, full.rates, cfg.gate.solved_threshold, cfg.gate.preserve_threshold)
        child_noise = noise_band(full)
        combined = math.sqrt(self.noise[parent.id] ** 2 + child_noise**2)
        d2 = confirm(cmp2, combined, n, cfg.gate, tok_delta)

        node.gain, node.regression, node.klass = cmp2.gain, cmp2.regression, cmp2.klass
        node.train_rates = full.rates
        node.tokens_per_rollout = self._tokens_per_rollout(full)
        node.reason = d2.reason
        if d2.accept:
            node.status = CONFIRMED
            node.lineage_gain = parent.lineage_gain + cmp2.gain
            verdict = "confirmed"
            self.log(f"  + {prop.hypothesis!r}: {d2.reason} -> {node.id} [{cmp2.klass}]")
        else:
            node.status = PROMOTED  # may be an ancestor via broadening, may not steer
            node.lineage_gain = parent.lineage_gain + min(cmp2.gain, 0.0)
            verdict = "promoted_unconfirmed"
            self.log(f"  ~ {prop.hypothesis!r}: passed screen, failed re-test ({d2.reason})")
        self.archive.add(node)
        self.evals[node.id] = full
        self.noise[node.id] = child_noise
        self.history.append(HistoryEntry(prop.hypothesis, verdict, node.reason, node.gain))

    def _try_merge(self, it: int, n: int) -> None:
        """Recombine complementary specialists; keep the child only if it covers the union of their wins."""
        cfg = self.cfg
        specs = self.archive.specialists(cfg.gate.solved_threshold)
        if len(specs) < 2:
            return
        best_pair, best_cover = None, -1
        for i, a in enumerate(specs):
            for b in specs[i + 1 :]:
                if _is_ancestor(self.archive, a, b) or _is_ancestor(self.archive, b, a):
                    continue
                cover = len(a.solved() | b.solved())
                if cover > best_cover:
                    best_pair, best_cover = (a, b), cover
        if best_pair is None:
            return
        a, b = best_pair
        if any(set(m.merged_from) == {a.id, b.id} for m in self.archive.nodes.values()):
            return
        try:
            lineage = Harness.check_mergeable([a.lineage, b.lineage])
        except EditConflict as exc:
            self.log(f"  merge {a.id}+{b.id} skipped: {exc}")
            return
        new_edits = [e for e in lineage if e not in a.lineage]
        why = self.critic.review(new_edits or lineage, budget=max(1, len(new_edits)))
        if why:
            self.log(f"  merge {a.id}+{b.id} rejected by critic: {why}")
            return
        child_h = self.archive.root_harness.apply(lineage)
        if any(m.fingerprint == child_h.fingerprint() for m in self.archive.nodes.values()):
            return
        parent = a if a.lineage_gain >= b.lineage_gain else b
        res = self._eval(child_h, self.train, cfg.k_confirm)
        cmp = compare(self.evals[parent.id].rates, res.rates, cfg.gate.solved_threshold, cfg.gate.preserve_threshold)
        need = a.solved() | b.solved()
        covered = res.solved(cfg.gate.solved_threshold) >= need
        node = Node(self.archive.new_id(), parent.id, new_edits, lineage, child_h.fingerprint(), REVERTED,
                    hypothesis=f"merge {a.id}+{b.id}", iteration=it, merged_from=[a.id, b.id],
                    gain=cmp.gain, regression=cmp.regression, klass=cmp.klass, train_rates=res.rates,
                    tokens_per_rollout=self._tokens_per_rollout(res))
        if covered and cmp.gain > 0 and cmp.regression <= cfg.gate.max_regression:
            node.status, node.reason = CONFIRMED, f"covers union of parents' wins ({len(need)} tasks), g={cmp.gain:+.2f}"
            node.lineage_gain = parent.lineage_gain + cmp.gain
            self.log(f"  merge {a.id}+{b.id} -> {node.id}: {node.reason}")
        else:
            node.reason = "merge did not cover both parents' wins" if not covered else f"merge not better than parent (g={cmp.gain:+.2f})"
            self.log(f"  merge {a.id}+{b.id} reverted: {node.reason}")
        self.archive.add(node)
        self.evals[node.id] = res
        self.noise[node.id] = noise_band(res)
        self.history.append(HistoryEntry(node.hypothesis, node.status, node.reason, node.gain))

    def _finish(self) -> RunResult:
        cfg = self.cfg
        best = self.archive.best()
        root = self.archive.root
        root_h, best_h = None, None
        if best.id == root.id:
            decision = Decision(False, "no variant beat the seed harness on train; nothing to ship")
            self.log(decision.reason)
        elif not self.heldout:
            decision = Decision(False, "no held-out tasks: refusing to claim an improvement without one")
            self.log(decision.reason)
        else:
            self.log(f"Held-out check: seed vs best {best.id} on {len(self.heldout)} never-seen tasks (k={cfg.k_heldout})...")
            root_h = self._eval(self.archive.root_harness, self.heldout, cfg.k_heldout, seed_offset=10_000)
            best_h = self._eval(self.archive.harness(best), self.heldout, cfg.k_heldout, seed_offset=10_000)
            noise = math.sqrt(noise_band(root_h) ** 2 + noise_band(best_h) ** 2)
            decision = heldout_gate(root_h.mean, best_h.mean, noise, cfg.gate.noise_mult)
            root.heldout_mean, best.heldout_mean = root_h.mean, best_h.mean
            train_gain = sum(best.train_rates.values()) / len(best.train_rates) - sum(root.train_rates.values()) / len(root.train_rates)
            self.log(f"  train: {train_gain * 100:+.1f} pts | held-out: {(best_h.mean - root_h.mean) * 100:+.1f} pts -> {'SHIP' if decision.accept else 'DO NOT SHIP'} ({decision.reason})")
        return RunResult(self.archive, best, root_h, best_h, decision, self.history, self.tokens, self.rollouts, self.lines)


def _is_ancestor(archive: Archive, a: Node, b: Node) -> bool:
    """True if a is an ancestor of b."""
    cur = b
    while cur.parent_id is not None:
        if cur.parent_id == a.id:
            return True
        cur = archive.nodes[cur.parent_id]
    return False
