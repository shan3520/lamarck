"""A *simulated* agent world for tests and the zero-cost demo.

Nothing here is evidence that harness evolution works — the research papers are
that evidence. The simulation exists to exercise every part of the engine
(mining, proposing, the critic, preserve-and-extend selection, merging, the
held-out gate) deterministically and for free, including the failure modes the
papers warn about:

* task-specific "cheat" edits that boost a training task but not held-out tasks
  (overfitting; RRSI, Bad Genius),
* edits that help one task family and hurt another (cross-task interference; DarwinX),
* harness bloat that costs tokens and slightly lowers pass rates,
* noisy pass/fail outcomes (avg@k needed; DarwinX §2.5).

A harness gains capability NAME when any component contains the marker ``[cap:NAME]``.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from .harness import Edit, Harness
from .proposers.base import HistoryEntry, Proposal
from .tasks import HELDOUT, TRAIN, Task, TaskSuite
from .targets.base import Trial

CAP_RE = re.compile(r"\[cap:([a-z0-9_:\-]+)\]")

# capability -> (failure message when missing, generic fix skill)
CATALOG: dict[str, tuple[str, str]] = {
    "verify": (
        "FAIL: declared done without checking the output against the acceptance criteria",
        "Before finishing, restate the task's acceptance criteria as a checklist and check the final output against every item. [cap:verify]",
    ),
    "json": (
        "FAIL: output is not valid JSON (prose around the JSON)",
        "When a task asks for JSON, output only the JSON object: no prose, no code fences. Validate it parses. [cap:json]",
    ),
    "tests": (
        "FAIL: changed code without running the test suite",
        "After any code change, run the project's tests and fix failures before declaring success. [cap:tests]",
    ),
    "context": (
        "FAIL: lost track of earlier instructions in a long session",
        "In long tasks, keep a short running summary of constraints and re-read it before each major step. [cap:context]",
    ),
    "deps": (
        "FAIL: tool call failed because a dependency was not installed",
        "If a command fails with a missing module or binary, install the dependency and retry once before changing approach. [cap:deps]",
    ),
}

# capabilities that hurt tasks tagged with the listed need
INTERFERENCE: dict[str, tuple[str, str]] = {
    # cap -> (need it breaks, failure message)
    "terse": ("explain", "FAIL: answer omitted the explanation the task asked for"),
}
TERSE_SKILL = "Keep every answer as short as possible; never add explanations. [cap:terse]"


def capabilities(h: Harness) -> set[str]:
    caps: set[str] = set()
    for text in h.components.values():
        caps.update(CAP_RE.findall(text))
    return caps


def _unit(*parts) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


@dataclass
class SimulatedTarget:
    """Deterministic stand-in for 'frozen model + harness'."""

    name: str = "simulated"
    ceiling: float = 0.95
    bloat_free_chars: int = 2500
    bloat_penalty_per_1k: float = 0.03

    def pass_probability(self, h: Harness, task: Task) -> tuple[float, str]:
        caps = capabilities(h)
        needs: list[str] = task.meta.get("needs", [])
        base: float = task.meta.get("base", 0.25)
        met = [n for n in needs if n in caps]
        p = base + (self.ceiling - base) * (len(met) / len(needs) if needs else 1.0)
        reason = ""
        missing = [n for n in needs if n not in caps and n in CATALOG]
        if missing:
            reason = CATALOG[missing[0]][0]
        if f"task:{task.id}" in caps:
            p += 0.5  # memorized this exact task: big in-loop gain, zero transfer
        for cap, (need, msg) in INTERFERENCE.items():
            if cap in caps and need in task.meta.get("traits", []):
                p -= 0.45
                reason = msg
        over = max(0, h.size_chars() - self.bloat_free_chars)
        p -= self.bloat_penalty_per_1k * over / 1000
        return max(0.02, min(0.98, p)), reason or "FAIL: flaky failure with no clear cause"

    def run(self, harness: Harness, task: Task, seed: int) -> Trial:
        p, reason = self.pass_probability(harness, task)
        passed = _unit(harness.fingerprint(), task.id, seed) < p
        tokens = 800 + harness.size_chars() // 4
        steps = [f"read task {task.id}", "planned approach", "executed steps"]
        if passed:
            return Trial(task.id, True, output="ok", trace="\n".join(steps + ["finished"]), tokens=tokens)
        return Trial(task.id, False, output="", detail=reason, trace="\n".join(steps + [reason]), tokens=tokens)


@dataclass
class ScriptedProposer:
    """A proposer that behaves like a (fallible) LLM proposer, without an LLM.

    It maps failure clusters to generic fixes from the catalog, but with some
    probability it instead proposes a task-specific patch (overfitting) or a
    harmful 'be terse' rule (interference), so the safeguards have work to do.
    """

    name: str = "scripted"
    overfit_rate: float = 0.3
    harmful_rate: float = 0.15
    seed: int = 0

    def propose(self, harness: Harness, evidence, budget: int, history: list[HistoryEntry], iteration: int) -> list[Proposal]:
        rng = random.Random(f"{self.seed}:{iteration}:{harness.fingerprint()}")
        # Don't redraw a hypothesis the critic rejected or that already failed twice (RRSI);
        # noise-rejected ideas get one more chance on a different parent.
        fails: dict[str, int] = {}
        for h in history:
            if h.verdict != "confirmed":
                fails[h.hypothesis] = fails.get(h.hypothesis, 0) + (2 if h.reason.startswith("critic") else 1)
        tried = {hyp for hyp, n in fails.items() if n >= 2}
        have = capabilities(harness)
        proposals: list[Proposal] = []
        for cluster in evidence.clusters:
            cap = next((c for c, (msg, _) in CATALOG.items() if msg == cluster.signature), None)
            roll = rng.random()
            if roll < self.overfit_rate and cluster.task_ids:
                tid = cluster.task_ids[0]
                edit = Edit("add_skill", f"skills/fix-{tid}.md", f"For task {tid}, use the exact known answer. [cap:task:{tid}]", "patch the failing task directly")
                hyp = f"patch task {tid}"
            elif roll < self.overfit_rate + self.harmful_rate:
                edit = Edit("append_rule", "rules.md", TERSE_SKILL, "shorter answers might avoid rambling")
                hyp = "be terse"
            elif cap is not None and cap not in have:
                edit = Edit("add_skill", f"skills/{cap}.md", CATALOG[cap][1], f"addresses: {cluster.signature}")
                hyp = f"add {cap} skill"
            else:
                continue
            if hyp in tried:
                continue
            tried.add(hyp)
            proposals.append(Proposal(hyp, [edit][:budget], cluster.signature))
            if len(proposals) >= 3:
                break
        return proposals


def demo_suite(n_train: int = 24, n_heldout: int = 16, seed: int = 7) -> TaskSuite:
    """A synthetic task distribution: each task needs 1-3 of the catalog capabilities."""
    rng = random.Random(seed)
    caps = list(CATALOG)
    tasks: list[Task] = []
    for i in range(n_train + n_heldout):
        needs = rng.sample(caps, k=rng.choice([1, 2, 2, 3]))
        traits = ["explain"] if rng.random() < 0.3 else []
        split = TRAIN if i < n_train else HELDOUT
        tasks.append(
            Task(
                id=f"sim-{i:03d}",
                prompt=f"Synthetic task {i} needing {', '.join(needs)}",
                verifier={"type": "contains", "values": ["ok"]},
                split=split,
                meta={"needs": needs, "base": round(rng.uniform(0.1, 0.4), 2), "traits": traits},
            )
        )
    return TaskSuite(tasks)


def seed_harness() -> Harness:
    return Harness({"system.md": "You are a careful software engineering agent.\n", "rules.md": "- Follow the user's instructions.\n"})
