"""A/B test of two versions of an agent's instructions (e.g. CLAUDE.md before and after a change).

Teams change their agent's instruction file by feel. Lamarck's own experiments show why that
is risky: plausible, generic edits often *hurt* on tasks they were not tuned on (real-repo
run 7: +12.5 pts in-loop, -5.0 held-out; run 8: +5.0 held-out on 4 rollouts per task, then
-7.5 on 12). `compare` gives a change the same treatment: both versions run on the same
verified tasks, and a difference counts only if it clears the bootstrap noise band.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .evaluate import EvalResult, TrialCache, evaluate, noise_band
from .harness import Harness
from .targets.base import Target
from .tasks import Task

COMPARE_SEED = 30_000  # disjoint from run (0.., 10_000..) and calibration (20_000..) seeds
BETTER, WORSE, UNCLEAR, INCONCLUSIVE = "BETTER", "WORSE", "NO CLEAR DIFFERENCE", "INCONCLUSIVE"
EXIT_CODES = {BETTER: 0, UNCLEAR: 2, INCONCLUSIVE: 2, WORSE: 3}


@dataclass
class TaskRow:
    task_id: str
    base_passes: int
    cand_passes: int
    k: int


@dataclass
class CompareResult:
    verdict: str
    reason: str
    tasks: int
    k: int
    base_mean: float
    cand_mean: float
    base_pass: float
    cand_pass: float
    delta_pts: float
    noise_pts: float  # also the smallest difference this test could have detected
    base_tokens: float
    cand_tokens: float
    infra_errors: int
    rows: list[TaskRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(res: EvalResult) -> float:
    n = sum(len(v) for v in res.trials.values())
    return res.tokens / n if n else 0.0


def decide(base_mean: float, cand_mean: float, noise: float, n_rollouts: int, min_rollouts: int = 20) -> tuple[str, str]:
    delta = cand_mean - base_mean
    if n_rollouts < min_rollouts:
        return INCONCLUSIVE, f"only {n_rollouts} rollouts per version (need >= {min_rollouts})"
    if delta > noise:
        return BETTER, f"{delta * 100:+.1f} pts clears the noise band of {noise * 100:.1f} pts"
    if delta < -noise:
        return WORSE, f"{delta * 100:+.1f} pts is below minus the noise band of {noise * 100:.1f} pts"
    return UNCLEAR, f"{delta * 100:+.1f} pts is inside the noise band of {noise * 100:.1f} pts"


def compare_harnesses(target: Target, base: Harness, cand: Harness, tasks: list[Task], k: int = 4, workers: int = 3,
                      cache: TrialCache | None = None, noise_mult: float = 1.0, min_rollouts: int = 20) -> CompareResult:
    """Same agent, two versions of its instructions."""
    return compare_variants(target, base, target, cand, tasks, k, workers, cache, cache, noise_mult, min_rollouts)


def compare_variants(base_target: Target, base: Harness, cand_target: Target, cand: Harness, tasks: list[Task], k: int = 4,
                     workers: int = 3, base_cache: TrialCache | None = None, cand_cache: TrialCache | None = None,
                     noise_mult: float = 1.0, min_rollouts: int = 20) -> CompareResult:
    """Two agent variants: they may differ in instructions, agent settings (steps, memory, timeout) or model.

    Saved attempts are keyed by the instructions' fingerprint only, so variants that share instructions
    but differ in settings MUST use separate caches, or the candidate would replay the base's results."""
    if base_cache is not None and base_cache is cand_cache and base_target is not cand_target and base.fingerprint() == cand.fingerprint():
        raise ValueError("variants with the same instructions but different agents need separate caches")
    b = evaluate(base_target, base, tasks, k, seed=COMPARE_SEED, workers=workers, cache=base_cache)
    c = evaluate(cand_target, cand, tasks, k, seed=COMPARE_SEED, workers=workers, cache=cand_cache)
    noise = noise_mult * math.sqrt(noise_band(b) ** 2 + noise_band(c) ** 2)
    verdict, reason = decide(b.mean, c.mean, noise, len(tasks) * k, min_rollouts)
    rows = [TaskRow(t.id, sum(x.passed and x.error is None for x in b.trials[t.id]), sum(x.passed and x.error is None for x in c.trials[t.id]), k)
            for t in tasks]
    return CompareResult(verdict, reason, len(tasks), k, b.mean, c.mean, b.pass_rate, c.pass_rate, (c.mean - b.mean) * 100,
                         noise * 100, _tokens(b), _tokens(c), b.infra_errors + c.infra_errors, rows)


def render(r: CompareResult, base_name: str, cand_name: str, settings: dict | None = None) -> str:
    """`settings`: {"what changed": [(label, base value, candidate value), ...]} for the report's header table."""
    tok = f"{(r.cand_tokens / r.base_tokens - 1) * 100:+.0f}%" if r.base_tokens else "n/a"
    changed = [label for label, b, c in (settings or {}).get("rows", []) if b != c]
    what = " and ".join(changed) if changed else "instructions"
    lines = [
        f"# Agent change test ({what}): {r.verdict}",
        "",
        f"**{cand_name}** vs **{base_name}** on {r.tasks} verified tasks, {r.k} attempts each per version.",
        "",
    ]
    if settings and settings.get("rows"):
        lines += ["| Setting | Base | Candidate |", "|---|---|---|"]
        lines += [f"| {label}{' (changed)' if b != c else ''} | `{b}` | `{c}` |" for label, b, c in settings["rows"]]
        lines.append("")
    lines += [
        "| | Base | Candidate |",
        "|---|---|---|",
        f"| Mean score | {r.base_mean * 100:.1f}% | {r.cand_mean * 100:.1f}% |",
        f"| Full-pass rate | {r.base_pass * 100:.1f}% | {r.cand_pass * 100:.1f}% |",
        f"| Tokens per attempt | {r.base_tokens:,.0f} | {r.cand_tokens:,.0f} ({tok}) |",
        "",
        f"**Verdict: {r.verdict}.** {r.reason[0].upper() + r.reason[1:]}.",
        f"With this many tasks and attempts, differences smaller than about **{r.noise_pts:.1f} pts** can't be told apart from luck.",
    ]
    if r.infra_errors:
        lines.append(f"{r.infra_errors} attempts failed for technical reasons and count as failures.")
    lines += ["", "## Per task", "", "| Task | Base | Candidate |", "|---|---|---|"]
    lines += [f"| {row.task_id} | {row.base_passes}/{row.k} | {row.cand_passes}/{row.k} |" for row in r.rows]
    return "\n".join(lines) + "\n"


def load_target_spec(path) -> dict:
    """An agent variant: a JSON file holding a target spec, or a whole lamarck.json (its "target" is used)."""
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = data.get("target", data) if isinstance(data, dict) else None
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"{path}: expected a target spec with a 'type' (or a lamarck.json with a 'target')")
    return spec


def settings_rows(base_label: str, base: Harness, cand_label: str, cand: Harness, base_spec: dict, cand_spec: dict) -> dict:
    b = f"{base_label} ({base.fingerprint()})"
    c = b if cand.fingerprint() == base.fingerprint() else f"{cand_label} ({cand.fingerprint()})"  # same content = unchanged
    rows = [("instructions", b, c)]
    for key in sorted(set(base_spec) | set(cand_spec)):
        rows.append((f"agent {key}", str(base_spec.get(key, "")), str(cand_spec.get(key, ""))))
    return {"rows": rows}


def load_instructions(path) -> Harness:
    """A harness directory (system.md, rules.md, skills/) or a single instruction file such as CLAUDE.md."""
    from pathlib import Path

    from .harness import SYSTEM

    p = Path(path)
    if p.is_dir():
        return Harness.load(p)
    return Harness({SYSTEM: p.read_text(encoding="utf-8")})
