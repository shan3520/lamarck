"""Selection: when does a harness change earn its place?

Implements, clean-room from the papers:

* DarwinX preserve-and-extend contract (§2.1):
    Δ_t = p̂_t(child) − p̂_t(parent);  g = ΣΔ_t;  R = Σ(−Δ_t)₊
    admit iff g > 0 and R <= δ, and only a *re-tested* child may steer search.
* DarwinX solved-set classification (§2.3): improver / neutral / stepping stone / archived.
* RRSI selection rules: a noise-adjusted floor (gains inside evaluation noise
  don't count) and a cost rule (added inference tokens must be paid for by gain).
* A final held-out gate (the overfitting defence every paper converges on).
"""

from __future__ import annotations

from dataclasses import dataclass

IMPROVER = "improver"
NEUTRAL = "neutral"
STEPPING_STONE = "stepping_stone"
ARCHIVED = "archived"


@dataclass
class GateConfig:
    max_regression: float = 1.0  # δ: total solve-rate loss tolerated on the preservation set (task units)
    preserve_threshold: float = 0.8  # tasks the parent solves at least this reliably form the preservation set
    noise_mult: float = 1.0  # gain must exceed noise_mult × noise band (mean-rate units)
    tokens_per_point: float = 4000.0  # cost rule: each +1 task-unit of gain may add this many tokens per rollout
    solved_threshold: float = 0.5


@dataclass
class Comparison:
    gain: float  # g, in task units (sum of per-task rate deltas)
    regression: float  # R
    deltas: dict[str, float]
    klass: str

    @property
    def mean_gain(self) -> float:
        return self.gain / len(self.deltas) if self.deltas else 0.0


@dataclass
class Decision:
    accept: bool
    reason: str


def compare(parent: dict[str, float], child: dict[str, float], solved_threshold: float = 0.5, preserve_threshold: float = 0.8) -> Comparison:
    common = sorted(set(parent) & set(child))
    deltas = {t: child[t] - parent[t] for t in common}
    gain = sum(deltas.values())
    sp = {t for t in common if parent[t] >= solved_threshold}
    sc = {t for t in common if child[t] >= solved_threshold}
    # Preservation is about what the parent *reliably* solved; dips on tasks it only
    # sometimes solved are mostly noise, and already count against g.
    keep = {t for t in common if parent[t] >= preserve_threshold}
    regression = sum(max(0.0, -deltas[t]) for t in keep)
    return Comparison(gain, regression, deltas, classify(sp, sc))


def classify(parent_solved: set[str], child_solved: set[str]) -> str:
    if child_solved == parent_solved:
        return NEUTRAL
    if child_solved > parent_solved:
        return IMPROVER
    if child_solved < parent_solved:
        return STEPPING_STONE
    return ARCHIVED


def screen(cmp: Comparison, noise: float, n_tasks: int, cfg: GateConfig, token_delta_per_rollout: float = 0.0) -> Decision:
    """Cheap first stage: permissive about *trying* (DarwinX's 'enabler')."""
    if cmp.gain <= 0:
        return Decision(False, f"no net gain (g={cmp.gain:+.2f})")
    if cmp.regression > cfg.max_regression:
        return Decision(False, f"regresses solved work (R={cmp.regression:.2f} > delta={cfg.max_regression})")
    return _cost_rule(cmp, token_delta_per_rollout, cfg) or Decision(True, f"g={cmp.gain:+.2f}, R={cmp.regression:.2f}")


def confirm(cmp: Comparison, noise: float, n_tasks: int, cfg: GateConfig, token_delta_per_rollout: float = 0.0) -> Decision:
    """Strict second stage on more rollouts: strict about *trusting*."""
    floor = cfg.noise_mult * noise * n_tasks  # noise band converted to task units
    if cmp.regression > cfg.max_regression:
        return Decision(False, f"regression on re-test (R={cmp.regression:.2f} > delta={cfg.max_regression})")
    if cmp.gain <= floor:
        return Decision(False, f"gain within noise (g={cmp.gain:+.2f} <= floor {floor:.2f})")
    return _cost_rule(cmp, token_delta_per_rollout, cfg) or Decision(True, f"confirmed g={cmp.gain:+.2f} > floor {floor:.2f}, R={cmp.regression:.2f}")


def _cost_rule(cmp: Comparison, token_delta: float, cfg: GateConfig) -> Decision | None:
    if token_delta > 0 and token_delta > cfg.tokens_per_point * max(cmp.gain, 0.0):
        return Decision(False, f"too expensive: +{token_delta:.0f} tokens/rollout for g={cmp.gain:+.2f}")
    return None


def heldout_gate(base_mean: float, cand_mean: float, noise: float, noise_mult: float = 1.0) -> Decision:
    """Ship only if the held-out improvement clears the held-out noise band."""
    delta = cand_mean - base_mean
    floor = noise_mult * noise
    if delta > floor:
        return Decision(True, f"held-out +{delta * 100:.1f} pts > noise {floor * 100:.1f} pts")
    return Decision(False, f"held-out {delta * 100:+.1f} pts does not clear noise {floor * 100:.1f} pts")
