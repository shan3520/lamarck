"""The critic: screen proposals *before* paying to evaluate them.

RRSI's critic rejects benchmark-specific logic; "Bad Genius" (CHASE) shows a
proposer will happily find shortcuts that only work on the evaluation set.
This critic is deliberately cheap and deterministic:

1. leakage  - an edit that mentions a task id or a literal that identifies a
              single task is memorizing that task, not learning a behavior;
2. size     - one edit may not add more than `max_edit_chars` (bloat costs tokens
              on every future rollout);
3. budget   - a proposal may not bundle more edits than the annealed budget;
4. surface  - edits may only touch allowed components (permissions live outside
              the loop — Lilian Weng's caution about self-editing harnesses).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

from .harness import Edit


@dataclass
class Critic:
    denylist: set[str] = field(default_factory=set)
    max_edit_chars: int = 1200
    allowed_targets: tuple[str, ...] = ("system.md", "rules.md", "skills/*.md")

    def review(self, edits: list[Edit], budget: int) -> str | None:
        """Return a rejection reason, or None if the proposal may be evaluated."""
        if not edits:
            return "empty proposal"
        if len(edits) > budget:
            return f"bundles {len(edits)} edits; budget is {budget}"
        for e in edits:
            if not any(fnmatch.fnmatch(e.target, pat) for pat in self.allowed_targets):
                return f"edit touches non-editable surface {e.target!r}"
            if len(e.content) > self.max_edit_chars:
                return f"edit to {e.target} is {len(e.content)} chars (max {self.max_edit_chars})"
            text = f"{e.target}\n{e.content}".lower()
            for lit in self.denylist:
                if lit.lower() in text:
                    return f"task-specific leakage: edit mentions {lit!r}"
        return None


def edit_budget(iteration: int, total_iterations: int, start: int = 3, end: int = 1) -> int:
    """RRSI's temporally annealed budget: early candidates may bundle a few edits,
    late candidates only one, so late search makes small, attributable changes."""
    if total_iterations <= 1:
        return start
    frac = iteration / (total_iterations - 1)
    return max(end, round(start - (start - end) * frac))
