"""Human-readable run report: what changed, why, and whether it is proven on held-out tasks."""

from __future__ import annotations

from collections import Counter

from .archive import CONFIRMED, PROMOTED, REVERTED, Archive, Node
from .evolve import RunResult


def lineage(archive: Archive, node: Node) -> list[Node]:
    chain = []
    cur: Node | None = node
    while cur is not None:
        chain.append(cur)
        cur = archive.nodes.get(cur.parent_id) if cur.parent_id else None
    return list(reversed(chain))


def _mean(rates: dict[str, float]) -> float:
    return sum(rates.values()) / len(rates) if rates else 0.0


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render(result: RunResult, title: str = "Lamarck run report") -> str:
    arc, best, root = result.archive, result.best, result.archive.root
    nodes = [n for n in arc.nodes.values() if n.id != root.id]
    status = Counter(n.status for n in nodes)
    rejected = Counter(_reason_kind(n) for n in nodes if n.status == REVERTED)

    lines = [f"# {title}", ""]
    verdict = "SHIP" if result.decision.accept else ("INCONCLUSIVE" if result.decision.reason.startswith("inconclusive") else "DO NOT SHIP")
    lines += [f"**Verdict: {verdict}** - {result.decision.reason}", ""]
    lines += [
        "| | Seed harness | Best variant |",
        "|---|---|---|",
        f"| Train score (seen by the loop) | {_pct(_mean(root.train_rates))} | {_pct(_mean(best.train_rates))} |",
        f"| Held-out score (never seen) | {_pct(root.heldout_mean)} | {_pct(best.heldout_mean)} |",
        f"| Held-out full-pass rate | {_pct(root.heldout_pass)} | {_pct(best.heldout_pass)} |",
        f"| Harness size | {len(arc.root_harness.render())} chars | {len(arc.harness(best).render())} chars |",
        "",
    ]
    if root.heldout_mean is not None and best.heldout_mean is not None:
        train_gain = _mean(best.train_rates) - _mean(root.train_rates)
        held_gain = best.heldout_mean - root.heldout_mean
        lines += [
            f"In-loop gain {train_gain * 100:+.1f} pts vs held-out gain {held_gain * 100:+.1f} pts "
            f"(gap {(train_gain - held_gain) * 100:+.1f} pts). A large positive gap means the loop fit its own tasks; "
            "only the held-out number is evidence.",
            "",
        ]

    lines += ["## Accepted changes (best lineage)", ""]
    chain = lineage(arc, best)
    if len(chain) == 1:
        lines.append("_None: the seed harness was not beaten._")
    for n in chain[1:]:
        lines.append(f"### {n.id}: {n.hypothesis}")
        lines.append(f"g = {n.gain:+.2f} tasks, regression R = {n.regression:.2f}, class: {n.klass or '-'}. {n.reason}")
        for e in n.edits:
            lines.append(f"- `{e.op}` -> `{e.target}`: {e.rationale}".rstrip(": "))
            lines.append("")
            lines.append("  > " + e.content.strip().replace("\n", "\n  > "))
        lines.append("")

    lines += [
        "## Search summary",
        "",
        f"- Variants considered: {len(nodes)} (confirmed {status.get(CONFIRMED, 0)}, promoted-but-unconfirmed {status.get(PROMOTED, 0)}, rejected {status.get(REVERTED, 0)})",
    ]
    for kind, cnt in rejected.most_common():
        lines.append(f"  - rejected by {kind}: {cnt}")
    lines += [
        f"- Rollouts: {result.total_rollouts}; tokens (agent + verifier side): {result.total_tokens:,}",
        "",
        "## Harness diff (seed -> best)",
        "",
        "```diff",
        arc.root_harness.diff(arc.harness(best)).rstrip() or "(no change)",
        "```",
        "",
        "## Notes",
        "- Selection uses train tasks only. Held-out tasks were evaluated once, for the pre-registered best variant.",
        "- Gains inside the noise band are not counted (avg@k rollouts, bootstrap noise estimate).",
        "- Scores give partial credit (fraction of a task's checks passed); the full-pass rate is the strict number.",
    ]
    return "\n".join(lines) + "\n"


def _reason_kind(n: Node) -> str:
    r = n.reason.lower()
    if r.startswith("critic"):
        return "critic (leakage / size / budget / surface)"
    if r.startswith("screen"):
        return "screening (no gain, regression, or cost)"
    if "merge" in n.hypothesis:
        return "merge acceptance"
    if r.startswith("no-op"):
        return "no-op"
    return "other"
