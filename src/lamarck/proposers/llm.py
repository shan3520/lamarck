"""A Claude-powered proposer: reads the harness, the mined evidence and the history
of earlier attempts, and proposes small, generic, attributable harness edits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..harness import ALL_OPS, Edit, Harness
from ..llm import DEFAULT_MODEL, EmptyResponseError, RefusalError, complete, make_client
from ..miner import Evidence
from .base import HistoryEntry, Proposal

SYSTEM = """You improve the *harness* of an AI agent whose model weights are frozen. The harness is a set of \
markdown components: `system.md` (core instructions), `rules.md` (a bullet list of rules) and `skills/<name>.md` \
(reusable procedures). The rendered harness is what the agent reads before every task.

You will see the current harness, clusters of recurring failures mined from real rollouts, and the history of \
edits already tried. Propose harness edits that fix the *recurring cause* of failures.

Hard requirements:
- Edits must be GENERIC and REUSABLE across many tasks. Never mention a specific task, ticket, id, file name, \
expected answer or detail that only applies to one example. Edits that memorize examples are rejected automatically \
and waste the budget.
- Prefer additive edits: `add_skill` (target `skills/<kebab-name>.md`) or `append_rule` (target `rules.md`, one rule). \
Use `set_component` only when an existing instruction is actively wrong.
- Keep each edit short (a few sentences). Every token is paid on every future task.
- Each proposal may contain at most {budget} edit(s). Make proposals independent so each can be tested alone.
- Do not repeat a hypothesis that the history shows was already rejected unless you change it materially.
- Procedural fixes usually generalize best: state the acceptance criteria, then verify the output against them \
before finishing.

Respond with only a JSON object: {{"proposals": [{{"hypothesis": str, "cluster": str, "edits": [{{"op": str, \
"target": str, "content": str, "rationale": str}}]}}]}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "cluster": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string", "enum": list(ALL_OPS)},
                                "target": {"type": "string"},
                                "content": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["op", "target", "content", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["hypothesis", "cluster", "edits"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


@dataclass
class LLMProposer:
    model: str = DEFAULT_MODEL
    effort: str | None = "high"
    n_proposals: int = 3
    client: Any = None
    provider: str = "anthropic"  # or "groq" (see chat_compat.PROVIDERS)
    name: str = "llm"
    tokens_used: int = 0

    def propose(self, harness: Harness, evidence: Evidence, budget: int, history: list[HistoryEntry], iteration: int) -> list[Proposal]:
        if self.client is None:
            self.client = make_client(self.provider)
        for attempt in range(2):  # one retry: free gateway models occasionally return nothing
            try:
                comp = complete(
                    self.client,
                    model=self.model,
                    system=SYSTEM.format(budget=budget),
                    user=build_prompt(harness, evidence, budget, history, self.n_proposals),
                    effort=self.effort,
                    json_schema=SCHEMA,
                    fallbacks=True,
                )
            except (EmptyResponseError, RefusalError):
                continue
            self.tokens_used += comp.tokens
            if isinstance(_load_json(comp.text), dict) and "proposals" in _load_json(comp.text):
                return parse_proposals(comp.text, budget)  # a valid answer, even if it proposes nothing
        return []


def build_prompt(harness: Harness, evidence: Evidence, budget: int, history: list[HistoryEntry], n: int) -> str:
    comps = "\n\n".join(f"<component name=\"{k}\">\n{v.strip()}\n</component>" for k, v in sorted(harness.components.items()))
    hist = "\n".join(f"- {h.hypothesis!r}: {h.verdict} ({h.reason})" for h in history[-25:]) or "(none yet)"
    return (
        f"## Current harness\n{comps}\n\n"
        f"## Evidence from rollouts\n{evidence.render()}\n\n"
        f"## Edits already tried\n{hist}\n\n"
        f"Propose up to {n} independent proposals, each with at most {budget} edit(s), targeting the largest failure clusters first."
    )


def _load_json(text: str):
    """Structured output gives clean JSON; gateways that don't enforce it may wrap it in prose or fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}


def parse_proposals(text: str, budget: int) -> list[Proposal]:
    data = _load_json(text)
    if not isinstance(data, dict):
        return []
    out: list[Proposal] = []
    for p in data.get("proposals", []):
        edits = []
        for e in p.get("edits", [])[:budget]:
            try:
                edits.append(Edit(e["op"], e["target"].strip(), e.get("content", ""), e.get("rationale", "")))
            except (KeyError, ValueError):
                continue
        if edits:
            out.append(Proposal(str(p.get("hypothesis", "")).strip() or "unnamed", edits, str(p.get("cluster", ""))))
    return out
