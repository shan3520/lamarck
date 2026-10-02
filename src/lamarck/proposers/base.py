from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..harness import Edit, Harness
from ..miner import Evidence


@dataclass
class Proposal:
    hypothesis: str
    edits: list[Edit]
    cluster: str = ""  # failure signature this proposal targets

    def to_dict(self) -> dict:
        return {"hypothesis": self.hypothesis, "cluster": self.cluster, "edits": [e.to_dict() for e in self.edits]}


@dataclass
class HistoryEntry:
    """What the proposer is told about earlier attempts, so a falsified hypothesis is not redrawn (RRSI)."""

    hypothesis: str
    verdict: str
    reason: str = ""
    gain: float = 0.0
    extra: dict = field(default_factory=dict)


class Proposer(Protocol):
    name: str

    def propose(self, harness: Harness, evidence: Evidence, budget: int, history: list[HistoryEntry], iteration: int) -> list[Proposal]: ...
