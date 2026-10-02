"""The archive: a tree of every harness variant ever scored (DarwinX §2.2-2.3).

Nothing is thrown away. A variant that loses overall may hold the one edit that,
merged with another lineage's, unlocks a task neither solves alone.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .harness import Edit, Harness

# node status
ROOT = "root"
REVERTED = "reverted"  # failed screening, critic, or confirmation
PROMOTED = "promoted"  # passed screening: may be an ancestor, may not steer yet
CONFIRMED = "confirmed"  # passed re-test + preservation probe: may steer search


@dataclass
class Node:
    id: str
    parent_id: str | None
    edits: list[Edit]  # this node's own delta
    lineage: list[Edit]  # all edits from the root (what builds this harness)
    fingerprint: str
    status: str
    hypothesis: str = ""
    reason: str = ""
    klass: str = ""
    gain: float = 0.0  # g vs parent, task units
    regression: float = 0.0  # R vs parent
    lineage_gain: float = 0.0  # G = G(parent) + g
    train_rates: dict[str, float] = field(default_factory=dict)
    heldout_mean: float | None = None  # mean partial-credit score on held-out tasks
    heldout_pass: float | None = None  # strict full-pass rate on held-out tasks
    tokens_per_rollout: float = 0.0
    iteration: int = 0
    merged_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edits"] = [e.to_dict() for e in self.edits]
        d["lineage"] = [e.to_dict() for e in self.lineage]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        d = dict(d)
        d["edits"] = [Edit.from_dict(e) for e in d["edits"]]
        d["lineage"] = [Edit.from_dict(e) for e in d["lineage"]]
        return cls(**d)

    def solved(self, threshold: float = 0.5) -> set[str]:
        return {t for t, r in self.train_rates.items() if r >= threshold}


class Archive:
    def __init__(self, root_harness: Harness):
        self.root_harness = root_harness
        self.nodes: dict[str, Node] = {}
        self._counter = 0

    # ---- construction ---------------------------------------------------
    def new_id(self) -> str:
        self._counter += 1
        return f"n{self._counter:04d}"

    def add_root(self, train_rates: dict[str, float], tokens: float) -> Node:
        node = Node("n0000", None, [], [], self.root_harness.fingerprint(), ROOT, hypothesis="seed harness", train_rates=train_rates, tokens_per_rollout=tokens)
        self.nodes[node.id] = node
        return node

    def add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def harness(self, node: Node) -> Harness:
        return self.root_harness.apply(node.lineage)

    @property
    def root(self) -> Node:
        return self.nodes["n0000"]

    # ---- queries --------------------------------------------------------
    def steering_set(self) -> list[Node]:
        """Variants allowed to steer search: the root and every confirmed node."""
        return [n for n in self.nodes.values() if n.status in (ROOT, CONFIRMED)]

    def best(self) -> Node:
        return max(self.steering_set(), key=lambda n: (n.lineage_gain, -len(n.lineage), n.id))

    def select_parent(self, rng: random.Random, beta: float = 0.3) -> Node:
        """p* ~ (1-β)·δ[argmax_S G] + β·Broaden(P)  (DarwinX §2.2)."""
        steering = self.steering_set()
        if rng.random() >= beta or len(self.nodes) == 1:
            return max(steering, key=lambda n: (n.lineage_gain, n.id))
        pool = [n for n in self.nodes.values() if n.status in (ROOT, CONFIRMED, PROMOTED)]
        return rng.choice(pool)

    def specialists(self, threshold: float = 0.5) -> list[Node]:
        """Confirmed variants that solve at least one task no other confirmed variant solves."""
        confirmed = [n for n in self.nodes.values() if n.status == CONFIRMED]
        out = []
        for n in confirmed:
            others = set().union(*(m.solved(threshold) for m in confirmed if m.id != n.id)) if len(confirmed) > 1 else set()
            if n.solved(threshold) - others - self.root.solved(threshold):
                out.append(n)
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {"root_harness": self.root_harness.components, "nodes": [n.to_dict() for n in self.nodes.values()]}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Archive":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        arc = cls(Harness(data["root_harness"]))
        for d in data["nodes"]:
            arc.add(Node.from_dict(d))
        arc._counter = max((int(n[1:]) for n in arc.nodes), default=0)
        return arc
