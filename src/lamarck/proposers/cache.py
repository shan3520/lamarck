"""Record proposer answers so a resumed run replays the same search.

LLM proposers are not deterministic. On resume, the n-th proposal request with the same
inputs must get the same answer the original run got, or the replayed search would diverge
from what was already evaluated (and paid for)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..harness import Edit, Harness
from ..miner import Evidence
from .base import HistoryEntry, Proposal


class CachingProposer:
    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.name = getattr(inner, "name", "proposer")
        self.path = Path(path)
        self._data: dict[str, list[dict]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["proposals"]

    @staticmethod
    def key(harness: Harness, evidence: Evidence, budget: int, history: list[HistoryEntry], iteration: int) -> str:
        blob = json.dumps([iteration, harness.fingerprint(), evidence.render(), budget, [(h.hypothesis, h.verdict) for h in history]])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def propose(self, harness: Harness, evidence: Evidence, budget: int, history: list[HistoryEntry], iteration: int) -> list[Proposal]:
        k = self.key(harness, evidence, budget, history, iteration)
        if k in self._data:
            return [Proposal(p["hypothesis"], [Edit.from_dict(e) for e in p["edits"]], p.get("cluster", "")) for p in self._data[k]]
        out = self.inner.propose(harness, evidence, budget, history, iteration)
        self._data[k] = [p.to_dict() for p in out]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "proposals": self._data[k]}, ensure_ascii=False) + "\n")
        return out
