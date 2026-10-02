from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..harness import Harness
from ..tasks import Task


@dataclass
class Trial:
    """One rollout of a harness on a task."""

    task_id: str
    passed: bool
    output: str = ""
    detail: str = ""  # verifier explanation (task-agnostic), e.g. "FAIL: JSON is missing required key 'id'"
    trace: str = ""  # whatever the agent did, for the weakness miner
    tokens: int = 0
    error: str | None = None  # infrastructure error (not the agent's fault)
    score: float | None = None  # partial credit in [0, 1]; defaults to 1.0 / 0.0

    def __post_init__(self) -> None:
        if self.score is None:
            self.score = 1.0 if self.passed else 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "output": self.output[-4000:],
            "detail": self.detail,
            "trace": self.trace[-4000:],
            "tokens": self.tokens,
            "error": self.error,
        }


class Target(Protocol):
    """Something that runs a frozen model under a harness on a task."""

    name: str

    def run(self, harness: Harness, task: Task, seed: int) -> Trial: ...


@dataclass
class TargetInfo:
    name: str
    model: str = ""
    notes: dict = field(default_factory=dict)
