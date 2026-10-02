"""Run a frozen Claude model under a harness: the harness becomes the system prompt.

This is the cheapest real target: good for instruction-following / convention
tasks (support replies, extraction, triage) where the 'agent' is one call.
For coding agents that edit files, use CommandTarget.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..harness import Harness
from ..llm import DEFAULT_MODEL, RefusalError, complete, make_client
from ..tasks import Task, verify
from .base import Trial


@dataclass
class AnthropicTarget:
    model: str = DEFAULT_MODEL
    effort: str | None = "low"
    max_tokens: int = 4000
    client: Any = None
    name: str = "anthropic"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _client(self) -> Any:
        with self._lock:
            if self.client is None:
                self.client = make_client()
            return self.client

    def run(self, harness: Harness, task: Task, seed: int) -> Trial:
        # `seed` cannot pin API sampling; repeated rollouts are genuinely independent samples.
        try:
            c = complete(self._client(), model=self.model, system=harness.render(), user=task.prompt, max_tokens=self.max_tokens, effort=self.effort)
        except RefusalError:
            return Trial(task.id, False, detail="FAIL: the model declined the request", trace="(refusal)")
        v = verify(task, c.text)
        trace = f"OUTPUT:\n{c.text[-2500:]}\n\nVERIFIER: {v.detail or 'PASS'}"
        return Trial(task.id, v.passed, output=c.text, detail=v.detail, trace=trace, tokens=c.tokens)
