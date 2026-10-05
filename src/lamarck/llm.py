"""Thin wrapper over the Anthropic SDK (optional dependency: ``pip install lamarck[llm]``).

Credentials resolve the SDK's usual way (ANTHROPIC_API_KEY, or an ``ant auth login``
profile). Never pass keys on the command line or commit them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .env import model_id, provider

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class RefusalError(LLMError):
    pass


class EmptyResponseError(LLMError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RateLimiter:
    """Spaces requests evenly across all threads (free OpenRouter models allow 20/minute)."""

    def __init__(self, per_minute: float | None = None):
        self.per_minute = per_minute
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.per_minute:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + 60.0 / self.per_minute
        if slot > now:
            time.sleep(slot - now)


LIMITER = RateLimiter()


def set_rate_limit(per_minute: float | None) -> None:
    LIMITER.per_minute = per_minute


def is_anthropic_model(model: str) -> bool:
    """Claude models accept `output_config` (effort, JSON schema); other gateway models may not."""
    return "/" not in model or model.startswith("anthropic/")


def make_client(provider_name: str = "anthropic") -> Any:
    """`anthropic` (Anthropic SDK; also OpenRouter via ANTHROPIC_BASE_URL) or an OpenAI-compatible
    provider from chat_compat.PROVIDERS (e.g. `groq`)."""
    from .chat_compat import PROVIDERS, ChatClient

    if provider_name in PROVIDERS:
        return ChatClient.for_provider(provider_name)
    if provider_name != "anthropic":
        raise LLMError(f"unknown provider {provider_name!r}; expected 'anthropic' or one of {sorted(PROVIDERS)}")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LLMError("the Anthropic SDK is not installed; run: pip install 'lamarck[llm]'") from exc
    return anthropic.Anthropic(max_retries=5)  # rides out per-minute 429s with backoff


def complete(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 16000,
    effort: str | None = None,
    json_schema: dict | None = None,
    fallbacks: bool = False,
) -> Completion:
    """One Messages API call. Thinking is left at the model default (adaptive on current models).

    `fallbacks=True` lets the API re-run a refused request on another model. Use it for the
    *proposer* only: the *target* model must stay frozen, or measurements are meaningless.
    """
    from .chat_compat import ChatClient, strip_reasoning

    if isinstance(client, ChatClient):
        for _ in range(3):
            LIMITER.wait()
            data = client.chat(model=model, system=system, user=user, max_tokens=max_tokens, effort=effort, json_mode=bool(json_schema))
            choice = (data.get("choices") or [{}])[0]
            # Gemini sometimes attempts a native tool call although no tools are declared; the API then
            # filters the reply ("function_call_filter: MALFORMED_FUNCTION_CALL", ~1 in 3 replies in a
            # real run). That's a serving quirk, not the model's answer: ask again.
            if "function_call_filter" not in str(choice.get("finish_reason") or ""):
                break
        text = strip_reasoning((choice.get("message") or {}).get("content") or "")
        if not text:
            raise EmptyResponseError(f"{model} returned an empty response (finish_reason={choice.get('finish_reason')})")
        usage = data.get("usage") or {}
        return Completion(text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0), str(data.get("model", model)), str(choice.get("finish_reason")))

    output_config: dict = {}
    if effort:
        output_config["effort"] = effort
    if json_schema:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    kwargs: dict = {
        "model": model_id(model),
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if output_config and is_anthropic_model(model):
        kwargs["output_config"] = output_config  # non-Claude models: JSON is requested in the prompt instead
    LIMITER.wait()
    if fallbacks and provider() == "anthropic":  # server-side fallbacks are an Anthropic API feature
        resp = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    if not getattr(resp, "content", None):  # some gateway providers occasionally return an empty body
        raise EmptyResponseError(f"{model} returned an empty response (stop_reason={getattr(resp, 'stop_reason', None)})")
    if resp.stop_reason == "refusal":
        raise RefusalError(f"{model} declined the request")
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    usage = resp.usage
    return Completion(text, int(usage.input_tokens or 0), int(usage.output_tokens or 0), str(getattr(resp, "model", model)), str(resp.stop_reason))
