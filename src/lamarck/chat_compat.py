"""Client for OpenAI-compatible chat-completions providers (Groq today).

Stdlib only, so the core stays dependency-free. It handles the parts that matter for
long evaluation runs on free tiers:
  * per-minute 429s: wait for `retry-after`, then retry,
  * per-day 429s: raise immediately (fatal: the run must stop, not loop),
  * 5xx: retry with backoff,
  * reasoning models: hidden reasoning, and any stray <think> blocks stripped from the answer.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "tokenharbor": ("https://tokenharbor.ai/v1", "TOKENHARBOR_API_KEY"),
}

# Offered to models that insist on native tool calls (Gemini): without it, the API filters their
# replies as MALFORMED_FUNCTION_CALL. A call is converted back into the agent's text protocol.
SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one shell command in the repository root and see its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def tool_call_as_text(message: dict) -> str:
    """Render a `bash` tool call as the agent's ```bash block protocol ('' if there is none)."""
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        cmd = args.get("command") if isinstance(args, dict) else None
        if cmd:
            return f"```bash\n{cmd}\n```"
    return ""


_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
# Groq: "tokens per day (TPD)"; Gemini: quota ids like "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
_DAILY = re.compile(r"per[- ]?day|\(RPD\)|\(TPD\)", flags=re.IGNORECASE)


class GatewayHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.retry_in = retry_hint(message)


# Groq: "Please try again in 2m51.9s"; Gemini: "Please retry in 41.5s".
_HINT = re.compile(r"(?:try again|retry) in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?", flags=re.IGNORECASE)


def retry_hint(text: str) -> float | None:
    """Seconds until a quota frees up, from messages like 'Please try again in 2m51.9s'.

    Groq's per-day limits are a rolling 24-hour window, so this is often minutes, not hours."""
    m = _HINT.search(text or "")
    if not m or not any(m.groups()):
        return None
    h, mins, secs = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(secs or 0)


@dataclass
class ChatClient:
    provider: str
    base_url: str
    api_key: str
    max_retries: int = 8
    quota_wait: float = 0.0  # seconds this client may spend waiting out per-day limits before giving up
    network_wait: float = 600.0  # seconds to keep retrying through a network outage before giving up
    timeout: float = 180.0

    @classmethod
    def for_provider(cls, provider: str) -> "ChatClient":
        from .llm import LLMError

        base, env_var = PROVIDERS[provider]
        key = os.environ.get(env_var, "")
        if not key:
            raise LLMError(f"{env_var} is not set; put it in the .env file next to lamarck.json (never in config or chat)")
        return cls(provider, base, key)

    _net_tries: int = 0

    def chat(self, *, model: str, system: str, user: str, max_tokens: int, effort: str | None, json_mode: bool, shell_tool: bool = False) -> dict:
        body: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if shell_tool:
            body["tools"] = [SHELL_TOOL]
            body["tool_choice"] = "auto"
        if "gpt-oss" in model:
            body["reasoning_effort"] = effort or "low"
            body["include_reasoning"] = False
        elif "qwen3" in model and self.provider == "groq":
            body["reasoning_format"] = "hidden"
        elif self.provider == "gemini" and effort:
            body["reasoning_effort"] = effort
        return self._post("/chat/completions", body)

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        attempt = -1
        while attempt < self.max_retries:
            attempt += 1
            req = urllib.request.Request(
                self.base_url + path,
                data=data,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "lamarck/0.1"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                msg = exc.read().decode("utf-8", "replace")[:500]
                if exc.code == 429 and not _DAILY.search(msg) and attempt < self.max_retries:
                    time.sleep(_retry_after(exc, attempt))
                    continue
                if exc.code >= 500 and attempt < self.max_retries:
                    time.sleep(min(60, 2**attempt))
                    continue
                if exc.code == 429:
                    wait = retry_hint(msg)
                    if wait is not None and wait + 5 <= self.quota_wait:
                        # rolling per-day window: tokens free up as old usage ages out; wait instead of failing
                        self.quota_wait -= wait + 5
                        time.sleep(wait + 5)
                        attempt -= 1  # waiting out the quota doesn't use up a retry
                        continue
                    raise GatewayHTTPError(429, "daily quota (per-day limit) reached: " + msg) from None
                raise GatewayHTTPError(exc.code, msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                # Network outage (seen in a real run: DNS failures). Keep retrying for a while without
                # using up normal retries; if it persists, raise status 0, which callers treat as
                # fatal ("pause the run"), never as the agent failing the task.
                pause = min(60.0, 5.0 * 2 ** min(self._net_tries, 4))
                if pause <= self.network_wait:
                    self.network_wait -= pause
                    self._net_tries += 1
                    time.sleep(pause)
                    attempt -= 1
                    continue
                raise GatewayHTTPError(0, f"network error: {exc}") from None
        raise GatewayHTTPError(0, "retries exhausted")


def _retry_after(exc: urllib.error.HTTPError, attempt: int) -> float:
    try:
        return min(90.0, float(exc.headers.get("retry-after", "")) + 0.5)
    except (TypeError, ValueError):
        return min(60.0, 2.0**attempt)


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text or "").strip()
