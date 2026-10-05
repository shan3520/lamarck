"""LLM-backed components against a fake Anthropic client (no network, no spend)."""

import json
from types import SimpleNamespace

from lamarck.harness import Harness
from lamarck.miner import Evidence
from lamarck.proposers.base import HistoryEntry
from lamarck.proposers.llm import LLMProposer, parse_proposals
from lamarck.targets.anthropic_api import AnthropicTarget
from lamarck.tasks import Task


def response(text, stop="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        model=model,
    )


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


class FakeClient:
    def __init__(self, replies):
        self.messages = FakeMessages(replies)
        self.beta = SimpleNamespace(messages=FakeMessages(replies))


TASK = Task("t-1", "Reply to ticket 12345", {"type": "contains", "values": ["The Acme Support Team"], "label": "replies must be signed"})


def test_target_passes_and_fails_with_verifier_detail():
    client = FakeClient([response("Thanks!\nThe Acme Support Team"), response("Thanks!")])
    target = AnthropicTarget(client=client, effort="low")
    h = Harness({"system.md": "Be nice."})
    ok, bad = target.run(h, TASK, 0), target.run(h, TASK, 1)
    assert ok.passed and ok.tokens == 120
    assert not bad.passed and bad.detail == "FAIL: replies must be signed"
    call = client.messages.calls[0]
    assert call["system"] == h.render() and call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert not client.beta.messages.calls, "the frozen target must never use model fallbacks"


def test_target_refusal_is_a_failed_trial():
    target = AnthropicTarget(client=FakeClient([response("", stop="refusal")]))
    trial = target.run(Harness({"system.md": "x"}), TASK, 0)
    assert not trial.passed and "declined" in trial.detail


def test_proposer_uses_structured_output_and_fallbacks(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    payload = {"proposals": [{"hypothesis": "sign replies", "cluster": "FAIL: replies must be signed", "edits": [
        {"op": "append_rule", "target": "rules.md", "content": "Sign every reply as 'The Acme Support Team'.", "rationale": "unsigned replies fail"}]}]}
    client = FakeClient([response(json.dumps(payload))])
    client.beta.messages.replies = [response(json.dumps(payload))]
    prop = LLMProposer(client=client)
    ev = Evidence([], ["t-1"], [], [], [], {"t-1": 0.5})
    out = prop.propose(Harness({"system.md": "x"}), ev, 1, [HistoryEntry("old idea", "rejected", "screen: no net gain")], 0)
    assert len(out) == 1 and out[0].edits[0].op == "append_rule"
    call = client.beta.messages.calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "old idea" in call["messages"][0]["content"]


def test_parse_proposals_drops_invalid_edits_and_enforces_budget():
    text = json.dumps({"proposals": [
        {"hypothesis": "a", "cluster": "", "edits": [
            {"op": "add_skill", "target": "skills/a.md", "content": "A", "rationale": ""},
            {"op": "add_skill", "target": "skills/b.md", "content": "B", "rationale": ""}]},
        {"hypothesis": "bad", "cluster": "", "edits": [{"op": "add_skill", "target": "rules.md", "content": "x", "rationale": ""}]},
    ]})
    out = parse_proposals(text, budget=1)
    assert [p.hypothesis for p in out] == ["a"] and len(out[0].edits) == 1
    assert parse_proposals("not json", 1) == []


def test_parse_proposals_tolerates_prose_and_fences():
    text = 'Here you go:\n```json\n{"proposals": [{"hypothesis": "h", "cluster": "", "edits": [{"op": "append_rule", "target": "rules.md", "content": "Sign replies.", "rationale": ""}]}]}\n```'
    assert [p.hypothesis for p in parse_proposals(text, 1)] == ["h"]


def test_openrouter_gateway(monkeypatch, tmp_path):
    from lamarck import env

    for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(k, "placeholder")  # registers the original value so teardown restores it
        monkeypatch.delenv(k)
    (tmp_path / ".env").write_text("# comment\nOPENROUTER_API_KEY=sk-or-test\n")
    env.load_dotenv(tmp_path / ".env")
    assert env.configure_gateway() == "openrouter"
    import os

    assert os.environ["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test" and os.environ["ANTHROPIC_API_KEY"] == ""
    assert env.model_id("claude-opus-5-5") == "anthropic/claude-opus-5.5"
    assert env.model_id("claude-haiku-4-5") == "anthropic/claude-haiku-4.5"
    assert env.model_id("anthropic/claude-sonnet-5.5") == "anthropic/claude-sonnet-5.5"
    # an inherited default Anthropic base URL + token must not override the user's OpenRouter key
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-token")
    assert env.configure_gateway() == "openrouter"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "sk-or-test"
    # fallbacks are Anthropic-only: through a gateway the proposer uses the plain endpoint
    payload = {"proposals": []}
    client = FakeClient([response(json.dumps(payload))])
    LLMProposer(client=client).propose(Harness({"system.md": "x"}), Evidence([], [], [], [], [], {}), 1, [], 0)
    assert client.messages.calls and not client.beta.messages.calls
    assert client.messages.calls[0]["model"] == "anthropic/claude-opus-5.5"


def test_empty_gateway_response_is_retried_then_skipped():
    empty = SimpleNamespace(content=None, stop_reason=None, usage=SimpleNamespace(input_tokens=0, output_tokens=0), model="m")
    client = FakeClient([empty, empty])
    assert LLMProposer(client=client).propose(Harness({"system.md": "x"}), Evidence([], [], [], [], [], {}), 1, [], 0) == []
    assert len(client.beta.messages.calls) == 2  # proposer calls go through the fallback-enabled endpoint
    trial = AnthropicTarget(client=FakeClient([empty])).run(Harness({"system.md": "x"}), TASK, 0)
    assert trial.error and not trial.passed


def test_groq_chat_client(monkeypatch):
    import io
    import urllib.error

    from lamarck import chat_compat
    from lamarck.evaluate import is_fatal
    from lamarck.llm import complete, make_client

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = make_client("groq")
    sent = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    replies = [
        urllib.error.HTTPError("u", 429, "rate", {"retry-after": "0"}, io.BytesIO(b"Rate limit reached on tokens per minute (TPM)")),
        Resp(json.dumps({"model": "llama-3.1-8b-instant", "choices": [{"message": {"content": "<think>hmm</think>Hello"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 3}}).encode()),
    ]

    def fake_urlopen(req, timeout):
        sent.append(json.loads(req.data))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(chat_compat.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(chat_compat.time, "sleep", lambda s: None)
    c = complete(client, model="llama-3.1-8b-instant", system="s", user="u", max_tokens=100, json_schema={"type": "object"})
    assert c.text == "Hello" and c.tokens == 13
    assert len(sent) == 2 and sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0]["messages"][0] == {"role": "system", "content": "s"}

    def daily(req, timeout):
        raise urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(b"Rate limit reached on requests per day (RPD): Limit 1000"))

    monkeypatch.setattr(chat_compat.urllib.request, "urlopen", daily)
    try:
        complete(client, model="llama-3.1-8b-instant", system="s", user="u", max_tokens=100)
        raise AssertionError("expected an error")
    except chat_compat.GatewayHTTPError as exc:
        assert exc.status_code == 429 and is_fatal(exc)


def test_groq_requires_key(monkeypatch):
    import pytest

    from lamarck.llm import LLMError, make_client

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        make_client("groq")


def test_retry_hint_parsing():
    from lamarck.chat_compat import retry_hint

    assert abs(retry_hint("Please try again in 2m51.936s. Need more tokens?") - 171.936) < 1e-6
    assert retry_hint("try again in 1h2m3s") == 3723
    assert retry_hint("try again in 12.5s") == 12.5
    assert retry_hint("try again in 45ms") is None or retry_hint("try again in 45ms") < 1
    assert retry_hint("no hint here") is None


def test_groq_client_waits_out_short_daily_quota(monkeypatch):
    import io
    import urllib.error

    from lamarck import chat_compat
    from lamarck.llm import complete, make_client

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = make_client("groq")
    client.quota_wait = 600
    slept = []
    monkeypatch.setattr(chat_compat.time, "sleep", lambda s: slept.append(s))

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    daily = b"Rate limit reached on tokens per day (TPD): Limit 200000, Used 199579. Please try again in 2m51.9s."
    replies = [urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(daily)),
               Resp(json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}).encode())]

    def fake(req, timeout):
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(chat_compat.urllib.request, "urlopen", fake)
    assert complete(client, model="m", system="s", user="u", max_tokens=10).text == "ok"
    assert slept and 170 < slept[0] < 190 and client.quota_wait < 600

    client.quota_wait = 10  # too little budget left: give up with a fatal error that carries the hint
    monkeypatch.setattr(chat_compat.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 429, "rate", {}, io.BytesIO(daily))))
    try:
        complete(client, model="m", system="s", user="u", max_tokens=10)
        raise AssertionError("expected an error")
    except chat_compat.GatewayHTTPError as exc:
        assert exc.status_code == 429 and 170 < exc.retry_in < 175


def test_gemini_quota_messages():
    from lamarck.chat_compat import _DAILY, PROVIDERS, retry_hint

    assert "gemini" in PROVIDERS and "nvidia" in PROVIDERS
    msg = "Quota exceeded for metric: generate_content_free_tier_requests, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 41.5s."
    assert _DAILY.search(msg) and abs(retry_hint(msg) - 41.5) < 1e-9
    assert not _DAILY.search("quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier")


def test_gemini_filtered_tool_call_is_retried(monkeypatch):
    from lamarck import chat_compat
    from lamarck.llm import complete

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = chat_compat.ChatClient.for_provider("gemini")
    replies = [
        {"choices": [{"message": {"content": "Let me look"}, "finish_reason": "function_call_filter: MALFORMED_FUNCTION_CALL"}], "usage": {}},
        {"choices": [{"message": {"content": "```bash\nls\n```"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    ]
    monkeypatch.setattr(client, "chat", lambda **kw: replies.pop(0))
    c = complete(client, model="gemini-3.5-flash-lite", system="s", user="u", max_tokens=10)
    assert c.text.startswith("```bash") and c.tokens == 8 and not replies


def test_gemini_shell_tool_call_becomes_bash_block(monkeypatch):
    from lamarck import chat_compat
    from lamarck.llm import complete

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = chat_compat.ChatClient.for_provider("gemini")
    seen = {}

    def fake_chat(**kw):
        seen.update(kw)
        return {"choices": [{"message": {"content": None, "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "pytest -q tests/x.py"})}}]},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    monkeypatch.setattr(client, "chat", fake_chat)
    c = complete(client, model="gemini-3.5-flash-lite", system="s", user="u", max_tokens=10, shell_tool=True)
    assert seen["shell_tool"] is True
    assert c.text == "```bash\npytest -q tests/x.py\n```"


def test_shell_tool_is_sent_in_request_body(monkeypatch):
    from lamarck import chat_compat

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = chat_compat.ChatClient.for_provider("gemini")
    bodies = []
    monkeypatch.setattr(client, "_post", lambda path, body: bodies.append(body) or {})
    client.chat(model="m", system="s", user="u", max_tokens=5, effort=None, json_mode=False, shell_tool=True)
    client.chat(model="m", system="s", user="u", max_tokens=5, effort=None, json_mode=False)
    assert bodies[0]["tools"][0]["function"]["name"] == "bash" and "tools" not in bodies[1]
