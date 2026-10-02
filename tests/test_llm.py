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
