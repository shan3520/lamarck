import pytest

from lamarck.evaluate import FatalRunError, evaluate
from lamarck.harness import Harness
from lamarck.tasks import Task


class Billing(Exception):
    status_code = 402


class Flaky(Exception):
    status_code = 529


class Target:
    name = "t"

    def __init__(self, exc):
        self.exc = exc

    def run(self, harness, task, seed):
        raise self.exc


TASKS = [Task("a", "p", {"type": "exact", "value": "x"})]


def test_billing_error_aborts_the_run():
    with pytest.raises(FatalRunError, match="out of credits"):
        evaluate(Target(Billing()), Harness({}), TASKS, k=2, workers=1)


def test_transient_error_is_recorded_not_fatal():
    res = evaluate(Target(Flaky()), Harness({}), TASKS, k=2, workers=2)
    assert res.infra_errors == 2 and res.rates["a"] == 0.0


class DailyQuota(Exception):
    status_code = 429

    def __str__(self):
        return "Rate limit exceeded: free-models-per-day"


def test_daily_free_quota_aborts_the_run():
    with pytest.raises(FatalRunError, match="daily free-model request quota"):
        evaluate(Target(DailyQuota()), Harness({}), TASKS, k=1, workers=1)


def test_rate_limiter_spaces_requests():
    import time

    from lamarck.llm import RateLimiter

    lim = RateLimiter(per_minute=600)  # one slot every 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        lim.wait()
    assert time.monotonic() - t0 >= 0.29


def test_non_claude_models_get_no_output_config():
    from types import SimpleNamespace

    from lamarck.llm import complete

    calls = []
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")], stop_reason="end_turn",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1), model="m")
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: calls.append(kw) or resp))
    complete(client, model="qwen/qwen3.8-27b:free", system="s", user="u", effort="low", json_schema={"type": "object"})
    complete(client, model="claude-opus-5-5", system="s", user="u", effort="low")
    assert "output_config" not in calls[0] and calls[0]["model"] == "qwen/qwen3.8-27b:free"
    assert calls[1]["output_config"] == {"effort": "low"}


def test_max_rollouts_stops_search_but_keeps_heldout_check():
    from lamarck.evolve import EvolveConfig, Evolver
    from lamarck.sim import ScriptedProposer, SimulatedTarget, demo_suite, seed_harness

    cfg = EvolveConfig(iterations=20, workers=1, max_rollouts=600)
    r = Evolver(SimulatedTarget(), ScriptedProposer(), demo_suite(), seed_harness(), cfg, log=None).run()
    assert r.total_rollouts <= 600
    assert any("Budget reached" in line for line in r.log)


def test_single_rollout_noise_is_not_zero():
    from lamarck.evaluate import EvalResult, noise_band
    from lamarck.targets.base import Trial

    trials = {f"t{i}": [Trial(f"t{i}", i % 3 == 0)] for i in range(6)}
    res = EvalResult(rates={k: float(v[0].passed) for k, v in trials.items()}, trials=trials, k=1)
    assert noise_band(res) > 0.05


def test_tiny_heldout_sample_is_inconclusive_not_ship():
    from lamarck.evolve import EvolveConfig, Evolver
    from lamarck.sim import ScriptedProposer, SimulatedTarget, demo_suite, seed_harness

    cfg = EvolveConfig(iterations=6, workers=1, k_heldout=1)
    r = Evolver(SimulatedTarget(), ScriptedProposer(), demo_suite(24, 3), seed_harness(), cfg, log=None).run()
    assert not r.decision.accept and r.decision.reason.startswith("inconclusive")


def test_network_outage_is_retried_then_fatal_not_a_task_failure(monkeypatch):
    import urllib.error

    from lamarck import chat_compat
    from lamarck.evaluate import describe_fatal, is_fatal

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = chat_compat.ChatClient.for_provider("groq")
    client.network_wait = 30  # allows a few short retries
    slept = []
    monkeypatch.setattr(chat_compat.time, "sleep", lambda s: slept.append(s))

    def down(req, timeout):
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(chat_compat.urllib.request, "urlopen", down)
    with pytest.raises(chat_compat.GatewayHTTPError) as ei:
        client.chat(model="m", system="s", user="u", max_tokens=5, effort=None, json_mode=False)
    assert slept and sum(slept) <= 30
    assert is_fatal(ei.value) and "network unavailable" in describe_fatal(ei.value)


def test_agent_crash_is_an_infra_error(tmp_path):
    import subprocess

    from lamarck.gitmine import MineConfig, mine_git_tasks
    from lamarck.targets.command import CommandTarget

    r = tmp_path / "repo"
    r.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@e.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(r), *args], check=True)
    (r / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "init"], check=True)
    (r / "a.py").write_text("x = 2\n")
    (r / "test_a.py").write_text("import a\nassert a.x == 2\n")
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "commit", "-qm", "x=2"], check=True)
    task = mine_git_tasks(r, MineConfig(test_cmd="{python} {tests}", heldout_fraction=0.0)).tasks[0]
    trial = CommandTarget(agent_cmd='{python} -c "raise SystemExit(70)"').run(Harness({}), task, 0)
    assert trial.error and "crashed" in trial.error and not trial.passed
