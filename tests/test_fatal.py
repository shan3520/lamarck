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
