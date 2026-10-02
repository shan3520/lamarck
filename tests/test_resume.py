import pytest

from lamarck.evaluate import FatalRunError, TrialCache
from lamarck.evolve import EvolveConfig, Evolver
from lamarck.proposers.cache import CachingProposer
from lamarck.sim import ScriptedProposer, SimulatedTarget, demo_suite, seed_harness


class QuotaTarget:
    """Simulated target whose 'daily quota' runs out after `quota` real rollouts."""

    name = "quota-sim"

    def __init__(self, quota=None):
        self.inner = SimulatedTarget()
        self.quota = quota
        self.calls = 0

    def run(self, harness, task, seed):
        if self.quota is not None and self.calls >= self.quota:
            raise FatalRunError("daily quota exhausted")
        self.calls += 1
        return self.inner.run(harness, task, seed)


def evolve(target, tmp, tag):
    cfg = EvolveConfig(iterations=6, workers=1)
    cache = TrialCache(tmp / tag / "trials.jsonl")
    prop = CachingProposer(ScriptedProposer(), tmp / tag / "proposals.jsonl")
    return Evolver(target, prop, demo_suite(), seed_harness(), cfg, log=None, cache=cache).run()


def test_paused_run_resumes_to_the_same_result(tmp_path):
    reference = evolve(QuotaTarget(), tmp_path, "ref")

    day1 = QuotaTarget(quota=300)
    with pytest.raises(FatalRunError):
        evolve(day1, tmp_path, "paused")
    day2 = QuotaTarget()
    resumed = evolve(day2, tmp_path, "paused")

    assert resumed.best_harness.fingerprint() == reference.best_harness.fingerprint()
    assert resumed.decision == reference.decision
    assert resumed.total_rollouts == reference.total_rollouts
    # day 2 paid only for what day 1 had not done
    assert day1.calls == 300 and day2.calls == reference.total_rollouts - 300


def test_trial_cache_round_trip(tmp_path):
    from lamarck.targets.base import Trial

    c = TrialCache(tmp_path / "t.jsonl")
    c.put("k1", Trial("a", True, detail="", trace="t", tokens=5, score=1.0))
    again = TrialCache(tmp_path / "t.jsonl")
    got = again.get("k1")
    assert got.passed and got.tokens == 5 and len(again) == 1 and again.get("missing") is None
