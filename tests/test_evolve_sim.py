"""End-to-end behavior of the loop in the simulated world (no network, deterministic)."""

from lamarck.archive import CONFIRMED
from lamarck.critic import Critic
from lamarck.evolve import EvolveConfig, Evolver
from lamarck.harness import Edit
from lamarck.proposers.base import Proposal
from lamarck.report import render
from lamarck.sim import CATALOG, ScriptedProposer, SimulatedTarget, capabilities, demo_suite, seed_harness


def run(proposer, critic=None, iterations=10, seed=0):
    cfg = EvolveConfig(iterations=iterations, seed=seed, workers=1)
    return Evolver(SimulatedTarget(), proposer, demo_suite(), seed_harness(), cfg, critic=critic, log=None).run()


def test_generic_fixes_are_found_confirmed_and_shipped():
    r = run(ScriptedProposer())
    assert r.decision.accept, r.decision.reason
    caps = capabilities(r.best_harness)
    assert len(caps & set(CATALOG)) >= 3
    assert not any(c.startswith("task:") for c in caps), "a memorized-task edit reached the best harness"
    assert r.best.heldout_mean > r.archive.root.heldout_mean


def test_critic_blocks_task_specific_patches():
    r = run(ScriptedProposer(overfit_rate=0.6))
    blocked = [n for n in r.archive.nodes.values() if n.reason.startswith("critic: task-specific leakage")]
    assert blocked
    assert all(n.status != CONFIRMED for n in blocked)


class CheatOnly:
    """A 'bad genius' proposer: only memorizes training tasks."""

    name = "cheat"

    def propose(self, harness, evidence, budget, history, iteration):
        done = {h.hypothesis for h in history}
        out = []
        for c in evidence.clusters:
            for tid in c.task_ids:
                hyp = f"memorize {tid}"
                if hyp not in done and f"task:{tid}" not in capabilities(harness):
                    out.append(Proposal(hyp, [Edit("add_skill", f"skills/m-{len(done) + len(out)}.md", f"[cap:task:{tid}]")]))
                    break
            if len(out) >= 3:
                break
        return out


def test_heldout_gate_refuses_overfitting_when_critic_is_bypassed():
    # With the critic disabled, memorization wins in the loop...
    r = run(CheatOnly(), critic=Critic(denylist=set()), iterations=8)
    root_train = sum(r.archive.root.train_rates.values()) / len(r.archive.root.train_rates)
    best_train = sum(r.best.train_rates.values()) / len(r.best.train_rates)
    assert best_train > root_train + 0.1
    # ...but the held-out gate catches it.
    assert not r.decision.accept
    assert "does not clear noise" in r.decision.reason


def test_regression_gate_blocks_interference_from_best_lineage():
    r = run(ScriptedProposer(harmful_rate=0.5, overfit_rate=0.0))
    assert "terse" not in capabilities(r.best_harness)


def test_report_renders():
    r = run(ScriptedProposer(), iterations=4)
    text = render(r)
    assert "Verdict:" in text and "Held-out tasks" in text and "```diff" in text


def test_runs_are_reproducible():
    a, b = run(ScriptedProposer(), iterations=5), run(ScriptedProposer(), iterations=5)
    assert a.best_harness.fingerprint() == b.best_harness.fingerprint()
    assert a.decision == b.decision
