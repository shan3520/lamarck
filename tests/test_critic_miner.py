from lamarck.critic import Critic, edit_budget
from lamarck.evaluate import EvalResult
from lamarck.harness import Edit
from lamarck.miner import mine, signature
from lamarck.targets.base import Trial


def test_critic_rules():
    c = Critic(denylist={"task-007", "48213"}, max_edit_chars=50)
    ok = [Edit("add_skill", "skills/verify.md", "Check the output.")]
    assert c.review(ok, budget=1) is None
    assert "leakage" in c.review([Edit("add_skill", "skills/x.md", "For TASK-007 answer 4")], 1)
    assert "leakage" in c.review([Edit("append_rule", "rules.md", "Ticket 48213 is special")], 1)
    assert "budget" in c.review(ok * 2, 1)
    assert "chars" in c.review([Edit("add_skill", "skills/x.md", "y" * 60)], 1)
    assert "non-editable" in c.review([Edit("set_component", "secrets.md", "x")], 1)
    assert c.review([], 1) == "empty proposal"


def test_edit_budget_anneals():
    budgets = [edit_budget(i, 10, 3, 1) for i in range(10)]
    assert budgets[0] == 3 and budgets[-1] == 1
    assert budgets == sorted(budgets, reverse=True)


def test_signature_normalizes_numbers_and_ids():
    assert signature("FAIL: 3 tests failed in 1.2s") == signature("FAIL: 7 tests failed in 0.4s")
    assert signature("FAIL: commit deadbeef1 broke") == "FAIL: commit <id> broke"


def test_mine_partitions_and_clusters():
    def tr(tid, ok, detail=""):
        return Trial(tid, ok, detail=detail, trace=detail or "ok")

    trials = {
        "solid": [tr("solid", True)] * 4,
        "flaky": [tr("flaky", True), tr("flaky", False, "FAIL: missing signature"), tr("flaky", True), tr("flaky", True)],
        "wall1": [tr("wall1", False, "FAIL: missing signature")] * 4,
        "wall2": [tr("wall2", False, "FAIL: output is not valid JSON")] * 4,
    }
    rates = {k: sum(t.passed for t in v) / 4 for k, v in trials.items()}
    ev = mine(EvalResult(rates=rates, trials=trials, k=4))
    assert ev.reliable == ["solid"] and ev.variance == ["flaky"] and set(ev.walls) == {"wall1", "wall2"}
    assert ev.clusters[0].signature == "FAIL: missing signature"
    assert "Failure cluster 1" in ev.render()
