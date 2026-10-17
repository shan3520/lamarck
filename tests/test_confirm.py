"""The pre-registered confirmation: pools the run's held-out rollouts with new ones (new seeds)."""

from lamarck.archive import CONFIRMED, Archive, Node
from lamarck.confirm import HELDOUT_SEED, confirm
from lamarck.evaluate import TrialCache
from lamarck.harness import Edit, Harness
from lamarck.selection import GateConfig
from lamarck.targets.base import Target, Trial
from lamarck.tasks import HELDOUT, Task


class Stub(Target):
    """The evolved harness (it has a rules.md) solves every task; the seed solves none."""

    def __init__(self):
        self.seeds = []

    def run(self, harness, task, seed):
        self.seeds.append(seed)
        ok = "rules.md" in harness.components
        return Trial(task.id, ok, score=1.0 if ok else 0.0)


def _archive():
    a = Archive(Harness({"system.md": "hi"}))
    a.add_root({}, 0.0)
    e = [Edit("append_rule", "rules.md", "- Re-run the tests.")]
    a.add(Node("n0001", "n0000", e, e, a.root_harness.apply(e).fingerprint(), CONFIRMED))
    return a


def test_confirm_pools_old_and_new_rollouts_with_new_seeds(tmp_path):
    tasks = [Task(f"t{i}", "p", {"type": "exact", "value": "x"}, HELDOUT) for i in range(3)]
    cache = TrialCache(tmp_path / "trials.jsonl")
    first = Stub()
    confirm(first, _archive(), "n0001", tasks, k_done=0, extra_k=4, gate=GateConfig(), cache=cache, workers=1)  # stands in for the run
    stub = Stub()
    c = confirm(stub, _archive(), "n0001", tasks, k_done=4, extra_k=2, gate=GateConfig(), cache=cache, workers=1)
    assert c.k_total == 6 and c.ship and c.delta_pts == 100.0
    assert sorted(set(stub.seeds)) == [HELDOUT_SEED + 4, HELDOUT_SEED + 5]  # the run's rollouts came from the cache
