from lamarck.calibrate import CALIBRATION_SEED, calibrate, select, table, zone
from lamarck.critic import Critic
from lamarck.evaluate import TrialCache
from lamarck.harness import Edit, Harness
from lamarck.targets.base import Target, Trial
from lamarck.tasks import Task, TaskSuite, task_specific_literals


class Stub(Target):
    """wall: never solved; easy: always; coin: solved on even seeds."""

    def __init__(self):
        self.seeds = []

    def run(self, harness, task, seed):
        self.seeds.append(seed)
        ok = task.id == "easy" or (task.id.startswith("coin") and seed % 2 == 0)
        return Trial(task.id, ok, score=1.0 if ok else 0.0, tokens=10)


def _suite():
    mk = lambda i, split="train": Task(i, f"do {i}", {"type": "exact", "value": "x"}, split)
    return TaskSuite([mk("wall"), mk("easy"), mk("coin-a"), mk("coin-b", "heldout")])


def test_zone_boundaries():
    assert zone(0.0) == "wall"
    assert zone(0.5) == "learnable"
    assert zone(1.0) == "solved"
    assert zone(0.95, high=0.9) == "solved"


def test_calibrate_keeps_only_learnable_tasks_and_their_splits(tmp_path):
    target = Stub()
    cals, _ = calibrate(target, Harness({"system.md": "hi"}), _suite(), k=2, workers=1)
    zones = {c.task_id: c.zone for c in cals}
    assert zones == {"wall": "wall", "easy": "solved", "coin-a": "learnable", "coin-b": "learnable"}
    kept = select(_suite(), cals)
    assert [t.id for t in kept.train()] == ["coin-a"] and [t.id for t in kept.heldout()] == ["coin-b"]
    assert "2 learnable, 1 walls" in table(cals)


def test_calibration_seeds_never_overlap_run_seeds():
    # tasks are selected on these outcomes; a run must not reuse them (selection bias)
    target = Stub()
    calibrate(target, Harness({"system.md": "hi"}), _suite(), k=3, workers=1)
    assert min(target.seeds) == CALIBRATION_SEED


def test_calibration_resumes_from_cache(tmp_path):
    cache = TrialCache(tmp_path / "c.jsonl")
    calibrate(Stub(), Harness({"system.md": "hi"}), _suite(), k=2, workers=1, cache=cache)
    again = Stub()
    calibrate(again, Harness({"system.md": "hi"}), _suite(), k=2, workers=1, cache=TrialCache(tmp_path / "c.jsonl"))
    assert again.seeds == []


def test_critic_rejects_code_identifiers_named_by_a_single_task():
    suite = TaskSuite([
        Task("t1", "Raise SignatureExpired and set date_signed", {"type": "exact", "value": "a"}),
        Task("t2", "Fix slugify with a custom_separator", {"type": "exact", "value": "b"}),
        Task("t3", "Use custom_separator in humanize", {"type": "exact", "value": "c"}),
    ])
    lits = task_specific_literals(suite)
    assert {"SignatureExpired", "date_signed"} <= lits
    assert "custom_separator" not in lits  # shared by two tasks: allowed
    critic = Critic(denylist=lits)
    bad = Edit("add_skill", "skills/timestamps.md", "Always set date_signed to an aware datetime.")
    ok = Edit("append_rule", "rules.md", "- Run the failing tests before editing, and again before finishing.")
    assert "leakage" in critic.review([bad], budget=3)
    assert critic.review([ok], budget=3) is None


def test_python_builtin_names_are_not_task_specific():
    suite = TaskSuite([
        Task("t1", "Raise ValueError when SignatureExpired", {"type": "exact", "value": "a"}),
        Task("t2", "Fix the parser", {"type": "exact", "value": "b"}),
    ])
    lits = task_specific_literals(suite)
    assert "SignatureExpired" in lits and "ValueError" not in lits


def test_saved_suite_keeps_repo_paths_relative(tmp_path):
    (tmp_path / "repos" / "lib").mkdir(parents=True)
    f = tmp_path / "tasks.json"
    f.write_text('{"tasks": [{"id": "t1", "prompt": "p", "verifier": {"type": "exact", "value": "x"}, "meta": {"repo": "repos/lib"}}]}', encoding="utf-8")
    s = TaskSuite.load(f)
    assert s.tasks[0].meta["repo"] == str((tmp_path / "repos" / "lib").resolve())
    out = tmp_path / "out.json"
    s.save(out)
    assert '"repo": "repos/lib"' in out.read_text(encoding="utf-8")
    assert TaskSuite.load(out).tasks[0].meta["repo"] == s.tasks[0].meta["repo"]
