"""A/B testing a change to the agent's instructions (lamarck compare)."""

import json

import pytest

from lamarck.cli import main
from lamarck.compare import BETTER, INCONCLUSIVE, UNCLEAR, WORSE, compare_harnesses, decide, load_instructions
from lamarck.harness import SYSTEM, Harness
from lamarck.sim import CATALOG, TERSE_SKILL, SimulatedTarget, demo_suite


def test_decide_needs_the_difference_to_clear_the_noise_band():
    assert decide(0.40, 0.55, 0.10, 40)[0] == BETTER
    assert decide(0.40, 0.25, 0.10, 40)[0] == WORSE
    assert decide(0.40, 0.45, 0.10, 40)[0] == UNCLEAR
    assert decide(0.10, 0.90, 0.10, 8)[0] == INCONCLUSIVE  # too few attempts, however big the gap


def test_compare_detects_a_helpful_and_a_harmful_change():
    tasks = demo_suite().tasks
    base = Harness({SYSTEM: "You are a careful engineer.\n"})
    helpful = Harness({SYSTEM: "You are a careful engineer.\n", "skills/all.md": "\n".join(fix for _, fix in CATALOG.values())})
    harmful = Harness({SYSTEM: "You are a careful engineer.\n", "rules.md": f"- {TERSE_SKILL}\n"})
    up = compare_harnesses(SimulatedTarget(), base, helpful, tasks, k=4, workers=1)
    assert up.verdict == BETTER and up.delta_pts > up.noise_pts > 0
    down = compare_harnesses(SimulatedTarget(), helpful, Harness({**helpful.components, "rules.md": f"- {TERSE_SKILL}\n"}), tasks, k=4, workers=1)
    assert down.verdict in (WORSE, UNCLEAR) and down.delta_pts < 0
    assert len(up.rows) == len(tasks) and all(r.k == 4 for r in up.rows)
    assert harmful.fingerprint() != base.fingerprint()


def test_single_instruction_file_is_a_harness():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "CLAUDE.md"
        f.write_text("Always run the tests.\n", encoding="utf-8")
        assert load_instructions(f).components == {SYSTEM: "Always run the tests.\n"}


def test_compare_cli_end_to_end_and_plan_cannot_change(tmp_path):
    demo_suite().save(tmp_path / "tasks.json")
    (tmp_path / "lamarck.json").write_text(json.dumps({"tasks": "tasks.json", "target": {"type": "simulated"},
                                                       "evolve": {"workers": 1}}), encoding="utf-8")
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "system.md").write_text("You are a careful engineer.\n", encoding="utf-8")
    (tmp_path / "CLAUDE.new.md").write_text("You are a careful engineer.\n" + "\n".join(f for _, f in CATALOG.values()), encoding="utf-8")
    out = tmp_path / "cmp"
    args = ["compare", "-c", str(tmp_path / "lamarck.json"), "--base", str(tmp_path / "harness"),
            "--candidate", str(tmp_path / "CLAUDE.new.md"), "--out", str(out), "-y", "--allow-sleep"]
    assert main(args) == 0  # BETTER
    report = (out / "compare_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Agent change test (instructions): BETTER") and "| Task | Base | Candidate |" in report
    assert json.loads((out / "compare.json").read_text(encoding="utf-8"))["verdict"] == BETTER
    assert main(args) == 0  # same plan: replayed from saved attempts
    with pytest.raises(SystemExit, match="different comparison plan"):
        main(args + ["-k", "2"])


def test_compare_agent_settings_with_the_same_instructions(tmp_path):
    # e.g. more steps or another model: same CLAUDE.md, different agent. Shared instructions must not
    # make the candidate replay the base's saved attempts (they're keyed by instructions only).
    demo_suite().save(tmp_path / "tasks.json")
    (tmp_path / "lamarck.json").write_text(json.dumps({"tasks": "tasks.json", "target": {"type": "simulated", "ceiling": 0.45},
                                                       "evolve": {"workers": 1}}), encoding="utf-8")
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "system.md").write_text("You are a careful engineer.\n" + "\n".join(f for _, f in CATALOG.values()), encoding="utf-8")
    (tmp_path / "stronger.json").write_text(json.dumps({"type": "simulated", "ceiling": 0.95}), encoding="utf-8")
    out = tmp_path / "cmp"
    code = main(["compare", "-c", str(tmp_path / "lamarck.json"), "--candidate-agent", str(tmp_path / "stronger.json"),
                 "--out", str(out), "-y", "--allow-sleep"])
    result = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert code == 0 and result["verdict"] == BETTER and result["cand_mean"] > result["base_mean"]
    report = (out / "compare_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Agent change test (agent ceiling): BETTER")
    assert "| agent ceiling (changed) | `0.45` | `0.95` |" in report
    assert (out / "trials-base.jsonl").exists() and (out / "trials-candidate.jsonl").exists()


def test_identical_variants_are_refused(tmp_path):
    demo_suite().save(tmp_path / "tasks.json")
    (tmp_path / "lamarck.json").write_text(json.dumps({"tasks": "tasks.json", "target": {"type": "simulated"}}), encoding="utf-8")
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "system.md").write_text("hi\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="identical"):
        main(["compare", "-c", str(tmp_path / "lamarck.json"), "-y", "--allow-sleep"])


def test_shared_cache_with_same_instructions_and_different_agents_is_rejected():
    h = Harness({SYSTEM: "hi\n"})
    from lamarck.compare import compare_variants
    from lamarck.evaluate import TrialCache

    cache = TrialCache()
    with pytest.raises(ValueError, match="separate caches"):
        compare_variants(SimulatedTarget(ceiling=0.4), h, SimulatedTarget(ceiling=0.9), h, demo_suite().tasks, 2, 1, cache, cache)


def test_model_switch_inside_a_variant_makes_the_test_inconclusive():
    # a gateway (OmniRoute) swapped models on rate limits; a verdict would measure the swap, not the change
    from lamarck.targets.base import Trial

    class Gateway(SimulatedTarget):
        def run(self, harness, task, seed):
            t = super().run(harness, task, seed)
            t.models = ["model-a"] if seed % 2 == 0 else ["model-b"]  # falls back on every other attempt
            return t

    class Pinned(SimulatedTarget):
        def run(self, harness, task, seed):
            t = super().run(harness, task, seed)
            t.models = ["main-model", "small-model"]  # two models per attempt is fine if it never changes
            return t

    tasks = demo_suite().tasks
    base = Harness({SYSTEM: "You are a careful engineer.\n"})
    helpful = Harness({SYSTEM: "You are a careful engineer.\n", "skills/all.md": "\n".join(fix for _, fix in CATALOG.values())})
    swapped = compare_harnesses(Gateway(), base, helpful, tasks, k=4, workers=1)
    assert swapped.verdict == INCONCLUSIVE and "changed between attempts" in swapped.reason
    pinned = compare_harnesses(Pinned(), base, helpful, tasks, k=4, workers=1)
    assert pinned.verdict == BETTER and pinned.base_models == ["main-model", "small-model"]
    assert Trial.from_dict(Trial("t", True, models=["m"]).to_dict()).models == ["m"]
