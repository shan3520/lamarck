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
    assert report.startswith("# Instruction change test: BETTER") and "| Task | Base | Candidate |" in report
    assert json.loads((out / "compare.json").read_text(encoding="utf-8"))["verdict"] == BETTER
    assert main(args) == 0  # same plan: replayed from saved attempts
    with pytest.raises(SystemExit, match="different comparison plan"):
        main(args + ["-k", "2"])
