import json

from lamarck.cli import main
from lamarck.sim import demo_suite


def test_demo_writes_run_directory(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--iterations", "4", "--out", str(out)]) == 0
    for f in ("report.md", "archive.json", "summary.json", "best_harness.rendered.md", "log.txt"):
        assert (out / f).exists(), f
    assert json.loads((out / "summary.json").read_text())["rollouts"] > 0


def test_init_then_run_with_simulated_backends(tmp_path):
    proj = tmp_path / "proj"
    assert main(["init", str(proj)]) == 0
    demo_suite(12, 8).save(proj / "tasks.json")
    conf = json.loads((proj / "lamarck.json").read_text())
    conf["target"] = {"type": "simulated"}
    conf["proposer"] = {"type": "scripted"}
    conf["evolve"] = {"iterations": 4, "workers": 1}
    (proj / "lamarck.json").write_text(json.dumps(conf))
    code = main(["run", "-c", str(proj / "lamarck.json"), "--out", str(proj / "run")])
    assert code in (0, 2)  # 0 = ship, 2 = nothing shippable; both are valid outcomes
    assert (proj / "run" / "report.md").exists()


def test_init_from_existing_claude_md(tmp_path):
    src = tmp_path / "CLAUDE.md"
    src.write_text("Use pnpm, not npm.\n")
    assert main(["init", str(tmp_path / "p"), "--from", str(src)]) == 0
    assert (tmp_path / "p" / "harness" / "system.md").read_text() == "Use pnpm, not npm.\n"


def test_resume_replays_from_saved_rollouts(tmp_path, capsys):
    proj = tmp_path / "proj"
    assert main(["init", str(proj)]) == 0
    demo_suite(12, 8).save(proj / "tasks.json")
    conf = json.loads((proj / "lamarck.json").read_text())
    conf.update(target={"type": "simulated"}, proposer={"type": "scripted"}, evolve={"iterations": 3, "workers": 1})
    (proj / "lamarck.json").write_text(json.dumps(conf))
    run_dir = proj / "run"
    main(["run", "-c", str(proj / "lamarck.json"), "--out", str(run_dir)])
    first = json.loads((run_dir / "summary.json").read_text())
    assert (run_dir / "trials.jsonl").exists() and (run_dir / "run_meta.json").exists()
    capsys.readouterr()
    main(["run", "--resume", str(run_dir)])
    assert "saved rollouts will be replayed" in capsys.readouterr().out
    assert json.loads((run_dir / "summary.json").read_text()) == first


def test_run_wait_sleeps_through_quota_and_finishes(tmp_path, monkeypatch, capsys):
    import time as _time

    from lamarck import cli
    from lamarck.evaluate import FatalRunError
    from lamarck.sim import SimulatedTarget

    class QuotaOnce(SimulatedTarget):
        hits = 0

        def run(self, harness, task, seed):
            if QuotaOnce.hits == 0 and seed == 1:
                QuotaOnce.hits += 1
                raise FatalRunError("daily model quota exhausted: Please try again in 1m30s.")
            return super().run(harness, task, seed)

    monkeypatch.setattr(cli, "build_target", lambda spec: QuotaOnce())
    slept = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    proj = tmp_path / "proj"
    main(["init", str(proj)])
    demo_suite(12, 8).save(proj / "tasks.json")
    conf = json.loads((proj / "lamarck.json").read_text())
    conf.update(target={"type": "simulated"}, proposer={"type": "scripted"}, evolve={"iterations": 2, "workers": 1})
    (proj / "lamarck.json").write_text(json.dumps(conf))
    code = main(["run", "-c", str(proj / "lamarck.json"), "--out", str(proj / "run"), "--wait"])
    out = capsys.readouterr().out
    assert code in (0, 2) and "Waiting 2 min" in out and slept == [120.0]
    assert (proj / "run" / "report.md").exists()


def test_run_wait_backs_off_when_quota_waits_finish_nothing(tmp_path, monkeypatch, capsys):
    import time as _time

    from lamarck import cli
    from lamarck.evaluate import FatalRunError
    from lamarck.sim import SimulatedTarget

    class Starved(SimulatedTarget):
        hits = 0

        def run(self, harness, task, seed):
            if Starved.hits < 3:  # the first three tries hit the quota before anything is saved
                Starved.hits += 1
                raise FatalRunError("quota: Please try again in 2m0s.")
            return super().run(harness, task, seed)

    monkeypatch.setattr(cli, "build_target", lambda spec: Starved())
    slept = []
    monkeypatch.setattr(_time, "sleep", lambda s: slept.append(s))
    proj = tmp_path / "proj"
    main(["init", str(proj)])
    demo_suite(12, 8).save(proj / "tasks.json")
    conf = json.loads((proj / "lamarck.json").read_text())
    conf.update(target={"type": "simulated"}, proposer={"type": "scripted"}, evolve={"iterations": 1, "workers": 1})
    (proj / "lamarck.json").write_text(json.dumps(conf))
    main(["run", "-c", str(proj / "lamarck.json"), "--out", str(proj / "run"), "--wait"])
    assert slept == [150.0, 1230.0, 2430.0]  # provider hint, then 20 and 40 minutes (+30 s margin)
