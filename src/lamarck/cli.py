"""Command line interface.

    lamarck demo                         zero-cost simulated run (no API key needed)
    lamarck init DIR [--from CLAUDE.md]  scaffold a project (harness + config)
    lamarck mine-git --repo PATH         mine verified tasks from git history
    lamarck calibrate -c lamarck.json    keep only tasks the agent sometimes solves
    lamarck run -c lamarck.json          evolve the harness; writes a run directory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

from .critic import Critic
from .env import configure_gateway, load_dotenv
from .evaluate import describe_fatal, is_fatal
from .evolve import EvolveConfig, Evolver, RunResult
from .gitmine import MineConfig, mine_git_tasks, validate_suite
from .harness import RULES, SYSTEM, Harness
from .report import render
from .selection import GateConfig
from .tasks import TaskSuite, task_specific_literals

DEFAULT_CONFIG = {
    "harness_dir": "harness",
    "tasks": "tasks.json",
    "target": {"type": "anthropic", "model": "claude-opus-5-5", "effort": "low"},
    "proposer": {"type": "anthropic", "model": "claude-opus-5-5", "effort": "high"},
    "evolve": {"iterations": 6, "k_screen": 2, "k_confirm": 4, "k_heldout": 4, "workers": 4, "max_total_tokens": 3_000_000},
    "gate": {},
}


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def build_target(spec: dict):
    kind = spec.get("type")
    if kind == "simulated":
        from .sim import SimulatedTarget

        return SimulatedTarget()
    if kind in ("anthropic", "llm"):
        from .targets.anthropic_api import AnthropicTarget

        return AnthropicTarget(model=spec.get("model", "claude-opus-5-5"), effort=spec.get("effort"), max_tokens=spec.get("max_tokens", 4000),
                               provider=spec.get("provider", "anthropic"))
    if kind == "command":
        from .targets.command import CommandTarget

        return CommandTarget(agent_cmd=spec["agent_cmd"], harness_file=spec.get("harness_file", "CLAUDE.md"), timeout=spec.get("timeout", 1800))
    raise SystemExit(f"unknown target type {kind!r} (expected simulated | llm | command)")


def build_proposer(spec: dict):
    kind = spec.get("type")
    if kind == "scripted":
        from .sim import ScriptedProposer

        return ScriptedProposer()
    if kind in ("anthropic", "llm"):
        from .proposers.llm import LLMProposer

        return LLMProposer(model=spec.get("model", "claude-opus-5-5"), effort=spec.get("effort", "high"), n_proposals=spec.get("n_proposals", 3),
                           provider=spec.get("provider", "anthropic"))
    raise SystemExit(f"unknown proposer type {kind!r} (expected scripted | llm)")


def _pick(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise SystemExit(f"unknown {cls.__name__} option(s): {sorted(unknown)}")
    return cls(**d)


def write_run(result: RunResult, out: Path, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result.archive.save(out / "archive.json")
    result.best_harness.save(out / "best_harness")
    (out / "best_harness.rendered.md").write_text(result.best_harness.render(), encoding="utf-8")
    (out / "report.md").write_text(render(result, title), encoding="utf-8")
    (out / "log.txt").write_text("\n".join(result.log) + "\n", encoding="utf-8")
    summary = {
        "ship": result.decision.accept,
        "reason": result.decision.reason,
        "best": result.best.id,
        "root_heldout": result.archive.root.heldout_mean,
        "best_heldout": result.best.heldout_mean,
        "rollouts": result.total_rollouts,
        "tokens": result.total_tokens,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---- commands ------------------------------------------------------------
def cmd_demo(args) -> int:
    from .sim import ScriptedProposer, SimulatedTarget, demo_suite, seed_harness

    print("SIMULATION: exercises the engine end to end for free. It is not evidence that harness evolution works.\n")
    suite = demo_suite()
    cfg = EvolveConfig(iterations=args.iterations, seed=args.seed)
    result = Evolver(SimulatedTarget(), ScriptedProposer(seed=args.seed), suite, seed_harness(), cfg).run()
    out = Path(args.out)
    write_run(result, out, "Lamarck demo (simulated)")
    print(f"\nReport: {out / 'report.md'}")
    return 0


def cmd_init(args) -> int:
    root = Path(args.dir)
    hdir = root / "harness"
    if (root / "lamarck.json").exists():
        raise SystemExit(f"{root / 'lamarck.json'} already exists")
    hdir.mkdir(parents=True, exist_ok=True)
    if args.from_file:
        text = Path(args.from_file).read_text(encoding="utf-8")
        Harness({SYSTEM: text}).save(hdir)
    else:
        Harness({SYSTEM: "You are a careful software engineering agent working in this repository.\n", RULES: "- Follow the user's instructions.\n"}).save(hdir)
    (root / "lamarck.json").write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
    print(f"Created {root / 'lamarck.json'} and {hdir}/. Next: add tasks (lamarck mine-git, or write {root / 'tasks.json'}), then lamarck run -c {root / 'lamarck.json'}")
    return 0


def cmd_mine_git(args) -> int:
    cfg = MineConfig(test_cmd=args.test_cmd, max_commits=args.max_commits, max_tasks=args.max_tasks, heldout_fraction=args.heldout)
    suite = mine_git_tasks(args.repo, cfg)
    suite.save(args.out)
    print(f"Mined {len(suite.tasks)} tasks ({len(suite.train())} train / {len(suite.heldout())} held-out) -> {args.out}")
    if not suite.tasks:
        print("No commits changed both tests and source. Try a larger --max-commits or a different --test-cmd/glob.")
    return 0


def cmd_validate(args) -> int:
    suite = TaskSuite.load(args.tasks)
    print(f"Validating {len(suite.tasks)} tasks (fail before fix, pass after fix)...")
    kept, _ = validate_suite(suite, args.max_seconds)
    if args.heldout is not None:
        kept = kept.assign_splits(args.heldout)
    kept.save(args.out)
    print(f"Kept {len(kept.tasks)} of {len(suite.tasks)} ({len(kept.train())} train / {len(kept.heldout())} held-out) -> {args.out}")
    return 0 if kept.tasks else 1


def _load_project(config: str):
    """Config, tasks, seed harness and credentials for a project (shared by run and calibrate)."""
    cfg_path = Path(config).resolve()
    base = cfg_path.parent
    conf = {**DEFAULT_CONFIG, **json.loads(cfg_path.read_text(encoding="utf-8"))}
    suite = TaskSuite.load(base / conf["tasks"])
    harness = Harness.load(base / conf["harness_dir"])
    load_dotenv(base / ".env", Path.cwd() / ".env")
    via = configure_gateway()
    from .llm import set_rate_limit

    set_rate_limit(conf.get("requests_per_minute"))
    return cfg_path, base, conf, suite, harness, via


def _stay_awake(args) -> None:
    """Long runs stalled for hours when the PC idle-slept (frozen agents, network drops on wake)."""
    if not getattr(args, "allow_sleep", False):
        from .awake import keep_awake

        if keep_awake():
            print("Keeping this computer awake while the run is in progress (--allow-sleep to disable).")


def cmd_calibrate(args) -> int:
    from .calibrate import calibrate, select, table
    from .evaluate import TrialCache

    cfg_path, base, conf, suite, harness, _ = _load_project(args.config)
    _stay_awake(args)
    if conf["target"].get("type") == "simulated":
        raise SystemExit("calibration needs a real target (the simulated one has no task difficulty to measure)")
    target = build_target(conf["target"])
    out = Path(args.out) if args.out else base / "tasks.calibrated.json"
    cache = TrialCache(out.with_suffix(".trials.jsonl"))  # resumable: rerun the same command after a pause
    n = len(suite.tasks)
    print(f"Calibrating {n} tasks with the seed harness: {args.k} rollouts each (~{n * args.k} agent runs).")
    if len(cache):
        print(f"Resuming: {len(cache)} saved rollouts are replayed from disk at no cost.")
    if not args.yes and input("This runs the agent on every task. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted.")
        return 1
    workers = args.workers or conf.get("evolve", {}).get("workers", 3)
    from .chat_compat import retry_hint

    while True:
        try:
            cals, _ = calibrate(target, harness, suite, k=args.k, workers=workers, cache=cache, low=args.low, high=args.high)
            break
        except Exception as exc:
            if not is_fatal(exc):
                raise
            # an overnight network outage paused a calibration for 7 h; with --wait it checks back instead
            wait = (retry_hint(str(exc)) or 300.0) if args.wait else None
            if wait is None:
                print(f"\nPaused: {describe_fatal(exc)}. {len(cache)} rollouts are saved; rerun the same command to continue.")
                return 4
            wait = min(wait, args.max_wait * 3600)
            print(f"\nPaused: {str(exc)[:300]}\nWaiting {wait / 60:.0f} min ({len(cache)} rollouts saved), then continuing...", flush=True)
            time.sleep(wait + 30)
    print(table(cals))
    kept = select(suite, cals, args.heldout)
    kept.save(out)
    (out.with_suffix(".report.json")).write_text(json.dumps([c.__dict__ for c in cals], indent=2), encoding="utf-8")
    print(f"\nKept {len(kept.tasks)} learnable tasks ({len(kept.train())} train / {len(kept.heldout())} held-out) -> {out}")
    if len(kept.train()) < 4 or len(kept.heldout()) < 3:
        print("Too few learnable tasks for a trustworthy run. Mine more tasks, raise the agent's step budget, "
              "or use a stronger model; walls give harness evolution nothing to learn from.")
        return 2
    print(f"Next: point \"tasks\" in {cfg_path.name} at {out.name}, then lamarck run -c {cfg_path.name}")
    return 0


def cmd_run(args) -> int:
    if args.resume:
        meta_path = Path(args.resume) / "run_meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"nothing to resume: {meta_path} not found")
        args.config, args.out = json.loads(meta_path.read_text(encoding="utf-8"))["config"], args.resume
    cfg_path, base, conf, suite, harness, via = _load_project(args.config)
    _stay_awake(args)
    if not suite.heldout():
        suite = suite.assign_splits(0.3)
        print("No held-out split in the task file; assigned 30% of tasks to held-out deterministically.")
    gate = _pick(GateConfig, conf.get("gate", {}))
    ecfg = _pick(EvolveConfig, {**conf.get("evolve", {}), "gate": gate})
    target, proposer = build_target(conf["target"]), build_proposer(conf["proposer"])

    est = len(suite.train()) * ecfg.k_confirm + ecfg.iterations * (ecfg.proposals_per_iter * len(suite.train()) * ecfg.k_confirm + 1) + 2 * len(suite.heldout()) * ecfg.k_heldout
    if ecfg.max_rollouts:
        est = min(est, ecfg.max_rollouts + ecfg.iterations)
    llm_roles = [conf[r] for r in ("target", "proposer") if conf[r].get("type") in ("anthropic", "llm")]
    paid = conf["target"].get("type") != "simulated" or bool(llm_roles)
    free = bool(llm_roles) and conf["target"].get("type") != "command" and all(
        ":free" in r.get("model", "") or r.get("provider") == "groq" for r in llm_roles)
    print(f"Tasks: {len(suite.train())} train / {len(suite.heldout())} held-out. Upper-bound requests: ~{est}. Token cap: {ecfg.max_total_tokens}. Rate limit: {conf.get('requests_per_minute') or 'none'}/min.")
    if paid:
        names = sorted({r.get("provider", "anthropic") for r in llm_roles})
        print("Model provider: " + ", ".join(via if n == "anthropic" else n for n in names))
    if paid and not args.yes:
        what = "free-tier models (uses your daily free quota, no charges)" if free else "paid models / runs agents"
        reply = input(f"This run calls {what}. Continue? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    out = Path(args.out) if args.out else base / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps({"config": str(cfg_path)}, indent=2), encoding="utf-8")
    from .evaluate import TrialCache
    from .proposers.cache import CachingProposer

    cache = TrialCache(out / "trials.jsonl")
    if len(cache):
        print(f"Resuming: {len(cache)} saved rollouts will be replayed from disk at no cost.")
    proposer = CachingProposer(proposer, out / "proposals.jsonl")
    critic = Critic(denylist=task_specific_literals(suite))
    from .chat_compat import retry_hint

    stalled, last_saved = 0, -1
    while True:
        try:
            result = Evolver(target, proposer, suite, harness, ecfg, critic=critic, cache=cache).run()
            break
        except Exception as exc:
            if not is_fatal(exc):
                raise
            wait = retry_hint(str(exc))
            # internet outage or per-minute overrun: no quota is being burned, so just check again in a
            # few minutes (a real outage overnight left a run sleeping 2.5 h after the network was back)
            transient = wait is None and ("network" in str(exc).lower() or "rate limit still exceeded" in str(exc))
            if transient:
                wait = 300.0
            # The provider's hint only says when the *next request* fits. If waiting that long
            # finished no rollout, a partial attempt was wasted: back off (20, 40, 80 min ...) so the
            # rolling window frees enough quota for whole attempts.
            stalled = stalled + 1 if len(cache) == last_saved else 0
            last_saved = len(cache)
            if wait is not None and stalled and not transient:
                wait = max(wait, 1200 * 2 ** (stalled - 1))
            if wait is not None:
                wait = min(wait, args.max_wait * 3600)
            if args.wait and wait is not None:
                # Rolling per-day quotas free up over time: sleep, then replay from the saved rollouts.
                resume_at = time.strftime("%H:%M", time.localtime(time.time() + wait + 30))
                print(f"\nPaused: {str(exc)[:300]}", flush=True)
                print(f"Quota reached ({len(cache)} rollouts saved). Waiting {wait / 60:.0f} min, until about {resume_at}, then continuing...", flush=True)
                time.sleep(wait + 30)
                continue
            print(f"\nRun paused: {describe_fatal(exc)}. Progress is saved ({len(cache)} rollouts).")
            print(f"Resume later with:  lamarck run --resume {out}" + (" --wait" if not args.wait else ""))
            return 4
    write_run(result, out, f"Lamarck run: {cfg_path}")
    print(f"\nReport: {out / 'report.md'}\nBest harness: {out / 'best_harness.rendered.md'}")
    return 0 if result.decision.accept else 2


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    p = argparse.ArgumentParser(prog="lamarck", description="Evolve your agent's harness against verified tasks, with held-out proof.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="zero-cost simulated run")
    d.add_argument("--iterations", type=int, default=10)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default="runs/demo")
    d.set_defaults(fn=cmd_demo)

    i = sub.add_parser("init", help="scaffold a project directory")
    i.add_argument("dir")
    i.add_argument("--from", dest="from_file", help="seed the harness from an existing CLAUDE.md / AGENTS.md")
    i.set_defaults(fn=cmd_init)

    m = sub.add_parser("mine-git", help="mine verified tasks from git history")
    m.add_argument("--repo", default=".")
    m.add_argument("--test-cmd", default="{python} -m pytest -q {tests}", help="verifier command; {tests} = the task's test files, {python} = the interpreter Lamarck runs in")
    m.add_argument("--out", default="tasks.json")
    m.add_argument("--max-commits", type=int, default=300)
    m.add_argument("--max-tasks", type=int, default=60)
    m.add_argument("--heldout", type=float, default=0.3)
    m.set_defaults(fn=cmd_mine_git)

    v = sub.add_parser("validate-tasks", help="keep only tasks whose tests fail before the fix and pass after it")
    v.add_argument("tasks")
    v.add_argument("--out", required=True)
    v.add_argument("--max-seconds", type=float, default=120.0, help="drop tasks whose tests take longer than this")
    v.add_argument("--heldout", type=float, default=None, help="re-assign the train/held-out split with this held-out fraction")
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser("calibrate", help="keep only tasks the agent sometimes solves (drop walls and already-solved tasks)")
    c.add_argument("-c", "--config", default="lamarck.json")
    c.add_argument("--out", help="calibrated task file (default: tasks.calibrated.json next to the config)")
    c.add_argument("-k", type=int, default=2, help="rollouts per task (default 2)")
    c.add_argument("--low", type=float, default=0.0, help="score at or below this is a wall (default 0)")
    c.add_argument("--high", type=float, default=0.9, help="score at or above this is already solved (default 0.9)")
    c.add_argument("--heldout", type=float, default=None, help="re-assign the train/held-out split with this held-out fraction")
    c.add_argument("--workers", type=int, help="parallel agent runs (default: evolve.workers from the config)")
    c.add_argument("-y", "--yes", action="store_true")
    c.add_argument("--wait", action="store_true", help="on a quota, throttle or network outage, wait and continue instead of pausing")
    c.add_argument("--max-wait", type=float, default=12.0, help="with --wait: longest single wait in hours (default 12)")
    c.add_argument("--allow-sleep", action="store_true", help="let the computer idle-sleep during the run")
    c.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("run", help="evolve the harness")
    r.add_argument("-c", "--config", default="lamarck.json")
    r.add_argument("--resume", metavar="RUN_DIR", help="continue a paused run (e.g. after a daily quota ran out)")
    r.add_argument("--out")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before spending on paid models")
    r.add_argument("--wait", action="store_true", help="when a rolling daily quota is reached, wait for it and continue instead of pausing")
    r.add_argument("--max-wait", type=float, default=12.0, help="with --wait: longest single wait in hours (default 12)")
    r.add_argument("--allow-sleep", action="store_true", help="let the computer idle-sleep during the run")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
