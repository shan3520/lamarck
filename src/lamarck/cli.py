"""Command line interface.

    lamarck demo                         zero-cost simulated run (no API key needed)
    lamarck init DIR [--from CLAUDE.md]  scaffold a project (harness + config)
    lamarck mine-git --repo PATH         mine verified tasks from git history
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


def cmd_run(args) -> int:
    if args.resume:
        meta_path = Path(args.resume) / "run_meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"nothing to resume: {meta_path} not found")
        args.config, args.out = json.loads(meta_path.read_text(encoding="utf-8"))["config"], args.resume
    cfg_path = Path(args.config).resolve()
    base = cfg_path.parent
    conf = {**DEFAULT_CONFIG, **json.loads(cfg_path.read_text(encoding="utf-8"))}
    suite = TaskSuite.load(base / conf["tasks"])
    if not suite.heldout():
        suite = suite.assign_splits(0.3)
        print("No held-out split in the task file; assigned 30% of tasks to held-out deterministically.")
    harness = Harness.load(base / conf["harness_dir"])
    gate = _pick(GateConfig, conf.get("gate", {}))
    ecfg = _pick(EvolveConfig, {**conf.get("evolve", {}), "gate": gate})
    load_dotenv(base / ".env", Path.cwd() / ".env")
    via = configure_gateway()
    from .llm import set_rate_limit

    set_rate_limit(conf.get("requests_per_minute"))
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
    try:
        result = Evolver(target, proposer, suite, harness, ecfg, critic=critic, cache=cache).run()
    except Exception as exc:
        if is_fatal(exc):
            print(f"\nRun paused: {describe_fatal(exc)}. Progress is saved ({len(cache)} rollouts).")
            print(f"Resume later with:  lamarck run --resume {out}")
            return 4
        raise
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
    m.add_argument("--test-cmd", default="python -m pytest -q {tests}", help="verifier command; {tests} is replaced by the task's test files")
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

    r = sub.add_parser("run", help="evolve the harness")
    r.add_argument("-c", "--config", default="lamarck.json")
    r.add_argument("--resume", metavar="RUN_DIR", help="continue a paused run (e.g. after a daily quota ran out)")
    r.add_argument("--out")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before spending on paid models")
    r.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
