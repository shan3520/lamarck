# Lamarck

**Coding agents that get measurably better on *your* repo, with held-out proof.**

Lamarck evolves your agent's *harness*: the instructions, rules and skills it reads (`CLAUDE.md`, `AGENTS.md`, skill files). The model itself stays frozen. Every change has to win on tasks verified by real checks, and a change only ships if it also wins on **held-out tasks the loop never saw**.

```
your git history ──► verified tasks (commits that made tests pass) ──► train / held-out split
                                                                         │
seed harness ──► run agent ──► mine failures ──► propose small edit ──► critic ──► screen ──► confirm
      ▲                                                                                        │
      └──────────────────── archive of variants · merge complementary specialists ◄────────────┘
                                                                                               │
                                       held-out gate: ship only if the gain clears the noise ◄─┘
```

## Why this exists
Research in 2026 showed that you can rewrite the harness around a frozen model and get large gains on agent benchmarks. Sources are listed in [NOTICE.md](NOTICE.md):
- Meta-Harness: up to 7.7 points
- Self-Harness: up to 132% relative
- DarwinX: about 17 points on average

The same research showed the catch. In-loop gains overfit, sometimes badly: DarwinX measured a 31.7-point gap between in-loop and held-out scores, and RRSI saw +14.1 in-distribution fall to +4.7 out of distribution. Lamarck turns the best published selection method into a tool a team can run on its own codebase:
- the DarwinX preserve-and-extend contract, which has no public implementation
- RRSI-style regularization
- a strict held-out gate

The analysis behind this repo is in [docs/research/brief.md](docs/research/brief.md) and [docs/research/thesis.md](docs/research/thesis.md).

## Quick start
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -e ".[llm,dev]"
```
```bash
.venv/Scripts/lamarck demo
```
`demo` runs the whole loop in a **simulated** world. It's free and needs no API key. The report lands in `runs/demo/report.md`. The simulation exercises every safeguard, but it is *not* evidence that harness evolution works; the papers are that evidence.

### Real run: convention learning with Claude (about 1–2M tokens)
`examples/acme-support` contains 22 support tasks. The verifiers enforce house conventions that the task prompts never state:
- ticket references formatted as `Ticket #12345`
- a fixed sign-off
- no promised timelines
- `USD 0.00` amounts
- strict triage JSON

The loop has to discover these conventions from failures and write them into the harness.
```bash
export ANTHROPIC_API_KEY=...   # or: ant auth login
```
```bash
.venv/Scripts/lamarck run -c examples/acme-support/lamarck.json
```
**Using OpenRouter instead of an Anthropic key:** put `OPENROUTER_API_KEY=sk-or-...` in a `.env` file at the repo root (`.env` is git-ignored). Lamarck then routes the Anthropic SDK through OpenRouter's Anthropic-compatible endpoint and maps model ids (`claude-opus-5-5` becomes `anthropic/claude-opus-5.5`). Server-side refusal fallbacks are an Anthropic API feature, so they are switched off on OpenRouter.

**Free models (no spend):** `examples/acme-support/lamarck.free.json` uses OpenRouter `:free` models. A weaker target (`qwen/qwen3.8-27b:free`) leaves more room to improve, and the strongest free proposer is `nvidia/nemotron-3-ultra-550b-a55b:free`. The config sets these limits:
- `requests_per_minute: 18` (OpenRouter allows 20)
- `max_rollouts: 900`, so a run fits in one day's quota and always reserves the held-out check
- a stop with a clear message if the daily quota runs out

OpenRouter allows **50 free requests/day**, or **1,000/day once you have bought at least $10 of credit, ever**. Free requests don't consume that credit. A meaningful run needs about 450 requests, so it needs the 1,000/day tier.

Free providers may log or train on prompts, so don't send private code to `:free` models.
```bash
.venv/Scripts/lamarck run -c examples/acme-support/lamarck.free.json
```

The run asks for confirmation before it spends anything. A token cap is set in `lamarck.json` (`max_total_tokens`).

### Your repo: Claude Code / Codex as the target
```bash
lamarck init .lamarck --from CLAUDE.md
```
```bash
lamarck mine-git --repo . --test-cmd "python -m pytest -q {tests}" --out .lamarck/tasks.json
```
Then set the target in `.lamarck/lamarck.json`:
```json
"target": {"type": "command", "agent_cmd": "claude -p \"$(cat {prompt_file})\" --permission-mode acceptEdits", "harness_file": "CLAUDE.md"}
```
```bash
lamarck run -c .lamarck/lamarck.json
```
Each task starts from the parent of a real commit, brings in that commit's tests, and asks the agent to make them pass. Every rollout runs in a throwaway `git worktree`. **Agents execute commands, so run this on a sandboxed machine or container.**

## How a change earns its place
| Stage | Rule | Source |
|---|---|---|
| Critic (free) | Rejects edits that mention a task id or a task-unique literal (memorization); also enforces edit size, an annealed edit budget, and allowed surfaces only | RRSI critic; *Bad Genius* (CHASE) |
| Screen (k=2) | Net gain g = ΣΔₜ > 0; regression R on reliably-solved tasks ≤ δ; added tokens must be paid for by gain | DarwinX §2.1; RRSI cost rule |
| Confirm (k=4) | Gain must exceed the bootstrap noise band; regression re-checked. Only confirmed variants steer search | DarwinX two-speed promotion; RRSI noise floor |
| Merge | Complementary specialists are recombined, and the child is kept only if it covers the union of their wins | DarwinX §2.3 |
| Held-out gate | The pre-registered best variant ships only if its held-out gain clears held-out noise | every paper's overfitting lesson; *Rethinking…* |

## Layout
```
src/lamarck/
  harness.py      components + additive, mergeable edits
  tasks.py        tasks, verifiers, train/held-out split, task-specific literal detection
  evaluate.py     avg@k evaluation, bootstrap noise band
  miner.py        reliable / variance / wall partition, failure clustering, contrasts
  critic.py       leakage / size / budget / surface screen, annealed edit budget
  selection.py    preserve-and-extend, solved-set classes, screen/confirm, held-out gate
  archive.py      variant tree, parent selection, specialists
  evolve.py       the loop
  proposers/llm.py   Claude proposer (structured output)
  targets/anthropic_api.py   frozen Claude model with the harness as system prompt
  targets/command.py         any agent CLI in a git worktree
  gitmine.py      verified tasks from git history
  sim.py          simulated world for tests + demo
  report.py, cli.py
```

## Model notes
- The defaults use `claude-opus-5-5` for both proposer and target. Change `model` and `effort` in `lamarck.json`. Models without `effort` support (e.g. Haiku 4.5) need `"effort": null`.
- The **proposer** has server-side refusal fallbacks enabled (`server-side-fallback-2026-07-01`, `fallbacks: "default"`).
- The **target** never uses fallbacks. A silent model swap would invalidate the measurement, so a refusal is counted as a failed rollout.

## What's proven vs. not
- **Proven by the cited research:** harness edits on frozen models raise agent pass rates, and regression-gated loops compound.
- **Not yet proven:**
  - that git-mined tasks drive edits that generalize on real teams' repos
  - the cost per accepted improvement

  That is what the pilot phase must measure. See the kill criteria in `thesis.md`.

## License
Proprietary. All rights reserved. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
