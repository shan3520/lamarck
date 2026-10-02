# Lamarck

**An automatic coach for AI coding agents: it improves the agent's instruction file and proves each improvement on tasks the agent never practised on.**

Coding agents like Claude Code, Codex and Cursor read an instruction file before every task (`CLAUDE.md`, `AGENTS.md`, skill files). That file has a large effect on how well the agent works, and teams tune it by guesswork. Lamarck tunes it **with evidence**:

1. **Builds a practice exam from your git history.** Every past commit that made failing tests pass becomes a task, graded automatically by those tests.
2. **Watches the agent take it** and groups its mistakes.
3. **Has a second model propose a small, general fix** to the instruction file.
4. **Keeps the fix only if the score rises and nothing that already worked breaks.**
5. **Ships only if the improved file also wins on held-out tasks**, a slice of the exam the loop never sees.

The model itself is never retrained. What the agent learns is written into its instructions and inherited by every future run, which is why the project is named after Lamarck, who argued that acquired traits are inherited.

```
your git history ─► verified tasks (fail before the fix, pass after) ─► train / held-out split
                                                                         │
seed instructions ─► run agent ─► mine failures ─► propose small edit ─► critic ─► screen ─► confirm
       ▲                                                                                       │
       └──────────────── archive of variants · merge complementary specialists ◄───────────────┘
                                                                                               │
                              held-out gate: ship only if the gain clears the noise band ◄─────┘
```

> **Status: research prototype.** It works end to end on real models, with zero cost on free tiers. It is not yet proven on real teams' repositories; see [Results](#results) and [Limitations](#limitations).

---

## Contents
- [Glossary](#glossary)
- [Results](#results)
- [Quick start (free)](#quick-start-free)
- [Use it on your own repo](#use-it-on-your-own-repo)
- [How a change earns its place](#how-a-change-earns-its-place)
- [Long runs on free tiers: pause and resume](#long-runs-on-free-tiers-pause-and-resume)
- [Model providers](#model-providers)
- [Safety and integrity](#safety-and-integrity)
- [Limitations](#limitations)
- [Documentation map](#documentation-map)
- [Development](#development)

## Glossary
| Term | Meaning |
|---|---|
| **Harness** | Everything around the frozen model that Lamarck may change: `system.md` (core instructions), `rules.md` (bullet rules), `skills/*.md` (reusable procedures). Rendered into one instruction file. |
| **Target** | The agent being improved: a coding-agent CLI, Lamarck's built-in mini agent, or a single model call. |
| **Proposer** | The model that reads failures and suggests harness edits. |
| **Task** | A prompt plus a verifier (an automatic pass/fail check, such as a test command). |
| **Rollout** | One attempt by the target at one task. |
| **Train / held-out** | Train tasks drive the search. Held-out tasks are hidden from it and used once, to decide whether to ship. |
| **Score** | Partial credit: the fraction of a task's checks passed. **Full-pass rate** is the strict number. |

## Results
| Run | What it tested | Outcome | Write-up |
|---|---|---|---|
| Simulation (`lamarck demo`) | Every safeguard, deterministically | All safeguards behave as designed. Not evidence about real models. | `lamarck demo` |
| Support conventions, free Groq models | Can the loop discover unstated house rules? | **SHIP**: held-out score 39.7% → 96.0%, full-pass 0/7 → 6/7, $0. On an easy benchmark I wrote. | [2026-10-02-groq-full.md](docs/results/2026-10-02-groq-full.md) |
| Real repos (itsdangerous + python-slugify) | Does a real coding agent improve on real bug fixes? | **In progress**: a multi-day free run. The seed agent passed 2/10 attempts on day 1. | `examples/real-repo/` |

Earlier smoke run and the bugs it surfaced: [2026-10-02-free-smoke.md](docs/results/2026-10-02-free-smoke.md).

## Quick start (free)
Requires Python 3.10+ and git. Commands are shown for Windows (Git Bash or PowerShell); on macOS/Linux use `.venv/bin/` instead of `.venv/Scripts/`.

**1. Install**
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -e ".[llm,dev]"
```

**2. See the whole loop for free, no API key**
```bash
.venv/Scripts/lamarck demo
```
The report is written to `runs/demo/report.md`.

**3. Run it on real models for $0 with Groq's free tier.** Create a key at [console.groq.com/keys](https://console.groq.com/keys) and put it in a file named `.env` at the repo root (git-ignored):
```
GROQ_API_KEY=gsk_...
```
Then run:
```bash
.venv/Scripts/lamarck run -c examples/acme-support/lamarck.groq.json
```
This takes about 45 minutes and fits in one day's free quota. It has to discover 5 house conventions that the task prompts never state, and you get a report with a SHIP / DO NOT SHIP / INCONCLUSIVE verdict.

## Use it on your own repo
```bash
lamarck init .lamarck --from CLAUDE.md
```
```bash
lamarck mine-git --repo . --test-cmd "{python} -m pytest -q {tests}" --out .lamarck/tasks.mined.json
```
```bash
lamarck validate-tasks .lamarck/tasks.mined.json --out .lamarck/tasks.json --heldout 0.4
```
Then point the target at your agent in `.lamarck/lamarck.json`, using one of these:
```json
"target": {"type": "command", "harness_file": "CLAUDE.md", "agent_cmd": "claude -p \"$(cat {prompt_file})\" --permission-mode acceptEdits"}
```
```json
"target": {"type": "command", "harness_file": "AGENT.md", "agent_cmd": "{python} -m lamarck.agents.mini --model qwen/qwen3.8-27b --prompt-file {prompt_file}"}
```
```bash
lamarck run -c .lamarck/lamarck.json
```

- **`mine-git`** turns every commit that changed both source and tests into a task: start from the commit's parent, add the commit's tests, make them pass. The commit message is the issue text.
- **`validate-tasks`** keeps only tasks whose tests **fail before** the real fix and **pass after** it, the same check SWE-bench uses. In the itsdangerous repo, 11 of 20 mined tasks failed this check.
- The **mini agent** (`lamarck.agents.mini`) is a small, token-frugal, bash-only coding agent that runs on free Groq models. Its instruction file `AGENT.md` is the harness Lamarck evolves.
- A complete worked example is in [`examples/real-repo/`](examples/real-repo/). Run `examples/real-repo/setup.sh` to clone its repos.

## How a change earns its place
| Stage | Rule | Source |
|---|---|---|
| **Critic** (free, before any rollout) | Rejects edits that mention a task id or a literal unique to one task (memorization); limits edit size, edits per proposal (annealed over time) and which files may be edited | RRSI critic; *Bad Genius* |
| **Screen** (few rollouts) | Net gain > 0; no more than δ loss on tasks the parent reliably solved; added tokens must be paid for by gain | DarwinX preserve-and-extend; RRSI cost rule |
| **Confirm** (more rollouts) | Gain must exceed the bootstrap noise band. Only confirmed variants may steer the search | DarwinX two-speed promotion; RRSI noise floor |
| **Merge** | Variants that solve different tasks are combined; the child is kept only if it keeps both parents' wins | DarwinX recombination |
| **Held-out gate** | The best variant ships only if its held-out gain clears held-out noise. Fewer than 20 held-out rollouts → **INCONCLUSIVE**, never SHIP | Every paper's overfitting lesson |

Each run writes `report.md`, which contains the verdict, the train vs held-out gap, every accepted edit with its rationale, and the full harness diff. Details: [docs/architecture.md](docs/architecture.md).

## Long runs on free tiers: pause and resume
Free tiers cap tokens per day: Groq allows about 200K tokens per model per day, and a coding-agent attempt costs about 15–20K. Lamarck saves **every finished rollout and every proposal** in the run directory. When a daily quota runs out, the run **pauses** with exit code 4 and prints the resume command:
```bash
lamarck run --resume examples/real-repo/runs/run-1 --yes
```
Resuming replays the search from disk at zero cost and continues exactly where it stopped. A test checks that a paused-and-resumed run ends identical to an uninterrupted one.

## Model providers
| Provider | Set in `.env` | Notes |
|---|---|---|
| **Groq** (free tier) | `GROQ_API_KEY` | Recommended free route. Set `"provider": "groq"` on an `llm` target or proposer. Per-minute limits are waited out automatically; daily limits pause the run. |
| **Anthropic** | `ANTHROPIC_API_KEY` (or `ant auth login`) | The default (`claude-opus-5-5`). The proposer gets server-side refusal fallbacks; the target never does, because a silent model swap would invalidate the measurement. |
| **OpenRouter** | `OPENROUTER_API_KEY` | Routed through OpenRouter's Anthropic-compatible endpoint. `:free` models allow 50 requests/day (1,000/day after a one-time $10 credit purchase). |

Free providers may log prompts. Don't send private code to free tiers. Full config reference: [docs/reference.md](docs/reference.md).

## Safety and integrity
- **No peeking at the answer:** each rollout runs in a fresh **snapshot** of the repo at the task's base commit, with **no `.git` directory**, so the agent cannot look up the fixing commit. A test checks this.
- **Agents run real commands.** Use a sandboxed machine or container for repositories you don't trust.
- **Secrets** live only in `.env`, which is git-ignored. Keys are never written to configs, logs or reports.
- **Spend:** `run` asks before calling paid models, and `max_total_tokens` / `max_rollouts` cap a run.

## Limitations
- **Not yet proven on real teams' repositories.** The real-repo run is the first test of that.
- **The support benchmark is easier than real work**: its failure messages name the broken rule.
- **Memorization risk:** older commits in popular libraries may already be known to the model. Recent commits (e.g. python-slugify, Sept 2026) avoid this.
- **Small samples:** verdicts rest on tens of rollouts. The noise band and the 20-rollout minimum guard against over-claiming, but they don't replace larger studies.
- **Harness bloat:** accepted rules can be redundant. A pruning step is planned.

## Documentation map
| Document | What's in it |
|---|---|
| [docs/reference.md](docs/reference.md) | Every CLI command and option, every `lamarck.json` setting with defaults, run outputs, exit codes |
| [docs/architecture.md](docs/architecture.md) | How the engine works, module by module, plus the design decisions and why |
| [docs/research/brief.md](docs/research/brief.md) | The research survey this is built on: 5 papers, agreements, disagreements, GitHub check |
| [docs/research/thesis.md](docs/research/thesis.md) | Product thesis, competition, kill criteria |
| [docs/results/](docs/results/) | Write-ups of every real run, including failures and caveats |
| [NOTICE.md](NOTICE.md) | Credits for the research methods (clean-room implementation) |

## Development
```bash
.venv/Scripts/python -m pytest -q
```
77 tests, all offline. Tests never see real credentials or `.env`; a fixture isolates them. The simulated world in `src/lamarck/sim.py` exercises the full loop deterministically.

## License
Proprietary. All rights reserved. See [LICENSE](LICENSE). Research credits are in [NOTICE.md](NOTICE.md).
