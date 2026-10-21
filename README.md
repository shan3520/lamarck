# Lamarck

**A/B tests for your AI coding agent: its instructions, its settings and its model.**

Coding agents like Claude Code, Codex and Cursor read an instruction file before every task (`CLAUDE.md`, `AGENTS.md`, skill files). Teams edit that file all the time, by gut feeling. A new rule *sounds* helpful, so it ships.

Lamarck tests the change instead. It runs your agent with the old and new instructions on **verified tasks from your own repository's history** and tells you whether the change helped, hurt, or made no difference you can measure:

```
lamarck compare --base CLAUDE.md --candidate CLAUDE.new.md

Score: 49.2% -> 41.7%  (-7.5 pts, noise 5.4 pts)
Verdict: WORSE (-7.5 pts is below minus the noise band of 5.4 pts)
```
*(Numbers from a real re-test in our experiments: 10 never-seen tasks, 12 attempts per task and version.)*

> **Status: research prototype.** The engine is complete and tested (138 tests). The measurements below come from free models on open-source repositories, not yet from customer pilots.

---

## Why test instruction changes
In our own experiments on real repositories, **plausible, generic instruction changes often made the agent worse**. Small tests were also easy to fool:

| Change (found by Lamarck's own search) | Looked like | Measured on tasks it wasn't tuned on |
|---|---|---|
| "Re-run the tests after every edit; finish only when they pass" | **+12.5 pts** | **−5.0 pts** |
| "After any edit, run an import check; fix or revert on failure" | **+7.8 pts**, then +5.0 pts on 4 attempts per task | **−7.5 pts** on 12 attempts per task (pre-registered re-test) |

The second change used up the agent's step budget on extra checks, so more attempts ran out of steps before finishing. Nobody would have caught that by reading the rule. The full story, including what went wrong in our own tooling, is in [docs/results/2026-10-08-real-repo.md](docs/results/2026-10-08-real-repo.md).

The agent's **settings** told a different story. A pre-registered `compare` of the same model and instructions, with a bigger step budget, more memory and more command output, on 16 randomly chosen real tasks:

| | Default settings | Roomy settings |
|---|---|---|
| Tasks fully solved | 28.1% | **68.8%** (+40.6 pts, noise ±4.7 pts) → **BETTER** |
| Tokens per attempt | 45k | 135k (+201%); per *solved* task +23% |

That's a large, real effect, with a cost you can weigh. [Write-up](docs/results/2026-10-21-showcase-agent-settings.md) · [report](docs/results/2026-10-21-showcase2-compare-report.md).

## How it works
1. **Tasks from your git history** (`mine-git`). Every past commit that changed code *and* tests becomes a task: start from the commit's parent, make its tests pass. The commit message is the issue text.
2. **Only real tasks** (`validate-tasks`). A task is kept only if its tests **fail before** the real fix and **pass after** it, the same check SWE-bench uses.
3. **Only informative tasks** (`calibrate`). Your agent tries every task. Tasks it never solves or always solves score the same under any instructions, so they're dropped. In our runs, only 15–35% of real tasks were informative.
4. **The A/B test** (`compare`). Both versions run on the same tasks, several attempts each. A difference counts only if it clears the **bootstrap noise band**, and the report states the smallest difference the test *could* have detected.
5. **No re-rolling.** The plan (tasks, attempts, rule) is written to disk *before* anything runs. Re-running with a different plan into the same folder is refused, so a test can't be repeated until it happens to pass.

Each run starts from a fresh snapshot of the repository at the task's base commit, **with no `.git` directory**, so the agent can't look up the real fix.

## Use it on your repo
```bash
lamarck init .lamarck --from CLAUDE.md
```
```bash
lamarck mine-git --repo . --test-cmd "{python} -m pytest -q {tests}" --out .lamarck/tasks.mined.json
```
```bash
lamarck validate-tasks .lamarck/tasks.mined.json --out .lamarck/tasks.json
```
Point the target at your agent in `.lamarck/lamarck.json`:
```json
"target": {"type": "command", "harness_file": "CLAUDE.md", "timeout": 1800,
           "agent_cmd": "claude -p --model sonnet --output-format json --permission-mode acceptEdits --allowedTools Bash PowerShell --strict-mcp-config --no-session-persistence < {prompt_file}"}
```
This works on Windows, macOS and Linux. Notes on the Claude Code target:
- It uses your normal Claude Code login, so run `claude` once and log in first. Usage counts against your plan or API key.
- Lamarck reads its token use from the JSON output. An API problem (not logged in, rate limit, outage) pauses the run; it's never scored as the agent failing.
- `--no-session-persistence` keeps test runs out of your Claude Code history.
- Your personal Claude Code settings and instructions still apply, equally to both versions in a comparison.
- **Gateways such as OmniRoute:** Lamarck records which model answered each attempt. If the models behind one side change during a test, for example because a gateway fell back to another model on a rate limit, the verdict is **INCONCLUSIVE**. Pin one model, with fallback off, for testing.
- Claude Code's file tools stay inside the task folder, but shell commands are not sandboxed. Use a container for repositories you don't trust.
Keep only informative tasks, then set `"tasks": "tasks.calibrated.json"` in the config:
```bash
lamarck calibrate -c .lamarck/lamarck.json --wait
```
Test a change:
```bash
lamarck compare -c .lamarck/lamarck.json --base CLAUDE.md --candidate CLAUDE.new.md --wait
```
The verdict and a per-task table are written to `compare_report.md`.

**Test agent settings or a new model, not just instructions.** In our experiments, the agent's settings (step budget, memory, how much command output it sees) mattered more than its instruction file. Give each variant as a JSON agent spec:
```bash
lamarck compare -c .lamarck/lamarck.json --base-agent agents/current.json --candidate-agent agents/more-steps.json --wait
```
The same works for a model migration: point `--candidate-agent` at a spec that uses the new model. The report shows exactly which settings changed. Ready-made examples are in [`examples/real-repo/agents/`](examples/real-repo/agents/).

**In CI**, the exit code is the verdict: `0` better, `2` no clear difference, `3` worse. You can block a pull request that makes the agent measurably worse.

**How many attempts?** `-k 4` (the default) on 10–20 informative tasks detects differences of roughly 8–12 points. Smaller real effects need more tasks or attempts. The report says how small a difference your test could have caught, so "no clear difference" is never mistaken for "no difference".

## Experimental: let Lamarck suggest changes
`lamarck run` searches for instruction changes automatically. It mines the agent's failures, has a second model propose small general edits, and keeps only edits that survive staged testing and a final check on held-out tasks (DarwinX- and RRSI-style selection, see [docs/architecture.md](docs/architecture.md)). `lamarck confirm` re-tests a run's verdict once, with a plan registered in advance.

**Honest status:** on a benchmark of unstated house conventions it improved held-out scores from 39.7% to 96.0% ([write-up](docs/results/2026-10-02-groq-full.md)). On real repositories with free models, **no suggested change has survived confirmation** (8 runs). The safeguards rejected every false improvement. Treat suggestions as hypotheses to test with `compare`, not as fixes.

## What we learned building it
- **Most real tasks are uninformative for a given agent.** With Nemotron-3-Super, 60 of 74 validated tasks were never solved and 3 were always solved, at first. Calibration is not optional.
- **The agent's settings mattered more than its instructions.** Giving the same model 50 steps instead of 25, longer command output and more memory took it from 1 of 10 hard tasks to 9 of 9 (DeepSeek V4.1 Flash), and a pre-registered `compare` with Nemotron-3-Ultra measured 28% → 69%. That's why `compare` tests agent settings and models too.
- **Small samples lie.** 4 attempts per task said +5 points; 12 said −7.5.
- **Long free-tier runs need plumbing.** Lamarck now waits out quotas, throttles and network drops, keeps the PC from idle-sleeping while it runs, and resumes after a shutdown without redoing finished work.

## Quick start (free)
Requires Python 3.10+ and git. Commands are shown for Windows; on macOS/Linux use `.venv/bin/` instead of `.venv/Scripts/`.
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -e ".[llm,dev]"
```
See the whole engine work offline, no API key needed:
```bash
.venv/Scripts/lamarck demo
```
A complete real-repository example (task pool, calibrated tasks, configs for free providers) is in [`examples/real-repo/`](examples/real-repo/). Run `examples/real-repo/setup.sh` to clone its repositories.

## Model providers
Keys go in a git-ignored `.env` file, never in configs or chat.

| Provider | Key in `.env` | Notes |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | The agent under test never gets model fallbacks: a silent model swap would invalidate the measurement. |
| Groq, Gemini, NVIDIA, Mistral, Token Harbor | `GROQ_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `MISTRAL_API_KEY`, `TOKENHARBOR_API_KEY` | OpenAI-compatible; free tiers work, with daily or per-minute limits that Lamarck waits out (`--wait`). |
| OpenRouter | `OPENROUTER_API_KEY` | Anthropic-compatible endpoint. |

Free tiers may log prompts. Don't send private code to them.

## Safety and integrity
- **No peeking at the answer:** fresh repository snapshots with no `.git`. Lamarck's built-in agent also blocks commands that reach outside the task folder. Third-party agents need their own sandbox settings: OpenCode, for example, read the original repository on disk until it was configured to deny outside folders.
- **Technical failures are not task failures.** Quota limits, network outages and provider hangs pause the run or are recorded as infrastructure errors; they're never scored as the agent failing.
- **Pre-registration:** `compare` and `confirm` record their plan before running and refuse to change it.
- **Secrets** stay in `.env` (git-ignored) and never reach logs or reports.

## Limitations
- **No customer pilots yet.** Every number above comes from free models on open-source Python libraries.
- **Python test suites only, so far.** Mining and validation assume a test command that names test files.
- **Cost and time:** an A/B test on 15 tasks × 4 attempts × 2 versions is 120 agent runs. That's minutes to hours with a paid API, and days on free tiers.
- **Memorization:** popular libraries' older commits may already be known to the model. Prefer recent commits.

## Documentation map
| Document | What's in it |
|---|---|
| [docs/reference.md](docs/reference.md) | Every command and option, config settings, outputs, exit codes |
| [docs/architecture.md](docs/architecture.md) | How the engine works and why |
| [docs/results/](docs/results/) | Every real run, including failures, caveats and pre-registered plans |
| [docs/research/](docs/research/) | The research survey and product thesis this started from |
| [NOTICE.md](NOTICE.md) | Credits for the research methods (clean-room implementation) |

## Development
```bash
.venv/Scripts/python -m pytest -q
```
138 tests, all offline. Tests never see real credentials or `.env`. The simulated world in `src/lamarck/sim.py` exercises the engine deterministically.

## License
[Apache License 2.0](LICENSE). You may use, modify and distribute Lamarck, including commercially, under its terms. Copyright and research credits are in [NOTICE.md](NOTICE.md).
