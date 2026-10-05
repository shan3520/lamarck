# Lamarck reference

## CLI

### `lamarck demo`
Runs the full loop in a simulated world: no API key, no cost. Not evidence about real models.

| Option | Default | Meaning |
|---|---|---|
| `--iterations` | 10 | Search iterations |
| `--seed` | 0 | Random seed |
| `--out` | `runs/demo` | Run directory |

### `lamarck init DIR`
Creates `DIR/lamarck.json` and `DIR/harness/`.

| Option | Meaning |
|---|---|
| `--from FILE` | Seed the harness from an existing `CLAUDE.md` / `AGENTS.md` |

### `lamarck mine-git`
Turns commits that changed both source and tests into tasks.

| Option | Default | Meaning |
|---|---|---|
| `--repo` | `.` | Repository to mine |
| `--test-cmd` | `{python} -m pytest -q {tests}` | Verifier command. `{tests}` is replaced by the task's test files; `{python}` by the interpreter Lamarck runs in |
| `--out` | `tasks.json` | Output task file |
| `--max-commits` | 300 | How far back to look |
| `--max-tasks` | 60 | Stop after this many tasks |
| `--heldout` | 0.3 | Fraction of tasks held out (deterministic, hash-based) |

Skipped automatically: merge commits, root commits, commits touching more than 15 files, commits with no added or modified test file.

### `lamarck validate-tasks TASKS --out FILE`
Keeps only tasks whose tests **fail** on the base snapshot (with the new tests written in) and **pass** at the fixing commit. Prints `OK` / `DROP` and the reason for each task.

| Option | Default | Meaning |
|---|---|---|
| `--max-seconds` | 120 | Drop tasks whose tests are slower than this |
| `--heldout` | (keep) | Re-assign the train/held-out split with this held-out fraction |

### `lamarck run`
Evolves the harness.

| Option | Meaning |
|---|---|
| `-c, --config` | Config file (default `lamarck.json`) |
| `--out` | Run directory (default `runs/<timestamp>` next to the config) |
| `--resume RUN_DIR` | Continue a paused run. Config and output come from `RUN_DIR/run_meta.json` |
| `-y, --yes` | Don't ask for confirmation before calling models |
| `--wait` | When a rolling daily quota is reached, wait until it frees up and continue, instead of pausing |
| `--max-wait` | With `--wait`: longest single wait, in hours (default 12) |

`.env` is loaded from the config's folder and the current folder. Variables already set in the environment take precedence.

### Exit codes (`run`)
| Code | Meaning |
|---|---|
| 0 | **SHIP**: a variant beat the seed on held-out tasks by more than the noise band |
| 1 | Aborted at the confirmation prompt |
| 2 | Not shipped: DO NOT SHIP, INCONCLUSIVE, or no variant beat the seed |
| 4 | **Paused**: out of credits/quota, or an authentication error. Progress is saved; fix the cause, then `--resume` |

## `lamarck.json`
```json
{
  "harness_dir": "harness",
  "tasks": "tasks.json",
  "requests_per_minute": 20,
  "target":   {"type": "command", "harness_file": "AGENT.md", "agent_cmd": "{python} -m lamarck.agents.mini --model qwen/qwen3.8-27b --prompt-file {prompt_file}"},
  "proposer": {"type": "llm", "provider": "groq", "model": "openai/gpt-oss-120b", "effort": "medium", "n_proposals": 1},
  "evolve":   {"iterations": 3, "proposals_per_iter": 1, "k_screen": 1, "k_confirm": 2, "k_heldout": 2, "workers": 1},
  "gate":     {}
}
```
Paths are relative to the config file.

### Top level
| Key | Default | Meaning |
|---|---|---|
| `harness_dir` | `harness` | Folder of `.md` components: `system.md`, `rules.md`, `skills/*.md` |
| `tasks` | `tasks.json` | Task file. Relative `meta.repo` paths inside it resolve against its folder |
| `requests_per_minute` | none | Client-side rate limit for LLM calls in this process |

### `target`
| `type` | Fields |
|---|---|
| `llm` (alias `anthropic`) | `provider` (`anthropic` \| `groq`, default `anthropic`), `model`, `effort` (`low`/`medium`/`high`, or `null` for models without effort), `max_tokens` (default 4000). The harness becomes the system prompt and the task the user message: one call per rollout. |
| `command` | `agent_cmd` (template with `{prompt_file}`, `{workdir}`, `{python}`), `harness_file` (default `CLAUDE.md`), `timeout` seconds (default 1800). Runs a coding agent in a snapshot; see the agent protocol below. |
| `simulated` | none. For tests and demos. |

### `proposer`
| `type` | Fields |
|---|---|
| `llm` (alias `anthropic`) | `provider`, `model`, `effort` (default `high`), `n_proposals` (default 3) |
| `scripted` | none. For tests and demos. |

### `evolve`
| Key | Default | Meaning |
|---|---|---|
| `iterations` | 8 | Search iterations |
| `proposals_per_iter` | 3 | Proposals evaluated per iteration |
| `k_screen` | 2 | Rollouts per train task for the cheap screen |
| `k_confirm` | 4 | Total rollouts per train task after confirmation (must exceed `k_screen`) |
| `k_heldout` | 4 | Rollouts per held-out task, for the seed and for the best variant |
| `beta` | 0.3 | Chance of picking a random archive node instead of the best lineage |
| `merge_every` | 3 | Try merging complementary variants every N iterations (0 = never) |
| `budget_start`, `budget_end` | 3, 1 | Edits allowed per proposal, annealed from start to end |
| `workers` | 4 | Parallel rollouts. Use 1 for coding agents on free tiers |
| `seed` | 0 | Random seed |
| `max_total_tokens` | none | Stop searching after this many target tokens |
| `max_rollouts` | none | Stop searching early enough to keep the held-out check within this many rollouts |

### `gate`
| Key | Default | Meaning |
|---|---|---|
| `max_regression` | 1.0 | δ: total score loss tolerated on tasks the parent reliably solved (task units) |
| `preserve_threshold` | 0.8 | A parent "reliably solves" a task at or above this score |
| `noise_mult` | 1.0 | A gain must exceed `noise_mult` × the bootstrap noise band |
| `tokens_per_point` | 4000 | Cost rule: added tokens per rollout allowed per +1 task of gain |
| `solved_threshold` | 0.5 | Score at which a task counts as solved (for solved-set classes) |
| `min_heldout_rollouts` | 20 | Below this many held-out rollouts the verdict is INCONCLUSIVE |

## Task file
```json
{"tasks": [{
  "id": "git-2f23599bfb",
  "prompt": "Fix add_uppercase_char ...\nRun them with: {python} -m pytest -q test_x.py",
  "verifier": {"type": "command", "cmd": "{python} -m pytest -q test_x.py"},
  "split": "train",
  "tags": ["git"],
  "meta": {"repo": "repos/python-slugify", "base": "<parent sha>", "commit": "<fix sha>", "test_files": ["test_x.py"]}
}]}
```
`split` is `train` or `heldout`. Git-mined tasks need `meta.repo`, `meta.base` and `meta.commit`.

### Verifier types
| `type` | Fields | Passes when |
|---|---|---|
| `exact` | `value` | Output equals `value` (trimmed) |
| `contains` | `values`, `label`? | Every string appears in the output |
| `not_contains` | `values`, `label`? | No string appears |
| `regex` | `pattern`, `label`? | The pattern matches (multiline) |
| `json` | `strict`?, `required_keys`?, `equals`?, `matches`?, `labels`? | The output parses as a JSON object meeting the constraints. `strict` forbids prose and code fences |
| `command` | `cmd`, `timeout`? | The command exits 0 in the rollout's working directory |
| `all` | `checks` | All sub-checks pass. Gives **partial credit** and lists every failing check |

`label` sets the failure message the proposer sees. Write it the way your test output would describe the failure.

## Run directory
| File | Contents |
|---|---|
| `report.md` | Verdict, train vs held-out table, accepted edits and rationale, search summary, harness diff |
| `summary.json` | Machine-readable verdict and numbers |
| `best_harness/`, `best_harness.rendered.md` | The winning harness, as components and as the rendered instruction file |
| `archive.json` | Every variant tried: lineage, edits, scores, verdicts |
| `log.txt` | The run log |
| `trials.jsonl`, `proposals.jsonl` | Saved rollouts and proposals (used by `--resume`) |
| `run_meta.json` | Which config the run used |

## Agent protocol for `command` targets
- Lamarck writes the rendered harness to `harness_file` in the working directory, and the task prompt to `{prompt_file}`.
- The working directory is a snapshot **without `.git`**.
- After the agent exits, the task's verifier runs in the same directory.
- Optional: print `LAMARCK_TOKENS=<n>` to report token use.
- Optional: exit with code **75** when the model provider's daily quota is exhausted; the run then pauses instead of recording failures.
- Subprocesses get UTF-8 I/O and colorless output (`NO_COLOR=1`, `PY_COLORS=0`).

### Built-in mini agent: `python -m lamarck.agents.mini`
| Option | Default | Meaning |
|---|---|---|
| `--model` | required | e.g. `qwen/qwen3.8-27b` |
| `--provider` | `groq` | `groq` or `anthropic` |
| `--prompt-file` | required | Task prompt |
| `--harness` | `AGENT.md` | Instruction file read as the system prompt |
| `--max-steps` | 12 | Commands per attempt |
| `--max-tokens-total` | 30000 | Token cap per attempt |
| `--obs-chars` | 1500 | Command output kept per step (head + tail) |
| `--keep` | 4 | Recent steps kept in full; older ones are summarized |
| `--quota-wait` | 600 | Seconds per attempt the agent may spend waiting out a rolling daily quota (Groq says "try again in 2m51s") instead of abandoning the attempt |
| `--cmd-timeout` | 120 | Seconds per command; on timeout the whole process tree is stopped |

Commands that reach outside the working directory (a bare `/`, absolute paths other than `/dev/null` and `/tmp`, `~`, drive paths like `C:\`, or `../..`) are blocked with exit code 126 and a message to the agent. The interpreter path from `{python}` is allowed.

## Environment variables
| Variable | Used for |
|---|---|
| `GROQ_API_KEY` | Groq provider |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` | Anthropic provider |
| `OPENROUTER_API_KEY` | Routes Anthropic-SDK calls through OpenRouter. Overrides an inherited default Anthropic base URL |
| `LAMARCK_BASH` | Path to the bash the mini agent uses. On Windows it otherwise finds Git Bash and never uses WSL |
