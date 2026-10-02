# How Lamarck works

## The loop
```
tasks ──► evaluate seed (avg@k) ──► archive root
             │
             ▼  for each iteration
   select parent ──► mine its failures ──► proposer: small edits ──► critic
   (best lineage, sometimes                                            │
    a random archive node)                                             ▼
                                                    screen (k_screen rollouts)
                                                               │ passes
                                                               ▼
                                                    confirm (k_confirm rollouts)
                                                               │
                                     confirmed ──► may steer the search
                                     promoted  ──► kept, may be picked when broadening
                                     reverted  ──► kept as a record only
             every merge_every iterations: try merging complementary specialists
             │
             ▼
   held-out gate: seed vs best variant on never-seen tasks ──► SHIP / DO NOT SHIP / INCONCLUSIVE
```

## Modules
| Module | Responsibility |
|---|---|
| `harness.py` | A harness is a set of markdown components. Edits are `add_skill`, `append_rule` (additive), `set_component`, `remove_component`. Additive edits from different lineages can be merged; conflicting rewrites raise `EditConflict`. |
| `tasks.py` | Tasks, verifiers (partial credit for `all`), the deterministic train/held-out split, detection of task-specific literals (for the critic), and the `{python}` placeholder. |
| `evaluate.py` | Runs k rollouts per task (`avg@k`) and estimates a bootstrap noise band. Infrastructure errors count as failures; fatal errors (out of credits, auth, daily quota) stop the run. `TrialCache` stores every rollout for resume. |
| `miner.py` | Splits tasks into reliable / variance / wall, clusters failures by a normalized signature, and pairs passing with failing rollouts of the same task. Raw traces are kept, because summarized feedback loses the signal (Meta-Harness). |
| `critic.py` | A cheap, deterministic screen before any rollout is spent: leakage, size, edit budget, allowed files. Plus the annealed edit budget. |
| `selection.py` | DarwinX preserve-and-extend: gain g = ΣΔ, regression R on the preservation set; solved-set classes; screen and confirm decisions; cost rule; held-out gate. |
| `archive.py` | Tree of every variant. Parent selection exploits the best confirmed lineage with probability 1−β and broadens otherwise. Finds specialists for merging. |
| `evolve.py` | The loop above. |
| `proposers/llm.py` | LLM proposer: sees the current harness, failure clusters with trace excerpts, and the history of tried hypotheses; returns JSON proposals. |
| `proposers/cache.py` | Saves proposals so a resumed run replays the same search. |
| `targets/anthropic_api.py` | `llm` target: one model call with the harness as the system prompt. |
| `targets/command.py` | `command` target: snapshot → harness file → agent → verifier. |
| `agents/mini.py` | Built-in bash-only coding agent for free tiers. |
| `chat_compat.py` | Minimal OpenAI-compatible client (Groq): `retry-after` handling, daily-limit detection, removal of reasoning text. |
| `llm.py`, `env.py` | Provider dispatch, model-id mapping, `.env` loading, rate limiting. |
| `gitmine.py` | Mining tasks from git history and fail-to-pass validation. |
| `report.py`, `cli.py` | Run reports and the command line. |
| `sim.py` | A deterministic simulated world (capabilities, interference, cheats, bloat) for tests and the demo. |

## Design decisions and why
**Noise-aware selection.** Agent outcomes are random: the same agent on the same task passes some runs and fails others. Every decision uses average scores over several rollouts, and a gain only counts if it beats a bootstrap noise estimate. With one rollout per task, the noise estimate falls back to resampling tasks, because zero within-task variance would wrongly read as zero noise. This bug was found in a real run.

**Two-speed promotion.** The screen is permissive, so the search keeps moving. Confirmation is strict, so lucky results can't steer it. This follows DarwinX.

**Preservation set.** A change may not lose more than δ on tasks the parent **reliably** solved. Dips on tasks it only sometimes solved are mostly noise, and they already count against the gain.

**Partial credit.** Real tasks often require several things at once. With binary pass/fail, fixing one of them looks like no progress at all; a correct fix scored 0 in the first real run. The `all` verifier scores the fraction of checks passed and lists every failing one.

**A held-out gate decided in advance.** Held-out tasks are never shown to the proposer and never used for selection. Only the single best variant, chosen on train tasks, is evaluated on them, once. This is the defence against the overfitting every paper in `docs/research/brief.md` reports.

**The critic before any spend.** Edits that name a task id, a ticket number, or a literal unique to one task are rejected before any rollout. This is RRSI's critic and the *Bad Genius* concern: a proposer will happily memorize the exam.

**Snapshots without git history.** In a git worktree, an agent can run `git show <fixing commit>` and copy the answer. Snapshots are created with `git archive` and contain no `.git`.

**Resume by replay.** Instead of serializing the evolver's internal state, every rollout and proposal is saved as it finishes. A resumed run repeats the deterministic search; finished work is answered from disk, and new work continues from where it stopped. That keeps resume logic out of the loop and makes "resumed = uninterrupted" testable.

**The target never uses model fallbacks.** If a refused request were silently re-run on another model, measurements would mix models. Fallbacks are allowed only for the proposer.

## Integrity checklist for a new task suite
1. `mine-git`, then `validate-tasks`: only fail-to-pass tasks remain.
2. Prefer commits newer than the target model's training data, to avoid memorization.
3. Drop huge refactors. A task should be a focused change.
4. Use at least 20 held-out rollouts (tasks × `k_heldout`), or the verdict will be INCONCLUSIVE.
5. Use `{python}` in commands, so tests run in the same environment as Lamarck.
