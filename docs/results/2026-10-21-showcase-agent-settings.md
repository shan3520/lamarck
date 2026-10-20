# Showcase: A/B test of agent settings (2026-10-20 to 10-21)

**Question:** does giving a coding agent more room (a bigger step budget, more memory of earlier steps, more command output) make it solve more real tasks, and at what cost?

**Answer: yes, by a wide margin.** With Nemotron-3-Ultra and Lamarck's mini agent, the roomy settings solved **68.8%** of attempts against **28.1%** for the defaults: **+40.6 pts against a noise band of 4.7 pts**. That's **BETTER**, at about 3× the tokens per attempt, or +23% tokens per *solved* task.

## Setup (pre-registered)
- **Tasks:** 16 tasks picked at random by ID hash, without looking at any past results, from a pool of 74 validated tasks (more-itertools 14, itsdangerous 1, python-slugify 1). They were not calibrated for either setting, so the test can't favor either side.
- **Variants:** the same model (`nvidia/nemotron-3-ultra-550b-a55b`, free tier) and the same instructions in both. Only the agent settings differ:

  | Setting | Default | Roomy |
  |---|---|---|
  | `--max-steps` | 25 | 50 |
  | Steps kept in memory (`--keep`) | 4 | 8 |
  | Characters of command output shown (`--obs-chars`) | 1,500 | 4,000 |
  | Token cap per attempt | 80k | 400k |
  | Timeout | 40 min | 2 h |

- **Design:** 4 attempts per task per variant (128 runs). The plan was committed before any run ([plan](2026-10-20-showcase2-compare-plan.json)).
- **Command:** `lamarck compare --base-agent agents/nemotron-ultra-default.json --candidate-agent agents/nemotron-ultra-roomy.json`

## Result
| | Default | Roomy |
|---|---|---|
| Full-pass rate | 28.1% | **68.8%** |
| Tokens per attempt | 44,755 | 134,560 (+201%) |
| Tokens per solved task | ~159k | ~196k (+23%) |
| Technical failures | 0 / 64 | 0 / 64 |

- **Spread across tasks:** 11 of 16 tasks improved, 2 got slightly worse, 3 were unchanged (2 of those never solved by either). Several tasks went from 0/4 to 4/4.
- **Full report:** [2026-10-21-showcase2-compare-report.md](2026-10-21-showcase2-compare-report.md).

## Showcase 1 was invalid, and what it fixed
The first attempt used `nemotron-3-super-120b-a12b`. Partway through, NVIDIA's free service started answering **HTTP 404 "Not Found"** for that model on this account: it had been removed. The default half had already run. **60 of the 64 roomy attempts crashed**, Lamarck counted those crashes as failures, and it reported a confident **WORSE** (14.1% → 4.7%). That verdict measured the outage, not the settings. Three fixes came from it:
1. `compare` reports **INCONCLUSIVE** when more than 10% of either side's attempts fail for technical reasons. Re-scored from its saved attempts, showcase 1 now reads "INCONCLUSIVE: too many technical failures (base 0/64, candidate 60/64)".
2. A provider "model not found" (404) pauses the test instead of counting as failures.
3. `--wait` only waits out problems that pass by themselves: rate limits, quotas, network drops. A missing model or a bad key pauses with an explanation instead of waiting forever.

Showcase 2 ran on the same 16 tasks with a model that was available, under a new pre-registered plan.

## What this shows, and what it doesn't
- **Shows:** `compare` works end to end on real tasks with a real agent. For this agent, the settings change is a large, clean effect: 9× the noise band, with no technical failures. Settings can matter far more than instruction edits: across 8 runs on the same repositories, no instruction change survived testing.
- **Doesn't show:** that the same holds for Claude Code or other production agents, which already have generous budgets. Nor is "more room" free: it tripled tokens per attempt. Whether +40 pts is worth 3× the cost is a business decision, and the report gives the numbers to make it.
