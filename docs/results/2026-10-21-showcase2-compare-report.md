# Agent change test (agent agent_cmd and agent timeout): BETTER

**candidate** vs **base** on 16 verified tasks, 4 attempts each per version.

| Setting | Base | Candidate |
|---|---|---|
| instructions | `harness (1442b1034d76)` | `harness (1442b1034d76)` |
| agent agent_cmd (changed) | `{python} -m lamarck.agents.mini --provider nvidia --model nvidia/nemotron-3-ultra-550b-a55b --prompt-file {prompt_file} --max-steps 25 --max-tokens-total 80000 --rpm 9` | `{python} -m lamarck.agents.mini --provider nvidia --model nvidia/nemotron-3-ultra-550b-a55b --prompt-file {prompt_file} --max-steps 50 --keep 8 --obs-chars 4000 --max-tokens-total 400000 --rpm 9` |
| agent harness_file | `AGENT.md` | `AGENT.md` |
| agent timeout (changed) | `2400` | `7200` |
| agent type | `command` | `command` |

| | Base | Candidate |
|---|---|---|
| Mean score | 28.1% | 68.8% |
| Full-pass rate | 28.1% | 68.8% |
| Tokens per attempt | 44,755 | 134,560 (+201%) |

**Verdict: BETTER.** +40.6 pts clears the noise band of 4.7 pts.
With this many tasks and attempts, differences smaller than about **4.7 pts** can't be told apart from luck.

## Per task

| Task | Base | Candidate |
|---|---|---|
| git-71b46b06fb | 3/4 | 2/4 |
| git-def2dabea8 | 2/4 | 4/4 |
| git-6c58e969a5 | 0/4 | 2/4 |
| git-670b715b30 | 0/4 | 0/4 |
| git-5bd5c263f9 | 0/4 | 0/4 |
| git-edb9970d02 | 1/4 | 4/4 |
| git-5e6ae76a61 | 3/4 | 4/4 |
| git-a16e9c1c9b | 0/4 | 4/4 |
| git-f51a53bfd2 | 1/4 | 4/4 |
| git-6887a7bd5a | 0/4 | 0/4 |
| git-0e6acdf9b6 | 2/4 | 4/4 |
| git-f89d7a3331 | 0/4 | 4/4 |
| git-fb33cf2c24 | 0/4 | 1/4 |
| git-2f23599bfb | 2/4 | 4/4 |
| git-619ab9ce9a | 4/4 | 3/4 |
| git-069b30002a | 0/4 | 4/4 |
