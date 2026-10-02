# Examples

| Folder / config | What it is | Cost |
|---|---|---|
| `acme-support/lamarck.groq.json` | 22 support tasks with 5 unstated house conventions; frozen `gpt-oss-20b`, proposer `gpt-oss-120b` | Free (Groq), about 45 min |
| `acme-support/lamarck.free.json` | The same suite on OpenRouter `:free` models | Free, but needs the 1,000/day OpenRouter tier |
| `acme-support/lamarck.smoke.json` | A 7-task subset for a quick end-to-end check | Free, about 25 requests |
| `acme-support/lamarck.json` | The same suite on Claude Opus 5.5 | Paid |
| `real-repo/` | 13 validated bug-fix tasks from itsdangerous and python-slugify; built-in mini agent on `qwen/qwen3.8-27b` | Free (Groq), multi-day; uses pause/resume |

## real-repo
```bash
bash examples/real-repo/setup.sh
```
```bash
.venv/Scripts/python -m pip install freezegun text-unidecode
```
```bash
.venv/Scripts/lamarck run -c examples/real-repo/lamarck.json --out examples/real-repo/runs/run-1
```
When Groq's daily quota runs out, the run pauses (exit code 4). Continue the next day with:
```bash
.venv/Scripts/lamarck run --resume examples/real-repo/runs/run-1 --yes
```
Task provenance: mined with `lamarck mine-git` and filtered with `lamarck validate-tasks`. A 1,000-line refactor commit was removed by hand as unrealistic for a focused agent.
