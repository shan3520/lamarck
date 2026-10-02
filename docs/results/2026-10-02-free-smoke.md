# First real run: free models, smoke-sized (2026-10-02)

**Setup**
- Suite: `examples/acme-support/lamarck.smoke.json`, with 4 train and 3 held-out tasks.
- Target (frozen): `qwen/qwen3.8-27b:free`.
- Proposer: `nvidia/nemotron-3-ultra-550b-a55b:free`.
- Access: OpenRouter free tier.
- Limits: 1 iteration; train k=2; held-out k=1; 22 rollouts, 13.5k tokens, $0.

**What happened**
- The seed harness was one sentence. With nothing in the prompts saying so, the loop found 3 of the 5 hidden conventions from failure traces alone: the exact sign-off, the `Ticket #12345` format, and the `USD 0.00` amounts. It wrote all three as **generic** rules, so the leakage critic had nothing to block.
- Train score went from 17.7% to 68.8% (partial credit). The edit was confirmed above noise with no regression.
- Held-out full-pass rate went from 0/3 to 1/3.

**Verdict: inconclusive.** Three held-out rollouts can't support a claim. At the time, the gate printed "SHIP" because with k=1 the noise estimate came out as 0. That was a bug and is now fixed: the noise estimate falls back to resampling tasks, and any result with fewer than 20 held-out rollouts is reported as INCONCLUSIVE.

**Bugs this run surfaced (all fixed)**
1. **No partial credit.** Tasks require several conventions at once, so binary pass/fail made single fixes invisible, and the first run scored g=0 for a correct edit. Fix: `all` verifiers now give partial credit and list every failing rule.
2. **Edit budget collapsed to 1 when `iterations=1`.** Fix: the budget annealing now starts at full budget for single-iteration runs.
3. **Empty responses from free gateway models crashed the proposer.** Fix: retry once, then skip.
4. **The noise estimate read 0 at k=1.** Fix: task-level bootstrap fallback, plus the INCONCLUSIVE verdict.

**Next:** the full suite (`lamarck.free.json`, about 450 requests). That needs OpenRouter's 1,000/day free tier, which requires a one-time purchase of at least $10 in credits.
