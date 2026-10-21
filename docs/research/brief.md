# Research Brief #01: Can AI agents actually improve themselves?

*Date: 2026-10-02 · Method: question first, then 5 papers, synthesis, GitHub check, and a full read only for papers with no code (Keshav three-pass + critical/creative reading)*

## The question
**"How do AI agents 'improve themselves' without retraining? Does it generalize or just overfit, and where are the limits?"**

**Why this topic.** As of today, Hugging Face's trending papers are dominated by "recursive self-improvement" (RSI) of agent harnesses: Raven 503 upvotes, NeoHorse-1 326, SkillOpt 263, RRSI 219, ModularRSI 214. Most people hear "RSI" and picture a model rewriting its own weights. That's not what is actually happening in these papers.

## TL;DR
- **Today, "self-improving AI" mostly means the model stays frozen while an agent rewrites its harness**: the prompts, skills, tools, memory, and control flow around the model.
- The loop is the same everywhere:
  1. run tasks
  2. read the traces
  3. propose a small edit
  4. keep the edit only if it passes a regression gate
- It works. Gains are +5 to +50 points on agent benchmarks.
- **The real fight is over generalization.** Gains measured on the tasks the loop evolved against shrink a lot on held-out tasks, and a matched-compute paper finds simple test-time search can match the gains.
- Whoever solves **trustworthy measurement** (verifiers, held-out gates, noise control) owns this layer.

---

## The 5 papers (first pass: the five Cs)

| # | Paper | Category | Context | Correctness | Contribution | Clarity | Code status |
|---|---|---|---|---|---|---|---|
| 1 | **Meta-Harness** ([arXiv:2603.28052](https://arxiv.org/abs/2603.28052)): Lee, Khattab, Finn et al. (Stanford), COLM 2026 | New system | GEPA, TTT-Discover, OpenEvolve, ACE | Mostly evolves and tests on the same benchmark; the math result tests on 5 held-out models | The proposer gets a *filesystem* of all prior harness code, traces, and scores (about 10M tokens/iter vs. GEPA's about 8K). Results: +7.7 pts on text classification with 4x fewer tokens; +4.7 pts on IMO-level math across 5 held-out models; #1 Haiku 4.5 agent on TerminalBench-2 | High | ✅ **Implemented**: [stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness), MIT, about 1.6k stars, pushed today |
| 2 | **Self-Harness** ([arXiv:2606.09498](https://arxiv.org/abs/2606.09498)): Zhang et al. (Shanghai AI Lab), Jun 2026 | New method | Meta-Harness, DGM | Held-out split is within the same benchmark | The model improves *its own* harness, with no stronger external agent. Loop: weakness mining, then minimal proposal, then regression-validated acceptance. All 9 model×benchmark combinations improved held-in **and** held-out, up to +132% relative (e.g. TB2 held-out 40.5% → 61.9%) | High | ✅ **Implemented**: [qzzqzzb/Self-Harness](https://github.com/qzzqzzb/Self-Harness), about 117 stars, ⚠️ **no license** (code cannot legally be reused) |
| 3 | **DarwinX** ([arXiv:2608.07545](https://arxiv.org/abs/2608.07545)): Salesforce AI Research, Jul 2026 | New method + system | DGM, HGM, SkillOpt, HarnessX | Strong held-out design, but individual operators aren't ablated; TerminalWorld has only 41 tasks (p=0.45) | Self-evolution as *population selection*: a **preserve-and-extend** contract, an archive of lineages, and **merging** of complementary specialists. Average gain about +17 pts. TB2.1 75.5 → 83.2%; WebArena-Infinity 43.5 → 93.0% (audit-clean); held-out TerminalWorld 25 → 28/41 | High | ❌ **No code.** It evolves Salesforce's *proprietary* agent "Monet". → **Deep-read below** |
| 4 | **RRSI** ([arXiv:2609.24972](https://arxiv.org/abs/2609.24972)): Google Research, Sep 2026, 219 HF upvotes | New method | Meta-Harness, regularization theory | Honest about overfitting; 8 benchmarks | Brings ML-style *regularization* to harness search: an annealed edit budget, a history-aware proposer, a critic that rejects benchmark-specific logic, a noise floor, a cost rule, and a pruner. +14.1 pts on the evolve split but only **up to +4.7 out of distribution**; 30% fewer tokens | High | ✅ **Implemented**: [google-research/rrsi](https://github.com/google-research/rrsi), Apache-2.0, about 1.2k stars, has tests |
| 5 | **Rethinking the Evaluation of Harness Evolution** ([arXiv:2607.12227](https://arxiv.org/abs/2607.12227)), Jul 2026 | Measurement / critique | All of the above | Matched-budget design is the right control | Under matched feedback and inference budgets, harness evolution **does not consistently beat simple test-time scaling** and shows **limited generalization** (TB2.1, GPT-5.4 and Opus 4.6) | Medium-high | ✅ **Implemented**: [rethinking-harness-evolution/code](https://github.com/rethinking-harness-evolution/code), about 40 stars, no license |

**Supporting papers** (read at abstract level):
- [Do Agent Optimizers Compound?](https://arxiv.org/abs/2607.14004): gains compound *only* when regression control sits inside the loop. GEPA transferred *below* baseline.
- [Do Self-Evolving Skills Generalize?](https://arxiv.org/abs/2609.39148): of 21 skills that improved on training tasks, only 5 kept all of the gain on test tasks.
- [RHO](https://arxiv.org/abs/2606.05922): label-free optimization from past trajectories; SWE-Bench Pro 59 → 78%.
- [HSI](https://arxiv.org/abs/2608.08466): no gain on tasks beyond what the backbone model can do.
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954): the ancestor; SWE-bench 20 → 50%.
- [Lil'Log: Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/): a survey. Cites Lin et al.: a 9B model *proposes* harness edits about as well as Opus.

> Note: the citation-graph tool returned no edges for these 2026 papers (too recent). The papers were chosen by relevance, rigor, HF signal, and spread of viewpoints. Every ID was confirmed in the Firecrawl research index.

---

## Where they agree
1. **The harness is a first-class lever.** With weights frozen, harness edits alone move agent benchmarks by large margins. *(All 5; even #5 sees in-distribution gains.)*
2. **Everyone runs the same inner loop**: batch rollouts, reflect on traces, propose a bounded edit, gate against a regression or held-out signal. DarwinX says this explicitly. *(#1–#4)*
3. **Raw traces beat summaries.** The proposer needs to see actual failures, not compressed feedback. *(#1 filesystem history; #2 weakness mining; #3 failure/teacher/self evidence)*
4. **Regression gating is non-negotiable.** *(#2 validation, #3 preservation, #4 selection; "Do Optimizers Compound" shows that without it, gains don't compound)*
5. **What evolves is mostly generic procedure, not domain knowledge.** Examples: *verification before finalizing*, explicit acceptance contracts, output truncation, history compaction, independent review. *(#3 §8; Self-Evolving Harness 2609.38372; #2 qualitative analysis)*

## Where they disagree
| Fault line | Side A | Side B |
|---|---|---|
| **Does it generalize?** | DarwinX: "general competence, not patches"; Self-Harness: held-out improves in 9/9 combinations | RRSI: +14.1 in-split shrinks to +4.7 OOD; Rethinking: "limited generalization"; Skills paper: only 5/21 skills keep all of their gain |
| **Is it the harness or just more compute?** | DarwinX §4.1: "the gain is the harness, not compute" (extra tokens go only to newly solved tasks) | Rethinking: matched-budget test-time scaling performs comparably |
| **How to search** | Meta-Harness: one proposer with *maximal* history | DarwinX: *population* + merge (though every merge was reverted on WebArena-Infinity); RRSI: one path, *heavily regularized* |
| **How much may the agent edit?** | RRSI / Meta-Harness: open edit space, constrain the *search* | Self-Harness: 9 fixed editable surfaces. Lilian Weng warns that permissions must live *outside* the loop |
| **Who proposes** | Self-Harness: the frozen model itself | Meta-Harness / RRSI: a strong separate proposer (Opus-class) |

## What's genuinely new vs repackaged
- **New idea:** Meta-Harness's *uncompressed filesystem history* as proposer context (about 1000x more context per iteration than GEPA).
- **New selection rule:** DarwinX's *preserve-and-extend* contract plus two-speed promotion: permissive about *trying* an edit, strict about *trusting* it. Archives themselves are not new (DGM); the rule is.
- **New framing:** Self-Harness. Harnesses are *model-specific*, and a model can write its own without a stronger teacher.
- **Unifying framework:** RRSI imports classic regularization (budgets, pruning, noise floors) into harness search. The individual pieces are simple; the framing is the contribution.
- **Methodological contribution:** Rethinking's matched-budget baselines. These should become the standard.
- **Repackaged:** the reflect-and-edit inner loop itself (GEPA, Reflexion, PromptBreeder lineage).

---

## GitHub check result
| Paper | Status | Implication |
|---|---|---|
| Meta-Harness | ✅ Real, MIT, active, ecosystem (incl. [SuperagenticAI/metaharness](https://github.com/SuperagenticAI/metaharness) CLI) | No read needed; README + repo is the explanation |
| Self-Harness | ✅ Real, but **no license** | Ideas usable; code is not reusable |
| DarwinX | ❌ **None** (proprietary agent) | **Full read done ↓**. Also the clearest *productization gap* |
| RRSI | ✅ Real, Apache-2.0, method-to-code table in README | No read needed; Apache-2.0 means ideas *and* code are reusable with attribution |
| Rethinking | ✅ Real, no license | Abstract is enough; use as our evaluation standard |

## Deep read: DarwinX (no implementation exists)
*Second pass on method, results, and limitations; third pass ("virtually re-implement it") on the selection rule.*

**One-sentence summary.** DarwinX treats agent improvement as natural selection over a tree of harness variants. A child harness is kept only if it solves something new without losing what its parent solved, and specialists from different branches are merged.

**Mechanism, re-derived for implementation:**
- **Per-task solve rate.** p̂ₜ(v) is avg@k over k rollouts. For a child c with parent p, let Δₜ = p̂ₜ(c) − p̂ₜ(p).
  - Net gain: **g(c) = Σ Δₜ**
  - Regression: **R(c) = Σ (−Δₜ)₊**
  - Admit the child iff **g > 0 and R ≤ δ**.
- **Two speeds.** A promoted child becomes an *ancestor* on cheap screening (small k). It may *steer search* only after a higher-k re-test plus a preservation probe on tasks the parent already solved.
- **Parent selection.** With probability 1−β, pick argmax of lineage gain G(v) = G(parent) + g(v) among confirmed nodes; otherwise broaden across the population. Cumulative gain is used because raw scores on different task subsets aren't comparable.
- **Children are classified by solved set S** relative to the parent:
  - improver: S(c) ⊋ S(p)
  - neutral: S(c) = S(p)
  - stepping stone: S(c) ⊊ S(p)
  - archived: trades some solves for others

  Only improvers and neutral children are eligible for inheritance. The rest contribute only their distilled lessons.
- **Merge.** H = H₀ ⊕ Δcode ⊕ Δskill ⊕ Δprompt ⊕ Δtool, kept iff S(child) ⊇ ∪ S(vᵢ). Edits are **additive**, which is what makes merging possible.
- **Evidence routing.**
  - Tasks with no passing rollout ("walls") get teacher-derived evidence.
  - Tasks with mixed pass and fail ("variance band") get self-contrast evidence.
  - Failures get failure diagnosis.

**Critical reading** (from the Stanford guide: suspicious, check data and assumptions):
- The authors admit the operators (archive, selector, merge) are **not independently ablated**. The gains are a system-level claim.
- On WebArena-Infinity **every merge was reverted**. Recombination is unproven there; it only shows up on TerminalWorld (specialists 24–27 → merged 28/41).
- **The in-loop proxy overfits.** The training subset saturates from 0.505 to 1.000 while held-out reaches only 68.3%, a **31.7-pt gap**. Held-out gating is not optional.
- Cross-benchmark transfer (84.2% on SWE-V) sits within a narrow 80.8–84.2 band.
- The authors name two deployment blockers themselves:
  1. **Production tasks rarely arrive with a verifier.**
  2. **avg@k costs k rollouts per candidate per task**, which is fine as a periodic offline job but not per request.

  Their suggested workaround: evolve against a maintained proxy suite and deploy the winner.

**Creative reading** (the good ideas and what to do next):
- The "preservation set" doubles as a **policy interface**. Any compliance probe can become a "never regress" constraint. That's a sellable enterprise feature.
- The harness is an **asset that outlives model generations**. Re-selection from a warm archive on a model swap is untested, cheap to try, and valuable.
- **Next step if building on it:** solve blocker #1 (no verifiers). For coding agents, a repo's **own git history is a free supply of verified tasks**: commits that made failing tests pass.

---

## Opportunity scorecard (Phase B)
Scored 1–5 on: proven gain · productization gap · solo-buildable · buyer pain · moat.

| Candidate | Gain | Gap | Solo | Pain | Moat | **Total** |
|---|---|---|---|---|---|---|
| **A. Harness CI for coding agents**: evolve a team's `CLAUDE.md` / `AGENTS.md` / skills using DarwinX-style selection, verified on tasks **mined from the team's own git history** | 4 | 4 | 4 | 4 | 3 | **19** |
| B. Generic self-improving layer for any production agent | 4 | 2 (RELAI raised $6.9M Jun 2026; LangSmith Engine) | 2 (needs per-customer verifiers, enterprise sales) | 4 | 3 | 15 |
| C. Model-migration autopilot: re-tune the harness when switching to a cheaper model | 3 | 3 | 3 | 5 | 2 | 16 |

**Winner: A.** C becomes a feature on the same engine.

## Next question to ask (research #02)
*"How do you build trustworthy evals from production traces when there's no ground truth?"* RHO, LLM-judge reliability, and verifier synthesis. This is the bottleneck every paper above names.
