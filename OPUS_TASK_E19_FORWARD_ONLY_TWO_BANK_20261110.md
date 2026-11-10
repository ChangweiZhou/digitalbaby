# One cloud job for a Claude Opus 5.5 worker: E19, a two-bank learner whose credit is exact and forward-only (10 November 2026, 12:45 EST)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (fetch and check out; confirm this file is present). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy. This job writes new code. It is for one worker; say in your first message that you are running E19.

**Branch rule.** If your session can push to `codex/minifly-consolidated-20261023`, commit there (pull with `--rebase` before each push, never force, at most one push an hour while runs are in flight). Otherwise commit to the branch your session is given and name the branch and final commit in your last message. Do not create any other branch.

**Read first:** `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md` (Q1 and Q5); K-prop itself (Liu, Haider, Benitez, Petrovici, Senn, arXiv 2611.12530, Section 3.1 and Appendix A) if your session has web access; `arch17.py` (header, `Net17`, `forward`, `install`); `develop17` and `adamw16` in `exp16_integrated.py`; `exp45/E17_ALIGN_AT_BIRTH.md`; review sections 9.67 and 9.72 in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`; `.agents/skills/causal-experiment-loop/SKILL.md` (Mr. Zhou's statement of how experiments here are run).

## The one question

In the two-bank learner, the layers above each write and the first bank as a whole lose their credit in time under the rule as run (measured: cosine about 0 with the true gradient on the first bank's projection). E17 Phase 4 repairs that with a backward pass over the item. The literature check found that for two stacked layers of leaky units with fixed retention the same credit can be carried **forward only and exactly**, at one extra trace per lower synapse per distinct upper retention rate, provided the upper layer's input is linear in a function of the lower layer's state. The existing two-bank network does not meet that condition: its second bank is written through a nonlinear projection and a gate.

**Question: does a two-bank learner that meets the condition (a linear write into the second bank) still have enough computing power to be born on the development task, and if it does, can it be born by the forward-only rule with weight-sign feedback?** Stage A answers the first half with backprop and is a feasibility gate. Stage B runs only if the gate passes.

## What to build (new files only)

`arch19.py` and `exp19_kbank.py`. Edit no existing file. Use the harness constants (`E.C` = 512 units per bank, `E.H` = 512 readout units, `arch17.RATES` = five retention rates assigned in equal groups, 258 symbols in, 257 out), the standard development stream and panels exactly as `develop17` uses them, and the `install` pattern of `arch17.py` so that `E.respond` and `E.score` work unchanged.

**The network (`kbank2`).** With x_t the byte at step t, ρ = tanh, R1 and R2 the fixed diagonal retention of each bank:

```
h1_t = R1 h1_{t-1} + A[:, x_t]                      A is a table, C x 258: the first bank's write, one column per byte
h2_t = R2 h2_{t-1} + W2 ρ(h1_t) + b2                the second bank integrates a linear map of ρ of the first bank's state
z_t  = tanh(Wm1 ρ(h1_t) + Wm2 ρ(h2_t) + E[:, x_t])  readout hidden layer; E is a table, H x 258, for the current byte
logits_t = W3 z_t + b3
```

The first bank of the existing network is already a function of the current byte alone (embedding, first layer, projection, then gate and content), so the table A loses nothing against it and removes every layer above the write. What is given up against the existing network is the gated, nonlinear write into the second bank. Parameter count is about 1.18 million against the existing 1.13 million; report yours.

**The forward-only rule.** With δ1_t and δ2_t the readout's error with respect to h1_t and h2_t at step t (within the step, through `Wm1`, `Wm2` and ρ′), every gradient is a sum over steps of a current error times a trace kept forward in time:

```
dL/dW2[c, j] = Σ_t δ2_t[c] e2_t[c, j]          e2_t[c, j] = r2_c e2_{t-1}[c, j] + ρ(h1_t)[j]      (depends on c only through its rate: five traces per j)
dL/dA[j, b]  = Σ_t ( δ1_t[j] e1_t[j, b] + Σ_g K^g_t[j] ζ^g_t[j, b] )
               e1_t[j, b] = r1_j e1_{t-1}[j, b] + [x_t = b]
               ζ^g_t[j, b] = r_g ζ^g_{t-1}[j, b] + ρ′(h1_t)[j] e1_t[j, b]        one trace per upper rate g: the deep traces
               K^g_t[j]   = Σ_{c in rate group g} δ2_t[c] W2[c, j]               the error relayed down, grouped by the upper unit's rate
```

Readout weights and the table E learn within the step. Derive it yourself and correct this sketch if it is wrong; the acceptance test below is the authority, not these lines. Under **weight-sign feedback**, every place where an error passes back through a weight matrix (`W3`, `Wm1`, `Wm2`, and `W2` inside K) uses sign(W) times the mean absolute weight of that matrix, as `SignLinear` in `exp3_state.py` does; nothing else changes. No step may look at a later step or revisit an earlier one: one pass over the item, state and traces only. Because the input is one byte per step, traces indexed by the position of the write and merged by byte at the end are an acceptable equivalent of traces per table entry; say which you used and what a true online version would store.

**Optimizer.** AdamW, learning rate 0.001. Backprop births: decay 0.3, as every birth. Rule births: decay 0.25, as every life. The per-byte tables A and E are in a group without decay (the E17 lesson: decay erases the rows of symbols that the stream has not shown yet).

## Acceptance tests (commit `exp45/E19_CHECK.json` before any birth)

1. **Equality.** With exact weights in every feedback path, the forward-only gradient equals PyTorch autograd's backprop through time on every parameter, maximum relative error below 1e-4, on three random initialisations and, later, on the three Stage A births, on development batches of 64 with distractors up to twelve letters.
2. **Forward-only.** A version of the update that is fed the item one byte at a time, holding nothing but the two bank states and the traces, gives the same gradient as your batched implementation (maximum relative error below 1e-4).
3. **What the deep traces are worth, before any training.** At each Stage A birth: cosine with the true gradient, per parameter group, of (i) the full rule with weight signs and (ii) the rule with the deep traces removed (ζ replaced by its current-step term ρ′(h1_t) e1_t). Report the table; it is a measurement, not a gate.

If test 1 or 2 fails, stop and report; do not launch births.

## Runs, seeds 73129, 93683, 116089

Standard development: 24,576 updates, batches of 64, scored on the sealed panels at k = 0, 4, 8, 12, 20 every 4,096 updates, as `develop17`.

- **Stage A, the feasibility gate (three births):** `kbank2` by backprop through time.
- **Gate:** Stage B runs only if the Stage A hard mean (mean of add89 and sub89 at k = 12, final checkpoint) is at or above 60 in at least two of the three seeds. If it is not, stop there, report Stage A in full, and do not vary the architecture: that the linear write is too weak is the result.
- **Stage B (six births):** R1, the forward-only rule with weight signs; R2, the same with the deep traces removed (the matched control: one difference).

Files `exp45/e19_dev_kbank2_{bptt,r1,r2}_<seed>.json` and `.pt` (`git add -f` the `.pt`), logs `exp45/logs/e19_<arm>_<seed>.log`. Record that none exists before launch. Dry-run each arm for 512 updates first and archive under `exp45/archive_e19_dry/`. Launch births detached and poll logs with short commands. No reruns of a finished birth, no tuning of sizes, rates, learning rate or decay, no change of seeds: a miss is a result.

References on file (owner's machine; compare levels, not digits): the existing two-bank network by backprop, hard mean 91.2 / 94.5 / 90.3; one bank alone, 21.7 / 19.3 / 15.2.

## Predictions (copy into `E19_PREDICTIONS.md` and commit before Stage A; written by the coordinating session before any `kbank2` code existed)

- **P19.1 (acceptance)** Tests 1 and 2 pass.
- **P19.2 (the gate)** Stage A's hard mean is at or above 60 in at least two of the three seeds, and below the existing two-bank reference in all three. Held at about 50%: the second bank now accumulates at every step, distractor letters included, with no gate to refuse a write.
- **P19.3 (the rule against its own ceiling)** If Stage B runs: R1's hard mean is at or above 0.85 of the Stage A birth of the same seed in at least two of the three seeds.
- **P19.4 (the deep traces matter)** R1's hard mean is at least 10 points above R2's in at least two of the three seeds.
- **P19.5 (cost)** Reported, not predicted: CPU-milliseconds per update for backprop, R1 and R2; traces held per synapse and per sample; peak memory.

Reading, written before the result. P19.2 and P19.3 holding give a learner with no frozen part, no backprop, no weight transport and no backward pass, whose work per byte is bounded and known before birth; its lives and its retention are then the next job, and nothing about them is claimed here. P19.2 missing says the gated write into the second bank is doing necessary work; the next step is then K-prop's approximate form for a gated write, or more rates, and the Phase 4 backward pass remains the only repair on file. P19.3 missing with P19.2 holding says weight signs, not the credit path, are the limit for this network. P19.4 missing says the credit through the second bank's memory is not what the first bank needs here, which would contradict this morning's measurement on the gated network and should be checked against acceptance test 3.

## Report

`E19_REPORT.md`: the derivation as implemented; the three acceptance results; per arm and seed the six checkpoints of the birth curve (four operations at five k), the hard mean beside both references, and the silenced score if you implement silencing; the gate decision; the five verdicts; run identity (machine, core count, torch version, commit, environment of each process); CPU-seconds per birth; forecast from the dry runs against measured; what went wrong. One paragraph as `E19_REVIEW_SECTION_9_73.md`. State the evidence level as E2 (development panels, sealed, operand-disjoint); no life, no retention and no reuse on never-reinforced inputs is tested; no learning-core improvement is claimed.

**Standing notes.** Mr. Xu's working answers of 10 November (review 9.67): feedback relayed layer by layer or by a sub-network is allowed; a learner with a frozen part does not meet the aim. Whether weight-sign feedback meets the project's locality constraint is not settled by Mr. Zhou and is marked as disputed in every claim. The literature memo and this brief decide nothing.

**Cost.** A backprop birth of the existing two-bank network takes about 1,730 CPU-s on the owner's machine and roughly twice that on the cloud containers used so far. Expect the three Stage A births to take about an hour of wall on four cores and the six Stage B births two to four hours, depending on your implementation of the traces. Forecast from your dry runs.
