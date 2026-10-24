# V28 recommendation: build a task that can test compatibility at all

Written 7 September 2026, after two independent V27 runs:
`v27_compatibility_results` (ChatGPT, 1760 trajectories) and `v27_claude_results`
(570 trajectories, source SHA256 `d340bf12…c40472`).

**Both V27 runs retired competitive plasticity, and both agree. But neither run
actually tested representation-memory compatibility — for opposite reasons. One
had no headroom for adaptation; the other had no memory to damage. V28 should
construct and validate a task that has both, and should not test another
mechanism until that task exists.**

Intervals are descriptive paired Student 95% intervals over independent seed
differences, six seeds, unadjusted for multiplicity.

---

## 1. What V27 settled

### 1.1 Competitive plasticity is retired, confirmed twice

| run | arms moving the frontier |
|---|---|
| ChatGPT V27 | **0 of 32** (28 inconclusive, 4 `sparse_plasticity_effect`) |
| Independent V27 | **0 of 12** (5 `update_budget_effect`, 6 `partial`, 1 `sparse_plasticity_effect`) |

The control patterns differ slightly and both are damning. ChatGPT's run: competitive
beat dense by +11.86 [+9.20, +14.53] but was matched by random-subset
(+1.10 [−0.85, +3.05]) and by anti-competitive bottom-k (−0.17 [−2.85, +2.51]),
losing to bottom-k in 8 of 18 arms. The independent run: competitive beat bottom-k
in 6 arms but was matched by random-subset everywhere, and its frontier offset was
negative or null in all 12 — significantly *below* the dense curve at FL r1
(−1.44 [−2.58, −0.30]) and FL r3 (−1.71 [−2.62, −0.80]).

Selecting *by magnitude* is no better than selecting *at random*. Competition adds
nothing. Combined with V26's result that novelty, shuffled and norm-matched gates
were all matched by a random gate and by simply lowering the rate, the conclusion
across two levels of the hierarchy is the same: **what matters is how much the
representation changes, not which cells change or when.**

That class of mechanism is closed. Do not test another selection heuristic.

### 1.2 Contrast geometry is confirmed a third time

| task | baseline | contrast | residual |
|---|---:|---:|---:|
| context (blended) | 62.75% | **69.44%** | 64.20% |
| association (blended) | 72.54% | **75.34%** | 73.72% |
| association one-shot | 91.40% | 91.58% | 91.41% |

Independent replication of ChatGPT's D family (context 51.14% → 67.05% there).
Smaller margin, same direction, no one-shot cost. Three separate runs now agree.
Contrast is the encoder default.

### 1.3 A claim of mine that did not replicate

I wrote after V26 that "the geometry that wins structured transfer is not the one
that wins compatibility," based on ChatGPT's practical panel where
`contrast_stable` (+5.09 / −12.65) sat worse on the frontier than
`balanced16_stable` (+17.71 / −4.94). On the independent shift task, contrast
dominates on **both** axes:

| arm | new | old |
|---|---:|---:|
| **B_contrast_dense_r2** | **51.35%** | **46.42%** |
| B_residual_dense_r2 | 48.02% | 44.32% |
| B_random8_dense_r2 | 46.80% | 40.49% |
| B_balanced_dense_r2 | 44.87% | 42.30% |

The separate-axes claim does not survive. It was one panel on one task.

---

## 2. Why neither V27 run tested compatibility

This is the finding that should drive V28.

### 2.1 ChatGPT's run had no headroom

Its frozen anchor reached **98.22%** new-cohort accuracy. Every adapting arm
therefore traded 6–40 pp of old-cohort retention for under 1 pp of new
performance. A frontier measured in that regime is a pure loss curve: adaptation
has nothing to buy, so no mechanism can demonstrate value against it.

The cause is now identifiable. V26's causal panel used random8 geometry and its
frozen FF arm sat at 74.19% new-cohort — ample room, which is why adaptation looked
valuable there (+23.61 pp for `practical_live`). V27 switched the default to
contrast, and the frozen anchor jumped ~24 pp to saturation. **Improving the fixed
encoder removed the need for encoder adaptation rather than solving the
compatibility problem.**

### 2.2 The independent run had no memory to damage

The independent runner carries a headroom diagnostic added for exactly this
failure. It reported `usable`: frozen anchor 46.17%, best adaptation gain
+8.46 [+3.36, +13.56]. Headroom was genuinely present.

But the frontier it measured is not a frontier at all:

| arm | new | old |
|---|---:|---:|
| A0_frozen | 46.17% | 40.74% |
| A1_dense_LL_r1 | 49.16% | 42.51% |
| A1_dense_LL_r2 | 51.35% | 46.42% |
| **A1_dense_LL_r3** | **54.63%** | **47.81%** |

Adaptation improves the old cohort by **+7.07 pp** while improving the new cohort by
+8.46 pp. Both rise monotonically with rate. There is no tradeoff to navigate.

The cause is in the stream construction. In `shift_stream`, old-cohort probes are
built from `cue_bases(rng_for(seed, 'v26_shift_probe_pairs'), ...)` — **fresh
random latent pairs**, sharing only the label bits and the direction with training.
Success therefore requires generalizing the category rule, not recalling a stored
association. A better representation improves rule generalization on direction A
and direction B alike, so adaptation is genuinely Pareto and nothing about memory
compatibility is being measured.

### 2.3 A correction

After ChatGPT's V27 I attributed my V26 `pareto_improvement` verdicts (R1_live
+7.70 new / +2.48 old; R3 +7.03 / −0.08) to low statistical power — "a −2.2 pp loss
was indistinguishable from zero with ±5 pp intervals." That explanation was wrong.
The real cause is the probe design above: on this stream adaptation *is* Pareto,
and the verdict was correct for the task as built. The task simply was not
measuring what the verdict was named after.

Headroom was necessary but not sufficient. The diagnostic I added catches one of
the two failure modes and is blind to the other.

---

## 3. What a valid compatibility task requires

Four conditions, all of which must hold simultaneously:

1. **Recall, not generalization.** Old-cohort probes must be noisy renderings of
   the *same* latent identities that were trained, so accuracy depends on the
   stored association surviving. The V24/V25 association stream already has this
   discipline ("noisy held-out rendering of learned identity"); the shift stream
   discarded it.
2. **Arbitrary bindings.** Old associations must not be derivable from a rule the
   encoder can relearn. If the label is a function of the input that a better
   representation computes more accurately, adaptation repairs what it damages.
3. **Headroom.** A frozen encoder of the *quality actually being used* (contrast,
   not random8) must be measurably below ceiling on the new cohort. Demonstrating
   headroom with a weak encoder and then running mechanisms with a strong one
   measures nothing.
4. **No replay.** Old items are never re-presented for training after the shift.

Conditions 2 and 3 are in tension, and that tension is the real design problem.
Arbitrary bindings are what make recall meaningful; but a new cohort hard enough to
defeat a contrast encoder tends to be rule-structured, which reintroduces
generalization. V28's first job is to resolve this, not to assume it away.

The most promising resolution: **separate the rule from the instances.** Old cohort
carries arbitrary input→label bindings probed by noisy variants of the trained
inputs (recall-dependent). New cohort introduces sensory structure the contrast
encoder demonstrably underrepresents — a different temporal scale, or a direction
in a subspace the fixed afferents sample poorly — so that headroom exists without
the old cohort becoming rule-derivable.

---

## 4. V28 design

### Block A — construct and validate the task (primary; nothing else runs until this passes)

Build candidate shift streams and *screen them for validity before any mechanism
is tested*. For each candidate, with a contrast encoder throughout:

- **A1 — headroom probe.** Frozen contrast encoder on the new cohort. Record the
  level and the best paired gain any dense-adapting arm achieves over it.
- **A2 — tension probe.** Dense adaptation at several rates, scoring the old
  cohort. The task is valid only if adaptation measurably *hurts* old-cohort
  accuracy.
- **A3 — recall-dependence probe.** Score the old cohort twice: with noisy variants
  of trained inputs, and with fresh latents of the same category. A valid task
  shows a large gap. No gap means the old cohort is measuring generalization.

**Task-validity gate, all three required:**

1. frozen new-cohort gain available ≥ 5 pp with a paired lower bound above zero;
2. dense adaptation reduces old-cohort accuracy with an upper bound below zero;
3. trained-input recall exceeds fresh-latent accuracy by a wide margin.

A candidate stream failing any of these is discarded, not patched. Report the
screen even for discarded candidates — knowing which task constructions cannot
express compatibility is itself the deliverable.

### Block B — decompose the frontier (only on a task that passes Block A)

With a valid task, ask *why* old associations break, before proposing repairs.
Using the existing FF/FL/LF/LL factorial plus counterfactual probes that hold one
pathway at its old state while the other moves:

- how much old-cohort loss is attributable to **address drift** (the probe routes
  to a different bank than it was written to);
- how much to **content drift** (same bank, changed coordinates);
- how the split shifts with adaptation rate.

This has never been measured on a task where the question is well posed. V25
attempted it under a random8 encoder and a stream whose properties we now know
were confounded.

### Block C — one mechanism, chosen by Block B (only if Blocks A and B succeed)

If address drift dominates: an invariant or slowly-tracking key. If content drift
dominates: explicit key–value factorization, or local transport of stored weights
into the new coordinates. **One specified mechanism with matched controls**, held
to the standard the novelty gate and competitive plasticity both failed: it must
beat a control matched on total representation change, and a control matched on
sparsity with the selection rule inverted.

If Block A finds no valid task, Block C does not run. That is a legitimate outcome
and should be reported as one.

---

## 5. Not in scope

Further plasticity-selection heuristics; gating variants; language modelling;
hierarchical, scratchpad or multi-horizon architectures; dendritic units;
re-tuning anything in the closed list (output gain modulation, readout addressing,
learning-rate cascades, spillover, variance salience, novelty gating, competitive
plasticity).

**Scaling remains untested since V20** — every experiment from V20 through V27 uses
264 identities. It stays deferred, but the reason has changed. It was previously
deferred behind the encoder bottleneck, which is now resolved; it is now deferred
behind task validity. If Block A fails to produce a valid compatibility task, the
right next move is scaling with a frozen contrast encoder rather than continuing
to search for compatibility mechanisms.

That is a live possibility worth stating plainly: ChatGPT's V27 showed a good
frozen encoder reaching 98.22% on the new cohort. If a contrast encoder at scale
keeps saturating the tasks we can construct, then **continual representation
learning may simply not be needed in this regime**, and the program's remaining
questions are capacity and scaling rather than compatibility.

---

## 6. Provenance

- `v27_compatibility_results` — 1760 trajectories. Frontier screen, matched
  controls (dense-norm, random-norm, fixed-norm, bottom-norm), width sweep,
  geometry panel, robustness across shift variants, confirmation.
  Verdict `not_established`.
- `v27_claude_results` — 570 trajectories, seeds 701–739. Independent
  implementation reusing the V25 bank substrate and V26 shift stream. Headroom
  diagnostic, budget-matched and random-subset and bottom-k controls,
  geometry × plasticity block, variance-salience and context closure arms.
- `v26_merged_research_plan.md`, `v27_experiment_recommendation.md`,
  `v26_experiment_recommendation.md`, `v24_complete_results_review.md`.
- Probe construction read from `shift_stream` in
  `elm_fly_representation_v26.py`; frontier levels from `SUMMARY.csv` in
  `v27_claude_results`.

Corrections to earlier Claude analysis recorded here: the separate-axes claim
(§1.3), and the low-power explanation for the V26 Pareto verdicts (§2.3), whose
real cause is the probe design. The variance-salience closure arm in the
independent run returned `task_specific_salience_exploit` but with intervals
spanning ±13–25 pp; it is underpowered and ChatGPT's full task-variant grid
remains the stronger evidence on that question.
