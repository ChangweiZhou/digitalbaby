# V69 design — from ownership information to plastic action

**Status** design proposal, pre-outcome. Nothing here is a result.
**Parent** unchanged `iteration5/bi`, instantiated from the frozen `iteration10` reference, with the
fixed `iteration8/current_RBF64` observer. 5,177 KCs, 165,688 mutable bytes, no cue-specific banks.
**Scope** V69 does not change the encoder (except in the isolated Stage C diagnostic), KC count,
feedback parameters, physiology fit, anatomical routing or the reader. No new download. No
specimen identities joined.

---

## 0. Why V69 exists

V68 settled the question V67 left open. The model **does** contain ownership information, and it is
accessible: single-KC decoding is AUROC 0.608, averaging 32 local probabilities reaches 0.832, and
active-population statistics reach **0.951 [0.926, 0.975]** for reversal versus unrelated and
**0.983 [0.974, 0.992]** for reversal versus anomaly. Memory-removal shifts the S1 reversal
probability by 0.148 [0.091, 0.205] at standardised effect 1.26, and stable associations are
flagged at 0.21% against 87.4% sensitivity. The information is real and history-dependent.

V68 also showed that acting on it is a separate and unsolved problem, and gave two hard facts
about why:

1. **Category information does not imply action utility, inside one feature family.** The failed
   pilot's two scores are members of S1: E is feature 27, `fast_aligned_to_slow_conflict_ratio`, and
   F is `slow_L1`/`slow_L2`. Drawn from the same 28 statistics that decode at 0.951, they retained
   3.27% of reversal-conflicting mass against 26.83% of unrelated-conflicting mass — backwards — and
   the fresh run lost 26.56 points of durable reversal.
2. **The failure was selection, not dose.** The gate retained *more* slow-write mass than the scalar
   control (90.35% versus 89.66%) and still performed worse. This is not the old learning-rate
   tradeoff.

And the privileged LP witness found that even an oracle restricted to retaining or attenuating the
existing slow direction at 80% endpoint L1 mass produced **no old-choice improvement** (0.00 pp at
load 12, −0.30 pp at load 48), feasible in only 10/16 and 7/16 cases.

V69 therefore asks the question V68's own conclusion names: **which compact runtime-accessible
population quantities predict the benefit or harm of a particular write, rather than its episode
category** — and, separately, whether a richer action than "turn the slow write down" exists.

---

## 1. The one inference V69 is built on

The LP held **both fast traces fixed at the intact endpoint**. Its negative result therefore
constrains only the retain-or-attenuate family on the slow direction. Routing a conflicting write
into fast state instead of discarding it lies **outside** the family the oracle searched, and is
not excluded by that result. This is the single load-bearing inference in this design and it should
be checked by the implementer before Stage B is built.

---

## 2. Stage A — action value on frozen snapshots (primary, unconditional)

Stop classifying episodes. Score actions.

### A.1 Snapshots

From existing reconstructed checkpoints, take snapshots at loads **12, 24, 48, 96**. At each
snapshot a new association is presented and a write is proposed. Nothing is promoted; every action
branches from the identical pre-write state.

### A.2 The bounded action set

Fixed before running; no additions afterwards.

| id | action |
|---|---|
| `NORMAL` | write as the parent does (baseline) |
| `SUPPRESS` | zero the conflicting slow component (the V68 action — a known-bad reference) |
| `ROUTE_FAST` | conflicting slow component is added to alpha-fast instead; slow component zeroed |
| `DELAY_PROMOTE` | as `ROUTE_FAST`, but promote accumulated fast evidence into slow after conflict persists for `k` bouts (`k` in {2, 4}) |
| `REDISTRIBUTE` | move conflicting mass onto non-conflicting active coordinates, preserving total L1 |

`SUPPRESS` is included specifically so a known-failing action sits in the comparison.

### A.3 The measured quantity

For every (snapshot, action) pair, at +5 min and +24 h, measure the new association's oriented
margin and every previously stored association's margin, against the exposure-matched no-write twin.
Define

```
U = Δ(new margin)  −  λ · Σ_j max( −Δ(old margin_j), 0 )
```

**λ is fixed before any Stage A outcome** at the parent's own observed exchange rate, computed from
frozen V67 data: the ratio of mean new-association gain to mean negative focal disturbance under
INTACT (4.022350 / 0.131162). Record λ in `config.json` before the run. Report U at that λ and, as
a sensitivity only, at λ/2 and 2λ. No λ is selected after seeing results.

### A.4 Controls

- **Same-dose sham** for every action that changes total written mass: rescale `NORMAL` so its
  endpoint slow L1 matches the action's realised L1, per the V68 convention. An action must beat its
  own dose sham, not just `NORMAL`.
- **No-write twin** at matched exposure and adaptation for every branch.
- **Stable and unrelated conditions** carried from V67/V68 so the action set is scored on writes that
  should be permitted as well as writes that should not.
- **Exact replay check**: nonlinear replay of each chosen action must agree with its predicted
  endpoint to < 1e-12 Hz, as V68's LP replay did to 8.9e-15.

### A.5 The selector question

Only after A.3 is measured: can a **small** statistic pick the best action?

Search over the 28 S1 features for subsets of size 1 and 2 exhaustively, size 3 by greedy forward
selection, with one fixed L2 logistic specification, whole-stream holdout, and training-fold
standardisation. No feature engineering, no threshold search, no nonlinear models.

Report selector accuracy against two references: always-`NORMAL`, and the oracle that picks the
true argmax action. A selector that cannot beat always-`NORMAL` is worthless regardless of its
AUROC on category.

### A.6 Stage A gates, declared now

| gate | criterion |
|---|---|
| **A1** | at least one action beats `NORMAL` on U at some load, paired whole-stream 95% interval excluding zero |
| **A2** | that action also beats its own same-dose sham, interval excluding zero |
| **A3** | a subset of ≤3 S1 features selects actions better than always-`NORMAL` on held-out streams, interval excluding zero |
| **A4** | A1–A3 hold at two or more loads |

**If A1 or A2 fails, V69 stops and reports.** Do not build a runtime rule. Do not retune λ, the
action set, or the loads. An action set with no positive member is a complete and valid deliverable
and is the outcome most consistent with the LP witness.

---

## 3. Stage B — conditional runtime test

Built **only if A1–A4 pass**, and its protocol frozen after Stage A passes but before any
calibration, following the V68 convention.

### B.1 What is built

The winning action plus the ≤3-feature selector, as an analytic rule. Mutable state must not grow
beyond what `DELAY_PROMOTE` requires; report the exact byte cost. Fixed coefficients count as fixed
bytes, not mutable state.

### B.2 Endpoints

The V68 fresh-validation endpoints, on a fresh reserved bank with cues globally disjoint from all
prior banks, against **both** the parent and a dose-matched scalar control:

| endpoint | parent (V68 fresh) |
|---|---:|
| stored choices, load 12 +24 h | 89.58% |
| stored choices, load 24 +24 h | 87.50% |
| stored choices, load 48 +24 h | 79.82% |
| stored choices, load 96 +24 h | 71.35% |
| each new association after isolated +24 h | 81.25% |
| reversal after 12 bouts +24 h, all assigned | 82.81% |

All-assigned denominators, as V68 used, so changed acquisition cannot hide revision failure.

### B.3 The schedule falsification — required, not optional

Alpha-fast retains **1.065%** over a 15,840 s rest (iteration 6 `state_geometry.csv`), giving
τ ≈ **58 minutes** and a half-life near 40 minutes. A 6-bout block is 1,980 s, so fast retains ~57%;
a 12-bout block ~32%; across 24 h, ~1e-11.

Any mechanism that buffers evidence in fast state therefore has a shelf life of roughly one hour,
and the existing reversal protocol sits inside that window. **Run the same 12-bout reversal dose
spread across 24 h as well as within one hour.** If the mechanism's advantage disappears on the
spread schedule, it is fitted to the bout schedule and must be reported as such. Declare this
before running; do not add it afterwards.

### B.4 Stage B gates

| gate | criterion |
|---|---|
| **B1** | acquisition noninferior: new-association margin within 0.25 Hz and joint success within 5 points of parent |
| **B2** | stored choices superior to parent at ≥2 loads, paired intervals excluding zero |
| **B3** | durable reversal noninferior to parent (≥ 82.81% − 5 points) |
| **B4** | beats the dose-matched scalar on B2 endpoints |
| **B5** | advantage survives the spread schedule of B.3 |
| **B6** | no increase in false flips versus parent |

Failing B5 does not void B1–B4 but must be reported in the headline, not a footnote.

---

## 4. Stage C — recruitment-spread diagnostic (independent, cheap, non-blocking)

This is a **diagnostic**, not a candidate learner, and it runs independently of A and B.

### C.1 The measurement that motivates it

From `iteration11/data/runs/*/footprints.npz`, slow component, 32 streams × 4 focal memories:

| quantity | value |
|---|---|
| KCs written per memory | 231 |
| writable pool per stream (union of its 4) | 351 |
| global union across 32 streams | 1,170 |
| anatomically alpha-writable | 1,226 |
| pairwise support Jaccard, observed | 0.536 [0.509, 0.563] |
| same, random-within-pool null | 0.438 [0.408, 0.467] |
| signed cosine mean | +0.002, indistinguishable from zero |
| \|cosine\| observed vs shuffled null | 0.070 vs 0.012; 22% of pairs above 0.1, null 0% |

Each memory occupies two-thirds of the substrate its stream uses, and a stream's four memories touch
30% of what is anatomically available. At 231-into-351 an overlap near 0.44 is forced arithmetically.
Combined with signed alignments that average zero but spread 6× wider than null, this is V66's
0/62-versus-62/62 stated geometrically: heavy shared substrate, random signed angles, and the
unlucky ones destroyed.

The model's sparsification is a fixed deterministic top-5%; iteration 5 states it "is not a fitted
APL circuit."

### C.2 What to run

At **fixed KC count and fixed write rule**, vary recruitment spread only. Measure pairwise support
Jaccard and the load curve, against the existing parent curve.

### C.3 The objection, stated in advance

`cue_overlap` correlates **0.0175 [−0.0031, +0.0392]** with the delayed effect while `signed_cue_cos`
correlates 0.771 and `cos_slow` 0.906. The defence is that overlap has almost no variance here — all
pairs sit between roughly 0.44 and 0.54 — and a variable with no variance cannot correlate while
still setting the scale. That argument may be wrong, and only intervention settles it.

### C.4 Gate

**C1** — if support overlap falls materially and delayed damage does not fall with it, the
variance argument is refuted and Stage C is closed permanently. Record that outcome; do not iterate
on recruitment schemes.

Note iteration 10 found that changing KC *count* degrades choices (647 KCs: 73.89% versus 88.02%).
This is a different manipulation at constant count, but carries the same risk of degrading the code,
and C1 must be read alongside an acquisition check.

### C.5 Interaction to record, not to resolve

If spreading works, it reduces exactly the conflict statistics S1 reads. The signal would weaken
because the problem shrank. A and C must therefore be evaluated separately and never stacked in one
arm.

---

## 5. Sequencing

`§12`-style connectome questions — which circuits could pool population statistics — are deferred
until Stage A names the one or two quantities worth pooling. Biologically plausible pooling is low
dimensional; a single APL-like cell computes a scalar, and the one two-dimensional compression
tried so far failed backwards. Searching the connectome for an implementation of an unidentified
statistic is premature.

Do **not** deploy the 28-feature logistic as a runtime controller. It would exploit the information
while moving the project toward an engineered ML controller, and V68's diagnostic models are
explicitly offline.

---

## 6. What V69 cannot establish

No measured odor→PN activity enters this design, so no new biological-fit score is produced and the
existing physiology failures stand unchanged. Frozen snapshots and generated cue banks are streams
through one specimen's anatomy, not independent animals. Action utility measured on snapshots is not
longitudinal performance of a learner using that action from zero — Stage B exists precisely because
those differ. A compact selector, if found, is an information result about this fitted model; no
anatomical circuit computing it is thereby identified. Stage C changes a modelling choice, not a
biological finding about APL.

---

## 7. Budget and stopping

V68's replay, witness and pilot worker time totalled about 91 seconds at ≤267 MB RSS. Stage A is of
that order; Stage B adds one fresh bank generation. One compute budget, one declared action set, no
post-outcome tuning, no replacement family after a failure. If Stage A fails its gates, return the
smallest reproducible failing case, the affected assumption, and the completed results, and stop.
An inconclusive or negative V69 is a valid completed deliverable.
