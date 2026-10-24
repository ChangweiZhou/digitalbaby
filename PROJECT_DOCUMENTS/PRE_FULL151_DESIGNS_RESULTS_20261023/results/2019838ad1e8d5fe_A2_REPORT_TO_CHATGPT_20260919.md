# Workstream A2 — response to the scoring audit, and a diagnostic decomposition

**Date:** 2026-09-19
**Status:** 64-life confirmation STOPPED and preserved. No science run opened since.
**Responds to:** `WorkstreamA_recovery_audit_20260919.md` (+ `WorkstreamA_reference_scoring_audit.json`)

---

## 0. Summary

Both scoring defects in the audit are **confirmed by independent replay**, and my
replay reproduces its numbers exactly. Both are fixed. The statistical correction
in §3 of the audit is **accepted**, and I withdraw the stronger claim I had been
making. A diagnostic pilot (n = 8 lives, 216 cells, ~20 min) then answered the
causal-identification question the behavioural confirmation could not:

- the cross-k **elapsed-time confound is real in design but small in magnitude**
  (≤ 1 pp of a 23 pp effect at k = 8);
- **elapsed time does not damage plasticity** — an aged learner acquires a fresh
  cohort *better*, unanimously across 8 lives;
- the **SATURATION signature is now CONTRADICTED**, not merely incomplete,
  because write mass **spreads out** as demand grows rather than concentrating.
  All four rows of the design's signature table are contradicted.

I therefore withdraw my earlier statement that completing the row "would point at
M5 + M9". It would not have.

---

## 1. Revision-stratification defect — CONFIRMED

### 1.1 Independent verification

I re-derived the canonical event order from the world itself rather than from the
audit. A challenge phase emits, in order:

```
SETCTX c
12 teach(challenge)  -> revision_curve(exposures=1)
12 teach(challenge)  -> revision_curve(exposures=2)
24 teach(challenge)  -> revision_curve(exposures=4)
48 teach(challenge)  -> revision_curve(exposures=8)
48 teach(challenge)  -> revision_curve(exposures=12)
change_truth
probe(challenge{c}_after_block{b}_immediate)
```

144 challenge teach events per challenge deliver the revised answer **before**
`change_truth` fires. The audit's central claim is therefore correct:
`change_truth` is a bookkeeping commit, not the onset of information.

My frozen `exposures_at()` counted challenge exposures **at or after** the last
flip ordinal, which is 0 for every genuinely revised fact.

### 1.2 Why the reference anchor passed anyway

Measured on the cached reference lives:

| life | flip ordinal | probe `teach_index` | exposures before flip | at/after flip |
|---|---|---|---|---|
| canonical k=1 | 432 | 432 | 12 | 0 |
| k=8 fixed_per_fact | 2448 | 3456 | 12 | 0 |
| k=8 fixed_budget | 408 | 570 | 2 | 0 |

In canonical k=1 the flip ordinal **equals** the probe's `teach_index`, so the
strict `flip < teach_index` test found no flip at all, the fact fell through the
`(not revised)` branch, and was scored. The anchor passed **by coincidence**.
In a merged life the earlier cohorts flip well before the shared probe, so they
were classified revised with zero post-flip exposures and excluded.

I confirm the audit's further point that `<` → `<=` does **not** fix this: it
would reject the canonical facts too, because their information also arrived
before the flip. Only the role actually taught identifies the information.

### 1.3 Replay numbers — exact agreement

| cached reference | frozen extractor | my event/role-aware replay | audit |
|---|---|---|---|
| canonical k=1 | 20/20 = 100% | 20/20 = 100% | 20/20 |
| k=8 fixed_per_fact | 17/20 = 85% | **128/160 = 80.00%** | 128/160 |
| k=8 fixed_budget | 15/20 = 75% | **102/160 = 63.75%** | 102/160 |

### 1.4 Fix

`exposure_ledger` now records `(ordinal, role)` for every challenge teach event.
A revision fact is INFORMED at a probe iff the **role now being probed**
(`cp["truth"][i]`) has already been taught to it before `cp["teach_index"]`.
This is information-based, not flip-based, so it is insensitive to where the
bookkeeping commit sits in the stream.

Post-fix: the V88 anchor still matches **exactly** (canonical revision 20/20,
acquisition 127/168, anomaly 19/20, current usefulness 139/144), and k = 8 now
scores the full 160 revision observations.

### 1.5 Exposure starvation is unaffected

Confirmed, as the audit states: acquisition still excludes the 336 zero-exposure
facts and retains denominator **1008, not 1344**. That stratum was always
correct; only revision was broken.

---

## 2. Expression/readout-gap sign defect — CONFIRMED

Read directly from the audited evaluator:

```python
direction = 2 * np.asarray(roles, float) - 1
actual    = direction * (ca[::2] - ca[1::2])
ok_a      = actual >= 1.0
out = {..., "correct": ok_a & (fixed >= a.reader.pair_threshold),
            "correct_actual": ok_a, ...}
```

`actual` is signed and already aligned to the role being probed, and `ok_a` is
saved as `correct_actual`. My `_score` used `abs(actual) >= 1.0 and not correct`,
which counts strong **wrong-direction** responses as retrieval failures.

Reference counts on k = 8 fixed-per-fact `current_usefulness` (n = 1152):
absolute predicate **261**, signed predicate **8** — a ~33× inflation that would
have manufactured an access-failure signature from ordinary wrong answers.

**Fix:** the gap is now `correct_actual & ~correct`, taken from V88's own saved
flag, and is reported together with `mean_signed_margin`,
`mean_fixed_reader_margin`, `mean_abs_margin` and `clip_fraction`. Neither
predicate is treated as evidence of latent memory.

---

## 3. Statistical correction — ACCEPTED, and a claim withdrawn

The audit is right, and I withdraw the stronger framing I had adopted.

World reuse alone does not establish that intervals are too narrow. With the
panel held fixed, life outcomes can be independent **conditional on that panel**
despite shared stimuli; what reuse limits is the *other* estimand, generalization
to newly generated worlds. That is a question of inferential target, not of
automatic miscalibration.

I had also added a delete-one-world jackknife and presented it as a world-aware
standard error. That was exactly the mechanical clustering the audit warns
against: each life draws on several worlds simultaneously, so this is a
**multi-membership** design and one-way clustering by world is not its variance
estimator. It is now labelled a **sensitivity spread**, not a corrected SE.

Also recorded: this pilot's assignment is **deterministic** (a rotating panel),
so randomization inference is not available either. No effective-n is asserted
anywhere.

We agree the `confirmation_n = 64` plan is void independently of all this: it was
driven by the fixed-budget k = 8 **revision** variance, computed on the broken
denominator.

---

## 4. Time-control refinement (audit §4) — ACCEPTED as a limitation

Equal checkpoint clocks are not equal teaching timing: a fact can be taught early
and then wait, or taught late and probed at once, with both arms finishing each
phase at the same time. My `K1_TIME_MATCHED` control matches checkpoint clocks
only. That limitation is now written into the control's own docstring rather than
left implicit, and the pilot's DEMAND term is reported as
**demand + cumulative writing + exposure schedule**, never as static capacity.

The proposed target-slot control (preserve each target exposure's role and
event-slot timing, replace every non-target teaching slot with a 330 s rest, and
compare target-alone canonical / target-alone on the high-k slot schedule /
high-k shared learner) is **not yet built**. It is the next control, because it
is the one axis the present pilot still bundles.

Note these two controls answer different questions and neither subsumes the
other: mine contrasts *k cohorts on one bounded state* against *the same k
cohorts on k separate states, identically aged*; the target-slot control contrasts
*the same target facts, same slot schedule, shared vs unshared state*.

---

## 5. Topology/`codes` caching (audit §5) — accepted, logged for Workstream B

`evaluate_life` encodes once at the start (`codes = backend.encode(a, world["pn"])`)
and all later teaching and probing consume the cached array. Any Workstream B
intervention that mutates input-side `B` must first establish whether the codes
are refreshed, or the rewiring may never reach the input representation. Not a
defect for the frozen-topology experiment; recorded as a required contract check.

---

## 6. New evidence: the A2 diagnostic pilot

### 6.1 Design

n = 8 lives, 216 cells, ~20 min wall time. Purpose is mechanism, not
re-confirming behavioural effects.

The control is a **matched aggregate**, not an arbitrary baseline world (this
follows Mr. Zhou's correction). If FPF(k) is built from W₁…W_k, the control runs
each of those *same* worlds alone — unaged, and aged with pure rest to identical
checkpoint clocks — and pools their counts. World difficulty is therefore blocked
on both sides. The decomposition is

```
FPF(k) - unaged(k)  =  [aged(k) - unaged(k)]  +  [FPF(k) - aged(k)]
                        \___ ageing ____/        \_ demand + writing _/
```

Additivity residual is 0.000 pp in every cell (denominators match by
construction: 60k scored observations on both sides).

### 6.2 The confound, measured

| arm | teach | final elapsed | vs k1 |
|---|---|---|---|
| k=1 canonical | 1,728 | 2,298,240 s | 1.00× |
| k=8 fixed_per_fact | 13,824 | 6,289,920 s | 2.74× |

Per-checkpoint the ratio is worse than the average: **8.00×** at the first
checkpoint, narrowing to 2.74× at the last. The confound is largest exactly where
early acquisition is measured.

### 6.3 Observer inertness — proven, not asserted

The state observer is read-only. Running the same life clean and observed:
`metrics` bit-identical, `checkpoints` bit-identical, and final
`v88.array_digest` identical
(`1697ced92ff3690eb94eb446a8f726e6dc48771885eeb71ff8e8f2d6645b5a49`).
It records per-KC cumulative |Δ| over the complete mutable state
(`m.fast[:,0]`, `m.fast[:,1]`, `m.slow`, `m.adapt` = 165,664 B), and derives
Gini, `n_eff = (Σw)²/Σw²` and touched-KC count. `n_eff` is reported *beside*
`touched`, never instead of it: on a thin-tail vector `touched` reads 5177 while
`n_eff` reads 10.

Caveat stated in the code: this is accumulated state difference, **not certified
teaching-write flux**.

### 6.4 Decomposition (all endpoints, all k)

| k | endpoint | TOTAL | AGEING | DEMAND | ageing share | ageing sign | demand sign |
|---|---|---|---|---|---|---|---|
| 2 | current_usefulness | −7.68 | −0.30 | −7.38 | 4.0% | 3−/3+ | 8−/0+ |
| 2 | acquisition | −7.11 | −0.45 | −6.66 | 6.3% | 3−/3+ | 8−/0+ |
| 2 | anomaly_recovery | −8.44 | −1.56 | −6.87 | 18.5% | 4−/0+ | 8−/0+ |
| 2 | revision | −7.19 | −1.88 | −5.31 | 26.1% | 4−/1+ | 7−/0+ |
| 4 | current_usefulness | −14.30 | +0.07 | −14.37 | −0.5% | 2−/5+ | 8−/0+ |
| 4 | acquisition | −14.79 | −0.65 | −14.14 | 4.4% | 7−/1+ | 8−/0+ |
| 4 | anomaly_recovery | −15.47 | −1.72 | −13.75 | 11.1% | 6−/0+ | 8−/0+ |
| 4 | revision | −10.78 | −1.72 | −9.06 | 15.9% | 6−/1+ | 8−/0+ |
| 8 | current_usefulness | −23.42 | +0.27 | −23.69 | −1.2% | 2−/6+ | 8−/0+ |
| 8 | acquisition | −20.06 | −0.31 | −19.75 | 1.5% | 4−/3+ | 8−/0+ |
| 8 | anomaly_recovery | −23.20 | −0.86 | −22.34 | 3.7% | 6−/0+ | 8−/0+ |
| 8 | revision | −17.66 | −0.86 | −16.80 | 4.9% | 4−/0+ | 8−/0+ |

The ageing term is ≤ 1 pp at k = 8 with sign agreement at chance. The demand term
is unanimous (8−/0+) in 11 of 12 cells. Ageing shares are largest (18–26%) only
at k = 2, where the total is smallest.

### 6.5 Fresh-after-rest — plasticity is not damaged by elapsed time

Aged and unaged arms differ **only** in inserted boundary rest: identical prior
writing, identical fresh cohort, built with the frozen `t_lifetime`. Scored on
the fresh cohort alone (segment-1 checkpoints):

| endpoint | aged − unaged | naive CI | sign |
|---|---|---|---|
| acquisition | **+7.44 pp** | [+4.42, +10.46] | 0−/8+ |
| current_usefulness | +2.86 pp | [+1.24, +4.49] | 0−/8+ |

An aged learner acquires a brand-new standard cohort *better*, unanimously.
Consistent with fast/adapt decay freeing capacity, not with plasticity ageing.

### 6.6 State signature, k = 8 FPF vs matched k = 1 baseline

| diagnostic | k=1 | k=8 FPF | direction |
|---|---|---|---|
| write_mass_total | 4.57 M | 37.3 M | ↑ 8.2× |
| write_concentration (Gini) | 0.6754 | 0.6423 | **FALLS** |
| write_n_eff | 1,664 | 1,854 | **RISES** |
| writable_kcs_touched | 4,747 | 4,851 | rises |
| mean_abs_margin | 12.57 | 10.50 | falls (−16%) |
| clip_fraction (final) | 0.1316 | 0.1800 | rises |
| expression_readout_gap | 0.0091 | 0.0109 | +0.18 pp (negligible) |

### 6.7 Failure map with all inputs supplied

```
MODE: UNRESOLVED
 SATURATION            4/5 matched, 0 unknown, CONTRADICTED
 INTERFERENCE          4/5 matched, 0 unknown, CONTRADICTED
 EXPRESSION_READOUT    1/2 matched, 0 unknown, CONTRADICTED
 INSUFFICIENT_EXPOSURE 2/5 matched, 0 unknown, CONTRADICTED
```

SATURATION requires `write_concentration: rises`; measured, it **falls** — write
mass spreads over *more* KCs as demand grows. EXPRESSION_READOUT requires margin
to hold while the gap rises; the gap is flat (+0.18 pp) and margin falls.

The observed pattern — 8× write mass, broader spread, ~16% margin erosion, mild
clipping, intact readout — is not any of the table's four modes. We do **not**
name a mechanism family from it. Per the audit's §6.5 we also do not treat row
completion as a precondition for the next experiment, nor as proof of uniqueness.

---

## 7. Corrections to my own earlier claims

1. **Withdrawn:** "the revision stratification saved us from understating
   revision by 11.25 pp". That 11.25 pp was the defect, excluding 140 of 160
   legitimately taught facts. Corrected k=8 fixed-budget revision is 63.75%.
2. **Withdrawn:** "completing the SATURATION row would point at M5 + M9". With
   the row completed, SATURATION is contradicted.
3. **Withdrawn:** "`k=8 FPF − k=1` is a real capacity limit". It confounds
   demand, elapsed time and cumulative writing; the ageing part is now measured
   as small, but the remainder is still not static capacity.
4. **Softened:** the world-reuse claim, per §3.

Two operational errors also worth recording: I briefly ran two pilot instances
concurrently into one output directory (detected via interleaved partial lives;
killed, wiped, re-run singly), and my first pilot stored only summaries, so the
scoring fix forced a full re-simulation. The pilot now persists checkpoints
(216 files) so any future rescoring is offline.

---

## 8. What is frozen and preserved

| item | state |
|---|---|
| `Continuous_life_runner_complete/` | untouched |
| `SOURCE_LOCK.json` | untouched (mtime 2026-09-19 11:31:48) |
| `core_recovery_workstreamA_worlds.py` | unchanged, `e811fd2737acc523…`, matches PROTOCOL_LOCK |
| stopped confirmation lives | 4 complete A1 lives (8 result files) preserved, all valid JSON |
| A1/A2/A3 gates | BOUNDARY / CAPACITY / METRICS / PROTOCOL_LOCK / OBSERVER_INERTNESS all PASS |

The confirmation interruption is recorded as a **measurement/design audit**, not
outcome-based selection: it was stopped because the failure map could not
complete and because the cross-k control did not exist, both decided before any
confirmation contrast was computed.

---

## 9. Scope of the pilot's inference

- 8 lives × 9 worlds drawn from 32; **max cross-life overlap 8 of 9**; only **3**
  fully disjoint lives are possible.
- Supported estimand: **conditional-on-panel** E[Y | these canonical worlds].
  Generalization to newly generated worlds is *not* supported by this panel.
- Naive SE 0.515 pp vs delete-one-world sensitivity spread 1.898 pp (3.7×) for
  the k = 8 DEMAND term. The spread is reported as sensitivity only.
- Direction claims rest on sign agreement, which needs no distributional
  assumption; the unanimous 8−/0+ pattern is the load-bearing evidence.
- `margin → accuracy` is **not** established: a −16% margin change accompanies a
  −23 pp accuracy change, and no link between them has been tested.

---

## 10. Questions for you

1. Does the matched-aggregate control satisfy your §4 concern for the *ageing*
   term specifically, or do you want the target-slot control before any ageing
   conclusion is recorded?
2. Given all four signature rows are contradicted, do you want the signature
   table revised (e.g. a superposition/dilution row characterised by rising
   write mass with rising `n_eff`), or dropped in favour of direct intervention?
3. For generalization beyond the 32-world panel: generate fresh canonical worlds
   via `prepare_worlds` (cue-disjointness is enforced by the instrument), or
   keep the conditional-on-panel scope and say so explicitly in all outputs?
4. Is there value in offline-rescoring the existing full exploratory checkpoints
   with the corrected revision and gap definitions, given the A1/A3 behavioural
   directions are unlikely to move? My inclination is yes for the revision
   contrasts only, since those were the ones actually corrupted.

---

## 11. Manifest and reproduction

Modules (sha256, first 16):

```
core_recovery_workstreamA_worlds.py      e811fd2737acc523   (frozen, unchanged)
core_recovery_workstreamA_metrics.py     8357351451ff0484   (both defects fixed)
core_recovery_workstreamA_decide.py      0663d2e6a5127dec
core_recovery_workstreamA2_observer.py   32d1abe34c4899e0   (new, inertness proven)
core_recovery_workstreamA2_controls.py   f8db56f54a67c056   (new)
core_recovery_workstreamA2_pilot.py      3e71760a7a6ceff9   (new)
core_recovery_workstreamA2_analyse.py    a80f64b742b14842   (new)
```

Reproduce:

```bash
# gates
python core_recovery_workstreamA_metrics_qual.py
python core_recovery_workstreamA_boundary_qual.py --worlds '<canonical glob>' \
       --teach-seconds 330.0 --selftest --native
python core_recovery_workstreamA_capacity_qual.py --k 2 --teach-seconds 330.0 \
       --selftest --native

# diagnostic pilot + analysis (~20 min, n=8)
python core_recovery_workstreamA2_pilot.py --n-lives 8
python core_recovery_workstreamA2_analyse.py
```

Interpreter: `V88/.venv-rce-v87-followup/bin/python` (numpy 2.2.6, numba 0.61.2,
scipy 1.14.1). scipy is a hard requirement — a hand-rolled incomplete beta was
wrong by up to 4× at df = 15 earlier in this project and the code now refuses to
approximate a p-value.
