# Workstream A2 follow-up — every item from the review, executed

**Date:** 2026-09-19
**Responds to:** `WorkstreamA2_review_and_next_steps_20260919.md`
**Status:** 64-life confirmation still stopped and preserved. `Continuous_life_runner_complete/` and `SOURCE_LOCK.json` untouched.

---

## 0. What was done

All five numbered steps of the review's "Concrete next run", plus the definite
offline repair and the bookkeeping items:

| item | status |
|---|---|
| Offline rescore of all saved output, corrected revision + gap | done, 184 cells, 0 unverified |
| Final-clipping probe-selection repair | done, root cause removed |
| Relabel AGEING / DEMAND in derived reporting | done |
| Per-contrast world membership instead of plan-level | done, reproduces your table |
| Reader-gate-only upper bound on one endpoint | done, reproduces your identity |
| Slow-component-only concentration | done, reproduces your numbers |
| 64 fresh canonical worlds, qualified generator, cue checks | done |
| Target-slot control, 8 blocks x 3 arms = 24 cells | done, all qualified |
| Observer non-perturbation on the transformed control | done, bit-identical |
| Bookkeeping (stale labels, SHA256SUMS self-hash, header) | done |

Your recalculation of all 288 per-life differences, the reference replay
(20/20, 128/160, 102/160; fixed-budget acquisition 467/1008) and the two negative
examples all agree with our side. Every quantity below that appears in your review
reproduced to the digits you reported.

---

## 1. Offline rescore — only revision moved

`core_recovery_workstreamA2_rescore.py`. No native simulation. Originals
untouched; corrected values written to `CORE_RECOVERY_workstreamA2/rescored/`.

Each cell's world is rebuilt deterministically from `RUN_PLAN.json` and then
**verified against its own saved checkpoints** — stage, phase, probe_type, prefix
width, teach_index, full truth vector and all four index lists must match, exactly
as your reproducer does. A cell that fails verification is reported UNVERIFIED and
excluded rather than rescored on a guess.

**184 cells rescored, 0 unverified, 0 missing** (176 exploratory + 8 preserved
confirmation).

Only the revision endpoint changed. Everything else is bit-identical:

| contrast | endpoint | old | corrected |
|---|---|---|---|
| A2 distractor − wait | current usefulness | −2.60 | −2.60 |
| A3 k=8 budget − per_fact | current usefulness | −22.99 | −22.99 |
| A3 k=8 FPF − k1 | current usefulness | −24.61 | −24.61 |
| A1 rest_only − full | current usefulness | −9.13 | −9.13 |

Revision moved substantially:

| contrast (revision) | old | corrected | shift |
|---|---|---|---|
| A3 k=8 FPF − k1 | −6.87 | **−20.51** | −13.63 |
| A3 k=8 budget − k1 | −26.88 | −37.38 | −10.51 |
| A3 k=4 FPF − k1 | −5.62 | −13.20 | −7.58 |
| A3 k=2 FPF − k1 | −3.44 | −7.66 | −4.22 |
| A3 k=8 budget − per_fact | −20.00 | −16.88 | +3.12 |

So revision was **not** relatively spared under fixed-per-fact; the previous
appearance of sparing was the broken denominator.

**The precision plan.** You were right that `confirmation_n = 64` came from the
fixed-budget k=8 revision variance on the wrong denominator:

| | sd | n required | n final |
|---|---|---|---|
| old | 11.529 pp | 85 | 64 (clamped) |
| corrected | **4.289 pp** | **14** | **24** |

The size-64 plan stays superseded regardless; recorded so nobody re-derives it.

A1's revision was already 320 and did not move: in a concatenated lifetime each
segment's flip lands at its own probe's teach_index, reproducing the same
coincidence that hid the defect in canonical k=1.

---

## 2. Final-clipping selection error — confirmed and repaired at the root

Confirmed exactly. The observer admitted any probe with `len(roles) >= 48`, which
includes the untrained-population diagnostic (`codes[2*total:]`), and that is a
life's LAST probe call. Measured on one life:

| cell | trained_facts | my old "last" | population-aligned final |
|---|---|---|---|
| k=1 | 168 | n=168, clip 0.1146 | same (right by luck) |
| k=2 | 336 | n=48, clip 0.0521 | n=336, clip 0.1235 |
| k=4 | 672 | n=96, clip 0.1016 | n=672, clip 0.1150 |
| k=8 | 1344 | n=192, clip 0.1784 | n=1344, clip 0.2015 |

Repaired two ways, not one. The analysis now selects the last snapshot whose
width equals that cell's `trained_facts`; and the **root cause** is removed — the
observer now persists explicit identity (`probe_population`, `is_trained_prefix`,
`trained_facts`) so no downstream selection infers population from size.

Corrected 8-life means, matching your table:

| k | unaged baseline | aged baseline | FPF |
|---|---|---|---|
| 2 | 0.1403 | — | 0.1474 |
| 4 | 0.1279 | — | 0.1705 |
| 8 | 0.1316 | — | 0.2012 |

Clipping now rises monotonically with k (+53% at k=8), where the old selection
made it fall at k=2.

---

## 3. Relabelling, membership, the gate bound, and the slow channel

**Labels.** Derived reporting now uses `checkpoint_rest_control_effect` and
`shared_FPF_minus_checkpoint_rest_control`, with the note that the three-term
split is an algebraic identity and a zero additivity residual identifies nothing.
Original files and labels are intact for provenance.

**Per-contrast membership** — reproduces your table exactly:

| contrast | worlds/unit | unique | max overlap |
|---|---|---|---|
| A3 k=2 | 2 | 15 | 1 |
| A3 k=4 | 4 | 29 | 3 |
| A3 k=8 | 8 | 32 | 7 |
| fresh-after-rest | 2 | 16 | 0 |

The 8/9 figure we quoted was plan-level and overstated the dependence of every
contrast. The delete-one-world quantity is now labelled a sensitivity spread, not
a standard error, since lives are multi-membership.

**Reader-gate-only bound**, one endpoint, k=8 current usefulness:

| condition | C | G | A |
|---|---|---|---|
| checkpoint-rest control aggregate | 95.2257% | 0.3689% | 95.5946% |
| shared FPF8 | 71.5386% | 1.0308% | 72.5694% |

ΔC = −23.6871 = ΔA (−23.0252) − ΔG (+0.6619), identity residual −3.6e−15.
Relaxing only the fixed-reader gate can recover at most **1.0308 pp**. A counting
identity and a restricted upper bound — not proof that latent memory is absent.

**Slow channel alone**, k=1 → FPF8: mass 260,108 → 2,126,643 (8.176×), effective
support 902.84 → 1003.98 (+11.2%), touched 1651.20 → 1709.50 (+3.5%), Gini
0.83141 → 0.81357. We accept the reading: these do not establish saturation, Gini
is scale-invariant, and `n_eff(write_mass)` is not memory capacity. The recorded
quantity is accumulated absolute state difference during teach calls, pooled
across channels with differing units, and is **not** certified teaching-induced
flux. We do not claim any M-family is excluded; a contradicted Boolean row rules
out that stipulated conjunction only.

---

## 4. Fresh worlds — 64, qualified generator, verified disjoint

`core_recovery_workstreamA2_makeworlds.py` uses `v13.prepare_worlds`, the already
qualified path: a disjoint **global** history index per phase/epoch (V88's
`derive_history_seed` treats every round name except "A" as its B namespace, so a
new name alone recycles seeds), `v88.create_world`, raw-cue reuse forbidden, and
`v88.audit_world` on each result. New phase `a2_targetslot` at base 470000.

- 57 existing campaign worlds inherited as exclusions: **21,888 raw cues**
- 64 new worlds, **24,576 new cues**, global indices 470000–470063
- all 64 seeds distinct; zero collisions within the new set or against the
  inherited set; `audit_world` passes on all 64

One bug found and fixed in our own script before use: the inherited copies were
renamed, so neither our check nor `prepare_worlds`'s `rglob("world_*.json.gz")`
matched them and the exclusion set was silently **empty**. A guard now aborts if
inherited worlds contribute zero cues.

---

## 5. Target-slot control — the schedule costs nothing

`core_recovery_workstreamA2_targetslot.py` + `..._run.py`. 8 blocks of 8 distinct
worlds (64 total, **no world in two blocks**), one preassigned target cohort per
block covering positions 0–7, same target facts scored in all three arms.

Every non-target teach slot is replaced by a rest of the measured 330 s. Target
roles, exposure order, event times, probe times and indices are preserved. Probes
are read-only in this evaluator, so the other cohorts' probes, revision curves and
reactivation probes are kept for slot fidelity without reintroducing learning;
`change_truth` consumes no time and only flips non-target roles, so it is kept.

**Qualification, all 8 blocks pass:** target exposure counts equal in all three
arms; (b) and (c) identical in target first/last times, inter-exposure gaps and
roles; all probe times and widths identical between (b) and (c); total elapsed
identical between (b) and (c); target kept + non-target silenced = the shared
arm's total teach; zero non-target teaching remains. Observer
non-perturbation re-verified **on the transformed control**: metrics, checkpoints
and final state digest bit-identical.

The manipulation is real, not nominal: (a) elapsed 2,298,240 s vs (b) and (c)
6,289,920 s; 1,728 target teach kept and 12,096 silenced; and the target's mean
inter-exposure gap is **23,532 s in (a) vs 79,612 s in (b), a 3.38× stretch**,
with non-identical gap sequences.

### Per-block current usefulness on the target cohort

| blk | target pos | (a) canonical | (b) slot-only | (c) shared | a→b | b→c |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9514 | 0.9514 | 0.6319 | +0.00 | −31.94 |
| 1 | 1 | 0.9583 | 0.9653 | 0.7431 | +0.69 | −22.22 |
| 2 | 2 | 0.9583 | 0.9583 | 0.6944 | +0.00 | −26.39 |
| 3 | 3 | 0.9514 | 0.9583 | 0.6944 | +0.69 | −26.39 |
| 4 | 4 | 0.9444 | 0.9375 | 0.7431 | −0.69 | −19.44 |
| 5 | 5 | 0.9375 | 0.9583 | 0.6736 | +2.08 | −28.47 |
| 6 | 6 | 0.9931 | 0.9931 | 0.8056 | +0.00 | −18.75 |
| 7 | 7 | 0.9514 | 0.9444 | 0.7500 | −0.69 | −19.44 |

### Decomposition, all endpoints

| endpoint | SCHEDULE (b−a) | SHARED STATE (c−b) | schedule share |
|---|---|---|---|
| current usefulness | **+0.26 pp** CI[−0.50,+1.02], 2−/3+/3⁰ | **−24.13 pp** CI[−28.21,−20.05], 8−/0+ | −1.1% |
| acquisition | −0.74 CI[−1.62,+0.13], 4−/1+/3⁰ | −19.79 CI[−23.97,−15.61], 8−/0+ | +3.6% |
| anomaly recovery | −0.62 CI[−3.30,+2.05], 2−/1+/5⁰ | −20.00 CI[−28.06,−11.94], 8−/0+ | +3.0% |
| revision | −1.25 CI[−3.19,+0.69], 2−/0+/6⁰ | −15.00 CI[−25.95,−4.05], 6−/0+/2⁰ | +7.7% |

**Reading.** Delivering the target's identical exposures on the shared schedule's
event slots and clock — a 3.4× stretch of its own spacing — costs nothing
detectable and stays far below the 5 pp threshold on every endpoint. Adding the
other cohorts' learning, with target timing held identical, costs 15–24 pp and is
unanimous in sign.

**What this does NOT say.** (b) → (c) is not automatically pure synaptic
overwrite and not automatically static capacity; both remain consistent with it,
and separating them needs a further intervention. Eight blocks are a diagnostic
design, not a precision guarantee. Blocks use disjoint world sets, so per-block
differences are independent across blocks in stimulus terms; we report that rather
than converting it into a new-population claim.

---

## 6. Bookkeeping

- `PILOT_PLAN.json`'s `estimand_supported` label corrected; the withdrawn text and
  the reason are retained in the file as provenance. Plan geometry and all measured
  values unchanged. `A2_ANALYSIS.json` regenerated.
- `SHA256SUMS.txt`: the previous bundle hashed the sums file while writing it,
  producing an inconsistent self-entry. It is now computed over all other files and
  excludes itself, with that stated in the file.
- Report header: corrected. The stopped item is the **64-life behavioural
  confirmation**; the A2 diagnostic pilot and this target-slot control ARE native
  science runs, deliberately small and diagnostic. Earlier phrasing conflated the
  two.

---

## 7. Where the program now stands

Best supported, and now with the timing confound addressed:

1. Irrelevant-content interference is below the 5 pp practical threshold (A2,
   cross-life overlap 0).
2. At fixed k, reducing exposure costs a great deal more (same-k paired design).
3. Under matched target timing and matched clock, sharing the state with other
   cohorts' learning costs 15–24 pp, while the schedule itself costs nothing.
4. Elapsed time did not damage plasticity in the one arrangement tested: an aged
   learner acquired a fresh cohort 7.44 pp better, 8/8 positive.

Not claimed: any specific M-family; that `n_eff` measures capacity; that the
absent readout gap excludes latent memory; that these panels generalize to newly
generated worlds beyond the 64 we made for the target-slot blocks.

Kept separate, per your instruction: topology/rewiring. The once-per-life
`codes = backend.encode(a, world["pn"])` contract is confirmed, so an input-side
`B` intervention needs a proven live-encoding path and a no-rewire live-encoding
control before its effects mean anything.

---

## 8. Open question

The one thing we did not do is decide between overwrite and static capacity for
the (b) → (c) effect. The natural next contrast is to hold the *amount* of
non-target learning fixed while varying whether it competes for the same
substrate — but that is a design question we would rather agree on with you than
pick unilaterally, since it determines which M-families the next run can speak to.
