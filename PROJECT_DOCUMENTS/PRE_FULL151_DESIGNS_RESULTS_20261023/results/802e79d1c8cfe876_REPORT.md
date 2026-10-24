# MiniFly V9.5 — four-cohort byte-learning lifetime stress: report

**Date:** 2026-09-25.
**Protocol:** `../DESIGN.md` (Codex) with the binding amendments in `../REFINEMENT_AUDIT.md` (A1–A9).
**Implementation:** `lifetime_assay.py`, `test_lifetime.py`; self-audit in `CODE_AUDIT.md`.
**Run:** 256 sealed worlds (401–656), 10.2 minutes of wall time on 4 workers. All validity checks passed.

## Verdict

**Interpretable, and harmful on both primary readouts.** All three learning and floor gates passed. After four contiguous, successfully learned new byte cohorts, the inherited value pathway leaves the old byte association weaker 24 h later than the identical history without writing.

| Primary: WWWW − NNNN, old set, at D4 (n = 256 worlds, 97.5% t) | Estimate | Label |
|---|---|---|
| Deterministic choice | **−11.3 pp [−14.5, −8.2]**; SD 22.4 pp; worlds −96 / 0: 143 / +17 | `PERSISTENT_WRITE_HARM_CHOICE` |
| Analog margin | **−0.0227 Hz [−0.0264, −0.0191]**; worlds −208 / +48 | `PERSISTENT_WRITE_WEAKENING_MARGIN` |

**Robustness:**
- **Bootstrap** (10,000 resamples, seed 20260926): choice [−14.6, −8.2] pp; margin [−0.0263, −0.0191] Hz.
- **Labels are stable:** no label changes under bootstrap, and no single world's removal changes a label.
- **Recomputed independently** from the raw receipts, with identical results.
- **The optional "< 5 pp cost" statement does not apply**, because the whole interval lies below −5 pp.

## Gates, all passed (95% t)

| Gate | Result |
|---|---|
| G1: old set acquired | OLD_WRITE 73.3% [70.6, 76.1]; minus OLD_NO_WRITE +24.2 pp [+21.1, +27.3] |
| G2: each cohort acquired by its own writes, against a same-prestate no-write shadow | cohort 1: 86.5%, +35.8 pp · cohort 2: 84.9%, +36.2 · cohort 3: 84.2%, +32.6 · cohort 4: 86.5%, +36.7 (every lower bound > 0.5 and > 0) |
| G3: no-write old baseline above floor at D4 | NNNN 72.9% [70.2, 75.7] |

Last-only acquisition was verified too: NNNW learns cohort 4 at 87.5%, +38.4 pp over NNNN.

## When the harm arises (secondary, descriptive)

**Old-set choice, write history minus no-write history:**

| Point | W − N | Incremental write effect of that cohort (same prestate) |
|---|---|---|
| E1 | −8.5 pp [−11.8, −5.2] | (= W − N) |
| E2 | −10.0 pp [−13.5, −6.5] | W − S₂: −7.4 pp [−10.5, −4.3] |
| E3 | −14.5 pp [−17.8, −11.1] | W − S₃: −6.6 pp [−9.6, −3.7] |
| E4 | −15.9 pp [−19.5, −12.3] | WWWW − WWWN: −3.7 pp [−6.8, −0.5] |
| D4 (+24 h) | −11.3 pp (the primary) | WWWW − WWWN: −2.5 pp [−4.6, −0.3] |

**Decomposition at D4:**

| Contrast | Choice | Margin (Hz) |
|---|---|---|
| First three writes (WWWN − NNNN) | −8.9 pp [−11.6, −6.1] | −0.0185 |
| Fourth write after three (WWWW − WWWN) | −2.5 pp | −0.0042 |
| Fourth write alone (NNNW − NNNN) | −2.6 pp | −0.0077 |
| History × fourth-write interaction | +0.1 pp [−2.6, +2.8] | +0.0035 |

At E4 the interaction was +6.4 pp: immediately, a fourth write costs less after three earlier ones. At D4 this sub-additivity has gone for choice.

**What these numbers show:**
- **Every cohort's writing costs the old set something.** The cost accumulates through cohort 3, partly recovers over the final 24 h, and persists.
- **The recovery is not V9.4's.** V9.4 saw a single cohort's choice cost vanish at +24 h. After four cohorts, the cost remains at −11.3 pp.
- **The exposure-only benefit is gone by 24 h.** At D4, NNNN equals CLOCK_ONLY (difference 0.0), so seeing bytes without writing has no lasting effect.

**Pair-level losses.** 563 of the 768 old pairs were correct just before the challenge. Of these, WWWW had lost **123** by D4, against **3** for NNNN and 3 for CLOCK_ONLY (179 against 14 at E4). The worst old pair's write effect per world averaged −0.070 Hz at D4.

**Joint competence (WWWW choice):**

| Point | Old | New 1 | New 2 | New 3 | New 4 |
|---|---:|---:|---:|---:|---:|
| E1 | 0.690 | 0.865 | | | |
| E2 | 0.676 | 0.755 | 0.849 | | |
| E3 | 0.633 | 0.711 | 0.695 | 0.842 | |
| E4 | 0.617 | 0.680 | 0.668 | 0.673 | 0.865 |
| D4 | 0.616 | 0.690 | 0.656 | 0.685 | 0.741 |

All five sets stay above chance at D4 (the old set's lower bound is 0.586), so the narrow five-set competence label holds. Earlier cohorts, however, also lose 15–20 pp once later cohorts are written.

## Exploratory and sensitivity

- **Overlap load (A1).** Old pairs sharing symbols with 2 cohorts and with 3 cohorts show the same margin effect at D4 (−0.0231 and −0.0225 Hz). The harm is **not graded by symbol sharing**.
- **KC overlap.** Old-versus-new code overlap (Jaccard) is about 0.70 for every cohort, the same as in V9.4. This shows overlap exists; it does not identify the cause.
- **Assumed readout noise.** With independent Gaussian noise on each option (evaluator-side), the choice effect at D4 is −9.7 pp at SD 0.02 Hz and −8.0 pp at SD 0.05 Hz.

## Byte prediction (4 sentinels × 2 authors)

After cohort 4, every sentinel still beats the locked train-only Markov-2 on full and content bytes, so there is no regression label. Drift has grown, though:
- Doyle full-stream bits per byte rise from 2.815 to 2.870–2.883 (Markov-2: 3.001).
- Stevenson rises from 2.728 to 2.784–2.801 (Markov-2: 2.926).
- Content-byte bits rise by 0.09–0.12.

The predictor was identical across all write policies. The drift therefore comes from reading the synthetic stream, not from value writing, and it is larger per cohort than in V9.4.

## Integrity and deviations

**What was checked:**
- **A4 regression.** Before any V9.5 world, the V9.5 engine reproduced V9.4 world 305's saved margins exactly (difference 0.0). Its negative control (comparison with world 306's receipt) was caught.
- **Technical gate** (9901–9904) passed:
  - flush equals the passive probe at P and at D4;
  - reward-label inertness on the parent, the old-trained prestate and W3;
  - save/load roundtrip;
  - all six negative controls caught.
- **Every science receipt** passed exact clocks, prestates, predictor identity after each record, byte/Teach/write-flag ledgers for all branches (including shadows), the time ledger, the probe roster and purity.

**Deviation (A3).** The implementation was **audited by the implementer only**; no independent code review took place.

**Scope.**
- One fly/model seed: intervals describe byte-world variation, not variation across flies.
- Contiguous cohorts, with three reversed pairs per set.
- An external reward bit.
- This is not the original PN-cue A11 grid.

## What this decides

Under the current FULL_151 value pathway, a 30/14 s teaching dose and this overlapping ordered-byte workload, continued learning **does** leave a persistent cost on older byte associations, about 11 pp after four cohorts. It accumulates while the cohorts continue, and a 24 h rest undoes only part of it. Earlier new cohorts suffer too.

The design cannot identify the site: a synapse, a compartment or a neuron-count limit. The symbol-overlap load does not grade the effect. The next decision (consolidation, spacing, KC coding or topology) needs its own locked experiment.

## Files

- `results/SUMMARY_FULL.json`, `results/SUMMARY_TECHNICAL.json`, `results/SOURCE_LOCK.json`
- 260 per-world receipts (about 17 MB, local only)
- `results/full_run.log`
- `REPORT.md`, `CODE_AUDIT.md`, `../REFINEMENT_AUDIT.md`

To reproduce: run `test_lifetime.py`, then `lifetime_assay.py --phase technical`, then `--phase full` (V88 venv, `OPENBLAS_NUM_THREADS=1`).
