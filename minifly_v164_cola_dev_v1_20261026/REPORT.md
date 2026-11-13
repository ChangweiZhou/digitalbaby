V164 COLA V1: FAIL. On the task where credit across time matters (S-COPY), COLA recovers none of BPTT's 1.32-bit advantage over one-step truncation. DEV_NOT_EVIDENCE.

# V164 report: a small test of COLA itself (26 October 2026)

**Question:** where credit across time demonstrably matters, does COLA's local rule recover what exact back-propagation through time (BPTT) gains over one-step truncation (TBPTT-1)?

**Answer: no.** On S-COPY, COLA matches TBPTT-1 to within 0.001 bits and recovers 0% of the 1.32-bit gap. On text it is slightly worse than TBPTT-1.

**Files:**
* Design: `minifly_v164_cola_design_20261026/DESIGN.md`.
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`:
  * 6 attempts, 2 of them failed operationally and fixed by amendments 1 and 3;
  * 1 attempt invalid, caught by the text parity check and fixed by amendment 2;
  * cycles testbed *continue*, technical *continue*, dev *stop*.

## C0 testbed (references before COLA code)
* Network: 256 tanh units with a linear softmax readout; near-critical gain 3.15 from the paper's two-point estimate (λ about −0.11).
* All arms share one schedule: batches of 16 lines, AdamW.
* The text references reproduce V163's calibration curves and gain exactly.

| Task (test bits per scored position) | RAND-R | TBPTT-1 | BPTT | TBPTT-1 − BPTT (SD) | Qualifies (≥ 0.3 bits, ≥ 3 SDs) |
|---|---:|---:|---:|---:|---|
| S-COPY: target x_{t−8} | 2.858 | 2.473 | **1.152** | 1.321 (0.012) | yes |
| S-SUM: target (x_{t−2} + x_{t−6}) mod 8 | 3.014 | 2.987 | 2.871 | 0.115 (0.006) | no |
| Text (*Thursday*, at most 4 epochs) | 3.945 | 3.006 | 3.015 | −0.009 (0.006) | descriptive |

All synthetic references stopped at the 10-epoch budget, so their numbers are lower bounds.

## C1 technical: 4 of 4 checks pass
* With μ fixed at 0, COLA's gradient equals TBPTT-1's autograd gradient (3.5e-17 on S-COPY, 8.3e-17 on text).
* The teaching signal is exactly Eq. 23, including the cap.
* α and μ equal a hand-computed case of Eqs. 25–30.
* Training is deterministic.

## C2 dev
| Task | COLA | TBPTT-1 | BPTT | COLA's gain over TBPTT-1 (SD) | Share of the BPTT gap |
|---|---:|---:|---:|---:|---:|
| **S-COPY** | 2.474 | 2.473 | 1.152 | −0.001 (0.0002) | **0%: FAIL** |
| Text | 3.047 | 3.006 | 3.015 | −0.041 (0.004) | – |

**COLA's statistics:**
* **S-COPY:** α median 0.0006 (5–95%: −0.008 to 0.014); μ median 0.09; μφ′ never at its cap.
* **Text:** α near 0; μ median 0.85; at the cap on 0.07% of steps.

## What it shows
1. **COLA's teaching signal cannot carry credit to an earlier step that had no error of its own.**
   * δ̃_t = e_t / (1 − μφ′_t) is always a per-unit rescaling of the *current* step's error e_t.
   * In S-COPY, the information needed at step t (write x_t somewhere it survives) only pays off at t + 8. At t, the error says nothing about it.
   * BPTT sends that later error back to t; COLA cannot.
   * Its statistics show the mechanism directly. The step-to-step continuation α of the error is about 0, because consecutive targets are independent. So the closure contributes nothing, and COLA reduces to TBPTT-1.
2. **This matches the paper.**
   * COLA's wins there are on tasks where the error itself is temporally persistent: Adding, and rollout regression with smooth targets. That is exactly Assumption 1, δ_{t+1} ≈ α δ_t.
   * On character language modelling it matched TBPTT-1.
   * Prediction tasks with independent per-step targets (bytes of text, symbol recall) violate Assumption 1.
3. **On text it is mildly worse than TBPTT-1.** μ around 0.85 amplifies the immediate gradient by roughly 1/(1 − μφ′), which acts like a larger, uneven step size.
4. **For MiniFly:**
   * COLA is not a route to credit across time for byte prediction or for tasks like addition, where a digit's correct handling pays off only steps later with no immediate error in between.
   * That role still needs something that genuinely transports later errors back: BPTT, or exact online methods such as RTRL and its approximations.
   * Credit across time remains a non-issue on text at 40 KB (V163, and again here).

## Limits
* Small synthetic tasks, one seed, and AdamW instead of the paper's SGD (stated in the design).
* The synthetic references are cut off at 10 epochs. COLA's S-COPY curve was still falling at epoch 10 (2.495 → 2.469 in calibration), but it tracks TBPTT-1's, not BPTT's.

## Accounting
* 0.070 CPU-h over 6 attempts.
* 3,852 learner records (3,820 dev, 32 technical).
* 2 of 3 test cycles; designs tried k = 4.
