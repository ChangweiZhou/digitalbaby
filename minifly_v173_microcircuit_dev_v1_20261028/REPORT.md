V173 MICROCIRCUIT V1: CANNOT_TEST_CLAIM_AS_SPECIFIED. Even with exact temporal credit, isolated gated 8-neuron circuits did not reach half of dense BPTT's delayed-copy gain within the budgets screened (best 1.998 bits against the required 1.81; still improving at the budget edge). No learner arm was run. DEV_NOT_EVIDENCE.

# V173 report: gated recurrent microcircuits with exact within-circuit credit, on delayed copy (28 October 2026)

**Question:** in small, isolated, gated circuits, exact temporal credit is computable forward in time with circuit-local machinery. Does that architecture preserve the delayed-dependency learning that a dense recurrent network achieved only with BPTT?

**Answer: not testable as specified.**
* The architecture's own exact-gradient reference (MC-BPTT) never reached the qualifying level within the screened budgets. This held for 6 configurations, at 10 and 20 epochs, with 2 learning rates and 2 gate initialisations.
* The required level was half of the gap between V164's dense one-step truncation and dense BPTT: M = 1.8125 bits.
* By the identity stated in design §0, the candidate (MC-RTRL) computes the same gradients as MC-BPTT. So its scores here are MC-BPTT's.
* The prestated rule therefore stopped the delivery before any learner arm was run.

**Files:**
* Design: `minifly_v173_microcircuit_design_20261028/DESIGN.md`, committed before code (6b7e0898).
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`: 3 testbed attempts; testbed cycles *revise*, *stop*; amendments 1 (operational) and 2 (testbed_invalid), both before any learner run.

## Setup
* **Task:** V164's S-COPY (target x_{t−8}; 8 symbols; 100/20/50 lines of 120), with V164's schedule (batches of 16 lines, AdamW, clip 5, best calibration epoch kept).
* **MC architecture:**
  * 32 circuits × 8 gated neurons = 256 state units, each circuit reading the one-hot input;
  * q = tanh(A x + R s + b), g = σ(C x + U s + b_g), s ← (1 − g)s + g·q;
  * linear readout;
  * 8,704 recurrent parameters, against about 68,000 in V164's dense RNN.
* **References** (autograd evaluator instruments, 0 records): MC-BPTT, the ceiling; MC-T1, one-step truncation, the floor.

## C0 testbed
**Original grid:** learning rate 1e-3, gates initialised at σ(0) = 0.5.
* testbed-001 crashed after training, in the bootstrap SD (list input). Amendment 1 (operational) fixed it.
* testbed-002 reproduced the numbers exactly:

| Reference | Test bits / scored position |
|---|---:|
| MC-BPTT, 10 epochs | 2.757 |
| MC-T1, 10 epochs | 2.861 |
| MC-BPTT, 20 epochs | 2.431 |
| *V164 dense: RAND-R / TBPTT-1 / BPTT (10 epochs)* | *2.858 / 2.473 / 1.152* |

The calibration curves fell almost linearly and were still falling at 20 epochs. At initialisation the gates halve the state every step. **Verdict:** CANNOT_TEST_CLAIM_AS_SPECIFIED.

**Amendment 2 (testbed_invalid)** extended the grid with generic settings only, and fixed the selection rule before running:
* learning rate 3e-3;
* a "spread" gate-bias initialisation: initial retention times log-uniform over 2–64 steps across circuits, independent of the task's lag.

| Config | Epochs | MC-BPTT | MC-T1 | Floor gap (T1 − BPTT) |
|---|---:|---:|---:|---:|
| G-lr (lr 3e-3, gate σ(0)) | 10 | 2.331 | 2.664 | 0.33 |
| G-spread (lr 1e-3, spread) | 10 | 2.752 | 2.800 | 0.05 |
| G-spread-lr (lr 3e-3, spread) | 10 | 2.484 | 2.684 | 0.20 |
| **G-lr** | **20** | **1.998** | 2.492 | **0.49** |
| G-spread | 20 | 2.555 | 2.713 | 0.16 |
| G-spread-lr | 20 | 2.221 | 2.587 | 0.37 |

No configuration reached M = 1.8125. Every MC-BPTT curve was still falling at its budget edge; the best calibration epoch equals the budget in all six rows. **Verdict:** CANNOT_TEST_CLAIM_AS_SPECIFIED. **Decision:** stop, by the rule fixed in amendment 2.

## What it shows (reference level only)
1. **Temporal credit matters in this architecture, and exact local credit supplies it.**
   * The gap between one-step truncation and exact credit grows with training: 0.33 → 0.49 bits for G-lr.
   * MC-T1 at 20 epochs (2.492) sits at the dense one-step level (2.473).
2. **The limit at these budgets is learning speed or capacity, not credit.**
   * With the same exact credit, the isolated circuits recover 36% of the dense BPTT gain at 20 epochs: (2.473 − 1.998)/(2.473 − 1.152).
   * Dense BPTT reached 1.152 in 10 epochs at a lower learning rate.
   * The asymptote of MC is unknown, because every curve was still falling.
3. **Long initial retention slowed learning.** The spread initialisation was worse than gates at σ(0) at both learning rates.
4. **The candidate's score is known without running it.** MC-RTRL's forward-sensitivity gradient equals MC-BPTT's, because the circuits are isolated and the readout is linear. Its score at each configuration would be the MC-BPTT row. The C1 check of this identity in code was not run, because the stop came first.

## Process deviations
* `forward_grad`, the learner code, existed in `v173_lib.py` before C0. It was called once on 16 lines for a timing smoke test, with no training and no outcome. That deviates from "C0 before learner code".
* testbed-001 crashed after its references had trained; amendment 1 fixed it.
* The grid was extended after the original endpoint failed (amendment 2, testbed_invalid). Both happened before any learner run.

## What a follow-up would need
* **A larger budget at lr 3e-3:** 40–60 epochs. C2 with three arms at 60 epochs needs about 18,000 records, so the owner would have to write a cap.
* Alternatively, fewer arms: MC-RTRL against MC-DIAG only.
* Whether the gap reaches M at any budget is unknown. The decisive question would be whether MC-DIAG falls short of MC-RTRL once the architecture can learn the task.

## Predictions, scored
| Prediction | p | Outcome |
|---|---:|---|
| The ceiling passes at 10 epochs | 0.45 | no |
| P1, P2, P3 | – | unresolved (not run) |

## Accounting
* **CPU:** 0.038 CPU-h, all evaluator references.
* **Records:** 0 learner records.
* **Cycles:** 0 of 3 test cycles used.
