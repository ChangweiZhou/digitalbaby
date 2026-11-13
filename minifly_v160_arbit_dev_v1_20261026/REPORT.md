V160 ARBIT V1: X2 FAIL as specified (PoE 0.740 / 0.702 on unseen members, pass >= 0.80); X3 not run. DEV_NOT_EVIDENCE (synthetic streams).

# V160 report: separate pathways combined as a product of experts (26 October 2026)

**Question:** if the specific KC pathway (S) and the shared predictive pathway (G) each learn their own prediction and are then combined, does MiniFly generalise across members of a class that share a future?

**Answer:**
* **Partly.** The combination improves accuracy on members never seen before a target by 0.09–0.11, and leaves memory of seen members intact.
* **It falls short of the pre-stated 0.80,** because S is confidently wrong on contexts it has never seen, which drags the product down.
* By the design (no variant), V160 stops here, and X3 (text) was not run.

* **Design:** `minifly_v160_arbit_design_20261026/DESIGN.md`. §6 replaced one check, before any code.
* **Grant:** `AUTHORIZATION.json` (default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (2 attempts), `runlog/CYCLES.jsonl` (testbed *continue*, technical *stop*).

## C0 testbed (evaluator-side, before learner code)
V159's class-transfer streams, seeds 0 and 1. Accuracy at target positions, given as unseen / seen members.

| Arm | Seed 0 | Seed 1 |
|---|---:|---:|
| S alone (floor ≤ 0.65 on unseen) | 0.634 / 0.996 | 0.616 / 1.000 |
| Oracle expert alone (R1 on the true-class trace) | 1.000 / 0.989 | 1.000 / 0.996 |
| ORACLE-PoE (ceiling ≥ 0.90 / ≥ 0.95) | 0.945 / 1.000 | 0.934 / 1.000 |

The verdict was CONTINUE. Even a perfect shared expert loses 5–7 points on unseen members once multiplied with S. This was recorded at C0, before the learner arms ran.

## C1 technical
* **Checks: 5 of 5 pass.**
  * PoE equals the normalised product;
  * β = 0 gives S exactly;
  * the predictive layer is deterministic;
  * G alone reproduces V159's diagnostic exactly (0.8170 / 0.8017);
  * validity: G-SHUF alone is at chance (0.498 / 0.500), which supplies the control V159 lacked.
* **X2 failed.**

  | Arm | Seed 0 (unseen / seen) | Seed 1 (unseen / seen) | Bits at targets, unseen (seed 0 / 1) |
  |---|---:|---:|---:|
  | S alone | 0.634 / 0.996 | 0.616 / 1.000 | 1.37 / 1.57 |
  | G alone | 0.817 / 0.801 | 0.802 / 0.777 | 4.68 / 4.63 |
  | G-SHUF alone | 0.498 / 0.510 | 0.500 / 0.500 | – |
  | **PoE (β = 1)** | **0.740 / 0.996** | **0.702 / 1.000** | 2.22 / 2.48 |
  | PoE-SHUF | 0.638 / 0.996 | 0.603 / 1.000 | 1.97 / 2.25 |

  Pass required PoE ≥ 0.80 on unseen and ≥ 0.95 on seen members, with PoE-SHUF ≤ 0.65.

## What it shows
1. **Keeping the pathways apart recovers part of the shared pathway's generalisation without losing memory.** V159's joint readout added nothing (PF-READ 0.651 / 0.595, equal to SHUFFLED). The product adds 0.106 and 0.086 over S, while seen members stay at 0.996–1.0. The shuffled control adds nothing.
2. **The shortfall comes from the specific pathway's overconfidence on novel contexts.** S has no way to say "I have not seen this", so a product cannot defer to G when S is wrong but confident. The same loss appears with a perfect shared expert (ORACLE-PoE 0.94, against 1.00 for the expert alone).
3. **Probability quality got worse while choice accuracy improved.** At target positions on unseen members, the bits rose from 1.4–1.6 (S) to 2.2–2.5 (PoE). G on its own spreads probability over all symbols, and β = 1 lets that spread through. On text, where loss is measured in bits, β would have to be chosen on calibration, as X3 specified.

## Next (hypothesis, for the owner's decision)
* **The missing piece is a signal of how familiar a context is to the specific memory,** so that S's weight falls when it has no evidence.
* The fly has a natural candidate: novelty detection through KC→MBON depression, in which output synapses weaken for patterns already stored (Dasgupta et al. 2018, cited in Senge 2026).
* A small X2 test of familiarity-gated arbitration would follow the same pattern as V160, with the same streams, controls and pass marks.

## Accounting
* 0.134 CPU-h (C0 0.062, C1 0.072).
* 1,200 learner records (seed 0's predictive layer was learned twice, for the determinism check).
* 1 of 3 test cycles; designs tried k = 1.
