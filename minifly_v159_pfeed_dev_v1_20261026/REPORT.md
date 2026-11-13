V159 PFEED V1: X2 FAIL as specified (variant FAIL); X3 not run; X4 EVEN keys -0.096 bits. DEV_NOT_EVIDENCE (synthetic streams and development text).

# V159 report: a predictive feedback layer (26 October 2026)

**Question:** does a locally learned prediction layer, fed back through a fading trace, let MiniFly generalise across symbols that share a future?

**Answer (as specified):**
* **No.** Added to the same readout as the KC code (PF-READ), the feedback did not improve generalisation on either seed. The design's one variant failed too, so the tested form is falsified and X3 (text) was not run.
* **A post-hoc diagnostic narrows this.** The feedback trace does carry transferable class information: read on its own, it reaches 0.80–0.82 on unseen members. A joint readout ignores it, because the member-specific KC features fit the training data.
* **Separately, X4 found a technical gain:** equalising receptor fan-out improves the frozen code by 0.096 bits/symbol.

* **Design:** `minifly_v159_pfeed_design_20261026/DESIGN.md`. §6 is the testbed-driven readout redesign.
* **Grant:** `AUTHORIZATION.json` (default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (6 attempts), `runlog/CYCLES.jsonl` (testbed *continue*, technical *stop*), `runlog/AMENDMENTS.jsonl` (3 rows: one testbed redesign, two operational).

## C0 testbed (evaluator-side, before any predictive-layer code)
**X2 task:** synthetic class transfer. Bytes a–h and i–p each reveal their class through a following byte. The target after "s -" depends on the class of s. During training, only half of each class's members ever appear before a target.

| Attempt | Readout | FROZEN unseen accuracy (seed 0 / 1) | ORACLE unseen accuracy (seed 0 / 1) | Verdict |
|---|---|---:|---:|---|
| 1 | as designed | 0.638 / 0.607 | 0.723 / 0.694 | CANNOT_TEST (ceiling < 0.90) |
| 2 | §6: extra channels normalised to the KC block, wider L2 grid | 0.634 / 0.616 | 0.940 / 0.971 | CONTINUE |

* **Why attempt 1 failed:** with L2 regularisation, the readout explains the seen members far more cheaply through many member-specific KC features than through a few shared channels.
* **Caveat:** ORACLE chose the top of the widened L2 grid in attempt 2.

**X4, even fan-out (*Thursday*, R1):**
* EVEN keys keep each KC's inputs and weights but give every receptor channel 313 or 314 synapses. They scored **2.867** against FROZEN's 2.963: −0.096 bits, line-bootstrap SD 0.011.
* MiniFly's birth matrix is very uneven: per-channel fan-out has a coefficient of variation of 0.91, from 1 to 1,154 KCs, and 323 KCs have no input.
* This is the largest gain to the frozen code found so far without learning: about 40% of the dense learned reference's 0.22-bit headroom.

## C1 technical
* **Code checks: 5 of 5 pass.** They cover the delta-rule step against the formula, batch against per-position prediction, a feedback trace that holds only earlier predictions (no look-ahead), resets at line starts, and determinism.
* **X1 failed.**

  | Stream | Same-future index of r | Required | FROZEN codes |
  |---|---:|---:|---:|
  | seed 0 | 0.010 | ≥ 0.5 | 0.001 |
  | seed 1 | 0.011 | ≥ 0.5 | 0.004 |

  The index is a plain cosine. r predicts non-negative receptor patterns, so all r vectors share a large common component, and plain cosine is then near 1 for every pair. The index was a poor choice of measure for graded r. It is reported as failed, as specified.
* **X2 failed (unseen-member accuracy; pass required PF-READ ≥ 0.80 with FROZEN and SHUFFLED ≤ 0.65):**

  | Arm | Seed 0 | Seed 1 |
  |---|---:|---:|
  | PF-READ (delta rule, γ = 0) | 0.651 | 0.595 |
  | SHUFFLED | 0.647 | 0.612 |
  | PF-READ, variant (TD target, γ = 0.5) | 0.647 | 0.603 |
  | SHUFFLED, variant | 0.647 | 0.603 |
  | FROZEN (testbed) | 0.634 | 0.616 |

  Seen-member accuracy was 0.99–1.00 in every arm. With the variant, PF-READ and SHUFFLED are identical, so the readout made no use of the feedback channels.

## Post-hoc diagnostic (not gating; written after the failures and labelled as such)
| | Seed 0 | Seed 1 |
|---|---:|---:|
| D2: R1 on the feedback trace **alone** (centred), unseen members | **0.817** | **0.802** |
| D2: the same, seen members | 0.801 | 0.777 |
| D2b: R1 on KC code + centred, normalised trace, unseen members | 0.655 | 0.612 |
| D4: r just after a class byte, projected on mean φ(F1) − mean φ(F2): mean margin, class 1 minus class 2 | 0.038 | 0.039 |

**Reading:**
* The feedback trace carries class information, and it transfers to members never seen before a target. On its own it generalises about equally to unseen and seen members, so it holds no member-specific memory. Centring and normalising it does not stop a joint readout from relying on the member-specific KC features instead.
* **The failure is a competition between the specific pathway and the shared one,** not an absence of shared structure.
* D4's fixed zero threshold gave 0.84 on seed 0 and 0.50 on seed 1. That is because of a common offset in the projection: the class margins differ by the same amount on both seeds.
* **Missing control:** no shuffled trace-alone readout was run.

## What this means
1. **A learned prediction of the future does produce shared, transferable structure.** That is the property the KC layer cannot supply (V155, V158).
2. **Merging it into the KC readout does not work.** A memorising pathway beside a generalising one, both feeding one readout trained on the same data, lets memorisation win. Any design needs to keep the two apart and arbitrate between them, for example:
   * separate predictions from the specific (KC) pathway and the shared (feedback) pathway;
   * deferring to the shared pathway when the specific memory has no evidence for the current context.

   This is the pattern of complementary learning systems. It is a hypothesis for a next design, not a result.
3. **Equalising fan-out is a cheap fix to MiniFly's frozen code** that is worth carrying forward.

## Accounting
* 0.875 CPU-h of child CPU over 6 attempts (C0 0.40, C1 0.47), including two failed operational attempts (10.7 CPU-s) and two failed science attempts (X2 and its variant, recorded as failures).
* 3,602 learner records (predictive-layer line presentations), 1 of 3 test cycles, designs tried k = 3.
