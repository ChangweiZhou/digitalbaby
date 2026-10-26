V160 DESIGN (owner go-ahead; written and committed before any V160 code)

# V160 design: keep the specific and shared pathways apart, and combine their predictions (26 October 2026)

**Owner's words:** "Use the recursive improvement skill again to iterate."

**Basis:** V159 (`minifly_v159_pfeed_dev_v1_20261026/REPORT.md`).
* A locally learned prediction of the next input, fed back as a fading trace z, carries transferable class information. A readout of z alone reached 0.817 and 0.802 on unseen members (post-hoc diagnostic).
* A single readout of [KC code, z] ignored z, because member-specific KC features fit the training data (0.655 and 0.612).

**Scope:** X2 (synthetic, decisive) and X3 (text, only if X2 passes). Default caps: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 1. The hypothesis and the single change
**Hypothesis.** Generalisation fails because the two pathways compete inside one readout. It returns if each pathway learns its own prediction and the two predictions are combined.

**The change.** The specific and shared pathways are trained separately and combined as a product of experts:
* **S (specific):** R1 on the binary KC code (with the empty-state column). This is V159's FROZEN arm.
* **G (shared):** R1 on the feedback trace z alone, centred on the training-stream mean. This is V159's predictive layer, unchanged: delta rule, η 0.1, γ 0, 2 passes over the training stream, then frozen.
* **Combination (PoE):** p(c) ∝ p_S(c) · p_G(c)^β, over the same classes (symbols seen in training plus OTHER).
  * When S has no evidence for a context (a flat p_S), G decides.
  * When S has learned the context, S dominates.

  There is no learned gate.
* **β:** X2 fixes β = 1 before running. X2's calibration stream contains no unseen member, so a β chosen there would only reward S. For X3, β ∈ {0, 0.25, 0.5, 1, 2} is chosen on calibration bits; β = 0 is S alone.

**Reuse.** V159's modules are imported unchanged: the streams, the readout utilities, the predictive layer and the birth-key loader. Birth keys are kept, not EVEN keys, so that the comparison with V159 changes one factor only.

## 2. C0 testbed (evaluator-side, before any V160 learner code)
* **Streams:** V159's class-transfer streams, seeds 0 and 1 (V159 design §2).
* **Arms:**
  * S alone;
  * ORACLE-PoE, in which G is replaced by R1 on the true-class trace alone (V159's ORACLE channels, centred).
* **CONTINUE** if, on both seeds:
  * ORACLE-PoE scores ≥ 0.90 on unseen and ≥ 0.95 on seen members (ceiling);
  * S alone scores ≤ 0.65 on unseen members (floor).
* Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED: stop and report.

## 3. C1 technical
**Checks:**
* PoE equals the normalised product;
* β = 0 gives S exactly;
* the predictive layer reproduces V159's learned matrix bit for bit on seed 0;
* determinism.

**X2 arms** (both seeds; score = accuracy at target positions, as in V159):

| Arm | What it is |
|---|---|
| PoE | S × G, β = 1 |
| PoE-SHUF | S × G-SHUF, where G-SHUF is trained on a trace built from predictions permuted across positions (seed 7) |
| G alone | reported |
| G-SHUF alone | the control missing from V159 |

**Rules:**
* **Pass:** on both seeds, PoE ≥ 0.80 on unseen members **and** ≥ 0.95 on seen members, and PoE-SHUF ≤ 0.65 on unseen members.
* **Validity:** if G-SHUF alone scores > 0.65 on unseen members, the shared pathway's gain is not due to the content of its predictions. That is INVALID, and it stops the delivery.
* **No variant.** If X2 fails, V160 stops and reports.

## 4. C2 dev: X3 text (only if X2 passes)
* **Texts:** *Thursday* as A (40 / 5 / 10 KB; V155's cached traces, SHA-verified) and *Edgeworth* as B (10 / 5 / 10 KB).
* **Predictive layer:** learned on A's taught text (2 passes), then frozen. The traces z are computed on every split of A and B.
* **Arms:** S, PoE and PoE-SHUF.
  * Same-text: readouts on A; β chosen on A's calibration bits.
  * Carry-over: readouts trained on B's 10 KB; β chosen on B's calibration bits.
* **L2 grids:** S on A {100, 30, 10}; S on B {300, 100, 30, 10}; G {10, 1, 0.1, 0.01}. 300 iterations.
* **Pass:** PoE beats S on B's held-out lines by ≥ 0.02 bits **and** ≥ 3 line-bootstrap SDs, and PoE-SHUF does not. The same-text result is reported.

## 5. Accounting, predictions, limits
* **Learner records:** predictive-layer line presentations. C1 is 800; C2 is 1,410.
* **Estimated CPU:** C0 0.15 h, C1 0.2 h, C2 0.5 h.
* **Predictions** (my record is poor):
  * P1: PoE passes X2, probability 0.65;
  * P2: PoE passes X3, probability 0.25.
* **Not tested:**
  * a native (local delta-rule) version of the two readouts and their combination;
  * EVEN keys;
  * own-output generation;
  * unseen addition.

## 6. Pre-code refinement of one check
V159 did not save its learned matrix, so the check "the predictive layer reproduces V159's learned matrix bit for bit" cannot be run as written. It is replaced by a stronger end-to-end parity check: **G alone must reproduce V159's diagnostic accuracies on unseen members exactly** (0.8170 and 0.8017). It uses the same predictive layer, trace, centring, readout and grid.
