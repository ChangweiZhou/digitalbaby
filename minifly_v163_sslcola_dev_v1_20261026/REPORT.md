V163 SSL/COLA V1: lane L PASS by its rule (partial, φ 0.31), but the learned layer does not beat the KC code it reads; lane C (COLA) CANNOT_TEST at C0. DEV_NOT_EVIDENCE.

# V163 report: a local self-supervised layer under a back-propagated top, and COLA (26 October 2026)

**Question:** using the ideas of Zihan et al. (layer-local contrastive predictive learning) and Wang et al. (COLA), with back-propagation allowed only in the top block, does a new layer above MiniFly's frozen KC code learn features that carry over to a new author?

**Answer:**
* **Lane L (local self-supervised layer): a partial result.**
  * The local rule's layer beats the same layer left random by 0.13 bits on carry-over, which passes the pre-stated rule. It recovers 31% of what the same layer gains when back-propagation learns it.
  * But it is still worse than reading the KC code directly with the same top. So it repairs part of what an extra random layer loses; it does not add information beyond the KC code.
* **Lane C (COLA): CANNOT_TEST.** Exact back-propagation through time was no better than one-step truncation (worse, in fact), so COLA had nothing to approximate. It was not built.

**Files:**
* Design: `minifly_v163_sslcola_design_20261026/DESIGN.md`.
* Grant: `AUTHORIZATION.json`. Grant 1 had default caps; grant 2 raised them to 3 CPU-h and 20,000 records. Its departure from AUT-02 is flagged: the numbers came from my request, and the owner chose the full option.
* Ledgers in `runlog/`: 5 attempts; cycles testbed *revise*, testbed *continue*, technical *continue*, dev *stop*; amendment 1.

## C0 testbed (reference runs before any learner code)
**Instrument checks:**
* The probe reproduced the headroom check's T3 exactly (2.69947).
* *Thursday*'s exported codes equal the saved ones.

**Lane L carry-over** (layer learned on *Thursday*; fresh top on 10 KB of *Edgeworth*; held-out bits/symbol, mean of 3 seeds):

| Arm | Carry-over |
|---|---:|
| BP-H: layer learned by back-propagation (ceiling) | **3.070** |
| TOP: no extra layer, top on the KC code | 3.330 |
| V162: R1 on the KC code | 3.320 |
| RAND-H: the layer left random | 3.487 |

* **Ceiling met:** BP-H beats RAND-H by 0.418 bits (SD 0.015; 0.06 required). A layer learned on one author clearly helps another, when back-propagation learns it.
* **Noise check failed at first:** the seed-1 refit moved by 0.044 bits (0.01 allowed). By the design's pre-stated remedy, every lane L carry-over probe is now averaged over seeds 0, 1 and 2 (amendment 1).
* **Recorded probe defect:** for layers trained with labels on *Thursday*, a probe refitted on *Thursday* is best after one epoch and overfits after that. Their same-text numbers are therefore descriptive only. The gating readings do not use them.

**Lane C** (256 recurrent units above the KC code; gain 3.15 from the paper's two-point Lyapunov estimate, giving λ −0.11 at that gain):

| Arm | Same text | Carry-over | Training readout, calibration |
|---|---:|---:|---:|
| RAND-R | 3.245 | 3.795 | – |
| TBPTT-1 | 3.028 | 3.120 | 2.991 |
| BPTT | 3.155 | 3.178 | 3.055 |

* The reference learns: BPTT beats RAND-R by 0.089 (0.05 required).
* **The ceiling fails:** BPTT is 0.127 worse than TBPTT-1 (0.03 better required). It is worse on every reading, so the verdict does not depend on the probe defect.
* COLA was not built. This matches the COLA paper's own character-level result.

## C1 technical: 4 of 4 checks pass
* The autograd update equals the explicit three-factor formula (difference 3.3e-16, float64).
* No gradient from the probe reaches the layer.
* One pass on 50 lines is deterministic.
* After one pass, the true next state scores above a random one on 97.4% of *Thursday*'s calibration pairs (60% required).
* Cost: 12.3 s per pass.

## C2 dev (lane L)
**Training:** 8 passes of the local rule (5,640 records). The contrastive loss fell from 1.62 to 0.088.

**Checkpoints** (*Edgeworth* calibration under the 3-seed probe):

| Passes | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Calibration | 3.371 | 3.306 | 3.282 | **3.258** |
| Held-out (for information) | 3.462 | 3.406 | 3.376 | 3.358 |

8 passes were chosen. Calibration was still improving, so the chosen checkpoint sits at the edge of the budget.

**Gating result:**
* **SSL-H 3.358 against RAND-H 3.487: gain 0.129 bits, SD 0.012. PASS** (≥ 0.03 and ≥ 3 SDs required).
* φ = 0.129 / 0.418 = **0.31**.

**Comparisons that the rule did not gate but that decide the meaning:**
* SSL-H is 0.028 worse than TOP (3.330) and 0.038 worse than V162's R1 on the KC code (3.320).
* With a linear probe, SSL-H scores 3.535 (seed 0), against 3.430 for TOP and 3.074 for BP-H (C0).

**Diagnostics (non-gating):**
* **Same text:** SSL-H with the probe scores 2.851, against TOP 2.783 and RAND-H 2.887 (single seed).
* **Alignment:** the cosine between the local update and the back-propagated next-byte gradient is 0.019, essentially orthogonal.
* **Separation:** the rule solves its own task almost perfectly, 99.7% on *Thursday* and 99.6% on *Edgeworth* calibration pairs. So it generalises across authors, but the task is easy.
* **Units:** no dead units; about 59 of 512 units are active per position.
* **Effective rank:** 138 for SSL-H against 238 for RAND-H. The rule concentrates the code onto fewer directions.

## What it shows
1. **The slot above the KC code has large transferable headroom, but only back-propagation finds it.** Learned by back-propagation on *Thursday*, the layer cuts *Edgeworth* carry-over to 3.07, from 3.32 for the best KC-code readout. That is the largest carry-over gain in the V-series so far, and it is the size of prize a local rule must reach.
2. **The local contrastive rule learns something real, but not what next-byte prediction needs.**
   * It beats its own random start by 0.13 bits and goes on improving with passes.
   * Yet its update is orthogonal to the back-propagated gradient (cosine 0.02), and its layer stays below the KC code it reads.
   * Telling consecutive states from random ones is too easy for byte-level text (99.7% after training), so the objective stops shaping the features long before they become predictive.
3. **By the reading fixed in advance (pass with φ < 0.5), the next step is Zihan et al.'s direct feedback,** where the rule's reference signal comes from the top layer rather than from the layer's own next state. The alignment result supports this: the missing ingredient is the next-byte error.
4. **Credit across time is not a bottleneck at 40 KB.** One-step truncation beat exact back-propagation through time. This is consistent with the COLA paper's character-level result and with PPM-5 being no better than PPM-4 here.

## Limits
* One text pair and one development seed per learner; DEV_NOT_EVIDENCE.
* The SSL-H checkpoint is at the budget edge (8 passes), so more passes might narrow the gap to TOP. The trend (3.376 → 3.358 from 4 to 8 passes) suggests a small effect.
* The top uses back-propagation, so this is not a learner free of back-propagation.
* Lane C trained with AdamW rather than the paper's SGD.

## Accounting
* 0.541 CPU-h of child CPU over 5 attempts:
  * C0: 0.012 export, 0.319 references, 0.092 3-seed revision;
  * C1: 0.006;
  * C2: 0.113.
* 6,445 learner records; 2 of 3 test cycles; designs tried k = 4 (each lock rebuild counts).
