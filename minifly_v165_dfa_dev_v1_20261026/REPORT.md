V165 DFA V1: PASS. A broadcast next-byte error through fixed random feedback recovers 86% of back-propagation's carry-over gain in V163's layer, and beats the KC readout by 0.20 bits. DEV_NOT_EVIDENCE.

# V165 report: direct feedback alignment for V163's layer (26 October 2026)

**Question (the review's P1-A):** V163's forward model stays fixed: the frozen KC code, a 512-unit layer H, and a top block trained by back-propagation. If H learns from the next-byte error broadcast through fixed random feedback, instead of by back-propagation into it, does it close the gap to the back-propagated layer on carry-over to a new author?

**Answer: mostly, yes.**
* On *Edgeworth* carry-over, DFA-H scores 3.126 against BP-H's 3.070. That recovers 86% of the gain over the random layer.
* It beats the direct KC readout by 0.20 bits, which the local self-supervised layer of V163 did not.
* The matched control, the same rule fed errors from other positions, gains almost nothing. So the gain comes from input-matched credit.

**Files:**
* Design: `minifly_v165_dfa_design_20261026/DESIGN.md`.
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`: 3 attempts; cycles testbed *continue*, technical *continue*, dev *stop*; amendment 1 (operational).
* The review's P1-B runs separately as V166.

## The rule
* **Forward pass:** a = W x + b and h = ReLU(a), with x the frozen KC code. The top (2,048 ReLU units, dropout 0.5, softmax) learns by back-propagation within itself, on a detached h.
* **H's teaching signal:** δ = ReLU′(a) ⊙ (e F), where e = (p − onehot(y)) / N and F is a fixed random K × 512 matrix.
* **H's update:** ΔW_ji ∝ −Σ δ_j x_i, the product of presynaptic KC activity, the unit's own derivative and a broadcast error. There is no weight transport and no backward pass through the top.
* **DFA-SHUF:** the same, with the error rows permuted within each batch.
* **Training:** at most 3 epochs on *Thursday* (the record cap), AdamW, the same initialisation and batches as V163's BP-H.

## C0 testbed and C1 technical
* **C0:**
  * V163's BP-H, retrained under the 3-epoch budget, reproduced V163's cached layer exactly;
  * RAND-H's 3-seed carry-over reproduced exactly;
  * ceiling met (BP-H beats TOP by 0.260).
  * Amendment 1: the run's import record was written into V163's runlog, because V163's `runtime.py` shadowed V165's. V163's file was restored and the record rebuilt in V165; no result is affected.
* **C1, 5 of 5 checks pass:**
  * the update equals the explicit formula (1.4e-17);
  * with F = 0, no gradient reaches W and the top's gradient is exact;
  * the shuffle is a within-batch permutation;
  * runs are deterministic;
  * the rule learns: after one epoch on 200 lines, *Thursday* calibration is 3.715 against 3.845 with H frozen.

## C2 dev: *Edgeworth* carry-over (held-out bits/symbol, mean of seeds 0, 1, 2)
| Arm | Carry-over | Source |
|---|---:|---|
| BP-H: back-propagation into H | **3.070** | V163 |
| **DFA-H: broadcast error, fixed random feedback** | **3.126** | V165 (per seed 3.118 / 3.130 / 3.131) |
| TOP: no extra layer | 3.330 | V163 |
| V162's R1 on the KC code | 3.320 | V162 |
| SSL-H: local self-supervised layer | 3.358 | V163 |
| DFA-SHUF: errors from other positions | 3.440 | V165 (per seed 3.412 / 3.436 / 3.473) |
| RAND-H: random layer | 3.487 | V163 |

**Gating comparisons:**
* DFA-H beats TOP by **0.204** (SD 0.013): pass (≥ 0.03 and ≥ 3 SDs required).
* DFA-H beats DFA-SHUF by **0.314** (SD 0.012): pass.
* **φ = 0.86** of BP-H's gain over RAND-H.

**Also reported:**
* DFA-H beats SSL-H by 0.231 (SD 0.013).
* BP-H beats DFA-H by 0.057 (SD 0.009).
* DFA-SHUF beats RAND-H by 0.047 (SD 0.012): a small effect that needs no input-matched credit, probably the layer adapting to its input statistics.

**Training (*Thursday* calibration under the jointly trained top):**
* DFA-H: 3.298 → 3.122 → 3.051, best epoch 3, which is the budget edge. BP-H reached 2.963 at its best epoch, 2.
* DFA-SHUF: 3.543 → 3.409 → 3.338.

**Alignment** (cosine of the DFA update with the back-propagated next-byte gradient, on 2,048 calibration positions; non-gating):
* DFA-H: 0.48, 0.59 and 0.65 after epochs 1–3.
* DFA-SHUF: −0.01, 0.03 and 0.01.
* V163's local self-supervised rule: 0.019.

## What it shows
1. **The missing ingredient in V163 was the task error, not depth or capacity.**
   * Same layer, same top, same data, same probe. Replacing the contrastive self-supervised signal by a broadcast next-byte error moves carry-over from 3.358 to 3.126.
   * The layer now adds 0.20 bits beyond the KC code instead of losing 0.03.
2. **The error need not be back-propagated.** A fixed random feedback pathway suffices: the layer comes to align with it, so its update aligns with the true gradient (cosine 0.65 after three epochs), and it recovers 86% of back-propagation's gain.
3. **The credit must be input-matched.** The same errors, delivered to the wrong positions, give almost nothing (3.440, against 3.487 for the random layer). The gain is credit assignment, not error-shaped noise.
4. **By the reading fixed in advance (pass, φ ≥ 0.5),** the next step is to make the top block local as well (DFA for its hidden layer, a delta rule for the readout), then to test on a second text pair.

## Limits
* **Budget:** DFA-H's best epoch is the last (3) and its calibration was still falling, so 3.126 is a lower bound. The 0.057 gap to BP-H may shrink with more epochs, which needs larger caps.
* **Scale:** one text pair and one learner seed; DEV_NOT_EVIDENCE.
* **Not fully local yet:** the top block still learns by back-propagation, and the error e is computed from the top's output. The rule needs a signed, position-specific error signal to reach each unit of H. In the fly, the candidate carriers would be dopaminergic teaching signals; that mapping is not tested here.
* **No native integration:** this is not a native MiniFly result.

## Accounting
* 0.093 CPU-h over 3 attempts (C0 0.028, C1 0.003, C2 0.061).
* 4,480 learner records (C1 250, C2 4,230).
* 2 of 3 test cycles; designs tried k = 3.
