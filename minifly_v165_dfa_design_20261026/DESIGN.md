V165 DESIGN (owner go-ahead; written and committed before any V165 code)

# V165 design: a task-aligned local learning signal for V163's layer (direct feedback alignment) (26 October 2026)

**Owner's words:** "Activate the recursive improvement skill, turn this to V165." The owner attached the joint review of V163–V164 (ChatGPT).

**Scope (my interpretation, stated to the owner before acting):** V165 is the review's **P1-A**.
* V163's forward model stays fixed: the frozen KC code, a 512-unit layer H, and a top block trained by back-propagation.
* Only H's learning signal changes: the next-byte error is broadcast to H through fixed random feedback.
* This is also the next step that V163's reading fixed in advance (a pass with φ < 0.5 calls for feedback from the top, because the local self-supervised update was orthogonal to the task gradient, cosine 0.019).

**Not in V165:**
* The review's P1-B, the memory-to-workspace architecture test, needs its own task design. It is proposed as V166, to run in parallel, not on condition of V165.
* P2 (eligibility traces) comes later.
* P0 bookkeeping is already in the V163 and V164 reports: SSL-H at its 8-pass edge, and V164's references and COLA at the 10-epoch edge.

**Caps:** the defaults, since the owner wrote no numbers: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 1. The rule
* **Forward pass (as in V163):**
  * a_t = W x_t + b and h_t = ReLU(a_t), where x_t is the frozen KC code with its empty-state column;
  * a top block (2,048 ReLU units, dropout 0.5, softmax over A's classes) reads h_t.
  * H starts from RAND-H's initialisation; the top starts from V163's seed-0 initialisation.
* **The top block** learns by back-propagation within itself, as in V163, with the gradient stopped at its input.
* **H's learning signal (DFA-H):**
  * output error: e_t = (p_t − onehot(y_t)) / N, the gradient of the batch-mean cross-entropy with respect to the logits;
  * teaching signal: δ_t = ReLU′(a_t) ⊙ (e_t F);
  * update: ΔW_ji ∝ −Σ_t δ_j(t) x_i(t), and likewise for b.
  * F (K × 512) is fixed and random: Gaussian with variance 1/K, seed 21.
  * Each synapse's change is the product of presynaptic KC activity, the unit's own derivative, and an error broadcast to it through the fixed pathway. There is no weight transport and no backward pass through the top.
* **Control (DFA-SHUF):** identical, except that the error rows are permuted within each batch (fixed seed). Every unit still receives an error with the same statistics, but from a different position than the one whose input it saw.
* **Optimiser and schedule for both:** AdamW, learning rate 1e-3, weight decay 1e-4, batch 256. At most 3 epochs (the record cap). The epoch with the best A calibration bits under the jointly trained top is kept. BP-H's best epoch in V163 was 2.

## 2. Arms and readings
**Arms reused from V163, unchanged:** the same probe, the same three seeds and the same lines.

| Arm | *Edgeworth* carry-over (3-seed mean) |
|---|---:|
| RAND-H (random layer) | 3.487 |
| TOP (no extra layer) | 3.330 |
| SSL-H (local self-supervised layer) | 3.358 |
| BP-H (back-propagated layer) | 3.070 |

**New arms:** DFA-H and DFA-SHUF, with the same carry-over reading. H is learned on *Thursday* and frozen; a fresh top is fitted on 10 KB of *Edgeworth* with seeds 0, 1 and 2 (V163 amendment 1), then scored on *Edgeworth*'s held-out lines.

**Same text:** the jointly trained top's calibration bits are reported. V163 showed that a probe refitted on *Thursday* is invalid for layers trained with labels.

## 3. Cycles and rules
**C0 testbed** (no learner code):
* **Parity:**
  * V163's BP-H, retrained under V165's 3-epoch budget, reproduces V163's cached layer exactly;
  * RAND-H's 3-seed carry-over equals V163's.
* **Ceiling under the budget:** BP-H beats TOP on carry-over by ≥ 0.06 bits (V163: 0.26).

**C1 technical:**
* the update equals the explicit formula (float64, ≤ 1e-10);
* with F = 0, W receives no gradient, and the top's gradient equals back-propagation within the top;
* the shuffle is a permutation within the batch;
* one epoch on 25 lines is deterministic;
* **learns:** after one epoch on A's first 200 lines, the joint model's A calibration bits are ≥ 0.01 lower than the same top trained over frozen RAND-H on the same lines and schedule (the RAND-H run is an evaluator reference).

**C2 dev:**
* **Pass, on carry-over:**
  * DFA-H beats **TOP** by ≥ 0.03 bits and ≥ 3 line-bootstrap SDs, so it adds value beyond the KC code, which SSL-H did not;
  * **and** DFA-H beats DFA-SHUF by ≥ 0.03 bits and ≥ 3 SDs, so the gain comes from input-matched credit.
* **Reported:**
  * φ = (RAND-H − DFA-H) / (RAND-H − BP-H);
  * DFA-H against SSL-H and RAND-H.
* **Diagnostics (non-gating):** the cosine between the DFA update of W and the back-propagated next-byte gradient at the same parameters, on 2,048 A calibration positions at each epoch end, for both arms. This is the review's ⟨g_local, ∇J⟩.

## 4. Readings fixed in advance
* **Pass, φ ≥ 0.5:** a broadcast output error through fixed random feedback recovers most of what back-propagation gains in this slot. Next: make the top block local too (DFA for its hidden layer and a delta rule for the readout), then a second text pair.
* **Pass, φ < 0.5:** task-aligned feedback adds value beyond the KC code, but leaves most of the gain unrecovered. Next: feedback that learns to align (for example Kolen–Pollack), instead of fixed random feedback.
* **Beats RAND-H and DFA-SHUF but not TOP:** task-aligned feedback helps, but like SSL-H it does not beat the KC code. Then the gap to BP-H is not closed by a broadcast error at this depth and budget.
* **Does not beat DFA-SHUF:** any error-shaped signal does as well, so no task-specific credit reaches H.
* **The budget edge:** if DFA-H's best epoch is the last (3), its result is a lower bound. The fix is larger caps, not another rule.

## 5. Accounting and predictions
* **Learner records:**
  * DFA-H ≤ 2,115 and DFA-SHUF ≤ 2,115 (3 epochs × 705 lines each);
  * C1: 50 (determinism) + 200 (learns);
  * total ≤ 4,480.
  * References (BP-H, RAND-H, the frozen-H top) are evaluator fits, as in V163.
* **Estimated CPU:** about 0.3 h. One epoch of a layer plus top costs about 12 s, each 3-seed carry-over probe about 2 min.
* **Predictions:**
  * DFA-H beats RAND-H: 0.8;
  * the pass (beats TOP and DFA-SHUF): 0.45;
  * φ ≥ 0.5: 0.25.
