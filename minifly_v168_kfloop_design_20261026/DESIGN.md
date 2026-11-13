V168 DESIGN (owner go-ahead relayed by the main session; written and committed before any V168 code)

# V168 design: training V166's memory loop without BPTT (26 October 2026)

**Owner's words:** "Okay, you can do these." This answers the main session's list of the two follow-ups: V165's (run by the main session as V167) and V166's, "train the loop without back-propagation through time, then use MiniFly's real KC code and byte-level tasks". Earlier: "Assign a separate agent to work on P1-B to speed things up." V168 is the **first step** of V166's follow-up, run by a separate agent: synthetic tasks, with V166's memory, tasks and architecture. MiniFly's real KC code and bytes are not in V168.

**Question:** V166 showed with BPTT that the memory → workspace loop solves the two-hop query (Q2 0.009 bits against 3.09 without the loop). Can an **online rule without back-propagation through time** train the loop's recurrent and query weights, so that the error at the answer step reaches the query made two steps earlier?

**Caps:** the defaults, since the owner wrote no numbers: 2 CPU-h of child CPU, 5,000 learner records, 3 test cycles. No official run, no adoption. The main session has asked the owner for larger caps (proposed: 4 CPU-h, 20,000 records, 3 cycles). Until the owner writes numbers, V168 stays within the defaults (section 6).

## 0. Basis
* **V166:** with BPTT, LOOP solves Q2 (0.009 bits). Its query targets the cue at the Q step (99.7%) and the retrieved intermediate one step later (99.9%). Cutting r → h at test sends it to chance.
* **V164:** COLA's teaching signal is a rescaled immediate error. On an 8-step copy task it recovered 0% of BPTT's gain over one-step truncation (TBPTT-1). Credit must be carried to steps that have no error of their own.
* **V165:** a broadcast error through fixed random feedback trained a feedforward layer (86% of back-propagation's gain). That rule has no time dimension.

## 1. Fixed from V166 (unchanged)
* **Memory, lines, tasks:** V166's KC-like code (512 KCs, fan-in 6, top 16, seed 7), Hebbian write M ← M + e_b k(a)ᵀ/16 at study steps, the 34-step line and the MIXED task (2 Q1 and 2 Q2 episodes per line). Chance 5 bits; the candidate-set level on Q2 is 3 bits (log2 8).
* **Network:** V166's LOOP cell, H = 64, 47,072 parameters, initialisation seed 0:
  * h_t = tanh(a_t), a_t = W_x x_t + W_h h_{t−1} + W_r r_{t−1} + b_h;
  * q_t = ReLU(W_q h_t + b_q), r_t = M_t q_t;
  * y_t = W_o h_t + W_or r_t + b_o; loss and scores at ANS steps only.
* **Probe and scoring:** test bits per scored position, −log2 p(target), on Q1 and Q2 ANS steps; paired line bootstrap (1,000 resamples, seed 0); V166's query-targeting diagnostic.
* **Code:** V168 loads V166's `v166_lib.py` read-only by file path (generator, memory read, `Net`, `evaluate`, `summary`, `boot_sd`, `targeting`, `oracle`), and adds its own world seeds to that module's in-memory world table. No V166 file changes.
* **New worlds** (V166's dev worlds are not reused):
  * technical (C0, C1): train seed 1681, calibration 1682 (200 lines), test 1683 (1,000 lines);
  * dev (C2): 1691, 1692 and 1693, the same sizes, never used before C2.
  * The first N lines of a train seed are used, where N is set by the budget configuration (section 4).

**Only the training rule changes.** Every arm uses the same forward pass, initialisation, data order (seed 13), AdamW (weight decay 1e-4), gradient clipping at 5 and early stopping (best calibration epoch, patience 10). Arms differ only in the gradient estimate they hand to the optimiser.

## 2. The credit the loop needs, and which rule carries it
**Weight groups:**
* recurrent θ_a = [W_x | W_h | W_r | b_h] (64 × 167), with presynaptic vector u_t = [x_t; h_{t−1}; r_{t−1}; 1];
* query θ_q = [W_q | b_q] (512 × 65), with presynaptic vector [h; 1];
* readout [W_o | W_or | b_o].

**Exact quantities (state h):**
* D_t = ∂h_t/∂h_{t−1} = diag(φ′_t)(W_h + W_r M_{t−1} diag(ρ′_{t−1}) W_q). The second term is the memory path h_{t−1} → q_{t−1} → r_{t−1} → h_t.
* The immediate influence of θ_q on h_t is B^q_t ⊗ [h_{t−1}; 1], with B^q_t = diag(φ′_t) W_r M_{t−1} diag(ρ′_{t−1}).
* The learning signal is L_t = ∂ℓ_t/∂h_t = W_oᵀ e_t + W_qᵀ δq_t, with e_t = ∂ℓ_t/∂y_t and δq_t = ρ′_t ⊙ M_tᵀ W_orᵀ e_t.

**Credit for a Q2 episode** (Q at t, WAIT at t+1, ANS at t+2):

| Credit (what must be learned) | Steps back from ANS | TBPTT-1 | TBPTT-2 | TBPTT-3, BPTT | Diagonal traces (e-prop, RFLO, SnAp-1 on θ_a) | KF-RTRL |
|---|---:|---|---|---|---|---|
| Readout; the ANS step's own query (output path); θ_a at ANS | 0 | exact | exact | exact | exact | exact |
| Second-hop query: W_q at t+1 (q ≈ k(b)), via r_{t+1} → h_{t+2} | 1 | no | exact | exact | **no**: the query units have no internal state, so their trace is immediate | exact (one-step term) |
| Routing and holding: θ_a at t+1 (e_b into h_{t+1}), via D_{t+2} | 1 | no | exact | exact | only the diagonal of D_{t+2}: 1 of 64 paths per unit, with \|D_jj\| ≈ 0.1 at initialisation | unbiased estimate |
| **First-hop query: W_q at t (q ≈ k(a)), via r_t → h_{t+1} → D_{t+2}** | 2 | no | no | exact | **no** | unbiased estimate |
| Uses during study and delay | ≥ 3 | no | no | BPTT only | diagonal only | estimate (adds variance) |

* The path in the brief, r_t → h_{t+1} → q_{t+1} → r_{t+1} → h_{t+2}, is the memory term of D_{t+2} applied to W_q's influence at t. It is cross-unit at every link: the query writes into KC space, the memory maps KC space to items, and W_r spreads items over all hidden units.
* **So diagonal traces should matter here, and decisively.**
  * For θ_q, e-prop's update (memoryless query units, immediate learning signal) is **identical to TBPTT-1's**.
  * For θ_a, e-prop adds only D_t's diagonal entries.
  * C1 check 5 verifies the identity numerically and reports how close e-prop's whole update is to TBPTT-1's.
  * No records are spent training a rule whose trace does not contain the path.
* **SnAp-1 is not cheap here.** With h as the state, the one-step pattern of W_q's influence is dense (through W_r M), so SnAp-1 on θ_q equals exact RTRL for θ_q: about 9 GFLOP per line, or about 75 CPU-min per arm at 9,600 line presentations. Beyond the caps.
* **UORO** (rank-1, unbiased) carries the path in expectation, but its variance is known to be much larger than Kronecker-factored estimates (Benzing et al. 2019; Marschall et al. 2020).

### Learner: KF-RTRL (Mujika, Meier and Steger 2018), with exact immediate terms
For each weight group g, the influence J^g_t = ∂h_t/∂θ_g is approximated per line by one Kronecker product Ã^g_t ⊗ ũ^g_t: Ã^a is 64 × 64, Ã^q is 64 × 512, and ũ is the presynaptic vector. Per step:
1. Propagate: A1 = D_t Ã_{t−1}, u1 = ũ_{t−1}. New term: A2 = B_t, u2 = u_t, where B^a_t = diag(φ′_t) with u^a_t, and B^q_t with [h_{t−1}; 1].
2. **Gradient** at scored steps, where the (h, r)-TBPTT-1 gradient is computed exactly:
   ĝ = g_TBPTT-1 + (L_tᵀ D_t Ã^a_{t−1}) ⊗ ũ^a_{t−1} + (L_tᵀ D_t Ã^q_{t−1}) ⊗ ũ^q_{t−1} + (L_tᵀ B^q_t) ⊗ [h_{t−1}; 1].
   The last term is exact; it is W_q's one-step credit.
3. **Trace update** (random-sign mixing, variance-optimal scale):
   * ρ_i = √(‖A_i‖/‖u_i‖), and s = ±1 with equal probability, one draw per line, step and group;
   * Ã_t = A1/ρ1 + s A2/ρ2 and ũ_t = ρ1 u1 + s ρ2 u2;
   * if either term is zero, the other is kept exactly.
* **Unbiasedness:** E[Ã_t ⊗ ũ_t] = J_t. The cross terms carry the factor s and cancel in expectation. The forward pass does not depend on the signs, and parameters change only between batches. So E[ĝ] equals the exact BPTT gradient.
* **Dropping the carried terms gives TBPTT-1 exactly** (C1 check 1).
* **Not local:** KF-RTRL needs the exact one-step Jacobian D_t, including the transposed memory read, and a 64 × 512 trace per line. It tests whether an **online forward rule with a small, approximate trace** can carry this credit. It is not a fly-plausible rule. If it works, the next step is a more local trace; if it fails, the diagonal local rules carry strictly less of this path (table above).
* Signs come from seed 31.

### Matched control: KF-SHUF
* The same rule, the same sign stream and the same TBPTT-1 part, but every carried term uses the learning signal L_t of **another line in the batch**: a derangement of the batch, drawn per batch with seed 37, at the same step.
* The control keeps the update's size and noise and destroys its line-specific credit. In each line the memory and the pairs are random, so only the line's own error can say which query was right.
* The control with zeroed traces is TBPTT-1 itself (C1 check 1), so the TBPTT-1 reference serves as that control at 0 records.

## 3. References (evaluator instruments, 0 learner records, run in C0 before any learner code)
* **BPTT:** full back-propagation within the line (the ceiling).
* **TBPTT-k:** for each ANS step s, gradient through steps s−k+1 … s only. The carried state (h, r) is detached on entering step s−k+1.
  * Episodes are 3 steps long and losses fall only on ANS steps, so the windows for k ≤ 3 never overlap and this is exact TBPTT-k.
  * **TBPTT-1** (the floor): no credit across steps, since every parameter gets credit only for its use in the step where the loss falls.
  * **TBPTT-2 and TBPTT-3** (descriptive): how many steps of credit the task needs. By the table above, TBPTT-2 reaches the second hop and TBPTT-3 the first.

## 4. Budget and cycles (rules fixed in advance)
**The budget problem.**
* V166 trained on 4,000 lines × up to 60 epochs (240,000 line presentations per arm, batches of 32). Its LOOP got below 1 bit after about 1,100–1,250 updates.
* Learner records count lines × epochs for every learner arm, KF-SHUF included. So V168 trains on far fewer line presentations, and uses small batches to keep the number of updates near what V166 needed.

**Budget configurations** (all with the same AdamW, clip and patience):

| Config | Train lines N | Max epochs | Line presentations per arm R | Batch | lr | Updates | C2 learner records (2 arms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 400 | 6 | 2,400 | 2 | 1e-3 | 1,200 | 4,800 |
| A2 | 400 | 6 | 2,400 | 2 | 2e-3 | 1,200 | 4,800 |
| B1 | 800 | 12 | 9,600 | 4 | 1e-3 | 2,400 | 19,200 |
| B2 | 800 | 12 | 9,600 | 4 | 2e-3 | 2,400 | 19,200 |
| D | 1,000 | 24 | 24,000 | 4 | 1e-3 | 6,000 | 48,000 |

**C0 testbed** (technical worlds; references only; no learner code exists yet):

| Check | Rule |
|---|---|
| T0-a, wiring | V168's reference forward pass, without detaching, equals V166's `Net` bit for bit. Its BPTT gradient equals autograd through V166's `Net` (float64, ≤ 1e-12 relative) |
| T0-b, memory | V166's oracle reads on the new technical test world: ≥ 99% on Q1 and Q2 |
| T0-c, per config: ceiling | BPTT: Q2 ≤ 1.0 bit and Q1 ≤ 0.5 bits (V166's ceiling) |
| T0-c, per config: floor | TBPTT-1 is worse than BPTT on Q2 by ≥ 1.0 bit **and** ≥ 3 paired SDs |

* **Selection:** the first config in the order A1, A2, B1, B2, D that passes both the ceiling and the floor is the C2 budget. The order is fewest records first, then V166's learning rate. Selection uses technical-world results only.
* TBPTT-2 and TBPTT-3 run at every config, as descriptive values.
* **Continue** if T0-a and T0-b pass and some config qualifies. **Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED:** no budget in the grid lets the references show the gap, and V168 stops.

**C1 technical** (technical worlds; small fixtures; learner fixtures counted as records, 1 per line per pass; ≤ 200 records):
1. **TBPTT-1 identity:** KF's gradient with the carried terms dropped equals autograd TBPTT-1 (float64, 2 lines, ≤ 1e-10 relative).
2. **Exact RTRL identity:** the TBPTT-1 part plus the carried terms computed with full influence matrices equals autograd BPTT (float64, 2 lines, ≤ 1e-10). This validates D_t, B^q_t and the recursion.
3. **Unbiasedness, exact:** on a 6-step prefix of one line, the mean of Ã_5 ⊗ ũ_5 over all 2⁵ sign sequences equals the exact J_5, separately for each group (float64, ≤ 1e-10 relative).
   * The line is the first technical train line with a write at step ≤ 1, so the query group's check is not trivial.
4. **SHUF:**
   * every batch's permutation is a derangement;
   * KF-SHUF's TBPTT-1 part equals KF's exactly;
   * its carried part differs;
   * with the identity permutation it reproduces KF exactly.
5. **Diagonal traces:** the e-prop (diagonal, symmetric learning signal) update for θ_q equals TBPTT-1's (≤ 1e-12). Also reported: cosines of e-prop's, TBPTT-1's and KF's updates with BPTT's, and the share of BPTT's carried gradient that e-prop's diagonal captures.
6. **Determinism:** two 1-epoch KF runs on 16 lines (the selected config's batch and learning rate) give identical parameters.
7. **Learns:** one KF epoch on 32 lines lowers the training loss on those lines.
* **Alignment at initialisation (non-gating):** on 8 lines, the cosine with BPTT's gradient for one KF draw and for the mean of 4 draws.
* **Planned C1 records:** 2 + 2 + 64 + 12 + 8 + 32 + 32 + 32 = 184.

**C2 dev** (dev worlds; the selected config; learners KF and KF-SHUF; references BPTT, TBPTT-1, TBPTT-2 and TBPTT-3 retrained on the dev worlds; Q2 test bits; SDs from the paired line bootstrap on the dev test lines):
* **R-dev, reference check:** BPTT's Q2 ≤ 1.0 and Q1 ≤ 0.5, **and** TBPTT-1 − BPTT on Q2 ≥ 1.0 bit and ≥ 3 SDs. Call that gap G.
* **P1, recovery:** TBPTT-1 − KF on Q2 ≥ 0.5 G **and** ≥ 3 SDs.
* **P2, credit:** KF-SHUF does **not** meet P1's criterion, **and** KF-SHUF − KF on Q2 ≥ 3 SDs.
* **G1, guard:** KF's Q1 ≤ TBPTT-1's Q1 + 0.5 bits. The gain must not come at one-hop's expense.
* **Verdict:**
  * **INVALID** if R-dev fails (the gap to recover is not there);
  * **PASS** if P1, P2 and G1 hold;
  * **FAIL** otherwise.
* **Non-gating diagnostics:**
  * recovery share φ = (TBPTT-1 − arm)/G for KF, KF-SHUF, TBPTT-2 and TBPTT-3;
  * Q1 bits and accuracies;
  * V166's query targeting for every trained net;
  * best epochs and whether a best epoch is the last;
  * records and CPU.
* A learner whose best epoch is the last is reported as a lower bound on its recovery.

## 5. Readings fixed in advance
* **PASS:**
  * An online forward rule with a small, approximate trace carries the 2–3 steps of credit the memory loop needs, within this budget.
  * The gain needs line-matched credit.
  * Next: make the trace local, for example the learning signal through fixed random feedback (V165) and a trace through the memory that a fly circuit could compute. Then MiniFly's real KC code and bytes.
* **FAIL with P1 failing:**
  * Even an unbiased approximate-RTRL trace does not recover half of BPTT's gain at this budget. Its noise or the budget is the obstacle.
  * The diagonal local rules carry strictly less of this path (section 2, C1 check 5).
  * Options for later: lower-variance estimators (Kronecker sums of rank r, several samples), a larger budget (records), or an architecture with a shorter credit path.
* **P1 passes and P2 fails:** the gain does not need line-matched credit (noise or regularisation), so it is not evidence for the trace.
* **INVALID:** the dev references do not reproduce the testbed's gap. The verdict cannot be read.
* **CANNOT_TEST at C0:**
  * If no config qualifies because BPTT misses its ceiling: the loop is not learnable from ≤ 24,000 line presentations even with exact gradients. The obstacle is the instrument's budget, not a rule.
  * If the TBPTT-1 floor is not met: the task does not need credit across steps at these budgets, so it cannot test the rule.
* **TBPTT-2 and TBPTT-3, descriptive:** if TBPTT-2 stays near TBPTT-1 and TBPTT-3 reaches BPTT, the first hop's 2-step credit is the part that matters.
* **Not decided by V168:** whether any fly-plausible local rule learns the loop; MiniFly's real KC code; bytes.

## 6. Cost, caps and predictions
**Measured cost** (scratch timing on random tensors, one thread; not delivery code):
* a BPTT update takes about 22 ms at batch 2 or 4;
* the dominant per-step products of a Kronecker trace take about 5 ms per line.

**Estimated CPU:**
* C0: 20 reference fits, about 0.35 h;
* C1: about 0.03 h;
* C2: about 0.1 h at A configs and about 0.25 h at B configs;
* total under 0.7 CPU-h.
* At most 2 processes run in parallel; the main session shares the machine.

**Learner records:**
* C0: 0.
* C1: ≤ 200 (184 planned).
* C2: 2 × R:
  * A1 or A2 selected: 4,800 records, total ≤ 4,984, **within the default 5,000**. C2 runs.
  * B1 or B2 selected: 19,200 records, total ≤ 19,384. C2 runs only if the owner writes caps that cover it, recorded as grant 2. Otherwise V168 stops after C1 and reports these numbers.
  * Only D qualifies: 48,000 records, total ≤ 48,184, beyond the proposed raise. V168 stops after C1 and reports.
* Before launching C2, the remaining records and CPU are checked against the caps by hand. The launch wrapper checks only spent totals.

**Predictions:**

| Event | p |
|---|---:|
| T0-a and T0-b pass | 0.95 |
| Some config in A1–B2 qualifies | 0.6 |
| A1 or A2 qualifies (C2 fits the defaults) | 0.2 |
| TBPTT-1's Q2 at the selected config ≥ 2.5 bits | 0.75 |
| TBPTT-2 recovers less than half of G | 0.6 |
| TBPTT-3 within 0.3 bits of BPTT on Q2 | 0.6 |
| C1 check 5: e-prop's θ_q update equals TBPTT-1's | 0.97 |
| P1, given C2 runs | 0.35 |
| KF-SHUF fails P1's criterion, given C2 runs | 0.9 |
| PASS overall, given C2 runs | 0.3 |

## 7. What this cannot show
* **Scale:** small synthetic tasks, one seed per arm; DEV_NOT_EVIDENCE.
* **Plausibility:** KF-RTRL is online and forward-only but not local. A pass shows that a small approximate trace can carry this credit, not that a fly circuit can.
* **Budget:** every arm sees at most 24,000 line presentations, a tenth of V166's. A rule that needs more data than BPTT may fail here and pass at a larger budget.
* **Memory:** V166's KC-like code (512 units), not MiniFly's 5,177-unit layer; a dense learned query; study-gated writes.
* **Scope:** no bytes, no addition, no native integration.
