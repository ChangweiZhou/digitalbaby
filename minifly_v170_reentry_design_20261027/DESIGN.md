V170 DESIGN (owner grant relayed by the main session; written and committed before any V170 code)

# V170 design: learning two-hop composition without BPTT; re-entry through the fixed expansion, and a three-factor rule (27 October 2026)

**Owner's words** (grant 1, AUTHORIZATION.json): "No need. 1) Push all the discussions above as a memo after streamlining to the repo in academic language. Be well organized. 2) Launch the two experiments using the skills." The two experiments are V169 and V170 of `minifly_memo_credit_horizon_20261027/MEMO.md` section 7. V170 is run by a separate agent in parallel with V169.

**Caps:** the defaults, since no number was written: 2 CPU-h of child CPU, 5,000 learner records, 3 test cycles. No official run, no adoption. Pushes go to the session branch only. V170 fits these caps or stops after its code checks and reports what it needs.

**Question (memo section 7, V170):** can the two-hop composition that V166's loop solves with BPTT be learned **without** back-propagation through time? Two strategies, in this order:
* **A, architectural (primary):** the retrieved item is re-encoded through the same fixed PN→KC expansion that encodes input and serves as the next query. Only one-step decisions remain to be learned (whether to answer with the first retrieval or re-query, and how to read out), trained with one-step credit.
* **B, learning rule:** V166's learned loop, trained by a three-factor rule: eligibility tags (pre × post) decaying over about 3 steps, combined with the delayed answer error broadcast through fixed random feedback.

In both, **order generalisation** is tested explicitly with held-out order-dependent instruction pairs.

## 0. Basis
* **V166:** with BPTT, LOOP (h_t = tanh(W_x x_t + W_h h_{t−1} + W_r r_{t−1} + b_h), q_t = ReLU(W_q h_t + b_q), r_t = M_t q_t) solves the two-hop query: Q2 0.009 bits against 3.09 without re-entry. Memory: KC-like code (512 KCs, fan-in 6 from a 32-dimensional PN pattern, top 16, seed 7), Hebbian KC→MBON writes e_b k(a)ᵀ/16 at study steps, reset per line.
* **V168:** the loop needs credit about 3 steps back, across units (answer → memory read → earlier query). TBPTT-3 recovers 97% of BPTT's gain over TBPTT-1, TBPTT-2 4.5%. Diagonal traces cannot carry it (e-prop's query update equals TBPTT-1's to 1.5e-16); KF-RTRL recovered 3.3%. BPTT needed about 9,600 line presentations (E1: 800 lines × 12 epochs, batch 4, lr 3e-3). At 2,400 presentations per arm no budget let BPTT learn (testbed-003).
* **V165, V167:** an output error broadcast through fixed random feedback trains a feedforward layer well. That settles spatial credit for one layer; V170 is about temporal credit.

## 1. Task: COMPOSE-ORD (new; fixed now)
**Line** (34 steps, as V166). Input x_t is the concatenation of four one-hots (none = all-zero): marker (5: STUDY, DISTRACT, INSTR, QUERY, ANS), cue item (32), value item (32), instruction token (6: tokens 0–5). D_IN = 75.
1. **Pairs:** 12 distinct items form 4 chains x0 → x1 → x2, giving 8 pairs (x0, x1), (x1, x2) in random order.
2. **Study (16 steps):** the 8 pair steps (STUDY, cue a, value b; memory written) interleaved at random with 8 distractor steps (DISTRACT, cue uniform over 32 items, nothing written).
3. **Delay:** 6 distractor steps.
4. **Query phase:** 4 episodes of 3 steps:
   * INSTR, token u;
   * QUERY, token v, cue a;
   * ANS, the scored step.
5. **Episodes per line:** 2 TWO-HOP episodes (cues: 2 distinct chain heads x0; target f(f(a)) = x2) and 2 ONE-HOP episodes (cues: 2 distinct keys from the 6 other keys; target f(a)), in random order.
6. **The instruction is an ordered pair (u, v), u ≠ v.** Ascending (u < v) means TWO-HOP; descending (u > v) means ONE-HOP. So (u, v) and (v, u) always have different answers, and the hop count can be read only from the order of the two tokens.
7. **Held-out ordered pairs (order generalisation):** the unordered pairs H = {0,3}, {1,2}, {4,5} (both orders, 6 ordered pairs) never occur in training or calibration lines. The other 12 unordered pairs (24 ordered pairs) are the trained set. Every token occurs in training in both positions and in both orders.
   * The instruction pair of an episode is drawn uniformly from the allowed list (ascending or descending part), by an index from its own generator, default_rng([seed, i, 1]). Everything else in a line comes from default_rng([seed, i]).
   * **test-T** (trained pairs) and **test-H** (held-out pairs) use the same test seed and the same index draws, so the two test sets have identical lines and differ only in the instruction tokens. That gives a paired comparison.
8. **Worlds:** technical (C0, C1): train 1701 (first N lines, N set by the budget), calibration 1702 (200 lines), test 1703 (1,000 lines, as test-T and test-H). Dev (C2): 1711, 1712, 1713, the same sizes, never used before C2.
9. **Levels:** chance is 5 bits. Knowing only that the answer is one of the line's 8 stored values gives 3 bits. A system that knows the two candidates b and c but not the order rule, and splits evenly, gets about 1 bit on both hop types.

**Scoring:** test bits per scored ANS step, −log2 p(target), averaged separately over ONE-HOP (Q1) and TWO-HOP (Q2) episodes, on test-T and test-H. The SD of a difference comes from a paired bootstrap of 1,000 resamples of test lines, seed 0.

## 2. Strategy A: the re-entry circuit
**Fixed parts (no learned parameter; task-independent; the same rule at every step of every line):**
* PN patterns P (32 items × 32 PN dimensions) and the PN→KC wiring J (512 × 32, fan-in 6, weight 1) are V166's (seed 7). The input encoding of an item x is k(x) = top16(J p(x)), binary. C0 checks that these equal V166's codes.
* **Re-entry encoding:** K(r) = top16(J Pᵀ r), binary: the retrieved MBON pattern r (over items) is decoded into the r-weighted mixture of item PN patterns and passed through the **same** expansion as input. K is 0 if Σ r ≤ 1e-9. For a one-hot r = e_x, K(e_x) = k(x) exactly (C0 check).
* **KC drive:** at a step with a cue, q_t = k(cue_t) (sensory input has precedence); at a step without a cue, q_t = K(r_{t−1}) (re-entry). This applies at every cue-less step: INSTR, ANS and any other.
* **Read:** r_t = M_t q_t / 16, with V166's memory (writes up to and including step t).
* **Front end for the decision:** z_t = [x_t; x_{t−1}; x_{t−2}] (225 dimensions; zeros before the line start), a fixed delay line over **all** input channels, of the same kind as MiniFly's lag-tagging front end. No task-specific feature is selected.

**Learned parts (1,282 parameters):**
* **Gate:** g_t = σ(w_g · z_t + b_g): answer with the previous retrieval or with the re-queried one.
* **Readout:** y_t = W_o m_t + b_o over the 32 items, with m_t = (1 − g_t) r_{t−1} + g_t r_t.
* Initialisation: weights N(0, 1/fan-in), biases 0, seed 0.

In an episode: at QUERY, q = k(a) retrieves b. At ANS there is no cue, so q = K(r_QUERY) ≈ k(b) retrieves c. The gate must pick r_QUERY (b) for descending pairs and r_ANS (c) for ascending pairs, reading the order of u (lag 2) and v (lag 1). The readout must map MBON patterns to output logits.

**Why only one-step credit is needed, and what that means.** No learned parameter influences q, r or M at any step, and all learned parameters act at the scored step. So the exact gradient through the whole line (BPTT) equals the one-step (TBPTT-1) gradient for this circuit. C1 check A1 verifies this numerically. **This is the architectural claim, and it is also its limit:** A removes the temporal-credit problem, rather than solving it with a learning rule. That is why strategy B runs whenever it fits the caps (section 3).

**Learner A (the "A" arm):** one-step credit, written explicitly (not autograd):
* readout: the delta rule ∂ℓ/∂W_o = e_t m_tᵀ, ∂ℓ/∂b_o = e_t, with e_t = softmax(y_t) − onehot(target);
* gate: ∂ℓ/∂w_g = g_t(1 − g_t) [(W_oᵀ e_t) · (r_t − r_{t−1})] z_t, and the same factor for b_g.
* Averaged over the scored steps of a batch, then AdamW (weight decay 1e-4), clipping at 5, batch order seed 13, best calibration epoch kept (calibration bits over all scored steps).
* This is one-step credit with exact within-step spatial credit (as V168's TBPTT-1). Spatial locality of the gate's update (W_oᵀ e) is not addressed here; V165–V167 address it.

**Matched controls (learner arms, same learner, same budget, same initialisation):**
* **A-SHUF:** the re-entry is present, but its content is deranged. K(Π r) with Π a per-line derangement of the 32 items: no fixed point, the same in every epoch, from default_rng([17, seed, i]). The cue-driven reads are untouched.
* **A-CUT:** no re-entry. At cue-less steps q_t = 0.

**Guards against hand-wiring:**
* the re-entry pathway is generic: it has no parameter and no task input, and it runs at every cue-less step;
* the same circuit answers ONE-HOP and TWO-HOP, choosing from the input order;
* A-SHUF tests whether the re-entered content matters;
* the untrained circuit must be near chance (C1 check A6);
* held-out instruction pairs test whether the learned decision generalises an order rule rather than memorising pairs.

## 3. Strategy B: three-factor rule on the learned loop
**Network:** V166's LOOP cell, re-implemented for D_IN = 75 (H = 64, initialisation seed 0), on COMPOSE-ORD. C0 checks the re-implementation against V166's `Net` on V166's own lines.

**Rule (B arm):**
* Eligibility tags, reset at line start, with λ = e^(−1/3) ≈ 0.717 (decay over about 3 steps):
  * recurrent group θ_a = [W_x | W_h | W_r | b_h]: E^a_t = λ E^a_{t−1} + φ′_t ⊗ u_t, with u_t = [x_t; h_{t−1}; r_{t−1}; 1] and φ′ the tanh derivative;
  * query group θ_q = [W_q | b_q]: E^q_t = λ E^q_{t−1} + ρ′_t ⊗ [h_t; 1], with ρ′ the ReLU derivative.
* Readout [W_o | W_or | b_o]: the delta rule e_t ⊗ [h_t; r_t; 1], which is immediate and exact.
* At each ANS step, the output error e_t is broadcast through fixed random matrices B_h (64 × 32) and B_q (512 × 32), N(0, 1/32), seed 41. The updates are ∂θ_a ≈ (B_h e_t) ⊙_rows E^a_t and ∂θ_q ≈ (B_q e_t) ⊙_rows E^q_t.
* The same optimiser and training loop as the references.

**Matched control B-SHUF:** the same rule, but the broadcast error (only in the feedback terms) comes from another line of the batch: a derangement drawn per batch, seed 43. The readout uses its own error.

**References (evaluator instruments, 0 records):** LOOP trained by BPTT and by TBPTT-1 (h and r detached on entering each ANS step), at the same budget.

**Prior expectation, stated before any run:** the first-hop credit runs through the line-specific memory M_t. A fixed random feedback matrix cannot align with a matrix that changes every line, and decaying tags at the query units carry only pre × post, not the memory path. So B is expected to fail (section 7).

## 4. Budgets, cycles and rules (fixed in advance)
**Budget configs** (line presentations R = lines × epochs):

| Config | Lines × epochs | R | Batch | lr (A arms) | lr (LOOP references and B) |
|---|---|---:|---:|---:|---:|
| G1 | 100 × 2 | 200 | 4 | 1e-2 | 3e-3 |
| G2 | 250 × 2 | 500 | 4 | 1e-2 | 3e-3 |
| G3 | 500 × 2 | 1,000 | 4 | 1e-2 | 3e-3 |
| G4 | 800 × 2 | 1,600 | 4 | 1e-2 | 3e-3 |
| E1 | 800 × 12 | 9,600 | 4 | – | 3e-3 (V168's selected budget) |

* The LOOP and B use lr 3e-3, the best learning rate V168 found for this loop. The A arms use 1e-2, since they train only a linear readout and a gate.
* E1 is run for the LOOP references only, to size B and as a descriptive order-generalisation reference.
* Learner records count lines × epochs for every learner arm, controls included.
* The C1 learner records are capped at 200.

### C0 testbed (technical worlds; references only; no learner code exists)
| Check | Rule |
|---|---|
| T0-a, wiring | V170's LOOP with V166's D_IN equals V166's `Net` bit for bit (float32 forward) on V166 technical lines, and its BPTT gradient equals autograd through `Net` (float64, ≤ 1e-12 relative). V170's P, J and k(x) equal V166's CODES exactly |
| T0-b, re-entry identity | K(e_x) = k(x) for all 32 items (exact) |
| T0-c, oracle | The A circuit with the ideal gate (1 for ascending, 0 for descending) and argmax of m: ≥ 99% correct on Q1 and Q2 of technical test-T and test-H |
| T0-d, A ceiling (per G config) | A circuit trained with autograd BPTT (exact gradient, all parameters): test-T Q2 ≤ 1.0 and Q1 ≤ 0.5 bits |
| T0-e, A floors (per G config) | A-CUT trained the same way: Q2 worse than T0-d's by ≥ 1.0 bit and ≥ 3 SDs; **and** LOOP-TBPTT-1: Q2 worse than T0-d's by ≥ 1.0 bit and ≥ 3 SDs |
| T0-f, LOOP qualification (per config, G1–G4 and E1) | LOOP-BPTT: Q2 ≤ 1.0 and Q1 ≤ 0.5; and LOOP-TBPTT-1 − LOOP-BPTT ≥ 1.0 bit and ≥ 3 SDs on Q2 |

**Selection:**
* **R_A:** the first of G1–G4 that passes T0-d and T0-e.
* **R_B:** the first of G1–G4, E1 at which T0-f holds.
* **C2 composition:**
  * A runs if R_A exists: 3 R_A records (A, A-SHUF, A-CUT).
  * B runs if R_B exists **and** 3 R_A + 2 R_B + 200 ≤ 5,000. Otherwise B stops after its C1 checks, and the report states what it needs (2 R_B records plus C1, or "no qualifying reference up to E1").
* **Continue** if T0-a, T0-b and T0-c pass and R_A exists. If R_A does not exist: CANNOT_TEST_CLAIM_AS_SPECIFIED for A. Then continue for B only if R_B fits; otherwise stop.
* Descriptive (not used for selection): A-BPTT on test-H; LOOP fits on test-H.

### C1 technical (technical worlds; small fixtures; ≤ 200 records; one cycle for both learners)
A checks:
1. **A1, one-step = full credit:** the explicit A learner's gradient equals autograd BPTT through the whole line and autograd TBPTT-1 on the A circuit (float64, 4 lines, ≤ 1e-10 relative).
2. **A2, learned-parameter isolation:** q_t and r_t at every step are bitwise identical under two different parameter settings.
3. **A3, SHUF:**
   * every line's Π is a derangement, identical across epochs;
   * distinct permutations occur at a rate at least 0.95 of the rate expected for uniform draws, on 8 lines;
   * the cue-driven reads are identical to A's;
   * the re-entered codes differ.
4. **A4, CUT:** r_t = 0 exactly at cue-less steps; the cue-driven reads are identical to A's.
5. **A5, generic re-entry:** the code path's re-entry function, applied to one-hots, equals the input encoding, and it is invoked at exactly the cue-less steps of every line (count).
6. **A6, structure audit:** the untrained A circuit (initial parameters) scores ≥ 4.0 bits on technical test-T.
7. **A7, determinism:** two 1-epoch runs on 16 lines give identical parameters.
8. **A8, learns:** one epoch on 32 lines lowers training loss on them.

B checks:
1. **B1, identity:** with λ = 0, B_h e replaced by the true learning signal L_t = W_oᵀ e_t + W_qᵀ δq_t, and B_q e replaced by M_tᵀ W_orᵀ e_t (masked by ρ′), the B update equals autograd LOOP-TBPTT-1 (float64, 2 lines, ≤ 1e-10 relative).
2. **B2, SHUF:**
   * every batch permutation is a derangement;
   * the readout part equals B's;
   * the feedback part differs;
   * the identity permutation reproduces B.
3. **B3, determinism:** as for A, on 16 lines.
4. **B4, learns:** one epoch on 32 lines lowers training loss.

### C2 dev (dev worlds; selected budgets; all arms retrained)
**Strategy A** (learners A, A-SHUF, A-CUT; reference LOOP-TBPTT-1 and LOOP-BPTT at R_A):
* **P1, composition with one-step credit:**
  * A's test-T Q2 ≤ 1.0 and Q1 ≤ 0.5;
  * **and** LOOP-TBPTT-1 − A on Q2 ≥ 1.0 bit and ≥ 3 SDs.
* **P2, content:** A-SHUF − A ≥ 1.0 bit and ≥ 3 SDs on Q2, **and** A-CUT − A ≥ 1.0 bit and ≥ 3 SDs on Q2.
* **P3, order generalisation:** on test-H, A's Q1 ≤ 0.5 and Q2 ≤ 1.0. An undecided gate gives about 1 bit on Q1, so this needs the order rule applied to unseen pairs.
* **G1, guard:** A-SHUF's and A-CUT's test-T Q1 ≤ 0.5: the controls differ only in the re-entry.
* **Verdict A:**
  * **INVALID** if G1 fails;
  * **PASS** if P1, P2 and P3 hold;
  * **FAIL** otherwise.

**Strategy B** (if it runs; learners B, B-SHUF; references LOOP-BPTT and LOOP-TBPTT-1 at R_B):
* **R-dev-B:** LOOP-BPTT test-T Q2 ≤ 1.0 and Q1 ≤ 0.5, and G_B = LOOP-TBPTT-1 − LOOP-BPTT ≥ 1.0 bit and ≥ 3 SDs on Q2.
* **PB1:** LOOP-TBPTT-1 − B ≥ 0.5 G_B and ≥ 3 SDs on Q2.
* **PB2:** B-SHUF does not meet PB1's criterion, and B-SHUF − B ≥ 3 SDs on Q2.
* **GB1:** B's Q1 ≤ LOOP-TBPTT-1's Q1 + 0.5.
* **Verdict B:**
  * **INVALID** if R-dev-B fails;
  * **PASS** if PB1, PB2 and GB1 hold;
  * **FAIL** otherwise.
  * **NOT_RUN** (caps) if B does not fit.
  * B's test-H is descriptive.

**Non-gating diagnostics:**
* gate decision accuracy (g > 0.5 ⇔ ascending) on test-T and test-H, per held-out pair;
* A with the gate forced to 0 or 1 at test;
* LOOP-BPTT at E1 on test-T and test-H (dev), as the learned-loop order reference;
* best epochs, records and CPU per arm.

## 5. Readings fixed in advance
* **A PASS:**
  * With the retrieved item re-encoded through the input expansion, two-hop composition is learned from a small budget with one-step credit only.
  * The re-entered content is necessary.
  * The learned decision generalises the order rule to held-out ordered pairs.
  * The temporal-credit problem of V168 is **removed by architecture, not solved by a rule** (BPTT ≡ TBPTT-1 for A). Whether that counts as supplied structure turns on the guards in section 2. The report states the judgment.
* **A FAIL, P3 only:** composition is learned, but the order decision does not transfer to unseen pairs. The lag-tagged front end supports discrimination, not generalisation, here (memo section 5.2).
* **A FAIL, P1 or P2:** re-entry through the fixed expansion does not give composition at this budget, or its gain does not need the re-entered content.
* **A INVALID:** a control is broken; nothing is read.
* **B PASS:** a three-factor rule with decaying tags and fixed random feedback carries the 2-step credit through the memory. That would contradict the prior expectation.
* **B FAIL:** the expected outcome. Fixed broadcast feedback cannot align with a line-specific memory path.
* **B NOT_RUN:** the needed records are reported. Nothing about B is claimed.
* **Stopping (memo section 7):** if A passes, the next step is the memo's route toward native tasks. A negative result for both strategies indicates that a mechanism has not yet been identified, and is not a reason for tuning.

## 6. Cost and caps
* **Estimated CPU:**
  * C0: 8 A-type fits (seconds each) and 10 LOOP fits (≤ 20 s at G budgets, about 65 s at E1), about 0.1 h;
  * C1: under 0.02 h;
  * C2: A arms in seconds, LOOP references about 0.05 h (0.1 h with E1), B arms (if run) about 2–3 times BPTT;
  * total about 0.3 CPU-h.
* At most 2 processes run in parallel, with one thread each.
* **Learner records:** C0 0; C1 ≤ 200; C2 3 R_A (+ 2 R_B if B fits). At G4, A alone needs 4,800, so 5,000 is the binding limit.
* The launch wrapper checks spent totals. The C2 fit is checked by the selection rule above, before launch.

## 7. Predictions
| Event | p |
|---|---:|
| T0-a, T0-b and T0-c pass | 0.95 |
| R_A exists (some G config qualifies for A) | 0.85 |
| R_A = G1 | 0.5 |
| LOOP-TBPTT-1 Q2 ≥ 2.5 at R_A (technical) | 0.85 |
| LOOP-BPTT qualifies at some G config (≤ 1,600) | 0.15 |
| LOOP-BPTT qualifies at E1 | 0.5 |
| B runs in C2 (fits caps) | 0.15 |
| C1 check A1 (one-step = full gradient for A) | 0.97 |
| P1, given C2 runs A | 0.85 |
| P2, given C2 runs A | 0.9 |
| P3, given C2 runs A | 0.6 |
| A PASS, given C2 runs A | 0.5 |
| PB1, given B runs | 0.1 |
| LOOP-BPTT at E1 meets P3's thresholds on dev test-H, given it meets them on test-T | 0.35 |

## 8. What this cannot show
* **Scale:** small synthetic tasks, one seed per arm; DEV_NOT_EVIDENCE.
* **Structure:** A's composition is executed by a fixed pathway. A pass shows that the architecture makes the credit local in time, not that a learning rule can carry multi-step credit.
* **Spatial locality:** A's gate update uses W_oᵀ e within the step.
* **Order:** held-out **combinations** at fixed positions are tested. Positions never seen in training are not: the fixed 3-step delay line cannot reach them by construction.
* **Memory:** V166's KC-like code (512 units), study-gated writes; no bytes, no addition, no native integration.
