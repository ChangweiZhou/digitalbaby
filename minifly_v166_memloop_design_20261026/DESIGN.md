V166 DESIGN (owner go-ahead relayed by the main session; written and committed before any V166 code)

# V166 design: architecture sufficiency audit, memory → workspace (review P1-B) (26 October 2026)

**Owner's words:** "Assign a separate agent to work on P1-B to speed things up." Earlier in the same exchange: "Activate the recursive improvement skill, turn this to V165." V165 (the review's P1-A) runs in the main session. V166 is the review's **P1-B**, run by a separate agent, in parallel and not on condition of V165.

**Question:** when an acquired memory is read, can the retrieved content change the internal working state and take part in the next computation, h_{t+1} = F(h_t, x_t, r_t) with r_t = Retrieve(M_t, h_t)? Does that path give anything that the same retrieval read out only at the output, h_{t+1} = F(h_t, x_t), does not?

**What V166 is:** an architecture-sufficiency test, with back-propagation through time (BPTT) as the instrument, used as a positive control. V163's layer is feedforward, and V164's recurrent network has no memory it can retrieve from. If even BPTT cannot use the retrieval loop here, there is no basis for looking for a biological learning rule for it (the review's own criterion).

**What V166 is not:** no local learning rule, no MiniFly native integration, no bytes and no addition. The memory is MiniFly-like but small and synthetic.

**Caps:** the defaults, since the owner wrote no numbers: 2 CPU-h of child CPU, 5,000 learner records, 3 test cycles. No official run, no adoption. Pushes go to the session branch only.

## 0. Basis
* **V163:** a 512-unit feedforward layer above the frozen KC code, learned by back-propagation, cut *Edgeworth* carry-over from 3.33 to 3.07 bits. Exact BPTT was no better than one-step truncation on text at 40 KB.
* **V164:** on an 8-step copy task, BPTT beat one-step truncation by 1.32 bits; COLA recovered none of it.
* **Neither tested the path in question.** No V-series network so far reads a memory written during the line and feeds what it read back into its own state.

## 1. Memory (MiniFly-like, fixed, not learned)
* **Items:** 32 symbols. Each item x has a fixed random "PN" pattern p(x) ~ N(0, I) in 32 dimensions.
* **KC code:** a frozen random expansion to 512 KCs.
  * Each KC sums exactly 6 PN dimensions, chosen at random without replacement, with weight 1.
  * k(x) is the binary indicator of the 16 most active KCs (3.1%; MiniFly's KC layer has about 3.5% active).
  * Codes and wiring use seed 7 and are shared by every arm and world.
* **Write (Hebbian, input-driven):** at a study step that shows the pair (cue a, value b):
  * M ← M + e_b k(a)ᵀ / 16, where e_b is the one-hot of b over the 32 items.
  * M is a 32 × 512 KC→MBON matrix with one output unit per item, reset to 0 at each line start.
  * The write depends only on the input. No gradient reaches it.
* **Read:** r_t = M_t q_t, with a learned query q_t = ReLU(W_q h_t + b_q) in KC space (512 dimensions).
  * q_t is dense and differentiable, not top-k.
  * Sending the query through the frozen PN→KC expansion is left for later.
* **Implementation:** r_t = Σ_i [w_i ≤ t] e_{b_i} (k(a_i) · q_t) / 16, summed over the line's pairs i with write step w_i. This equals M_t q_t exactly. C0 checks the equality in float64.

## 2. Tasks (fixed now)
**Line layout** (34 steps). Each step carries three channels: a marker (one of STUDY, DISTRACT, Q1, Q2, WAIT, ANS), a cue (an item, or none) and a value (an item, or none).

1. **Pairs:** 12 distinct items form 4 chains x0 → x1 → x2, giving 8 pairs (x0, x1) and (x1, x2) in random order.
2. **Study phase (16 steps):** the 8 pair steps (STUDY, cue a, value b; memory written) are interleaved at random with 8 distractor steps.
   * A distractor step is DISTRACT, with a cue drawn uniformly from all 32 items and no value. Nothing is written.
3. **Delay:** 6 distractor steps.
4. **Query phase:** 4 episodes of 3 steps each:
   * Q1 or Q2, with the query cue;
   * WAIT;
   * ANS, the scored step.

   The state sees the cue and value channels at every step, so a plain recurrent network could hold pairs in its own state.

**Two tasks on these lines:**
* **ONE-HOP:** all 4 episodes are Q1, with 4 distinct keys drawn from the 8 keys. The target at ANS is the cue's value. This is the delay task; it needs one retrieval.
* **MIXED** (the composition task):
  * 2 episodes are Q2, with 2 distinct chain heads x0; the target is x2 = f(f(x0)).
  * 2 episodes are Q1, with 2 distinct keys drawn from the 6 other keys; the target is f(cue).
  * The episodes come in random order.
  * The Q1 positions are the one-hop reading and the Q2 positions the two-hop reading. Training on both is a built-in curriculum: the composition reuses the one-hop primitive.

**What a network can get without the loop:**
* Chance is log2 32 = 5 bits.
* A network that knows the set of tails but not which chain the cue belongs to gets log2 4 = 2 bits on Q2. That is the best possible without retrieving the specific chain twice.
* NO-LOOP can do one retrieval per step, but its query depends only on a state that never saw a retrieval. It can do better than 2 bits on Q2 only by holding the first pair (a, b) in its own recurrent state, and then querying the memory with k(b).

**Worlds** (fixed seeds):
* **Technical** (C0, C1): train 4,000, calibration 200 and test 1,000 lines, with seeds 101, 102 and 103.
* **Dev** (C2): seeds 201, 202 and 203, the same sizes, never used before C2.
* Test sets have 4,000 scored positions: 2,000 Q1 and 2,000 Q2 on MIXED.

## 3. Network and arms
**The cell:**
* LOOP: h_t = tanh(W_x x_t + W_h h_{t−1} + W_r r_{t−1} + b_h).
* q_t = ReLU(W_q h_t + b_q) and r_t = M_t q_t.
* logits_t = W_o h_t + W_or r_t + b_o, over the 32 items.
* x_t is the concatenation of three one-hots: 6 markers, 32 cue items and 32 value items, with none as all-zero.
* Loss and scores are taken at ANS steps only.

**Timing:** a hop takes one step. At Q (t), q ≈ k(a) retrieves b. At WAIT (t+1), r_t enters h, so q ≈ k(b) can retrieve c. ANS (t+2) gives one step of slack.

| Arm | Path r → h | Memory written | H | Parameters | Role |
|---|---|---|---:|---:|---|
| LOOP | yes (W_r) | true pairs | 64 | 47,072 | the test |
| NO-LOOP | no: r_t reaches only the readout | true pairs | 67 | 47,262 | the review's ablation that closes the retrieval feedback |
| LOOP-SHUF | yes | values permuted within the line by a derangement (every key gets a wrong value; same keys and values; a fixed permutation per line) | 64 | 47,072 | the gain must use the retrieved content |
| LOOP-NULL | yes | nothing (M = 0, so r = 0) | 64 | 47,072 | plain recurrent network: floor |

**Parameter matching:**
* LOOP has H² + 647H + 1,568 parameters; NO-LOOP has H² + 615H + 1,568.
* NO-LOOP's width is the smallest whose count is at least LOOP's: H = 67, 0.4% more. The control is never smaller.
* Compute per step differs by under 5%. Measured CPU is reported per arm.

**Training (identical for every arm; the instrument):**
* Full BPTT within the line.
* AdamW, learning rate 1e-3, weight decay 1e-4; batches of 32 lines; gradient norm clipped at 5.
* At most 60 epochs; stop after 10 epochs without improvement in calibration bits over all scored positions; keep the best calibration epoch.
* Initialisation:
  * weights N(0, 1 / fan-in), biases 0;
  * seed 0 for every arm, so LOOP, LOOP-SHUF and LOOP-NULL start identical;
  * batch order seed 13.
* One thread per process.

**Accounting:** every arm is a BPTT-trained evaluator instrument (a positive control and its matched ablations). As with the headroom check's and V163/V164's back-propagation references, the arms count as 0 learner records. Epochs and CPU are reported.

## 4. Cycles and rules fixed in advance
**Scoring:**
* Test bits per scored position, −log2 p(target), averaged over Q1 or Q2 ANS steps.
* A difference's SD comes from a paired bootstrap of 1,000 resamples of test lines, seed 0.
* A fit whose best epoch is the last (60) is reported as a lower bound.

**C0 testbed (technical worlds; references and qualification only; no comparison code exists yet):**

| Check | Rule |
|---|---|
| T0-a, read | The factored read equals the explicit M_t q_t (float64, max abs difference ≤ 1e-12) |
| T0-b, oracle memory | Reads with the true codes, b̂ = argmax M k(a) and then ĉ = argmax M k(b̂), are correct on ≥ 99% of test Q1 and Q2 episodes |
| T0-c, one-hop sanity, both architectures | LOOP and NO-LOOP trained on ONE-HOP: each ≤ 0.5 test bits |
| T0-d, ceiling: the positive control learns the composition | LOOP trained on MIXED: Q2 ≤ 1.0 bit and Q1 ≤ 0.5 bits |
| T0-e, floor: the plain state does not solve it | LOOP-NULL trained on MIXED: Q2 ≥ 1.5 bits |

* **Continue** only if all five pass.
* **Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED.**
  * One testbed redesign is allowed, by amendment with reason testbed_invalid, and only for this case: T0-d fails with LOOP's best epoch at the 60-epoch edge. Then the budget is doubled to 120 epochs, with nothing else changed.
  * Any other failure stops V166.

**C1 technical (technical worlds; small instances; 0 records):**
1. **NO-LOOP isolation:** with the memory zeroed or shuffled, NO-LOOP's states are bit-identical, and the autograd gradient of its summed states with respect to the memory values is exactly 0. The same gradient for LOOP is nonzero.
2. **Parameter counts:** NO-LOOP ≥ LOOP and within 1%. LOOP-SHUF and LOOP-NULL equal LOOP.
3. **Shuffle:** on every line, the written values are a derangement of the true ones, with the multiset kept; permutations differ across lines and are the same in every epoch.
4. **Cut:** a LOOP with W_r = 0 gives states identical to a pass that ignores r (float64).
5. **Gradient:** autograd's gradient for LOOP equals central finite differences on a tiny float64 instance (relative error ≤ 1e-6).
6. **Determinism:** two 1-epoch runs on 64 lines give bit-identical parameters.

**C2 dev (dev worlds; all four arms trained on MIXED):**
* **P1:** LOOP beats NO-LOOP on Q2 by ≥ 0.5 bits **and** ≥ 3 SDs.
* **P2:** LOOP beats LOOP-SHUF on Q2 by ≥ 0.5 bits **and** ≥ 3 SDs.
* **Guards:** LOOP's and NO-LOOP's Q1 bits ≤ 0.5 on dev, so both retrieve.
* **Verdict:**
  * **SUPPORTED** if P1, P2 and the guards pass;
  * **INVALID** if a guard fails;
  * otherwise **NOT_SUPPORTED**.
* **Diagnostics (non-gating):**
  * LOOP-NULL's Q1 and Q2 bits on dev;
  * the trained LOOP with W_r set to 0 at test (LOOP-CUT), and with its memory values deranged at test;
  * accuracies;
  * **query targeting:** at the Q, WAIT and ANS steps of each Q2 episode, which of the line's 8 keys has the largest k_i · q_t, reported as the share of episodes where that key is the cue a and where it is the intermediate b, for LOOP and NO-LOOP.

## 5. Readings fixed in advance
* **SUPPORTED:**
  * With exact gradients, retrieved memory that re-enters the working state is usable, and it is needed for composition here: the same retrieval read out only at the output does not do it, and a memory with the right statistics but the wrong content does not either.
  * The positive control uses the connection. That licenses, but does not prove, a search for a learning rule that can train the r → h path without BPTT. Candidates: the review's P2 eligibility traces, given V164's finding that the credit must cross steps.
  * Then move toward MiniFly's real KC code as the key code, and toward byte-level tasks.
* **NOT_SUPPORTED, P1 fails:** even BPTT gets no composition advantage from the loop here. By the review's criterion, there is no basis for promoting a memory → workspace loop as a mechanism. Revisit the memory formulation or the task before any learning rule.
* **NOT_SUPPORTED, P1 passes but P2 fails:** LOOP's advantage does not depend on the correct memory content, so it comes from extra recurrence, not retrieval.
* **CANNOT_TEST at C0:**
  * If T0-d fails: the BP positive control cannot learn to use the path within this budget. There is no basis for a local rule now; the next question is the instrument (budget, width, curriculum), not a mechanism.
  * If T0-e fails: the plain state solves the task, so the task does not need the memory. Redesign the task.
  * If T0-b or T0-c fails: the memory or the retrieval itself is broken.
* **Not decided by V166:** whether any biologically plausible rule can learn the loop, or whether MiniFly's real KC code and byte inputs behave the same way.

## 6. Cost, caps and predictions
**Measured cost** (random tensors, no task, one thread; scratch timing, not delivery code): one BPTT batch of 32 lines × 34 steps at H = 64 takes 38 ms, so one epoch of 4,000 lines takes about 5 s.

**Estimated CPU:**
* each fit ≤ 5 min (60 epochs);
* C0: 4 fits, ≤ 0.33 h;
* C1: ≤ 0.02 h;
* C2: 4 fits, ≤ 0.33 h;
* worst case about 0.7 CPU-h, expected about 0.3.
* At most 2 processes run in parallel; the main session shares the machine.

**Learner records:** 0, since every arm is an evaluator instrument.

**Predictions:**

| Event | p |
|---|---:|
| T0-b, the oracle memory, passes | 0.97 |
| T0-c, the one-hop sanity check, passes for both architectures | 0.8 |
| T0-d: LOOP learns the composition (Q2 ≤ 1.0) | 0.65 |
| T0-e: the plain state fails Q2 (≥ 1.5) | 0.9 |
| P1 passes, given C0 continues | 0.9 |
| P2 passes, given C0 continues | 0.9 |
| NO-LOOP's Q2 within 0.5 bits of LOOP-NULL's | 0.6 |
| LOOP-CUT raises LOOP's Q2 by ≥ 1 bit, given SUPPORTED | 0.8 |
| SUPPORTED overall | 0.55 |

## 7. What this cannot show
* **Scale:** small synthetic tasks, one seed per arm, a 32-item memory; DEV_NOT_EVIDENCE.
* **Instrument:** BPTT is not a biological rule. A pass shows the architecture can use the path, not that a fly-like learner can learn it.
* **Memory:** the KC code is KC-like (512 units, fan-in 6, top-16), not MiniFly's 5,177-unit layer. The query is a dense learned projection into KC space, not a top-k pattern through the PN→KC expansion.
* **Scope:** no bytes, no addition and no native integration.
