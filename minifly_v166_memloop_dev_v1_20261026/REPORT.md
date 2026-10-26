V166 MEMLOOP V1: SUPPORTED. With BPTT as a positive control, retrieved memory that re-enters the working state solves a two-hop composition: Q2 0.009 bits, against 3.09 when the same retrieval reaches only the output and 3.29 with the memory contents deranged. DEV_NOT_EVIDENCE.

# V166 report: architecture sufficiency audit, memory → workspace (review P1-B) (26 October 2026)

**Question:** can retrieved memory change the internal working state and take part in the next computation, h_{t+1} = F(h_t, x_t, r_t) with r_t = M_t q_t? Does it give anything that the same retrieval read out only at the output does not?

**Answer: yes, at least for an exact-gradient learner on a small synthetic task.**
* With the loop, BPTT learns a two-hop query (a → b → c) almost perfectly: 0.009 bits, 99.75% correct.
* The matched network whose retrieval reaches only the readout stays at 3.09 bits. That is the level of knowing that the answer is one of the 8 stored values (log2 8 = 3).
* With the memory contents deranged, the loop network stays at 3.29 bits.
* Both one-hop guards hold, so the control network can retrieve; it cannot compose.
* By the review's criterion, the BP positive control can use this connection. That licenses, but does not prove, a search for a non-BPTT rule that trains it.

**Files:**
* Design: `minifly_v166_memloop_design_20261026/DESIGN.md`, committed before any code.
* Grant: `AUTHORIZATION.json`. Owner's words: "Assign a separate agent to work on P1-B to speed things up." Default caps.
* Ledgers in `runlog/`:
  * 3 attempts, none failed;
  * cycles testbed *continue*, technical *continue*, dev *stop*;
  * amendments 1–2, operational, made before the technical attempt, plus erratum row 3.
* Results: `results/testbed/TESTBED.json`, `results/technical/CHECKS.json`, `results/dev/RESULTS.json`.
* Run by a separate agent in parallel with V165. The harness did not let the agent write this file, so the main session saved its report text unchanged apart from this note and one wording fix in C1 (the threshold "set at design time").

## Setup (as designed)
**Memory:** a fixed KC → MBON matrix M (32 items × 512 KCs), reset per line.
* Writes: at a study step showing (cue a, value b), M gains e_b k(a)ᵀ / 16. k(a) is a frozen KC-like code: each KC has fan-in 6 from a 32-dimensional PN pattern, and the top 16 of 512 are kept.
* Reads: r_t = M_t q_t, with q_t = ReLU(W_q h_t + b_q) learned.
* No gradient reaches the writes.

**Lines** (34 steps):
* 8 pairs forming 4 chains x0 → x1 → x2, interleaved with 8 distractor cues;
* a delay of 6 distractor steps;
* 4 query episodes of 3 steps each (query, wait, answer).

**Tasks:** ONE-HOP (the target is f(cue)) and MIXED (2 one-hop and 2 two-hop episodes per line; the two-hop target is f(f(head))).

**Reference levels:** chance is 5 bits. Knowing only which items are tails gives 2 bits on Q2.

| Arm | r → h | Memory | H | Parameters |
|---|---|---|---:|---:|
| LOOP | yes | true | 64 | 47,072 |
| NO-LOOP | no: r reaches the readout only | true | 67 | 47,262 (+0.4%) |
| LOOP-SHUF | yes | values deranged within each line | 64 | 47,072 |
| LOOP-NULL | yes | empty | 64 | 47,072 |

**Training (every arm):**
* full BPTT, AdamW at learning rate 1e-3, batches of 32 lines;
* at most 60 epochs, patience 10, best calibration epoch kept;
* 4,000 / 200 / 1,000 lines for training, calibration and test.

## C0 testbed (technical worlds; all five checks pass)
| Check | Result | Rule |
|---|---|---|
| T0-a: factored read = explicit M_t q_t | max difference 0.0 | ≤ 1e-12 |
| T0-b: oracle reads with the true codes | Q1 100%, Q2 100% | ≥ 99% |
| T0-c: ONE-HOP, LOOP and NO-LOOP | 0.0007 and 0.0005 bits | each ≤ 0.5 |
| T0-d: ceiling, LOOP on MIXED | Q2 0.024, Q1 0.002 bits | Q2 ≤ 1.0, Q1 ≤ 0.5 |
| T0-e: floor, LOOP-NULL on MIXED | Q2 4.97 bits | ≥ 1.5 |

* The memory holds the answers.
* Both architectures learn one retrieval.
* The positive control learns the composition.
* A plain recurrent state of the same size learns nothing usable within the budget: it stays at chance, and does not even reach the 2-bit tail-set level.
* Three fits had their best epoch at the 60-epoch edge, so their values are lower bounds. All were far inside their thresholds.

## C1 technical (technical worlds; 6 of 6 checks pass)
1. **NO-LOOP isolation:** its states are bit-identical with the memory zeroed or deranged, and the gradient of its states with respect to the memory is exactly 0. LOOP's gradient is nonzero.
2. **Parameter counts:** NO-LOOP has 0.40% more parameters than LOOP. LOOP-SHUF and LOOP-NULL equal LOOP.
3. **Shuffle:** a derangement on every line, with the multiset of values kept. Distinct permutations occur at the rate expected for uniform draws (0.876 against 0.877 expected).
4. **Cut:** setting W_r = 0 gives exactly the states of the cut pass.
5. **Gradient:** the BPTT gradient equals central finite differences (relative error 1.6e-8).
6. **Determinism:** two runs give identical parameters.

**Amendments before this attempt:**
* Amendment 1 corrected the shuffle check's threshold, set at design time at 0.9; the birthday expectation, 0.877, makes that unreachable.
* Amendment 2 kept that check off the dev worlds.
* No task, metric, margin, arm or rule changed.

## C2 dev (fresh dev worlds; MIXED; all four arms retrained)
**Test bits per scored position** (2,000 Q1 and 2,000 Q2 answers; accuracy in brackets):

| Arm | Q1 (one-hop) | Q2 (two-hop) | Best epoch / run | CPU s |
|---|---:|---:|---:|---:|
| LOOP | 0.003 (99.95%) | **0.009** (99.75%) | 60 / 60 | 311 |
| NO-LOOP | 0.028 (100%) | 3.092 (13.0%) | 30 / 40 | 202 |
| LOOP-SHUF | 3.271 (12.7%) | 3.286 (12.8%) | 10 / 20 | 103 |
| LOOP-NULL | 4.968 (4.7%) | 4.930 (6.3%) | 14 / 24 | 124 |

**Gating, all pass:**

| Rule | Gain on Q2 (bits) | Paired line-bootstrap SD | Required |
|---|---:|---:|---|
| P1: LOOP vs NO-LOOP | **3.083** | 0.012 | ≥ 0.5 and ≥ 3 SDs |
| P2: LOOP vs LOOP-SHUF | **3.277** | 0.018 | ≥ 0.5 and ≥ 3 SDs |
| Guards: LOOP Q1, NO-LOOP Q1 | 0.003 and 0.028 | – | each ≤ 0.5 |

**Verdict: SUPPORTED.**

**Compute:** about 5.1 CPU-s per epoch for every arm. The controls stopped early because their calibration bits had not improved for 10 epochs; the budget they did not use was available to them. LOOP's best epoch is at the edge, so its number is an upper bound on its bits.

**Diagnostics (non-gating):**
* **Cutting r → h at test** (LOOP-CUT) sends the trained LOOP to 5.27 bits on Q2 and 5.20 on Q1, which is chance. The trained network routes even one-hop answers through its state, although NO-LOOP shows that the output-only route suffices for one hop.
* **The trained LOOP read with deranged memory at test** is confidently wrong: 14.1 bits on Q2 and 17.7 on Q1. It uses the retrieved content.
* **Query targeting:** on 2,000 Q2 episodes, the key with the largest k_i · q_t at each step:

  | | Q step: cue a | WAIT step: intermediate b | ANS step: b |
  |---|---:|---:|---:|
  | LOOP | 99.65% | 99.85% | 60.3% |
  | NO-LOOP | 11.1% | 11.8% | 11.9% |

  For NO-LOOP the share is at the 1-in-8 level at every step.
  * LOOP performs exactly the designed two hops: it queries the cue, then the item it retrieved, one step later.
  * On two-hop episodes NO-LOOP does not target any key. Its 3.09 bits match a diffuse query that returns the set of stored values (accuracy 13% ≈ 1/8).
* **Against the empty memory on Q2:**
  * NO-LOOP is 1.84 bits better (SD 0.02) and LOOP-SHUF 1.64 bits better (SD 0.03). The memory without the loop gives the set of candidates, not the answer.
  * LOOP is 4.92 bits better.

## Predictions (design §6)
| Event | p | Outcome |
|---|---:|---|
| T0-b passes | 0.97 | yes |
| T0-c passes for both architectures | 0.8 | yes |
| T0-d: LOOP learns the composition | 0.65 | yes |
| T0-e: the plain state fails Q2 | 0.9 | yes |
| P1, given C0 continues | 0.9 | yes |
| P2, given C0 continues | 0.9 | yes |
| NO-LOOP's Q2 within 0.5 bits of LOOP-NULL's | 0.6 | **no**: 1.84 bits apart. NO-LOOP extracts the candidate set from the memory |
| LOOP-CUT raises LOOP's Q2 by ≥ 1 bit | 0.8 | yes (+5.26) |
| SUPPORTED overall | 0.55 | yes |

## What it shows
1. **The architecture can use the path, and the task needs it.**
   * With exact gradients, the memory → workspace loop turns a one-retrieval memory into a composable one. The second query is computed from what the first retrieval returned.
   * The matched output-only control solves one hop as well as LOOP, but on two hops it gets only the candidate set (about 3 bits).
   * A memory with correct statistics but wrong content gives nothing beyond that.
   * This is the review's missing connection: read an acquired memory, change the working state, retrieve again.
2. **The mechanism is the designed one, not a shortcut.**
   * Query targeting shows the two hops step by step.
   * Removing the r → h path at test destroys the behaviour.
   * Deranging the memory at test makes the network confidently wrong.
3. **The credit the path needs crosses time steps.**
   * The answer error at step t + 2 must train the query at t (first hop) and the routing W_r at t + 1.
   * V164 showed that COLA cannot carry credit to a step that has no error of its own. A local rule for this loop therefore needs something that transports later errors back, such as eligibility traces (the review's P2), not a rescaled immediate error.
4. **By the reading fixed in advance (SUPPORTED):**
   * The next step is a rule that trains the r → h path and the query without BPTT.
   * Then MiniFly's real KC code as the key code, and byte-level tasks.

## Limits
* **Scale:** a small synthetic task, one seed per arm, 32 items and 8 pairs per line; DEV_NOT_EVIDENCE. The task was built so that the plain state cannot solve it: the 64-unit recurrent network stayed at chance, so in-state memorisation was not a competitor here.
* **Instrument:** BPTT is not a biological rule. The result shows the architecture can use the path, not that a fly-like learner can learn it.
* **Memory:** the memory is KC-like (512 units, fan-in 6, top-16), not MiniFly's 5,177-unit layer. The query is a dense learned projection into KC space, not a top-k pattern through the PN→KC expansion. The writes are hand-gated by the study marker.
* **Scope:** no bytes, no addition, no native integration. V166 says nothing about V165's representation question.
* **Budget:** LOOP's best epochs were at the 60-epoch edge in C0 and C2. Its bits would at most fall further, so the verdict does not depend on it.

## Accounting
* **CPU:** 0.499 CPU-h of child CPU over 3 attempts:
  * C0: 0.286;
  * C1: 0.002;
  * C2: 0.212.
* **Records:** 0 learner records. Every arm is a BPTT evaluator instrument, as V163/V164's back-propagation references were.
* **Cycles:** 2 of 3 test cycles; designs tried k = 3 (each lock rebuild counts).
* **Evaluator epochs:**
  * C0: 60, 60, 60 and 22;
  * C1: 2;
  * C2: 60, 40, 20 and 24.
