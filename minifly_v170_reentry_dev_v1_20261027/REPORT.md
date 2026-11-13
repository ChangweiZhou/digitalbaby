V170 REENTRY V1: INVALID for strategy A (pre-registered guard G1 fails: the controls answer one-hop queries 100% correctly but with 1.07 and 1.05 bits, against a 0.5-bit guard); strategy B NOT_RUN (no qualifying BPTT reference up to 9,600 line presentations). Not read as a verdict: re-entry through the fixed expansion, trained with one-step credit only, reaches two-hop Q2 0.72 bits from 500 line presentations, but does not generalise the order rule to held-out instruction pairs (Q2 2.05). DEV_NOT_EVIDENCE.

# V170 report: learning two-hop composition without BPTT; re-entry through the fixed expansion, and a three-factor rule (27 October 2026)

*Run by a separate agent in parallel with V169. The harness did not let the agent write this file. The main session checked the key numbers against `results/dev/RESULTS.json` and saved the agent's text unchanged apart from this note. Full text with every table: the agent's final message, summarised here section by section.*

**Question (memo section 7):** can the two-hop composition that V166's loop solves with BPTT be learned without back-propagation through time? Two strategies were tested, and order generalisation was measured explicitly.

**Answer, by the rules fixed in advance: not decided.**
* **Strategy A (architectural)** is **INVALID**. Its guard on the matched controls failed, so by the pre-registered reading nothing is read as a verdict.
* **Strategy B (three-factor rule)** did **not run**. Even exact BPTT on the learned loop did not reach the ceiling within 9,600 line presentations on this task, so there was no qualifying reference to test it against.

**What the numbers show, outside the verdict (non-gating):**
1. **Composition with one-step credit.**
   * The retrieved item is re-encoded through the same PN→KC expansion as input and used as the next query. A gate and a readout, trained with one-step credit only, then answer two-hop queries: 100% correct at 0.72 bits, from 500 line presentations.
   * The learned loop (V166's cell) at the same budget stays at chance with one-step credit (4.77 bits) and also with BPTT (4.74). At 9,600 presentations, BPTT reaches only 2.46 bits.
2. **The re-entered content is necessary.** Deranging it, or removing it, costs 4.6–4.7 bits on two-hop queries.
3. **The order rule does not generalise.** On held-out ordered instruction pairs, two-hop accuracy is 55% and Q2 is 2.05 bits. The gate's decision is right 100% of the time on trained pairs and 75% on held-out pairs; it is at chance on the pair {4,5}. No exact-gradient reference of the A circuit generalised either, at any budget.
4. **For A, BPTT ≡ TBPTT-1 holds by construction** (verified to 5e-16). A removes V168's temporal-credit problem rather than solving it with a learning rule.

**Files:**
* Design: `minifly_v170_reentry_design_20261027/DESIGN.md`, committed before any code (655ee8f9).
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`:
  * 4 attempts, none failed;
  * cycles: testbed *revise*, testbed *continue*, technical *continue*, dev *stop*;
  * amendment 1 (testbed_invalid), made before any learner code.
* Results: `results/testbed/TESTBED.json` and `TESTBED_2.json`; `results/technical/CHECKS.json`; `results/dev/RESULTS.json`.

## Setup
**Task COMPOSE-ORD:** V166's memory and lines, with 4 query episodes per line (INSTR token u; QUERY token v with cue a; ANS).
* The hop count is given only by the order of the two tokens: ascending means two-hop, descending means one-hop, with tokens 0–5.
* The unordered pairs {0,3}, {1,2} and {4,5} are held out of training.

**Strategy A:** the cue-less KC drive is K(r_{t−1}) = top16(J Pᵀ r_{t−1}), the retrieved item re-encoded through the same expansion as the input.
* Learned: a gate over a 3-lag delay line, and a readout (1,282 parameters), trained with explicit one-step credit.
* Controls: A-SHUF (re-entered content deranged) and A-CUT (no re-entry).

**Strategy B:** V166's LOOP cell trained by eligibility tags (λ = e^(−1/3)) times the answer error broadcast through fixed random feedback. Control B-SHUF; references LOOP-BPTT and LOOP-TBPTT-1.

## C0 testbed
* The fixed checks pass: wiring identical to V166; re-entry K(e_x) = k(x); the oracle 100% correct on both test sets.
* The design grid G1–G4 qualified nothing (testbed-001).
* Amendment 1 added budgets H1–H8 at learning rates 0.05 and 0.1, with arms and rules unchanged. H3 was selected: 250 lines × 2 epochs, lr 0.05. On technical worlds A-BPTT scored Q2 0.712, Q1 0.415, and A-CUT-BPTT already had Q1 0.99.
* The LOOP never qualified, reaching at best Q2 2.13 at E1 (9,600 presentations).

## C1 technical: 12 of 12 checks pass (154 records)
* For A, the explicit one-step gradient equals autograd BPTT (5.2e-16).
* The SHUF and CUT controls behave as specified.
* The re-entry identity holds.
* Runs are deterministic, and the learners learn.
* B's update equals autograd TBPTT-1 when the true one-step signals replace the random feedback (2.1e-16).

## C2 dev (fresh dev worlds; H3; 1,500 records)
| Arm | test-T Q1 | test-T Q2 | test-H Q1 | test-H Q2 | Gate correct (T / H) |
|---|---:|---:|---:|---:|---:|
| **A** | 0.384 (100%) | **0.723** (100%) | 1.233 (96.5%) | **2.051** (54.7%) | 100% / 75.3% |
| A-SHUF | 1.067 (100%) | 5.354 (1.2%) | 1.195 | 5.867 | 80.2% / 65.7% |
| A-CUT | 1.048 (100%) | 5.407 (0.3%) | 1.176 | 5.920 | 74.4% / 63.3% |
| LOOP-BPTT (500) | 4.745 | 4.744 | 4.745 | 4.746 | – |
| LOOP-TBPTT-1 (500) | 4.820 | 4.773 | 4.812 | 4.777 | – |
| LOOP-BPTT (9,600; descriptive) | 1.332 | 2.461 | 1.433 | 2.501 | – |

**Rules:**
* P1 holds: A's composition; LOOP-TBPTT-1 − A = 4.05 bits on Q2 (z 198).
* P2 holds: SHUF −A = 4.63, CUT − A = 4.68.
* **P3 fails:** held-out Q1 1.233 and Q2 2.051, against required values of ≤ 0.5 and ≤ 1.0.
* **G1 fails:** the controls' Q1 is 1.067 and 1.048, against ≤ 0.5.

**Verdict A: INVALID. Verdict B: NOT_RUN.**

**Non-gating diagnostics:**
* **Gate on held-out pairs:** {0,3} 100%, {1,2} 81%, {4,5} 44%.
* **Forced gate:** the circuit has learned both answers (gate forced to 0: one-hop 0.34 bits; forced to 1: two-hop 0.64 bits). Only the order decision fails.
* **G1's failure** reflects confidence, not retrieval: the controls retrieve one-hop answers 100% correctly. C0 had already shown it (A-CUT-BPTT Q1 0.99 at H3). The design did not screen the guard there, and the agent did not catch it before launching. Nothing was changed after the outcome.

## What it shows
1. **By the readings fixed in advance:** A is INVALID and B is NOT_RUN.
2. **Non-gating:**
   * Re-entry through the input expansion makes composition learnable with one-step credit, from a twentieth of the data that BPTT needed for the learned loop.
   * This is supplied structure, not a learning rule. For A, BPTT ≡ TBPTT-1, and the number of hops is fixed by episode timing.
3. **Order generalisation fails, in both the learner and its exact-gradient reference.** The lag-tagged input supports discrimination of (u, v) from (v, u), but not generalisation of the order rule to unseen pairs (memo section 5.2).
4. **COMPOSE-ORD is harder for the learned loop than V166's task.** BPTT did not qualify within 9,600 presentations.

## What unrun stages need
* **Strategy A with a valid guard:** a post-outcome guard change (for example, guarding on one-hop accuracy) is an owner decision, and it needs fresh dev worlds. About 1,654 records, within the default caps.
* **Strategy B:**
  * a C0 extension (0 records, 0.1–0.3 CPU-h) to find a budget where LOOP-BPTT qualifies; the next candidate is 60,000 presentations;
  * then a dev run of 2 × that budget in records (at least 19,200; 120,000 at 60,000 presentations), with caps the owner must write.

## Accounting
* **CPU:** 0.167 CPU-h (testbed 0.090 + 0.040; technical 0.002; dev 0.036).
* **Records:** 1,654 learner records (C1 154; C2 1,500).
* **Cycles:** 2 of 3 test cycles; designs tried k = 4. Default caps; no official run.
