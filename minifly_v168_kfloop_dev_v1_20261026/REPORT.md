V168 KFLOOP V1: STOPPED AFTER C1 (records cap); no learner verdict. Even exact BPTT needs about 9,600 line presentations per arm to learn V166's two-hop loop here, so the KF-RTRL comparison (2 learner arms) needs 19,200 learner records against the 5,000 granted. C1 verified the learner exactly. Diagonal e-prop's query update is identical to TBPTT-1's. DEV_NOT_EVIDENCE.

# V168 report: training V166's memory loop without BPTT, first step (26–27 October 2026)

**Question:** can an online rule without back-propagation through time train the loop's recurrent and query weights, so that the error at the answer step reaches the query made two steps earlier?

**Answer: not decided.**
* **Why C2 did not run:** the reference runs show that the comparison cannot fit 5,000 learner records.
  * At ≤ 4,800 line presentations even exact BPTT does not learn the loop.
  * At 9,600 it does: Q2 0.56 bits against TBPTT-1's 3.50.
  * With the learner and its matched control, C2 needs 19,200 records. V168 therefore stopped after the code checks (C1), as the owner's grant 2 instructs.
* **Settled without training:**
  1. **The task needs exactly the first hop's 2-step credit.** At every budget where BPTT learns, TBPTT-3 recovers 98–100% of BPTT's gain over TBPTT-1, while TBPTT-2 recovers 3–6%.
  2. **Diagonal eligibility traces cannot carry that credit.**
     * e-prop's update for the query weights equals TBPTT-1's to 1.5e-16.
     * For the recurrent weights, e-prop's diagonal adds 0.3% of BPTT's carried gradient.
  3. **The chosen learner is correct and unbiased.**
     * KF-RTRL with its carried terms dropped equals autograd TBPTT-1; with exact carried terms it equals autograd BPTT.
     * Its Kronecker trace averages exactly to the true influence matrix over all sign sequences.
     * A single draw is noisy: its cosine with BPTT's gradient at initialisation is 0.23–0.44.

**Files:**
* Design: `minifly_v168_kfloop_design_20261026/DESIGN.md`, committed before any code.
* Grants: `AUTHORIZATION.json`:
  * grant 1, "Okay, you can do these." (default caps);
  * grant 2, "3 hours should be fine. No need for a lot of work as we are not testing scaling law etc.": 3 CPU-h; records stay 5,000.
* Ledgers in `runlog/`:
  * 4 attempts, none failed;
  * cycles: testbed *revise*, testbed *revise*, testbed *continue*, technical *stop*;
  * amendments 1 (testbed_invalid) and 2 (operational), both before any learner code existed in the delivery.
* Results:
  * `results/testbed/TESTBED.json`, `TESTBED_2.json`, `TESTBED_3.json`;
  * `results/technical/CHECKS.json`.
* Run by a separate agent in parallel with V167. The harness did not let the agent write this file; the main session checked the key numbers against the result files and saved the agent's text unchanged apart from this note.

## Setup (as designed)
* **Fixed from V166:**
  * the memory: KC-like code, 512 KCs, top 16, Hebbian KC → MBON writes, reset per line;
  * the 34-step lines, the MIXED task (2 one-hop and 2 two-hop episodes per line);
  * the LOOP cell (H = 64, 47,072 parameters, initialisation seed 0), scoring and line bootstrap.
  * V166's library is loaded read-only by file path.
  * New worlds: technical seeds 1681–1683; dev seeds 1691–1693, never used.
* **Learner: KF-RTRL** (Mujika, Meier and Steger 2018), with exact immediate terms.
  * Each weight group's influence ∂h_t/∂θ is carried per line as one Kronecker product. The trace is 64 × 64 for [W_x | W_h | W_r | b_h] and 64 × 512 for [W_q | b_q].
  * The trace is updated by random-sign mixing, so its expectation is exact RTRL.
  * The gradient is the exact TBPTT-1 gradient plus the carried terms.
* **Matched control: KF-SHUF.** The same rule and signs, with each carried term using another line's learning signal.
* **References** (evaluator instruments, 0 records): BPTT (the ceiling), TBPTT-1 (the floor), and TBPTT-2 and TBPTT-3 (descriptive). TBPTT-k detaches the state (h, r) on entering step s−k+1 for each answer step s.
* **Why diagonal traces were not trained:**
  * The first-hop credit runs q_t → M → r_t → W_r → h_{t+1} → D_{t+2}, cross-unit at every link.
  * The query units have no internal state, so e-prop, RFLO and SnAp-1 on θ_a carry none of it to W_q.
  * SnAp-1 on W_q equals exact RTRL there, about 75 CPU-min per arm.

## C0 testbed (technical worlds; references only; 3 attempts, 0 records)
T0-a (wiring) and T0-b (oracle reads) pass in every attempt:
* V168's reference forward pass equals V166's `Net` bit for bit;
* its BPTT gradient differs by 0.0;
* oracle reads are 100% on Q1 and Q2.

**Test bits per scored position, Q2 (Q1 in brackets):**

| Config (lines × epochs, batch, lr; presentations per arm) | BPTT | TBPTT-1 | TBPTT-2 | TBPTT-3 | Ceiling / floor |
|---|---:|---:|---:|---:|---|
| A1 (400 × 6, 2, 1e-3; 2,400) | 3.909 (4.034) | 4.223 | 3.895 | 3.900 | fail / fail |
| A2 (400 × 6, 2, 2e-3; 2,400) | 3.741 (3.917) | 4.094 | 3.787 | 3.729 | fail / fail |
| B1 (800 × 12, 4, 1e-3; 9,600) | 3.353 (3.541) | 3.499 | 3.374 | 3.177 | fail / fail |
| B2 (800 × 12, 4, 2e-3; 9,600) | 1.787 (1.219) | 3.435 | 3.315 | 0.908 (0.639) | fail / pass |
| D (1,000 × 24, 4, 1e-3; 24,000) | 2.467 (1.865) | 3.334 | 3.196 | 0.385 (0.309) | fail / fail |
| *amendment 1* E1 (800 × 12, 4, 3e-3; 9,600) | **0.556 (0.364)** | **3.497** | 3.326 | 0.546 (0.363) | **pass / pass** (gap 2.94, SD 0.04) |
| E2 (800 × 12, 2, 2e-3; 9,600) | 0.453 (0.244) | 3.491 | 3.322 | 0.515 (0.315) | pass / pass (gap 3.04, SD 0.04) |
| E3 (800 × 24, 4, 2e-3; 19,200) | 0.996 (0.607) | 3.435 | 3.315 | 0.432 (0.292) | fail (Q1) / pass |
| E4 (4,000 × 15, 32, 1e-3; 60,000) | 0.041 (0.075) | 2.709 | 2.642 | 0.059 (0.080) | pass / pass (gap 2.67, SD 0.02) |
| *amendment 2* F1 (400 × 6, 2, 4e-3; 2,400) | 4.047 (4.133) | 4.024 | 3.953 | 3.816 | fail / fail |
| F2 (400 × 6, 2, 6e-3; 2,400) | 4.105 (4.309) | 4.160 | 4.080 | 4.013 | fail / fail |
| F3 (400 × 6, 2, 8e-3; 2,400) | 4.381 (4.542) | 4.516 | 4.389 | 4.389 | fail / fail |
| F4 (1,200 × 2, 2, 6e-3; 2,400) | 3.791 (4.048) | 3.739 | 3.769 | 3.540 | fail / fail |

Ceiling: BPTT Q2 ≤ 1.0 and Q1 ≤ 0.5. Floor: TBPTT-1 − BPTT on Q2 ≥ 1.0 bit and ≥ 3 paired SDs.

**The three attempts:**
* **testbed-001** (design grid A1–D): no config qualifies, so the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED. Closed with decision *revise*.
* **Amendment 1** (testbed_invalid; no learner code existed): extended the budget grid with E1–E4. The ceiling arm, task, metric, thresholds and selection rule are unchanged. TBPTT-3 meeting the ceiling numbers at D was recorded but not used: the ceiling stays BPTT.
* **testbed-002:** CONTINUE, with E1 selected. But C2 at E1 needs 19,200 records. The owner's grant 2 then kept records at 5,000, so the cycle was closed with decision *revise*.
* **Amendment 2** (operational, before any learner outcome):
  * C2 must fit the cap: 2 arms × R + 184 ≤ 5,000, so R ≤ 2,400;
  * configs F1–F4 at 2,400 presentations, with higher learning rates, since the learning rate mattered most at 9,600;
  * the selection is cap-aware.
* **testbed-003:** no cap-fitting config qualifies; higher learning rates made BPTT worse.
  * testbed-002's calibration curves show BPTT still at 2.6–3.0 bits at 4,800 presentations (epoch 6 of E1 and E2). It breaks through only at epochs 7–9.
  * So even a one-learner form (KF only, with TBPTT-1 as the zeroed-trace control) would not fit.
  * Closed with decision *continue*, to the C1 code checks only.

## C1 technical (technical worlds; 7 of 7 checks pass; 184 records; fixture config E1)
1. **TBPTT-1 identity:** KF with the carried terms dropped equals autograd TBPTT-1 (relative error 1.3e-16, float64).
2. **Exact-RTRL identity:** the TBPTT-1 part plus exact carried terms (full influence matrices) equals autograd BPTT (9.5e-16). The carried part is 88% of BPTT's gradient norm at initialisation.
3. **Unbiasedness, exact:** on a 6-step prefix, the mean of the Kronecker trace over all 32 sign sequences equals the exact influence matrix:
   * recurrent group 4.3e-16, query group 3.7e-16;
   * a single draw is off by 3.1 and 1.9 (relative).
4. **SHUF:**
   * every batch's permutation is a derangement;
   * the TBPTT-1 part is identical to KF's;
   * the carried part differs;
   * the identity permutation reproduces KF.
5. **Diagonal traces:** e-prop's query update equals TBPTT-1's (1.5e-16). Non-gating values:
   * cosine with TBPTT-1's update 0.992;
   * cosine with BPTT's gradient 0.393, against 0.389 for TBPTT-1 itself;
   * the recurrent diagonal captures 0.3% of BPTT's carried gradient by projection (norm ratio 0.05).
6. **Determinism:** identical parameters.
7. **Learns:** one KF epoch on 32 lines takes the training bits from 5.110 to 4.879.

**Alignment at initialisation (non-gating, 8 lines):**
* KF single draws: cosine with BPTT's gradient 0.31, 0.23, 0.44 and 0.32; the mean of 4 draws 0.59; TBPTT-1 0.39.
* On the carried part alone: single draws 0.20–0.43, mean of 4 draws 0.54.

**No amendment touched C1:** it ran as designed.

## C2 dev: not run
**What C2 needs** (the design's C2 unchanged, at the smallest qualifying budget E1: 800 lines × 12 epochs, batch 4, lr 3e-3):
* **Learner records:** KF 9,600 + KF-SHUF 9,600 = **19,200**. With the 184 already spent, a records cap of at least **19,384** (for example 20,000).
* **CPU:** about **0.2 CPU-h**:
  * KF about 90 ms per batch-4 update × 2,400 updates, about 4 min per learner arm;
  * four dev references about 1 min each.
  * V168's total would be about 1.2 CPU-h, inside the 3 CPU-h grant.
* **Cycles:** 2 of 3.
* **E2** (batch 2) needs the same records.
* **A one-learner form** (KF only, with the zeroed-trace control, which is exactly the TBPTT-1 reference) needs 9,600 records (9,784 in all). That is still above 5,000, and it loses the noise-matched control.
* **C2 runner:** the design (section 4) fixes the runner, its rules and readings. The runner itself was not written into the delivery.

## Predictions (design §6)
| Event | p | Outcome |
|---|---:|---|
| T0-a and T0-b pass | 0.95 | yes |
| Some config in A1–B2 qualifies | 0.6 | **no** |
| A1 or A2 qualifies (C2 fits the defaults) | 0.2 | no |
| TBPTT-1's Q2 at the selected config ≥ 2.5 bits | 0.75 | yes (3.497 at E1, selected after amendment 1) |
| TBPTT-2 recovers less than half of G | 0.6 | yes (6% at E1) |
| TBPTT-3 within 0.3 bits of BPTT on Q2 | 0.6 | yes at E1 (0.546 against 0.556) |
| C1 check 5: e-prop's θ_q update equals TBPTT-1's | 0.97 | yes |
| P1, KF-SHUF failing P1, PASS overall (given C2 runs) | 0.35, 0.9, 0.3 | unresolved (C2 not run) |

## What it shows
1. **The loop's credit is short but cross-unit, and it is the first hop that needs it.**
   * TBPTT-3 does everything BPTT does, and TBPTT-2 almost nothing beyond TBPTT-1.
   * The answer error must reach the query made two steps earlier, through the memory and W_r.
   * This is exactly the path that diagonal eligibility traces (e-prop, RFLO, SnAp-1 on the recurrent weights) do not contain. For the query weights, e-prop *is* TBPTT-1.
   * A local rule for this loop needs traces that cross units: through the memory read, not only within a neuron.
2. **An unbiased approximate-RTRL trace with a small state carries it in expectation, but noisily.**
   * KF-RTRL's mean is exact.
   * One draw has cosine of about 0.3 with the true gradient at initialisation, against TBPTT-1's 0.39. Averaging 4 draws gives 0.59.
   * Whether that noise still lets it learn the loop is what C2 would measure.
3. **The data budget is the obstacle, not compute.**
   * Exact BPTT shows no progress on the loop until about 5,600–7,200 line presentations, and it qualifies at 9,600 with lr 2e-3 to 3e-3.
   * A learner comparison with a matched control therefore needs about 20,000 learner records. CPU is small, about 0.2 h.
4. **Observation, non-gating:**
   * At small budgets, truncated TBPTT-3 learned faster than full BPTT: at D, Q2 0.385 against 2.467; at B2, 0.908 against 1.787. At larger budgets they are equal.
   * KF-RTRL estimates the full BPTT gradient, including the long study-phase paths. A trace with a short memory (reset or decay over about 3 steps) would target the TBPTT-3 gradient instead, with less variance from old terms.
   * That is a candidate refinement for a later design. It is not tested here.

## Limits
* **Scale:** small synthetic tasks, one seed per arm; DEV_NOT_EVIDENCE. No learner was trained beyond the C1 fixtures, so nothing here measures KF-RTRL's learning.
* **Reference search:**
  * The budget grid was extended twice after reference outcomes (amendments 1–2, both before any learner code or outcome entered the delivery, and both with the arms, task, metric and thresholds unchanged).
  * designs_tried_k = 4.
  * The qualifying configs were found on technical worlds only; the dev worlds are unused.
* **Plausibility:** KF-RTRL is online and forward-only but not local. It needs the exact one-step Jacobian, including the transposed memory read, and a 64 × 512 trace per line.
* **Process notes:**
  * The learner code was drafted in the scratchpad while testbed-002 ran. It entered the delivery after the testbed cycle closed.
  * Before launch it was smoke-tested outside the delivery, on random tensors and on a scratch generator seed (990001–990003), not a V168 world. That smoke used about 184 fixture line passes plus random-tensor passes; nothing was kept, and these are not counted as V168 records.

## Accounting
* **CPU:** 0.973 CPU-h of child CPU over 4 attempts:
  * testbed-001 0.387;
  * testbed-002 0.428;
  * testbed-003 0.155;
  * technical-004 0.003.
* **Records:** 184 learner records, all C1 fixtures. References count 0 (evaluator instruments).
* **Cycles:** 1 of 3 test cycles (technical), plus 3 testbed cycles (not counted); designs tried k = 4.
* **Caps:** 3 CPU-h (grant 2), 5,000 records, 3 cycles. No official run, no adoption.
