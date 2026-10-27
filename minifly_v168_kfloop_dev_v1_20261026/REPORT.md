V168 KFLOOP V1: FAIL. KF-RTRL, an online forward rule without back-propagation through time, recovers 3.3% of BPTT's gain over TBPTT-1 on V166's two-hop query: Q2 3.335 bits against TBPTT-1's 3.431 and BPTT's 0.526, with a pass needing 50%. Its matched control does worse (3.657). DEV_NOT_EVIDENCE.

# V168 report: training V166's memory loop without BPTT, first step (26–27 October 2026)

**Question:** can an online rule without back-propagation through time train the loop's recurrent and query weights, so that the error at the answer step reaches the query made two steps earlier?

**Answer: not with KF-RTRL at this budget.**
* With 9,600 line presentations, BPTT learns the two-hop query (0.526 bits) and TBPTT-3 nearly does (0.605).
* KF-RTRL stays at the level of one-step truncation: Q2 3.335 against TBPTT-1's 3.431, or 3.3% of the 2.905-bit gap.
* Its query never learns the two hops. At the Q step of two-hop episodes it targets the cue in 2.5% of cases, against BPTT's 90.9%.
* Its traces do carry some line-specific credit: it beats the same rule fed other lines' errors by 0.32 bits (9 SDs). But the signal is too weak and noisy to find the loop.

**Settled along the way:**
1. **The task needs exactly the first hop's 2-step credit.** At every budget where BPTT learns, TBPTT-3 recovers 97–100% of BPTT's gain over TBPTT-1, while TBPTT-2 recovers 3–6%.
2. **Diagonal eligibility traces cannot carry that credit.**
   * e-prop's update for the query weights equals TBPTT-1's to 1.5e-16.
   * For the recurrent weights, e-prop's diagonal adds 0.3% of BPTT's carried gradient.
3. **KF-RTRL is implemented correctly and is unbiased.**
   * With its carried terms dropped it equals autograd TBPTT-1; with exact carried terms it equals autograd BPTT.
   * Its trace averages exactly to the true influence matrix.

**Files:**
* Design: `minifly_v168_kfloop_design_20261026/DESIGN.md`, committed before any code.
* Grants: `AUTHORIZATION.json`, the owner's words relayed by the main session:
  * grant 1, "Okay, you can do these." (default caps);
  * grant 2, "3 hours should be fine. No need for a lot of work as we are not testing scaling law etc." (3 CPU-h);
  * grant 3, "Run V168 with 20,000 records" (20,000 records).
* Ledgers in `runlog/`:
  * 5 attempts, none failed;
  * cycles: testbed *revise*, testbed *revise*, testbed *continue*, technical *stop*, dev *stop*;
  * amendments 1 (testbed_invalid) and 2 (operational), both before any learner code existed in the delivery.
* Results:
  * `results/testbed/TESTBED.json`, `TESTBED_2.json`, `TESTBED_3.json`;
  * `results/technical/CHECKS.json`;
  * `results/dev/RESULTS.json`.
* Run by a separate agent in parallel with V167. The harness did not let the agent write this file; the main session checked the key numbers against the result files and saved the agent's text unchanged apart from this note.

## Setup (as designed)
* **Fixed from V166:**
  * the memory: KC-like code, 512 KCs, top 16, Hebbian KC → MBON writes, reset per line;
  * the 34-step lines, the MIXED task (2 one-hop and 2 two-hop episodes per line);
  * the LOOP cell (H = 64, 47,072 parameters, initialisation seed 0), scoring and line bootstrap.
  * V166's library is loaded read-only by file path.
  * New worlds: technical seeds 1681–1683; dev seeds 1691–1693, first used in C2.
* **Learner: KF-RTRL** (Mujika, Meier and Steger 2018), with exact immediate terms.
  * Each weight group's influence ∂h_t/∂θ is carried per line as one Kronecker product. The trace is 64 × 64 for [W_x | W_h | W_r | b_h] and 64 × 512 for [W_q | b_q].
  * The trace is updated by random-sign mixing, so its expectation is exact RTRL.
  * The gradient is the exact TBPTT-1 gradient plus the carried terms.
* **Matched control: KF-SHUF.** The same rule and signs, with each carried term using another line's learning signal. The zeroed-trace control is TBPTT-1 itself.
* **References** (evaluator instruments, 0 records): BPTT (the ceiling), TBPTT-1 (the floor), and TBPTT-2 and TBPTT-3 (descriptive).
* **Shared by every arm:** the same forward pass, initialisation, data order, AdamW, clipping at 5 and best-calibration-epoch selection. Only the gradient estimate differs.
* **Why diagonal traces were not trained:**
  * The first-hop credit runs q_t → M → r_t → W_r → h_{t+1} → D_{t+2}, cross-unit at every link.
  * The query units have no internal state, so e-prop, RFLO and SnAp-1 on θ_a carry none of it to W_q.

## C0 testbed (technical worlds; references only; 3 attempts, 0 records)
T0-a (wiring) and T0-b (oracle reads) pass in every attempt:
* V168's forward pass equals V166's `Net` bit for bit;
* its BPTT gradient differs by 0.0;
* oracle reads are 100%.

**Test bits, Q2 (Q1 in brackets):**

| Config (lines × epochs, batch, lr; presentations per arm) | BPTT | TBPTT-1 | TBPTT-2 | TBPTT-3 | Ceiling / floor |
|---|---:|---:|---:|---:|---|
| A1 (400 × 6, 2, 1e-3; 2,400) | 3.909 (4.034) | 4.223 | 3.895 | 3.900 | fail / fail |
| A2 (400 × 6, 2, 2e-3; 2,400) | 3.741 (3.917) | 4.094 | 3.787 | 3.729 | fail / fail |
| B1 (800 × 12, 4, 1e-3; 9,600) | 3.353 (3.541) | 3.499 | 3.374 | 3.177 | fail / fail |
| B2 (800 × 12, 4, 2e-3; 9,600) | 1.787 (1.219) | 3.435 | 3.315 | 0.908 (0.639) | fail / pass |
| D (1,000 × 24, 4, 1e-3; 24,000) | 2.467 (1.865) | 3.334 | 3.196 | 0.385 (0.309) | fail / fail |
| *amendment 1* **E1 (800 × 12, 4, 3e-3; 9,600)** | **0.556 (0.364)** | **3.497** | 3.326 | 0.546 (0.363) | **pass / pass** (gap 2.94, SD 0.04) |
| E2 (800 × 12, 2, 2e-3; 9,600) | 0.453 (0.244) | 3.491 | 3.322 | 0.515 (0.315) | pass / pass |
| E3 (800 × 24, 4, 2e-3; 19,200) | 0.996 (0.607) | 3.435 | 3.315 | 0.432 (0.292) | fail (Q1) / pass |
| E4 (4,000 × 15, 32, 1e-3; 60,000) | 0.041 (0.075) | 2.709 | 2.642 | 0.059 (0.080) | pass / pass |
| *amendment 2* F1–F4 (2,400; lr 4e-3 to 8e-3, batch 2) | 3.79–4.38 | 3.74–4.52 | 3.77–4.39 | 3.54–4.39 | fail / fail |

Ceiling: BPTT Q2 ≤ 1.0 and Q1 ≤ 0.5. Floor: TBPTT-1 − BPTT on Q2 ≥ 1.0 bit and ≥ 3 paired SDs.

**The three attempts:**
* **testbed-001** (design grid A1–D): no config qualifies, so the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED.
* **Amendment 1** (testbed_invalid; no learner code existed): extended the grid with E1–E4. The ceiling arm, thresholds and selection rule are unchanged. TBPTT-3 meeting the ceiling numbers at D was recorded but not used.
* **testbed-002:** E1 selected.
* **Grant 2** kept records at 5,000, so amendment 2 (operational) required C2 to fit the cap and added F1–F4 at 2,400 presentations per arm, with cap-aware selection.
* **testbed-003:** no config that fits the cap qualifies. BPTT is still at 2.6–3.0 calibration bits at 4,800 presentations.
* **C1 and the stop:** V168 ran C1 and stopped for the cap. **Grant 3** then raised records to 20,000, and the same cap-aware rule selects E1 (C2 19,200 records, total 19,384).

## C1 technical (technical worlds; 7 of 7 checks pass; 184 records)
1. **TBPTT-1 identity:** KF with the carried terms dropped equals autograd TBPTT-1 (1.3e-16).
2. **Exact-RTRL identity:** the TBPTT-1 part plus exact carried terms equals autograd BPTT (9.5e-16). The carried part is 88% of BPTT's gradient norm at initialisation.
3. **Unbiasedness, exact:** the mean of the Kronecker trace over all 32 sign sequences equals the exact influence matrix (4.3e-16 for the recurrent group, 3.7e-16 for the query group). A single draw is off by 3.1 and 1.9 (relative).
4. **SHUF:**
   * every batch's permutation is a derangement;
   * the TBPTT-1 part is identical to KF's;
   * the carried part differs;
   * the identity permutation reproduces KF.
5. **Diagonal traces:** e-prop's query update equals TBPTT-1's (1.5e-16). Non-gating values:
   * cosine with TBPTT-1's update 0.992;
   * cosine with BPTT's gradient 0.393, against 0.389 for TBPTT-1 itself;
   * the recurrent diagonal captures 0.3% of BPTT's carried gradient.
6. **Determinism:** identical parameters.
7. **Learns:** 5.110 → 4.879 training bits in one epoch on 32 lines.

**Alignment at initialisation (non-gating):** a single KF draw has cosine 0.23–0.44 with BPTT's gradient; the mean of 4 draws has 0.59; TBPTT-1 has 0.39.

## C2 dev (fresh dev worlds; E1; all six arms trained)
**Test bits per scored position** (2,000 Q1 and 2,000 Q2 answers; accuracy in brackets):

| Arm | Q1 (one-hop) | Q2 (two-hop) | Best epoch / run | CPU s |
|---|---:|---:|---:|---:|
| BPTT (ceiling) | 0.338 (93.4%) | **0.526** (88.9%) | 12 / 12 | 65 |
| TBPTT-1 (floor) | 3.863 (15.8%) | **3.431** (18.7%) | 6 / 12 | 64 |
| TBPTT-2 | 3.705 (16.2%) | 3.302 (20.6%) | 5 / 12 | 65 |
| TBPTT-3 | 0.428 (90.1%) | 0.605 (86.7%) | 12 / 12 | 64 |
| **KF** | 3.649 (15.4%) | **3.335** (19.2%) | 6 / 12 | 220 |
| **KF-SHUF** | 3.769 (13.8%) | 3.657 (13.7%) | 12 / 12 | 218 |

**Gating:**

| Rule | Value | Required | Result |
|---|---|---|---|
| R-dev: BPTT ceiling and gap | Q2 0.526, Q1 0.338; G = 2.905 bits (SD 0.041) | Q2 ≤ 1.0, Q1 ≤ 0.5; G ≥ 1.0 and ≥ 3 SDs | pass |
| P1: TBPTT-1 − KF on Q2 | **0.096** bits (SD 0.032, z 2.98) = **3.3% of G** | ≥ 0.5 G and ≥ 3 SDs | **fail** |
| P2: KF-SHUF fails P1, and KF-SHUF − KF ≥ 3 SDs | KF-SHUF −0.226 (−7.8% of G); KF beats it by 0.322 (SD 0.036, z 9.0) | both | pass |
| G1: KF's Q1 ≤ TBPTT-1's Q1 + 0.5 | 3.649 against 3.863 | holds | pass |

**Verdict: FAIL** (P1 fails).

**Diagnostics (non-gating):**
* **Recovery shares of G:** KF 3.3%, KF-SHUF −7.8%, TBPTT-2 4.5%, TBPTT-3 97.3%. KF is 2.809 bits worse than BPTT (SD 0.039).
* **Query targeting** (2,000 two-hop episodes): the share where the key with the largest k_i · q_t is the cue at the Q step, and where it is the intermediate at the WAIT step:

  | | Q step: cue | WAIT step: intermediate |
  |---|---:|---:|
  | BPTT | 90.9% | 89.2% |
  | TBPTT-3 | 78.2% | 83.5% |
  | KF | 2.5% | 21.9% |
  | KF-SHUF | 8.6% | 16.7% |
  | TBPTT-1 | 1.8% | 22.9% |

  KF behaves like TBPTT-1: no two-hop retrieval.
* **Calibration curves:**
  * KF follows TBPTT-1: best at epoch 6 (3.49), then rising to 3.66 by epoch 12.
  * BPTT and TBPTT-3 break through at epochs 7–9.
  * KF-SHUF falls slowly to 3.66 (best epoch at the edge).

## Predictions (design §6)
| Event | p | Outcome |
|---|---:|---|
| T0-a and T0-b pass | 0.95 | yes |
| Some config in A1–B2 qualifies | 0.6 | no |
| A1 or A2 qualifies | 0.2 | no |
| TBPTT-1's Q2 at the selected config ≥ 2.5 bits | 0.75 | yes (3.497 on technical, 3.431 on dev) |
| TBPTT-2 recovers less than half of G | 0.6 | yes (4.5% on dev) |
| TBPTT-3 within 0.3 bits of BPTT on Q2 | 0.6 | yes (0.605 against 0.526 on dev) |
| C1 check 5: e-prop's θ_q update equals TBPTT-1's | 0.97 | yes |
| P1, given C2 runs | 0.35 | **no** |
| KF-SHUF fails P1's criterion, given C2 runs | 0.9 | yes |
| PASS overall, given C2 runs | 0.3 | **no** |

## What it shows
1. **By the reading fixed in advance (FAIL, P1 failing):**
   * Even an unbiased approximate-RTRL trace with a small state does not recover half of BPTT's gain at this budget. Its noise, or the budget, is the obstacle.
   * The diagonal local rules carry strictly less of this path (C1 check 5).
   * Options named in advance: lower-variance estimators (Kronecker sums of rank r, several samples), a larger budget, or an architecture with a shorter credit path.
2. **The trace's signal is real but too weak.**
   * KF beats its shuffled control by 0.32 bits (9 SDs), and beats TBPTT-1 by 0.10 (about TBPTT-2's level).
   * Fed the wrong line's error, the same carried term costs 0.23 bits against TBPTT-1. The noise is large enough to matter.
   * At initialisation one KF draw is less aligned with the true gradient (cosine about 0.3) than TBPTT-1's biased gradient (0.39). Over 2,400 updates this did not add up to the breakthrough that BPTT makes at epochs 7–9.
3. **The credit needed is short and cross-unit.**
   * TBPTT-3 learns the loop as well as BPTT; TBPTT-2 does not.
   * The error must reach the query two steps back through the memory and W_r. That is the path that within-neuron traces lack.
   * A local rule for this loop needs a trace that crosses units through the memory read.
4. **Next candidates (not tested):**
   * a trace with a short memory that targets TBPTT-3's gradient, which learns at least as fast as BPTT here and drops the variance of old terms;
   * lower-variance Kronecker estimates (several samples or rank r);
   * then, if one learns the loop, a local version (random feedback for the learning signal) and MiniFly's real KC code and bytes.

## Limits
* **Scale:** small synthetic tasks, one seed per arm, one budget; DEV_NOT_EVIDENCE.
* **Reference search:**
  * The budget grid was extended twice after reference outcomes (amendments 1–2, before any learner code or outcome entered the delivery; arms, task, metric and thresholds unchanged).
  * designs_tried_k = 5.
  * The selected budget uses lr 3e-3. A noisier estimator may suffer more from a high learning rate than BPTT does, and that was not varied.
* **Plausibility:** KF-RTRL is online and forward-only but not local. It needs the exact one-step Jacobian, including the transposed memory read, and a 64 × 512 trace per line.
* **Process notes:**
  * The learner code was drafted in the scratchpad while testbed-002 ran; it entered the delivery after the testbed cycle closed.
  * The learner and the runners were smoke-tested outside the delivery, on random tensors and scratch generator seeds (990001–990003, 990011–990013), not V168 worlds. Nothing was kept; these passes are not counted as records.
* **Grant 3** reopened C2 after the technical cycle had been closed with decision *stop* for the record cap. The C2 design was unchanged.

## Accounting
* **CPU:** 1.175 CPU-h of child CPU over 5 attempts:
  * testbed-001 0.387;
  * testbed-002 0.428;
  * testbed-003 0.155;
  * technical-004 0.003;
  * dev-005 0.202.
* **Records:** 19,384 learner records: C1 184, and C2 19,200 (KF 9,600 + KF-SHUF 9,600). References count 0.
* **Cycles:** 2 of 3 test cycles (technical, dev), plus 3 testbed cycles (not counted); designs tried k = 5.
* **Caps:** 3 CPU-h (grant 2), 20,000 records (grant 3), 3 cycles. No official run, no adoption.
