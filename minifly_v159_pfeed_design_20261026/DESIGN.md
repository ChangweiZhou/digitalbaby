V159 DESIGN (owner go-ahead; written and committed before any V159 code)

# V159 design: a predictive feedback layer, tested by small falsifiable experiments (26 October 2026)

**Owner's words:** "Let us activate the recursive improvement skill and do the experiments."

**Scope:** X1 and X2 (synthetic), X3 (text, only if X2 passes) and X4 (an independent technical check). Default caps: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 1. The hypothesis
**Why the current system cannot generalise.** MiniFly's KC code is a similarity-preserving hash of recent input. Two contexts share code only if they share recent bytes. V155 and V158 found no way to make key learning group contexts by their *futures*.

**The proposed component.** A predictive feedback layer (PF):
* **Prediction.** r_t = M · [x̃_t; 1], where x̃_t is the binary KC code divided by its number of active units. r_t predicts the receptor pattern φ(s_t) of the next symbol: φ is the fixed 88-channel byte pattern (`BC.R_TABLE`), and the zero vector for end-of-line.
* **Learning.** M learns by the local delta rule:

  M ← M + η · (φ(s_t) − r_t) · [x̃_t; 1]ᵀ, with η = 0.1.

  Each output unit uses only its own error and its presynaptic activity. There is no back-propagation.
* **Feedback.** The predictions enter a trace that fades like the byte trace:

  z_t = Σ_{k≥1} 0.807^(k−1) · r_{t−k}, reset at each line start.

  z_t holds only past predictions; r_t itself is excluded.
* **What MiniFly reads.** The readout reads [x_t, z_t] instead of x_t alone. This arm is called PF-READ.

**Why it should help.** Contexts followed by similar things get similar r_t, by construction. Through z, that shared information is still present at *later* positions. A linear readout of z can therefore carry what was learned for one member of a class over to another member. Feeding r_t into the same step's readout would add nothing, because r_t is linear in x_t. That is why only past predictions enter z.

**Deferred.** PF-KC, in which z enters the KC layer's input, is left for later. PF-READ tests the information question without a moving KC code.

## 2. Experiments and rules (all fixed here)
Common to all:
* front end P81, untaught (V153);
* birth keys from V155 `codes.py`, frozen (except in X4);
* evaluator readout R1 (V149 logistic regression, L2 chosen on a calibration stream).

The native-learnable ONLINE readout (V153; η 0.01, 3 passes) is reported where it is cheap. It does not gate any decision.

### X1. Does PF have the property? (C1, synthetic)
* **Streams:** V155's two fixture streams (seeds 0 and 1, 300 lines). Bytes of a set share their futures; their pasts are identical in distribution.
* **Learning:** M learns over 2 passes, then is frozen.
* **Index:** the same-future index on r: mean cosine of r over 2,000 pairs of positions ending in different bytes of the same set, minus that over 2,000 pairs from different sets (V155 pair sampling, seeds 11 and 12).
  * For binary codes with 184 active units, the cosine equals V155's overlap index. FROZEN scored 0.004 and 0.005.
* **Pass:** index ≥ 0.5 on both streams. Failure means an implementation defect (stop, fix within the cycle, report).

### X2. Does PF's shared signal produce generalisation? (C0 sets the ceiling and floor; C1 runs the PF arms)
**Alphabet:**
* class 1: S1 = a…h, which reveals its class through F1 = q…x;
* class 2: S2 = i…p, which reveals its class through F2 = y z A…F;
* neutral byte n = '-';
* targets: g1 = '!' for class 1, g2 = '?' for class 2.

**Lines:** 10 segments, separated by spaces. Each segment picks a class and a member uniformly at random.
* **Type 1:** "s f", with f drawn from F of s's class. Every member appears this way.
* **Type 2:** "s - g", with g the target of s's class.
  * In training and calibration, type 2 occurs only for the **seen** members (a–d and i–l), with probability 0.5 when such a member is drawn.
  * In the test stream, type 2 occurs for every member with probability 0.5, so **unseen** members (e–h and m–p) appear before g only there.

**Streams per seed:** 200 training lines (seed s), 50 calibration lines (seed s + 100), 100 test lines (seed s + 200), for s ∈ {0, 1}.

**Arms** (features at each position; the readout is R1, fitted on training and calibration):

| Arm | Features |
|---|---|
| FROZEN | x |
| ORACLE | x plus a trace of the true class of each byte read (2 channels, same fade, reset per line) |
| PF-READ | x plus z |
| SHUFFLED | x plus a trace built from r values permuted across positions (seed 7) |

**Score:** accuracy at every g position in the test stream. A position counts as correct if P(true g) > P(other g), and as 0.5 on a tie. Accuracy is reported separately for unseen and seen members. The bits at g positions are reported too.

**C0 (testbed, before PF code):** FROZEN and ORACLE only.
* CONTINUE if ORACLE's unseen-member accuracy is ≥ 0.90 (ceiling) and FROZEN's is ≤ 0.65 (floor) on both seeds.
* Otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED, and the task is redesigned before any PF code.

**C1 pass (X2):** on both seeds, PF-READ's unseen-member accuracy is ≥ 0.80, while FROZEN and SHUFFLED stay ≤ 0.65.

**If X2 fails:** one variant is allowed. M then predicts a fading sum of future patterns: target φ(s_t) + 0.5 · r_{t+1} within a line, a TD-style local rule. Everything else stays the same. If the variant also fails, the hypothesis is falsified in this form: stop and report. X3 is not run.

### X3. Text (C2; only if X2 passes)
* **Texts:** *Thursday* as A (40 / 5 / 10 KB, V155's cached traces, verified against V155's recorded SHA-256) and *Edgeworth* as B (10 / 5 / 10 KB).
* **Learning:** M learns on A's taught text over 2 passes with η = 0.1 (no grid), then is frozen.
* **Arms:** FROZEN, PF-READ and SHUFFLED.
* **Scored with:**
  * same-text: A held-out bits;
  * carry-over: readout trained on B's 10 KB, scored on B held-out;
  * by R1 (L2 grids as V155) and ONLINE.
* **Pass:** PF-READ beats FROZEN on B held-out by ≥ 0.02 bits **and** ≥ 3 line-bootstrap SDs (R1). SHUFFLED must not.
  * The same-text result is reported. It is expected to be small, because counting is near the limit at 40 KB.

### X4. Even fan-out (C0; evaluator-side, independent)
* **EVEN keys:** each KC keeps its number of inputs and its weights. Each synapse's source is reassigned so that every one of the 88 receptor channels reaches the same number of synapses, ±1. Within a channel, a projection neuron is chosen uniformly (seed 4).
* **Scored by** R1 on *Thursday* (V155 cache) against FROZEN, which is 2.963.
* Descriptive only: the difference is reported with its line-bootstrap SD, and nothing is gated on it.

## 3. Accounting
* **Learner records:** M's line presentations. X1 is 1,200 and X2 is 800 (a further 800 if the variant runs). X3 is 1,410. The total is at most 4,210.
* **Evaluator fits** (R1, ONLINE) are reported separately.
* **Estimated CPU:** C0 about 0.2 h, C1 about 0.1 h, C2 about 0.4 h.

## 4. Predictions (my record on these is poor)
| | Prediction | My probability |
|---|---|---:|
| P1 | X1 passes | 0.9 |
| P2 | X2 PF-READ passes | 0.6 |
| P3 | X3 PF-READ passes | 0.3 |
| P4 | X4: EVEN improves on FROZEN by ≥ 0.02 bits | 0.4 |

## 5. What this does not test
* PF-KC, in which predictions shape the KC code.
* Native integration, meaning the 16-compartment port and own-output generation.
* Retention.
* Unseen addition, which probably also needs multi-step computation.

## 6. Testbed-driven redesign of X2's readout (after C0 attempt 1; before any predictive-layer code)
**C0 attempt 1:** X2 was CANNOT_TEST_CLAIM_AS_SPECIFIED. ORACLE reached only 0.723 and 0.694 on unseen members (0.90 required). FROZEN gave 0.638 and 0.607. Both arms chose L2 = 10, the top of the grid.

**Cause (evaluator-side).** With L2 regularisation, explaining the seen members' targets costs far less when small weights are spread over the 184 active, member-specific KC features than when weight is put on 2 shared class channels. A contribution c costs about c²/184 one way and c²/0.65 the other. So the readout under-uses any low-dimensional shared signal. Any fed-back channel block, including PF's, would meet the same bias.

**Change, applied identically to every arm with extra channels (ORACLE now; PF-READ and SHUFFLED in C1; X3):**
* **Block normalisation.** The extra-channel block is multiplied by one factor, computed on the training stream so that its mean squared norm per position equals that of the KC block, including the empty-state column. The same factor is applied to the calibration and test streams.
* **Wider L2 grid:** {100, 30, 10, 3, 1, 0.3, 0.1}.

Thresholds, streams and arms are unchanged. **This is the only testbed redesign:** if the ceiling still fails, X2 is untestable as designed, and V159 stops X2 and reports.

**X4 (no change):** EVEN keys scored 2.867 against FROZEN's 2.963 (−0.096, line-bootstrap SD 0.011).
