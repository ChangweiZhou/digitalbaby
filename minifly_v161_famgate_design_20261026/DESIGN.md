V161 DESIGN (owner go-ahead; written and committed before any V161 code)

# V161 design: let the specific memory step aside when it has no evidence (26 October 2026)

**Owner's words:** "Yes, run V161 with the recursive improvement skill."

**Basis:** V160 (`minifly_v160_arbit_dev_v1_20261026/REPORT.md`).
* Training the specific pathway S (R1 on the KC code) and the shared pathway G (R1 on the feedback trace) separately, then multiplying their predictions, raised accuracy on unseen members from 0.634 / 0.616 to 0.740 / 0.702, with seen members intact.
* It fell short of 0.80 because S is confidently wrong on contexts it has never seen. Even a perfect shared expert lost 5–7 points in the product.

**Scope:** X2 (synthetic, decisive), then X3 (text) only if X2 passes. Default caps: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 1. The single change: a familiarity gate on S
**Evidence table (local, fly-plausible).** For each output class c and each KC j:

e[c, j] = the number of training positions with target c at which KC j was active.

This is a count at the KC→output synapse, like the synaptic change that marks a stored pattern in the fly's KC→MBON novelty detector (Dasgupta et al. 2018).

**Familiarity at a position t:**
* A_t is the set of active KCs.
* ĉ_t is S's most probable class.
* f_t = the fraction of A_t that has evidence for ĉ_t: f_t = |{j ∈ A_t : e[ĉ_t, j] > 0}| / |A_t|, with f_t = 0 for an empty code.

**Threshold τ.** τ is the 5th percentile of f over the calibration stream's positions with a non-empty code. Those are known kinds of contexts in fresh occurrences, and the threshold uses no test data.

**Gated combination:**
* p ∝ p_S^{a_t} · p_G^β, where a_t = 1 if f_t ≥ τ and 0 otherwise.
* β = 1 in X2, as in V160.
* When the gate is off, G decides alone.

**Unchanged from V160:** S, G, the predictive layer, the trace, its centring, the readout grid, the streams and the scoring. V160's and V159's modules are imported unchanged.

## 2. C0 testbed (evaluator-side, before any V161 learner code)
**Streams:** V159's class-transfer streams, seeds 0 and 1.

**Arms and readings:**
* S alone;
* the gate's separation: the share of unseen and of seen target positions that the gate turns off, and the AUC of 1 − f for unseen against seen target positions;
* ORACLE-GATED: the gated combination with V160's oracle expert in G's place.

**CONTINUE if, on both seeds:**
* ORACLE-GATED scores ≥ 0.90 on unseen and ≥ 0.95 on seen members (ceiling);
* S alone scores ≤ 0.65 on unseen members (floor).

Otherwise the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED: the familiarity gate cannot do its job here, and V161 stops before any learner code.

## 3. C1 technical
**Checks:**
* the gate formula against an independent computation;
* τ uses calibration positions only;
* with a = 1 everywhere, the result reproduces V160's PoE exactly (0.7404 / 0.7025 on unseen members);
* G-SHUF alone reproduces V160 (0.4979 / 0.5000);
* determinism.

**X2 arms:** GATED (S gated × G) and GATED-SHUF (S gated × G-SHUF).

**Pass:** on both seeds, GATED scores ≥ 0.80 on unseen **and** ≥ 0.95 on seen members, and GATED-SHUF scores ≤ 0.65 on unseen members.

**Validity:** G-SHUF alone ≤ 0.65 on unseen members.

**If X2 fails:** there is no variant. V161 stops and reports.

## 4. C2 dev: X3 text (only if X2 passes)
**Setup:** as V160's X3. *Thursday* is text A and *Edgeworth* text B. The predictive layer is learned on A and frozen. The comparison is S against GATED and GATED-SHUF.

**What is chosen on calibration bits:**
* β ∈ {0, 0.25, 0.5, 1, 2};
* τ ∈ {0 (no gating), the 1st, 5th and 10th percentile of f on calibration};
* both are chosen jointly, on A's calibration stream for the same-text readouts and on B's for carry-over.

**Pass:** GATED beats S on B's held-out lines by ≥ 0.02 bits **and** ≥ 3 line-bootstrap SDs, and GATED-SHUF does not.

## 5. Accounting, predictions, limits
* **Learner records:** predictive-layer line presentations. C1 is 800; C2 is 1,410.
* **Estimated CPU:** C0 0.1 h, C1 0.1 h, C2 0.5 h.
* **Predictions (my record is poor):**

  | | Prediction | My probability |
  |---|---|---:|
  | P1 | the C0 ceiling passes | 0.6 |
  | P2 | X2 passes | 0.4 |
  | P3 | X3 passes | 0.2 |

* **Not tested:** native local readouts, EVEN keys, own-output generation, unseen addition.
