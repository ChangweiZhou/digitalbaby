V162 DESIGN (owner go-ahead; written and committed before any V162 code)

# V162 design: freeze the local fixes, then test the new architecture on text (26 October 2026)

**Owner's words:** "Okay. Activate the skill to ship V162."

**Basis:**
* V159–V161 showed on a synthetic task that a shared predictive pathway G, combined with the KC memory S through a familiarity gate, generalises to never-seen members while keeping memory intact.
* What remained was local:
  * G's slow feedback trace: fade 0.3 was better post hoc (V161 diagnostic);
  * the collapse of probability quality when S is switched off.

**Scope:** (a) synthetic confirmation on fresh streams, and (b) text carry-over. **(b) runs whatever (a) shows:** (a) only confirms parameters chosen after the fact, while (b) answers the question that decides the direction. Default caps: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 1. Fixed mechanism (no tuning in V162)
* **S:** R1 on the binary KC code (birth keys) plus the empty-state column.
* **G:** V159's predictive layer: delta rule, η 0.1, γ 0, 2 passes, then frozen. Its feedback trace now fades at **0.3** per byte (it was 0.807), is centred on the training mean, and is read by its own R1.
* **Familiarity and τ:** f_t is V161's familiarity. τ is a percentile of f on calibration.
* **Soft gate:** p ∝ p_S^{a_t} · p_G^β, where a_t = 1 if f_t ≥ τ, else **a_low**. When the memory lacks evidence it is tempered, not removed.

## 2. (a) Synthetic confirmation: fresh streams, seeds 2 and 3 (training, calibration and test at seed, seed + 100, seed + 200)
**Fixed values:** fade 0.3, τ = 5th percentile, β = 1, a_low = 0.25.

**C0 testbed (before learner code):**
* ORACLE-GATED (the soft gate with the oracle expert) must score ≥ 0.90 on unseen and ≥ 0.95 on seen members;
* S alone must score ≤ 0.65 on unseen members;
* on both seeds. Otherwise CANNOT_TEST, but (b) still runs.

**C1 technical:**
* **Checks:**
  * with fade 0.807, a_low = 0 and β = 1, the code reproduces V161's GATED on seeds 0 and 1 (0.8170 and 0.7727 on unseen members);
  * the soft gate equals its formula;
  * determinism.
* **Pass (a)**, on both fresh seeds:
  * GATED ≥ 0.80 on unseen and ≥ 0.95 on seen members;
  * GATED-SHUF ≤ 0.65 on unseen members;
  * mean bits at unseen targets for GATED ≤ S's + 0.5.
* **Validity:** G-SHUF alone ≤ 0.65.

## 3. (b) Text carry-over (C2 dev)
* **Texts:**
  * A = *Thursday*: 40 KB taught, V155's cached traces, SHA-verified;
  * B = *Edgeworth*: 10 KB taught, 5 KB calibration, 10 KB held-out (V155 corpus rules), traces computed here.
* **Predictive layer:** learned on A's taught text (2 passes, 1,410 learner records), then frozen. r and z (fade 0.3) are computed on B.
* **S on B:** R1 with L2 grid {300, 100, 30, 10}, 300 iterations.
* **G on B:** R1 on centred z, L2 grid {10, 1, 0.1, 0.01}.
* **Evidence for the gate:** from B's taught text.
* **Combination, chosen on B's calibration bits:**
  * β ∈ {0.25, 0.5, 1};
  * τ ∈ {none (gate always open), 1st, 5th, 10th percentile};
  * a_low ∈ {0, 0.25, 0.5};
  * plus "S alone".
* **GATED-SHUF:** the same procedure with G trained on a shuffled trace (seed 7). Any gain from merely tempering S then shows up in this control.
* **Pass (b):**
  * GATED beats S on B's held-out lines by ≥ 0.02 bits **and** ≥ 3 line-bootstrap SDs;
  * GATED-SHUF does not, meaning its gain is < 0.02 bits or < 3 SDs.
* **Not run, to save compute:** same-text (A) results.

## 4. Readings fixed in advance
* **(b) passes:** the new architecture helps where language needs it. The next round integrates it with the technical fixes (even fan-out, output port, native local readouts).
* **(b) fails:** the architectural question reopens. The candidates are a richer shared pathway (feeding the KC layer, multi-step prediction, word-scale units) or a different level of prediction. The next design changes the pathway; it does not tune the toy.

## 5. Accounting and predictions
* **Learner records:** about 2,210 (800 synthetic + 1,410 text).
* **Estimated CPU:** about 0.6 h.
* **Predictions (my record is poor):**
  * P1: (a) passes, probability 0.55;
  * P2: (b) passes, probability 0.25.
