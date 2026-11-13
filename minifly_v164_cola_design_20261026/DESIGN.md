V164 DESIGN (owner go-ahead; written and committed before any V164 code)

# V164 design: a small test of COLA itself (26 October 2026)

**Owner's words:** "Okay, can we try COLA then? It does not need to be a big experiment, a small one is fine."

**Basis:** V163 never built COLA. On *Thursday*, exact back-propagation through time (BPTT) was worse than one-step truncation (TBPTT-1), so COLA had nothing to approximate there.

**The fix:** test COLA where credit across time is shown to matter, using two small synthetic tasks fixed now. The text run is reported as description.

**Caps:** the defaults (2 CPU-h, 5,000 learner records, 3 test cycles). No official run.

## 1. Learner (V163 lane C, unchanged)
* **Network:** a single tanh recurrent layer of 256 units with a linear softmax readout, reset at each line start.
* **Initialisation:** near-critical gain from the COLA paper's two-point Lyapunov estimate (Eq. 12), computed once per task and copied to every arm. Input weights have variance 1 / (number of active inputs).
* **Training:** batches of 16 lines; one AdamW step per batch (learning rate 1e-3, weight decay 1e-4, norm clip 5); best calibration epoch kept.
* **Arms:**
  * RAND-R: frozen recurrence, trained readout;
  * TBPTT-1;
  * BPTT within the line;
  * COLA (V163 design §3): Eq. 23 teaching signal; α and μ by Eqs. 25, 28 and 30; ρα 0.995, ρμ 0.98; α clipped to ±0.99; |μ φ′| ≤ 0.99; denominator floor 1e-3; statistics averaged over the lines in a batch and carried across batches.
* **Deviation from the paper, as in V163:** AdamW, not SGD.

## 2. Tasks
**Synthetic tasks** (one-hot input over 8 symbols; random lines of 120 symbols; 100 training, 20 calibration and 50 test lines; fixed seeds):
* **S-COPY:** the target at t is x_{t−8}.
* **S-SUM:** the target at t is (x_{t−2} + x_{t−6}) mod 8.
* Positions without a defined target are not scored.
* Up to 10 epochs, stopping after 3 without improvement.
* Metric: bits per scored position on test, from the trained readout.

**Text** (*Thursday*, V163's KC-code input):
* up to 4 epochs for every arm (a matched budget), stopping after 3 without improvement;
* metric: held-out bits from the trained readout, which avoids V163's probe defect;
* descriptive only.

## 3. Cycles and rules
**C0 testbed** (references only, before COLA code):
* RAND-R, TBPTT-1 and BPTT on both synthetic tasks and on text.
* A synthetic task **qualifies** if BPTT beats TBPTT-1 by ≥ 0.3 bits and ≥ 3 line-bootstrap SDs on test.
* COLA is tested on every qualifying task. If none qualifies, the verdict is CANNOT_TEST and COLA is not built.

**C1 technical:**
* with μ fixed at 0, COLA's gradient equals TBPTT-1's (float64, ≤ 1e-10);
* the teaching signal equals Eq. 23;
* α and μ match Eqs. 25–30 on a hand-computed two-unit case;
* runs are deterministic.

**C2 dev:**
* **Pass, per qualifying task:** COLA beats TBPTT-1 by ≥ half of (TBPTT-1 − BPTT) **and** ≥ 3 SDs.
* **Text:** COLA against TBPTT-1 and BPTT, reported.
* **Diagnostics:** the distributions of α and μ, and the share of steps where μφ′ sits at its cap. The paper found PTB's μ at its cap.

## 4. Readings fixed in advance
* **Pass:** COLA's local credit across time works in our network, where it matters. It is a candidate for tasks that need multi-step state, such as addition.
* **Fail on a qualifying task:** COLA's scalar closure does not capture what BPTT gains here, at this scale and with this optimiser.
* **Text:** expected to be close to TBPTT-1, as in V163 and the paper.

## 5. Accounting and predictions
* **Learner records** (lines × epochs for COLA only; references and C1 fixtures count as in V163):
  * synthetic: ≤ 1,000 per qualifying task;
  * text: ≤ 2,820;
  * C1: about 100.
* **Estimated CPU:** about 0.4 h.
* **Predictions:**
  * at least one task qualifies: 0.7;
  * COLA passes on a qualifying task: 0.35.
