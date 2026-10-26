V163 DESIGN (owner asked for a design; written and committed before any V163 code; no run is authorised yet)

# V163 design: local self-supervised learning under a back-propagated top, and COLA, on MiniFly's KC code (26 October 2026)

**Owner's words:** "While ChatGPT is doing the review, we try to use the two paper's ideas into the next iteration. So we are going to try local SSL with last few layer doing back-propagation or COLA. Can you design a small experiment to do this?"

**What changes in the programme:** back-propagation enters the learner for the first time, confined to the top layers, at the owner's direction. Everything below the top is local, and the KC layer stays frozen. A pass would show that "local learning below, back-propagation on top" works here. It would not show that a learner free of back-propagation does.

**Interpretation of "last few layers doing back-propagation" (to be confirmed before lock):**
* The top block (one hidden layer plus the softmax output) is trained by back-propagation on next-byte cross-entropy.
* The gradient stops at the top block's input.
* Below the top block, lane L uses a layer-local self-supervised rule. Lane C uses COLA for a recurrent layer's credit across time; its within-step error arrives through the readout weights, which is the back-propagated part.

## 0. Basis
**Measured facts that fix where learning can go:**
* *Thursday*, 40 KB taught, held-out bits/symbol:
  * frozen KC code with the linear readout R1: 2.963;
  * a hidden layer trained by back-propagation on the 88-channel trace (T3): 2.699;
  * PPM-4: 2.606.
* Learning the PN→KC keys on MiniFly's native wiring has almost no headroom, even with back-propagation:
  * REF-N (graded keys on the native synapses): 2.931, only 0.032 better than frozen;
  * REF-D (the same with dense connections): 2.740.
  * So V163 learns **above** the KC layer, where connections can be dense, not in the keys. V155 and V158 failed in the keys.
* Carry-over to *Edgeworth* with 10 KB of readout data: R1 on the KC code scores 3.320 (V162).

**The two papers:**
* **Zihan et al. 2026.** A layer-local contrastive predictive rule (CLAPP) approximates self-supervised back-propagation and matches it on three of four image benchmarks. The conditions are:
  * continuous units, every one of which receives the learning signal;
  * unconstrained weights;
  * a learnable lateral prediction matrix;
  * explicit negatives;
  * Adam and many epochs.

  V155 tested a rule of this type under MiniFly's KC constraints (binary top-k winners, a fixed weight budget per KC), which break the first two. Lane L restores all five.
* **Wang et al. 2026 (COLA).** Two per-unit scalars (α, μ) replace back-propagation through time.
  * On PTB-Char it matches one-step truncation (1.835 against 1.832 for TBPTT-1).
  * Its PTB-Char results are flat across its whole sensitivity grid (EMA rates and window) because μ sits at its projection cap. On character language modelling it therefore behaved like one-step truncation with a per-unit rescaling.
  * Lane C first measures whether exact back-propagation through time has any advantage over one-step truncation here. COLA is built only if it does.

## 1. Shared protocol
* **Texts:** V155 corpus rules, SHA-pinned.
  * A = *Thursday*: 40 / 5 / 10 KB (41,017 / 5,132 / 10,277 positions; 705 taught lines).
  * B = *Edgeworth*: 10 / 5 / 10 KB (10,287 / 5,146 / 10,266 positions).
* **Input to every arm:** MiniFly's frozen KC code (cell P81, birth keys, top-92 per side) plus V149's empty-state column. That gives 5,178 binary inputs, about 181 of them active.
  * *Thursday*'s codes are the headroom check's saved codes, checked equal to V155's cached encode.
  * *Edgeworth*'s codes are computed from V162's cached traces with the same birth keys; the SHA is recorded.
* **Probe (the back-propagated top):** the headroom check's T3 learner, unchanged.
  * Architecture: input → 2,048 ReLU units → dropout 0.5 → softmax over the symbols seen in the taught text, plus OTHER.
  * Training: AdamW, learning rate 1e-3, weight decay 1e-4, batch 256, seed 0, one thread.
  * Stopping: at most 60 epochs; stop after 6 epochs without improvement in calibration bits; keep the best calibration epoch.
  * A **linear probe** (softmax regression, same optimiser and stopping) is reported alongside. It is the evaluation Zihan et al. use.
* **Two readings per arm:**
  * **Same text:** layer and probe both learned on A's taught text, scored on A's held-out.
  * **Carry-over:** the layer is learned on A and frozen. A fresh probe is fitted on B's 10 KB taught text, stopped on B's calibration and scored on B's held-out. This reading is comparable with V162 (R1 on the KC code, 3.320).
* **Scoring:**
  * bits/symbol with V149's class rules;
  * differences with line-bootstrap SDs (1,000 resamples of held-out lines).
* **Accounting:**
  * Learner records are passes over A's 705 taught lines by the layer under test (the SSL layer, the COLA layer).
  * Probes and back-propagation references are evaluator fits, reported in epochs and CPU, not records. This is how the headroom check and V155's references were counted. If the owner wants probe epochs counted, the caps in §6 must change.

## 2. Lane L: a local self-supervised layer under a back-propagated top
**Architecture:** KC code → H (512 ReLU units, dense) → probe.

**Arms** (only H differs; all H start from the same random initialisation, W ~ N(0, 1/184), b = 0):

| Arm | H | Role |
|---|---|---|
| TOP | none: the probe reads the KC code directly | does back-propagation in the top alone close the gap to T3? |
| RAND-H | the initialisation, frozen | control; also SSL-H's starting point |
| SSL-H | learned from that initialisation by the local rule below, with no next-byte labels | the test |
| BP-H | learned on A end-to-end with a probe, by back-propagation | ceiling: what this slot can give |

**The local rule** (CLAPP adapted to a sequence; the CLAPP row of Zihan et al.'s Table 1):
* **Activity:** h_t = ReLU(W x_t + b), where x_t is the KC code at position t.
* **Pairs:** the positive pair is (h_t, h_{t+1}) within a line. One negative h_n is drawn from a uniformly random taught position, with a fixed seed.
* **Loss:** L = max(0, 1 − h_{t+1}ᵀ B h_t) + max(0, 1 + h_nᵀ B h_t).
  * B is a learnable 512 × 512 prediction matrix, initialised at 0.
* **Update:** the gradient of L with respect to W, b and B, with every other layer detached.
  * Each update of W is a product of three factors: presynaptic activity and the unit's own derivative; a lateral prediction (B h); and a global gate that is nonzero only when a margin is violated.
  * C1 checks that it equals this explicit three-factor formula.
* **Training:** Adam, learning rate 1e-3, batch 256 pairs, 8 passes.
  * Checkpoints after 1, 2, 4 and 8 passes.
  * The checkpoint is chosen by B's calibration bits under the probe; B's held-out lines are scored once.

## 3. Lane C: COLA for a learned recurrent layer (gated by C0)
**Architecture:**
* KC code → r_t = tanh(W_rr r_{t−1} + W_rk x_t + b), with 256 units, reset at each line start.
* During training the readout is linear softmax, as in the COLA paper. The trained layer is then frozen and read by the same probes as lane L.

**Initialisation** (paper §3.2, computed once and copied to every arm):
* W_rk ~ N(0, 1/184);
* W_rr = g · W̄, where W̄ is Gaussian with operator norm 1;
* g comes from the paper's two-point Lyapunov estimate (Eq. 12), with probes at g = 1 and 1.1 on 100 taught lines.

**Arms:**
* RAND-R: the initialisation, frozen.
* TBPTT-1: r_{t−1} is treated as a constant.
* BPTT: exact back-propagation within each line.
* COLA:
  * teaching signal by Eq. 23; α and μ by Eqs. 25, 28 and 30;
  * the paper's settings: ρα 0.995, ρμ 0.98, α clipped to [−0.99, 0.99], |μ φ′| ≤ 0.99, denominator floor 1e-3;
  * statistics averaged over the lines in a batch.

**The same update schedule for every arm:**
* gradients are summed over a batch of 16 lines;
* one AdamW step per batch (learning rate 1e-3, norm clip 5);
* at most 15 epochs, stopping after 4 without improvement in A's calibration bits.

**Deviation from the paper:** AdamW instead of SGD at 1e-3. Our 40 KB give about 400 times fewer character presentations than their PTB-Char runs (0.6 million against 255 million). Adam partly normalises COLA's per-unit rescaling, so a COLA result here is for this optimiser.

## 4. Cycles and rules fixed in advance
**C0 testbed** (evaluator-side, before any learner code; no learner records):
* **Parity:**
  * the probe on the 88-channel trace reproduces T3's 2.6995 within 0.001;
  * the exported *Thursday* codes equal the headroom check's codes.
* **Lane L:** TOP, RAND-H and BP-H on both readings, plus seed-1 repeats of TOP and RAND-H on carry-over.
  * **Ceiling:** BP-H must beat RAND-H on carry-over by ≥ 0.06 bits. Otherwise lane L is CANNOT_TEST and no SSL code is written.
  * **Noise:** the seed-1 repeats must move carry-over bits by ≤ 0.01. Otherwise every probe is averaged over 3 seeds (one testbed redesign, recorded).
* **Lane C:** RAND-R, TBPTT-1 and BPTT on both readings.
  * **The reference learns:** BPTT must beat RAND-R on the same text by ≥ 0.05 bits. Otherwise REFERENCE_NOT_READY.
  * **Ceiling:** BPTT must beat TBPTT-1 on the same text by ≥ 0.03 bits. Otherwise lane C is CANNOT_TEST and COLA is not built.
* **Budget:** any chosen fit whose best epoch is the last is reported as a lower bound.

**C1 technical:**
* **The local rule:**
  * the autograd update equals the explicit three-factor formula (float64, ≤ 1e-10);
  * no gradient from the probe reaches W in SSL-H;
  * runs on 50 lines are deterministic;
  * after one pass, the score h_{t+1}ᵀ B h_t exceeds h_nᵀ B h_t for ≥ 60% of A's calibration pairs (chance is 50%; this catches a broken rule);
  * the cost per pass is measured.
* **COLA, if lane C passed C0:**
  * with μ fixed at 0 its update equals TBPTT-1's exactly;
  * the teaching signal equals Eq. 23;
  * α and μ match Eqs. 25–30 on a hand-computed case;
  * the BPTT gradient equals autograd's;
  * runs are deterministic.

**C2 dev:**
* **Lane L passes if** SSL-H beats RAND-H on carry-over by ≥ 0.03 bits **and** ≥ 3 line-bootstrap SDs.
  * Also reported: φ = (RAND-H − SSL-H) / (RAND-H − BP-H), the share of back-propagation's gain that the local rule recovers.
  * The same-text reading and the linear probe are reported but do not gate.
* **Lane C passes if** COLA beats TBPTT-1 on the same text by ≥ half of (TBPTT-1 − BPTT) **and** ≥ 3 SDs. Carry-over is reported.
* **Diagnostics (non-gating):**
  * the cosine between the local update of W and the back-propagated next-byte gradient at the same W (Zihan et al.'s alignment measure);
  * dead units and the effective rank of H;
  * the distributions of α and μ, and how often μ sits at its cap.

## 5. Readings fixed in advance
**Lane L:**
* **CANNOT_TEST:** even back-propagation finds nothing in this slot that carries over to a new author at 10 KB. The slot is then closed for carry-over at this scale, whatever the rule. The next question is the amount of text A (the scale of pre-training) or the unit of prediction (words).
* **Pass, φ ≥ 0.5:** the local rule learns most of what back-propagation learns here. Next: replace the back-propagated top with local rules and place the layer natively (KC → H → output).
* **Pass, φ < 0.5:** partial. Next: Zihan et al.'s direct feedback, where the rule's reference signal comes from the top layer. It is their targeted refinement.
* **Fail with the ceiling met:** next-step contrastive learning gives no transferable features at this scale, so the useful signal is the next-byte error itself. The next local candidate is a rule driven by a broadcast output error (direct-feedback type), not self-supervision.
* **In every case,** TOP against R1 on the same text separately answers whether a back-propagated top on the frozen code closes the 0.26-bit gap to T3.

**Lane C:**
* **CANNOT_TEST:** at 40 KB with this input there is no headroom for credit beyond one step, as in COLA's own PTB-Char result. Credit across time is not a current bottleneck; COLA waits for a task that needs a multi-step state, such as addition.
* **Pass:** online local credit across time works on text, and COLA is kept for multi-step tasks.
* **Fail with the ceiling met:** COLA's scalar closure misses what back-propagation through time gains here.

## 6. Cost, caps and predictions
**Measured cost** (random tensors, no text, one thread):
* one epoch of a 512-unit layer with a top: 7.4 s;
* one CLAPP pass: 13 s;
* one BPTT epoch with 256 units: 2.7 s;
* T3's fit: 138 s.

**Estimated CPU:** C0 about 1.1 h, C1 about 0.25 h, C2 about 0.4 h, about 1.8 h in total. My estimates have run ten times low before (V155's local rule), so the request leaves room.

**Learner records:**
* SSL-H, 8 passes: 5,640;
* COLA, at most 15 epochs: 10,575;
* C1: about 1,500;
* total: about 17,700.

**Caps requested:** 3 CPU-h, 20,000 learner records, 3 test cycles. These are inside the self-checked tier. Within the default caps (2 CPU-h, 5,000 records), lane L runs with at most 4 SSL passes, lane C runs C0 only, and COLA is not built in this delivery.

**Predictions (my record is poor):**
* lane L's C0 ceiling is met: 0.6;
* TOP reaches ≤ 2.75 on the same text: 0.5;
* SSL-H passes, if C0 is met: 0.3;
* lane C's C0 ceiling is met: 0.3;
* COLA passes, if its C0 is met: 0.25.

## 7. What this cannot show
* **Scale:** one text pair and one seed per fit; DEV_NOT_EVIDENCE. Both papers used far more data than 40 KB, so a failure holds at this scale only.
* **Learning rule:** the top uses back-propagation. A pass does not make a learner free of back-propagation.
* **Anatomy:** the 512-unit dense layer is not fly anatomy. The nearest analogue is plastic KC→output synapses under a dopamine-like third factor.
* **COLA:** with AdamW, COLA is not the paper's SGD setting.
* **Scope:** native integration, the output port, even fan-out and arithmetic are left out.
* **Review:** if ChatGPT's review changes the basis, this design is revised before lock.
