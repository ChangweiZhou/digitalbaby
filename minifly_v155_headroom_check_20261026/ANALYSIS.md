V155 HEADROOM CHECK: ANALYSIS (development text; companion to REPORT.md)

# Headroom of a learned representation, and its consequences for V155 (26 October 2026)

## 1. Motivation
MiniFly encodes each position in a text as a sparse binary Kenyon-cell (KC) code:
1. A fixed front end turns the byte history into a decaying trace over 88 receptor channels.
2. The trace is mapped onto 302 projection neurons.
3. A fixed random matrix connects those neurons to 5,177 KCs, and only the 92 most strongly driven KCs on each side stay active.

In MiniFly as it stands, only the KC-to-output weights learn. Every readout it can implement is therefore a linear function of a representation fixed at initialisation.

V153 measured how well such a readout can do. A regularised multinomial logistic regression (R1) on the frozen code reaches 2.963 bits per symbol on held-out text. That is close to an order-2 PPM model (2.986), but clearly behind higher-order counting models (PPM-3 2.667, PPM-4 2.606). Since R1 is the best linear readout found, no change confined to the output side can do much better than 2.963. Any further improvement has to come from changing the representation, that is, from learning the input-to-KC weights (the keys).

The headroom check asked a prior question before any local learning rule is designed. Given the same receptor trace and the same amount of data (40 KB), how much better could a learned representation be than the frozen random one?

## 2. Method
All measurements use the development text (*The Man Who Was Thursday*) with V153's splits: 40 KB for training, 5 KB for calibration and 10 KB held out. Hyperparameters were chosen on the calibration set only. The decision thresholds were committed to the repository before any code was run (`PLAN.md`).

| Reading | What it does |
|---|---|
| T1 | linear readout (R1) of the raw 88-channel receptor trace |
| T2 | linear readout (R1) of the frozen random KC code; reproduces V153 |
| T3 | one hidden layer trained by backpropagation on the same trace, in place of the random keys |
| T4 | the same learner as T3, given the previous four symbols explicitly |

* **T3** is an evaluator-only upper reference. It is never part of MiniFly.
* **T4** is a positive control. It checks that the reference learner can exploit context at this data size, so that a weak T3 cannot be blamed on a defective learner.

## 3. Results (held-out bits per symbol)
| Reading | Held-out |
|---|---:|
| T1: linear readout of the receptor trace | 3.463 |
| T2: linear readout of the frozen random KC code | 2.963 |
| T3: learned representation of the trace | 2.699 |
| T4: learned model of the explicit 4-symbol context | 2.625 |
| PPM-4 (best counting model) | 2.606 |

* T3 improves on T2 by 0.263 bits (line-bootstrap SD 0.016).
* T4 differs from PPM-4 by +0.018 (SD 0.019), which is not significant.
* T2 and the counting baselines reproduce V153 exactly, and T4 passed its qualification threshold.

## 4. Interpretation
1. **The representation has substantial headroom.** A representation learned from the receptor trace improves on the frozen random code by about a quarter of a bit per symbol. The limitation therefore lies on the representational side, not in the readout.
2. **At 40 KB, counting models are close to the attainable optimum for this class of learner.**
   * Even when given the preceding four symbols explicitly, a backpropagation-trained network only matches PPM-4.
   * The reference's training loss (1.82 bits) is far below its held-out loss (2.70 bits), so it is limited by data rather than capacity.
   * Outperforming the best n-gram model on the same text is therefore not a realistic criterion at this data scale.
   * A learned representation is more likely to show its advantage in transfer: reusing structure learned on one corpus when data from another corpus is scarce, which counting cannot do.
3. **The random expansion is informative on text.**
   * A linear readout of the raw trace is 0.50 bits worse than one of the KC code. The nonlinear sparse expansion supplies conjunctions of recent symbols that a linear model of the trace cannot form.
   * Codex (V154) reported the opposite on arithmetic, so the expansion's contribution depends on the task.
4. **The trace keeps most of the recent context.** The learned model of the trace (T3) is only 0.07 bits behind the explicit four-symbol model (T4). The fading, superposed trace loses little; the larger gap is between the frozen random code and what the trace makes available.

## 5. Limitations
* **Wiring mismatch.**
  * The reference's hidden layer is densely connected.
  * MiniFly's keys are sparse: each KC receives input from about 5 of the 302 projection neurons, about 1.8% of possible connections.
  * The 0.26-bit headroom therefore applies to dense learned keys. How much of it is reachable under the native connectivity was not measured.
* **Grid edge.** The selected reference setting lies at the edge of its hyperparameter grid, so the estimated headroom is a lower bound.
* **Scope.** The results come from one development text, with one random seed per setting.

## 6. Consequences for the V155 design
The design is in `minifly_v155_keys_design_20261026/DESIGN.md`.

V155 asks whether a local, label-free plasticity rule can learn the keys. It compares three arms that differ only in the key matrix:
* the frozen keys;
* a competitive Hebbian rule, in which winning KCs move their weights towards the inputs that activate them;
* a predictive three-factor rule of the CLAPP type, in which a broadcast scalar signal gates Hebbian updates so that successive codes come to predict each other.

Both rules keep the native connectivity and use non-negative, normalised synaptic weights.

**First step: a feasibility gate.** Two backpropagation references, one restricted to the native connectivity and one dense, measure the headroom available under each wiring. If the native-connectivity reference finds less than 0.08 bits of headroom, the study stops before any local rule is evaluated.

**Evaluation criteria.** Because of finding 4.2, matching counting on the training corpus is no longer a criterion. V155 uses two:
* **Within-corpus capture:** the fraction of the native-connectivity headroom that each local rule recovers.
* **Transfer:** whether keys learned on one author improve a readout trained on only 10 KB of text by a different author.

**Safeguards specified in advance:**
* a synthetic testbed that the predictive rule must pass before any natural text is used;
* pre-registered outcome rules and predictions;
* separate development, rehearsal and held-out text pairs.

**Owner decisions required before implementation:**
1. Confirm the evaluation criteria, or alternatively allow more training text and keep outperforming counting as the criterion.
2. Resource limits for the test cycles.
3. Whether a later round may let the rules form new connections, if the native connectivity proves limiting.
4. Whether backpropagation references are acceptable as evaluation instruments.
