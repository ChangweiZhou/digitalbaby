MEMO (discussion record; no experiment, no claim promotion). Owner-requested summary of the discussion after V163–V168.

**ERRATUM (27 October 2026, same day):** section 4 originally reported counting baselines that were too high by exactly 1 bit/symbol. The exploratory script summed the pair (total bits, symbol count) returned by the ladder before dividing by the symbol count. With the error corrected, **counting pre-trained on the first text beats every learner in the low-data transfer regime, including the back-propagated layer, by about 0.4 bits**, and counting fitted on the new author's 10 KB alone ties the best learner. The original claim that the circuit's advantage lies in data efficiency is withdrawn. Sections 4 and 7 have been corrected. The erroneous output is kept as `explore/ppm_carry_output_ERRONEOUS_plus1bit.txt`.

# Credit assignment, order and horizons in MiniFly: where the programme stands (27 October 2026)

## 1. Purpose and scope
This memo consolidates the discussion that followed experiments V163–V168. It covers:
1. the problem the programme now faces;
2. what the evidence establishes;
3. how sequence order and temporal horizon are represented;
4. the plan for the next two experiments, V169 and V170.

All results cited are development results (DEV_NOT_EVIDENCE) unless stated otherwise. Section 4 reports one exploratory measurement, made outside any delivery (`explore/`).

## 2. Problem statement
The programme asks whether a mushroom-body circuit can learn, from small amounts of raw text and with biologically plausible local learning rules, two things:
* the internal representations that language prediction requires;
* the internal multi-step computation that tasks such as unseen multi-digit addition require (criterion E3 of the objective contract).

The experiments since V155 indicate that one problem underlies both: **credit assignment under locality constraints.** A synapse can use only locally available quantities: presynaptic activity, its own postsynaptic state, and broadcast signals. The question is whether those quantities determine a weight change that serves the task. The working criterion is the alignment ⟨g_local, ∇J⟩ between the local update and the gradient of the task objective.

The problem has two components:
* **Spatial credit:** which units of a hidden layer should change, given an error observed at the output at the same time step.
* **Temporal credit:** which earlier states or queries should change, given an error that appears only several steps later.

## 3. What the evidence establishes
### 3.1 The fixed expansion and its limits
* The random PN→KC expansion is a similarity-preserving hash. With a linear readout it predicts the training text at about counting level: 2.96 bits/symbol, against 2.61 for PPM-4 at 40 KB (headroom check, V153).
* Learning the PN→KC projection on MiniFly's native sparse wiring offers almost no headroom, even under back-propagation: 0.03 bits (V155).
* Learned capacity must therefore sit above the KC layer.

### 3.2 Spatial credit is resolved for one feedforward layer
Carry-over to a new author (*Thursday* → *Edgeworth*, 10 KB of the new text), for a 512-unit layer above the frozen KC code:

| Learning signal for the layer | Alignment with the task gradient (cosine) | Gain recovered, relative to back-propagation |
|---|---:|---:|
| Contrastive self-supervision (V163) | 0.02 | 31% |
| Next-byte error broadcast through fixed random feedback, back-propagated top (V165) | 0.65 | 86% |
| The same, with no back-propagation anywhere (V167) | 0.74 | 85% |

The decisive variable is whether the local signal carries task-relevant error. Depth and capacity were not limiting.

### 3.3 A sufficient architectural extension for composition
When retrieved memory content re-enters the working state, a recurrent network trained by back-propagation through time solves a two-hop chained query: 0.009 bits, against 3.09 for the matched network without re-entry (V166).

### 3.4 Temporal credit is not resolved
* **The credit the loop needs:** training the V166 loop requires credit propagated about three steps back, through cross-unit paths: from the answer, through the memory read, to the earlier query. Truncation at three steps recovers 97% of the gain from full back-propagation through time; truncation at two steps recovers 4.5% (V168).
* **Diagonal eligibility traces** (e-prop, RFLO) cannot represent this path. For the query weights, the e-prop update is identical to one-step truncation (V168).
* **An unbiased approximation of real-time recurrent learning** (KF-RTRL; Mujika et al., 2018) is too noisy at this data budget. It recovers 3% of the gap (V168).
* **COLA** (Wang et al., 2026) fails for a related reason. Its teaching signal is a rescaled immediate error, so it cannot assign credit to a step that produces no error of its own (V164).

## 4. Evaluation frame: counting in the low-data transfer regime (corrected)
The programme assessed progress against counting on the training text at 40 KB. The counting baselines for carry-over were specified in V155 but never computed. An exploratory computation, made outside any delivery (`explore/ppm_carry.py`, corrected; about 1.4 CPU-s), gives the following, all scored on the same held-out lines:

| Bits/symbol on B's held-out lines | *Thursday* → *Edgeworth* | *Emma* → *The Ball and the Cross* |
|---|---:|---:|
| PPM-4 fitted on B's 10 KB only | 3.07 | 3.17 |
| **PPM-4 fitted on A's 40 KB, then B's 10 KB** | **2.68** | **2.95** |
| Frozen KC code, linear readout fitted on B (V162) | 3.32 | – |
| Frozen KC code, back-propagated readout fitted on B (V163, V167 C0) | 3.33 | 3.37 |
| Layer learned with no back-propagation anywhere (V167) | 3.13 | not run |
| Layer learned by back-propagation (V163, V167 C0) | 3.07 | 3.17 |

*The Ball and the Cross* is cut at its Project Gutenberg end marker (V167 amendment 1).

**Interpretation:** counting wins in this regime as well.
* Counting pre-trained on the first text beats the best learner by about 0.4 bits.
* Counting fitted on the new author's 10 KB alone ties it.

**The comparison is not like for like in one respect, which matters for the plan.** Counting carries its entire predictor from A to B: the exact n-gram statistics of English transfer across authors. The learners carry only a representation from A, and refit their readout on B's 10 KB alone. A learner that is trained on A and then **continues** learning on B, which is what a persistent learner does, is the like-for-like comparison. It has not yet been made. The headroom check found back-propagated learners within about 0.1 bits of counting when they see all the data (2.70 against 2.61 on the same text), so the gap under a matched protocol may be much smaller than 0.4 bits.

## 5. Representation of order
### 5.1 Current mechanism
Since the CLS round (17 October), MiniFly has used an order-keeping front end (FE-PERM; cell P81 of V153). Each byte activates a fixed pattern r(x) on 6 of 88 receptor channels. Before each new pattern is added, the existing trace is passed through a fixed channel permutation P and attenuated by f = 0.807:

  s_t = r(x_t) + f·P r(x_{t−1}) + f²·P² r(x_{t−2}) + …

Each past symbol is therefore tagged by its lag. For example:
* "5-3" yields r(3) + f·P r(−) + f²·P² r(5);
* "3-5" yields r(5) + f·P r(−) + f²·P² r(3).

The two traces share only the operator term.

**Evidence:**
* Reordered prompts share 31% of their KC code with the permutation, against 69% without it.
* Taught order-dependent pairs (`9-2=` against `2-9=`) are kept apart 100% of the time, against 75% without it (CLS round).
* The permutation improves text prediction by 0.28–0.36 bits/symbol (V153).

### 5.2 Limitations
The code supports **discrimination** of orders but not **generalisation** of order-dependent operations, for three reasons:
1. **Short horizon** (section 6).
2. **Entangled identity and position.** A symbol at lag 2 and the same symbol at lag 3 occupy unrelated channels, so knowledge does not transfer across positions.
3. **Implicit roles.** "First operand" and "second operand" exist only as lags relative to the operator. The lags vary with operand length and with the progress of output generation, which is precisely what place-value alignment requires.

The earlier diagnosis of addition is consistent with this: `a+b` and `b+a` received the same answer only about half the time.

### 5.3 Relation to temporal credit
The standard route to an order representation that is both discriminative and generalisable is a **recurrent working state**:
* symbols are read one at a time;
* the update rule is shared across steps, which gives position invariance by construction;
* role and progress are carried in the state.

Training such a state requires temporal credit. **The order problem and the temporal-credit problem are therefore the same problem viewed from two sides.**

## 6. Two horizons
| Horizon | Meaning | Status |
|---|---|---|
| Context horizon (within a sequence) | How far back the current state can depend on earlier symbols | Not solved for text; solved only in a synthetic proof of concept |
| Persistence horizon (across episodes) | Whether what was learned survives rest and later learning | Addressed by the native fast memory for taught associations (E1 and E2); untested for the recent learned components; E3 not achieved |

### 6.1 Context horizon
* **The trace's limit.** The trace retains most, but not all, of the last four symbols (headroom check). The learned layers of V163–V167 read only its KC code and inherit this limit.
* **A mechanism without that limit.** The fast KC → MBON memory is content-addressed, so associations written earlier in a line remain retrievable at any lag (V166). This was demonstrated only with back-propagation through time, with hand-gated writes, a memory cleared at the start of each line, and synthetic tasks. Local training failed (V168).

**Why a slower fade does not help.** The attenuation 0.807 = e^(−Δ/τ) is inherited from the native receptor time constant: τ = 10 s, with a byte every Δ = 30/14 s. It was retained because it predicts better than 0.95: 2.963 against 3.833 bits/symbol (V153). The trace is a fixed 88-dimensional superposition:

| Fade f | Symbols effectively superposed, about 1/(1 − f) | Newest symbol's share of trace energy, 1 − f² |
|---|---:|---:|
| 0.807 | 5 | 35% |
| 0.95 | 20 | 10% |
| 0.997 | 333 | 0.6% |

* The total memory capacity of a linear dynamical system is bounded by its dimension (Jaeger, 2002). The fade redistributes a fixed capacity across lags; it does not increase it.
* The permutation also has order 120 (cycle lengths 2, 4, 12, 30 and 40). Positions 120 steps apart are encoded identically, and channels on the short cycles alias every 2 or 4 steps.

A slow fade therefore trades resolution for reach. Exact long-range information must be **written to and retrieved from the associative memory**, whose capacity scales with the KC and MBON populations rather than with the 88-channel trace.

### 6.2 Persistence horizon
* **Native fast memory:**
  * Taught facts are recalled at 84–88%, and at 81–85% after a day of rest.
  * With content-addressed keys, 99% of old facts survive four later cohorts of new learning and a day.
  * Forgetting is caused by interference from new learning, not by decay.
  * Replay protected an old skill during new learning: 70–85% of addition kept, against 4–6% without replay (CLS round; Byte-Full151 bridge).
* **The learned layer of V165–V167** was trained offline over several passes and then frozen. Its retention under continued learning, and its need for replay, have not been measured. Because it is a slow distributed learner, interference is the expected failure.

## 7. Plan
Two experiments, run in parallel under the research-loop skill, each with pre-registered criteria, matched controls and qualifying references.

**V169: replicate the local learner and compare a persistent learner with counting under a matched protocol.** On the second text pair (*Emma* → *The Ball and the Cross*):
* **Replication:** replicate the fully local learner of V167.
* **Matched comparison:** train the local network on the first text, continue learning on the second, and compare its held-out bits on both texts with counting fitted on the first text and then the second. Counting baselines are interpolated Kneser–Ney and a tuned PPM, with hyperparameters chosen on calibration lines.
* **Persistence:** measure retention of the first text with and without replay.

**V170: temporal credit for the memory loop, with order generalisation tested explicitly.** Two strategies, in this order:
1. **Architectural.** The retrieved item is re-encoded through the same fixed expansion that encodes input and serves as the next query, so that only one-step decisions (whether to re-query, how to read out) remain to be learned.
2. **Learning rule.** A three-factor rule: eligibility tags (pre × post) decaying over about three steps, combined with a delayed error broadcast through fixed random feedback. This is the temporal analogue of the rule validated in V165–V167.

In each strategy the task includes held-out order-dependent operand pairs, so that generalisation of order is measured directly.

**Stopping rules:**
* A positive result on V170, together with V169, would justify returning to native own-byte addition (E3).
* A negative result for both V170 strategies would indicate that multi-step computation under these locality constraints requires a mechanism not yet identified. That is a substantive finding, not a reason for further tuning.
* No further work is planned on COLA, on KF-RTRL variants, or on scaling studies.

## 8. Open questions
* **Arithmetic:** whether gains measured in next-symbol bits translate into arithmetic competence is untested.
* **Structure supplied by hand:** whether a fixed re-entrant pathway counts as generic architecture or as supplied structure must be judged by whether it is task-independent and whether learned components determine behaviour.
* **Where the signed error comes from:** the broadcast-error rule requires a signed, position-specific error signal at each unit. Its biological carrier, plausibly dopaminergic teaching signals, is not modelled.

## References
* Bellec, G. et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop). *Nature Communications*.
* Jaeger, H. (2002). Short term memory in echo state networks. GMD Report 152.
* Lillicrap, T. et al. (2016). Random synaptic feedback weights support error backpropagation for deep learning. *Nature Communications*.
* Mujika, A., Meier, F. and Steger, A. (2018). Approximating real-time recurrent learning with random Kronecker factors. *NeurIPS*.
* Murray, J. M. (2019). Local online learning in recurrent networks with random feedback (RFLO). *eLife*.
* Nøkland, A. (2016). Direct feedback alignment provides learning in deep neural networks. *NeurIPS*.
* Wang, W., Gao, K. and Chen, G. (2026). Global credit assignment via dynamical criticality (COLA). *ICML*.
* Zihan, W. S., Delrocq, A., Gerstner, W. and Bellec, G. (2026). Can local learning match self-supervised backpropagation? *ICML*.

Repository sources: the V153 and V155 headroom-check reports; V163–V168 designs and reports; `minifly_cls_dev_20261017/REPORT.md`; `addition_difficulty_diagnosis_dev_20261017/MEMO.md`; `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md`.
