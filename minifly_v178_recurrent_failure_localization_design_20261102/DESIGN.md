# MiniFly V178 — Recurrent Reference Reproduction and Failure Localization

**Identity:** V178-RRFL-D1  
**Date:** 2026-11-02  
**Status:** DESIGN_ONLY. Neither implementation nor any training or evaluation is authorized by publishing this document.  
**Review anchor:** digitalbaby at `207c2aab30f3196a63e7359808a4dfbcfa599772`; separate Claude source at `f4146856c3241d94f7fa948e0b7d30fa0b869bc1`  
**Research objective:** Identify the first controlled change between a historically successful BPTT reference (V164) and the unsuccessful V177 recurrent reference that removes reliable delayed-dependency learning at a *matched training budget*. Do not invent a new neural mechanism.

## 0. Executive contract

The last three Codex experiments failed to establish cumulative progress: V175 coactivity-based inhibition barely affected interference; V176's locally learned features were usable by fitted probes but not the native behavior; V177's *own BPTT positive reference* did not qualify, leaving the BS/BR local-credit hypotheses untested. V177 held accuracy was 21.13% and 14.54%; its underlying stochastic generator had enough training information for a task-specific lag estimator to recover the dependence. **The absence of a qualified positive reference is the first question to repair.**

V164 had already reported a 256-unit dense tanh RNN on the eight-class lag-eight S-COPY benchmark: **1.151632 test bits/symbol under BPTT**, 2.472613 under TBPTT-1, with validation curve ending at **1.14564**. V164's S-COPY source/setting snapshots—not a recollection or approximate restatement—are the control anchor. Its architecture, optimizer and benchmark are distinct from V177's 128-unit connected gated learner with 257 outputs, random byte projections, a stochastic input process and genuine EOS. V178 will **not** conflate those changes.

**Proposed ordered path:**

- **A: reproduce the literal V164 BPTT positive reference** on the exact historical opened S-COPY generator, with numerical and source equivalence audit;
- **B: change only the recurrent connectivity package** from dense to a frozen sparse modular mask, while holding width 256, tanh dynamics, input, eight-class output, task, optimizer and dose fixed;
- **C: add a trainable recurrent gate** to the *same sparse 256-unit graph*, with the same input/output/task/optimizer/dose; the gate change is a declared architecture package with additional parameters, not a clean one-parameter causal intervention;
- **Guard: only if A/B/C all qualify at the original dose, run A and C once on a fresh independent S-COPY world without choosing or retuning anything on that world.**

At the **first B or C failure**, no downstream stage runs. A predefined ten-additional-epoch diagnostic of that first failing stage is permitted if finite and resource-admitted, **but it does not convert the original ten-epoch gate to PASS, and downstream stages still do not run**. This is a budget-sensitivity diagnostic, not outcome-driven selection.

**Explicit deferrals:** no 128-unit downsizing, 257-class output, raw-byte remapping, noisy next-byte stream, newline/EOS curriculum, BS/BR, COLA/e-prop, mushroom-body Full151 integration, local consolidation, spiking neural networks, attention or Transformer in V178. Each would introduce an additional confound. If the sparse gated 256-unit reference survives, changes of width/vocabulary/task must be separately specified and tested in a new version.

**Evidence tier:** E0 fidelity/source and a bounded BPTT architecture-feasibility diagnosis, not local learner success, E2 persistent retention, E3 arithmetic or confirmation of a general cognitive architecture. A reference-only model that predicts S-COPY's delayed target through its own trained eight-way output is still a positive reference, *not* a deployable MiniFly neural kernel.

## 1. Immutable historical V164 reference

### 1.1 Exact source and task anchors

Use original `minifly_v164_cola_dev_v1_20261026/v164_lib.py` and `v164_testbed.py` on the Claude branch, the referenced `minifly_v163_sslcola_dev_v1_20261026/v163_lib.py`, their `SETTINGS.json` files and the imported `minifly_v149_input_ceiling_dev_20261025/readouts.py`. The original `LOCK.json` gives source/dependency hashes and historical Python/PyTorch metadata. Reuse source rather than paraphrasing its semantics; verify actual files against the pinned historical hashes. The historical *text* `codes.npz` cache is not needed to construct the synthetic S-COPY benchmark, but if an import or runtime contract truly depends on it, resolve that dependency or stop rather than replacing it.

The exact S-COPY dataset has 8 discrete symbols `{0,…,7}`; each of 100 training, 20 calibration and 50 test **independent** lines has length 120. All inputs `X_t` are iid uniform. Labels at positions `t=0,…,7` are unscored; at `t=8,…,119` the desired class is `Y_t=X_{t-8}`. This is a **delayed-copy supervisory target**, *not* an autoregressive next-byte prediction. Training loss and all reported bits/token apply only to the 112 scored positions; no EOS or newline exists. Train/validation/test original generator seeds are 1/2/3, with V164's source-defined task-offset multiplication and exact ordering rules. Save generated train/val/test SHA256 digests **before** training; never choose or replace records after looking at outcomes.

For this synthetic fixture, V163's `RNN` executes

\[
h_t=\tanh(W_{\rm in}e_{X_t}+W_{\rm rr}h_{t-1}+b),\qquad
z_t=O h_t+d,\qquad p_t=\operatorname{softmax}_{8}(z_t),
\]

where `e_{X_t}` is the one-active-index input in the original `EmbeddingBag`/input-weight representation. `h_{-1}=0` at each line. `H=256`. The original uses a random dense recurrent matrix divided by its spectral norm, then a gain `g` calibrated on *training-only* trajectories; the published S-COPY result used **g=3.1475108458573424**, train-driven Lyapunov exponent approximately −0.114. Preserve the V164 initialization RNG, original active input scaling, padding semantics, trainable bias and output-head initialization. Do not replace the full indexed input by a newly seeded eight-vector embedding and still call the result a literal reproduction.

Optimizer and schedule: full BPTT, PyTorch AdamW lr `0.001`, weight decay `0.0001`, global parameter-gradient norm clip 5, batch 16 complete lines, seed-13 per-epoch shuffled line order, at most 10 epochs, patience 3 with best calibration checkpoint retained in historical code. The published successful run selected **epoch 10**, so the controlled bridge will likewise execute exactly ten epochs, record final and best-calibration checkpoints separately, and report any divergence from original early-stop behavior. The historical source remains frozen; any compatibility adapter must preserve its original predictions and gradients before being used downstream.

The V164 numeric result is not a guarantee across a different machine, Torch version or BLAS implementation. Record environment and numeric tolerances rather than calling a functional recovery bitwise identity.

### 1.2 Source-fidelity gate A

A must pass **all** of:

1. Imported files match recorded source hashes (or an explicit, limited runtime-compatibility adapter has separately proven forward/gradient parity). Model data/input shapes, masking, record resetting and target shift are verified against independent minimal examples. A reimplementation that produces a similar test score without source-level qualification **does not reproduce V164**.
2. In a pinned deterministic float64 technical fixture: an independent reference computes every forward hidden state and full eight-class logits for a short line, and matches literal original source to max abs error ≤1e−9 (same backend/dtype); automatic and finite-difference gradients agree relative ≤1e−6 (absolute ≤1e−8 near zero), including a delayed recurrent weight.
3. For the original fixed generator and ten-epoch dose, final calibration S-COPY loss ≤**1.35 bits/scored position** and historic-test score ≤**1.35**; report comparison against published **1.14564 calibration / 1.151632 test**, the full epoch curve, train/val/test same-metric losses and exact source/runtime differences. A literal numeric-reproduction label requires additionally |test−1.151632|≤0.05; functional reproduction without that tighter agreement is separately labeled, not passed off as bitwise replay.
4. Record optimizer state, epoch-10 parameters and all reference scores before starting B. No change to rate, gain, seed, train lines, task, loss or max epochs based on the opened historical test result.

If A fails, stop `SOURCE_MISMATCH`, `NUMERIC_PARITY_FAILED`, or `REFERENCE_NOT_REPRODUCED` as appropriate. A failure on an unsupported environment is an **operational reproducibility limitation**, not a theorem about recurrent-learning capacity. Do not invoke V177 for an unrelated task as a fallback baseline.

## 2. Controlled architecture ladder

All bridge stages use **the identical historical S-COPY training/calibration/test lines**, eight-class output, BPTT supervision and the same fixed **10×100 =1,000** full training-record applications. No fitted evaluator readout replaces each network's trained eight-way output; no task labels, lag values, position index or generator oracle are supplied as neural inputs. The historical test was already opened, so A/B/C are *engineering/development* comparisons, not fresh confirmation. Each stage starts from a fresh, matched initialized model; do not hand B's learned checkpoint to C.

### 2.1 B — sparse topology package, not a new learning rule

Keep the **same 256 tanh neurons**, `W_{\rm in}`, output head, bias, training objective, training schedule and optimization hyperparameters. Replace the *dense* recurrent support only with a **32-module ×8-neuron sparse graph**, a width-matched extension of V177's 8-unit module pattern.

For destination neuron `j=(m,k)`, `m∈\{0,…,31\}` and `k∈\{0,…,7\}`, use:

- every neuron in the same module: 8 input edges, including self;
- for each offset `o∈\{-2,-1,+1,+2\}` modulo 32, source neurons with within-module indices `(k+o) mod8` and `(k+o+3) mod8`: 8 additional directed edges.

Each destination has exactly 16 incoming edges and the strongly connected graph has `256×16=4,096` trainable directed recurrent entries (vs 65,536 dense). Enforce this fixed support during training, including mask-aware gradients, AdamW moments/decay and checkpoint restoration; no previously masked connection may grow.

**Initialization control:** start from the **same original unnormalized dense V164 recurrent birth** `\bar W`. Define

\[
W_{\rm rr}^{B}
=
g\,\frac{M\odot\bar W}{\|M\odot\bar W\|_2},
\qquad
W_{\rm rr}^{A}=g\,\frac{\bar W}{\|\bar W\|_2}.
\]

With `g=3.1475108458573424`, both matrices have the same initial spectral operator norm. Hold the input projection and output-head initial values identical to A where possible, and hash the exact matched arrays. Zero/NaN masked matrix or failed strong-connectivity fixture stops technical qualification.

This is a **sparsity plus operator-norm stabilization package**; it cannot uniquely attribute a negative result to topology rather than initial-state Jacobian conditioning. Report the train-only unlabelled largest Lyapunov exponent, product-of-Jacobians singular-value profile and eight-step hidden-state sensitivity **without retuning `g`**.

### 2.2 C — gated transition package on the same sparse 256-unit graph

Keep B's graph, width, input support, eight-way output, same data, target and BPTT optimizer. Replace only the tanh transition with

\[
a_t=\tanh(W_{\rm in}e_{X_t}+W_{\rm rr}^{B}h_{t-1}+b),
\]
\[
q_t=V_{\rm in}e_{X_t}+S_{\rm rr}h_{t-1}+c,\quad
r_t=\sigma(q_t),\quad
h_t=(1-r_t)\odot h_{t-1}+r_t\odot a_t.
\]

Gate `S_{\rm rr}` has exactly B's sparse support. Initialize `V_{\rm in}=0`, `S_{\rm rr}=0` and `c=0`, so the gate begins at `r=1/2`. The candidate tanh weights, input projection, output head and initial biases are copied **from the same B initial birth before any fitting**, not its learned state. All gate weights subsequently train under the same BPTT, batch, clip and LR. The old `W_{\rm rr}^B` support must remain frozen throughout.

Introducing a trainable gate necessarily adds parameters and changes the dynamical update; this is a **gating/parameterization package**, not isolated proof that multiplicative gates alone help or hurt. It matches *the type* of V177's recurrent update while deliberately **not** reducing width or changing output vocabulary. Record gate distributions, saturation, memory-state sensitivity and same-batch gradient propagation before drawing a mechanism conclusion.

**Important:** C is not a literal V177 reproduction. V177 used 128 units, a different fixed byte embedding and 257 outputs, with a different synthetic task and optimizer. Failure of C does not settle those other changes, and success of C does not establish the original V177 protocol was otherwise sound.

### 2.3 Fixed screening and first-failure procedure

For B and C, at the **final epoch-10 checkpoint** (no validation-best hindsight selection) require:

- calibration delayed-copy cross-entropy **≤1.50 bits per scored position**, compared with approximately 3 bits for an eight-class marginal model;
- final held test loss (opened development fixture) reported once after the calibration gate, with a separate declared **≤1.65 bits** *descriptive reproducibility guard*, not a hyperparameter-selection signal;
- no NaNs, source/target mismatch, hidden extra position features, gradient/support violation, or resource failure.

If the calibration criterion fails at epoch 10:

1. Mark the original stage `FAIL_AT_MATCHED_DOSE` and **stop the architecture ladder**.
2. If technical and resource checks allow, resume the **same failing stage** from its exact epoch-10 checkpoint and AdamW state for **ten additional fixed epochs**, with identical data/order continuation, no rate or gain change. This **predeclared** diagnostic does not make the matched-dose comparison pass.
3. Report `BUDGET_SENSITIVE` only if calibration reaches ≤1.50 at epoch 20; otherwise `UNRESOLVED_OR_ARCHITECTURE_LIMITED_AT_20`. Do not label lack of success an impossibility theorem. No subsequent architecture stages or fresh guard run after a matched-dose failure.

If B passes, proceed to C. If C passes as well, proceed to the held-out independent guard. If either stage's held descriptive guard fails but calibration passes, **do not promote**; report a generalization warning, stop the ladder, and do not run a new grid.

The 10→20 diagnostic is part of the original design, not an authorization to add more epochs until a preferred result appears. It is impossible to infer a single causal culprit merely from a disconnected stage pair; the interpretation must say *which bundled transition was tested*.

## 3. Fresh-world guard, run only if A/B/C qualify at ten epochs

**Reservation:** `202611027811`, checked against every accessible branch/namespace before execution. Use an independent SHA256-derived PCG64 generator for 100/20/50 S-COPY lines of length 120 with uniformly sampled eight-class input symbols, and exactly the same `Y_t=X_{t-8}` scoring rule, 8-class output and 10-epoch schedule. Freeze the guard generator, immutable dataset hashes, initial weights, and source/optimizer code **before** opening any guard scores. Do not see its held targets in engineering cost probes.

Train **A dense tanh reference and C sparse gated candidate** from fresh, independently seeded but within-world matched initial weights (same common input/output births and candidate paths). B is not retrained on the guard; the guard tests the endpoint C relative to its already-qualified dense reference and is not intended to localize B versus C again. A/B/C design choices are frozen on the historical-world engineering data; **the guard world must not choose the gain, graph, model width, gate bias, optimizer or dose**.

**Guard interpretation gates**, on untouched final checkpoints:

- A calibration ≤1.50, C calibration ≤1.50 bits/scored position;
- A held ≤1.60 and C held ≤1.60;
- C held − A held ≤0.35 bits;
- no hidden evaluator probe in model output and no source/data leakage.

A guard failure is `NO_REPLICATED_REFERENCE_FEASIBILITY`, even when C did well on the historical opened fixture. Passing is `REFERENCE_BRIDGE_FEASIBLE_DEV`: a narrow two-world engineering witness that the **256-unit sparse gated architecture** can learn S-COPY with BPTT. It is **not** independent statistical confirmation, a demonstration of local learning or persistence, or evidence that the 128-unit/257-output V177 system would have worked.

## 4. Diagnostics required even when everything fails

### 4.1 Same-metric train / calibration / held curves

At every completed epoch, evaluate **the current checkpoint** with all stochastic optimization disabled on the exact same `t≥8` target subset:

\[
\ell_{\rm train}(e),\ \ell_{\rm calib}(e),\ \ell_{\rm test}(e).
\]

To reduce sequential test exposure, only train and calibration are opened every epoch; the **test is evaluated once at the final locked decision checkpoint**. Any training-loss value averaged *during* updates must be separately labeled `online_train_loss` and never compared directly to a final-checkpoint held metric as an overfitting certificate. Save curve values for calibration and training, plus best-calibration epoch descriptive metadata; score final model, not a post hoc best-test model.

Interpretations:
- both train and calibration near ~3 bits: insufficient learning or optimization/representation failure at the given dose;
- train approaches ~0 bits while calibration stays near ~3: strong memorization/overfit evidence;
- both improve but test degrades: distribution/selection dependence, not necessarily a defective forward graph;
- B or C training collapses while A trains: transition-package bottleneck at the fixed protocol.

No claim that architecture is inherently incapable follows just from these regimes.

### 4.2 State sufficiency / readout reachability

For every qualified or failed stage, use frozen training checkpoints to fit a **separate evaluator-only** train-only ridge/softmax linear probe `P(X_{t-8}\mid h_t)`, with a single predeclared regularization coefficient `1e−2` and frozen input/state normalization, and score untouched calibration, then test only at final diagnosis. Charge probe sample exposure/fit CPU. This probe is **not the model's own output**, not evidence that its native head could have learned the same readout, and must never select a production winner.

Decompose:
- low probe loss + high own-head loss: delay information is accessible but the native trained readout/optimization is deficient;
- high probe loss + high own-head loss: delay information is missing from an easily decoded feature; nonlinear recovery remains possible and cannot be ruled out;
- both low: supports successful ordinary learned prediction on this bounded task.

### 4.3 Temporal and numerical mechanism checks

For saved *read-only* checkpoints, measure the eight-step state sensitivity to the relevant input using exact Jacobian products:

\[
J_t=\partial h_t/\partial h_{t-1},\qquad
\frac{\partial h_t}{\partial e_{X_{t-8}}}
=J_t\cdots J_{t-7}\,\frac{\partial h_{t-8}}{\partial e_{X_{t-8}}},
\]

with indexing checked against the actual forward semantics. The derivative is for a continuous input-embedding perturbation, not a derivative with respect to a discrete token identity. Report per-world norms/singular-value summaries, saturation of tanh/gates, state variance and gradient decay. **High sensitivity does not prove learned delayed-copy accuracy; low sensitivity only supports a conditioning hypothesis.**

Repeat read-only state perturbations: recurrence-off, sparse-cross-edge-off, and output-head-only controls **where meaningful**, explicitly flagging distribution-shift lesions. The untrained dense random-recurrence readout (train-only head) is a useful reference control, but its fitted head must not be credited to the candidate. No expensive external probe or oracle is a substitute for complete candidate outputs.

### 4.4 Reproduce the code and the resource ledger

Check input/target shift and exact scored positions; row-major vs column-major weights; no optimizer moment for masked edges; no restoring a best calibration state before extra exposure; all gates see the same data/order; no seed reuse across guard; exact full-batch evaluation; training and total CPU. Publish any failed assertion, absent dependency, incorrect gradient norm or phase timing rather than mutating it silently after scoring.

## 5. Resource schedule and hard stop

Treat *record applications* as every complete training line consumed for an optimizer update, including repeats. **A** 100×10=1,000; **B** 1,000; **C** 1,000; **fresh A+C guard** 2×1,000=2,000 if all first three pass; or **one conditional failing-stage extension** up to 1,000 if any matched-dose bridge fails. Guard and extension are **mutually exclusive**.

| Conditional route | Maximum record applications |
|---|---:|
| A+B+C all pass; guard A+C runs | 5,000 |
| First failure at B, one 10-epoch extension | 3,000 |
| First failure at C, one 10-epoch extension | 4,000 |
| A fails (no bridge) | 1,000 |
| Technical fixtures (all routes, separately charged) | ≤64 |

Global **hard learner record cap 5,100** (proposed as part of the design, not approved training authorization). With 112 scored targets per S-COPY line, scheduled main-and-guard exposures are at most **560,000 supervised target applications**, plus ≤7,168 technical scored target applications. Distinguish unique lines (100 per training world) from repeated exposures, plus 20 calibration and 50 test lines per world, probe fits and extra forward evaluations. Do not count validation/test as new training or fail to count repeated optimizer presentations.

Compute cap: **7,200 aggregate process-tree CPU seconds**, wall cap **14,400 seconds**, peak RSS **2 GiB**, output files **2 GiB**; one CPU process at a time. These are prospective budgets; actual environmental feasibility must be checked by technical profiling. A ×1.5 preflight upper forecast must cover the **maximum conditional route**, full training-state/sensitivity/probe/evaluation cost, all dependency imports, checkpointing and failure receipts. If unadmitted, `RESOURCE_NOT_ADMITTED` before substantive training; do not shorten the roster or hide optional work.

No official V178 execution, CPU spending, branching integration or adoption follows automatically from this design. An experiment run must separately authorize a precise technical implementation and source-lock the runner/data/optimizer, repository asset closure, Python/Torch/NumPy versions, RNG seeds, dataset hashes and expected outputs. Never reconstruct a missing V164 original implementation and label it reproduction.

## 6. Publication and audit contract

Before training, write `SOURCE_LOCK.json`, `ENVIRONMENT.json`, `DATA_MANIFEST.json`, `PARAMETER_BIRTHS.json`, `MASK_AND_GAIN_AUDIT.json`, `EXECUTION_SPEC.json` and `RESOURCE_ADMISSION.json`, and record an immutable first-run timestamp. After each stage, retain source-verified train/val/test per-line scores (test *only once* per completed stage), optimizer states and checkpoint hashes, source and numerical parity, hidden-state-probe diagnostics, recurrence spectra, failure evidence and global CPU/record accounting. The version registry should only mark a stage complete if the corresponding receipts exist.

A separate saved-artifact reducer should recompute every gate, target count, line score and train/calibration held contrast from saved receipts **without rerunning learning**. It is not automatically an independent external audit. Consistent with the project's no-raw publication policy, compact GitHub reports may omit raw model weights/line contents; then clearly mark stronger raw replication as **NOT_PUBLICLY_VERIFIED**, rather than inventing accessible data.

**Stop and verdicts:**

| Event | Label | Consequence |
|---|---|---|
| Missing historical source/data or incompatible runtime | `SOURCE_UNAVAILABLE` / `REPRODUCTION_INVALID` | No B/C; no surrogate |
| A source/parity or functional gate fails | `REFERENCE_NOT_REPRODUCED` | Stop; no local-learning claim |
| B fails after ten epochs | `SPARSE_PACKAGE_FAIL_10` | Stop ladder; optional prespecified B extension only |
| C fails after ten epochs | `GATED_PACKAGE_FAIL_10` | Stop ladder; optional prespecified C extension only |
| B/C passes at 10, guard fails | `NO_REPLICATED_REFERENCE_FEASIBILITY` | Stop; no new core |
| All A/B/C and fresh guard pass | `REFERENCE_BRIDGE_FEASIBLE_DEV` | Design a separately approved width/vocabulary/next-byte study |
| Resource/protocol/sign/target error | `RESOURCE_NOT_ADMITTED` or `INVALID` | Stop, preserve failure |

Do **not** promote a candidate from one favorable run, a validation-best checkpoint without a frozen selection protocol, a fitted probe, a source-reproduced historical score, a changed threshold, a future-answer leak, or synthetic replay. V178 does not prove local learning or arithmetic transfer, so leave every E2/E3 and native Full151 adoption pointer unchanged.

## 7. Next scientific design, explicitly deferred

Only after a qualified V178 reference bridge should a later version isolate **width 256→128** (including the topology rescaling required for a smaller graph), then the **eight-way→257-way output expansion**, then the **raw byte representation**, then the **stochastic next-byte process**, and finally the **genuine newline/EOS boundary**. These transitions must not be run or selected on the V178 historical-world test, and any single comparison that necessarily changes both model width and topology must be labeled as such.

Then, and only then, can the BS/BR local temporal-credit rules be evaluated against the **actual successful** BPTT candidate, with independent fresh worlds, own outputs, and verified train-versus-held learning curves. Following a successful local learning result, test A→B→A corrections and retained useful knowledge in one continuing model. This avoids claiming that a fly-inspired associative lookup alone implements transferable computation.

**Version governance:** V178-RRFL-D1 is the next allocated design; V177 remains a frozen `REFERENCE_NOT_QUALIFIED` result. V178-R1 is reserved for documentation-only revisions with a diff; a materially different learning method or task requires a new experiment number, after checking all live branches. **Do not allocate V179 here.**

## 8. Exact sources

- [V164 original BPTT reference results, g and epoch curves](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v164_cola_dev_v1_20261026/results/testbed/TESTBED.json)
- [V164 original settings](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v164_cola_dev_v1_20261026/SETTINGS.json)
- [V164 original synthetic generator and training loop](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v164_cola_dev_v1_20261026/v164_lib.py)
- [V163 recurrent network's original dynamics](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v163_sslcola_dev_v1_20261026/v163_lib.py)
- [V164 historical source and environment hash lock](https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v164_cola_dev_v1_20261026/LOCK.json)
- [V177 stopped BPTT reference result](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v177_connected_recurrent_credit_dev_20261102/REPORT.md)
- [V177 actual model/data/optimizer source](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v177_connected_recurrent_credit_dev_20261102/core.py)
- [V175 design self-hostile audit](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_V175_SELF_HOSTILE_AUDIT_20261101.md)
- [Project no-attention kernel constraint](https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md)
