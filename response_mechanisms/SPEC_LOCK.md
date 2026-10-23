# Response-mechanism final specification

Prospective specification for the new, separate Full151 response-mechanism program.
Freeze together with executable code, tests, exact runtime and SOURCE_LOCK.json
only after cycle 3 pilot passes; do not open final worlds before independent launch
approval. No changes to A V3 science or existing receipts.

## Source and task
Canonical Full151 FE0 four-output stores from the immutable A V3 scaffold and
Mac-anchored PN→KC B. Four independent native births per system. Predictor learning
is frozen. Existing random balanced16-pair fact generator: row-major digits0..3
old,4..7 new,12-byte cues,30/14-second byte intervals,165s records. Six complete
randomized repetitions per cohort, with one86,400s rest after each cohort. Clock
checkpoints: birth0,old_end15,840,old_day102,240,new_end118,080,final204,480 seconds.
Probes clone each continuing state; no teacher, homeostasis or weight updates are
committed from probes.

Worlds 300001–300032 inclusive, exactly32; every world has all 7 arms. Pilot worlds
290000/290001/290002 and technical290099 are excluded. W teaches both cohorts;
N_old observes identical bytes/timing but old native and added-bank write flags
are literally false. New teaching is enabled in both. Unsupervised exposure-driven
T and H updates continue on both branches. Each original16×4 response tensor,
label mapping, training emission, write ledger and source/runtime record is saved.
This is taught random-pair learning/retention, not withheld-pair transfer.

## Arms and fixed parameters
All arms allocate the declared added arrays; their operative/writable capacity
is separately disclosed, and J versus FE0 is an architecture-package comparison.
- FE0: unchanged native FE0 value route
- T_OFF: identical added timing calculations with B writes disabled; exact FE0
  output parity required at all phases
- T: cue-byte antisymmetric PN/KC timing eligibility on canonical fixed support.
  Cue-only sensor never receives outcome/newline bytes; traces reset per record,
  tau10s. Sum12 cue-byte eligibility increments with epsilon.05 divided by12;
  positive relative weight floor1e-9; normalize each KC's fixed incoming mass.
  Apply after native teaching captures the original cue code. No labels/task IDs
  or value-memory state enter the timing sensor/encoder
- H: independent own-channel scalar EMA, beta1/16, of pre-teacher raw activity;
  output=raw−EMA. No direct label/task-ID/other-channel input; own activity can
  indirectly reflect past learning. No probe-based or label-based calibration
- J_ADD: additive recent/current receptor eligibility, same bank size/learning
  equation/gain/decay/bounds as J
- J: local recent×current receptor eligibility accumulated over cue bytes with
  recent tau10s and eligibility tau10s; reset each record. Four 88×88 float64
  output banks, born zero. Per-output normalized bank update eta.25 times signed
  local reward residual times eligibility/max(1,||eligibility||²), target+1 for
  rewarded channel and−1 otherwise. Clip each weight to±16, decay tau86,400s.
  Eligibility is local; sum-of-squares normalization is a declared shared
  postsynaptic resource, not a strictly independent synapse rule
- J_SHUFFLE: same J model but a target-independent per-record coordinate permutation
  at write; ordinary eligibility at read. Fixed SHA256 seed uses world+record.
  It matches architecture and rule, not realized residual-dependent update dose

J/J_ADD/J_SHUFFLE gain0.4237781016501581, fixed from cycle 1 FE0 W pre-teacher
channel-centered activity RMS, without labels/correctness. No other scaling,
threshold or candidate-selection fit is permitted. Resolved parameters are saved
in every receipt and checked against the lock.

## Measurements
For every complete grid V[a,b,c], B[c]=mean_ab V, F[a,c]=mean_b V−B,
G[b,c]=mean_a V−B, H=V−B−F−G. Bias RMS removes the channel-common mean of B.
Save all components, RMS values, reconstruction residuals and interaction/main
ratios; denominators≤1e-12 are explicitly undefined. Also decompose per-cue
channel-centered decision-space values and matched W−N_old tensors. For each
component, evaluator-only targets are2*one_hot(label)−1. Save component target
RMS, covariance, and covariance/RMS(target_component), with zero targets flagged.
Random target row/column effects are reported, not assumed zero.

Primary outcome: W final old-pair accuracy. Primary paired contrasts:
T−T_OFF,H−FE0,J−J_ADD,J−J_SHUFFLE. World is the independent unit. Two-sided paired
Student-t95% intervals and tests; Holm correction across these4 tests,alpha.05.
All 224 lives are required; no outcome-dependent stopping, selective exclusion,
positive-result selection or final-source revision. Missing/extra/duplicate worlds,
wrong dose, mixed parameters/revision or source/runtime changes stop validation.

Supportive mechanistic signatures, prespecified but not jointly familywise-confirmed:
- T: positive Holm-significant behavioral contrast; paired nominal95% lower bounds
  above zero for learned decision-space g12, g12/g1, g12/g2 and target-aligned H
  projection versus T_OFF. Mean learned g1,g2 each≥90% of T_OFF; this is an
  operational preservation floor, not a statistical noninferiority claim.
  Any undefined ratio blocks the signature
- H: positive Holm-significant behavioral contrast and nominal95% upper bound
  below zero for within-state corrected−raw bias RMS. F/G/H preservation is only
  an implementation invariant and does not count as independent biological evidence
- J: both behavioral contrasts positive and Holm-significant; nominal95% lower
  bounds above zero for learned decision-space g12 and target-aligned H projection
  against both matched controls. No dose-isolated mechanism claim

Secondary descriptive outcomes: all checkpoint old/new accuracies, old W−N_old
benefit, old interference loss, birth tensors, component/target-projection values,
raw second/first sensitivity, offline grand-mean centering, doses/clipping, memory
and runtime. Offline centering uses probe inputs and is not an online learner.
Historical2.3–2.5× and30.5→47.6% claims were not verified in available primary
sources and are not treated as facts or thresholds. Nonsignificance does not
establish equivalence or no possible benefit.

## Verification and resources
Receipt-only reconstruction independently recomputes fixtures, scoring and tensor
reductions, but shares a declared decomposition function and is not a simulation
replay. All 7 arms in final world 300001 are preselected for exact fresh-process
replay; compare every field except resource measurements. Record distinct PIDs,
all replay receipts and replay audit. Final audit enforces their presence/equality.

Python 3.11.15,NumPy 2.2.6,SciPy 1.14.1,Numba 0.61.2,pandas 2.2.3; numeric thread counts1.
One worker. Hard caps:8 worker-hours including replay and interrupted jobs,300s per
life,800MB worker RSS,200MB new results,12 elapsed wall-hours from initial launch,
including publication waits/resumes. No purchases. Cap violations or technical
failures stop execution and preserve logs. Unknown interrupted worker duration is
conservatively charged from its saved start time; budget does not reset on resume.

Receipts are write-once. Publish checkpoints after every complete world on branch
response-mechanisms-20261001 using ordinary fast-forward updates. Publication ACK
must bind world, lock digest and remotely verified commit and be retained. After
all 224 primary lives plus7 replays, run the locked analysis/audit and publish all
positive/negative/failure results with exact remote commit verification. No merging
or force-pushing is part of this experiment program.
