# Cycle3 prospective design: qualification, auditing and final lock

Prepare before running world290002. Retain all seven arms and the cycle2 equations
and shared gain; there is no selection of successful pilot variants. Cycle3 uses
six repetitions per cohort, the planned final-life dose, on one fresh exploratory
world. All scientific outcomes remain exploratory.

## Changes to qualify an auditable final run
- Log resolved parameters, literal teacher/end times, pre-update homeostatic state,
  actual versus proposed/clipped J steps and L2 step size
- Save canonical independent-birth certificates, frozen byte-predictor verification,
  final timing support/incoming-budget/positivity checks, precise added-state inventory
- Receipt-only auditor rebuilds exact fixture, verifies every teacher/write flag,
  rescores all predictions, recalculates all decompositions and paired deltas, checks
  source/runtime/row counts, exact FE0/T_OFF parity, H invariants and J_ADD's zero
  extra interaction. Deliberate receipt corruptions must be rejected
- Test the complete final analysis on pilot data before source lock. Include birth
  tensors and decision-centered W−N_old target-aligned interaction, not just activity
- Freeze analysis, mechanism, runner, audit and tests before opening final worlds

## Prespecified final design and inference
32 independent, previously unused worlds300001..300032. Every world runs FE0,
T_OFF,T,H,J_ADD,J,J_SHUFFLE with all16 old and16 new random pairs, six repetitions
per cohort, exactly the same labels/order/timing across arms, and W/N_old branches.
No final world may influence design, parameters, arm inclusion or endpoints.

Primary outcome: W final old-pair accuracy after new-cohort interference and a
second day. Four paired primary contrasts: T−T_OFF, H−FE0, J−J_ADD, J−J_SHUFFLE.
Two-sided Student-t paired-world tests and95% intervals; Holm correction across
these four p-values at family alpha.05. Report every contrast including negative
and null outcomes. World is the unit, never individual probe or branch.

Operational compound qualification:
- T: positive Holm-significant primary effect; paired95% lower bounds above zero
  for BOTH interaction/main ratios, absolute interaction RMS and target-aligned H
  projection in decision-centered W−N_old tensors. Mean learned F and G RMS each
  must be at least90% of T_OFF. This last point is an operational floor, not a
  separately demonstrated statistical noninferiority claim. Any undefined ratio
  blocks qualification and is reported
- H: positive Holm-significant primary effect plus95% upper bound below zero for
  within-same-state corrected minus raw channel-bias RMS. F/G/H preservation is a
  numerical invariant only
- J: both primary comparisons positive and Holm-significant, with paired95% lower
  bounds above zero for learned decision-space absolute H RMS and target-aligned
  H projection against both matched controls. This is an architecture/rule result,
  not dose-isolated because realized residual-driven writes can differ

Secondary descriptive endpoints: immediate and delayed old/new accuracy, old
W−N_old learning benefit, old interference loss, all B/F/G/H magnitudes and target
projections, raw FE0 second/first RMS ratio, and offline raw-value grand-mean
centering. Offline centering is a diagnostic using all probe inputs; it is not
credited as an online learner or used for tuning. Report target-table F/G/H RMS
variation, update doses, clipping, memory, timings and all world-level observations.

## Resource and persistence proposal
One worker, thread-count1 for numeric libraries. Based on cycle2 measurement so far
51–61s per4-repeat life, plan approximately4–6 worker-hours for224 final6-repeat
lives (32×7), with an8-worker-hour hard cap,300s individual-life cap,800MB per-worker
RSS cap,200MB new results cap and12h elapsed-session cap excluding explicit resume
time. These bounds will be finalized from cycle3 before lock. No purchases or new
paid machines. Immutable receipts; stop on technical failure or source drift;
ordinary fast-forward checkpoint publication on the separate authorized branch.
Scientific failure does not stop the roster. A technical abort is not a negative
scientific result and is preserved with its log; any source revision after launch
invalidates the affected final run and requires a new prospective decision.

## Independent-audit clarification before pilot
Final validation is automatically tied to SOURCE_LOCK.json, including exact
runtime import hashes and pandas2.2.3. Probe clocks are checked. All7 arms of
world300001 are prospectively selected for fresh-process replay; only resource
measurements may differ. Replay time counts in the8h worker cap. The12h wall cap
runs from initial launch and includes publication and resumes without reset.
Mechanistic signature gates remain prespecified supportive nominal95% checks,
not jointly familywise-confirmed mechanism claims. Receipt-only reconstruction
is distinguished from actual simulation replay.
