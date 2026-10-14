# Cycle 1 — design, hostile audit, test, revision

Design: unchanged ERROR trajectories with one local shared/private confidence gate; retain both reuse families and taught-memory guards.

Audit finding: per-option four-channel gating in binary choice allows unrelated channels 2/3 to change an L/R answer. A deterministic hostile witness found this effect. Revision: derive one gate from the two query-level channel-1-minus-0 scores, use one common weight, preserve original floating arithmetic.

Seven unit tests passed: hostile witness, strict boundaries/ties, forbidden target/task/world API, nonfinite/wrong-roster rejection, 800 source-baseline arithmetic checks, pair-order symmetry and bank-offset behavior. Regression logs include reruns after the cycle-2 revision; these are technical repeats only.

Read-only census matched ERROR/Q_HALF on 76,128 existing rows, with zero new learning records. The original shared-dominance gate left binary final held-out answers unchanged. That observation led to the algebraic audit in cycle 2. No threshold sweep was performed.

Evidence correction: +4.86 pp is the causal-contrast change, not raw accuracy; raw improvement is +3.47 pp. Latin raw publication is incomplete; only one official R_center world and one ERROR technical pilot can be rescored. The initial census remains preserved separately from the revised-rule census in cycle 2.

Outcome: first technical revision passed; scientific rule required the cycle-2 correction below. No efficacy or promotion claim.
