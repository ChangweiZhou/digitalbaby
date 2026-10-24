# RCE/V87 follow-up runner — engineering test report

**Exact runner:** `RCE-V87-FOLLOWUP-1.1-AUDITED`  
**Date:** 2026-09-16

## Static / hostile tests

- `python minifly_rce_v87_followup_runner.py --self-test` — **PASS**.
- `pytest -q` — **12/12 PASS**.
- `python -m py_compile minifly_rce_v87_followup_runner.py` — **PASS**.
- `bash -n RUN_FOLLOWUP.command` — **PASS**.

The hostile suite checks:

- exact RCE-FULL and TRACE source IDs;
- exact trace-restoration delta `{4,11,12,13}`;
- all 34 low/high per-gene box attacks;
- exact prior-cue exclusion count/hash;
- six valid deactivation/reactivation episodes;
- challenge-fresh semantics and no post-reactivation reuse;
- explicit active-fraction overrides with unchanged genome/B;
- learner-blind relevance API;
- clone-only lesion/mechanism probes;
- absence of collateral/churn gates from Round-A decision logic;
- Round-A decision tamper detection;
- balanced world labels;
- exact P1–P7 denominators.

## Exact-final integration rounds

The exact shipped source completed reduced engineering fixture runs through all pipeline stages.

### Round A

- 2 paired histories.
- 4 normal arms.
- RCE-FULL no-learning and unpaired controls.
- complete preflight life for every arm.
- lesions, reactivation, receipts, bootstrap decision, reports.
- tested with 1 worker and 2 workers.

Byte-identical across serial and parallel for:

- `DECISION.json`
- `NORMAL_RESULTS.jsonl.gz`
- `PAIRED_NONINFERIORITY.csv`
- `WORLD_PARTITION_AUDIT.json`

### Round B

- 2 paired histories.
- all 6 factorial causal cells + REF/V84E anchors.
- complete preflight life for every arm.
- active-fraction interventions, feedback contrasts, interactions, lesions, mechanism branches.
- tested with 1 worker and 2 workers.

Byte-identical across serial and parallel for:

- `RESULT.json`
- `NORMAL_RESULTS.jsonl.gz`
- `FACTORIAL_PAIRED_CONTRASTS.csv`
- `MECHANISM_EVENTS.csv`
- `WORLD_PARTITION_AUDIT.json`

### Receipt-level resume

From the exact-final serial Round-A fixture output, three sealed receipts and all derived reports were deleted, then the same experiment was rerun with two workers. Recomputed output was byte-identical for:

- `DECISION.json`
- `NORMAL_RESULTS.jsonl.gz`
- `PAIRED_NONINFERIORITY.csv`
- `CONTROL_SUMMARY.csv`

## Exact shipped hashes before package manifest generation

```text
runner   49b19beb9248c3a848c972659c7fe2fa714f313badecf086102048c5d898d435
design   e37c687452b53e552d3850897f0a123ada4364bbf77839add86903ad0adc344e
prior    83788cc0c410d9e51d97ae05a7d399657a558a9ec1e7d0e2d0f80f59283a972c
```

## Scope limitation

All executed lives in this report use the engineering fixture backend. They validate experimental plumbing and semantics only. They are **not biological MiniFly evidence**. The complete native inherited MiniFly checkout is not mounted in this environment, so native efficacy and native source integration must be established by the Mac preflight and subsequent scientific run.
