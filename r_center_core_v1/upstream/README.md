# MiniFly R3 observed-outcome deletion ablation: final result

**Completed and independently accepted on 2026-10-03.** All 192 scientific receipts cover the fixed 64 worlds (290001–290064) and three arms (R3, R_center, R0_signed). No worlds are missing. This publication supersedes the earlier running-status protocol checkpoint.

## Result

Centered signed teaching retains the demonstrated gain within the preregistered five-percentage-point tolerance for the causal old-teaching effect. Directional gain replication passes. A strictly positive contribution of learned shared prediction p is **not established**.

The primary R3-minus-centered effect is **+1.464844 percentage points**, with a Bonferroni simultaneous interval of **[−0.610368, +3.540056] percentage points**. Its upper bound is below the fixed five-point tolerance. This does not prove equality or the absence of every positive contribution.

The supported simplification is removal of learned-p dependence from private teaching in this observed exact cue-to-outcome retention assay. The shared bank remains active in learning and the fixed combined readout. Shared-bank deletion, capacity/runtime savings, broader generalization and autonomous learning are untested.

## Read the study

- [Scientific report](reports/R3%20Observed%20Outcome%20Ablation%20Scientific%20Report.docx)
- [Independent final result review](reviews/RESULT_REVIEW.md), including all seven contrasts, descriptive bank scores, recovery/resource limitations and identity bindings
- [Final machine-readable analysis](results/ANALYSIS.json)
- [Original result acceptance](reviews/RESULT_ACCEPTED.json)
- [Frozen protocol](PROTOCOL.md), [protocol lock](PROTOCOL_LOCK.json) and [science lock](LOCK.json)
- [Publication scope and omissions](PUBLICATION_SCOPE.md)

The protocol's historical “preregistration draft” heading and outcome-blind status are preserved exactly as locked before execution. They do not describe the current completion status. Likewise, the original scientific acceptance did not itself authorize publication; it remains unchanged as historical scientific evidence.

## Scientific design and interpretation

- 64 independent matched worlds; 192 world-arm receipts; 576 branch lives
- Arms: R3 uses p−y, centered teaching uses 1/4−y, and the contemporaneous original-gain reference uses 1−y
- Seven prespecified world-level contrasts, with paired Student-t Bonferroni nominal simultaneous 95% intervals and the locked exactly-zero-variance fallback
- The five-point tolerance concerns loss in the causal teaching effect, not necessarily raw W accuracy
- Student-t coverage is approximate for these bounded discrete outcomes
- Raw W guards are above 25% in this fixed sample only; no separate inferential above-chance claim follows
- Three infrastructure interruptions retain unknown unobserved resource peaks. Completed measured runs passed their caps; unknown peaks are not represented as measured passes

## Files and verification

The original scientific receipts, final analysis, final authenticated ledger and science manifest are unchanged. Qualification includes the original 27-test certificate and three technical pilot receipts, excluded from scientific inference. The frozen src/ and tests/ files are preserved; see the scope note for inherited vendor provenance-path handling.

PUBLICATION_MANIFEST.json binds every published file by SHA-256 and size, excluding the manifest itself to avoid a self-hash cycle. From this directory, run the standard-library inventory check:

```sh
python publication/verify_public.py
```

The inventory verifier checks publication bytes only. It does not rerun the science or certify source identity. **Eight inherited vendor files contain privacy-redacted local documentation paths. The original LOCK.json is unchanged and retains the original execution hashes. The frozen source-hash gate and default scientific integrity entrypoints therefore reject these public copies.** Original and published hashes are distinguished in PUBLICATION_MANIFEST.json. Exact byte-for-byte replay of the original locked source requires the unredacted originals, which are not part of this public projection.

The recorded scientific runtime was Python 3.11.15, NumPy 2.2.6, SciPy 1.14.1 and Numba 0.61.2. All src/ and tests/ files and all numeric model/calibration inputs retain their original bytes. No claim is made that the entire public vendor tree is byte-identical to the executed tree.

No simulation, design tuning or new behavioral data was used to prepare this publication. The ablation ends under the frozen plan.
