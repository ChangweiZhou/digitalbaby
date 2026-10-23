# Artifact-only reproducibility

The executed environment is recorded in ENVIRONMENT.json. Python 3.11 and NumPy are sufficient to reduce the saved artifacts; native imports, Numba and fitting are unnecessary for these commands.

From the repository root:

```sh
python minifly_recoverability_v1_dev_20261022/audit.py
python minifly_recoverability_v1_dev_20261022/reduce_details.py
```

The compact RESULT_BUNDLE.zip preserves repository-relative paths. Extract it into an empty directory and run those same commands there. The included authority snapshots allow hash checks without the full project checkout. The reducer reconstructs derived logits from raw response tensors and frozen weights using independent matrix equations; it never trains or reruns a native trajectory. The complete repository also retains the original derived logits and native checkpoints; FULL_ARTIFACT_MANIFEST.json indexes both included and omitted files.

`diagnose.py` and `payload_bypass.py` are posthoc artifact analyses. They use the separately retained original derived-logit files and therefore need those files from the repository. Their JSON results are included in the compact bundle. They do not change frozen head weights or the primary result.

`run.py` is the executed learning harness, preserved for inspection. **Do not run it to reproduce this report:** it performs native teaching/fitting and writes artifacts. The existing immutable results suffice for reduction. Cycle source snapshots document the pretraining and preassessment revisions.

The archive is a reproducibility packet for this bounded E1 pilot, not a standalone redistributable native simulator or a claim that all prospective protocol requirements passed. Read REPORT.md's limitations before interpreting FINAL_AUDIT.json.
