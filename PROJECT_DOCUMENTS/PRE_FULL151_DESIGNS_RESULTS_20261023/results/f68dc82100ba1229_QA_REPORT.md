# MiniFly V86 QA report

## Scope

This report distinguishes orchestration/source validation from an actual MiniFly numerical run. The ChatGPT sandbox does not contain the user's `iteration24/src/common79.py` / downstream legacy checkout, so the biological backend cannot be executed here. The user's Mac checkout previously demonstrated that those files exist locally. The required first real numerical test is therefore the shipped `--quick --workers 1` run on that checkout.

## Passed checks

- `minifly_V86_bridge_runner.py` and `qa/test_v86_runner.py` compile with Python.
- `RUN_V86.command` passes `bash -n` syntax checking.
- The V82E engine source locks match the audited hashes.
- The V84E and V85E champion JSON locks match the uploaded Claude archive exactly.
- The latest Claude archive's `source/model_evo.py` and `source/common_evo.py` hashes exactly match the V82E engine, so all five V86 arms use one phenotype implementation.
- Required model API used by the runner is present in the audited source (`EvoLearner`, `encode_sparse`, and learner methods used by V86).
- Archive verifier passes against `minifly_evolution_V83_V84_V85_20260915.zip`.
- Non-scientific deterministic mock integration passes for all five arms x two histories (10 jobs), including branched assays, uninterrupted-life scheduling, acquisition/revision curves, age-local copied-state damage probes, the 36-memory untouched cohort, SHA receipts/resume, and aggregation.
- The mock integration produced all declared aggregate outputs, including acquisition/revision history summaries and paired contrasts.

## QA-driven corrections before release

- Removed an invalid `initial48_correct_survival` measure: deliberate reversal changes the correct label, so that quantity mixed forgetting with intended updating.
- The clean retention endpoint is now the early 36-memory cohort that is never directly challenged after initial acquisition.
- Expanded exact-pattern exclusion scanning to include prior V*/V*E input banks both inside and beside the project tree when available.
- The launcher uses a private `.venv-v86` with the previously successful pinned stack: NumPy 2.2.6, pandas 2.2.3, SciPy 1.14.1, Numba 0.61.2; the runner also masks optional pyarrow/numexpr/bottleneck imports and preloads the environment Numba to avoid the earlier ABI/vendor collision.

## Required Mac acceptance sequence

Run `--prepare-only` first, then `--quick --workers 1`. Do not launch the full campaign if either fails. The quick run is explicitly marked non-scientific and uses two histories plus one long-life epoch. After it completes cleanly, run the full 24-history campaign with two workers.
