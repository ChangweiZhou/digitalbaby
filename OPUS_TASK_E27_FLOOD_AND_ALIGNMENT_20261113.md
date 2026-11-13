# Draft for hostile audit: E27, two measurements on saved weights (13 November 2026, 11:43 EST by the clock)

> **Status.** Draft. **No worker starts from this file** until a hostile audit has read it and the coordinator has answered the audit. From 13 November all implementation and all runs are done by Opus 5.5 workers; the coordinator writes briefs and has them audited (Mr. Xu's instruction of that date). Codex is not available; the auditors are fresh sessions of the same model family as the coordinator and the workers, so they are second readers and not independent ones.

## What this job is

A measurement on weights that already exist. Nothing is trained: no optimizer may appear in any file of this job. Diagnostic, evidence level E2. It claims nothing about any learner's ability.

**The hypothesis under test** (the audit session's, review 9.112 E): the working rule's backward path uses weight signs and carries no magnitudes, and that one fact explains both the every-byte collapse of E24 and the weaker composition of E26 and E26-X.

**Predictions P27.1 to P27.5 and their readings are already registered** in `MOD97_LOCAL_RULE_20261106/E27_PREDICTIONS.md` (commit 2b730193). They are not to be changed.

## What exists, and what you must not take on trust

The coordinator wrote a first implementation before the division of work changed: `e27_measure.py` and `e27_run.sh`, with fixtures run on untrained networks only (`exp45/E27_FIXTURES_<seed>.json`). It has been read by nobody else and run on no saved file. **Treat it as a draft by an interested party.** You may use it, fix it or replace it. Every defect you find is reported, with the line.

Commands that showed the claims above (run from `MOD97_LOCAL_RULE_20261106`):
- fixtures: `env -i PATH="$PATH" HOME="$HOME" E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 OMP_NUM_THREADS=1 python3 e27_measure.py fixtures 73129` exits 0 and writes the fixture record;
- no optimizer: `grep -n "optim" e27_measure.py` prints nothing;
- it reuses `e25_alignment.grads` and `e25_alignment.alignment`, whose written-out GRU step E26's stage 1 showed bit-identical to the worker's own (`exp45/e26_qual/E26_STAGE1_RECORD.md`, check 1).

## Your work, in order

1. **Review the implementation against the predictions file.** For each of P27.1 to P27.5, find the exact quantity in the script's output that it is judged on, and say whether the script computes what the prediction's words say. Where the words are ambiguous, stop and report the ambiguity; do not choose.
2. **Independent check of the split.** The script claims the every-byte gradient equals the share-weighted sum of the answer part and the other part, under both paths. Confirm it with code of your own that does not call `flood()`, on an untrained network and on one saved file.
3. **Positive and negative controls for "flood".** Build one fixture in which the amplification must be far above 1 (for example a network in which you scale down one layer's weights so that the exact path damps and the sign path, scaled by the mean magnitude, does not, or any construction you can justify) and one in which it must be 1 (a path with no weight matrix between the loss and the tensor). The measurement is admitted only if it moves in the first and not in the second.
4. **Write the scorer before the run.** `e27_score.py`, reading only `exp45/E27_MEASURE_<seed>.json`, judging the five predictions exactly as registered, with a fixture record showing it on fabricated inputs for a hold and a miss of each. Commit it before the run.
5. **Run** `e27_measure.py run <seed>` for seeds 73129, 93683 and 116089, one process, and then the scorer.
6. **Part B needs the end-of-life weights of the six E26 lives** (`exp45/e26/e26_life_{S,B}_<seed>.pt`), which were never committed. If your container has them, commit them first and verify each against the hash in its receipt. If it does not, run part B on the six crossover lives only, report the six as absent, and judge P27.4 and P27.5 on the files that exist, saying so.

## Report

`E27_REPORT.md`: the five verdicts with their figures per seed; the tables by file; the controls of step 3; what you changed in the coordinator's script and why; what was absent. One paragraph as `E27_REVIEW_SECTION.md`. Work on your own branch; one commit at completion; nothing to the default branch.

## Limits to state in the report

Gradients at saved weights are not training. A file trained by one update is measured where that update took it. Three seeds, 256 items a panel, one size.
