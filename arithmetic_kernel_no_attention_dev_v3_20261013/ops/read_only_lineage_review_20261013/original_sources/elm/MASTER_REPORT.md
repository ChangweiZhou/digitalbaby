# ELM research master report

Generated: 2026-09-04 16:21:07

## Stage status

- **p0**: `ok` — {'status': 'ok', 'runs': 3}
- **p1**: `ok` — {'status': 'ok', 'runs': 60}
- **p3**: `ok` — {'status': 'ok', 'sources': 3}
- **p4**: `already_complete` — {'status': 'already_complete'}
- **p5**: `ok` — {'status': 'ok', 'parts': [('dynamics', 'already_complete'), ('interference', 'already_complete')]}
- **p6**: `ok` — {'status': 'ok', 'parts': [('shd', 'skipped: tonic missing; rerun with --install-optional'), ('nomniglot', 'skipped: raw data/repo/dependencies unavailable; provide --nomniglot-data and optionally --install-optional')]}

Collected **183** run-level `summary.json` records.

Machine-readable aggregate: `MASTER_SUMMARIES.csv`.

## Interpretation order

1. P0 asks whether newly grown recurrent edges causally support long-delay memory.
2. P1 separates pruning, random rewiring, and gradient-guided topology learning under equal optimizer-step and unique-label budgets.
3. P3 asks whether a learned graph transfers after all weights are reset.
4. P4 asks whether the ELM cell itself beats simpler recurrent cells.
5. P5 tests whether E/I balance and temporal homeostasis repair blank-period runaway activity and whether memory survives distractors.
6. P6 moves the same sample-efficiency criterion to genuinely temporal / few-shot datasets.
