# MiniFly frozen QA1 generalization

The official QA1 generalization gate failed. All 54 jobs completed across 12 fresh story batches, three Python hash seeds, three renderings, two feedback-history conditions, and three write controls.

With no gold answer feedback, the learned gate scored 99.9% on the original wording, 92.6% on novel verbs retaining the location prefix, and 15.5% on changed location prefixes. In the last condition, it missed all 2,400 location-start positions per seed, while the explicit-position control reached 92.5%. Omitting gold answers left learned-arm accuracy unchanged in all three renderings. The random control matches primary triggers to the explicit-position budget. Follow-up and row write counts are reported separately.

The QA2 checks cover reference state semantics only. The QA2 learner experiment remains unrun.

## Reproduction

Use Python 3 with NumPy and SciPy. From the repository root, create the local aliases expected by the source-locked runner:

```sh
ln -s minifly_v202_babi_qa1 minifly-v202
ln -s minifly_v203_babi_qa1_binary_gate minifly-v203
cd minifly_qa1_frozen_generalization
OPENBLAS_NUM_THREADS=1 python -B qualify.py
OPENBLAS_NUM_THREADS=1 python -B official.py
```

OFFICIAL_PLAN.json records source and rendered-story digests. OFFICIAL_STORIES.json preserves the underlying story events. OFFICIAL_RESULTS.json contains the scored outputs and batch-paired comparisons.
