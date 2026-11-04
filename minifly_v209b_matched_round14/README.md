# MiniFly V209B — Matched Round 14 Comparison

The Round 14 validation spot check scored 46.0% on jointly new names and locations at hash seed 7, versus the reported 49.5% at seed 1.

The matched comparison ran 36 evaluations on three fresh story batches with hash seeds 7, 8 and 9. On jointly new words with familiar wording, the recovered learner, V209 identity and V209 contextual selectors scored 27.3%, 2.7% and 100.0%, respectively. With randomized candidate ordering they scored 51.0%, 23.0% and 58.3%. The contextual selector scored 0% on the unseen template despite 100% target availability.

## Reproduction

Use Python 3 and NumPy. From the repository root:

```sh
ln -s minifly_v209_contextual_answer_selection minifly-v209
ln -s minifly_v202_babi_qa1 minifly-v202
ln -s minifly_v205_round12_qa1_qa2 minifly-v205
cd minifly_v209b_matched_round14
OPENBLAS_NUM_THREADS=1 python -B run.py
```

PLAN.json records source and generated-data hashes. The recovered learner is included. The optional recovery script requires the original session export.
