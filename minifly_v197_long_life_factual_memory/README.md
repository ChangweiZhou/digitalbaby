# MiniFly V197 — Long-Life Factual Memory and Prose Prediction

V197 completed 28 synthetic-filler and eight real-prose-filler factual-memory lifetimes, plus five predictor runs on a chronological held-out suffix of Alice. The factual experiments isolated the recovered Round 7 float32 readout. The synthetic factorial compares 4-grams and importance-scaled writes. The scaling comparison tests 65,536 and 131,072 units at 4,096 facts.

The combined address and write rule achieved 94.5% oldest-wave recall at 1,024 facts with synthetic filler and 94.8% with Alice filler. At 4,096 facts, doubling the store raised oldest-wave recall from 64.6% to 73.7%. The held-out prose score was 2.44 bits/byte for the V9.2 hashed predictor and 5.65 for the Round 7 lag predictor.

## Reproduction

Use Python 3, NumPy and SciPy. Run with OPENBLAS_NUM_THREADS=1.

- python -B run.py reproduces the synthetic factual runs.
- V197_ALICE=/path/to/pg11.txt python -B filler.py reproduces real-prose filler runs.
- python -B prose.py /path/to/pg11.txt reproduces the held-out predictor comparison. This runner resolves a bundled extract of the V9.2 predictor source.

Use Project Gutenberg eBook 11 at https://www.gutenberg.org/cache/epub/11/pg11.txt and verify its digest against FILLER_PLAN.json or PROSE_NATIVE_TEMPERATURE_RESULTS.json. The recovered historical runner is retained for provenance. The V197 runners use selected components from it. The original session export is unnecessary for reruns.
