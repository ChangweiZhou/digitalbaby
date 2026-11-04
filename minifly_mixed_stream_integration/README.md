# MiniFly mixed-stream integration

All eight evaluation worlds completed. The factual and relation endpoints passed the proposed floors. Factual retention was 96.9%, oldest-wave recall 93.0%, and durable correction 96.2%. At 128 post-change blocks, the zero-label relation operator achieved 99.2% on changed held-out pairs and 98.4% on still-valid pairs. V9.2 outscored Round 8 at 2.757 versus 2.857 bits per byte on frozen held-out prose.

The factual and relation pathways replayed in one pass. Predictor comparisons replayed the same byte stream separately. Worker RSS is not a measurement of all three modules running in a single real-time loop.

## Reproduction

Use Python 3 with NumPy and a C compiler. From the repository root, create local aliases for the original dependency paths used by the source-locked runner:

```sh
ln -s minifly_v197_long_life_factual_memory minifly-v197
ln -s minifly_v193_reusable_relational_operator minifly-v193
cd minifly_mixed_stream_integration
cc -O3 -ffp-contract=off -dynamiclib predictor.c -o predictor.dylib
export MINIFLY_ALICE=/absolute/path/to/pg11.txt
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
python -B run.py dev
python -B run.py official
```

The compiler command targets macOS. Use Project Gutenberg eBook 11 and verify its SHA-256 against PLAN.json. The development result is world-20100.json. RESULTS.json contains the official paired roster and aggregate metrics.
