# MiniFly V201 — End-to-End Byte Interface

All eight paired worlds completed and passed the factual and relational floors. Still-valid fact recall was 96.9%, oldest-wave recall 93.0%, and durable correction for cohorts followed by another factual teaching wave was 95.7%. At block 128 after the graph change, changed-pair joint answer-plus-newline accuracy was 99.2% and still-valid accuracy was 98.4%. The post-change relation readout stayed frozen with zero new relation labels.

The explicit byte parser, V9.2 predictor, factual memory and relation pathway ran in one loop. Factual and relational state matched the tagged replay. V9.2 state and loss matched an isolated pass on the same physical stream. Mean throughput was about 155,424 input bytes per second versus 320,234 for the predictor-only pass. Worker peak RSS reached 356 MiB and includes harness overhead. Fresh-process checkpoint recovery was exercised on a development fixture.

## Reproduction

Use Python 3 with NumPy, a C compiler and the Project Gutenberg eBook 11 text with the digest in OFFICIAL_PLAN.json. From the repository root:

```sh
ln -s minifly_mixed_stream_integration minifly-integration
ln -s minifly_v197_long_life_factual_memory minifly-v197
ln -s minifly_v193_reusable_relational_operator minifly-v193
(cd minifly-integration && cc -O3 -ffp-contract=off -dynamiclib predictor.c -o predictor.dylib)
cd minifly_v201_end_to_end_byte_interface
export MINIFLY_ALICE=/absolute/path/to/pg11.txt
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1
python -B qualify.py
python -B resume_check.py
python -B live.py dev
python -B live.py official
```

The compiler command targets macOS. Runner source is unchanged from the scored run. The binary is built locally.
