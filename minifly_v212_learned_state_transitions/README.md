# MiniFly V212 — Learned State Transitions

V212 completed three development engineering cycles. The local sampled update agreed with a one-cell finite-difference fixture, and fresh-process checkpoint recovery reproduced the final state and evaluation scores. A corrupted checkpoint was rejected.

The candidate did not qualify for the official run. On probes where the location actually changed, full learning tied both controls at 28.3% in persistent coupling and scored 11.5% in one-time copying, versus 15.4% for both controls. Prediction loss remained near 2 bits per observation. The official roster stayed unopened, and forward transfer was not tested.

Use Python 3 with NumPy. Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B qualify.py` from this directory. ENGINEERING.json records per-arm results, the source digest and the checkpoint check.
