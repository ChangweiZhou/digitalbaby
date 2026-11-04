# MiniFly next-phase experiments

- FULL_RESULTS.json contains the 96 parity runs. The aligned map achieved 100% on both holdouts and passed the numerical gate.
- STORAGE_RESULTS.json summarizes the 16 storage runs. At 512 facts, float16 matched float64 on the measured accuracy endpoints and reduced readout storage from 128 MiB to 32 MiB.
- The graph reference used the generator's cycle order. The storage timing includes the bounded encoder cache. Mixed-stream integration remains unrun.

Use Python 3 with NumPy and SciPy. From this directory, run `OPENBLAS_NUM_THREADS=1 python -B full_parity.py` for parity, or `OPENBLAS_NUM_THREADS=1 python -B storage.py` followed by `python -B summarize_storage.py` for storage. The published storage runner resolves the recovered source relative to this repository.
