# MiniFly V191 — Continual Structural Learning

**Disposition:** Slow deployment did not pass the promotion gate. Direct online updates remain the reference under the tested conditions.

The official run completed 48 jobs across 12 paired worlds. After 32 feedback events, direct online scored 98.4% on changed pairs and slow deployment scored 78.6%. Both reached 100% by 128 events. Across post-change checkpoints, slow deployment was 0.7 percentage points below direct online on still-valid pairs (paired 95% interval: −1.8 to +0.5), short of the required +5 points.

The lifetime used 10% observation noise followed by one opposite-parity swap per world. The run did not measure runtime or peak memory by arm. The design PDF also describes separate factual-memory and integration experiments, which this package does not evaluate.

Use Python 3 with NumPy and SciPy. Run `OPENBLAS_NUM_THREADS=1 python -B official.py` to repeat the official roster, or `OPENBLAS_NUM_THREADS=1 python -B engineering.py` for the engineering worlds. OFFICIAL_RESULTS.json contains per-world scores and paired contrasts.
