# MiniFly V194 — Contrast-Normalized Successor Interface

All 96 official runs completed. Contrast normalization failed the promotion gate. At block 512 it scored 56.3% on noisy changed pairs versus 76.0% for raw frozen, a paired difference of −19.8 points (95% interval −31.0 to −8.6). It also missed the clean guardrail. The raw V193 interface remains the reference for this configuration.

The arms shared observations and query order within each condition. Clean and noisy continuations shared seeds, but corruption consumed RNG draws and changed later query order between conditions. Feature-drift and resource diagnostics were not captured.

Use Python 3 with NumPy and SciPy. Run `OPENBLAS_NUM_THREADS=1 python -B official.py` for the official roster. To rerun the engineering checks, create a sibling alias named `minifly-v193` pointing to `minifly_v193_reusable_relational_operator`, then run `OPENBLAS_NUM_THREADS=1 python -B qualify.py`. The archived runners retain their recorded source hashes.
