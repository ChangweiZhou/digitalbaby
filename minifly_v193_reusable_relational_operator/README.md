# MiniFly V193 — Reusable Relational Operator

The primary clean test passed. All 96 runs completed across 12 clean worlds and 12 continued-noise worlds, with four arms per world.

At 128 clean post-change blocks, the reusable operator scored 99.5% on changed held-out pairs and 100% on still-valid pairs, including its generated newline. The relational readout stayed frozen and received zero new relation labels. Under continued 10% observation corruption, it scored 69.8% and 85.4%, respectively. The jointly trained reference also missed both floors at that checkpoint under continued noise.

Use Python 3 with NumPy and SciPy. Run the engineering checks with `OPENBLAS_NUM_THREADS=1 python -B qualify.py` and the official roster with `OPENBLAS_NUM_THREADS=1 python -B official.py`.
