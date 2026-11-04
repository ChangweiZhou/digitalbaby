# MiniFly two-step parity prototype

All 96 runs completed. The aligned arm scored 100% on both holdout suites and cleared the numerical gate. See FULL_RESULTS.json for paired intervals and per-world results.

The prototype used operand byte IDs directly, fixed distance holdouts across worlds, and omitted the full shortcut-baseline ladder. The no-feedback arm emitted an invalid answer byte. Formal E3 qualification remains open.

Run with NumPy and SciPy installed using `OPENBLAS_NUM_THREADS=1 python -B full.py`.
