# Compact R_center: physical shared-bank ablation

Frozen next step authorized by the user, after three design/audit/trial/revision
work cycles. Parent `../r_center_core_v1` is required and remains unchanged.
Read PROTOCOL.md and SPEC.json. E1 only; E3 is not a gate.

Development uses worlds 490101–490103. Science uses 491001–491096, two arms and
four histories = 192 world-arm jobs / 768 lives. All actual native/signed store
calls are logged and independently checked, including no-write state equality.
Final private states/values must match across the physical deletion.

Use Python 3.11.5, NumPy 2.2.6, SciPy 1.14.1, Numba 0.61.2. Set
OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS/NUMBA_NUM_THREADS/
VECLIB_MAXIMUM_THREADS to 1 before starting trials or science.

```sh
python -u cycle.py --cycle 1
python -u cycle.py --cycle 2
python -u cycle.py --cycle 3
python seal.py
python launch.py
```

Developer sealing requires all three accepted receipts under the SAME final
source identity. It never repairs a failed lock at runtime. The detached
supervisor owns caffeinate and four workers, persists status and records,
stops on a scientific failure and does not automatically retry. A terminal
need not remain open. Completed receipts are never rerun. An interrupted job
requires explicit review/authorization before `compact_worker.py --resume`.

Read results/STATUS.json, results/LAUNCH.json, results/supervisor.log, active
worker cursors and failures. On complete receipt roster, the supervisor runs
compact_analysis.py and final_compact_audit.py and uploads the source, cycles and results to
`codex/r-center-compact-ablation-20261003` in ChangweiZhou/digitalbaby. Analysis
does not fit another output rule or enlarge n after an unfavorable result.
