# GATED-FEEDBACK V2 technical implementation

This folder contains the standalone KEEP/GRU and G62/LSTM learners, CE_ONLY,
OLD and SAFE learning rules, fixed technical dispatcher, separate receipt/data
auditor, tamper fixtures and the 20 locked-effect saved-output reducer.

Use Python 3.11 with NumPy 2.2.6 and Torch 2.8.0. The qualified local interpreter
is ../arithmetic_kernel_no_attention_dev_v3_20261013/.venv/bin/python. Torch and
BLAS use one thread per worker. No network, model downloads or pretrained weights
are required. The included dependency
../arithmetic_end_to_end_design_v1_20261013/validate_design.py is environment-only.

Read REPORT.md, QUALIFICATION.json and FINAL_AUDIT.json first after completion.
cycles/cycle1 contains generic implementation tests and byte-mapping fixtures;
cycle2 and cycle3 contain the exact source snapshots, native logs, accepted
receipts, independent certificates and full learning-state NPZ files. Each full
world/model job has seven correlated terminal histories and sixteen archives.

Do not launch runner.py or supervisor.py again in this directory. Existing
state directories/receipts deliberately block repeated trajectories. A partial
directory must be preserved and investigated, not deleted to bypass that block.
Do not rerun tests.py merely to view results: generic acquisition performs actual
training. tamper_tests.py uses synthetic receipt copies only. Read its saved
TAMPER_TESTS.json instead of repeating qualification for reporting.

The formal DEV and CONFIRM designs and effect definitions are retained in
DESIGN.json/EXPERIMENT.md. There is intentionally no authorized scientific launch
entry in this technical dispatcher. Formal admission would require a separately
authorized science-envelope/manifest adapter and fresh designated worlds; this
folder must not be relabelled as formal science. No configuration is adopted by
technical qualification alone. The existing adopted core is unchanged.

The local monitor is read-only at http://127.0.0.1:8881/. It reports per-cycle
live files and explicitly separates partial work from committed jobs. It cannot
start, stop, resume or alter a run. The only native worker entry points accept
the frozen cycle2/cycle3 worlds. The final bundle includes the implementation,
all three technical cycles and the environment dependency.
