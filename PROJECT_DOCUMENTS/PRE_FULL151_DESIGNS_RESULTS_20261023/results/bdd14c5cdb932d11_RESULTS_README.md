# V74 completed result supplement

REPORT.md interprets the completed493-job run. The original runner source/config/protocol remain frozen; results are not a new experiment or a model promotion. The prepared runner ZIP is unchanged.

From the project root, using the existing environments:

```sh
_omnibus10_env/bin/python minifly/iteration19/runner.py verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 _omnibus10_env/bin/python minifly/iteration19/reporting/audit_and_tables.py
MPLCONFIGDIR=minifly/.mplconfig .venv/bin/python - <<'PY'
import sys,runpy
sys.path.append('_omnibus10_env/lib/python3.11/site-packages')
runpy.run_path('minifly/iteration19/reporting/plot74.py',run_name='__main__')
PY
```

The audit needs excluded raw jobs and parent assets in the manifest. It verifies hashes, completeness, input exclusions and gates; replays capacity and independent direct-step probes; and writes final tables. It does not train a new learner or select new parameters. Reproducing the actual campaign uses the separately delivered runner and original parent assets, not report tables alone. Preserve the original completed directory and scientific lock.

The result bundle contains final_tables plus selected original aggregate decisions/gates, figures, configuration, report verification and reporting scripts. It references unchanged runtime/source files, raw job outputs, inputs and prior archives. The primary development/confirmation tables average focal ages before history-level analysis. The surgery/dose/information panels use their own explicit history counts and definitions; do not pool them as independent trials. A candidate confirmation was skipped; the32-history scale replication still completed.

Use reporting/verify_result_bundle.py for extracted payload verification, optionally --project /absolute/project/path to check parent references. Source-code, receipt and array integrity are independent from scientific acceptance. Read REPORT_DELIVERY.json for completed report status; the runner's STATUS.json is intentionally preserved as an execution record.
