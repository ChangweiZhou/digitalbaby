# Reproducing the preserved independent audit

These files preserve the exact audit evidence. They are not modifications to the
locked scientific implementation. Read `../INDEPENDENT_AUDIT.md` and
`ROUNDOFF_REVIEW.md` before interpreting invariant H-minus-FE0 p-values.

The original auditor's SHA256 is
`035df3473c4eba2843fe404b873d6e5f5cca61a0e8b07819d26a9b63cfa98fd5`.
The arithmetic-order-aligned auditor's SHA256 is
`ecb392e0332f4a833840f42c54dc4b1d826ce9081aa6da2e466f3240e20f84e9`.
Their exact one-line difference is in `ROUNDOFF_HELPER_DIFF.patch`. Preserve both
files unchanged, including their historical absolute `ROOT` constant.

Use the runtime and upstream package-restoration instructions in the experiment
README. Run the examples below from the repository root. They read the complete
224-primary-plus-seven-replay dataset and do not run new experimental lives.

## Portable numerical check

Import the preserved corrected helper and supply data from the current checkout.
The following checks every independent numerical comparison and qualification;
it is not the separate full receipt or operational audit. It creates no output
files and does not change any source or saved result.

```sh
python - <<'PY'
import importlib.util, json
from pathlib import Path
root = Path('response_mechanisms').resolve()
helper = root / 'results/final/audit/verify_complete_locked_order.py'
spec = importlib.util.spec_from_file_location('preserved_auditor', helper)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
a.ROOT = root  # Invocation-local relocation; the file and its hash stay unchanged.
lock = a.read(root / 'SOURCE_LOCK.json')
assert a.sha(root / 'SOURCE_LOCK.json') == a.LOCK_SHA
docs = {arm: [a.receipt(root / f'results/final/{arm}/{w}.json.gz')
              for w in lock['worlds']] for arm in lock['arms']}
result = a.numerical(docs, a.read(root / 'results/final/FINAL_METRICS.json'))
print(json.dumps(result, indent=2, allow_nan=False))
PY
```

Importing `verify_complete.py` instead reproduces the original numerical failure;
do not change its tolerances or present that failure as a new scientific result.

## Full combined check in a relocated checkout

In the same import pattern, set `a.ROOT = root`, set `sys.argv` to
`[str(helper), '--output', str(new_output_path)]`, then call `a.main()`.
Choose a new output path: the helper deliberately refuses to overwrite evidence.
Its parent directory must already exist.

The full check also needs the original upstream package/runtime, all operational
records and a `response_mechanisms/scratch/response-supervisor.lock` file. On a
clean, inactive reproduction checkout only, create the scratch directory and an
empty lock file if absent. Never replace a live coordinator's lock. The check
acquires that lock nonblockingly and reads the final ledger/status.

A new checkout's idle lock cannot establish the historical coordinator's
termination. Likewise, rerunning after the original deadline cannot establish
that the original audit preceded it. The saved combined audit and operational
records preserve the historical observations. A rerun's timestamp, deadline flag,
and directory-size measurement naturally differ; they are new observations.

## Reproducing the supplemental roundoff review

`roundoff_review.py` was originally located under
`response_mechanisms/scratch/final_audit/`. It derives the experiment root from
that location and imports the original helper from that same directory. Its
published copy is retained byte-for-byte as evidence, rather than silently
rewritten for the publication directory.

In a clean reproduction checkout, create that original scratch directory and
copy the published `roundoff_review.py` and `verify_complete.py` into it without
changing their bytes. Do not overwrite an existing file or prior diagnostic.
Run `python response_mechanisms/scratch/final_audit/roundoff_review.py`.
It writes a new, write-once `scratch/final_audit/ROUNDOFF_REVIEW.json`. The
diagnostic's `numerical()` calls do not use the helper's historical `ROOT` value.
The script compares both arithmetic orders in memory and retains all four
original p-value mismatches. It never replaces `FINAL_METRICS.json`, receipts,
the locked report, or scientific source.
