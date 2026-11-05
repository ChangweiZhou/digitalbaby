# V217 — MiniFly Round 23–24 reproduction

See REPORT.md for findings, scope and scoring caveats. All 150 fresh-seed jobs are complete. SUMMARY.json contains paired seed intervals. CHECKS.json records the verification results.

Install requirements.txt, then run `python -B run.py`, `python -B check.py` and `python -B summarize.py` from this directory. Generated results are resumed only when source, stream, specification, hash seed and Python-version receipts match. A different Python version reruns the jobs.

The recovered/ files preserve Claude's code. worker.py adds read-only traces. source/ contains the relevant exported creation/edit commands as provenance data; do not run those shell commands. recover.py reconstructs the recovered source using literal edits, without executing exported shell commands. RECOVERY.json records the original export checksum.

The original 81-question transfer score remains available; the report uses exactly 80 fresh-world questions. Native answer-bit scores have a gold-dependent fallback and do not establish valid compression. These limitations do not directly affect the pre-feedback top-1 predictions.

The new robust-role method, operation-grammar search and GLN integration remain future work. The four-operation repertoire and explicit QA parsing remain supplied scaffolds.
