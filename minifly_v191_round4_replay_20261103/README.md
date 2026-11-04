# Round 4 recovery from the session export

Recovered source at the Round 4 publication event (event index 1601), with earlier source ancestors and selected original console outputs. The export contained the initial file bodies and subsequent edits. Recovery applied literal text operations to the recorded bodies. It did not run any shell command or recovered code from the attachment.

- Main Round 4 files: integ.py, bytecyc4.py, addcyc.py, addsym.py.
- Ancestors and dependencies: rep.py, cyc.py, bytecyc.py, bytecyc2.py, bytecyc3.py. The pinned public V39/V38 dependencies are in dependencies/ and were recovered separately from GitHub commit b0e537a2.
- RECOVERY_MANIFEST.json records session checksum, zero-based event indices, event IDs, per-file SHA-256 and replacement counts. All nine session-derived Python files parsed successfully.
- SESSION_OUTPUT_EXCERPTS.json contains the recorded familiarity, reliability and addition outputs. The relation transcript reports comparison 0.972 and parity 0.993 on held pairs, from the original session, with no new execution here.

The recovered files preserve the original absolute /tmp imports. A future portable replay needs explicit path adapters and a pinned environment. The session export did not supply the complete redirected result files or experiment manifests, so this is source recovery with transcript excerpts, not a byte-for-byte archive of the original output directory.

The recovered relation source updates both signed successor maps throughout the observed stream and uses a first-encounter exponential reliability update with rate 0.1. V190-R1 froze the maps after background and used cumulative vote counts. The recovered Round 4 relation script scores the first output byte and checks a newline prediction using its bigram table, which differs from V190-R1's integrated own-byte/EOS endpoint. Preserve the two runs separately.
