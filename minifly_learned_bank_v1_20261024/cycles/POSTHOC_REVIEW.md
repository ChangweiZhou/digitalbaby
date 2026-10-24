# Post-lock review during the main run (24 Oct 2026)

A cross-session review (the session distilling MiniFly research-loop skills) checked the pushed branch at f33d8dee and flagged three gaps in the locked pipeline. All three are confirmed:

1. **`summarize.py` fails open on missing arms.** `gate_oracle` is `None` when ORACLE is absent, and `gates_ok` tests `is not False`, so a missing ORACLE passes. The T-COPY gate behaves the same when no COPY rows exist. In the main run every locked arm is present, so this cannot change the outcome there, but the rule as coded does not enforce it.
2. **`summarize.py` does not check the world count.** `overall = "supported"` whenever every present world passes, so 1 of 1 reads "supported". The rehearsal's `results/c3_rehearsal/SUMMARY.json` says "supported" for the single development world D3. That label is **not a verdict**: D3 is a development world and the rule needs the four locked main worlds (as `CYCLE3.md` says).
3. **The lock did not cover the runtime closure.** `LOCK.json` hashes `fastmem.py`, but not what it imports: `bootstrap.py`, `minifly_a_v3/src/stores.py`, `paths.py`, and 44 modules plus the birth data in the untracked extracted package `minifly_a_v3/package/`.

## What was done (the locked files were not edited)
Editing a hashed file mid-run would also make `run.py` refuse to start the T-COPY phase, which re-checks the lock. So `summarize.py` and `LOCK.json` stay as locked. `posthoc_check.py` was added instead, clearly labelled POSTHOC; it is not part of the lock and is never used by `run.py`. It reports alongside the locked summary:

* **Runtime closure.** It imports `lb1` and lists every repository file loaded, plus `run.py`, `summarize.py`, `main.sh` and the native birth data (`FULL151_BIRTH_FINGERPRINT.json`, `FULL151_CANONICAL_B.npz`).
  * Tracked files are verified against the git blob at the lock commit 1ce1eadb.
  * Files in the untracked package are verified against `minifly_a_v3/SOURCE_LOCK.json`.
  * Result (`POSTHOC_CLOSURE.json`): **55 files, all verified**. 9 match git at the lock commit and 46 match SOURCE_LOCK. The native birth additionally asserts its own fingerprint and canonical-B digest at every model construction.
* **Fail-closed verdict.** Required, or the result is `INVALID_FOR_VERDICT`:
  * exactly the locked main worlds M1–M4;
  * every locked (task, arm) history present, with no unlocked ones;
  * locked settings and the right record count in every history;
  * a final post-rest checkpoint.

  Every gate must actually be evaluated: ORACLE ≥ 0.90, FROZEN codes unchanged at every checkpoint, and all three T-COPY arms at system held ≥ 0.30. Balanced accuracies are recomputed from the raw per-prompt outputs with independent code. The outcome is compared with the locked `SUMMARY.json`; any disagreement is reported as a finding.
* Applied to the rehearsal (`results/c3_rehearsal/POSTHOC_VERDICT.json`): `INVALID_FOR_VERDICT`, because the worlds are not the locked main worlds. The per-world clauses for D3 agree with the locked summary (LEARN 0.944, FROZEN 0.361, SHUF 0.500, ORACLE 1.000). The overall label disagrees ("supported" against INVALID), which is exactly gap 2.

The main-run report will give both outputs: the locked rule as pre-registered, and the POSTHOC fail-closed check.
