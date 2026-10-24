# V78 compact results and reproduction

The report is `REPORT.md`; the scientific definition is `PROTOCOL.md`. `config.json` fixes all settings. There is no promoted model checkpoint. The 5,177-KC parent, fixed reader and V76 calibration remain unchanged external assets. `MANIFEST_RESULTS.json` gives path, size and SHA-256 references, including generated inputs and raw verification outputs that are deliberately absent from this ZIP.

From `/Users/pencilbard/Documents/second_try`, with the original project assets available:

```sh
_omnibus10_env/bin/python minifly/iteration23/runner.py verify
_omnibus10_env/bin/python minifly/iteration23/reporting/analyze78.py
_omnibus10_env/bin/python minifly/iteration23/reporting/audit78.py
_omnibus10_env/bin/python minifly/iteration23/reporting/metadata78.py
.venv/bin/python minifly/iteration23/reporting/figures78.py
_omnibus10_env/bin/python minifly/iteration23/reporting/verify_bundle.py minifly_V78_iteration23_results_20260914.zip
```

`runner.py prepare` validates an existing freeze without regenerating inputs. `runner.py run` resumes only missing jobs and validates completed receipts; it uses two subprocesses and caffeinate on macOS. A genuinely new reproduction can use a new project copy with the same frozen source, configuration, parent assets and generated inputs; do not delete existing scientific receipts or rewrite the release.

Final CSV tables contain aggregate metrics, paired contrasts, the 15-test corrected primary family, trace-fate counts and 192 diagnostic acceptance checks. `primary_history.csv` contains the final paired history-level contrasts used for inference; it is not a raw simulation dump. Raw victim arrays, bout histories, PN banks, checkpoint files and caches remain local and are referenced only. Figures have PNG/PDF versions.

M12 and native anchors reproduce inherited challenge semantics. BATCH6/HALF12/HALF_BATCH6 are fixed-request evaluator controls; ONLINE6 is the online two-bout queue. MIXED uses the native suffix for every condition, so its counts are not directly interchangeable with V77. Full/half source-vector replay is matched within each timing pair, but online-vs-replay requests need not have equal magnitude. All alpha-fast forecast diagnostics hold present activity fixed.
