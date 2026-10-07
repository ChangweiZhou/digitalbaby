# LINK completed results

Verdict: **SCREEN_NO_ADVANCE_RETAIN_ERROR**. Eight paired screening worlds completed 16 full world-assay jobs, 48 native lives and 31,104 records. The final audit accepted the experiment and the result archive is complete. The 64-world confirmation was not launched.

LINK's raw never-taught choice was 45.83%, below the 50% screen requirement. Old-key E1 was 94.27% versus ERROR's 96.35%, a 2.08 percentage-point loss exceeding the 2-point limit. LINK's paired causal increment relative to ERROR was +18.75 points, but this eight-world exploratory signal does not override the failed gates. This is a valid budget stop for the locked recipe, not falsification of the entire mechanism family.

Read [REPORT.md](../REPORT.md), [screen summary](../results/science/screen/SUMMARY.json), [final audit](../results/FINAL_AUDIT.json), [three-cycle report](../THREE_CYCLE_REPORT.md), and [execution qualification](../ops/run_20261007/QUALIFICATION.json). ERROR/LINK/PERM are readout conditions on qualified shared native trajectories, not three times the number of lives.

## Full evidence archive

The exact completed ZIP is split into four ordered binary parts, each at most 48 MiB. All complete science job and branch receipts are inside this archive; they are not duplicated as loose Git blobs.

From this publication directory:

```bash
cat archive/RESULT_BUNDLE.zip.part* > RESULT_BUNDLE.zip
unzip RESULT_BUNDLE.zip -d extracted_results
```

[ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json) records the original archive size and already-recorded package digest. [SOURCE_SNAPSHOT.json](SOURCE_SNAPSHOT.json) maps the 138 locked dependency files to exact copies under `locked_sources/`, preserving their workspace-relative layout for source review. Original locks retain the execution-machine paths. This is a scientific results/source handoff, not permission to rerun trajectories or a requalified portable execution kit.

Historical records saying publication was not requested describe the earlier authorization correctly. [AUTHORIZATION.json](AUTHORIZATION.json) records the later explicit GitHub request. Scientific results and frozen locks were not rewritten.
