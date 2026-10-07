# Operations repair and authorized checkpoint continuation

The user authorized: “Fix this and re-run in the background.” This amendment
does not change candidates, worlds, teaching, clocks, endpoints, sample sizes,
selection rules, statistical decisions, or resource/session limits.

The first dispatcher exited while a temporary atomic-write name disappeared
between `is_file()` and a second `stat()`. No complete science job or life was
committed. Eight workers saved complete-record checkpoints containing 767 W
teaching records. Original code, qualification, source lock, manifest, errors,
status and checkpoint bytes are preserved under `original/`.

Only `dispatcher.py` and `pipeline.py` changed in the execution source map:

- Disk counting uses one stat per enumerated path and tolerates only missing
  entries, while counting existing temporary files. Permission and other I/O
  errors still fail. The scratch/exclusion policy and disk cap are unchanged.
- An active heartbeat deleted when its worker exits is treated as absent.
- Packaging uses the same safe size inventory and excludes uncommitted
  `.pending-` names from the archive.

All 43 operations/protocol regressions passed. The stress case performs 1,024
atomic writes/deletes across eight threads and at least 500 inventory scans.
The original race was reproduced deterministically. This qualification runs
no new teaching trajectory and reuses prior unchanged scientific qualification.

Checkpoint rebinding changes only the execution-source identity and integrity
seal. Every operative array, state digest, cursor and complete record history
was checked unchanged; all 767 saved real native-write decisions were audited.
The immutable original checkpoint remains available. The screen manifest
changes only the execution identity. No committed science evidence is rebound.
The source lock and authorization are updated explicitly, with both identities
recorded in `RECOVERY_AUDIT.json`.

The failed dispatcher did not preserve its wait4 attempt ledger. The eight
worker logs show safe pauses. Their measured attempt CPU is retained. To avoid
undercharging, the resource ledger charges a disclosed conservative aggregate
upper bound: host logical CPU count multiplied by the entire launch-to-final-
exit wall window. This is not presented as a recovered kernel measurement.
Worker-wall charge also uses an upper bound. Recorded online learner timings
remain unchanged, including those from before the repair.

The next launch is an explicit detached `resume=True`, with eight workers and
caffeinate attached. Existing record prefixes continue from their saved cursor;
they are not taught again. The pipeline still screens eight worlds and confirms
at most one candidate on 48 fresh worlds. No scientific failure restarts
automatically; no GitHub upload is authorized.
