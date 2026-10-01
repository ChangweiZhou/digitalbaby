# 2026-10-01 host-loss recovery amendment

The cloud execution service disconnected at about21:16UTC and the restored
workspace no longer contained this checkout or its runtime. No continuation from
those processes is claimed. The last successful observation at21:15:06UTC had757
validated receipts,756 locally checkpointed, active_wall_s121717.77187667531 and
worker_s483839.86000360106. It had four active jobs and no reported failure.

The latest repository commit c9b0ba3f448baab3c8d0bc1c8ed8569e53d7653f preserved689
native receipts (all13 arms through190053) and a lossless archive forR3/190054.
All690 surviving receipts and all146 locked files were freshly validated under
the exact pinned runtime at21:34UTC.67 previously accepted trajectories have no
recoverable bytes: all arms190054–190058 exceptR3/190054, plusR1/R0/R0_signed190059.
Any unobserved later work is unknown. Recomputed files will be labelled as
reconstructions, never as the original lost bytes; their original hashes were
not recovered. No endpoint scores were inspected in this recovery.

The user authorized this recovery at21:33:56UTC in response to the explicit
request to regenerate those67 with identical locked seeds/code and retain the
original computation in the budget. The user also required regular GitHub
uploads and increased compute quota to finish. The existing90h active-wall and
360h worker-hour caps remain unchanged because they still provide ample room;
no scientific lock, seed, roster, equation, stopping rule, or analysis changes.

Restore690 exact receipts, generate only missing142 world-arms (including the67
reconstructions), preserve all surviving failure history, and retain cumulative
counters at least as large as the last observation. Reserve the entire uncertain
interval conservatively from21:14:00UTC (before the last saved observation) through
completed restart validation at four worker-seconds per second. This reservation
is not measured computation. Do not reset accounting to the stale remote ledger.

The superseded remote-blocking persistence is replaced by an additive local
journal and independent GitHub archive publisher. Every4 accepted receipts and
at stop, the frozen driver creates an immutable fsynced local checkpoint. Only a
separate publisher may acknowledge verified remote receipt hashes. Regular
publication cannot block ordinary compute. If8 accepted receipts lack verified
remote backup at a checkpoint, new dispatch stops as an infrastructure failure;
in-flight jobs finish with frozen live-budget enforcement. Resume only after
backup verification catches up. Integrity and per-job budget failures still need
explicit clearance; cumulative budget exhaustion remains final. Local fsync is
not off-host durability. Nothing is called backed up until the remote commit and
original receipt bytes or exact content hashes have been verified.

Final requirements remain832 fully audited science receipts, the literal causal
gate,52 predeclared independent mechanism replays, and unchanged m=26 analysis.
PackageB V2 E3 remains invalid for combined claims;P3 remains unavailable.
