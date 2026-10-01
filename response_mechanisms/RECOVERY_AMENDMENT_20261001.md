# Workspace-loss recovery: explicit operational amendment

On 2026-10-01 at 21:33:56 UTC, after being asked to increase the cumulative
worker-time cap from 8 to 12 hours to recover lost work without changing the
experimental design or analysis, the user approved increasing the compute quota
to finish the job and instructed regular GitHub backups going forward.

This amendment supersedes only the original **8 worker-hour cumulative cap** with
**12 worker-hours**. The original 12 elapsed-hour limit remains anchored at
2026-10-01 16:40:04.660057 UTC and expires on October 2 at 04:40:04.660057 UTC.
One simulation worker, 300 seconds per completed life, 800 MB worker RSS and
200 MB result storage remain the hard operational bounds. No purchase is involved.

The original scientific source, tests, specifications, parameters, endpoint,
paired-world sample, arm roster, statistical plan, runtime and source lock remain
unchanged. The lock is
`24943155d4b010c8e17746a265803155e13585a7b0b23fe261f3c0bae95a63fe`.
The separate, reviewable `ops/recover_after_reset.py` invokes the original assay
with the same arguments and produces original-schema, original-lock receipts.
It does not import itself into simulation subprocesses or change their source.
The final report must disclose this operational amendment; compliance with the
superseded 8-hour cap must not be claimed.

## Loss and preserved evidence

The last confirmed local count was 202/224 primary lives plus all seven selected
replays. After the executor outage, the project, virtual environment and runtime
files were absent, and old sessions could not resume. A search of permitted local
storage and current Library files found no additional experiment backup. The
verified GitHub checkpoint `901bcddf6e292711be76b72ed7bf1f9dc4b438cf` retained
world 300001's seven primary receipts, all seven replays, the original locked
source, all 21 development pilot lives and the checkpoint ledger.

The restored seven primary and seven replay receipts were revalidated under the
exact original runtime and lock. The replay contents still match excluding the
resource fields, with distinct saved process IDs. There are 217 missing primary
receipts: at least 195 had completed locally before the loss. Their original
outputs and full later ledger are unavailable; they are not counted as recovered.
Recovery reruns those same fixed missing world/arm jobs, without outcome-based
selection or new worlds. No final efficacy analysis was inspected before the loss.

## Conservative accounting and durable backups

The stale remote ledger is preserved verbatim as `PRE_RESET_REMOTE_LEDGER.json`.
The recovery ledger retains its original start, its 14 recorded jobs and its
original charges. An explicitly marked accounting adjustment raises prior total
charged time to **17,245.33994293213 seconds (4.790372206370036 hours)**, the entire
elapsed interval from original launch to confirmed reset at 21:27:30 UTC. This
conservative upper bound includes idle/publication time and maintenance downtime;
it is not a claim of measured active runtime. All recovery jobs and later
unobserved interruptions add to this cumulative total. Budget accounting is not
reset. See `operations/RESET_RECOVERY_20261001.json` for the evidence and limits.

The prior 374.0029-second interruption charge included unobserved downtime; it
was not a measured life runtime. The 300-second limit was observed for completed
jobs at their launcher checkpoints, but the exact active duration of that lost
interrupted attempt cannot now be independently established.

An exclusive supervisor lock, inherited by every simulation child, prevents
duplicate recovery launchers even if a supervisor disappears before its child.
Attempt-unique write-once stdout/stderr files preserve interrupted logs. Existing
receipts are validated and reused, never overwritten. Completed-world queue
records allow regular, independent GitHub checkpoints of inspected synthetic
receipts, current project source, operational records and reports. Only exact
verified remote commits count as durable backups. Computation does not wait for
uploads, and an upload failure must be reported promptly. Final receipt audit,
replay check, locked analysis and an independent operational reconciliation are
required before completion claims.
