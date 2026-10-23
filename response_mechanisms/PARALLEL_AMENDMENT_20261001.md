# Explicit concurrency amendment: one to two workers

At 2026-10-01 23:02:33 UTC, the user explicitly approved changing the locked
one-worker execution limit to two workers to shorten recovery, with unchanged
seeds, models, statistical analysis and approved cumulative compute budget.
This amendment documents that operational protocol exception. It does not claim
compliance with the original one-worker cap after the transition.

The original scientific source, source lock, runtime versions, numerical threads
per process, arms, 32-world roster, parameters, endpoint and statistical analysis
remain unchanged. The previously approved cumulative 12 worker-hour cap includes
all retained original charges, the conservative lost-work adjustment, all recovery
jobs and any interrupted attempts. The original elapsed deadline remains
October 2 at 04:40:04.660057 UTC. Per-life 300 seconds, per-worker 800 MB RSS and
200 MB results limits are preserved. No purchase or new persistent access is used.

The separate operational coordinator reserves at most two disjoint missing fixed
jobs, with a complete 300-second reservation for each. It charges the sum of both
running and completed job durations, never elapsed time just once for simultaneous
jobs. Every job uses the same original locked assay command, unique logs and
write-once receipt path. The original lock is intentionally not rewritten.

Approved operational coordinator SHA256:
`f1a775bbecbad25d0adafbd2dcf96cebe3d1a49ee86176cfd69555d16511f06a`

An independent code review reproduced and blocked an initial failure-admission
race during slow receipt validation. The revised candidate surfaces child failure
immediately, directly polls children before admission, drains newly completed
children after validation, and repeats checks after verification/reservation and
before spawn. All 24 isolated mocked tests passed independently. No extra
experimental lives were run for those tests.

Production launch requires both explicit approval and independent review records
bound to that exact operational source hash. A shared exclusive supervisor lock
is inherited by every child, preventing a surviving old or new worker from being
overlooked after a supervisor disappears. Process listings are only supplementary
because executor PID namespaces differ.

The transition waits for the old current life to finish, durably backs up the
completed records and counters, interrupts the old supervisor immediately, and
requires the inherited lock to be free. Because the old launcher has no drain
interface, a newly started next attempt may be interrupted; its saved-start
interval is conservatively charged and the missing receipt is rerun. No completed
receipt is discarded or replaced. The before-image ledger and transition evidence
are retained. The new active_jobs ledger must not be resumed with the obsolete
single-active-job launcher.

Regular GitHub checkpoints and private recovery backups continue independently.
Publication queues expose complete contiguous world prefixes even if two jobs
finish out of order. Final receipt/replay verification, unchanged locked analysis,
and a separate operational reconciliation remain necessary before any completion
or efficacy claim.
