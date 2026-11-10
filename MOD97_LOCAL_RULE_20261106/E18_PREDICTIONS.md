# E18 predictions: births with a next-byte target at every bank (10 November 2026, written before any birth)

**Brief:** `OPUS_TASK_E18_LOCAL_HEADS_BIRTHS_20261110.md`. This session runs the job whole: arms B0–B3, seeds 73129, 93683 and 116089, twelve births. The review paragraph is `E18_REVIEW_SECTION_9_71.md`.

**Harness:** commit 8d037ebb, the commit the brief ships with. No harness file is changed by this job.

**Before launch** (checked 16:15 UTC, 10 November): none of the twelve birth files `exp45/e17_dev_bank2_nbd{_bT_tall,_lh,_lh_bT_tall,_lh_bT}_{73129,93683,116089}.{json,pt}` existed.

## Check

`python3 e18_check.py` was run at 16:09 UTC, before any dry run. Its output is identical to `exp45/E18_CHECK.json`:

- every number is 0.0: the gradient difference from a3c0b675, the shared initialisation, the four leak terms and the two write-source terms;
- `head1_out_switch` is true;
- the new parameters are `l2a`, `lma` and `l3a` (weight and bias).

A copy of the output is `exp45/archive_e18_dry/cloud_20261110/e18_check_output.txt`.

## Runs

```
E16_ARCH=bank2 E16_BOOKDECAY=off <extra> python3 exp16_integrated.py dev <seed>
```

| Arm | Extra environment | Birth file tag |
|---|---|---|
| B0 | `E16_BIRTHRULE=T E16_TRACEUP=all` | `_bT_tall` |
| B1 | `E16_HEADS=local` | `_lh` |
| B2 | `E16_HEADS=local E16_BIRTHRULE=T E16_TRACEUP=all` | `_lh_bT_tall` |
| B3 | `E16_HEADS=local E16_BIRTHRULE=T` | `_lh_bT` |

- **Launch:** the twelve births start together, each detached (`setsid nohup`), one thread each, on four cores. Each birth runs 24,576 updates on the standard development stream.
- **Logs:** written to the session's scratch folder, then copied to `exp45/logs/e18_birth_<arm>_<seed>.log`.
- **Commits:** each birth's `.json` and `.pt` (the `.pt` with `git add -f`).

## Predictions (from the brief, copied verbatim; not to be changed)

"Hard mean" is the mean of add89 and sub89 at k = 12 at the final checkpoint, from the main head unless the first bank's head is named. The reference hard means are 91.2 / 94.5 / 90.3. These were written by the coordinating session after its own dry runs of 512 updates, in which B0, B1 and B2 stood level with the default birth and B3 behind; nothing longer has been run.

- **P18.1 (the first bank alone stays a one-bank learner)** In B1, the first bank's own head ends with add89 below 45 at k = 12 in all three seeds.
- **P18.2 (the second bank can build on it)** In B1, the main head's hard mean is at or above 70 in at least two of the three seeds, and below the reference in all three. Held at about 55%: the first bank was never asked to serve the second.
- **P18.3 (weight signs with the relay give birth)** In B0, the hard mean is at or above 0.80 of the reference in at least two of the three seeds.
- **P18.4 (the relay is worth more than exact weights)** B0's hard mean is at least 5 points above B1's in at least two of the three seeds.
- **P18.5 (signs cost little once the relay is gone)** B2's hard mean is within 10 points of B1's in at least two of the three seeds.
- **P18.6 (the cheapest version fails)** In B3 the hard mean is below 40 in all three seeds, and its max89/min89 mean is at least 20 points below B2's: the layers above the first bank's write get credit from the current step only.

**How I will score them:**
- "max89/min89 mean" is the mean of max89 and min89 at k = 12 at the final checkpoint, from the main head.
- P18.4 compares B0 and B1 seed by seed.
- P18.5 is |B2 − B1| ≤ 10, seed by seed.
- P18.6 compares B3 with B2 seed by seed, and its two clauses must both hold in all three seeds.

**Reading, written in the brief before the result:**
- If B1 comes within a few points of the reference, the relay between banks buys little on this task, and a learner with a target at every bank becomes the base design. The next step is to make each bank shallow enough that its credit is forward-only.
- If B1 falls far below while B0 holds, what backprop buys here is the relay itself, and a next-byte target at each bank is not enough. The first bank then has to be told what the second needs by some other pathway (for example a top-down prediction compared locally with what arrives).
- P18.5 and P18.6 price the two cheaper versions of the within-bank rule.

The report will state which case obtained, seed by seed.

## Dry run

The four commands were run once on seed 73129 with `E11_DEV_UPDATES=512`, all four together, at 16:10 UTC. Each finished without error in 1 min 40 s to 2 min 45 s of wall time. The files are in `exp45/archive_e18_dry/cloud_20261110/`.

Accuracy (%) at k = 12 after 512 updates, from the main head. The owner's dry runs (`exp45/archive_e18_dry/`) are shown for comparison; they are levels to compare, not digits, since the two machines do not reproduce bit for bit.

| Arm | add89 | sub89 | max89 | min89 | CPU-s | Owner: max89 / min89 (%), CPU-s |
|---|---:|---:|---:|---:|---:|---|
| B0 | 2.0 | 1.3 | 19.7 | 25.7 | 81 | 19.7 / 18.3, 109 |
| B1 | 1.7 | 1.7 | 19.7 | 22.7 | 102 | 19.7 / 22.0, 124 |
| B2 | 1.7 | 1.7 | 18.7 | 20.0 | 111 | 19.0 / 24.0, 133 |
| B3 | 0.7 | 0.0 | 7.0 | 6.3 | 125 | 5.7 / 6.3, 143 |

Every score is the count of exact answers out of 300 at k = 12, shown as a percentage. The owner's figures are taken from their logs the same way.

The pattern is as on the owner's machine: B0, B1 and B2 stand together, and B3 is behind on max89 and min89.

## Run identity

| Item | Value |
|---|---|
| Machine | `Linux vm 6.18.44-fc-v338 #1 SMP @0 x86_64 x86_64 x86_64 GNU/Linux`; 4 cores; Intel Xeon at 2.10 GHz |
| Software | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6; one thread per process (`exp11_cell.py` sets it) |
| Harness | 8d037ebb |

Each process's environment will be read from `/proc/<pid>/environ` while it runs and recorded in the report.

## Forecast

A 512-update dry run is mostly one checkpoint's scoring, so it cannot price the training updates by itself. On this container the dry runs cost less CPU than the owner's (81–125 against 107–143 CPU-s), which does not mean the births will be cheaper.

So I timed the updates directly. `exp45/archive_e18_dry/cloud_20261110/e18_time_updates.py` copies the birth's setup, writes nothing, and times 256 updates after 16 warm-up updates, plus one checkpoint's scoring. All five timings ran at once on four cores.

| Arm | CPU-s per update | Scoring per checkpoint | 24,576 updates + 6 checkpoints |
|---|---:|---:|---:|
| Default birth (reference) | 0.158 | 11.8 s | about 3,950 |
| B0 | 0.161 | 11.5 s | about 4,030 |
| B1 | 0.202 | 15.1 s | about 5,060 |
| B2 | 0.218 | 14.7 s | about 5,460 |
| B3 | 0.245 | 15.6 s | about 6,120 |

The default birth's rate here, 0.158 s per update, is about three times the owner's, which I estimate at about 0.05 s from their full birth of 1,730 CPU-s. This agrees with the brief's "roughly half that speed" for training.

**Expected:**
- about 62,000 CPU-s for the twelve births, plus a few hundred for the final silenced and first-head scoring;
- about 4.4 h of wall time on four cores;
- the last birth (a B3) finishing near 21:00 UTC if launched at 16:30.

The brief's estimate is about 55,000 CPU-s and 3.8 h. Scoring a trained model may cost more than scoring a fresh one, so the true figure could be somewhat higher.

## If the container restarts

A restart kills every process. A birth killed before it writes its file has not finished. It will be launched again from scratch with the same command and seed, and the report will say so. No finished birth is rerun.

## Push target

This session can push to `codex/minifly-consolidated-20261023`, so it commits there, pulls with `--rebase` before each push, and never forces.
