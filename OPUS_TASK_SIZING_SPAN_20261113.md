# The sizing learner's span. Revision 2, after a hostile audit (13 November 2026, 12:13 EST by the clock)

> **Status.** Released to one Opus 5.5 worker: admission, then the roster, in one job. Revision 1 (commit 1f23e4a7) was read by a fresh hostile auditor of the same model family, a second reader and not an independent one. It returned eleven confirmed findings; the coordinator reran the evidence for the four that changed the design and adopted all eleven (table at the end). The sizing roster of 84 fits stays held. This is a sizing exercise about one backprop learner; it says nothing about any learning rule.

## Why

Stage 1 of the sizing sweep (`TARGET_TEST_STEP0/sizing/SIZING_STAGE1_REPORT.md`) found that the sweep's GRU, from random weights, copies a byte 12 positions back at 99.7% and learns nothing at 24, 40 or 60. One seed (1104).

**The audit found that this comparison was not at equal dose.** A fixed distance D gives 256/(D+1) repeats per update: 19.7 at D = 12, 10.2 at 24, 4.2 at 60. At D = 12 the learner was still at 8% after 13,714 repeats and passed later. The run at 24 saw about 30,000 repeats in all, and the run at 60 about 42,000 over 10,000 updates. "Nothing at 24" may be a limit of span or of dose. Review 9.112 E drew a consequence for the target test from it; that is open until this job reports.

What hangs on it: arm B's ordinary stream needs the context letter 13 to 18 positions before its answer digits (`sz_stream.py` line 196: context, filler of 10 to 14, cue, `=`, two digits).

## The fixture

Stage 1's, unchanged: `sz_crosscut.fixture_stream` (lines 37 to 44 of `TARGET_TEST_STEP0/sizing/sz_crosscut.py`). Episodes `K <filler of D - 2> = K \n`, K one of 24 letters, chance 1/24. The learner is `sz_learner.StreamLearner`: every next byte a target, credit cut every 256 targets, state carried, backprop, AdamW at 0.001, decay 0.3. Predictions are read from each chunk's forward pass before its update. The code runs on Python 3.11 and later.

New file `TARGET_TEST_STEP0/sizing/sz_span.py`, importing `sz_learner` and `sz_crosscut` unedited:

- **Dose is counted in repeats.** A fixed-D run has **235 × (D + 1) chunks**, about 60,000 repeats at every D (3,055 chunks at D = 12, which is stage 1's dose there).
- **Mixed distances.** Each episode draws D uniformly from M = {4, 12, 24, 40}, from a generator of its own seeded `sizing-span-D|4,12,24,40|<seed>`, so that key and filler draws are those of stage 1's generator. 20,000 chunks, about 60,000 repeats per D.
- **A born start.** E26's backprop births, `MOD97_LOCAL_RULE_20261106/exp45/e26/e26_dev_B_<seed>.pt`: bare state dicts of thirteen tensors that load strictly into `sz_learner.Net` (auditor's `load.py`). Births 73129, 93683, 116089 are paired with stream seeds 1104, 1105, 1106 in that order. Load **in place** with `lr.m.load_state_dict(...)`: the optimizer is built on `lr.m` in the constructor and a replaced module would leave it on dead parameters. The optimizer starts fresh.
- **A permuted start.** The same born file with the entries of each tensor permuted within that tensor (generator seeded `sizing-span-perm|<birth seed>`): the birth's scale without its structure.

## Roster. Stream and model seeds 1104, 1105, 1106

| Block | Start | Distances | Chunks per run | Runs |
|---|---|---|---|---|
| A | random | fixed 4, 12, 16, 18, 20, 24, 40, 60 | 235 × (D + 1) | 24 |
| B | random | mixed M | 20,000 | 3 |
| C | born | fixed 4, 12, 24, 40 | 235 × (D + 1) | 12 |
| P | permuted | fixed 12, 24 | 235 × (D + 1) | 6 |
| D | born | mixed M | 20,000 | 3 |

48 runs, about 348,000 chunks. At stage 1's 0.087 to 0.093 CPU-seconds per chunk that is 8.4 to 9.0 CPU-hours. If the forecast after admission passes twelve, stop and report. **Distances of hundreds of bytes (the lines) are not in this job:** at equal dose one run at 384 costs over two CPU-hours. They are briefed only if block A passes at 60.

## Admission, before any roster run

1. Print Python, torch and the host architecture.
2. Through `sz_span.py`, fixed 12, seed 1104, 3,000 chunks: on stage 1's host class the last-quarter counts equal `runs/crosscut_diag_D12.json` (641 of 643 and 13,011 of 13,072). On another class, last-quarter accuracy must be at least 95%; otherwise stop.
3. The mixed stream: each repeat byte equals its key at its own D; a one-element set gives a stream byte-identical to stage 1's fixed stream; episodes per D are counted.
4. A one-off `sz_verify_born.py`, the only file allowed to import the arithmetic harness: each birth file's SHA-256 equals `weights_sha256` in its receipt, and on one birth batch the logits of `sz_learner.Net` and of `arch26.forward` agree to 1e-4 (the auditor measured 9.5e-6).
5. The permuted start: every tensor keeps its sorted values and changes its order.

## What every run records

Right and total by D, for crossing and not-crossing instances apart, in **windows of 250 chunks**; the same for the first chunk before any update; the repeats seen so far at each window; and the first window at which accuracy reaches 50%, as a count of repeats of that D (**repeats to 50%**).

## Classes, per run and per distance, on the last tenth of the stream

- *pass:* at least 100 instances and at least 50% right;
- *partial:* at least 10% and below 50%;
- *chance:* below 10%.

A run that passed in an earlier window and does not pass in the last tenth is "not held" and counts as not passing. **A block's span** is the largest D such that it and every smaller D run in that block pass in at least two of three seeds. One of three is "mixed" and supports no reading.

## Forecasts, registered (copy unchanged into `SZ_SPAN_FORECASTS.md` before the roster)

- **F1** Block A passes at 18 in at least two of three seeds. About 50%.
- **F2** Block A passes at 24 in at least two of three. About 40%.
- **F3** Block A passes at 60 in at least two of three. About 15%.
- **F4** Block B passes at 24 or at 40 in at least two of three seeds where block A does not. About 25%.
- **F5** At D = 12, repeats to 50% are more from the born start than from random weights in at least two of three pairs. About 75%.
- **F6** Block C passes at 40 in at least two of three. About 20%.

## Readings, fixed now

- **Stage 1's "nothing at 24".** Block A passing at 24: the earlier failure was dose, and the statements built on a span of 12 (review 9.112 E, the sizing brief's revision 4) get an erratum. Block A at chance at 24 with 12 passing: the limit is span at this dose, now on three seeds.
- **Arm B's ordinary stream.** Block A's span at 18 or more: arm B's held figures in the roster carry no span caveat. A span of 12 or 16: they are reported as limited by span. Fixed 12 not passing: arm B is not readable from random weights.
- **Mixed against fixed** (B against A at 24 and 40, about 60,000 repeats per D in both). Mixed helps only if B passes where A does not, in two of three seeds. Where both pass, repeats to 50% are compared and read only if they differ by a factor of two or more in all three seeds. Where A passes and B does not, mixing hurt, and that is said.
- **Born against random** at every shared D (4, 12, 24, 40), by class and by repeats to 50%, with the permuted start beside them at 12 and 24. A born advantage is read as the birth's structure only if the permuted start does not share it. On the Mac at D = 4 the auditor saw the born start learn two to three times slower than random weights (rerun by the coordinator on one birth: 215 of 3,200 against 3,200 of 3,200 in the fourth quarter of 300 chunks); a born failure at 24 or 40 is therefore read against its own D = 4 and 12 before anything is said about span.
- **What a born start cannot show here.** A failure does not show a birth cannot help: the birth was trained on answer targets, on items of 10 to 22 bytes with the state reset per item, never saw six of the eight filler bytes or the newline, and meets a fresh optimizer at a first loss of 10 to 21 nats.

## What this cannot say

One fixture, a copy over random filler with 24 keys. One learner and recipe, backprop only. Nothing about the lines' distances. E20-B's GRU holding operands across 40 letters (end of life, `exp45/e20b/e20_life_G1_73129.json`, add97 at k = 40) was digits held across letters it was trained to ignore, under answer targets with resets; this fixture is a different demand, and neither result predicts the other.

## Running it

One host, at most three processes on four cores, one thread each. New files only, under `TARGET_TEST_STEP0/sizing/`; records in `runs/span/`. Own branch; nothing to the default branch. Report: one table per block (D by seed: class, last-tenth figures, repeats to 50%), the spans, the forecasts' outcomes, CPU, and what was unclear. No conclusion about any rule or about the roster.

## The audit and what was done with it

| # | Finding | Coordinator's check | Done |
|---|---|---|---|
| 1 | The born start learns D = 4 slower than random; the roster could not see it | `short.py` rerun, random and birth 73129 | born fixed 4 and 12 added; reading conditioned on them |
| 2 | "First quarter passes" could never happen; quarters unequal | birth alphabet read from the report | first chunk before any update; windows of 250 chunks |
| 3 | Span confounded with repeats per update | `counts.py` rerun; stage 1's records | dose in repeats everywhere; mixed set cut to four distances |
| 4 | Fixed 16 does not cover arm B's 13 to 18 | `sz_stream.py` line 196, filler (10, 14) | mark moved to 18; 16, 18, 20 all run |
| 5 | No random 40 and 60 at new seeds; no control for trained weights | roster read | added to block A; permuted start added |
| 6 | Stream seed for born runs unspecified; load trap | `StreamLearner` constructor read | pairing stated; load in place |
| 7 | One-element set not byte-identical if D shares the generator | auditor's output | separate generator and seed string |
| 8 | Span definitions and F5 had holes | read against the draft | block span with monotone clause; classes; M2 dropped |
| 9 | Wrong section cited; the 40-letter result is a different demand | review line 2470 | cited as 9.112 E; differences stated |
| 10 | Admission item 4 infeasible as worded | — | `sz_verify_born.py` |
| 11 | Admission item 2 had no teeth off-class | — | 95% floor |
