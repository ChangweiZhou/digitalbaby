# Draft for hostile audit: the sizing learner's span (13 November 2026, 12:05 EST by the clock)

> **Status.** Draft. **No worker starts from this file** until a hostile audit has read it and the coordinator has answered. It replaces the item "added before the roster" in revision 4 of `OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md` and widens it. The roster of 84 fits stays held. Implementation and runs are an Opus 5.5 worker's. The auditors are fresh sessions of the same model family: second readers, not independent ones. This is a sizing exercise about one backprop learner; it says nothing about any learning rule.

## Why

Stage 1 of the sizing sweep (`TARGET_TEST_STEP0/sizing/SIZING_STAGE1_REPORT.md`) found that the sweep's GRU, from random weights on its own training path, copies a byte 12 positions back at 99.7% and learns nothing at 24, 40 or 60, at 60 even after 10,000 updates. One model seed (1104), one fixed distance per run. Three things about the target test hang on this and none is measured:

1. Arm B's ordinary stream needs the context letter 12 to 16 bytes back. Nothing between 12 and 24 was run.
2. The lines stand a median of 411 to 734 bytes before the probe that needs them.
3. E20-B's GRU held operands across 40 letters after a birth (review 9.110). Whether a fixed long distance from random weights is the hard case, and mixed distances or a born network the easy one, is not known. Review 9.110 E said to find out before the design is changed.

## The fixture

Stage 1's, unchanged: `sz_crosscut.fixture_stream` (`sed -n '36,45p' TARGET_TEST_STEP0/sizing/sz_crosscut.py`). Episodes `K <filler> = K \n` back to back, K one of 24 letters, the byte after `=` repeating the key D bytes earlier, chance 1/24. The learner is `sz_learner.StreamLearner`: every next byte a target, credit cut every 256 targets, state carried, backprop, AdamW at 0.001 with decay 0.3. Predictions are read from each chunk's forward pass before its update.

New, in a new file `TARGET_TEST_STEP0/sizing/sz_span.py` that imports `sz_learner` and `sz_crosscut` unedited:

- **Mixed distances.** Each episode draws its D uniformly from a set. Set M1 = {4, 8, 12, 16, 20, 24, 32, 40, 60}. Set M2 = {12, 24, 48, 96, 192, 384}. Every instance is recorded with its D.
- **A born start.** The weights of E26's backprop births, `MOD97_LOCAL_RULE_20261106/exp45/e26/e26_dev_B_<seed>.pt` for 73129, 93683, 116089, loaded with `strict=True` after their SHA-256 is checked against the birth receipt. The sizing learner has E20-B's G1 shape (stage 1's `sz_verify_shape.py`), and E26's network has the same thirteen tensors. The optimizer starts fresh. If a file does not load strictly, stop and report.

## Roster. Model seeds 1104, 1105, 1106 for random starts

| Block | Start | Distances | Chunks | Runs |
|---|---|---|---|---|
| A | random | fixed 12, 16, 20, 24 | 3,000 | 12 |
| A' | random | fixed 24 | 10,000 | 3 |
| B | random | mixed M1 | 10,000 | 3 |
| C | born | fixed 24, 40, 60 | 3,000 | 9 |
| D | born | mixed M1 | 10,000 | 3 |
| E | random and born | mixed M2 | 10,000 | 6 |

36 runs. Stage 1's runs cost 0.087 CPU-seconds per chunk (`runs/crosscut_*.json`: 256 to 270 s at 3,000 chunks, 934 s at 10,000), so about 5.5 CPU-hours. If the forecast after admission passes eight, stop and report.

## Admission, before any roster run

1. Print Python and torch versions; Python 3.12 or newer is needed.
2. Through `sz_span.py`, fixed D = 12, seed 1104, 3,000 chunks reproduces `runs/crosscut_diag_D12.json`: the same instance count and the same four last-quarter counts (643, 641, 13,072, 13,011) on a host of stage 1's class. On another class, report the differences and continue.
3. The mixed stream: every episode's repeat byte equals its key at its own D; the count of episodes per D over the stream is given; the stream for a one-element set is byte-identical to stage 1's fixed stream.
4. The born start: the hash check, the strict load, and the born network's scores on E26's sealed panel reproduced through its own scorer before the fixture touches it (so the file is shown to be the birth). State which fixture bytes (the 24 key letters, the eight filler bytes, `=`, newline) occurred in the birth's data and which never did.

## The criterion, per run and per distance

In the last quarter of the stream: at least 100 instances of that D, at least 50% right. Crossing and not-crossing instances are reported apart; the criterion is on all instances, since stage 1 showed the cut is not what limits. **A run's span** is the largest D that passes with every smaller D of that run passing. **A block's span** is the largest D at which at least two of its three runs pass. First-quarter accuracy is given beside every figure, so that what a born start already does is told from what it learns.

## Forecasts, registered (copy unchanged into `SZ_SPAN_FORECASTS.md` before the roster)

- **F1** Random start, fixed 16 passes in at least two of three seeds. About 55%.
- **F2** Random start, fixed 20 passes in at least two of three. About 35%.
- **F3** Block B's span is at least 24. About 50%. It reaches 60: about 20%.
- **F4** Born start, fixed 40 passes in at least two of three. About 45%.
- **F5** No block passes at 384. About 85%.

## Readings, fixed now

- **Arm B's ordinary stream.** Fixed 16 passing from random weights in two of three seeds: arm B's held figures in the roster are read without a span caveat. Otherwise they are reported as limited by the learner's span at 16. Fixed 12 failing at a new seed: stage 1's 99.7% was one seed's, and arm B is not readable from random weights at all.
- **Mixed against fixed.** Block B is read against block A' (the same dose). Mixed lifts the span only if B passes at 24 or beyond in two of three seeds and A' does not. Then a stream in which short dependencies come with long ones is an option worth a brief; nothing more.
- **Born against random.** Blocks C and D against A, A' and B at the same distances. A pass at 40 or beyond from born weights where random fails gives a measured answer on span to "what may a learner bring", and only on span: it says nothing about acquiring held pairs. If first-quarter accuracy already passes, the birth brought the copy and it was not learned here; say which.
- **The lines.** Only M2's 384 speaks to them. No block passing at 384: the lines stay unreadable in arm B for this learner, and the lines cannot become the base arm without a stream that puts each line near its use. A pass at 384: the lines' fits are readable for that start only.
- **A fixed distance failing inside a mixed set that passes it** (or the reverse) is reported as it stands; no span is then claimed for that block beyond the last distance on which both agree.

## What this cannot say

One fixture: a copy over random filler, 24 keys. One learner and recipe, backprop only. The born start comes from answer-target births on arithmetic with a fresh optimizer and a different decay (0.25 at birth, 0.3 here); a failure from it does not show a birth cannot help. Three seeds per block.

## Running it

One host, at most three processes on four cores, one thread each. New files only, under `TARGET_TEST_STEP0/sizing/`; records in `runs/span/`. Own branch; nothing to the default branch. Report: one table per block (D by seed, first and last quarter, crossing apart), the spans, the forecasts' outcomes, CPU, and what was unclear. No conclusion about any rule or about the roster.

## Questions for the audit

1. Is "at least 100 instances" reachable for every D of M2 in a last quarter of 640,000 bytes?
2. Does a born start with a fresh optimizer and decay 0.3 test anything, or does the first thousand updates erase the birth?
3. Is 50% the right pass mark when stage 1's outcomes were 99.7% or chance, with nothing between?
