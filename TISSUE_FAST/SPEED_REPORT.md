# Speed of the tissue's program: what was found, what was changed, what it gives (21 November 2026)

Development level. Measured on this session's machine (4 cores, gcc 13, Python 3.13, numpy 2.5), seed 73129, E63's stream of stage A. One run a figure unless stated; runs made side by side on the 4 cores are marked "side by side" and are rough.

## 1. Why the "two minutes" of 19 November did not hold

The figure "262,144 bytes in 1.2 minutes" (0.271 ms a byte) was measured on the kernel `ev2.c`. Its own design note calls it "timings of the design, not of the tissue": it had shortened cell dynamics, five input cells driven by hand, no making or giving up of connections, no learning, no records, and a result that depended on the order of the cells. The program written afterwards (`TISSUE_EVENT/` of `claude/e63-tissue-event-017dSE`) is the full tissue and cost 1.46 ms a byte for the first form here (core only), 3.98 for the coincidence form, 0.17 for the tissue whose strengths do not change, and about twice that with E63's records.

Measured causes, first form:

1. **The tissue never settles within a byte.** About 800 of the 2,048 cells are computed at every one of the 32 steps of a byte, nearly as many at step 32 as at step 1 (25,531 cell computations a byte against 6,846 in the kernel).
2. **Of those, 91% are cells that are silent before and after**, with threshold 0 and an arrival below the output floor: their output is 0 and stays 0. They stay on the list only because their state halves towards its arrival and the program takes a cell off the list only when its state no longer changes in any bit.
3. **About 1,100 senders pass on a change each byte, each over about 94 connections** (80,000 to 96,000 connection visits a byte). This is set by the tissue, not the program.
4. **94 connections a cell were never chosen as a number of connections.** The neighbourhood radius 1/8 puts about 94 cells within reach; a cell makes a connection every 128 steps while its threshold is 0 and its average activity is below a* = 1/8; the measured average is about 0.006, so every cell connects to everything within reach. The brief had forecast about 12 (not measured).
5. **E63's records cost as much as the core**: every 128 bytes the counts and distances read all 190,000 connections out of the core and build Python sets from them (about 0.2 s a time).

## 2. What was changed without changing any number

Each change was accepted only if every connection (sender, receiver, strength, trace, step of making), every emission, every cell quantity and the generator's state at the end of 16,384 bytes are identical, compared with `==`, for the first form, the coincidence form and the tissue whose strengths do not change, with the four switches on.

| Change | Check | Effect |
|---|---|---|
| Senders pass on in the order of their numbers | the order of floating-point additions into a cell's arrival no longer depends on the order of the work list. This changes E63's event program in the last bit (it is order-dependent), so the reference for the checks below is E63's event program with this one line added (`checks/evt_core_reference_fixed_order.c`) | needed for any exact check of the rest |
| A silent cell is put to sleep and its skipped steps are computed one by one, with the same formulas in the same order, when anything next reaches or reads it | identical in all three tissues, and with withdrawal by use on | first form: cell computations 25,531 → 8,720 a byte; the stationary tissue about 2.5 times cheaper; the plastic tissues little, because their cost is in passing on |
| The sort of the senders and the scan of all cells for making replaced by bitmaps; `-O3 -march=native` | identical | 10 to 25% |
| Records: one call into the core for a byte without a counting step; counts and distances computed with arrays from one reading of the connections a step | every receipt field (except times and file digests), both windows, the summary and the emissions identical to E63's recording loop | first form with records 2.96 → 2.03 ms a byte |
| Counts every N bytes (`--count-every`; E63: 128) | a choice of measurement | removes most of what is left of the records' cost |

**All program changes together** (core only, first 8,192 bytes, one at a time on this machine): the tissue whose strengths do not change 0.200 → 0.075 ms a byte (2.7 times); the first form 1.684 → 1.355 (1.24 times); the coincidence form 3.310 → 3.334 (no gain; its cells are mostly not silent with threshold 0, and the coincidence rule's sums are not deferred). **The exact changes do not make the plastic tissues much faster: their cost is in the events the tissue itself produces.**

## 3. What the tissue's numbers do to the cost (they change the tissue)

First form, bytes 8,192 to 16,383, core only, side by side:

| Steps a byte | Connections a cell | Connection visits a byte | ms a byte | 262,144 bytes |
|---|---|---|---|---|
| 32 | 93.8 (no cap) | 80,880 | 0.99 | 4.3 min |
| 32 | 31.0 (cap 32) | 6,956 | 0.33 | 1.4 min |
| 32 | 15.5 (cap 16) | 2,398 | 0.20 | 53 s |
| 8 | 93.8 | 46,534 | 0.69 | 3.0 min |
| 8 | 15.5 (cap 16) | 3,458 | 0.15 | 40 s |
| 4 | 15.5 (cap 16) | 2,665 | 0.10 | 27 s |
| 8, tolerance 1/256 | 15.5 (cap 16) | 2,038 | 0.13 | 34 s |

A cap is a number we fix; the definition asks the cells to settle it. Plan 1 (`PLAN_1.md`, outcome entered) asked whether they do when connections that bring nothing of use are withdrawn. They do not: in the first form what is withdrawn is made again (about 73 withdrawn and 97 made a byte), the number stays at about 93 and the churn makes the program 2.4 to 2.9 times dearer. The number of connections, and with it most of the cost, is set by the making rule, which makes a connection every 128 steps while a cell's activity is below a target that almost no cell reaches.

## 4. Files

`TISSUE_EVENT/evt_core.c` (the core), `evt.py` (E63's wrapper, with the cap's entry), `evt_run2.py` (the runner: `--steps`, `--cap`, `--count-every`, `--slow-records` for E63's loop). `TISSUE_STEP1/` holds E63's `tissue_step1.py` and `step1_read.py` unchanged. `checks/` holds the comparison and timing scripts as they were run in a scratch folder (their relative paths name the copies they compared: `.` the E63 event program, `ref2` the fixed-order reference, `fast` and `hp` this program) and the two reference cores.
