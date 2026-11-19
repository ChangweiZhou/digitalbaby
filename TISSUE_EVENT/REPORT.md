# E63: the tissue's program written so that it works only where something changes — report (19 November 2026)

Task: `TASK_E63_THE_TISSUE_PROGRAM_EVENT_DRIVEN_20261119.md` on `claude/vibrant-cerf-sje726`. It was released by decision 72, and part (a) was changed at 04:48 UTC. This branch, `claude/e63-tissue-event-017dSE`, starts from 040d3cf09, the head of `origin/claude/e63-tissue-step1`. No file of E63 was edited. Every figure below comes from a receipt in this folder. Nothing here is confirmed independently.

## Terms

- **E63's program**: `TISSUE_STEP1/tissue_step1.py`, imported unchanged. **The new program**: `evt_core.c` (C, gcc 13.3 `-O2 -ffp-contract=off`), called from Python through `evt.py`.
- **The four switches**:
  - the output floor: an output below it is 0;
  - the tolerance: a sender passes on the change of its output only when that change exceeds it;
  - one trace a sender, in place of a trace for each connection;
  - strengths once a byte, from sums kept by the receiving cell, in place of at every step.
- **Every switch off**: floor 0, tolerance 0, a trace for each connection, strengths at every step.
- **Ulp**: one unit in the last place of a double.
- **Host**: Linux x86-64, 4 cores, Python 3.13.18, numpy 2.5.5 (E63's stage A used numpy 2.5.3), one thread a process, at most three processes.

## The program (section 3 of the task)

1. **Connections.** They are stored once, in slots. Each cell keeps the list of its incoming slots in the order of making, and each sender (cell or line) the list of its outgoing slots. Both lists are corrected when a connection is made or given up.
2. **Kept arrival.** Each cell keeps its arrival from step to step.
   - At the start of a step, every sender whose value moved by more than the tolerance passes on σ·w·(change) to its receivers, which join the work list.
   - All arrivals of a step are collected before any cell is computed, so the result does not depend on the order of the cells (clarification 3).
   - A cell leaves the work list when it is stationary: computing it again would change neither its state, output and threshold nor any strength.
3. **Lazy relaxation.** The prediction, the activity average, the sliding threshold and the traces carry the step they hold. They are brought up to date when needed: by the recurrence itself for gaps up to 64 steps, and in closed form with `pow` beyond. Reading the state for the records computes these values without storing them (see "Departures").
4. **Making connections.**
   - Making is looked at only for cells whose period has passed, through a queue of timers. A full cell is left out until one of its connections is given up.
   - Giving up is tested where a strength has just changed, and when a connection reaches 128 steps of age.
   - The life generator is Python's Mersenne Twister. Its state is handed to the core, and `Random.choice` is reproduced exactly.
5. **The switches** are parameters, as listed under "Terms".
   - Strengths once a byte use, at the reading step, the receiving cell's sum over the byte (formula 7: Σ m·x(k−1); coincidence rule: Σ x(k−1)(x(k−1)−ϑ) and Σ (x(k−1)−a*)), times the trace held then.
   - The floor applies to x before every use of x.

The 512-byte tissue (formula 11 as first written) keeps its sending-side average ū for each connection, and its giving-up test at every step, densely.

## (a) Every switch off, against E63's program (`part_a.py`, receipts `qual/part_a_*.json`)

| Comparison | Result |
|---|---|
| Worked example, E63's check 1 parts (a) r4, (b) E4, (c) K, (d) r5 | **passes.** Rows against the expected values: 0 after rounding. Final strengths within 3.45e-05 (limit 0.6e-4). Against E63's own computation of the example: largest difference 2.2e-16 (rows), 2.8e-17 (final). |
| Fixed tissue, 2,048 bytes | **identical.** Emissions, connections (sender, receiver, step of making), and the generator are identical. Largest difference of strengths, traces, states, thresholds and averages: 1.7e-15. |
| First-form tissue with formula 11 as first written, 512 bytes | **identical.** 251,968 connections made and 265,850 given up, all at the same steps. Largest difference 1.4e-16. |
| Coincidence tissue, 2,048 bytes | **Not identical.** The first difference is the connection 325→145, given up at step 1594 here and at step 1595 by E63's program; its strength then differs by 7e-7. Outputs first differ at step 4 (1.7e-18), stay near 2e-16 to step 499, then grow about tenfold every 100 steps (5.7e-11 at step 999, 3.9e-6 at step 1399). |
| First-form tissue, 2,048 bytes | **not run.** Its reference run was stopped at the reviewer's change of 04:48 UTC, before it ended. |

**Why the coincidence tissue cannot be identical** (`nudge_test.py`, `qual/nudge_test.json`).

- I ran E63's own program twice on seed 73129. In the second copy, every strength between cells was moved up by one ulp after step 4.
- The largest output difference between the two copies is:

| Step | 499 | 599 | 799 | 999 | 1199 | 1399 |
|---|---|---|---|---|---|---|
| Largest output difference | 1.6e-16 | 9.6e-14 | 4.6e-12 | 3.9e-11 | 3.7e-8 | 2.6e-5 |

- That is the same growth as between the two programs. The coincidence tissue amplifies a difference of one ulp to about 1e-5 within about 1,000 steps.
- The new program keeps its arrivals by adding changes (point 2), so it rounds differently from numpy's `bincount`. It therefore cannot agree with E63's program over a life.
- The reviewer accepted part (a) on this ground (message of 05:03 UTC). They asked that the summation not be rewritten in numpy's order and that the run with point 3 off not be made. Neither was done.

## (b) Every switch on: floor 1/128, tolerance 1/1024, one trace a sender, strengths once a byte (`evt_run.py`, `PART_B_TABLES.md`)

Seed 73129, 16,384 bytes. The readings of section 11.2 are made by E63's `Run` class and `step1_read.read_run`, unchanged.

- **"Records"** is the cost with those records and readings.
- **"Core alone"** is the same tissue run again by the core without any recording. Its emissions are identical to the recorded run's in all seven runs.
- **E63's figures** are from its stage A receipts. They were measured on E63's own host, not on this one. On this host E63's program cost 0.0925 CPU-s a byte for the coincidence tissue and 0.0041 for the fixed tissue, both without records and over 2,048 bytes (`ref/*.json`). It was not timed here on the first-form tissue.

| Tissue | Program | CPU-s a byte, records | core alone | Connections a cell, end | Cells sending a step | Connection visits a byte | Emission share |
|---|---|---|---|---|---|---|---|
| first-form | E63 | 0.3631 | — | 91.79 | not counted | at least 6,015,680 (32 × connections) | 0.0050 |
| first-form | new | **0.00270** | **0.00128** | 93.84 | 37.1 | 113,156 | 0.0005 |
| coincidence | E63 | 0.0821 | — | 25.89 | not counted | at least 1,696,736 | 0.1901 |
| coincidence | new | 0.00381 | 0.00313 | 31.50 | 231.9 | 264,768 | 0.0674 |
| fixed | E63 | 0.0044 | — | 7.75 | not counted | at least 507,904 | 0.0000 |
| fixed | new | 0.00038 | 0.00021 | 7.75 | 3.4 | 580 | 0.0000 |

E63's program visits every connection at least once a step, so its visits are 32 times its connections. That is a count by construction, not a measurement. "Cells sending" is not counted by E63's program.

**Readings: where the new program's tissues differ** (full tables in `PART_B_TABLES.md`):

- **First-form tissue.**
  - Connections: 192,179 at byte 8,191 and to the end. That is every neighbourhood full: 192,179 is the total of the neighbourhoods of seed 73129 in E63's check 2. E63's tissue had 187,990 at the end.
  - Connections given up by the end: 138,212, against E63's 944,453.
  - Cells at d = 1 that are neither silent nor constant fall to 0.565 in the last stretch (E63: 0.848). In window A2 the share is 0.709 (E63: 0.954).
  - Cells at d = 2 and d = 4 are silent or constant in both programs, with shares 0 in A2.
  - Scores in A2: S(1,0) = 0.679 (E63: 0.684); S(2,0) = 0.335 (E63: 0.347).
- **Coincidence tissue.**
  - 64,517 connections at the end, against 53,023.
  - Emissions in 6.7% of positions, against 19.0%.
  - Share at d = 1 in A2: 0.616 (E63: 0.730). At d = 2 and 4: 0.992 and 1.000 (E63: 0.996 and 1.000).
  - Scores in A2: S(1,0) = 0.309 (E63: 0.234); S(2,0) = 0.026 (E63: 0.038).
- **Fixed tissue.** The connections and the shares of every stretch are equal. S(d,0) is equal to four decimals in A1 and A2. S(2,L) for L ≥ 1 differs at most in the fourth decimal (largest difference 0.0006).
- **The outcome of section 11.3 is unchanged.** The cells at d = 2 do not stay informative in either plastic tissue of either program:
  - first-form: share 0 at d = 2;
  - coincidence: S(2,0,A2) = 0.026 and 0.038, below 0.1146.

**Each switch alone, first-form tissue, 4,096 bytes**, the other three off:

| Setting | CPU-s a byte, records | core alone | Cells sending a step | Connection visits a byte | Given up by byte 4,095 | d = 1 / 2 / 4 informative, stretch 3 |
|---|---|---|---|---|---|---|
| E63's program (all off; first 4,096 bytes of stage A) | 0.3702 | — | not counted | at least 6,021,312 | 296,737 | 0.992 / 0.049 / 0.000 |
| floor 1/128 | 0.01678 | 0.01465 | 179.0 | 2,814,277 | 91,771 | 0.992 / 0.045 / 0.000 |
| tolerance 1/1024 | 0.08793 | 0.08308 | 194.5 | 4,912,959 | 101,826 | 0.983 / 0.030 / 0.000 |
| one trace a sender | 0.08157 | 0.07747 | 1,134.8 | 8,367,287 | 299,466 | 0.987 / 0.026 / 0.000 |
| strengths once a byte | 0.03702 | 0.03598 | 839.9 | 1,977,835 | 388,632 | 0.958 / 0.004 / 0.000 |
| all four on (first 4,096 bytes of the run above) | 0.00284 | — | — | — | 130,089 | 0.932 / 0.000 / 0.000 |

What each switch does alone:

- **The floor and the tolerance** each cut giving up to about a third: 91,771 and 101,826, against 296,737 in E63's program and 299,466 with one trace a sender. Giving up is therefore sensitive to the residues below 1/128 and to changes below 1/1024.
- **Strengths once a byte** gives up more (388,632) and lowers the share at d = 2.
- **One trace a sender** stays closest to E63's program in giving up (299,466 against 296,737). It leaves 1,134.8 cells sending a step, because without a floor almost every cell carries a residue.
- **Cost:** no switch is fast alone. The cost falls to 0.0028 CPU-s a byte only when the floor and the tolerance stop the residues from being passed on and strengths are changed once a byte.

## Cost of this task

All figures are CPU-hours, from the receipts:

| Item | CPU-hours |
|---|---|
| Part (b), final runs | 0.547 |
| Part (b), first attempt (discarded, see "Departures") | 0.532 |
| Reference traces of E63's program (ref/*.json, fixed, f11w, coincidence) | 0.056 |
| Part (a) comparisons, nudge test and tests | under 0.1 |
| **Total** | **about 1.2**, against the limit of 2 |

The stopped first-form reference run is not in its receipts (it was killed). It ran from 04:37:17 to about 04:49 UTC on one thread.

## Departures and errors

1. **A defect in the first attempt of part (b), found and fixed.** Reading the state for the records (connections every 128 bytes, distances every block) brought the lazily held traces and predictions up to date and stored them. That changed the later rounding of the tissue.
   - It showed as a difference of emissions between the recorded run and the core-alone run, in the coincidence run with every switch on and in the tolerance-alone run.
   - Reading now computes those values without storing them (`peek_trace`, `evt_cells`).
   - Part (a) was rerun: the same results.
   - All seven runs of part (b) were made again. In all seven, the emissions of the recorded and core-alone runs are now identical.
   - The first attempt's receipts and logs are kept in `runs_first_attempt/` and none of its figures is used here.
2. **Part (a) was changed by the reviewer** after two reference traces existed and two were running. The first-form trace was stopped and left nothing. The coincidence trace had ended (189.4 CPU-s) and is used.
3. **Processes and cores.** At most three processes, one thread each, were used, as section 6 of the task says. The reviewer's message suggested four cores; the task's limit was kept.
4. **The disk was full** at the start. The working tree is a sparse checkout (TISSUE_STEP1, TISSUE_EVENT, PROJECT_DOCUMENTS/KB); no commit was affected.
5. **Messages.** I told the reviewer once that the coincidence trace was not committed; it was committed at 50a05602c. I also stated once, without measuring it, that the lazy updates were not used before step 1594. Both were corrected by message.

## Files (`TISSUE_EVENT/`)

`evt_core.c`, `evt.py`, `part_a.py`, `nudge_test.py`, `evt_run.py`, `b_queue.sh`, `b_tables.py`, `ref_trace.py`, `ref_queue.sh`, `PART_B_TABLES.md`, this report, `qual/` (part (a) receipts, nudge test), `ref/` (reference traces), `runs/` (the seven runs: receipts, emissions, summaries, windows), `runs_first_attempt/` (receipts and logs only), `logs/`.
