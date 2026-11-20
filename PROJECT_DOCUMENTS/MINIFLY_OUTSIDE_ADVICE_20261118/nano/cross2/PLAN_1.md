# nano/cross2, first plan (E1): the gap text m1 with one and with two crossings (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as recorded in `ROADMAP_SECOND_REVISION_20261120.md` ("can you revise the road map and keep doing some experiments? Spin off some agents to speed up the experiment process."); the coordinating session's brief for track M names this experiment. **Source.** Branch `claude/nano-cross2-8mmhfm` at 1ce0337 (`hold.py`, `kern5.c` of this folder; both checks passed). Written before any run of 960,000 lines here; the only runs so far were the checks and 40,000-line timings on the development seed (cross=1: 12, 12, 8% by gap; cross=2: 12, 12, 12%, i.e. nothing learned yet in either). Development level; a small case; no checker gate.

**Question.** On the text m1 (`a+<0 to 8 letters>b=c`, c = (a + b) mod 10), does credit passed over a second connection (`cross=2`) learn faster and more reliably than over one (`cross=1`)?

**Learner, exactly.** `hold.py` of this folder, `text=m1 gap=8 eta1=1 ret=exact fast=1`, 256 cells, K = 16, no delay line (`chain=0`), 18 byte values, 960,000 lines; arms `cross=1` and `cross=2`, which differ in nothing else. Numbers that learn, the same in both arms: 2 × 256 × (18 + 16 + 1) = 17,920 in the cells and 2 × 18 × 256 + 18 = 9,234 at the answering cell and its return connections; 27,154 in all. The second crossing adds numbers a cell keeps (one per strength, two per cell: 17,920 + 512), none that learn.

**Measure.** Whole answer right on the 80 sums shown, mean over gaps of 0, 4 and 8 letters, read every 20,000 lines. *Final*: the mean of the readings at 800,000, 880,000 and 960,000 lines (the three readings the earlier five-seed runs averaged, so that the cross=1 arm can be set beside the coordinator's 91.2, 81.9, 100.0, 78.1, 100.0). *Lines to 90%*: the first reading at or above 90%; a run that never reaches it counts as more than 960,000.

**Order.** Development seed 1, both arms, first; its outcome is entered below before the reported seeds are read. No setting is open to choice in this plan (the step and the length are given), so the development seed serves to show that the programs run at full length; the five reported seeds (73129, 93683, 116089, 406364, 545353) are run in both arms whatever it shows.

**Hypothesis.** The second crossing makes the holding of the first number learnable sooner: over the five reported seeds the median of lines to 90% is lower with `cross=2` than with `cross=1`. *Refuted if* that median is equal or higher with `cross=2`.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| A1 | reported seeds: median lines to 90% lower with cross=2 than with cross=1 | 0.55 |
| A2 | reported seeds: median final score with cross=2 at least as high as with cross=1 in this folder's rerun | 0.55 |
| A3 | reported seeds: the lowest final score of the five is at least 85% with cross=2 (more reliable; the coordinator's cross=1 had 78.1 as its lowest) | 0.40 |
| A4 | this folder's cross=1 rerun is within 5 points of the coordinator's figure on at least four of the five seeds (the kernel sums in another order, and small differences can grow over a run) | 0.70 |
| A5 | development seed: cross=2 reaches 90% in at most half the lines cross=1 needs | 0.30 |

**Cost.** About 4 CPU-minutes a run with cross=1 and about 6 with cross=2 (from the 40,000-line timings); 12 runs, about one CPU-hour, a quarter of that in wall time on four cores.

**Not run.** The instrument as a learner at X=2L; other steps than eta1 = 1; return strengths other than the reader's own; the text g1; gaps beyond 8 letters; a look inside the networks.

## Outcome on the development seed (entered 20 November, before any reported seed was read)

**An error of mine in the launch, stated.** The first launch of the twelve runs used a launcher that dropped the first argument after the label, so `text=m1` was lost and the runs read the default text `d1`. I saw it on the first two finished files (their label was `text=m1`), stopped all runs, deleted their output without using it, corrected `run.sh` (the commit "launcher corrected") and started the twelve runs again. Nothing below comes from the first launch.

Development seed 1, 960,000 lines (`res/hold_e1_c1_1.json`, `res/hold_e1_c2_1.json`):

| | final % | lines to 90% | every later reading at least 90% from | at 240,000 lines | at the end by gap 0, 4, 8 | CPU s |
|---|---|---|---|---|---|---|
| cross=1 | 97.1 | 500,000 | 860,000 | 18.3 | 98, 100, 99 | 262 |
| cross=2 | 100.0 | 320,000 | 320,000 | 79.6 | 100, 100, 100 | 404 |

On this seed the second crossing reaches 90% in about two thirds of the lines, and from 360,000 lines on every reading is 100%, where the one-crossing arm still dips to 89% at 820,000 and 840,000 lines. **A5 failed** (320,000 is more than half of 500,000). The one-crossing arm here had 18.3% at 240,000 lines where the coordinator's run of the same seed and settings had 49%: the two programs sum in different orders, and at that point of the run the score is rising steeply, so I read this as the same curve shifted by some tens of thousands of lines, not as a different result; I have not tested that reading. One seed; nothing is claimed from it.

## Outcome on the five reported seeds (entered 20 November)

960,000 lines, sums shown, mean over gaps of 0, 4 and 8 letters (`res/hold_e1_c*_<seed>.json`; the table is `report_e1.txt`, made by `python3 report.py e1`).

| | 73129 | 93683 | 116089 | 406364 | 545353 | median | lowest |
|---|---|---|---|---|---|---|---|
| final %, cross=1 (this folder's rerun) | 83.2 | 94.4 | 99.7 | 89.9 | 100.0 | **94.4** | 83.2 |
| final %, cross=1 (the coordinator's runs, `loop_small_case`) | 91.2 | 81.9 | 100.0 | 78.1 | 100.0 | 91.2 | 78.1 |
| final %, **cross=2** | 99.0 | 100.0 | 100.0 | 95.7 | 100.0 | **100.0** | 95.7 |
| lines to 90% (first reading), cross=1 | 920,000 | 420,000 | 340,000 | 880,000 | 300,000 | **420,000** | |
| lines to 90% (first reading), **cross=2** | 240,000 | 340,000 | 200,000 | 480,000 | 300,000 | **300,000** | |
| every later reading at least 90% from, cross=1 | never | never | 400,000 | 940,000 | 300,000 | | |
| every later reading at least 90% from, cross=2 | 240,000 | 660,000 | 200,000 | 880,000 | 300,000 | | |
| at 240,000 lines %, cross=1 | 16.2 | 40.8 | 68.3 | 40.0 | 67.9 | 40.8 | |
| at 240,000 lines %, cross=2 | 93.3 | 33.8 | 98.8 | 40.8 | 70.8 | 70.8 | |

At the end by gap (0, 4, 8 letters): cross=1 85/88/90, 89/89/89, 99/100/100, 92/90/91, 100/100/100; cross=2 100/100/100 on four seeds and 95/95/96 on 406364. CPU: about 270 s a run with cross=1 and about 415 s with cross=2 (1.5 times).

**The hypothesis is not refuted.** The median of lines to 90% is 300,000 with two crossings against 420,000 with one; seed by seed it is lower on four seeds and equal on one (545353, 300,000 in both). In lines the gain is uneven: large on 73129 (240,000 against 920,000) and 406364 (480,000 against 880,000), small or none on the other three. In CPU time the two arms are about level at the median (300,000 lines at 1.5 times the cost of a line against 420,000). Final scores: 100.0 against 94.4 at the median and 95.7 against 83.2 at the lowest; every seed is at or above the one-crossing arm. Two crossings do not remove the dips: 93683 and 406364 fall below 90% late in the run (to 88% and 75 to 85% at single readings) and recover.

**Scores of the predictions.** A1 held (300,000 against 420,000). A2 held (100.0 against 94.4). A3 held (lowest 95.7). **A4 failed**: this folder's one-crossing rerun is within 5 points of the coordinator's figure on two seeds of five (116089, 545353); the other three differ by 8, 12.5 and 11.8 points, in both directions. A5 failed (above). The check of the kernel against numpy holds to 1e-10 over 300 lines, and with cross=1 the arithmetic is that of `kern4.c` in another order of sums, so my reading is that single runs of this text are not reproducible to better than about ten points between programs that agree to rounding; this was not tested here by running `loop_small_case`'s kernel again. It bears on how much one seed can show: the difference between the arms at the median (5.6 points in the final score) is smaller than this run-to-run spread, and the comparison rests on five seeds agreeing in direction (final score: four higher, one equal at 100.0), not on any one of them.

**What is and is not shown** (my reading). On this text, at this size and step, two crossings forward in time reach 90% sooner on four of five seeds and end higher or equal on all five. It is five seeds of one text; the reading that the gain comes from credit over two connections as such, rather than from a larger effective step on the same paths, was not separated (no arm with one crossing and a larger step was run here; `loop_small_case` has eta1 = 4 without a crossing only).
