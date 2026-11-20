# Kinds of cells, first plan (C6, question 1): cells that differ in how long they hold (20 November 2026)

By the agent session that owns line C6 and the folder `nano/kinds/`. Development level; a small case; no checker gate, seal or review is claimed. The readings are mine.

**Source of the question.** Mr. Xu, 20 November 2026 (recorded in `ROADMAP_SECOND_REVISION_20261120.md`): "Also the network could have many different types of cells that adjusts around different time clocks, having different memory ability, ability to output signals (chemicals) etc. But we need to make sure not testing everything at once and leave a mess." **Authorisation.** The coordinating session's brief for line C6, under Mr. Xu's words of 20 November as that brief gives them: "revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process".

**Source of the programs.** Branch `claude/friendly-darwin-8mmhfm` at 260c5d73. Copied from `loop_small_case/` into this folder: `hold.py`, `cells2.py`, `kern.c`, `kern3.c`, `fastk.py`, `fastk3.py`, `.gitignore`. Not copied: `kern2.c`, `kern4.c` and their wrappers (the copy refuses `chain`, `ret=local2`, and `fast=1|2` for `cells2.py`). The options added are described in the headings of `hold.py` and `cells2.py`. `check_kinds.py` (output in `check_kinds_output.txt`) compares the compiled kernels with the numpy programs with every option off and on: PASSED, differences of 1e-14 to 3e-8 of the largest strength. With the options off, the copy and the original gave identical readings on a run of 40,000 lines of `hold.py` (text `m1`, the crossing) and of 100,000 lines of `cells2.py` (`fast=3`); the five baseline runs below are a further check against `loop_small_case/res/hold_r7_m1_exact_<seed>.json`.

## 1. The question

In `Hold.__init__` each cell's gate has a threshold `cz`, drawn at birth uniformly from -1 to 3, so that at birth cells hold for different times (a gate of about 0.27 to 0.95 with nothing else arriving). The thresholds then learn like every other strength. **Does it matter, for learning to hold one digit across a gap of letters, that the cells differ at birth in how long they hold?**

## 2. Testbed and arms

T1: `python3 hold.py <label> text=m1 gap=8 seed=<seed> eta1=1 cross=1 ret=exact lines=960000 every=20000 fast=1`. Lines `a+<0 to 8 letters>b=c`, c = (a + b) mod 10; 80 of the 100 pairs shown; the score is the whole answer right on the sums shown, the mean over gaps of 0, 4 and 8 letters. The readings are taken every 20,000 lines here (the original: every 80,000); a reading changes nothing in the learner, so the readings at multiples of 80,000 are those of the original command, and **the reported score is the mean of the readings at 800,000, 880,000 and 960,000 lines**, as in `loop_small_case`.

| Arm | Thresholds at birth | Option |
|---|---|---|
| `spread` (baseline) | uniform from -1 to 3 | none |
| `alike` | every one 1 (the middle of the baseline's range) | `zalike=1` |
| `wide` | uniform from -1 to 5 | `zspread=-1,5` |
| `wide_c` (added by me) | uniform from -2 to 4 | `zspread=-2,4` |

One change against the baseline in each arm: the draw of the thresholds. Every other number at birth is the same in all four arms for a given seed (in `alike` the draw is made and thrown away; the other draws use the same random numbers, rescaled). The arm `wide` of the brief moves the mean of the thresholds from 1 to 2 as well as widening the spread; `wide_c` widens by the same amount about the baseline's mean, so that the two can be told apart. Nothing is changed in the compiled kernel for this question.

**Second measure.** Lines to 90%: the first reading (every 20,000 lines) at which the score is at least 90%; "not reached" if none by 960,000.

**The exact learner.** `hold.py` of this folder, class `Hold`, with `cross=1 ret=exact`: 256 gated cells, each reading the present byte's line (18 byte values) and 16 other cells; one trace for each strength, running forward; the answering cell's mismatch relayed over learned return strengths; one crossing. Numbers that learn: 9,216 strengths from the lines, 8,192 between cells, 512 thresholds (17,920 in the cells in between); 4,626 at the answering cell; 4,608 return strengths. The same in all arms.

## 3. Hypothesis, what would refute it, predictions (mine, before any run of these arms)

*Hypothesis.* Cells that differ at birth in how long they hold make holding across a gap easier to learn: with all thresholds alike the score is lower or the lines to 90% are more.

*Refuted if* `alike` is within the scatter of the baseline on both measures: a median score within 5 points and a median of lines to 90% within a factor of 1.5, with no more seeds below 90% than the baseline has. With five seeds of which the baseline's reported scores run from 78 to 100, a difference of a few points is not carried.

| | Statement | Chance |
|---|---|---|
| A1 | the baseline rerun gives the reported 91.2, 81.9, 100.0, 78.1, 100.0 (to 0.1) | 0.95 |
| A2 | `alike`: median score at least 5 points below the baseline's median | 0.40 |
| A3 | `alike`: median lines to 90% at least 1.5 times the baseline's (a seed that does not reach 90% counts as more than 960,000) | 0.45 |
| A4 | `wide`: median score at least as high as the baseline's | 0.55 |
| A5 | `wide`: median lines to 90% fewer than the baseline's | 0.50 |
| A6 | `wide_c` and `wide`: median scores within 5 points of each other | 0.60 |
| A7 | every arm has at least one seed at or above 99 and at least one below 90 | 0.50 |

## 4. Order, cost, what is not run

Development seed 1, all four arms (nothing is chosen on it for this question; it is a check that the arms run); then the five reported seeds 73129, 93683, 116089, 406364, 545353 for all four arms. A run is about 3.5 CPU-minutes; 24 runs, about 85 CPU-minutes, about 22 minutes on four cores. Not run: other gaps or texts (`g1`, two digits); thresholds that do not learn; a spread in the other gate strengths; more than five seeds; the system without the crossing.
