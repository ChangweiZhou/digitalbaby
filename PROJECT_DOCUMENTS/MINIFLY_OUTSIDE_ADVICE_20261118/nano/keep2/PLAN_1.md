# Keeping, repeated by another line: plan (written before any program of mine has run). 20 November 2026

Line `nano/keep2`, branch `claude/nano-keep2-8mmhfm`. An independent repeat of the final candidate of `nano/keep` (roadmap row K1), asked by the coordinating session under Mr. Xu's words as that session relayed them to me ("Keep experimenting with the agents." and "Respect evidence and keep trying new ideas."). **Development level.** No gate, seal or review is claimed. Everything marked "my reading" is mine.

**What I have read:** `AGENTS.md`; the roadmap's row K1 and section 5; `loop_small_case/REPORT.md` (sections 2, 9), `cells2.py`, `ERRATA_20261120.md`; `nano/keep/REPORT.md`; the prose of `nano/keep/PLAN_2.md` (its paragraph on joining only), `PLAN_3.md`, `PLAN_4.md`, `PLAN_6.md`, `PLAN_7.md`. **What I have not opened:** any `.py` or `.c` file of `nano/keep`, its tables, job lists and result files. I will not open them before my own program has given its first full result (candidate and plain, both conditions, the five reported seeds), and I will say when I do.

## 1. The claim

A system of cells with three rules keeps never-shown two-digit addition (400,000 lines) through 200,000 lines of subtraction: 85.8% with no old line and 81.6% with one old line in 32 (median, seeds 73129, 93683, 116089, 406364, 545353), with 128 cells joining at birth and 128 at the change of task; plain system 1.8 and 15.6; the program's choice (`reserve=128`) 4.4 and 67.0.

## 2. The three rules as I take them (my reading)

The plain system is that of `cells2.py` (ten delay lines of 14 byte values = 140 lines; cells in between with logistic output; one answering cell with 14 branches and 14 constants; relay over learned return strengths; eta1 1/4, eta2 1/16, fading 1e-5 a byte). A line of text has ten bytes; nine of them are "learned bytes" (the byte after the end-of-line byte is not predicted). A pool of 384 cells in between; one is present at the first byte, the others wait. A waiting cell is silent and unchanged. A cell joins with its birth strengths (answering strength nothing; return strengths at their random birth values).

At each learned byte, in this order:

1. *Who speaks.* A joined cell younger than 4,500 learned bytes (counted from its joining) speaks. An older joined cell speaks unless one of the ten active lines has a closed strength in that cell; then it is silent and unchanged at that byte: output 0, no step on its strengths from lines, its threshold, the answering strengths from it or the return strengths to it, and no fading of those answering and return strengths.
2. *The plain system's step*, among the cells that speak (the arithmetic of `cells2.py`, in its order). The answering cell's step on each of its 14 constants is multiplied by exp(-n / 10,000), n the number of bytes it has learned from before this one.
3. *Youth.* Each young cell adds one to its count for each of the ten active lines (stops at 255) and one to its age. At age 4,500 its strength from a line is open if that count is at least 30, closed otherwise, for good.
4. *Joining.* The answering cell's running mean A of the number of cells that spoke moves by 0.01 (heard - A); it starts at 0. X = max(0, 1 - A / 128) is added to a running sum; waiting cell number i (from 0) joins when the sum reaches (i + 1) * 20. A cell that joins takes part from the next learned byte.

Scoring is that of `cells2.py` (each of the three answer digits named from the true bytes before it; the whole answer right; 500 held-out pairs for each task; history before the line filled with the end-of-line byte), with rule 1 applied as while learning.

## 3. Where the description leaves a choice, and what I take

| # | Choice | Taken | Other reading |
|---|---|---|---|
| a | Which bytes count for a cell's youth and its meetings | learned bytes only (the report says "first 4,500 learned bytes") | every byte, the unpredicted one included |
| b | Whether the answering and return strengths of a silent cell fade while it is silent | no ("no fading of the strengths to and from it", `PLAN_3.md`) | they fade like the rest (the coordinator's sentence says only "silent and unchanged") |
| c | The law by which the constants become still | exp(-n / 10,000) (`PLAN_4.md`) | another law or time |
| d | n in that law | bytes learned before the present one | after it |
| e | The running mean: rate, start, what is counted | 0.01; 0; cells that spoke at this byte, the young included | start at K; joined cells |
| f | Order within a byte | as in section 2 | joining before the step |
| g | Size and birth values of the pool | 384 cells drawn as `cells2.py` draws 384 (so the first 256 are not the plain system's 256 cells) | a pool of 256; other draws |
| h | The stream of lines | drawn as the unbatched loop of `cells2.py` draws it (one draw for the old-line choice when reh is set, one for the line) | the line's program may draw otherwise; then single seeds differ and only medians can be compared |
| i | A young cell at a reading | scored as treated while learning (speaks) | scored by its marks |
| j | Counting stops at 255; count compared after youth only | as stated | - |
| k | Mean of what, for "repeats" | last reading, and mean of the last three readings of the second phase (lines 500,000, 550,000, 600,000) | - |
| l | The harsh change | subtraction written `aa=bb+ccc` (larger number first, aa = bb + ccc), scored on its last three digits | another layout |

**The two I judge to matter most before running:** (b) and (a). (b): with fading at 1e-5 a byte over 1.8 million bytes the old cells' answering strengths would shrink by e^-18, and I expect addition to be lost entirely. (a): under my reading the layout `aa=bb+ccc` is *not* free of new lines. In addition the byte `=` is never four bytes back at a learned byte and `+` never seven bytes back (both fall on the unpredicted byte); in `aa=bb+ccc` they are, at two learned bytes of nine. So under reading (a, taken) I expect the old cells to fall silent at those two bytes and some cells to join in the harsh change, which the line reports does not happen; under the other reading every byte value is met at every delay and nothing is new. I will run both readings of (a) on the harsh change and on the main claim, and the other reading of (b) on the main claim. If the first results show that some other choice matters more, I will run that too and say that it was chosen after seeing figures.

## 4. Runs

1. My numpy program against `cells2.py` with all cells present and no rule (every strength compared), and my compiled kernel against my numpy program with the rules on, on a short life that contains birth, the end of youth, a change of task and old lines. These checks are of my own programs against each other and against the plain system; they are not a comparison with the line's program.
2. Candidate and plain, `add:400000,sub:200000`, no old line and one in 32, seeds 73129, 93683, 116089, 406364, 545353 and 2001 to 2005: 40 runs. Readings every 50,000 lines. One of the candidate's lives also in numpy in full.
3. Subtraction beside its clean score: the held-out pairs whose written line (larger number first) no shown pair writes.
4. The harsh change (`aa=bb+ccc`), candidate and plain, both conditions, the five reported seeds; candidate under both readings of (a).
5. The other reading of (b) and of (a) on the main claim, five reported seeds, both conditions.
6. If time allows: the program's choice (`reserve=128`, from `loop_small_case/cells2.py`, run in a copy outside the repository) on seeds 2001 to 2005, so that the new seeds have their own comparator; otherwise the line's 67.0 is used.

## 5. What counts

**Repeats** (fixed now): in both conditions, on the five reported seeds, the median of my candidate's never-shown addition at the last reading is within 10 points of the line's (85.8 and 81.6), and with one old line in 32 it is above the program's choice (67.0, the line's figure; and my own figure for it if run); and the same on seeds 2001 to 2005 against the line's figures for its new seeds (84.8 and 81.2). **Repeats in part**: one condition or one set of seeds only; said as such. **Does not repeat**: otherwise. The mean of the last three readings is given beside each last reading; if the two disagree on the verdict I report the disagreement and do not choose. A result that holds under one reading of a choice of section 3 and not under the other is reported as holding only under that reading.

Falsifier of my own expectation: if the candidate under my reading keeps under 20% in either condition at the median, the description as written does not carry the result.

## 6. Predictions (mine, before any run)

- P1 (0.75): candidate, no old line, reported seeds: median addition at the end within 10 points of 85.8.
- P2 (0.60): candidate, one old line in 32, reported seeds: median within 10 points of 81.6 and above 67.0.
- P3 (0.80): 128 cells join at birth and 128 at the change of task on every seed of the main runs, none at another time.
- P4 (0.90): plain: at most 5% with no old line; between 5 and 30% with one in 32 (medians).
- P5 (0.55): single seeds do not agree with the line's to the digit (different stream or pool), though medians agree within 10 points.
- P6 (0.65): harsh change, my reading of (a): some cells join (more than 10) and addition at the end is still below 20% at the median in both conditions.
- P7 (0.70): harsh change, other reading of (a): no cell joins and addition is within 5 points of the plain system's.
- P8 (0.85): other reading of (b) (silent cells' strengths fade): addition at the end below 20% with no old line.
- P9 (0.70): the clean score of subtraction is within 6 points of the usual one for the candidate and for plain (medians).

**Cost expected.** About 60 compiled runs of one CPU minute; one numpy life of 10 to 15 minutes; four cores.

**Not run.** Other thresholds, youths, K, theta; a third task; the other order; one old line in 8; anything beyond the repeat.
