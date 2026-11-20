# nano/unite: the clock and the cells that count, in one system (20 November 2026)

**By** the session that holds this line of the outside work. **Authorisation.** Mr. Xu, 20 November 2026, in Chinese, in the coordinator's translation as given to this line: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The task (compose the pieces; report where the composition holds and where it breaks) is the coordinating session's. **Status.** Development level throughout: one program of mine, one small text, one development seed for choosing and the five reported seeds (73129, 93683, 116089, 406364, 545353) for the figures in bold; nobody has repeated any of it. No checker gate, seal or outside review is claimed. `PLAN_1.md` and `PLAN_2.md` were committed before their runs and carry every outcome and the score of every prediction; `PLAN_3.md` is a plan whose runs were not made. Files were changed only in this folder. The readings marked "mine" are this session's.

## 1. The question

Today a trial clock for the delay chain (`nano/skip`) and cells that count with look-ahead (`nano/sparse`, repeated by `nano/ahead2`) were each shown to work, each alone and on its own text. Do they compose into one system?

## 2. The system

Cells that count (`ahead2`'s arithmetic): a cell is keyed on at most four of nine lines and counts the next byte, the one after and the third; an answering cell guesses the byte three ahead, then two ahead, then answers, with a learned trust for each family, kind and level of experience; cells keyed on at most three lines and one of the guesses. **The nine lines** are the present byte and a chain of the last eight bytes that entered it. **The clock** (`skip`'s second form, its numbers, nothing retuned): one entry for each byte value, closed by chance for periods of 256 bytes, leaning to whichever state gave the lower mean of one number that the answering cell sends to all entries alike, the sum of its squared mismatches. No relay system. `unite.py` holds a plain Python reference and a compiled kernel that agree to 1e-14 on ten cases (`check_output.txt`).

Text `g2`: `aa+<0 to 8 letters>bb=ccc`. 500 pairs never shown, 8,000 shown. Score: the line is read to the `=` and the system writes the answer from its own bytes; whole answer right, at 0, 4 and 8 letters.

## 3. Step 1: do they compose? (`PLAN_1.md`)

Never-shown sums, %, median of the five reported seeds; one figure where the three gaps are equal.

| Arm | 40,000 lines | 100,000 | 200,000 | last three readings (the reading fixed beforehand) | cells that have counted; counts above zero |
|---|---|---|---|---|---|
| **the united system** | **93.8** (91.6 to 96.2) | 78.4 | 52.6 | **53.7** (50.3 to 58.3) | 10.2 million; 110 million |
| letters passed over by the program | 92.0 | 85.2 | 73.8 | 75.1 | 3.2 million; 42 million |
| no clock (three seeds to 60,000 lines; seed 1 to 200,000), gaps 0 / 4 / 8 | 67.8 / 0.0 / 0.0 | 85 / 0 / 0 (seed 1) | 88 / 0 / 0 (seed 1) | 86.7 / 0.0 / 0.0 (seed 1) | 7.2 million; 98 million (seed 1) |
| the relay system with the same clock (`nano/skip`'s files; true answer bytes before each place) | not read | 2.0 | 75.2 | 96.7 at 1,200,000 lines | 55,642 numbers that learn |

**The hypothesis** (at least 80% at every gap within 200,000 lines, read as the mean of the last three readings; refuted under 50% at the gap of 8) **is not met: 53.7.** It is not refuted by its letter, narrowly, and the readings are still falling.

What holds:
1. **The clock is found, and early.** Exactly the four letters are closed from 20,000 to 30,000 lines on all six seeds, and no other entry is closed at any reading. The relay system's clock needed 100,000 to 200,000 lines. The counting cells' mismatch is a good judge; their speed helps the clock.
2. **At 40,000 to 60,000 lines the two pieces are one system that works**: 93.8% and 92.2% of never-shown sums at every gap, level with the arm in which the program removes the letters.
3. **The look-ahead in real bytes does not clash with the clock.** The guess of the units digit made at the `=` is right on 100% on every seed; the guess of the tens digit as often as in the oracle arm.

Where it breaks:
4. **It does not hold its level.** After 60,000 lines the score falls, in the tens digit, to 54% at 200,000 lines.
5. **The fall is located in two parts** (development seed, one change each). (a) The cells' own fall with more reading, which the oracle arm has with no clock at all (94 to 76) and which `nano/sparse` reported on plain sums. (b) The clock's trials going on for life: with no entry tried after 30,000 lines the united system follows the oracle arm (75.8 against 76.8, where the united system is at 58.1). Each open entry is still closed in one period in a hundred, so some digit or sign is out of the chain in about one period in eight; the relay system forgets what such a period teaches, and cells that count do not.
6. `nano/sparse`'s cap (a cell speaks at the weight of at most 16 counts) carries over in part: 84.5 in the oracle arm and 82.2 in the united system at 200,000 lines on the development seed, both still 8 to 9 points under their reading at 40,000. Not run on the reported seeds.
7. **Size.** The united system holds 3.2 times the cells of the oracle arm, because everything counted while the chain was changing stays; 3.6 million of its cells are keyed on a letter at a chain line and can never be reached again. Against the relay system it is about 180 times the numbers.

## 4. Step 2: keeping (`PLAN_2.md`)

Life: 40,000 lines of addition, then 20,000 of subtraction, both with letters; nothing of addition shown again. Never-shown additions, last reading of each phase, medians of five seeds.

| Arm | before | after | kept, % (each seed) | clean subtraction at the end |
|---|---|---|---|---|
| **the united system** | 89.6 | 66.2 | **73.2** (24.5, 71.5, 86.5, 73.2, 80.8) | 87.5 |
| letters passed over by the program | 91.4 | 85.6 | **96.0** (94.8, 96.0, 91.7, 96.6, 96.0) | 86.7 |
| the united system, no entry tried after 30,000 lines (three seeds) | 86.0, 87.0 / 81.0 / 41.6, 93.6 | 80.4, 76.4 / 64.0 / 38.6, 82.8 | 93.5, 87.8 / 79.0 / 92.8, 88.5 | 94.1, 78.1 / 74.0 / 35.4, 86.2 |
| `nano/ahead2`, plain sums (its report) | 91.2 | 86.8 | 94.7 | 89.4 |

**The hypothesis (at least 85% kept) is refuted for the united system: 73.2%.** The cells keep (96.0% with the letters removed by the program); the clock's trials during the second task spend it. The loss falls in the first 5,000 lines of subtraction, when the new `-` entry is closed one period in five. With the trials stopped the same system keeps 93.5% on the seed where it otherwise keeps 24.5%. (On the middle seed of the frozen arm one letter was still open when I froze it; that seed tests the freezing badly.)

## 5. Step 3: three-digit numbers (`PLAN_3.md`): argued, not run

On `aaa+<letters>bbb=cccc` the units digit is four bytes after the `=`. With a look-ahead of three the guess of the tens digit can only come from cells of four digits, which is where the tens digit of two-digit sums stood without look-ahead; the hundreds and thousands lean on it. The smallest change is a look-ahead of four in the same form; `unite.py` supports it (`look=4`) and the check covers it. **No run was made**, so this is reasoning and nothing more. **And it is contradicted**: after my runs ended the coordinating session told me that `nano/ahead2` ran plain three-digit sums today with a look-ahead of three (99.7%) and of four (no gain). I have not verified that; as that line reports it, my argument does not hold (`PLAN_3.md`, correction).

## 6. Reading (mine)

1. The answer to the question is **yes, and not durably**. One system without a relay finds what to pass over by a rule that names no byte and answers 94% of never-shown sums across 0 to 8 letters within 40,000 lines, a fifth of the lines at which the relay system with the same clock is at 75%.
2. Both failures have one root, and it is in the joint, not in either piece: **a clock that never stops trying, feeding cells that never forget.** In the relay system the trials were free. Here each misaligned period leaves counts for good, in the first task (the fall) and across tasks (the keeping).
3. What I would test next, in this order, and have not: a clock whose entries stop trying when their evidence is firm and try a newly met byte value gently (a rule an entry can follow from its own two means; the freezing by line count used here is the program's, and failed on one seed of three); the cap together with it; cells that give back counts they are not reached by any more (a third of the cells are dead weight).
4. The relay system with the clock is small, slow and goes on rising; the united system is fast, large and falls. For the aim of one system, as the coordinator put it to this line, I take that as a plainer statement of where the pieces stand than either of today's separate results.

## 6a. Added after the runs ended (20 November 2026)

Two things from the coordinating session, entered as notes; nothing was run for them.

- Mr. Xu's words as that session passed them to this line (Chinese with English terms; the coordinator's translation): "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." Against that: no number of either piece was retuned for this text (the clock runs with `nano/skip`'s numbers, the cells with `nano/ahead2`'s). The stopping of trials at 30,000 lines is a line count laid down by the program; it is used here only to locate a cause and is not offered as a remedy. The look-ahead's depth of three is also laid down by hand, not found by the system. The same numbers were not run again on plain sums by this line; the oracle arm is the nearest thing to it (the chain then holds plain sums).
- The second cause, stated generally: **a store that keeps every count for good has to be protected from what the system is merely trying.** A general remedy the coordinator names: counts made while an entry is under trial are held apart and added to the cells only if the trial is not kept; no entry and no task is named. **NOT RUN.** For the first cause, also **NOT RUN** in this system: a trust that knows whether a more specific sender is present (`nano/table/PLAN_14.md`). The cap was run on the development seed only.

## 7. Predictions

Written before their runs: 24; held 17, failed 7. `PLAN_1.md`: 7 of 10, and 2 of 4 in its addition; `PLAN_2.md`: 8 of 10. Of the two hypotheses given to this line, the first is not met and the second is refuted. Failed: the oracle arm at 85% at the fixed reading (75.1); H1; the united system at or above the relay system at 200,000 lines; the cap holding the oracle arm within 5 points; the cap lowering the peak by 3 points; H2; the united system keeping within 8 points of the oracle arm.

## 8. Not run, and loose ends

- No clock on two of the five reported seeds, and beyond 60,000 lines on any reported seed; the capped arm on reported seeds; seeds 1001 to 1010; the frozen arm of keeping on two seeds; a third task; all of step 3.
- On one seed of the keeping runs the `+` entry closed and stayed closed (the loose end `nano/skip` reports for `=`). It cost nothing by itself there; with trials going on in the second task it was the worst seed.
- The split and the held-out pairs are `cells2.py`'s; the stream's draws are not `nano/skip`'s, and its relay figures are scored with true answer bytes where the cells are scored on their own.
- One run was lost to memory (three runs of 4.8 gigabytes at once) and was made again; nothing else was lost.

## 9. Commands and cost

    python3 unite.py check
    python3 unite.py run trial_73129 seed=73129 clock=trial            (200,000 lines of addition; clock=oracle | clock=none)
    python3 unite.py run dev_freeze_1 seed=1 clock=trial freeze=30000   |   ... ncap=16
    python3 unite.py run keep_trial_73129 seed=73129 clock=trial plan=add:40000,sub:20000 every=5000
    python3 table.py trial,oracle,none,keep_trial,keep_oracle,keep_freeze 73129,93683,116089,406364,545353

Job lists as run: `jobs_1.txt` to `jobs_5.txt` (queues `runq.sh`, `runq2.sh`; lists 1 to 3 were each cut short and continued by the next). A run of 200,000 lines takes about 9 CPU-minutes and 3.6 gigabytes (17 byte values) and twice the time with the cap; a keeping run about 3.5 CPU-minutes and 4.8 gigabytes (18 byte values). 32 runs kept and one lost and repeated, about 3.4 CPU-hours.
