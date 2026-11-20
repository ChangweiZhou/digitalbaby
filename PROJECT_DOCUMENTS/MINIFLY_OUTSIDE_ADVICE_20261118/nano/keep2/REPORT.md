# Keeping, repeated by another line: report. 20 November 2026

Line `nano/keep2`, branch `claude/nano-keep2-8mmhfm`. An independent repeat of the final candidate of `nano/keep` (roadmap row K1), asked by the coordinating session. **Development level.** No gate, seal or review is claimed. The plan (`PLAN_1.md`) was committed before any program ran; the readings marked "my reading" are mine.

## 1. The answer

**The result repeats**, by the measure fixed beforehand, on the five reported seeds and on five seeds no line had used (2001 to 2005), on the last reading and on the mean of the last three readings alike. My program was written from the line's prose (its report and the prose of its plans 2, 3, 4, 6 and 7) and gives, on the five reported seeds, the line's figures to the digit printed, seed by seed, for the candidate and for the plain system.

**It holds under one reading of one part of the description and not under the other** (section 4, choice b): a cell that is silent must also be exempt from the slow fading of the answering cell's strengths from it. If those strengths fade like the rest, addition is lost (0.0 and 11.6). The line states this in its third plan; the one-sentence form of the claim ("silent and unchanged") does not, and the result should be entered with it.

**The limit stands, and is a little wider than stated**: under the description as written (a cell's youth counts learned bytes) the text `aa=bb+ccc` is not free of new lines; 29 or 30 cells join, and addition is lost all the same (0.0 and 8.8). Under the other reading (every byte counts), which the line used for this test alone and says so in its eighth plan, no cell joins and I find the line's figures to the digit (0.4 and 13.4; plain 1.4 and 16.6).

## 2. The table

Never-shown addition at the end of `add:400000,sub:200000`, whole answer right, %. Five seeds and the median; **last reading | mean of the last three readings** (lines 500,000, 550,000, 600,000). Subtraction: usual score / clean score (held-out pairs whose written line no shown pair writes: 80 to 116 of the 500), medians at the last reading.

| System | Seeds | No old line: addition | subtraction | One old line in 32: addition | subtraction |
|---|---|---|---|---|---|
| candidate (mine) | reported | 89.6, 78.8, 78.6, 85.8, 87.0; **85.8 \| 85.8** | 86.8 / 85.1 | 86.4, 81.6, 80.4, 86.0, 80.4; **81.6 \| 81.6** | 85.8 / 86.1 |
| candidate, as the line reports | reported | 89.6, 78.8, 78.6, 85.8, 87.0; 85.8 | 86.8 / - | 86.4, 81.6, 80.4, 86.0, 80.4; 81.6 | 85.8 / - |
| plain (mine) | reported | 2.0, 1.6, 1.8, 1.4, 2.0; **1.8 \| 1.9** | 86.0 / 83.7 | 20.8, 12.2, 15.6, 17.2, 9.8; **15.6 \| 11.7** | 85.8 / 82.3 |
| the program's choice, as the line reports | reported | 4.4 | 74.0 | 67.0 | 75.4 |
| candidate (mine) | 2001 to 2005 | 74.0, 87.8, 77.4, 85.6, 91.8; **85.6 \| 85.6** | 89.6 / 91.4 | 76.2, 86.8, 80.4, 83.8, 91.6; **83.8 \| 83.1** | 90.4 / 91.4 |
| plain (mine) | 2001 to 2005 | 2.4, 1.6, 2.8, 4.0, 1.8; **2.4 \| 2.5** | 87.8 / 87.1 | 13.4, 17.2, 15.6, 14.4, 12.0; **14.4 \| 11.6** | 85.8 / 85.3 |
| the program's choice (`loop_small_case/cells2.py reserve=128`, run by me in a copy) | 2001 to 2005 | 4.8, 4.8, 5.6, 9.4, 4.0; **4.8 \| 6.3** | 70.0 / - | 68.2, 78.4, 66.2, 71.0, 55.4; **68.2 \| 63.2** | 73.6 / - |

Addition after its own phase: candidate 83.2 (reported seeds) and 85.6 (new seeds); plain 84.8 and 80.0. In all 20 lives of the candidate 128 cells have joined at every reading of the first phase and 256 at every reading of the second; at the end 128 cells may speak at the answer positions of held-out addition lines only, 128 of subtraction lines only, none of both. On seeds 2001 to 2005 the candidate with one old line in 32 is above the program's choice on five seeds of five (by 8 to 36 points). On seed 73129 my run of the program's choice gives the line's 5.0 and 75.2.

**By the measure of `PLAN_1.md`:** reported seeds, no old line: 85.8 against 85.8; one in 32: 81.6 against 81.6 and above 67.0. Seeds 2001 to 2005 against the line's figures for its own new seeds (84.8 and 81.2): 85.6 and 83.8, above the program's choice (68.2, mine). Repeats, in both conditions, on both sets of seeds, on both ways of reading the end.

Two things the table shows that were not asked. (i) For the plain system with one old line in 32 the last reading (15.6) is above the mean of the last three (11.7 and 11.6): a single last reading flatters the plain system there by about four points; the candidate's two figures agree. (ii) The clean score of subtraction is within 3 points of the usual one at the median in every row (seed by seed it differs by up to 13 points, on about a hundred lines). The scoring fault of the errata does not bear on the comparison.

## 3. What the agreement to the digit is, and is not

My program takes the birth values, the split, the text and the stream of lines from `cells2.py` as that program draws them (choices g and h of the plan), so on the same seed it lives the same life as the line's program if and only if the rules are the same. That every one of twenty figures on the reported seeds agrees to the digit therefore shows two things: the line's prose determines its program (I filled every gap of section 4 the way the line did, without seeing its code), and its program does what its prose says. It is **not** a second sample on those seeds: they are the same lives. The second sample is seeds 2001 to 2005. A fault shared by the plain system's own program (`cells2.py`: the scoring, the split) would be shared by both lines; I checked that no held-out addition line is written in training (asserted in every run) and scored subtraction cleanly.

## 4. Every choice the description left open

Taken as in `PLAN_1.md`, section 3; "matters" is from runs where run, otherwise my judgement.

| # | Choice | Taken | Does it matter |
|---|---|---|---|
| a | Bytes that count for youth and meetings | learned bytes only | **Run.** Not for the main claim (addition to the digit the same, 85.8 and 81.6; subtraction with old lines moves seed by seed, median the same). **Yes for the limit**: section 5. |
| b | Do the answering and return strengths of a silent cell fade | no | **Run. Decisive.** With fading: addition 0.0, 0.0, 0.0, 0.0, 0.0 with no old line and 12.0, 6.2, 20.4, 6.6, 11.6 (median 11.6) with one in 32; subtraction 86.8 and 86.8; the cells sort themselves as before (128 and 128). The result holds only under "no fading while silent". |
| c | Law of the constants | exp(-n / 10,000), from the fourth plan | not run; the line's fourth plan reports addition lost at a time constant of 1,000,000 |
| d | n before or after the byte | before | not run; a factor of exp(-1/10,000) |
| e | Running mean: rate 0.01, start 0, counts cells that spoke | as stated | not run |
| f | Order within a byte: speak, step, youth, joining | as stated | not run |
| g | Pool of 384 drawn as `cells2.py` draws 384 | as stated | not run; with this the figures are the line's to the digit |
| h | Stream drawn as the unbatched loop of `cells2.py` | as stated | as g |
| i | A young cell at a reading speaks | as stated | no cell is young at a reading of these runs |
| j | Count stops at 255, compared once at the end of youth | as stated | not run |
| k | Last reading and mean of last three | both given | see section 2 |
| l | The harsh text `aa=bb+ccc`, larger number first, last three digits scored | as stated | - |

Not in the description and taken from the line's plans: K = 128 with theta = 20 and one cell at birth (sixth plan); youth 4,500 and 30 meetings (seventh plan). I did not vary them.

**On choice b (my reading).** The candidate is described as three rules. There is a fourth ingredient inside the first: silence suspends forgetting. The plain system forgets everything slowly (fading 1e-5 a byte), and through 1.8 million bytes of subtraction that alone would remove addition from cells that are never touched. This is not a fault in the line's work, which states it, but the short form of the claim should carry it: "silent, unchanged, and not fading".

## 5. The limit: a change of task with no new byte

Subtraction written `aa=bb+ccc`. Five reported seeds; addition at the end, last reading | mean of last three (medians); subtraction; cells joined after the change.

| System | No old line: addition | subtraction | joined | One old line in 32: addition | subtraction | joined |
|---|---|---|---|---|---|---|
| plain | 2.2, 0.2, 0.6, 2.0, 1.4; **1.4 \| 0.9** | 83.8 | - | 20.2, 11.2, 13.6, 29.8, 16.6; **16.6 \| 19.2** | 82.2 | - |
| candidate, youth counts every byte (the line's reading for this test) | 0.2, 1.0, 0.4, 0.0, 0.4; **0.4 \| 0.3** | 75.2 | 0 | 5.2, 13.4, 19.4, 4.2, 14.2; **13.4 \| 12.4** | 70.6 | 0 |
| candidate as described (learned bytes) | 1.0, 0.0, 0.0, 0.0, 0.0; **0.0 \| 0.0** | 68.8 | 30 | 8.8, 4.2, 12.2, 4.0, 12.0; **8.8 \| 9.1** | 38.0 | 29 |

1. **The limit repeats**: under either reading the candidate keeps no more of addition than the plain system (the line: 0.4 against 1.4, and 13.4 against 16.6; mine under the same reading: the same figures).
2. **Under the description as written the swapped text does bring two lines never met** (`=` four bytes back, `+` seven bytes back: in addition they fall only on the byte from which nothing is learned). The old cells are then silent at two learned bytes of nine, 29 or 30 cells join, and the outcome is worse than plain: addition is lost and subtraction is learned less well (68.8 with no old line; 38.0 with one in 32, against 82.2 plain). So a change that is announced by a new line at *some* bytes only is not a mild case of the candidate's good behaviour; it is its worst case here. The line saw the two lines (its eighth plan) and set them aside for its test; the consequence of leaving them in was not reported and is new here.
3. In both readings the candidate learns the swapped subtraction less well than the plain system (75.2 and 70.6 against 83.8 and 82.2). I have not looked into why.

## 6. Predictions of `PLAN_1.md`, scored

P1 (0.75) held: 85.8. P2 (0.60) held: 81.6, above 67.0. P3 (0.80) held: 128 and 128 in 20 lives of 20. P4 (0.90) held: 1.8 and 15.6. P5 (0.55) **failed**: every seed agrees to the digit. P6 (0.65) held: 30 and 29 cells join; addition 0.0 and 8.8. P7 (0.70) held: no cell joins; 0.4 against 1.4 and 13.4 against 16.6. P8 (0.85) held: 0.0. P9 (0.70) held at the medians (largest gap 3.5 points, plain with one old line in 32).

## 7. When I opened the line's code

After my first full result (candidate and plain, both conditions, all ten seeds: commit 6a13de7) had been committed, pushed and reported to the coordinating session, and after the harsh-change runs of seed 73129 had finished. I then read, in `nano/keep/kage.py`, the header lines on `meet`, `openall` and `subtext`, the few lines that count meetings, and the text function of `subtext=swap`; and in `nano/keep/PLAN_8.md` the paragraph on test 3. Purpose: to learn why the line reports that no cell joins on the swapped text where mine join. Answer: for that test alone the line counts a young cell's lines at the unlearned byte as well (`openall=1`), and says so. My other reading of choice a also advances the cell's age at that byte (a youth of 450 text lines, not 500); the figures agree with the line's all the same. Nothing of my program was changed after the line's code was opened. `nano/keep/kage.c` and the other files of that folder were not opened.

## 8. What this does not show

- One pair of tasks, one order, one size, development level; ten seeds, of which five are the line's own lives over again.
- I repeated the final candidate only: not the steps that led to it, not the other order, not one old line in 8 (where the line reports 76.8 against the plain system's 71.2), not a third task.
- K, theta, the youth and the threshold were taken from the line and not varied; choices c to f and j were not run.
- The limits of the line's report (its section 4) are untouched by a repeat: keeping here is by separation, nothing the first task formed is used for the second, and the change is seen only through a line never met. Section 5 adds that a change announced at only some bytes does harm.
- `rep.py` is my reading of a description. It is small (kernel 60 lines) and was checked against the plain system and against its own numpy form; no third party has read it.

## 9. Files, commands, cost

- `PLAN_1.md` (before any run). `rep.py` (numpy, the reference, and the wrapper), `kern.c` (compiled; built to `kern.so`, not committed), `check.py` with `check_output.txt`: my numpy with no rule against `cells2.py` 8e-15; kernel against numpy on eight short lives with the rules on, strengths to 2e-13, counts, marks, ages, joined cells, running mean and scores identical. One full life in numpy (seed 73129, one old line in 32, 546 CPU seconds; `res/np_cand_reh32_73129.json`): the twelve readings are the kernel's to the digit.
- `jobs1.txt` (main, 40 lives), `jobs2.txt` (harsh, 30), `jobs3.txt` (other readings, 20), `runjobs.sh`, `tab.py`; `table1.txt`, `table1_new.txt`, `table2_harsh.txt`, `table3_other_readings.txt`; results in `res/` (91 files); the program's choice in `res_reserve/` (12 files, made by `loop_small_case/cells2.py ... reserve=128 fast=1 [reh=32]` in a copy outside the repository; nothing in `loop_small_case` was changed).
- The candidate: `python3 rep.py <label> seed=<s> arm=cand [reh=32] [harsh=1] [allbytes=1] [fade_silent=1]`; about 40 CPU seconds. Tables: `python3 tab.py m_cand_reh0 m_cand_reh32`, with `SEEDS=2001,2002,2003,2004,2005` for the new seeds.
- Cost: about 100 compiled lives and one in numpy, about 1.3 CPU hours, under an hour on the clock.

## 10. Decision record

- Anchor: `loop_small_case/cells2.py` at 4026bfa (imported, unchanged); the claim as in `nano/keep/REPORT.md`.
- Question: does the keeping line's final candidate repeat with another program written from its description.
- Falsifier (set beforehand): under 20% kept in either condition, or medians more than 10 points from the line's, or not above the program's choice with one old line in 32. Not met.
- Result: repeats (section 2); holds only with "no fading while silent" (section 4, b); the limit repeats and is wider than stated (section 5).
- Disposition (mine, for the coordinating session to weigh): the line's condition for entering K1, a repeat by another line with its own program, is met at development level. I would enter it with three things beside it: the line's limit 1; "silent cells do not fade" as part of the rule; and that the repeat on the reported seeds is the same lives, the new sample being seeds 2001 to 2005.
- One next step I would propose: the case of section 5, row 3: a change of task that brings a new line at some bytes only. It is where the candidate does worse than the plain system, and it is nearer to what real text does than either a wholly new byte or none.
