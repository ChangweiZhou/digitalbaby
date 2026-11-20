# Line `nano/life3`, first plan: a life of three tasks, the three routes of keeping side by side (20 November 2026)

By a separate agent session started by the coordinating session (`session_017dSEpVed7SthEH3ngGAdNc`); folder `nano/life3/`, branch `claude/nano-life3-8mmhfm`. **Development level.** No checker gate, seal or outside review is claimed. Readings and predictions marked "mine" are this session's.

**Authorisation.** Mr. Xu, 20 November 2026, as given to this session by the coordinating session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." And earlier, in English: "Keep experimenting with the agents." The question of this line was set by the coordinating session: Mr. Xu asked whether the system can really learn continually within some small range, and no line has run more than two tasks.

**Source.** `claude/friendly-darwin-8mmhfm` at commit 7016262 (the checkout this branch starts from). Read before this plan: `AGENTS.md`; `ROADMAP_SECOND_REVISION_20261120.md` (row K1, sections 5 and 7); `loop_small_case/REPORT.md`, `cells2.py`, `ERRATA_20261120.md`; `nano/keep/REPORT.md`, `kage.py`, `kage.c`; `nano/replay/REPORT.md`, `replay.py`; `nano/sparse/PLAN_1.md`, `PLAN_2.md`, `PLAN_5.md`, `sparse.py`; `nano/keep2` through the roadmap's account of it.

**What was copied, all from that commit.** Unchanged: `nano/replay/kernr.c`, `nano/replay/fastr.py` (the compiled relay system), `nano/keep/kage.c` (the compiled kernel of route 1), `nano/sparse/sparse.py` with the `kern_s.c` it writes (route 3, whole). `nano/keep/fastage.py` with one import line changed (it took its ctypes names from a file not copied). Written again in `life3.py`, from the sources named: the relay system and the counting cells with their replay (`nano/replay/replay.py`), the rules of route 1 for the options of its final candidate only (`nano/keep/kage.py`), the text and the split (`cells2.py`). `life3.py` is the numpy reference for the relay system and for route 1; `sparse.py` holds its own numpy reference.

## 1. The life, the tasks, the score

*Lines.* Ten bytes, as `cells2.py`: `aa+bb=ccc` and a newline. Task 1, addition. Task 2, subtraction `aa-bb=ccc`, the larger number first. Task 3: a third operator byte that no system has met, see section 2. The symbol set is that of `cells2.py` and one further byte (fifteen byte values).

*Life A.* 400,000 lines of task 1, then 200,000 of task 2, then 200,000 of task 3. The program shows nothing of an earlier task again (except in the arm that is the upper reference, where that is the arm), and tells no system that the task has changed.

*Split.* For addition and subtraction the split of `cells2.py`, from the same generator: of the 10,000 ordered pairs 8,000 are shown and 2,000 never; 500 of these are scored. The third task has a split of its own made the same way from a generator of its own, so that the first two phases are, line for line, the lives the other lines lived.

*Score.* As `cells2.py`: each of the three answer bytes named with the line's own bytes before it; a pair counts if all three are right; %. Addition and the third task: the 500 never-shown pairs (no never-shown pair of these tasks writes a line that a shown pair writes: the numbers are written in the order of the pair). **Subtraction: the clean score**, on the never-shown pairs, of all 2,000, whose written line no shown pair writes (about 400; `ERRATA_20261120.md`, item 1); the old score on the 500 is kept beside it in the files and is not used for any claim. For route 3 the score with the system's own bytes written before each next one is kept beside it in the files.

*Readings.* Every 50,000 real lines, every task of the life scored at every reading. Claimed: at the end of each phase, the last reading, with the mean of the last three readings of that phase beside it; the median of the five reported seeds, each seed listed.

*Seeds.* Development seed 1 for every choice. Reported: 73129, 93683, 116089, 406364, 545353, once each.

## 2. The third task, and how it is chosen (step 0, development seed 1 only)

It must be neither addition nor subtraction, and the plain system alone, from birth, must reach at least 70% of never-shown pairs within 200,000 lines (last reading; the mean of the last three is reported beside it). Candidates, tried in this order on seed 1, plain system, `plan=<task>:200000`; **the first that passes is taken and the others are not looked at for the choice**:

1. `max`: `aa^bb=ccc`, the larger of the two numbers (`07^85=085`). My reason for putting it first: it is a different kind of rule from the first two (a comparison and a copy, no carry), so a third task that is not a near relative of the first; its answer digits are still digits of the same places.
2. `sat`: `aa&bb=ccc`, the sum plus the first number's tens digit.
3. `srv`: `aa#bb=ccc`, the first number plus the second with its digits exchanged (addition with the roles of two delays exchanged).

All three candidates are run (30 seconds), so that the levels of the two not chosen are on file for a later fourth task; the choice follows the rule above whatever they show.

## 3. The arms (five systems, one life each for each seed)

| Arm | System | Numbers that learn |
|---|---|---|
| P | the plain system: 256 logistic cells in between, one answering cell of 15 branches, a mismatch relayed once (`cells2.py`; eta 1/4 and 1/16, fading 1e-5) | 256 x 151 + 15 x 257 + 256 x 15 = 46,351 |
| O, the upper reference | P; after each real line the program shows one shown line of an earlier task, an earlier task chosen with equal chance (in the first phase, where there is none, a line of the first task, as the control of `nano/replay` did) | as P |
| R, route 2 | P with cells that count runs of up to 8 bytes of the real lines; after each real line one replayed line drawn byte by byte from the longest active counting cell; the relay system learns from it as from a real line (`nano/replay`, r = 1, K = 8) | as P, and the counts (about 66,000 cells after two tasks there; reported) |
| K, route 1 | the final candidate of `nano/keep`, unchanged: 384 cells of which one is present at birth; cells join while the answering cell hears fewer than 128 (theta 20); a strength from a line opens if the line was active on at least 30 of the cell's first 4,500 learned bytes; a cell with a closed strength on an active line is **silent, unchanged and not fading**; the answering cell's constants become still (exp(-n/10,000)) | 384 x 151 + 15 x 385 + 384 x 15 = 69,519, of which a third wait |
| S, route 3 | cells that count, keyed on at most four of the delays 1 to 9 (255 kinds), read with a learned trust for each kind and level of experience; eta 0.1, e 2, trusts born at 0.02, the step of a trust shrinking with its use (`shrink=1`): the settings of `nano/sparse` for m = 4 | 10,200 trusts and the counts (1.4 million cells after two tasks there; reported) |

*Route 1's pool.* 383 cells wait at birth. In the two-task lives of `nano/keep` 128 joined at birth (the one present included) and 128 at the change. If the same happens at the second change, 128 join and none wait: the pool is exactly spent by three tasks. The number joined is recorded at every reading, and for each task the number of cells that may speak on its never-shown lines. What happens when the pool is spent is not seen in life A unless fewer than 128 remain; it is the subject of a fourth task (a later plan, if the budget allows).

*What the three routes are told.* Nothing about the tasks or the changes. The upper reference O is told both: it is the program that draws the old lines.

## 4. One question, one hypothesis

**Does each route still keep when there are three tasks?**

*Hypothesis.* At the end of life A (last reading, median of the five reported seeds) routes 1, 2 and 3 each have the first task at 60% or more of never-shown pairs, the second at 60% or more (clean score), and the third learned to 60% or more.

*A route is refuted for three tasks* if either earlier task is under 30% at the end. Between the two (an earlier task at 30 to 60, or the third task under 60 with both earlier tasks kept) the hypothesis fails for that route without the route being refuted, and I say which clause failed.

*Checks that the program is the other lines' program* (not hypotheses): at the end of the second phase, arms P, R, O and K on seed 73129 should give the figures in those lines' result files (P: addition 2.0, subtraction 82.0; R: 77.2 and 82.2; O: 88.4 and 75.4; K: 89.6 and 86.8), to the digit if the fifteenth byte value changes nothing before it is met. A difference of more than rounding is reported and looked into before anything is claimed.

## 5. Predictions (mine), written before any run of a life of three tasks

Disclosure: before this plan one run was made to see that the program runs, arm P on seed 73129 with `plan=add:100000,sub:50000`; it gave the readings of `nano/replay`'s file for that seed to the digit (68.2, 76.4, then 3.0 of addition). The check of the compiled kernels against numpy on lives of a few thousand lines (`check_life3.py`, `check_life3_output.txt`) had also printed its lines for the relay system and route 1 before the table below was written, and I had read them: after 1,600 lines of each task route 1 has addition at 0.0, subtraction at 0.4 and `max` at 41.6 (17.6 in a second case); the line for route 3 (after 100 lines of `max`: 39.2) I read after the table. So I knew, when I set T1, K3 and S3, that `max` begins to be learned within a few thousand lines; the chances stand as I wrote them with that knowledge. No life of three tasks at a length at which addition is learned has been seen.

| | Statement | Chance |
|---|---|---|
| T1 | step 0: `max` passes (70% or more at 200,000 lines, plain system alone, seed 1) | 0.70 |
| P1 | plain system at the end: addition under 10 and subtraction under 10 | 0.90 |
| P2 | plain system: the third task at 70 or more at the end | 0.70 |
| O1 | upper reference: addition and subtraction both 60 or more at the end | 0.75 |
| O2 | upper reference: the third task 60 or more | 0.60 |
| K1 | route 1: addition at the end 60 or more, and within 2 points of its level at the end of the first phase on at least four seeds | 0.80 |
| K2 | route 1: subtraction (clean) 60 or more at the end | 0.80 |
| K3 | route 1: the third task 60 or more | 0.60 |
| K4 | route 1: the whole hypothesis holds | 0.50 |
| K5 | route 1: refuted (an earlier task under 30) | 0.10 |
| K6 | route 1: on every seed 127 or 128 cells join at the second change and none at any other time after the first change | 0.70 |
| R1 | route 2: addition 60 or more at the end | 0.60 |
| R2 | route 2: subtraction (clean) 60 or more at the end | 0.40 |
| R3 | route 2: the third task 60 or more | 0.55 |
| R4 | route 2: the whole hypothesis holds | 0.20 |
| R5 | route 2: refuted | 0.20 |
| R6 | route 2: in the third phase, of the replayed lines, subtraction's are between 20% and 35% | 0.85 |
| S1 | route 3: addition 60 or more at the end | 0.65 |
| S2 | route 3: subtraction (clean) 60 or more at the end | 0.55 |
| S3 | route 3: the third task 60 or more | 0.40 |
| S4 | route 3: the whole hypothesis holds | 0.20 |
| S5 | route 3: refuted | 0.10 |
| A1 | at least one route fails the hypothesis | 0.85 |
| A2 | the route that fails most plainly (lowest of its three figures) is route 2 | 0.45 |

*Why (mine).* Route 1 keeps by separation: a third operator byte is a line never met at every learned byte, so the old cells should fall silent and a third set join; its risk is the third task itself, which 128 new cells must learn alone in 200,000 lines with constants that are still, and the pool, which is exactly spent. Route 2 replays in proportion to what was counted: at the start of the third phase two replayed lines in three are additions and one in three a subtraction, and by its end a quarter are of the third task; subtraction then has a quarter to a third of a replayed line for each real line, and `nano/replay` found on seed 1 that a quarter of a line kept 67.6 and an eighth 46.0 of one old task; and one set of 256 cells must hold three tasks. So I expect route 2 to keep addition and to lose part of subtraction: a failure that two tasks cannot show, since with two tasks the old task has the whole replay. Route 3 keeps counts apart by the operator where the operator is among a cell's four delays, and its trusts, by then used millions of times, hardly move: I expect the first two tasks about where they were, and the doubt is whether the third task is learned by trusts that no longer move, and whether its level on addition is itself above 60 (the line reported 67 to 81 for m = 4).

## 6. Cost, and what is not run

Per life of 800,000 real lines, on one core: P about 40 seconds; O and R about 2 minutes (1.6 million lines read); K about 80 seconds; S about 8 minutes and under 300 MB. Step 0: 30 seconds. Seed 1 for all five arms (to see that every arm runs to the end; no setting is chosen from it, all settings being the other lines'), then the five reported seeds: about 70 CPU-minutes in all, about 20 minutes on four cores. No checkpoints are kept; result files (JSON) hold every reading.

Not run in this plan: other orders of the three tasks; a fourth task; the third task written in the first task's byte values (the harsh case for route 2); other ratios of replay; route 1 with a larger pool; route 3 with three delays or without the shrinking step; further seeds. Each of the first three is a later plan with one change, if the budget allows.

*Decision that this serves.* Whether K1 of the roadmap (keeping) can be said of more than a pair of tasks for any of its three routes, and which route to carry to a longer life.

---

## 7. Outcome (20 November 2026; development level; readings are mine)

Runs: step 0 (three lives of 200,000 lines, seed 1), seed 1 for the five arms, the five reported seeds for the five arms (25 lives of 800,000 real lines); about 55 CPU-minutes. Files: `res/c0_*`, `res/d1_*`, `res/a1_*`; `table1.txt` (reported seeds, every seed listed), `table1_dev.txt` (seed 1); `check_life3_output.txt` (seven cases, compiled against numpy: largest difference 1.8e-12, every reading the same).

**Step 0.** `max` passes on seed 1: the plain system alone has 88.4% of never-shown pairs at 200,000 lines (mean of the last three readings 84.7). It is the third task. The two others, on file only: `sat` 15.4, `srv` 52.8 (neither passes; I have not looked into why `srv`, which is addition with two delays exchanged, is learned so much more slowly than addition on this seed).

**The checks against the other lines' files** (end of the second phase, seed 73129): P 2.0 and 82.0 (clean 81.3); R 77.2 and 82.2 (clean 79.8); O 88.4 and 75.4 (clean 70.2); K 89.6 and 86.8 (clean 86.2). All equal those lines' files to the digit. The first two phases of this life are the lives those lines lived.

**Life A, at the end of each phase.** Never-shown pairs, whole answer, %; median of the five reported seeds at the last reading; in brackets the median of the mean of the last three readings of the phase. Subtraction on the clean score (405 or 406 pairs a seed).

| Arm | after addition: add | after subtraction: add / sub | at the end of the life: add / sub / max |
|---|---|---|---|
| P plain | 84.8 [84.2] | 1.8 / 86.0 [1.9 / 82.5] | 2.8 / 1.6 / 70.4 [3.1 / 1.6 / 63.5] |
| O the program's old lines (upper reference) | 86.6 [86.1] | 84.4 / 71.9 [84.5 / 68.4] | 78.2 / 63.2 / 64.2 [77.7 / 63.8 / 59.5] |
| K route 1, cells that join | 83.2 [84.5] | 85.8 / 86.2 [85.8 / 87.2] | **85.8 / 86.2 / 90.8** [85.8 / 86.2 / 90.3] |
| R route 2, replay | 86.8 [86.3] | 77.2 / 78.9 [80.6 / 72.4] | **73.4 / 52.4 / 64.2** [75.4 / 54.7 / 60.6] |
| S route 3, cells that count | 55.8 [59.1] | 69.2 / 66.4 [71.9 / 66.6] | **65.2 / 65.0 / 97.4** [66.7 / 63.7 / 97.6] |

At the end of the life, each seed (73129, 93683, 116089, 406364, 545353): K addition 89.6, 78.8, 78.6, 85.8, 87.0; subtraction 86.2, 78.4, 83.0, 91.3, 92.0; max 88.2, 90.8, 90.4, 91.6, 92.2. R addition 74.0, 76.0, 72.2, 72.4, 73.4; subtraction 53.2, 52.5, 44.8, 52.4, 50.4; max 64.4, 60.2, 59.2, 73.2, 64.2. S addition 65.6, 69.2, 65.2, 64.4, 64.4; subtraction 65.0, 66.2, 63.4, 63.1, 65.9; max 97.4, 98.4, 97.6, 97.4, 97.0.

**The hypothesis: holds for routes 1 and 3; fails for route 2 on its second clause; no route is refuted.**

- *Route 1* keeps both earlier tasks where they stood and learns the third to 90.8. On every seed 128 cells join in each phase, the last of them 13,342 or 13,464 learned bytes into the phase, and none at any other time; at the end no cell waits; 128 cells may speak on each task's never-shown lines and no cell on two tasks' lines. The pool of 383 is exactly spent by three tasks.
- *Route 2* keeps addition (73.4) and learns the third task (64.2), and **loses a third of subtraction**: 52.4 at the end against 78.9 after its own phase, under 60 on five seeds of five (44.8 to 53.2), under 60 also as the mean of the last three readings (54.7). Not under 30, so not refuted.
- *Route 3* has all three at 60 or more (65.2, 65.0, 97.4), on every seed, on the last reading and on the mean of three. With its own bytes written before each next one the figures are the same.

**Predictions scored.**

| | Statement | Chance given | Outcome |
|---|---|---|---|
| T1 | `max` passes step 0 | 0.70 | held (88.4) |
| P1 | plain: addition and subtraction under 10 at the end | 0.90 | held (2.8, 1.6) |
| P2 | plain: third task 70 or more at the end | 0.70 | held by its letter (70.4; three seeds of five at 70 or more; mean of three readings 63.5) |
| O1 | upper reference: both earlier tasks 60 or more | 0.75 | held (78.2, 63.2; subtraction under 60 on one seed, 55.2) |
| O2 | upper reference: third task 60 or more | 0.60 | held (64.2; mean of three readings 59.5) |
| K1 | route 1: addition 60 or more and within 2 points of its level after the first phase on four seeds | 0.80 | held (85.8; differences -0.2, -0.4, -0.4, +2.6, +0.2) |
| K2 | route 1: subtraction 60 or more | 0.80 | held (86.2) |
| K3 | route 1: third task 60 or more | 0.60 | held (90.8) |
| K4 | route 1: the whole hypothesis | 0.50 | held |
| K5 | route 1: refuted | 0.10 | did not happen |
| K6 | route 1: 127 or 128 cells join at the second change, none at other times | 0.70 | held (128 on every seed) |
| R1 | route 2: addition 60 or more | 0.60 | held (73.4) |
| R2 | route 2: subtraction 60 or more | 0.40 | **failed** (52.4) |
| R3 | route 2: third task 60 or more | 0.55 | held (64.2; one seed at 59.2) |
| R4 | route 2: the whole hypothesis | 0.20 | did not hold |
| R5 | route 2: refuted | 0.20 | did not happen |
| R6 | route 2: subtraction's share of the replayed lines in the third phase 20 to 35% | 0.85 | held (29% on every seed) |
| S1 | route 3: addition 60 or more | 0.65 | held (65.2) |
| S2 | route 3: subtraction 60 or more | 0.55 | held (65.0) |
| S3 | route 3: third task 60 or more | 0.40 | held (97.4); I was wrong to doubt it |
| S4 | route 3: the whole hypothesis | 0.20 | **held, against my chance** |
| S5 | route 3: refuted | 0.10 | did not happen |
| A1 | at least one route fails | 0.85 | held |
| A2 | the route that fails most plainly is route 2 | 0.45 | held (the only one that fails) |

**Readings (mine).**

1. **Route 2's failure is the one two tasks could not show: replay is shared out by what was counted, and the middle task gets the least.** In the third phase 58% of the replayed lines are additions, 29% subtractions and 14% lines of the third task itself (every seed; every replayed line is a line that was shown). Subtraction has 0.29 of a replayed line for each real line and falls from 78.9 to 52.4; addition has 0.58 and falls from 77.2 to 73.4. The upper reference, which gives each old task half a line, ends with subtraction at 63.2 and addition at 78.2. So about half of route 2's loss of subtraction against the reference goes with its smaller share, and the rest of the fall is one that the reference has too (71.9 to 63.2): three tasks in one set of 256 cells. A life with more tasks, or with tasks of unequal length, makes this worse by the same arithmetic: the share of a task in replay is its share of all the lines ever read. This is a property of the replay rule and not of the store, which holds every line.
2. **Route 1 keeps three tasks as it kept two, by the same separation, and its pool is now empty.** Nothing in life A shows what a fourth task meets. The third task is learned by 128 fresh cells to 90.8, where the plain system, carrying two earlier tasks in the cells that must learn it, reaches 70.4 (and alone, on seed 1, 88.4). Read with the limit the keeping line states: every change here is announced by a byte never met, at every learned byte.
3. **Route 1's old cells are not exactly untouched at a change.** Addition after the first phase and after the second differ by up to 2.6 points on one seed (83.2 and 85.8 on seed 406364), and subtraction moves by up to 1 point at the second change, although the rule says silent and unchanged. A look inside (`look_k.py`, `look_k_406364.txt`): from the end of addition to 50,000 lines of subtraction the largest change of any strength of the first 128 cells is 0.0027 (strengths from lines; their largest size 11) and 0.0014 for the answering cell's strengths from them (largest size 0.146); the constants do not move. Addition reads 83.2 at the end of its phase, 84.4 after 1,000 lines of subtraction (while cells of the second set are still young and may speak on any line) and 85.8 from then on. Reading: the old cells learn at the first two bytes of the first line of the new task, where the new operator is not yet on the delay lines, and nowhere else; two steps of the usual size move the answer on some dozen borderline pairs. It is a small effect and it went the good way on this seed; it means the score of a kept task can move by a point or two at a change with no forgetting behind it.
4. **Route 3 holds, with thin margins on the two old tasks and an odd history on the first.** Addition falls while addition alone is read (about 76 at 50,000 lines, 55.8 at 400,000, on every seed, with the shrinking step), rises to 73 to 77 within 50,000 lines of subtraction with no addition shown, and then wears: 69.2 at the end of subtraction, 65.2 after the third task. Subtraction on the clean score wears from about 71 at 50,000 lines of its own phase to 66.4 at its end and 65.0 after the third task, while on the old score (mostly shown lines) it reads 87 and then 75: these cells hold shown lines exactly, and the old score flatters them, as the errata warned. The third task is nearly perfect (97.4) within 50,000 lines. So the route keeps, but its level on addition and subtraction is 20 points under route 1's and within 5 points of the 60 asked; a harder criterion would have failed it, and I do not read 65 as safe.
5. **The upper reference is not an upper bound.** Route 1 is above it on all three tasks and route 3 above it on the third. It bounds only routes that hold three tasks in the plain system's one set of cells.
6. **An earlier life costs the plain system the third task**: 70.4 after two tasks against 88.4 alone (seed 1 for the second figure; not run on the reported seeds).

**What this does not show.** One order of the tasks; one third task, and an easy one (a comparison and a copy; a third task with a carry was not learnable alone within 200,000 lines by the candidates tried); phases of fixed length; five seeds; settings all taken from the other lines, none tuned for three tasks; the score with the line's own bytes before each answer byte. Route 1's keeping rests on the new operator byte, as the keeping line says of two tasks. For route 3 the claim is on margins of about 5 points. Nothing here is a claim that any of these systems learns continually in general.

**Next (each one change from life A, each with its own plan committed before its runs):** `PLAN_2.md`, a fourth task (where route 1's pool is spent and route 2's shares thin further); `PLAN_3.md`, the harsh case for route 2 (the third task written in the first task's byte values); `PLAN_4.md`, the other orders of the three tasks for route 1, which did best.
