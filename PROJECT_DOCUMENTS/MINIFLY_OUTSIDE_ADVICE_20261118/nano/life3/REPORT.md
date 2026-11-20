# `nano/life3`: lives of three and four tasks, the three routes of keeping side by side. Report of 20 November 2026

By a separate agent session started by the coordinating session (`session_017dSEpVed7SthEH3ngGAdNc`); folder `nano/life3/`, branch `claude/nano-life3-8mmhfm`. Under Mr. Xu's words of 20 November as given to this session (in Chinese; the coordinator's translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well."; and earlier, in English: "Keep experimenting with the agents." **Development level throughout.** Nothing here has been repeated by another session; no checker gate, seal or review is claimed. Six plans, each committed before its runs, each with its outcome and every prediction scored (`PLAN_1.md` to `PLAN_6.md`); this report adds nothing that is not in them. Readings are mine.

## 1. The question

Three routes were found on 20 November by which a system keeps two-digit addition through subtraction: cells that join for lines never met (route 1, `nano/keep`), replay of lines made from the system's own counting cells (route 2, `nano/replay`), cells that count keyed on at most four delays (route 3, `nano/sparse`). Each was shown on two tasks. **Does each still keep when there are three tasks, and four?**

## 2. The lives

Lines of ten bytes. Addition `aa+bb=ccc` for 400,000 lines, subtraction `aa-bb=ccc` for 200,000, then a third task for 200,000: `max`, `aa^bb=ccc`, the larger of the two numbers, with an operator byte never met (chosen on seed 1 by a rule fixed beforehand: the plain system alone reaches 88.4% of never-shown pairs in 200,000 lines; the two candidates with a carry did not pass). Life B adds 200,000 lines of `min` (`aa_bb=ccc`). Life C writes the third task in addition's byte values (`aa=bb+ccc`). The program shows no old line and tells no system of a change. Scored on never-shown pairs, whole answer, with the line's own bytes before each answer byte; subtraction on the pairs whose written line no shown pair writes (about 405 a seed). Median of the five reported seeds, last reading; the means of the last three readings and every seed are in the plans and tables.

The systems are the other lines', unchanged, with one or two further byte values: at the end of the second phase the plain system, route 1, route 2 and the reference equal those lines' result files to the digit (seed 73129).

## 3. The answer in one table

| System | three tasks (life A): add / sub / max | four tasks (life B): add / sub / max / min | third task in addition's bytes (life C): add / sub / third |
|---|---|---|---|
| plain | 2.8 / 1.6 / 70.4 | 0.0 / 0.5 / 3.0 / 83.0 | 1.6 / 1.6 / 76.0 |
| the program's old lines, one for each real line (reference) | 78.2 / 63.2 / 64.2 | 66.0 / 58.3 / 62.0 / 60.8 | 65.2 / 46.6 / 56.4 |
| route 1, cells that join (384 cells) | **85.8 / 86.2 / 90.8** | 85.8 / 86.2 / 91.0 / **0.0** | **0.0** / 86.2 / 78.8 |
| route 1 with 512 cells | | **83.0 / 90.5 / 91.2 / 93.2** | |
| route 2, replay, one line for each real line | 73.4 / **52.4** / 64.2 | 64.0 / **42.7** / **55.2** / 62.4 | 58.8 / **36.3** / 65.2 |
| route 3, cells that count, four delays | **65.2 / 65.0 / 97.4** | **63.2 / 61.0 / 96.8 / 96.0** | 71.6 / 58.4 / 97.0 |

The hypothesis of the first plan (each task at 60 or more at the end; a route refuted if an earlier task is under 30) **holds for routes 1 and 3 on three tasks and fails for route 2**, which loses a third of the middle task. No route is refuted by the median on any life except route 1 on life C, which loses the first task entirely.

## 4. What was found

1. **Route 1 learns continually up to the size of its pool and no further.** Every 128 waiting cells take one task; on every seed and in every life with a new operator 128 cells join at each change and at no other time, and no cell speaks on two tasks' lines. Three tasks: kept to the digit or within 1.4 points, in all six orders (`PLAN_4.md`; every task 79 or more). Four tasks with 384 cells: the pool is empty, every cell is silent on the fourth operator, and the fourth task stands at 0.0 on every seed for 200,000 lines while the first three are untouched. With 512 cells all four stand at 83 to 93. It does not forget and, once the pool is spent, does not learn; cells are never given back; nothing one task formed serves another. Its limit as the keeping line stated it shows here with a kept task beside it: a third task that brings no new byte value (life C) silences addition's cells at some bytes only, and addition is lost entirely (0.0 on every seed) while subtraction, whose cells never met a '+', is kept to within 1.4 points.
2. **Route 2 fails in a way two tasks could not show.** Replay is shared out in proportion to what was counted, so in the third phase 58% of replayed lines are additions, 29% subtractions and 14% the third task's; subtraction falls from 78.9 to 52.4 (under 60 on five seeds of five), and with a fourth task to 42.7, with `max` at 55.2. Twice the replay gives the middle task about 8 points and does not restore it (56.9; `PLAN_5.md`): over four arms subtraction's level follows its part of all lines the relay system reads (14% gives 52, 17% gives 54, 19% gives 57, 25% gives 63), and under replay in proportion to counts that part cannot be raised much by replaying more. Sharing the replay equally with no knowledge of tasks (equal chance among the bytes that followed a run) does not help: an equal share of one line is a third, a third of it goes to the task being read, and addition loses 10 points. The program's own old lines, one for each real line, are themselves at 58 to 66 on four tasks: one set of 256 cells at this rate of rehearsal is near its limit there whatever chooses the lines.
3. **Route 3 holds three tasks and four, on thin margins that wear.** Addition 69.2, 65.2, 63.2 and subtraction 66.4, 64.5, 61.0 at the ends of the last three phases of life B; the two tasks without a carry stay at 96 to 97. Its level on the two tasks with a carry is some 20 points under route 1's. On the old score of subtraction (mostly shown lines) it reads 87 and then 75 where the clean score reads 66 and 65: the old score flatters cells that hold shown lines exactly. Its addition falls while addition alone is read (about 76 at 50,000 lines, 55.8 at 400,000, with the shrinking step) and rises again within 50,000 lines of subtraction; I have not explained that.
4. **The harsh third text is passed whole by no route** (life C): route 1 loses the first task, route 2 has the second at 36.3, route 3 the second at 58.4. It is dear to every system that holds the tasks in one set of cells, the reference included (65.2 / 46.6), and the task with the least rehearsal pays most.
5. **Replay chosen by the relay system's own mismatch** (`PLAN_6.md`, an idea of mine from the fifth outcome): its five lives were still running when this report was first committed; see the plan's outcome.

## 5. For the question whether the system can learn continually within some small range (my reading)

On tasks of this size that announce themselves by a byte never met: **route 1, yes, for exactly as many tasks as its pool holds** (one for every 128 cells, about 24,000 numbers a task), by keeping the tasks in separate cells. **Route 3, yes for four tasks at a modest level** (61 to 63 on the tasks with a carry), wearing by 2 to 4 points a task. **Route 2 at one replayed line for each real line: two tasks well, three with the middle one damaged, four with two damaged.** On a task that brings no new byte value none of the three is whole. None of this is continual learning in general: the tasks are four, short, of one kind, and three of the four changes are announced by a new byte.

## 6. What is not shown, and the limits

One third task and one fourth, both without a carry (a third task with a carry could not be learned alone in 200,000 lines by either candidate tried, so none was used); phases of fixed length; five seeds; one development seed; every setting taken from the other lines and none tuned for more than two tasks; the score with the line's own bytes before each answer byte (for route 3 the score with its own bytes is in the files and is the same). The orders were run for route 1 only. Route 3's margins over the 60 asked are 1 to 5 points; a stricter bar would have failed it. The "reference" is not an upper bound: route 1 is above it everywhere. Not run: a fifth task; unequal phases; routes 2 and 3 in other orders; route 1 with any way of giving cells back; anything on prose.

## 7. Errors of mine, as they were found

- `PLAN_1.md`: the first draft of the disclosure said no figure of a third task had been seen before the predictions; I had read the kernel check's lines (route 1: `max` at 41.6 after 1,600 lines). Corrected before the plan was committed; the chances were left as written.
- `PLAN_1.md`, reading 4: a range typed from memory (69 to 73) corrected against the files (73 to 77) in the next commit.
- Commit 6497183 holds a cut-off `check_life3_output.txt` (the check was being run again after `life3.py` changed, and I committed while it wrote); the next commit holds the whole file. The check passed both times.
- My first reading of route 2's failure (the share of replay) was half right: `PLAN_5.md`.
- I gave route 3 a chance of 0.20 and then 0.30 of holding whole on three and on four tasks; it held both times.

## 8. The decision record

- Source: `claude/friendly-darwin-8mmhfm` at 7016262. Copied unchanged from it: `nano/replay/kernr.c`, `fastr.py`; `nano/keep/kage.c`; `nano/sparse/sparse.py` and its `kern_s.c`. `nano/keep/fastage.py` with one import line changed. Written again in `life3.py` from the sources: the relay system, the counting cells and replay, the rules of route 1's final candidate, the text and split.
- Systems and sizes: plain system 256 cells, 46,351 numbers that learn with fifteen byte values; route 1 384 cells (69,519), or 512; route 2 the plain system and about 104,000 counting cells after three tasks (142,000 after four); route 3 10,200 trusts and about 1.52 million cells that have counted after three tasks (1.67 million after four).
- What ran: 129 lives of 200,000 to 1,000,000 real lines on the reported seeds and seed 1, a look inside route 1, the kernel check twice; about 4.7 CPU-hours; no checkpoints kept; every reading is in `res/*.json`.
- Result: section 3. Lesson: sections 4 and 5.
- Disposition (mine, for the coordinating session to weigh): K1 of the roadmap may be said of three tasks for routes 1 and 3 and of four for route 1 with a pool to match and for route 3, at development level, each with its limit beside it; not for route 2 beyond two tasks at this ratio.
- Next actions I would propose: for route 2, what `PLAN_6.md`'s outcome says; for route 1, some way for cells to be given back or shared, since its range is its pool; for route 3, why addition falls while it alone is read.

## 9. Files, commands, cost

`life3.py` (driver and numpy references; its header states every arm and option), `check_life3.py` and `check_life3_output.txt` (seven cases, compiled against numpy: largest difference 1.8e-12, all readings the same), `tab.py`, `runjobs.sh`, job lists `jobs*.txt`, tables `table1.txt` to `table6.txt` and `table1_dev.txt`, `look_k.py` with `look_k_406364.txt`, results in `res/`. A life: `python3 life3.py <label> arm=K seed=73129 [plan=add:400000,sub:200000,max:200000]`; a step: `sh runjobs.sh jobs1_rep.txt 4`; a table: `python3 tab.py a1`. One life of 800,000 real lines takes about 40 seconds for the plain system, 85 for routes 1 and 2, 5.5 minutes for route 3. Compiled `.so` files are not committed.
