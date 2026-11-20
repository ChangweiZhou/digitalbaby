# Keeping (roadmap K1): where a new task is written, settled by the cells. Report of 20 November 2026

By the agent session that owns line K1, folder `nano/keep/`, branch `claude/nano-keep-8mmhfm`. Started by the coordinating session under Mr. Xu's words of 20 November as recorded in `ROADMAP_SECOND_REVISION_20261120.md` ("revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process"). **Development level throughout.** Every setting was chosen on seed 1; figures are five seeds and their median. Nothing here has been repeated by another session; no checker gate, seal or review is claimed. Ten plans, each committed before its runs, each with its outcome and scored predictions appended (`PLAN_1.md` to `PLAN_10.md`); this report adds nothing that is not in them.

## 1. The question

The system of `loop_small_case` reads 400,000 lines of two-digit addition and then 200,000 of subtraction. The plain system then answers 1.8% of never-shown addition; with one addition line in 32 during subtraction, 15.6%. A control in which the program gives subtraction 128 new cells and freezes what addition wrote (`reserve=128`) keeps 4.4% with no old line and 67.0% with one in 32. Can the cells themselves settle where the second task is written, the program knowing nothing of the change, and keep that much? Is anything kept with no old line?

## 2. The answer in one table

Never-shown pairs, whole answer right, %; median of the five reported seeds (73129, 93683, 116089, 406364, 545353), and in brackets the median of five seeds not used before (1001 to 1005) where run. `plan=add:400000,sub:200000`.

| System | Cells | No old line: addition / subtraction | One old line in 32: addition / subtraction | Plan |
|---|---|---|---|---|
| plain | 256 | 1.8 / 86.0 [2.2 / 83.2] | 15.6 / 85.8 [10.8 / 85.2] | 1 |
| the program's choice (`reserve=128`), the level to reach | 256 + 128 | 4.4 / 74.0 [5.8 / 73.4] | **67.0** / 75.4 [66.0 / 71.4] | 1 |
| (b) cells silent most of the time | 256; 512 | 0.8 / 49.0; 1.0 / 72.4 | 5.4 / 50.6; 8.0 / 71.4 | 1 |
| joining at a steady rate, nothing else (control) | 384 | 1.8 / 87.4 | 14.2 / 86.4 | 2 |
| (a) age: joining, and steps that shrink with a cell's age | 384 | 6.2 / 53.2 | 33.6 / 56.0 | 2 |
| (c), first half: steps that shrink with a cell's own use | 256 | 1.8 / 82.0 | 12.8 / 81.6 | 9 |
| mine, step 1: joining by the clock; strengths closed unless met in youth | 384 | 74.6 / 76.2 (16 to 81 over the seeds) | 54.4 / 85.6 | 3 |
| step 2: and constants that become still | 384 | 85.4 / 80.0 | 56.8 / 84.8 | 4 |
| (c), second half: joining on the mismatch above its usual size | about 208 | 88.8 / 11.0 | 66.4 / 12.0 | 5 |
| step 3: joining when the answering cell hears too few cells | 128 + 128 | 85.8 / 86.8 | 71.6 / 90.8 | 6 |
| step 4: and a strength opens only if met on 30 young bytes | 128 + 128 | **85.8 / 86.8** [84.8 / 86.2] | **81.6 / 85.8** [81.2 / 87.6] | 7, 10 |

Addition after its own phase is 82 to 85 at the median in every row but (a) (73.4), (b) (71.4, 74.2) and the fifth plan (89.2).

## 3. What was found

1. **The candidates of the brief do not reach the level.** Silent cells keep less than logistic cells: about nine in ten of the cells active on addition lines are also active on subtraction lines, because which cells are active is decided by the digits, which both tasks share (`PLAN_1.md`). Age keeps twice what the plain system keeps and half of what the program's choice keeps, and pays in the learning of both tasks; on the development seed every law that kept over 20% of addition held subtraction below 62% (`PLAN_2.md`). Steps that shrink with a cell's own use keep nothing: cells used alike stiffen alike (`PLAN_9.md`). A common signal made of the size of the mismatch brings cells in at birth and not at the change of task, because most of the mismatch is bytes nothing can foretell (`PLAN_5.md`).
2. **Three rules on a cell's own quantities, with one common signal, reach it and pass it** (`PLAN_3.md`, `PLAN_4.md`, `PLAN_6.md`, `PLAN_7.md`):
   - *Closed unless met in youth.* Each strength from a line is closed at the cell's birth and opens if its line was active on at least 30 of the cell's first 4,500 learned bytes. Later, at a byte where a closed strength has its line active, the cell is silent and unchanged.
   - *Constants that become still.* The answering cell's step on its 14 constants shrinks with its age (exp(-n / 10,000)).
   - *Joining when too few are heard.* The answering cell keeps a running mean of the number of cells it hears; while that is below 128, waiting cells join (one number, the same for all of them).
   Nothing in them names a byte, a task or a moment. On every seed 128 cells join at birth and 128 at the change of task and none at any other time; at the end 128 cells may speak on addition lines only and 128 on subtraction lines only. With no old line addition stays where it was after its own phase (85.8; 83.2 after its phase) and subtraction is learned to 86.8; with one old line in 32, 81.6 and 85.8, against the program's 67.0 and 75.4, with the plain system's number of cells and of numbers that learn (43,278).
3. **Each rule was needed, and each was found from what the one before left.** With the first rule alone the old cells were untouched (their strengths changed by under 0.001 of their size) and yet addition fell to 16 and 31 on two seeds: the whole loss was in the 14 constants, which subtraction moved (put back, addition was at its own level on the three seeds looked at). With the clock, the cells that could learn subtraction arrived one by one. With a strength that opens on one meeting, old lines made things worse: late cells opened to them and spoke on both kinds of line.
4. **It holds in the other order and on new seeds.** Subtraction first: the first task 93.6 and 94.2 at the end (93.8 after its own phase), the second 79.0 and 83.4 (`PLAN_8.md`). Five seeds that had no part in the making of the candidate: 84.8 / 86.2 and 81.2 / 87.6, above the program's choice on every seed (`PLAN_10.md`). When addition is read again for 100,000 lines after subtraction, subtraction is unchanged (86.4 against 86.8; sixth plan's candidate).

## 4. What is not shown, and the limits

1. **The change of task is seen only through a line never met** (a byte value at a delay): here the operator byte, which is on the delay lines at every learned byte of a subtraction line. With subtraction written in the bytes of addition (`aa=bb+ccc`), so that no line is new, no cell falls silent, no cell joins, and the candidate loses addition as the plain system does: 0.4 against 1.4 with no old line, 13.4 against 16.6 with one in 32 (`PLAN_8.md`, test 3). A change of task made of met lines in a new arrangement is outside what these rules can see. This is the main limit.
2. **Nothing the first task formed is used for the second.** The two sets of cells share only the constants, which are still. The system has become two systems side by side, chosen by a byte. That is keeping by separation; it is not yet a system that goes on learning with what it has learned.
3. **The pool is finite**: each new kind of line takes 128 waiting cells. A third task was not run.
4. **More old lines are worse than fewer** for the final candidate until there are many: 85.8 with none, 81.6 with one in 32, 76.8 with one in 8 (plain: 71.2). The threshold of 30 meetings is tied to how rare the old lines are.
5. **The score** is that of `loop_small_case`: each of the three answer digits named from the true bytes before it, the whole answer right; 500 never-shown pairs of each task. The end-of-line byte is not scored. A single reading of a task that is being learned scatters by several points.
6. One pair of tasks; settings chosen on one development seed; the reported seeds were read between plans and shaped the later plans (hence `PLAN_10.md`); one session; no other session has repeated anything.
7. Not run: the candidate with the relay over two steps, with cells that read few lines, with one sign for each cell, or on any language task; steps that shrink with use on the joining frame; a rule that closes by pairs of lines.

## 5. Errors of mine, as they were found

- Two faults of the job launcher before the first plan's results (runs discarded, `PLAN_1.md`).
- A check script that printed zeros after an edit (the comparison had fallen into a comment); repaired and rerun before use.
- `PLAN_6.md`: a statement that no cell was in its youth at any reading was wrong for two mid-phase readings; no end-of-phase figure changes.
- `PLAN_7.md`: my account of a price to be paid on rare subtraction lines was wrong (a line is a byte at a delay, whatever its role); the price did not arise.
- `PLAN_8.md`: a figure quoted from the wrong plan, corrected beneath.
- `PLAN_3.md`: a clumsy count in the scoring of C1, corrected in `PLAN_4.md`.

## 6. The decision record

- Anchor: `loop_small_case` at commit 260c5d7 (the plain system and the control `reserve`), copied unchanged into this folder; `kage.py` is new.
- Question: can the cells settle where the second task is written and keep what the program's choice keeps.
- Falsifiers: stated in each plan; the one aimed at my own claim is test 3 of `PLAN_8.md`.
- What ran: ten plans, 285 result files, about 4.8 CPU hours, about two hours on the clock; all development.
- Result: section 2. Lesson: sections 3 and 4.
- Disposition (mine, for the coordinating session to weigh): retain the three rules as a development result on this pair of tasks; do not enter K1 as reached in the ledger until another line has repeated it, and not without limit 1 beside it.
- Next action I would propose, one: a change of task that the lines do not announce (limit 1), since that is where the candidate stands exactly where the plain system stands.

## 7. Files, commands, cost

- Copied from `loop_small_case/` at 260c5d7, unchanged: `cells2.py`, `kern.c`, `kern2.c`, `kern3.c`, `fastk.py`, `fastk2.py`, `fastk3.py`, `.gitignore`.
- New: `kage.py` (numpy, the reference; a subclass of the plain system; its header states every option), `kage.c` with `fastage.py` (compiled), `check_kage.py` and `check_kage_output.txt` (seventeen cases of the compiled kernel against the numpy program, every strength, count and signal compared: agreement to 2e-15; with all cells present and no rule `kage.py` equals `cells2.py` to 3e-15 and gives 81.5 on seed 1), `overlap.py`, `diag3.py`, `tab.py`, `runjobs.sh`, the job lists `jobs*.txt`, the tables `table*.txt`, every result in `res/`, the looks inside in `log/diag*.txt`. One full life was also run with the numpy program: the same twelve readings as compiled (`PLAN_4.md`).
- The final candidate: `python3 kage.py <label> seed=<s> plan=add:400000,sub:200000 H=384 H0=1 join=hear K=128 theta=20 young=4500 meet=30 blaw=exp btau=1e4 fast=1 [reh=32]`; about 40 to 50 CPU seconds.
- A table: `python3 tab.py r7_meet_reh32 r7_meet_reh0`; for the new seeds `SEEDS=1001,1002,1003,1004,1005 python3 tab.py r10_cand_reh32`.
