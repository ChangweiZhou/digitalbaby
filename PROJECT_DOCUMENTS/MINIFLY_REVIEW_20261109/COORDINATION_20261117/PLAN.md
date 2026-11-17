# What can be tested and has not been: the list, its order, and its state (from 17 November 2026)

Kept by the outside reviewer, who coordinates while Mr. Xu is away. This file is the one place where the order of work stands; it is updated whenever an item starts, reports or closes. Times are UTC. The log of what was sent to whom is `LOG.md` beside it.

## 1. The instruction

Mr. Xu, 17 November: "复盘一下，把这种所有能做的但没有做的都理清楚顺序一件一件做起来。要求已经很清楚了。你的工作很毛糙。我人在外面你自己安排。" (Review; put everything that could be done and has not been into a clear order and do it item by item. The requirements are already clear. Your work is careless. I am away; arrange it yourself.)

## 2. The requirements, in Mr. Xu's statements

| | Requirement | Stated |
|---|---|---|
| A | **Keeping.** Long-term memory should form by itself in the course of learning; some rehearsal is acceptable; it is not to be produced by our intervention. | 16 and 17 November |
| B1 | **Overtaking, in time.** Our learner may be slower at first and overtake the conventional learner in the course of continued learning. | 15 November |
| B2 | **Overtaking, in samples.** The end result may be about the same while our learner needs far fewer samples; it would then catch up in practice although it learns more slowly and computes more. | 17 November |
| C | **Locality.** Local learning in the place of full connection, at the input and the output as well; a cell cannot see the whole input. | 16 November |
| – | Standing: no full back-propagation in our learner; computation to a threshold is the measure; memory is fixed; learning continues. | October, 15 November |

## 3. What went wrong before, stated once

Several comparisons were made under settings chosen at the start for the dense network under back-propagation and carried unchanged into every later experiment: the starting size of the weights, the learning rate and decay, the taught fraction, the share of rehearsal. A method tested under settings chosen for another method can look worse than it is. Questions that needed such a setting to be varied were therefore never asked. The reviewer designed or audited those experiments and did not see it.

## 4. Rules of order

1. A setting that was carried over is varied before anything further is built on a conclusion that rests on it.
2. What needs only saved weights is done first, by the reviewer, without a worker.
3. Then, for each requirement, the experiment that answers it most directly.
4. Every job has a written brief with the record it rests on, checks before its runs, a trial on one seed before more, and its result verified from receipts and weights before it is read.
5. At most four lines of work at one time, the long lives with their helper counted as one. (The working rules say three. The reviewer raises it by one for the time of Mr. Xu's absence, because he asked for the items to be worked off, and because each result is verified by an agent of the reviewer before the reviewer reads it.)
6. A design is not fixed on an explanation that has not been tested; where an item waits for a result, the result is named.

## 5. The list

Status words: *running*, *waiting* (for the named item), *next* (brief to be written or in check), *queued*, *blocked*.

| No. | Item | Answers | Waits for | By | Status |
|---|---|---|---|---|---|
| 1 | Do differences between items pass from cell to cell at the starting weights of the local networks (E47, E49, E50 forms), and at what starting size would they? | C | – | reviewer, saved weights | running |
| 2 | Which cells and how many directions of the cells' state are in use in the limited and local networks (E42, E47, E49)? | A, C | – | reviewer, saved weights | running |
| 3 | An inventory of every setting of E26 to E50 that has only one value in the whole record | all | – | reviewer, record | running |
| 4 | E51: a new task learned in cells held in reserve; with arms of less rehearsal and of rehearsal at intervals | A | check of its brief | worker | next |
| 5 | The first long lives, release 1 (conventional learner; store; no decay), with a table of what is kept at a return by the number of earlier visits | A, B1 | – | worker | running |
| 6 | The sweep's long lives beside release 1 (with and without the store; 1,024 visits; four worlds) | B1 | – | helper | running |
| 7 | E50: a fixed input code in which a value stands for one byte; stages 1 to 3 | C | – | worker | running |
| 8 | E52: the sweep under its own learning rate and decay (and back-propagation under the same values), at half and at a quarter of the pairs | B1, B2, every comparison of the two rules | – | worker | next |
| 9 | E53: the local networks from a start at which differences pass | C | item 1 | worker | waiting |
| 10 | E54: the limited and the periodic network with a quarter and a tenth of the pairs taught | B2 | check of its brief; a free line | worker | next |
| 11 | E55: cells in reserve that only learning brings into use, on a birth with E50's code (no device) | A | items 4 and 7 | worker | waiting |
| 12 | E56: an unrelated task learned in E47's network, whose cells away from the boundary are not in use | A, C | item 2 | worker | waiting |
| 13 | Cells that learn at different speeds: does earlier knowledge stay in the slow ones? | A | items 4 and 5 | design | queued |
| 14 | A narrower width than the half width, to find the cheapest | computation | – | worker | queued |
| 15 | Does a life help with a task of another kind (the unrelated task from a lived and from a fresh network)? | B2 | – | worker | queued |
| 16 | Two more seeds for the conclusions of E47, E49 and E50 that stand after items 8 and 9 | C | items 7, 8, 9 | worker | queued |
| 17 | Confirmations on sealed tasks (E40, E43, E48) | all | the coordinator, who holds the sealed data | – | blocked |

The numbers E52 to E56 are the reviewer's; the coordinator may renumber them.

## 6. State

| Time | Event |
|---|---|
| 20:22 | Items 5 and 7 started in worker sessions (`LOG.md`). |
| 20:47 | Item 7: checks passed; stage 1 running. |
| 21:05 | Items 1, 2 and 3 started (agents of the reviewer, saved weights and the record only). Item 4's brief is in check. |
| 21:07 | Item 8: the brief of E52 written and given to a check. Items 1, 2, 3 running. |
| 21:10 | Item 5: the runner's checks passed and the runner change was read and accepted; the trial runs since 21:04. Item 6: the helper session for the sweep's long lives started. |
| 21:33 | Item 10: the brief of E54 written and given to a check. |
