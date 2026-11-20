# Line `nano/reuse`, second plan: a store read through other places, found by the system; the transfer experiments (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, since: "继续大胆假设，基于事实努力尝试。" "多想想，多试试。" "继续大胆尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-reuse-8mmhfm` at the commit that holds this file; base as in `PLAN_1.md`. Written after `reuse.py check` passed (`check_output.txt`) and before any run of the mechanism on any seed. Development level; no gate, seal or review is claimed.

## 1. What step 1 changed in the design

Step 1 found alike kinds, and found that the most-counted follower understates the likeness between plain kinds, because a cell keyed on digits alone counts at every place of a line where its values occur. So the judgment here is on what a store foretells of the stream with all its counts, not on its most-counted follower. And a kind is a set of distances back from the byte foretold, so the merging of kinds has to say *which place of one kind answers to which place of the other*: the object found is a store together with an ordered set of places through which it is read.

## 2. The system and the one change

*The base* is the plain system of `nano/merge/merge.py` (its mode `none`; `reuse.py check` shows the same score on a short run): nine delays; a kind for every set of at most four delays (255 kinds, 5.08 million possible cells, about 1.2 million that have counted after 40,000 two-digit lines); a cell says log((c(v) + 2) / (n + 14 x 2)); the answering cell holds a trust for each kind at each of 40 levels of experience, born 0.02, step 0.1 shrinking with use. No classes of cells are used in this plan (they are the merge line's; putting the two together is not run here, see section 6).

*The change: readings.*
1. **A reading** is a store (the cells of one kind A, of m places) and an ordered set P of m delays that is not A's own. At a byte, the reading addresses the cell of A's store whose values are the bytes at the places P, in P's order.
2. **Judged before counting.** The stream is taken in blocks of 10,000 bytes. Before a block is counted, each candidate reading is judged on it: the sum, over the block's bytes at which the addressed cell has counted, of what that cell says of the byte that follows less the mean of what the store's counted cells say of that byte. This is "the pooled store foretells the member": the bytes of the block at the places P are the new cells of the kind P, not yet counted.
3. **Against chance pooling.** The same store with its counted cells shuffled among their addresses is judged through every ordered set of places; the largest of these is the store's chance level. A reading is a candidate to keep only above it.
4. **Which stores, which places.** Stores: the M kinds of largest trust (mean over uses) with at least two delays, as the merge line takes its candidates. Places: every ordered set of m of the nine delays (3,024 for four places). Nothing says which delays, which bytes, or that anything is a digit.
5. **Kept readings.** The 32 of largest judgment above their chance level. Each speaks as a sender beside the plain kinds. Its trust starts as the trust of the kind whose store it reads (the merged kind has one store; I take it to bring its trust; the alternative, born at nothing, is an arm of the development choice) and is then its own; a reading kept again keeps its trust, also after an absence. **It counts into the store it reads** (one store, pooled, counting on); the alternative, reading without counting, is the other development arm.
6. **When.** At every block: the block is judged, then read with the readings that stood before it, and the readings judged on it stand from its end. So a reading is never judged on bytes its store has counted and never used on the bytes it was chosen on.

*Laid down, and what it costs in generality.* Blocks of 10,000 bytes; 32 readings; M stores by trust; stores of at most four places; a reading matches every place of a store to one delay (no reading of part of a store, and none through a class). The control "by chance": the same stores and the same number of readings at every block, the places drawn by chance.

*Numbers chosen once on development seed 1*: the trust a reading starts with (the store's kind's, or nothing) and whether it counts (2 x 2), with M = 32. Rule: the largest sum of (two-digit sums at 40,000 lines, mean of the last three readings) and (three-digit sums 5,000 lines after 40,000 two-digit lines). M = 32 and not the merge line's 16, because step 1 showed the store of the whole pair 17th by trust on this seed; chosen before any run. Then frozen for every experiment.

## 3. Experiments (five reported seeds 73129, 93683, 116089, 406364, 545353; the frozen setting)

Tasks: `add2` (`ab+cd=efg`, the split of `cells2.py`: 500 pairs scored, never shown; 8,000 shown), `add3` (the split of `nano/ahead2`: a fifth of the million pairs never shown, 500 scored), `add2w` (the same two-digit sums and the same split in another layout, `ab++cd==efg`: no new byte, every distance between the digits changed), `sub2` (`ab-cd=efg`, larger first, `cells2.py`'s split for subtraction). Score: never-shown lines, the system's own answer bytes, the whole answer. Readings of the second task at 1,000, 2,000, 3,000, 5,000, 7,500, 10,000 lines and every 5,000 to 40,000.

- (a) `add2` 40,000 lines then `add3` 40,000; against `add3` from nothing; each with readings and without (the plain system).
- (c) `add2` 40,000 then `add2w` 40,000; against `add2w` from nothing; each with readings and without.
- Control 1: (a) with the places of the readings drawn by chance. Control 2: `sub2` 40,000 then `add3` 40,000, with readings.
- (b) three-digit then four-digit sums needs eleven delays (562 kinds, 13 million possible cells, 7,920 ordered sets of four places): run on as many seeds as the cost allows after (a), (c) and the controls; a reading and not a claim if fewer than five.

*Lines needed*: the first reading at which the score of the second task is at least 90% (and, reported beside it, 80% and 70%); "more than 40,000" if never.

## 4. Hypothesis (the coordinating session's), what refutes it, predictions (mine)

*Hypothesis.* After the first task has been read, the system with readings needs at most half the lines to reach 90% of never-shown lines of the second task that the same system needs from nothing, in (a) and in (c). *Refuted if* the saving is under a fifth in either. If the system from nothing does not reach 90% within 40,000 lines and the system after the first task does, the saving is counted as held at the bound; if neither does, the same is read at 80%, and said.

| | Statement | Chance |
|---|---|---|
| R1 | (a): lines to 90% after `add2` at most half of those from nothing (median of five seeds; the system with readings) | 0.35 |
| R2 | (c): the same | 0.40 |
| R3 | within the first 3,000 lines of `add3` after `add2`, on at least four seeds, a reading of the store (3, 4, 6, 7) through the places (4, 5, 8, 9) or (8, 9, 4, 5) is kept (the two-digit rule read at the three-digit distances, straight or with the numbers exchanged) | 0.45 |
| R4 | `add3` at 2,000 lines: after `add2` at least 10 points above from nothing (median; with readings) | 0.60 |
| R5 | the plain system: `add3` at 5,000 lines after `add2` within 5 points of from nothing (nothing is reused without readings) | 0.50 |
| R6 | control 1 (places by chance): `add3` at 5,000 lines within 5 points of the plain system after `add2` | 0.70 |
| R7 | control 2 (`sub2` first): `add3` at 5,000 lines within 5 points of the system with readings from nothing | 0.50 |
| R8 | `add2` alone at 40,000 lines, last three readings: with readings at least 5 points above the plain system (the exchange of the two numbers found and used) | 0.40 |
| R9 | the same below the plain system | 0.20 |
| R10 | the hypothesis whole | 0.25 |

Where I expect it to fail, if it fails. (i) The two-digit store of the whole pair holds only the 8,000 shown pairs of 10,000 and one count in three in it is the rule's, so read at the three-digit distances it can answer about four tens digits in five and nothing sharper; 90% of whole three-digit answers needs the new lines as well. (ii) The stores of largest trust hold a layout byte and fit one place of the line only. (iii) In the second task a store goes on counting through its own places, which now foretell other things: the one store is diluted by its own kind.

## 5. Size and cost

Per run: 255 kinds and up to 32 readings; 5.08 million possible cells (about 310 MB), of which 1.2 to 2 million count. Cost unknown before the first run; my estimate is three seconds of judging a block, so three to five CPU-minutes for a life of 80,000 lines. About 50 runs on four cores.

## 6. Not run

Readings of stores keyed on classes (the merge line's class kinds), which step 1 showed to be the cleanest alike kinds; readings of part of a store; a store withdrawn from its own places when they no longer foretell; prose; more than two tasks in a life; look-ahead.

## 7. Addition 1 (20 November; written before any scored run of the mechanism; two changes, one of cost and one of the judgment)

*What happened.* The four development runs of section 2 were started and stopped unread after 17 CPU-minutes each without finishing: judging a block cost about 19 seconds, not 3. Nothing of them is kept. Two probes were then made on seed 1 that scored nothing: the judgment of one block after 3,000 and after 20,000 two-digit lines.

*Change of cost.* The judging loop runs over bytes outside and places inside (one table in use at a time), and a block is judged on its first 5,000 bytes. 1.3 seconds a block. `reuse.py check` again against numpy (`check_output.txt`).

*Change of the judgment.* The first probe showed 6,454 readings above the chance level of section 2, the largest being readings that say again what the plain kind of the same places already says (bytes of the layout). Judged against chance alone, the 32 kept would be readings that add nothing. So a reading is now judged on **what its store foretells of the block's bytes, read through its places, less what the plain kind of those places foretells of the same bytes with its own cells** (a cell that has not counted is taken to say 1/14 on both sides), and it is kept only above the largest such figure of the same store shuffled. In the words of the hypothesis: the pooled store must foretell the member kind's new cells better than the member's own store does. This is stricter than "alike": two kinds that say the same and have each read as much gain nothing from each other and are not merged.

*What this costs, seen in the second probe before any scored run.* After 20,000 two-digit lines one reading stands above chance and the exchange of the two numbers is not among the candidates: every line of the stream has been shown before, the kind's own cell has counted it, and the exchanged reading adds nothing *on the stream*. Its use would be on lines never shown, of which the stream holds none. So I now expect R8 to fail (my chance for it falls from 0.40 to 0.10) and say so before the runs. The transfer between tasks is what this judgment is made for: there the kind of the new places has no cells of its own.

*One more prediction, new.* R11: with readings that count into the store they read, the never-shown two-digit sums stand higher at the end of 40,000 lines of `add3` than at its start (median of five seeds; the three-digit lines fill, through the reading, the cells of the two-digit store that the two-digit task never showed): chance 0.45.

Everything else in sections 2 to 6 stands. The development choice (the trust a reading starts with; whether it counts) is made as stated, with the changed judgment.

## 8. Addition 2 (20 November; after the first development runs on seed 1, before the runs it describes and before any reported seed)

*What the first development runs show* (seed 1; `res/dev_*_first.json`, `res/dev_none.json`, `res/dev3_*.json`; 32 readings, the judgment of addition 1; never-shown three-digit sums, %, single readings):

| `add3` lines | 1,000 | 2,000 | 3,000 | 5,000 | 10,000 | 20,000 | CPU s |
|---|---|---|---|---|---|---|---|
| after `add2`, readings (trust the store's, counting) | 75.0 | 40.2 | 38.4 | 50.4 | 67.6 | 82.0 | 119 |
| after `add2`, readings (trust the store's, not counting) | 74.8 | 39.2 | 60.2 | 65.4 | 75.2 | 84.0 | 121 |
| after `add2`, readings (born at nothing, counting) | 73.0 | 43.4 | 40.0 | 50.0 | 67.4 | 82.6 | 121 |
| after `add2`, readings (born at nothing, not counting) | 71.8 | 42.2 | 61.0 | 66.0 | 76.2 | 84.2 | 118 |
| after `add2`, plain system | 25.4 | 33.8 | 40.4 | 51.4 | 68.6 | 79.8 | 28 |
| from nothing, readings | 34.0 | 40.6 | 52.4 | 62.0 | 74.2 | 84.6 | 46 |
| from nothing, plain system | 34.0 | 40.8 | 48.8 | 60.4 | 77.0 | 85.4 | 12 |

Two-digit sums at 40,000 lines are 76.2 to 76.4 in every arm: during the first task almost no reading stands above chance (as addition 1 expected).

*Reading (mine, one seed).* **At the first block of the second task the system finds the two-digit stores and reads them at the three-digit distances** (among the 32 kept after 770 lines: the store (3, 4, 6, 7) through (4, 5, 8, 9), and the stores that hold a layout byte through the places that fit, such as (2, 3, 4, 7) through (3, 4, 5, 9)); 75% of never-shown three-digit sums after 1,000 lines against 25% without readings and 34% from nothing. **At the next block it is lost**: 650 to 1,100 readings then stand above chance, because the plain kinds of the new places hold counts of the first task that foretell the new text worse than 1/14, so that almost any store read there beats them; the 32 slots fill with several orderings of the same small stores at the same places, and the readings that carry the rule are displaced. Counting through such readings then harms (38 against 60 at 3,000 lines).

*The one change.* **A set of places joins one store only**: among the readings through the same set of places (whatever the store and the order), only the one of largest judgment is kept. This is the rule by which a cell joins one class in `nano/merge` (the class that raises J most), which I had not carried over. And the number of readings kept is 64, not 32 (there are 246 sets of two to four places; more slots is more computation, not more knowledge). Nothing else changes. The development choice (2 x 2) is run again with this and decided by the rule of section 2.

*Predictions (mine).* A2-1: with the change the score of `add3` after `add2` at 2,000 and 3,000 lines stays at least 65 on seed 1 in at least one of the four arms (0.55). A2-2: counting arms are again below the arms that do not count at 5,000 lines (0.60).

## 9. Outcome of the development choice, and the frozen setting (20 November; before any reported seed)

Seed 1 with addition 2 (`res/dev_i?c?.json`; never-shown three-digit sums after 40,000 two-digit lines, %, single readings; 112 to 115 CPU-seconds a run, 2.3 million cells):

| `add3` lines | 1,000 | 2,000 | 3,000 | 5,000 | 10,000 | 20,000 |
|---|---|---|---|---|---|---|
| trust the store's, counting | 67.8 | 76.4 | 68.2 | 76.8 | 78.2 | 83.4 |
| **trust the store's, not counting** | 71.4 | 77.4 | 80.6 | **80.6** | 84.2 | 85.2 |
| born at nothing, counting | 70.4 | 78.0 | 69.0 | 78.0 | 79.8 | 85.2 |
| born at nothing, not counting | 69.0 | 76.4 | 71.4 | 80.4 | 83.8 | 85.4 |
| (plain system after `add2`, from the table above) | 25.4 | 33.8 | 40.4 | 51.4 | 68.6 | 79.8 |
| (from nothing, plain system) | 34.0 | 40.8 | 48.8 | 60.4 | 77.0 | 85.4 |

Two-digit sums at 40,000 lines: 76.9 (mean of the last three readings) in every arm. **By the rule of section 2 the frozen setting is: a reading starts with the trust of its store's kind and does not count into the store** (80.6 + 76.9; the margin over "born at nothing, not counting" is 0.2 points and decides nothing; counting is 2.4 to 3.8 points lower). A2-1 held (three arms of four stay at or above 65 at 2,000 and 3,000 lines). A2-2 held.

*What the choice costs the hypothesis, said before the reported seeds.* The hypothesis speaks of one pooled store. The setting chosen by the rule reads the store at the new places and does not pool: the store is used, not added to. R11 can then not happen (the two-digit store gains nothing from three-digit lines) and is scored as not tested. And on this seed no arm is at 90% within 20,000 lines, from nothing or after `add2`; the plain system from nothing is reported by its line at 92% after 80,000. I expect the reading at 90% to be undecided and the stated fallback (80%) to carry the result; I say so now.

*Runs* (`jobs_main.txt`): the experiments of section 3 on the five reported seeds; (b) on one seed (73129) with eleven delays and 20,000 lines of `add4`, a reading and not a claim.

## 10. Addition 3 (20 November; written AFTER the reported seeds of (a), (c) and control 1 were read; what follows it is a look after the fact and is marked so wherever it is reported)

*What was seen.* In (c) the system with readings is no better than the plain system after `add2`, and both are below the system from nothing. On seed 73129 the reading that carries the rule in the new layout, the store (3, 4, 6, 7) through the places (4, 5, 8, 9), is not among those kept at the first block, and no reading through that set of places is. The cause I find in the judgment as section 2 and addition 1 state it: a store's chance level is the largest figure of the shuffled store through *any* places, and after a change of task some plain kinds hold counts that foretell the new text far worse than 1/14, so that the shuffled store read at their places scores in the thousands; that one figure then bars the store's readings at every other set of places. In (a) the rule's readings happened to stand above it.

*The one change, for a look.* The chance level of a reading is the largest figure of the shuffled store **through the same set of places** (over its orders). `cset=1`. Nothing else. Runs: (c) and (a) again with it, five seeds, 10,000 lines of the second task (`jobs_3.txt`). The four-digit runs (b) were stopped unfinished to make room and are **not run**.

*Predictions (mine).* A3-1: (c) with the change: `add2w` at 1,000 lines at least 10 points above the system from nothing (55.2), median of five seeds (0.45). A3-2: (a) with the change at 1,000 lines within 5 points of the frozen setting's 71.2 (0.6).
