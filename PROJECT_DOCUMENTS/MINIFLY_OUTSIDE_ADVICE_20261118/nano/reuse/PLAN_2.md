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
