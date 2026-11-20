# Line `nano/reuse`: a store read through other places, found by the system (20 November 2026)

Development level; five reported seeds (73129, 93683, 116089, 406364, 545353), seed 1 for every choice. No gate, seal or review is claimed. Authorisation, the hypothesis (the coordinating session's), the system, the predictions and every change on the way are in `PLAN_1.md` and `PLAN_2.md` (three additions, the third written after reported seeds were read). The readings below are mine. Programs: `step1.py`, `reuse.py` (each with a check of the compiled or fast part against numpy or plain Python: `check1_output.txt`, `check_output.txt`); tables: `report_main.txt`, `report_3.txt`; results: `res/*.json`.

## 1. What was asked and what was built

*Asked.* Whether kinds of cell that say the same can be found by the system and merged, so that what one task taught is used for another; measured by the lines a second task needs after a first, against the same system from nothing and against the system without the mechanism.

*Step 1* (one seed; `PLAN_1.md`, section 5): in the system of `nano/merge` after 40,000 two-digit sums, alike kinds exist. The clean ones are class kinds held at one class against plain kinds (98% agreement for the case the coordinating session named, 2% for its complement; ten such pairs at 90% or more). Plain kinds agree in part only (50 to 87%), because a plain cell keyed on digits counts at every place of a line where its values occur.

*Step 2, the mechanism* (`PLAN_2.md`, sections 2, 7, 8, 9). On the plain system of `merge.py` (255 kinds of at most four of nine delays; no classes): a **reading** is the store of one kind read through another ordered set of places. Before each block of 10,000 bytes is counted, every reading of the 32 stores of largest trust through every ordered set of places is judged on the block's first 5,000 bytes: what the store foretells there, less what the plain kind of those places foretells with its own cells; kept only above the same store with its cells shuffled; a set of places joins one store only; at most 64 kept. A kept reading speaks as a sender, starting with its store's kind's trust; by the development rule it does not count into the store. Nothing says which places, which bytes, or anything about sums.

## 2. Results (never-shown lines, the system's own answer bytes, whole answer, %; median of five seeds)

**(a) Two-digit sums for 40,000 lines, then three-digit sums.** 2.3 million cells; 157 CPU-seconds a life with readings (119 of them judging), 33 without.

| `add3` lines | 1,000 | 2,000 | 3,000 | 5,000 | 10,000 | 20,000 | 40,000 | last three readings | lines to 70% | to 80% | to 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after `add2`, with readings | **71.2** | **76.6** | **78.4** | **80.4** | 81.0 | 83.8 | 90.0 | 87.1 | 1,000 | 5,000 (5, 5, 10, 15, 5 thousand) | 40,000 (two seeds: never) |
| from nothing, with readings | 33.2 | 42.6 | 50.8 | 62.6 | 74.8 | 86.2 | 91.2 | 89.2 | 10,000 | 15,000 | 40,000 (one seed: never) |
| after `add2`, plain system | 27.0 | 31.2 | 38.4 | 48.6 | 68.8 | 83.0 | 89.8 | 88.6 | 15,000 | 20,000 | never at the median |
| from nothing, plain system | 33.2 | 42.2 | 48.6 | 62.6 | 74.8 | 86.2 | 91.2 | 89.1 | 10,000 | 15,000 | 40,000 |
| control 1: after `add2`, places of the readings by chance | 26.6 | 30.2 | 36.0 | 43.6 | 62.2 | 76.4 | 89.8 | 86.7 | 20,000 | 25,000 | never |
| control 2: after `sub2`, with readings | 27.0 | 33.4 | 40.2 | 49.8 | 68.6 | 81.6 | 89.6 | 87.7 | 15,000 | 20,000 | never |

On every seed the reading of the two-digit store (3, 4, 6, 7) through the places (4, 5, 8, 9) is kept at the first block of the second task (after 770 lines) and at the next; at the end the readings of largest trust are two-digit stores that hold a layout byte read at the fitting three-digit places, such as (2, 3, 4, 7) through (3, 4, 5, 9). Tens and hundreds digits after 1,000 lines: 87 and 83 with readings, 54 and 52 without.

**(c) Two-digit sums for 40,000 lines, then the same sums in another layout (`ab++cd==efg`).** 1.5 million cells; 145 CPU-seconds with readings, 30 without.

| `add2w` lines | 1,000 | 2,000 | 3,000 | 5,000 | 10,000 | 20,000 | 40,000 | last three | lines to 80% |
|---|---|---|---|---|---|---|---|---|---|
| after `add2`, with readings | 60.4 | 61.2 | 56.6 | 66.2 | 76.2 | 80.8 | 79.6 | 79.1 | 20,000 (two seeds: never) |
| from nothing, with readings | 55.2 | 61.6 | 66.8 | 73.6 | 80.4 | 82.0 | 81.0 | 80.3 | 10,000 |
| after `add2`, plain system | 53.4 | 55.8 | 59.6 | 67.8 | 76.4 | 80.8 | 79.6 | 79.1 | 20,000 |
| from nothing, plain system | 55.2 | 61.6 | 66.8 | 73.6 | 80.4 | 82.0 | 81.0 | 80.3 | 10,000 |

No arm reaches 90%. The reading that would carry the rule, (3, 4, 6, 7) through (4, 5, 8, 9), is kept on no seed.

**Within one task** almost no reading is kept: two-digit sums alone stand at 77.5 with readings and 77.5 without (last three readings), three-digit sums from nothing at 89.2 and 89.1.

**The first task afterwards** (never-shown `add2` after 40,000 lines of the second task): 63.2 with readings and 58.2 without after `add3`; 11.4 in both after `add2w`. Without replay or separation the plain kinds lose the first task where the second writes the same bytes at other distances.

**(b) Three-digit then four-digit sums: not run** (eleven delays; stopped unfinished for cost).

**After the fact** (`PLAN_2.md`, section 10; written after the tables above were read): with the chance level taken at the same set of places, (a) is unchanged (73.2, 76.2, 80.0, 81.6 at 1,000, 2,000, 5,000, 10,000 lines) and (c) is not helped (56.4, 57.6, 66.2, 76.8); the rule's reading is still kept on no seed in (c). The cause I gave for (c) in that section is therefore not the cause, or not the whole of it.

## 3. The hypothesis and the predictions

*The hypothesis as stated* (at most half the lines to 90% in (a) and in (c); refuted if the saving is under a fifth in either): **refuted.** In (a) both arms reach 90% at 40,000 lines at the median (no saving at that level). In (c) no arm reaches 90%, and at the stated fallback of 80% the system after the first task needs twice the lines of the system from nothing.

| | Statement | Outcome |
|---|---|---|
| R1 | (a): lines to 90% at most half | failed (40,000 against 40,000) |
| R2 | (c): the same | failed (at 80%: 20,000 against 10,000) |
| R3 | the store (3, 4, 6, 7) read through (4, 5, 8, 9) or (8, 9, 4, 5) within 3,000 lines on at least four seeds | held (five of five, at the first block) |
| R4 | `add3` at 2,000 lines at least 10 points above from nothing | held (76.6 against 42.6) |
| R5 | plain system: after `add2` within 5 points of from nothing at 5,000 lines | failed (48.6 against 62.6: the first task harms the plain system) |
| R6 | places by chance within 5 points of the plain system after `add2` at 5,000 lines | held at the edge (43.6 against 48.6) |
| R7 | `sub2` first: within 5 points of from nothing at 5,000 lines | failed as stated (49.8 against 62.6; it equals the plain system after a first task, so the control shows what it was for: no saving without a related first task) |
| R8 | `add2` alone at least 5 points above the plain system | failed (77.5 against 77.5; expected to fail since addition 1) |
| R9 | `add2` alone below the plain system | failed (equal) |
| R10 | the hypothesis whole | failed |
| R11 | the pooled store lifts `add2` | not tested (the frozen setting does not count into the store) |
| A2-1, A2-2 | addition 2 | both held (seed 1) |
| A3-1 | (c) helped by the chance level at the same places | failed |
| A3-2 | (a) unchanged by it | held |

## 4. Readings (mine)

1. **Reuse found by the system exists in this programme for the first time, on one pair of tasks.** After two-digit sums the system, told nothing, reads the two-digit stores at the three-digit distances within 770 lines, and answers 71% of never-shown three-digit sums after 1,000 lines where it answers 33% from nothing and 27% without readings; 80% takes 5,000 lines against 15,000. Places drawn by chance give nothing (and cost 5 points); subtraction first gives nothing. So the gain is the use of what the first task taught, at places the system found.
2. **It is early help, not a higher level.** From 20,000 lines on the system after the first task is at or slightly under the system from nothing (83.8 against 86.2 at 20,000), and both end near 90. The two-digit store holds 8,000 of 10,000 pairs and one count in three in it is the rule's (step 1), and the plain kinds carry stale counts of the first task; the stated level of 90% is the plain system's ceiling here, which reuse does not move.
3. **Where the hypothesis fails, by the brief's four places.** (a): alike kinds are there, found, recognised early, and trusted. (c): **there but not found**; the rule's reading is never kept, the cause is not established (my one conjecture was tested after the fact and failed), and the first task then harms the second through shared plain kinds. Within one task: there (step 1: the exchange of the two numbers) but **not found by this judgment by construction**, since a stream of shown lines offers nothing that an exchanged reading foretells better than the kind's own cells.
4. **"Say the same" and "foretells better than my own cells" are different judgments, and I ended on the second.** Judged against chance alone, thousands of readings pass and the kept ones repeat what plain kinds already say. Judged against the own kind, a reading is kept only where its store knows more than the kind of those places, which is the case of a new task and never the case of a symmetry inside a known one. A judgment that keeps alike kinds *because* they agree, and still ranks the useful ones first, is not built here.
5. **One pooled store was not shown.** Counting through readings into the store was 2 to 4 points worse on the development seed and was not chosen; what is shown is a store used at new places, not added to.
6. *Limits.* Sums only; two tasks in a life; the plain system without the merge line's classes (the cleanest alike kinds of step 1 are class kinds, and readings of class stores are not built); 64 readings, blocks of 10,000 bytes, 32 stores by trust, the trust a reading starts with: chosen on one seed; two flaws of my first design were found and changed on that seed before the reported seeds (additions 1 and 2), one conjecture after them (addition 3). Judging costs four times the reading itself.

## 5. What I would do next

Find why (c) does not find its reading (list the judgment of every store at the places (4, 5, 8, 9) at the first block of `add2w`); readings of class stores, where step 1 found the agreement whole; a judgment by agreement on common cells for the symmetry within a task, with the own-kind judgment kept for ranking; (b) with eleven delays.
