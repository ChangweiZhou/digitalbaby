# The second answering cell added to this line's own prose systems (roadmap second revision, track L; the form of `PLAN_15.md`, nothing chosen) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "继续大胆尝试。" (my translation: "Go on trying boldly."). **Source.** `claude/friendly-darwin-8mmhfm` at 1f7aaf3df; `all.py`, `two.py`, `place.py`, `relay.py`, `words.py` as they stand; the form of `PLAN_15.md`. Written before any run of this plan's program (`gate_all.py`, which exists and has not been run). Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_15.md`, the part on prose, which is complete as I write (its sums on the reported seeds are still running): on the base system of runs of 0 to 6 bytes, a second answering cell with one part of the trust for each of the last six bytes, indexed by the byte there and born at nothing, lowers the loss by 0.0200, 0.0213, 0.0233 nats on three books that no line had opened, by 0.0210, 0.0236, 0.0299 on the three worn books, and by 0.0168 on the last tenth of the test text; the control whose parts read a constant gives between 0.005 worse and 0.001 better. That base has no cells that step once a word, no timing cells, no pool, no companions. This line's best system has all of them (`PLAN_10.md`: 1.4707 on the last tenth of the test text, where the full nanoGPT network of `nano/ref` has 1.454 on one seed).

## 2. The one change, and the question

The second cell of `PLAN_15.md`, unchanged in form and numbers (six parts, the bytes at delays 1 to 6, each part born at nothing and changed by the trust's step 0.00005 times the second cell's own mismatch), added to `all.py`'s system; the given sender (the relay system's opinions) has parts like any other sender. The first cell is `all.py`'s to the last digit in the same run.

**Does the gain stand beside the other kinds of cell, or do the cells that step once a word, the timing cells and the pool already hold what the bytes at the last six delays tell?**

*Systems.* T2 (the table's cells and the word cells: `without=companions,relay,since,pool`); all kinds but the relay system (`without=relay`); all kinds together.

*Measured.* (a) The test text: its stream read once, its last tenth with nothing changing (the measure on which 1.454 was measured; scored for some fifty arms and worn), and the last tenth with everything going on (another measure; no reference). T2, all but the relay system, all kinds. (b) The life of `PLAN_15.md` with the three books opened once there (`chesterton-ball`, `austen-emma`, `melville-moby_dick`): T2 and all kinds but the relay system (the relay system's opinions exist for the test text only); each book's last tenth right after its stream, the test text's last tenth at the end of the life and on the return with everything going on. (c) The same life with the three worn books, for comparison with `PLAN_10.md` and `PLAN_14.md`. The control (parts that read a constant) for T2 in (b).

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The gain stands beside the other kinds: on each of the three books of (b) the second cell is at least 0.010 nats below the first, for T2 and for all kinds but the relay system. *Refuted if* for either system fewer than two books are lower by 0.005.

| | Statement | Chance |
|---|---|---|
| X1 | T2, life (b): at least 0.010 lower on each of the three books | 0.70 |
| X2 | all kinds but the relay system, life (b): at least 0.010 lower on each | 0.55 |
| X3 | all kinds, the last tenth of the test text with nothing changing: the second cell at or under 1.4607 (0.010 below the first) | 0.60 |
| X4 | the same at or under 1.4540 (the figure of the full nanoGPT network on that tenth) | 0.35 |
| X5 | all kinds, going on through the last tenth: the second cell at least 0.010 below the first (1.3470) | 0.50 |
| X6 | the control on T2 gives less than a fifth of the gain on each book | 0.80 |
| X7 | keeping: after the three books of (b) the second cell's loss on the test text's last tenth has risen by under 0.03 (T2) | 0.45 |

If X4 holds it will be stated with its limits: one text, a last tenth scored for some fifty arms of this line, one seed of the reference, a system with 1.8 million or more cells against 10.7 million numbers, one reading against 4,000 steps of batches.

## 4. Size, cost, what is not run

The second cell holds 6 times (kinds + 1) times 40 times 65 parts: 171,600 for T2, about 1.7 million for all kinds; the number used is reported. T2 runs take under a minute; all kinds about five to ten minutes, and the relay system's opinions about ten minutes more, once. Not run: gates other than the bytes at the last six delays; a third cell; the second cell on the life of B only; the same on sums over long lives (an addition to `PLAN_15.md` if its claims hold).

## 5. Addition 1 (20 November; written after the runs of section 2 except one, and before the runs it describes): the life of (b) for all kinds, one book at a time

The life of (b) for all kinds but the relay system (the test text's stream and the three books in turn, 3.6 million bytes, 164 kinds) was killed twice for lack of memory on this machine, once beside other runs and once alone. It is NOT RUN as planned. In its place, to be read as a different measure: three lives, each the test text's stream and then **one** of the three books (`chesterton-ball`, `austen-emma`, `melville-moby_dick`), its last tenth scored right after its stream; all kinds but the relay system; the second cell. The largest of the three may also fail for memory and will then be entered as not run.

*Prediction (mine), in place of X2:* the second cell is at least 0.010 nats below the first on each book that runs (chance 0.55; on the worn books in turn it gave 0.020, 0.006, 0.025, so one book of three under 0.010 would not surprise me).

## 6. Outcome (entered 20 November after all runs; `res/gateall_*.json`, the printed lines in `out16/`)

The compiled loop equals numpy to 1.9e-15 on a prefix with companions and a given sender, and its first cell equals `all.py`'s kernel exactly (`check16_output.txt`). In every run the first cell gives the earlier plans' figures (1.4973; 1.4707; 1.4740; the worn books 1.4802, 1.3162, 1.4109 and 1.4546, 1.2413, 1.3418).

Nats a byte; first cell, second cell, and how much lower the second is.

*(a) The test text, its stream read once.*

| System | last tenth, nothing changing | last tenth, everything going on | kinds; parts used; cells; CPU s |
|---|---|---|---|
| T2 (the table's cells and the word cells) | 1.4973, 1.4820, 0.0153 | 1.3863, 1.3717, 0.0146 | 10; 80,896 of 171,600; 1.99 million; 9 |
| all kinds but the relay system | 1.4740, 1.4444, 0.0296 | 1.3462, 1.3192, 0.0270 | 164; 1.45 of 2.57 million; 4.49 million; 117 |
| **all kinds** | 1.4707, **1.4421**, 0.0287 | 1.3470, **1.3204**, 0.0266 | 164 and the given sender; 1.46 of 2.57 million; 4.49 million; 477 with the relay system's reading |

*(b) The test text's stream, then the three books of `PLAN_15.md` in turn; each last tenth right after its stream.*

| System | chesterton-ball | austen-emma | melville-moby_dick | the test text's last tenth at the end; on the return with everything going on |
|---|---|---|---|---|
| T2 | 1.6093, 1.5900, **0.0193** | 1.2056, 1.1861, **0.0195** | 1.4387, 1.4177, **0.0210** | 1.5061, 1.5072 (0.0011 higher); 1.3979, 1.3916 |
| T2, the control (parts read a constant) | 0.0017 higher | 0.0008 higher | 0.0008 higher | 1.5061, 1.5071; 1.3979, 1.3966 |
| all kinds but the relay system | **NOT RUN as planned** (killed for memory twice) | | | |
| all kinds but the relay system, addition 1: the test text and then that one book | 1.5950, 1.5741, **0.0209** | 1.1881, 1.1628, **0.0253** | 1.4215, 1.3974, **0.0241** | after the one book: 1.5704, 1.5737; 1.5713, 1.5662; 1.5356, 1.5319. On the return, going on: 0.021 to 0.023 lower |

*(c) The same life with the three worn books (bryant, burgess, carroll).* T2: 1.4802, 1.3162, 1.4109 against 1.4621, 1.2932, 1.3802 (0.0180, 0.0230, 0.0308 lower); the test text at the end 1.5067 and 1.5062. All kinds but the relay system: 1.4546, 1.2413, 1.3418 against 1.4342, 1.2357, 1.3165 (0.0204, 0.0056, 0.0253 lower); the test text at the end 1.5953 and **1.6176** (0.0224 higher); on the return with everything going on 1.3551 and 1.3314.

**The hypothesis holds**: the gain stands beside the other kinds of cell. For T2 the second cell is 0.019 to 0.021 below the first on each of the three books (0.010 asked). For all kinds the life as planned could not be run; on the three lives of one book each it is 0.021 to 0.025 below.

| | Statement | Outcome |
|---|---|---|
| X1 | T2: at least 0.010 lower on each of the three books | held (0.0193, 0.0195, 0.0210) |
| X2 | all kinds but the relay system, the life of (b) | NOT RUN (memory). In its place, addition 1: held (0.0209, 0.0253, 0.0241) |
| X3 | all kinds, last tenth: at or under 1.4607 | held (1.4421) |
| X4 | at or under 1.4540, the full nanoGPT network's figure on that tenth | held (1.4421) |
| X5 | going on: at least 0.010 below the first cell | held (0.0266) |
| X6 | the control on T2 under a fifth of the gain | held (it is slightly worse than no second cell) |
| X7 | T2, the second cell's rise on the test text after the three books under 0.03 | held (0.0252; the first cell's is 0.0088) |

Readings (mine).
1. **The bytes at the last six delays tell the answering cell something that none of the other kinds had told it**: the gain does not shrink when the word cells, the timing cells, the pool and the companions are there; with 164 kinds it is larger (0.029 on the test text against 0.015 with ten kinds). With more senders there is more to be read rightly.
2. **Against the reference.** With the second cell the system of cells of this line has 1.4421 on the last tenth of the test text, where the full nanoGPT network of `nano/ref` has 1.454. The limits belong beside it: one text; a last tenth scored for some fifty arms of this line (the second cell itself was fixed before it ran, and it gains as much on books it had never seen); one seed of the reference, taught for 4,000 steps at batch 16 and not for the published run; 4.5 million cells against 10.7 million numbers; one reading against 4,000 steps of batches; word ends given by hand for two kinds. A development-level figure; not a claim that this system is the better learner of English.
3. **What the second cell keeps.** Nothing of its own: after other books its loss on the old text is that of the first cell (T2: 0.001 above; all kinds, one book: 0.003 above to 0.005 below) and, where companions speak for the recent text, 0.022 above after three books. Allowed to go on changing on the return it is 0.006 to 0.024 below the first cell at once. So the system now has three parts that follow the recent text (companions, class cells that start empty in `nano/merge`, the second cell) and one that keeps (the cells that never fade, with the first cell). That division is found, not designed, and nothing yet uses it deliberately.
4. On one worn book of three the gain with all kinds is 0.006; I have no account of it.
5. *Not run:* the life with all kinds as planned (memory); the control with all kinds; the second cell on the life of the books only; a second seed of anything (the system has no chance in it except the relay system's birth).
