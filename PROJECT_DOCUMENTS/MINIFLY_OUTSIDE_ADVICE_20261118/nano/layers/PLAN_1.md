# Further answering cells and their gates, step 1: which parts of the second cell carry its gain (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/friendly-darwin-8mmhfm` at 7c653ed05; this line is the branch `claude/nano-layers-8mmhfm`, folder `nano/layers`. `nano/merge/merge.py` and `nano/table/all.py` (with `two.py`, `words.py`, `table.py`, `place.py`, `relay.py`, `trust.py`) are imported unchanged from that commit; `score.py` is copied unchanged from `nano/table`. Written before any run of this step (two runs that repeat the coordinator's figures were made first, section 2). Development level; no checker gate, seal or review is claimed.

## 1. What this line is for, and the hypothesis of the whole (the coordinator's, to be sharpened or refuted here)

A second answering cell that hears the same senders and the first cell's branches, with one part of the trust for each delay indexed by (kind of sender, level of counts, the byte at that delay), every part born at nothing and taught by the second cell's own mismatch, gains on sums and on prose with one form (`nano/table/PLAN_15.md`, `PLAN_16.md`). The hypothesis of the whole line: **with one frozen setting, further gates and further cells are at least 1 point above the single second cell on never-shown sums (five reported seeds, 40,000 lines, with classes, after a read line) and at least 0.010 nats below it on a book no line has opened (`edgeworth-parents`, read after the test text's stream). Refuted if neither is reached.** Steps 2 to 4 (other gates; cells on cells; gates drawn by search) each get their own plan before their runs. Choices are made on sums seed 1 and on the three worn books only.

## 2. The program, and what was verified before this plan

`layers.py`. The kernels of the two base systems are taken as text and one call is inserted after the first cell has answered and before anything is learned. The call runs a list of further cells, each with families of gated parts (the file's head lists the families). Nothing flows back into the base system. Several further cells are computed in one run, each from the same base system, so arms share a run.

*Checks* (`check_output.txt`): nine further cells of every family, a cell on a cell and a reading cell, compiled against a plain numpy reference, on both kernels: largest difference 4e-14; the first cell's trusts and counts equal the base kernels' own exactly. *Repeats of the coordinator's figures* (`res/sums_armssecond_seed1.json`, `res/life_armssecond_booksworn.json`): sums, seed 1, with classes, after a read line: first cell 92.24, second cell 96.48 (the coordinator's `gate_sums_gate0_modeclasses_seed1.json`: 92.24, 96.48; all eight figures equal); prose T2, the life with the worn books: first and second cell on the test text and the three books equal the coordinator's `gateall_life_booksworn_gate0_...json` to twelve digits (1.4802, 1.3162, 1.4109 and 1.4621, 1.2932, 1.3802).

## 3. Step 1: one question

**Which parts of the second cell carry its gain, and what is the least that holds it?** Arms, all in one run on each text, each a second cell that hears the first cell's branches:

| Arm | Parts |
|---|---|
| second | every delay line of the system (9 on sums, 6 on prose), each indexed by (kind, level, the byte there): the coordinator's cell |
| delay j, for each j | the part of delay j only |
| delays 1-2, delays 1-3 | the parts of the nearest two and three delays |
| no-level | every delay, indexed by (kind, byte) |
| no-kind | every delay, indexed by (level, byte) |
| byte-only | every delay, indexed by the byte only: one number for each delay and byte, shared by all senders |
| control | as many parts as delays, each reading a constant (the coordinator's control) |

*Systems and material.* Sums: `nano/merge`'s system with classes at that line's frozen setting (256 kinds of at most four of nine delays and 32 class kinds; e = 2; step 0.1 over the square root of one plus the uses), development seed 1, 40,000 lines, never-shown sums scored after a read line (mean of five draws; `loop_small_case/ERRATA_20261120.md`, section 3) with the empty past beside it; mean of the last three readings and the last. Prose: T2 (`all.py` without companions, relay, since, pool: 10 kinds; e = 0.02; step 0.00005), the life "the test text's stream, then bryant-stories, burgess-busterbrown, carroll-alice", each last tenth right after its stream with nothing changing. *The gain* of an arm is its figure less the first cell's in the same run; on prose the mean over the three books. One seed of sums carries about one point of noise (500 held sums); differences under a point between arms are not read.

## 4. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* A small part of the second cell holds most of its gain on both texts: parts indexed by (kind, byte) at no more than three delays hold at least three quarters of it. *Refuted if* on either text every arm with at most three delays holds under half. (The arm "kind and byte at three delays" itself is not in this run; the reading combines the arms no-level and delays 1-3, and the combination is run in step 2 if both hold.)

My reading of what the gate does on sums, which the predictions follow: the bytes at delays 2, 5 and 8 each say where in the line the byte to be named stands (the equals sign, the plus sign or the line end is there), and the byte at delay 1 does not for the tens digit.

| | Statement | Chance |
|---|---|---|
| S1 | prose: delay 1 alone gives at least half of the second cell's gain (mean of three books) | 0.60 |
| S2 | prose: delays 1-3 give at least 0.9 of it | 0.55 |
| S3 | sums: the best single delay gives at least half of the second cell's gain | 0.55 |
| S4 | sums: the best single delay is 2, 5 or 8 | 0.50 |
| S5 | sums: delay 1 alone gives under half | 0.60 |
| S6 | no-level gives at least three quarters of the gain on both texts | 0.50 |
| S7 | no-kind gives under half on both texts | 0.55 |
| S8 | byte-only gives under a quarter on both texts | 0.60 |
| S9 | the control gives under a quarter on both texts | 0.80 |

## 5. Size, cost, what is not run

Sums: 1,231,004 cells that have counted at seed 1; the first cell has 11,520 trusts; the second cell 1,451,520 parts (702,110 used); the other arms between 14 and 1.45 million. Prose: 2.42 million cells; 440 trusts; 156,000 parts (83,709 used). The numbers used are reported with each figure. One run on each text: about ten CPU-minutes on sums (sixteen further cells), two on prose. *Not run in this step:* the reported seeds; the unopened book; the new books; other gates; cells on cells; search; sums without classes; long reading.
