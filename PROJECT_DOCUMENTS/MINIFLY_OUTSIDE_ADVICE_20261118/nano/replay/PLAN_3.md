# Keeping by replay, third plan: a harsh change of task that brings no new byte (20 November 2026)

By the same separate agent session as `PLAN_1.md` and `PLAN_2.md`; same folder, same source. Written before any run of this step; the check of the program was run first (`check_replay3_output.txt`; 300 lines of teaching and an enumeration). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation and origin.** Asked by the coordinating session in a message of 20 November, after `PLAN_2.md`, whose change of task was a mild one (the plain system kept 63.7%). The question, the arms, the hypothesis and the bound of refutation are the coordinating session's. The second text is that of `nano/keep`'s `PLAN_8.md`, test 3 (read on `claude/nano-keep-8mmhfm` at `6f2f61e`; written anew in `replay3.py`, nothing copied): subtraction with the bytes of addition, `aa=bb+ccc` for aa - bb = ccc, larger number first. The coordinating session reports that line as finding there 1.4% of addition kept by the plain system and 0.4% by its candidate; I have not rerun those.

## 1. The question

Addition for 400,000 lines, then that second text for 200,000. No byte value arrives that addition did not use, and the answer stands at the same three bytes of a line. **Does replay from the counting cells keep addition through this change?**

## 2. What "no new line" means here (checked before the runs)

`nano/keep` checked by enumeration that the second text activates no line that addition does not, counting every byte of a line. In the relay system as it stands, one byte of each text line is not learned from (the first byte of a line, named after the newline). Counting only the bytes from which the system learns, my enumeration (`check_replay3.py`) finds that the second text activates **two lines that addition never activates there**: `=` four bytes back and `+` seven bytes back (in addition those two are active only at the byte that is not learned from). So in this system the change brings no new byte value, and two lines of the 140 whose strengths addition never changed; the usual subtraction text brings nine such lines. I run the text as given and say this beside every figure; it bears on how the plain system and the controls see the change, not on replay, which does not use the lines at all.

*The counting cells on the two texts.* With K = 8 the longest cell of each byte of a line holds the byte at the third place of the line (`+` or `=`) for every byte of the answer, so the two texts never share a cell there, and every replayed line is a line that was counted (the check: 30,000 of 30,000, on a mixture of the two texts). As in `PLAN_1.md`, the counting cells are a complete store of the lines shown.

## 3. Arms, measures

`replay.py`'s relay system, counting cells and replay, imported unchanged. Arms as in `PLAN_1.md`: **P** (no replay), **R** (one replayed line for each real line), **O** (control: one real addition line drawn by the program for each real line), **Z** (control: replay with the cells emptied at the change). Nothing is tuned; seed 1 is run as a development pass and nothing is chosen on it.

Measures at the last reading of the second phase: never-shown addition (the 500 pairs of before). The second task three ways, because the fault the coordinating session found in the subtraction split holds for this text as well (a never-shown pair (a, b) with a < b writes the line of (b, a), which may be shown): "mixed" (the 500 never-shown pairs as `cells2.py` scores them), "clean" (those of the 500 whose line no shown pair writes) and "clean, wide" (the same among all 2,000 never-shown pairs, about 400; the figure I report first). Also: the second task before it is taught; replayed lines by text and their share correct; cells; CPU.

## 4. Hypothesis and what would refute it (the coordinating session's)

**Never-shown addition at the end is at least 50% with replay (median of the five reported seeds). Refuted under 20%.** Between 20 and 50 the hypothesis fails without being refuted, and I would say so.

## 5. Predictions (mine), with chances

- F1 (0.8): P keeps under 5% of addition at the median.
- F2 (0.7): the hypothesis holds (at least 50%).
- F3 (0.5): R keeps at least 10 points less addition here than in `PLAN_1.md` (77.2), the two texts sharing every byte value.
- F4 (0.9): Z keeps under 5% of addition.
- F5 (0.6): with replay the second task (clean, wide) is at least 50% at the median.
- F6 (0.95): every replayed line is well formed and correct on every seed.
- F7 (0.6): R is within 8 points of O on addition at the median of the paired differences.

## 6. Cost, not run

24 compiled runs of 30 to 65 seconds: about 7 minutes on four cores. The numpy loop is the one checked in `PLAN_1.md` and, over two phases of this text, in `check_replay3.py`; no full-length numpy run. The ten further seeds where two arms are within five points and it matters. Not run: the text with the two lines made known to the system beforehand (`nano/keep`'s `openall`, which is a rule of that line's candidate and has no counterpart here); r = 1/2; K = 6; the other order of the tasks.

---

## 7. A dated addition, entered 20 November after the five reported seeds were read and before its runs

On the five reported seeds (scored in section 8) R keeps 65.6% of addition and O 80.8; paired, R less O is -19.2, -2.2, -7.4, -20.6, -4.4 (median -7.4, five of five below), while on the second task (clean, wide) R is 7.8 above O at the median. In `PLAN_1.md` a like difference on five seeds (-7.6) vanished on fifteen. Whether replay keeps less than the program's old lines on this harsher change matters for the reading, so R and O are run on 1001 to 1010 and the fifteen are reported together.
- G1 (0.5): on fifteen seeds R is more than 3 points below O on addition at the median of the paired differences.
- G2 (0.6): on fifteen seeds the sum of the two scores (addition, and the second task clean, wide) is lower for R than for O at the median of the paired differences.

## 8. Outcome (entered 20 November, after all runs of this plan)

Tables: `report3.txt` (made by `report3.py` from `res/h_*.json`); the check: `check_replay3_output.txt` (passed).

**The five reported seeds** (73129, 93683, 116089, 406364, 545353; median). Never-shown pairs, whole answer right, %, at the last reading of the second phase. The second task is given clean, wide (377 to 412 never-shown pairs whose line no shown pair writes), with clean (of the 500) and mixed beside it.

| | addition after its own phase | addition at the end | second task, clean, wide | clean | mixed |
|---|---|---|---|---|---|
| P, no replay | 84.2 | 2.2, 0.2, 0.6, 2.0, 1.4; **1.4** | 82.3, 80.9, 86.2, 85.9, 82.5; **82.5** | 85.6 | 83.8 |
| R, one replayed line for each real line | 86.3 | 62.0, 74.2, 65.4, 65.6, 76.4; **65.6** | 68.0, 66.7, 59.2, 73.5, 78.7; **68.0** | 72.1 | 72.2 |
| O, the program's old lines (control) | 86.1 | 81.2, 76.4, 72.8, 86.2, 80.8; **80.8** | 69.7, 57.1, 66.0, 65.8, 62.4; **65.8** | 58.7 | 65.8 |
| Z, replay, cells emptied at the change (control) | 86.3 | 1.8, 0.8, 0.6, 0.2, 2.2; **0.8** | 84.2, 80.9, 88.9, 90.3, 90.0; **88.9** | 86.1 | 89.0 |

Before it is taught, the second task is at 0.0% in every arm and seed. Every replayed line is well formed and correct on every seed (100.0%); in the second phase the replayed lines that are addition fall from 94% to 69% by reading (seed 73129: 47,097, 42,069, 38,203, 34,504 of 50,000). Counting cells at the end in R: 66,043; CPU 62 seconds.

**The hypothesis holds.** With one replayed line for each real line, never-shown addition after 200,000 lines of the second text is 65.6% at the median (62.0 to 76.4; every seed above 50), where the plain system keeps 1.4%, the figure `nano/keep` reports for it on this text. Replay keeps addition through a change of task that brings no new byte value, as the coordinating session expected and for the reason it gave: the replayed lines are drawn from the counts and do not depend on what the second task looks like.

**Fifteen seeds for R against O** (the five, then 1001 to 1010). Addition at the end: R 67.8, O 78.6; paired R less O -4.6 at the median, below on fourteen seeds of fifteen (-20.6 to +2.8). The second task (clean, wide): R 68.6, O 64.1; paired +7.8, above on ten of fifteen. The sum of the two scores: R 138.2, O 140.7; paired -4.7, seven seeds above and eight below.

**What I read from it.**

1. *Replay keeps, but this change is dearer than the one of `PLAN_1.md`, for replay and for the program's old lines alike.* There the two scores summed to about 156 under either; here to about 138 to 141, against a plain system that has 82.5 on the second task alone. Both tasks are held at 65 to 68 where the plain system holds one at 82 to 86. The risk the coordinating session named is what shows: the two texts use the same byte values at the same places but for two (the operators' places), and the relay system has to hold apart more than it did with a new operator.
2. *On this change replay keeps somewhat less of the old task than the program's old lines do, and gives somewhat more of the new* (-4.6 and +7.8 on fifteen seeds), unlike `PLAN_1.md`, where the two were level. The cause lies in what is replayed: the program's old lines are all addition, while a replay draws from everything counted, so by the end three replayed lines in ten are of the second task. The sum does not tell the two apart (seven seeds against eight). So replay from the cells' counts stands where the program's rehearsal stands, at another point of the same trade between the tasks; a replay that drew less from recent counts would move it along that trade, which is not run.
3. *It is again the old lines that keep*: with the cells emptied at the change, 0.8%.
4. *The reservation of section 2 stands beside the figures*: at the bytes from which the system learns, this text activates two lines that addition never trained (`=` four bytes back, `+` seven bytes back); and the reservation of `PLAN_1.md` stands too: the counting cells are a complete store of the lines shown.
5. For the coordinating session's summary sentence: on these runs replay needs no new byte; it does need a store that holds every shown line whole (the cells of eight bytes), and with a store that does not (six bytes, `PLAN_1.md`) it keeps nothing. It answers never-shown lines of the old task because the relay system does, not because the store holds them.

**Predictions.** F1 held (1.4). F2 held (65.6). F3 held (paired against `PLAN_1.md`'s R on the same seeds: -15.2, -10.8, -11.8, -10.0, -3.0; median -10.8). F4 held (0.8). F5 held (68.0). F6 held. F7 held on five seeds, narrowly (-7.4). G1 held (-4.6). G2 held by its letter (-4.7 at the median) with seven seeds above and eight below, which I do not count as a difference.

**Not run:** as section 6; P and Z on the ten further seeds; a replay that draws from older counts more than from recent ones.
