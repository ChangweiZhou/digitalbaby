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
