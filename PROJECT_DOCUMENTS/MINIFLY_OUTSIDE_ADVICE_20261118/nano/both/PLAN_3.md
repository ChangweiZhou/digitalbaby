# The joined system through a second task, with replay to the relay system from a store of counting cells (step 3) (20 November 2026)

**Authorisation and source** as `PLAN_1.md` (Mr. Xu's words quoted there; branch `claude/nano-both-8mmhfm`, written on top of commit 77a731d). Copied unchanged into this folder: `nano/replay/replay.py` (its class `Counting`, the store, and `kind`). Written before any run of `life.py`. Written when the runs of `PLAN_1.md` were nearly all in (their outcome is entered there after this plan): the joined system of `PLAN_1.md` falls with reading as the counting system does and hardly hears the relay system. This plan measures that system, unchanged, through two tasks; the changed trusts of `PLAN_2.md` are not used here unless an addition says so before its runs. Development level; no gate, seal or review is claimed.

## 1. The question and the life

**Does the joined system keep its addition through a second task and learn the second task, when the relay system is kept by replay as `nano/replay` found?**

Addition 400,000 lines, then subtraction 200,000 lines (`cells2.py`'s texts; the program shows no addition in the second phase); and the short life, 40,000 then 20,000. The cells that count and the two answering cells read the real lines only. After each real line the relay system learns one line drawn from the store (`replay.py`'s `Counting`, runs of at most 8 bytes inside a line, drawn exactly as `replay6.py`'s arm R: same random stream), in both phases, as that line does. The relay system's messages for a block of 50 real lines come from its strengths at the start of the block. The store is a third thing beside the two kinds of cell (118 thousand cells at most by that line's count); drawing the replayed line from the counting cells of the joined system itself is the natural next form and is not run here.

*Measured*, for the joined second cell, the joined first cell, the relay system alone (from the joined run; nothing of the counting cells reaches it) and the joined second cell with the relay system silent at the score; and for the counting system alone on two seeds (73129, 93683): never-shown additions at the end of each phase after a read addition line and after a read subtraction line; never-shown subtractions on the clean score (pairs never shown whose line no shown pair writes, of the 2,000 never shown, as `replay6.py`) after a read subtraction line. Whole answer, own bytes, five draws of the line before. Five reported seeds for the joined arm.

*Check.* The relay system with replay, scored with `replay6.py`'s own draws, against `nano/replay/res/m_z_sub_R_<seed>.json` (their field `past`).

## 2. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis (the coordinator's).* The joined system keeps at least 90% of its addition level through the second task and reaches at least 88 on subtraction (medians of five seeds, long life). *Refuted if* either fails.

| | Statement (joined second cell unless said; median of five seeds) | Chance |
|---|---|---|
| L1 | long life: addition after the second phase (after a subtraction line) is at least 90% of its level at the end of the first phase | 0.80 |
| L2 | long life: that is so because the level at the end of the first phase is low (under 85), and addition after the second phase is higher than before it | 0.55 |
| L3 | long life: subtraction (clean) at the end at least 88 | 0.35 |
| L4 | short life: addition kept at least 90% of its level | 0.60 |
| L5 | short life: subtraction (clean) at least 88 | 0.45 |
| L6 | the relay system from these runs repeats `nano/replay`'s figures within one point at the median (addition after the second phase 93.8, subtraction 88.0) | 0.85 |
| L7 | long life: the joined answer on addition at the end is under the relay system alone by more than 3 points | 0.65 |

## 3. Cost, what is not run

A long life is about 15 CPU-minutes for the joined arm; seven long and seven short lives, three at a time: about 40 minutes of wall time, about 0.5 GB each. Not run: no replay (the other line's 1.8% is taken as it stands); the harsher change of task; replay drawn from the joined system's own cells; the counting system alone on more than two seeds; any tuning.

## 4. Outcome (entered 20 November after all fourteen lives; `res/s3_*.json`, every seed in `report_s3.txt`, made by `report3.py`)

Never-shown lines, whole answer, own bytes, after a read line, %; median of five reported seeds (range); the counting system alone on two seeds (73129, 93683), both given. One thread; a long life takes 667 to 696 CPU-seconds joined and 569 alone, a short one about 70. At the end of the long life: 1.40 million cells that have counted; the store 66,043 cells. Clean subtractions scored: 377 to 412 a seed. The background runner was stopped by the session's 30-minute limit after eight lives and started again; the six remaining lives were run whole, and none was resumed part-way.

**The long life: addition 400,000 lines, then subtraction 200,000.**

| | Relay alone, with replay | Joined, second cell | Joined, first cell | Joined second cell, relay silent at the score | Counting alone, second cell (two seeds) | Counting alone, first cell |
|---|---|---|---|---|---|---|
| addition, end of the first phase | **97.5** (96.6 to 97.9) | 80.4 (77.7 to 92.5) | 82.6 | 74.7 | 75.5, 90.4 | 77.2, 81.2 |
| addition, end of the second phase, after a subtraction line | **93.8** (91.8 to 95.0) | 58.4 (38.9 to 70.0) | 72.9 | 46.4 | 12.3, 64.0 | 70.8, 67.0 |
| the same after an addition line | 93.9 | 58.4 | 72.9 | 46.8 | 11.9, 64.7 | 70.6, 67.4 |
| subtraction (clean), end of the second phase | **88.1** (87.0 to 92.3) | 75.6 (72.4 to 80.5) | 71.8 | 74.3 | 73.8, 74.5 | 64.7, 64.2 |
| share of the addition level kept | 96.3% | 72.6% (45.3 to 78.7) | | | 16.3%, 70.9% | |

**The short life: addition 40,000 lines, then subtraction 20,000.**

| | Relay alone, with replay | Joined, second cell | Joined, first cell | Relay silent at the score | Counting alone, second cell (two seeds) |
|---|---|---|---|---|---|
| addition, end of the first phase | 89.9 (88.4 to 91.9) | **97.2** (96.5 to 98.5) | 93.8 | 96.9 | 97.4, 98.2 |
| addition, end of the second phase, after a subtraction line | 83.7 (80.0 to 89.3) | **88.6** (80.1 to 90.2) | 80.2 | 86.0 | 86.2, 89.2 |
| subtraction (clean), end of the second phase | 64.0 (60.7 to 68.4) | **88.9** (87.3 to 90.1) | 82.1 | 88.9 | 89.3, 87.5 |
| share of the addition level kept | 94.6% | 91.3% (82.7 to 91.8) | | | 88.5%, 90.8% |

*Check.* The relay system with replay, scored with `replay6.py`'s own draws, gives `nano/replay/res/m_z_sub_R_<seed>.json` **to the digit on all five seeds** at the end of both phases (for example seed 73129: 98.04, 97.76 after the first phase; 94.96, 95.28 and subtraction 87.64 after the second). No replayed line was a wrong sum (the store returns lines it was shown).

**The coordinator's hypothesis is refuted in the long life and holds in the short one.** Long: the joined system keeps 72.6% of its addition (90% asked) and reaches 75.6 on subtraction (88 asked). Short: it keeps 91.3% and reaches 88.9.

Predictions. L1 failed (72.6%; I gave it 0.80). L2 failed: the level at the end of the first phase is low (80.4), but addition after the second phase is lower still, not higher. L3 failed as I expected (75.6). L4 held (91.3%). L5 held (88.9). L6 held (93.8 and 88.1 at the median, each seed to the digit). L7 held (58.4 against 93.8).

Readings (mine).
1. **In neither life is the joined answer the better of its parts on everything, and which part is better turns over between the lives.** After 40,000 and 20,000 lines the counting system (and the joined system, which is the counting system there) is ahead on all three measures: 97.2 against 89.9, 88.6 against 83.7, 88.9 against 64.0. After 400,000 and 200,000 lines the relay system kept by replay is ahead on all three: 97.5 against 80.4, 93.8 against 58.4, 88.1 against 75.6. The joined system of `PLAN_1.md` follows the counting cells in both.
2. **The second answering cell loses addition through a long second task**: alone it keeps 12.3 and 64.0 of 75.5 and 90.4 on the two seeds, while the first cell in the same runs keeps 70.8 and 67.0 of 77.2 and 81.2. The cells' counts are all still there; the parts indexed by the bytes at the delays are retaught by 200,000 lines of subtraction. This is new as far as I have read (the roadmap lists it as being measured by `nano/gate2`), on two seeds, and I did not predict it (L1).
3. In the long life the relay system's voice is worth 12 points of addition at the end (58.4 against 46.4 with it silent): the second task gives the mismatch on which it earns trust. It is far from enough.
4. Replay to the relay system is worth more during the first task than I had read from the other line's summary: at 40,000 lines the relay system with replay stands at 89.9, where it has 82.9 without (`PLAN_1.md`); at 400,000 lines 97.5 against 95.5. The relay system with replay is the best system of this line at the long reading on every measure.
5. What this does not show: the life with the changed trusts of `PLAN_2.md` (not run here); more than two seeds for the counting system alone.
