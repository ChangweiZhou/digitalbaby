# Line `nano/gate2`, sixth plan: a measurement. Is the even loss with long reading in the trusts or in what the cells say? (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written after the first results of `PLAN_5.md` had been read (its cell is far below the second cell at 40,000 lines and at 400,000 lines without classes: its hypothesis is refuted; the outcome is entered there when its last runs end) and before any run of this plan except a smoke test that the option runs (development seed 1, 6,000 lines).

## 1. The evidence, and the question

Every counting system loses with long reading. In this line: without classes the first and the second cell lose 17 to 18 points together between 40,000 and 400,000 lines (`PLAN_2.md`, section 7); with `nano/merge`'s setting C the first cell loses 9 to 11 and the second 2 to 7 (section 9); the first cell's reading of the plain kinds alone falls from 79% to 7 to 21%. `PLAN_5.md` supposed the cause was that trusts are fitted with the exact cell speaking, and its remedy made things much worse, so that account is at least incomplete. Two things change during a long reading: the trusts (and the second cell's parts) go on being taught, almost only on lines met before; and the cells go on counting, so that what a cell says, log((c(v) + e) / (n + S e)), grows sharper without limit (a follower never met is spoken against by log(n) more). No run so far separates them.

*Question.* If every trust and every part stops changing at 40,000 lines and the cells go on counting, does the level of 40,000 lines hold over 400,000 lines?

## 2. The measurement

No mechanism is added. `gate2.py` with `freeze_at=40000`: after the reading at 40,000 lines the step of the trusts and of the parts is nothing (first cell, second cell and control alike); the cells count as before; classes, where present, go on being found by `merge.py`'s rule (a pair kept keeps its trust; a new pair is born with nothing and stays there). Runs: 400,000 lines; seeds 116089, 406364, 545353; (a) without classes; (b) with classes at setting C (`keep=2`, in which the exact kind is not dropped; `PLAN_2.md`, section 9). Scores as before. Set beside the runs already made with nothing frozen (`s400_none_seed*`, `s400_C_seed*`), which are identical up to 40,000 lines.

*Size and cost.* As those runs: 1.22 to 1.24 million cells; about six to eight CPU-minutes a run; six runs.

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* The even loss is in what the cells say as their counts grow, not in the teaching of the trusts: with the trusts frozen at 40,000 lines the first cell without classes still loses at least 10 of its 17 to 18 points (it ends under 70 at the mean of the last three readings on at least two seeds of three). *Refuted if* it ends at 75 or more on at least two seeds (then the loss is mostly the trusts' drift, and the remedy is in how they are taught). Between: both.

My chance for the hypothesis: 0.5. I have one argument each way. For: a cell's speech against a follower it has not met grows as the logarithm of its count, so a kind that is wrong on a never-shown line speaks against the true byte ever more strongly while the trusts stand still. Against: the plain kinds alone fall to 7 to 21%, which is more than sharper speech from the same trusts would seem to give, and the control, whose step is nine times as large, ends above the first cell, so the size of the step matters.

| | Statement (never-shown additions after a read line, mean of the last three readings at 400,000 lines) | Chance |
|---|---|---|
| M1 | without classes, trusts frozen: the first cell under 70 on at least two seeds of three | 0.50 |
| M2 | without classes, trusts frozen: the first cell at least 3 points above the first cell of the run with nothing frozen (62.3, 61.8, 62.4), on at least two seeds | 0.60 |
| M3 | without classes, trusts frozen: the second cell at least 3 points above the second cell of the run with nothing frozen (68.1, 69.0, 69.1), on at least two seeds | 0.60 |
| M4 | setting C, trusts frozen: the second cell at 90 or more on all three seeds (with nothing frozen: 93.0, 90.0, 95.1) | 0.50 |
| M5 | setting C, trusts frozen: the first cell at least 3 points above the unfrozen run (82.8, 78.3, 83.4) on at least two seeds | 0.50 |

## 4. Not run

The other arm (counts frozen, trusts taught on): it needs a change to the compiled loop and is left for a later plan. Any remedy.

## 5. Outcome (20 November; `res/f_s400_none_seed*.json`, `res/f_s400_C_seed*.json`, `report_6a.txt`, `report_6b.txt`)

Every trust and part frozen after the reading at 40,000 lines; the cells count on to 400,000 lines. Up to 40,000 lines each run is identical to its run with nothing frozen. Never-shown additions after a read line; the reading at 40,000 lines, then the mean of the last three readings at 400,000 lines with the trusts frozen, and in brackets the same with nothing frozen (`PLAN_2.md`). CPU 329 to 332 s without classes, 451 to 466 s with setting C.

| Setting | Seed | First cell | Second cell | Control |
|---|---|---|---|---|
| without classes | 116089 | 80.1; **74.4** (62.3) | 86.3; **64.2** (68.1) | 83.8; 76.5 (66.7) |
| | 406364 | 78.6; **67.5** (61.8) | 84.6; **58.9** (69.0) | 80.0; 71.2 (65.5) |
| | 545353 | 77.8; **67.8** (62.4) | 85.6; **58.4** (69.1) | 82.5; 62.0 (63.5) |
| setting C | 116089 | 92.0; **89.8** (82.8) | 96.2; **94.0** (93.0) | 91.9; 80.8 (86.6) |
| | 406364 | 89.0; **80.2** (78.3) | 97.0; **86.1** (90.0) | 88.8; 81.5 (82.7) |
| | 545353 | 91.8; **76.9** (83.4) | 96.6; **94.3** (95.1) | 92.5; 64.0 (84.9) |

Without classes, the first cell with frozen trusts, along the reading (seed 116089; 50,000 to 400,000 lines): 79.4, 74.2, 75.6, 76.0, 76.4, 74.8, 74.2, 74.2; the second cell: 84.8, 72.8, 66.7, 64.8, 66.0, 64.4, 64.0, 64.2. All of the loss is again at the tens digit.

**What it shows (mine).**

1. *Both.* Without classes, with the trusts standing still, the first cell loses 5.7, 11.1 and 10.0 points; taught on, it loses 17.8, 16.8 and 15.4. So about a third to two thirds of its even loss is in what the cells say as their counts grow, with no trust moving, and the rest is added by going on teaching the trusts on lines met before. Neither account alone is right; by the lines I drew the hypothesis holds, narrowly (under 70 on two seeds of three: 67.5 and 67.8; the third at 74.4), and is not refuted.
2. *Going on teaching harms the first cell and helps the second.* Frozen, the first cell ends 12.1, 5.7 and 5.4 points *above* itself taught on; the second cell ends 3.9, 10.1 and 10.7 points *below* itself taught on, and below the frozen first cell. Parts fitted to what the cells said at 40,000 lines go stale as the cells' speech sharpens (the second cell loses 22 to 27 points with its parts frozen), and the second cell's own teaching wins part of that back; the first cell's teaching makes things worse. A cell that knows the bytes at the delays can use the later lines; a cell with one trust for a kind and level cannot.
3. *With setting C the picture is mixed and I draw nothing firm from it*: the second cell is within 1 point of its taught self on two seeds and 3.9 below on the third; the first cell is 7.0 and 1.9 above and 6.5 below. Class kinds found anew after the freeze are born with nothing and stay there, which muddies this arm.
4. *The loss with the trusts frozen levels off* (seed 116089: reached by 100,000 lines and flat after), which is not what a speech against unmet followers growing as the logarithm of the count would give by itself; I do not know why, and it is the next thing to look at: which kinds' cells change what they say between 40,000 and 100,000 lines (the kinds whose delays reach into the line before go on meeting new combinations long after the 8,000 shown lines have all been met).

**Predictions.** M1 held, narrowly (67.5 and 67.8 under 70; 74.4). M2 held (+12.1, +5.7, +5.4). M3 failed, the other way (-3.9, -10.1, -10.7). M4 failed (86.1 on one seed). M5 failed (one seed of three). Two of five.
