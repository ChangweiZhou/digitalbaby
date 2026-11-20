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
