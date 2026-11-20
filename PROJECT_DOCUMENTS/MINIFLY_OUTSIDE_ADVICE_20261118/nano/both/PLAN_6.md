# The line's one general question put to prose: does the relay system earn trust as a sender, and is its small trust where it is sure a fault of the reading? (20 November 2026)

**Authorisation.** Mr. Xu's words quoted in `PLAN_1.md`, and, as the coordinator relayed them on 20 November (Chinese, exactly): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (the coordinator's translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). **Source.** Branch `claude/nano-both-8mmhfm` on top of commit 0376af4; `nano/table/relay.py`, `trust.py`, `words.py`, `score.py` and `nano/loop/loop.py` are imported from their folders unchanged (their compiled kernels are built there as ignored files; no tracked file outside this folder is changed). Written before the program `prose.py` exists. Development level; no gate, seal or review is claimed.

**A fault of order, mine, stated first.** The coordinator's message with these words was queued at 21:42 and asked that step 4 (`PLAN_4.md`) not be started, as a matter of sums. I read my notifications only at 23:25, after the closing report: step 4 had been run (it ended about 21:53) and `PLAN_5.md` (sums again) as well. Nothing of them is withdrawn; they are what the budget went to, against the steer.

## 1. The question

On sums the relay system's trust ended next to nothing where it was sure (`PLAN_1.md`), and my account was: a trust moves only by the mismatch of the whole answer, so a sender that agrees with an answer already right is never credited. On sums the stream is closed (the same 8,000 lines), so the answer on the stream is right far more often than on new lines. Prose is an open stream: the text read is always new. **On prose, is the relay system's trust also small where it is sure; and is that a fault (the loss would be lower if it were heard more there) or what the loss asks for?**

## 2. What is run

`nano/table/relay.py`'s arm 1, unchanged: the cells that count of T2 (the table's cells to six bytes and the cells that step once a word), one answering cell, and the relay system of `nano/loop` (1,024 cells, 733,249 numbers, return strengths born at nothing) as a given sender with a trust for each of ten bins of its largest probability, reading the test text's stream once (relay seed 1; about 90 CPU-seconds). `prose.py` reads the stream in twenty pieces and keeps, at the end of each piece and for each bin: the relay system's trust; how many bytes fell in the bin; how often the relay system's first choice was the byte that came; how often the whole answer's was; the mean of what moves the trust (the mismatch times the message). Then, on the last tenth with nothing changing: the loss as learned (must be `relay.py`'s 1.4858); with the relay system silent; and with its trust in the top bin, and in the top three bins, set to one half, one and two.

NOT RUN, for want of budget (about an hour is left): the two answering cells of `gate_all.py` with all kinds (4 to 7 GB; the coordinator's first choice), the reported seeds, the books.

## 3. Predictions (mine, before the run)

| | Statement | Chance |
|---|---|---|
| R1 | at the end of the stream the relay system's trust in its top bin is under 0.3 and lower than in its bins under one half | 0.85 (the other line's file shows 0.16) |
| R2 | in the top bin the whole answer without... as it stands is right at least as often as the relay system's own first choice | 0.80 |
| R3 | setting the top bin's trust to one RAISES the loss on the last tenth (the small trust is what the loss asks, not a fault) | 0.70 |
| R4 | no setting tried lowers the loss by 0.002 nats or more | 0.70 |

If R3 holds, the reading for the programme is: the fault found on sums belongs to the closed stream (trusts taught on what was counted), not to the rule that a trust moves by the whole answer's mismatch; on an open stream that rule gives a sender what it adds.
