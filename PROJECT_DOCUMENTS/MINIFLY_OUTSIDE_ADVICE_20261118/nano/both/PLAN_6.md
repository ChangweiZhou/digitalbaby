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

## 4. Outcome (entered 20 November; `res/s6_prose_seed1.json`; one run, relay seed 1, 69 CPU-seconds; `prose.py`)

The last tenth of the test text, nothing changing, nats a byte. The run gives `relay.py`'s own figure, 1.4858 (that line's file: 1.4858), and the relay system alone 1.916.

| Setting at the score | Loss |
|---|---|
| as learned | **1.4858** |
| the relay system silent (the other trusts as they were learned beside it) | 1.5237 |
| its trust in the top bin (0.165 as learned) set to 0.5 / 1 / 2 | 1.4888 / 1.4957 / 1.5108 |
| its trust in the top three bins set to 0.5 / 1 / 2 | 1.5010 / 1.5473 / 1.6591 |
| its trust in all bins set to 0.5 / 1 / 2 | 1.5318 / 1.7423 / 2.3811 |

The relay system's trust by bin of its largest probability, along the stream (bins 0 to 9; born at 0.25 in that program):

| Bytes read | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 50,198 | 0.24 | 0.16 | 0.19 | 0.19 | 0.20 | 0.21 | 0.23 | 0.24 | 0.25 | 0.25 |
| 451,737 | 0.22 | 0.19 | 0.22 | 0.20 | 0.19 | 0.16 | 0.16 | 0.16 | 0.16 | 0.21 |
| 1,003,854 (the end) | 0.21 | 0.25 | 0.27 | 0.26 | 0.23 | 0.20 | 0.19 | 0.18 | 0.16 | 0.17 |

In the last twentieth of the stream, by bin: the relay system's first choice is the byte that came 9, 14, 27, 40, 51, 61, 70, 82, 89 and 96 times in 100; the whole answer's is 12, 26, 49, 57, 62, 67, 72, 83, 90 and 97 times in 100. In the top bin the whole answer gives the true byte a share of 0.941 and the relay system 0.904.

R1 held (0.165 in the top bin, 0.21 to 0.27 in the bins under one half). R2 held (96.6 against 96.4; the statement's wording in section 3 has a slip of mine, "without... as it stands"; it means the whole answer as it stands). R3 held: one raises the loss by 0.010. R4 held: every setting tried raises the loss.

Readings (mine; one text, one relay seed, one answering cell).
1. **On prose the relay system's trust is also smallest where it is surest, and there it is not a fault.** Where the relay system is sure the whole answer is as right as it is and surer; hearing it more there costs loss (0.010 nats at trust one in the top bin, 0.06 in the top three). The mismatch rule gives it what it adds: it is heard most where it is half sure and the counting cells have least to go on, and it is worth 0.038 nats inside this system (1.5237 without its voice) and 0.0115 against a system taught without it (the other line's 1.4973).
2. **So the fault found on sums does not belong to the rule that a trust moves by the whole answer's mismatch. It belongs to the closed stream.** The same rule, the same kind of sender, the same small trust at the sure bins: on prose every byte read is new, the trust is taught in the company in which it is used, and the taught values are the ones the unread tenth asks for. On sums the stream repeats 8,000 lines, the trust is taught where the cells that hold the line make the answer right, and it is used where they are silent.
3. This agrees with what the coordinator reports from `nano/table/PLAN_18.md` (in an open stream the second answering cell holds its level); I have not read that plan.
4. What follows for the lessons of `PLAN_5.md` (mine, untested): on an open stream there is nothing for them to repair in the trusts, because the stream itself supplies new cases. Their use, if any beyond sums, would be where a system must go on reading a closed or repeating text, or must keep an old text's trusts while reading another (the keeping of `PLAN_5.md`'s lives); that is the test to make on prose, and it is NOT RUN.

NOT RUN: `gate_all.py`'s two answering cells with all kinds; the reported seeds; the books.
