# A second new general idea, this line's own: a cell also says what it counted last (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). This idea is not on the coordinating session's list; it is mine, tried under "try more new ideas", and is reported to that session as such. **Source.** This branch at the commit that holds this file. Written before any run of the idea on any stream (it has run only inside the check, on 2,500 bytes against the reference). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_1.md`, `PLAN_3.md` and the first figures of `PLAN_4.md`: everything that has helped on the streams where the system is behind the compressors is a way of hearing *what was just read*: going on at all (0.3 to 1.1 nats), the sender that recalls the last place of the last eight bytes (0.07 to 0.12), a faster step (0.02 to 0.05). A cell of the system counts for ever and gives its first count and its latest the same weight. The recalling sender says "what followed the last time" for one long context only. The same question can be put to every cell.

## 2. The mechanism, exactly (`kinds=runs+last`; kind type 3 in the kernel)

Every cell keeps, beside its counts, **the byte it counted last** (one value). For each of the seven kinds of run there is one more sender: where the run's cell has counted, the sender says that byte **with the message of a cell that has counted once** (as the recalling sender does), and its trust is indexed by the level of the run's cell (its count and the number of different bytes it has met: the forty levels every cell has), born at nothing. In the second answering cell it has parts like any sender. Nothing else changes; there is no new number.

What it can and cannot add, said beforehand: where a cell has met one byte only, its last byte is that byte and the sender repeats the cell (the answering cell can give those levels no trust); where a cell has met several, the last one says which is current. It is recency without a rate of forgetting. It is known art in compressors that mix (a context's last byte as an input); what is this programme's is again only that the trust in it is learned by the local rule, level by level.

Cost: one value more for each cell (0.3 to 0.56 million); 7 kinds, 280 trusts, 7 x 6 x S parts; no new cells. CPU about a fifth more.

**The system it is tried on**: the system of `PLAN_4.md` with the rule at eta = 0.03 (`kinds=runs gate=1 recall=1 rule=2 eta=0.03`), with and without the seven senders. The rule of `PLAN_2.md` applies (going on first; the verdict with nothing changing beside it).

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** Going on, the seven senders lower the loss on at least three of the five kinds of stream by at least 0.001 and harm none by more than 0.005; the gain is larger on c and table than on either prose stream. **Refuted if** fewer than three kinds are lowered, or a stream is harmed by more than 0.005, or the gain on prose is the larger.

| | Statement (second cell, going on unless said) | Chance |
|---|---|---|
| E1 | the hypothesis holds | 0.50 |
| E2 | gain at least 0.01 on c and on table | 0.45 |
| E3 | gain under 0.005 on both prose streams | 0.70 |
| E4 | with nothing changing, at least one stream is harmed by more than 0.005 | 0.45 |
| E5 | the trust given to the senders is highest at levels where the cell has met more than four different bytes (mean over those levels above the mean over the levels with one byte met, on at least four streams, for the run of two bytes) | 0.60 |

## 4. Runs, what is not run

The two prose streams first (nothing can be chosen: there is no number), then the four others once. Not run: the last two or three bytes of a cell; counts that fade (the companions of `nano/table/two.py`: a rate of forgetting is a number, and it was chosen on English); the senders on the pool's kinds.

## 5. Outcome (entered 20 November, after the runs; `res/*__last.json`, `report_5.txt`)

Second cell; the system of `PLAN_4.md` at eta = 0.03 without and with the seven senders; the difference in brackets. Cells unchanged (one value more for each); numbers stored 1.00 to 1.01 times; about 5 to 7 CPU-seconds a stream.

| | measure | shakespeare | book | py | c | table | xml | kinds lowered | largest harm | by the rule |
|---|---|---|---|---|---|---|---|---|---|---|
| last | going on | 1.3427 (-0.0278) | 1.3953 (-0.0053) | 0.9657 (-0.0553) | 0.7172 (-0.0781) | 0.3483 (-0.1769) | 0.2953 (-0.0594) | 5 of 5 | -0.0053 | KEPT |
| last | nothing changing | 1.5502 (+0.0621) | 1.4625 (+0.0381) | 1.4681 (+0.0398) | 2.3036 (+0.0196) | 2.0622 (+0.0409) | 0.8696 (+0.0788) | 0 of 5 | +0.0788 | not kept |


**The hypothesis held on the going-on measure, and by more than I expected:** all five kinds are lowered (0.028 and 0.005 on prose; 0.055, 0.078, 0.177, 0.059 on py, c, table, xml) and the gain on c and table is the largest. With this one addition the system is below xz on c (0.717 against 0.724) and on table (0.348 against 0.381) for the first time. **With nothing changing it harms every stream** (0.020 to 0.079). So it is general by the rule going on and the opposite of kept with nothing changing: it is a sender of the present.

**Predictions scored.** E1 held. E2 held (0.078 and 0.177). **E3 failed**: the gain on tiny Shakespeare is 0.028, not under 0.005 (the book: 0.0053, also just over). E4 held (all six harmed). **E5 failed**: for the run of two bytes the mean trust at the levels where the cell has met more than four different bytes is above that at the levels with one byte met on c and xml only (shakespeare 0.10 against 0.14; book 0.07 against 0.23; py 0.15 against 0.21; c 0.21 against 0.19; table 0.17 against 0.19; xml 0.22 against 0.20). I had written that where a cell has met one byte the sender only repeats the cell and would get no trust; it gets the most. My reading now: a cell that has met one byte a few times says it with a message that the smoothing keeps mild, and the sender adds weight to it; the answering cell uses the sender to sharpen what is nearly certain as much as to say what is current.

**My readings.** (1) Together with the recalling sender this is the second idea of the day that is general going on and costs no cells; both say what was just read. (2) The harm with nothing changing is the same in kind as in `PLAN_3.md` and larger; `PLAN_7.md` tests my explanation of it. (3) The gain on tiny Shakespeare is a caution for the programme's older figures: part of what separates "nothing changing" from "going on" on that text (0.11 nats for the base) is of this kind, and no figure with nothing changing can show it.
