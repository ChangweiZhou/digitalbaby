# Why the senders of recency harm with nothing changing, as a hypothesis with a falsifier: trust by the age of what is said (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). This step is mine, under "verify with care": it tests the explanation I gave for a harm. **Source.** This branch at the commit that holds this file. Written before any run with `age=1` on any stream (it has run only inside the check). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

System Z (`PLAN_6.md`; its six runs are in `res/*__Z.json`, the outcome is entered there): going on it is below xz on all six streams; **with nothing changing it is above the base on five of six** (1.547 against 1.523 on tiny Shakespeare; 2.240 against 2.122 on c; 2.014 against 1.757 on table). Each of the senders that say what was just read showed this in its own step (`PLAN_3.md`: c harmed by 0.116; the last-byte senders: all six streams harmed by 0.02 to 0.08). My explanation so far, untested: while reading, what such a sender says is fresh, and the answering cell learns a trust that fits fresh sayings; with nothing changing the same sayings grow old and the trust stays.

## 2. The change, exactly (`age=1`)

If that is the cause, the answering cell lacks one thing it could know locally: **how old the saying is**. So the trusts of the senders of recency are indexed by age in place of their present levels; nothing else changes and no cell or sender is added.

* A last-byte sender: forty levels = (different bytes the run's cell has met: 1, 2, 3 or 4, more) x (bytes since that cell last counted: 1 to 3, 4 to 15, 16 to 63, ... four-fold, ten steps). Each cell keeps the time it last counted (one number more a cell). The count's size is no longer in the index.
* A recalling sender: forty levels = (bytes of agreement beyond eight: 0, 1, 2 to 3, 4 to 7, ..., 64 and more; eight steps) x (bytes back to the recalled place: under 64, under 1,024, under 16,384, under 262,144, more).

While reading, old sayings occur too (a cell not met for a long time; a place far back), so the trusts for great ages are learned from the system's own life. With nothing changing the time goes on (as `nano/table/PLAN_16.md`, correction 3, asks) while cells and places stand still, so sayings age and fall into the levels learned for old ones. The steps of the two scales are laid down here and not tuned.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With the trusts indexed by age, on the system Z: with nothing changing the loss is lower than Z's by at least 0.02 on at least four of the six streams; going on it is not above Z's by more than 0.005 on any stream. **Refuted if** the loss with nothing changing is lower by less than 0.02 on three or more streams (then age is not what the answering cell lacked), or going on is harmed by more than 0.005 somewhere (then the age index costs what the count's size gave).

| | Statement (second cell) | Chance |
|---|---|---|
| G1 | the hypothesis holds | 0.45 |
| G2 | with nothing changing the system with age is not above the base (first cell, `PLAN_1.md`) by more than 0.005 on at least four streams | 0.30 |
| G3 | going on, the system with age is below Z by at least 0.001 on at least three streams | 0.45 |
| G4 | the trust of the last-byte sender of the run of two bytes falls with age (mean over the three youngest steps above the mean over the oldest three used, on at least five streams) | 0.75 |

## 4. Runs, what is not run

The two prose streams first (there is nothing to choose), then the four others once; 8 CPU-seconds each. Not run: other scales of age; age for the cells themselves (counts that fade); an index of both the count's size and the age (more levels than the forty every sender has).

## 5. Outcome (entered 20 November, after the runs; `res/*__Zage.json`, `report_7.txt`)

System Z without and with the trusts by age; the difference in brackets. One number more a cell (the time it last counted); cells, trusts and parts as Z; 5 to 7 CPU-seconds a stream.

| | measure | shakespeare | book | py | c | table | xml | kinds lowered | largest harm | by the rule |
|---|---|---|---|---|---|---|---|---|---|---|
| age | going on | 1.3344 (-0.0046) | 1.3894 (-0.0017) | 0.9358 (-0.0138) | 0.6908 (-0.0160) | 0.3431 (+0.0016) | 0.2846 (+0.0002) | 3 of 5 | +0.0016 | KEPT |
| age | nothing changing | 1.5454 (-0.0017) | 1.4591 (+0.0010) | 1.4333 (-0.0329) | 2.1829 (-0.0567) | 1.7527 (-0.2613) | 0.8179 (-0.0251) | 4 of 5 | +0.0010 | KEPT |
| age, first cell | going on | 1.3799 (-0.0067) | 1.4280 (-0.0016) | 0.9800 (-0.0227) | 0.8029 (-0.0307) | 0.4586 (-0.0107) | 0.3412 (-0.0039) | 5 of 5 | -0.0016 | KEPT |
| age, first cell | nothing changing | 1.5123 (-0.0074) | 1.4586 (-0.0016) | 1.4211 (-0.0300) | 2.0939 (-0.0679) | 1.6558 (-0.2177) | 0.7723 (-0.0377) | 5 of 5 | -0.0016 | KEPT |

**The hypothesis held as written, and it is not the whole explanation.** With nothing changing the second cell's loss falls by 0.033, 0.057, 0.261 and 0.025 on py, c, table and xml (four streams of six by at least 0.02); going on nothing is harmed by more than 0.002, and four streams gain. But **on the two prose streams the loss with nothing changing does not move** (-0.002, +0.001), and there the second cell stays above the base (1.545 against 1.523 on tiny Shakespeare).

The first answering cell tells the two causes apart (the last two rows). For the first cell, with trusts by age, the loss with nothing changing is 1.5123, 1.4586, 1.4211, 2.0939, 1.6558, 0.7723: **below the base's 1.5226, 1.4676, 1.4568, 2.1222, 1.7567, 0.8077 on all six streams.** So for the first cell age was what was lacking: the senders of recency then help with nothing changing too. What remains is in the second cell: its parts are indexed by (kind, byte at a delay) and not by age, they are learned fast under the rule of `PLAN_4.md`, and for the senders of recency they fit the stretch of text just read. With nothing changing the second cell is 0.03 above the first on tiny Shakespeare and 0.10 above on table; going on it is 0.02 to 0.12 below it everywhere. This reading of the second cell is mine and is not tested here.

**Predictions scored.** G1 held (as written). **G2 failed**: the second cell is within 0.005 of the base or below on three streams (book, py, table), not four. G3 held (shakespeare, book, py, c lower by 0.002 to 0.016). **G4 failed** as written: the trust of the last-byte sender of the run of two bytes is higher over the three youngest steps of age than over the oldest three on py, c, table and xml, not on the prose streams (four of six, not five). The shape is the same on all six and is not the one I pictured: the trust is near nothing or negative when the cell counted 1 to 3 bytes ago, highest at 16 to 255 bytes, and falls to nothing beyond some thousands of bytes.

**My readings.** (1) Age is a general index by the rule going on (three kinds lowered, none harmed) and with nothing changing (four kinds lowered, largest harm 0.001), the only idea of this line kept on both measures. (2) A sender of recency should be trusted by how old its saying is; with that, the first cell gains from recency on both measures on every stream. (3) The second cell with fast-learned parts is itself a sender of the present; whoever scores with nothing changing should read the first cell or slow the parts. Not run: parts indexed by age.
