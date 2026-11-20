# One new general idea: a cell that recalls (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The idea is the coordinating session's; sharpening or refuting it is this line's task. **Source.** This branch at 1826e70. Written before any run with a recalling cell on any stream other than inside `suite.py check` (2,500 bytes, compared with the reference, no loss read). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_1.md`: going on, xz is below the base on source code, rows of numbers and markup by 0.02 to 0.43 nats; on prose the base is 0.34 below xz. The base's longest kind is six bytes, half of its cells have been met once, and it cannot use a repeat longer than its longest kind. A compressor of the xz family does little else than use long repeats. The known form of this in compressors that mix opinions is the match model.

## 2. The mechanism, exactly (`suite.py recall=1`; kind type 2 in the kernel)

One more sender, of no kind of text. At each byte it looks up the last place in what the system has read where the last M = 8 bytes stood before (a table from the run of M bytes to the place where it last ended; kept as the stream is read). If there is such a place it says the byte that followed there, **with the message of a cell that has counted once**: log((1 + e) / (1 + S e)) for that byte and log(e / (1 + S e)) for every other, e = 0.02 as everywhere. Its trust is indexed by **how long the two runs agree** (counted back from the present byte and from the place, to at most 2,048 bytes): one level for each length 8 to 23, then levels that double (24 to 31, 32 to 47, 48 to 79, ...), forty at most; every level born at nothing; step 0.00005 as every trust. In the second answering cell it has parts like any sender (six parts indexed by the byte at each of the last six delays). With nothing changing the table of places is not written, so only places in the nine tenths read are recalled (the bytes of the tenth are compared as context); with everything going on, any earlier place.

It is local in the sense of this programme: the sender holds an address and one byte; the answering cell learns how far to trust it at each length by the same rule as for any cell. There is no number in it but M and the cap; **M = 8 is the coordinator's and is not tuned** (longer than the longest kind by two). The levels are laid down by me, before any run.

Cost: one table of 4 million places and one place for each byte read (about 5 numbers a byte of memory in this program, which keeps the stream as 32-bit values); no cells; 40 trusts and 6 x S parts. CPU: a few seconds a stream.

## 3. Question, hypothesis, what refutes it

**Question.** Does one sender that recalls a long repeat close the distance to the compressors on the streams where the base is behind, without harm on prose?

**Hypothesis** (the coordinator's expectation, sharpened). Going on, with the second cell: the recalling sender lowers the loss by at least 0.02 nats a byte on at least three of the four streams py, c, table, xml, by less than 0.01 on tiny Shakespeare, and harms no stream by more than 0.005. **Refuted if** it lowers the loss by less than 0.02 on two or more of the four, or harms any stream by more than 0.005.

| | Statement (going on, second cell, against the same system without the sender) | Chance |
|---|---|---|
| C1 | the hypothesis holds as written | 0.50 |
| C2 | gain on c at least 0.05 | 0.50 |
| C3 | gain on table at least 0.05 | 0.50 |
| C4 | gain on tiny Shakespeare under 0.01, and on the book under 0.01 | 0.75 |
| C5 | with the sender the system is below xz on py and on xml (it is 0.004 and 0.02 from it with the second cell alone, if step 2's figures for those streams come as the first ones did) | 0.60 |
| C6 | it closes at least a third of the remaining distance to xz on c and on table | 0.30 |
| C7 | with nothing changing the gain is under half of the going-on gain on c and on table (the tenth repeats itself more than it repeats the nine tenths) | 0.65 |
| C8 | the trust the answering cell comes to give it rises with the length of agreement (mean trust over lengths 8 to 15 below that over lengths of 32 and more, on at least five streams) | 0.75 |

My own doubt, said beforehand: with the step 0.00005 a level of length that is used a few thousand times cannot bring its trust far from nothing (a use moves it by at most 0.0002), so on short streams the sender may be right and still unheard. If the gains are small, the trusts by level will show whether this is the cause; that would be an argument for step 4 (one rule for the step), not for a number tuned here.

## 4. The variant, only if the hypothesis holds

Several places recalled instead of one: three senders, the last, the one before and the third last place where the last 8 bytes stood (`recall=3`), each with its own trusts by length. One prediction: C9, the gain of three over one is under a quarter of the gain of one over none on at least four streams (chance 0.6).

## 5. Runs, order, what is not run

First the two development streams (tiny Shakespeare, the book): to see that nothing is broken and that C4 is not grossly wrong; M is not changed whatever they show. Then the four other streams once. Not run: other M; a recalling sender whose message is learned (a cell keyed on the length); recall over more than the stream read.
