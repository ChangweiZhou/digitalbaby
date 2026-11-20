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

## 6. Outcome (entered 20 November, after the runs; `res/*__recall1.json`, `res/*__recall3.json`, `report_3.txt`)

**A fault found on the development streams and repaired before any claim.** The first runs on tiny Shakespeare and the book gave 0.0002 nats a byte going on: the table of places was written one byte early, so the place looked up was the present one and the sender said the byte that was about to come. The plain-Python reference had the same fault, which is why the check had passed; the two development runs showed it at once (their outputs are not kept as results). Repaired in the kernel and the reference (a place is written after its byte is read), and `suite.py check` now has a test that would have caught it: every byte from some position on is replaced by another value and no answer up to and including the one for that position may move (`check_output.txt`, all ten cases). The four other streams were first run after the repair.

Nats a byte on the last tenth, second answering cell (parts by kind and byte), without and with the recalling sender; the difference in brackets. No cells are added; 40 trusts and 6 x S parts for a sender; a table of places (one place a byte read); 4 to 6 CPU-seconds a stream.

| system | measure | shakespeare | book | py | c | table | xml |
|---|---|---|---|---|---|---|---|
| without | nothing changing | 1.4978 | 1.4388 | 1.4349 | 2.1137 | 1.8376 | 0.7915 |
| without | going on | 1.3876 | 1.4184 | 1.1228 | 0.9470 | 0.6944 | 0.4398 |
| one place | nothing changing | 1.4928 (-0.0050) | 1.4312 (-0.0075) | 1.4342 (-0.0006) | 2.2299 (+0.1162) | 1.8497 (+0.0121) | 0.7773 (-0.0142) |
| one place | going on | 1.3783 (-0.0093) | 1.4084 (-0.0100) | 1.0351 (-0.0877) | 0.8448 (-0.1022) | 0.5700 (-0.1244) | 0.3707 (-0.0691) |
| three places | nothing changing | 1.4891 | 1.4273 | 1.4323 | 2.1654 | 1.8331 | 0.7680 |
| three places | going on | 1.3745 | 1.4043 | 1.0209 | 0.8270 | 0.5464 | 0.3570 |
| xz -9e | (goes on) | 1.7496 | 1.7876 | 1.1265 | 0.7244 | 0.3814 | 0.4333 |

**The hypothesis held on the measure it was written for** (going on): the gain is 0.088, 0.102, 0.124 and 0.069 on py, c, table and xml, 0.0093 on tiny Shakespeare, and no stream is harmed. **With nothing changing it does not hold as a general gain:** c is harmed by 0.116 and table by 0.012. My reading of the harm: in the tenth of c the sender recalls places in files unlike the present one; the trust it earned while reading (where its repeats were recent) is too high for them, and with nothing changing the trust cannot come down. This is the weakness of the frozen measure on joined streams again (`PLAN_1.md`, reading 2), but it is also a true statement about the sender: its worth is in going on.

**Predictions scored.** C1 held (going on). C2 held (c: 0.102). C3 held (table: 0.124). C4 held narrowly (0.0093 and 0.0100, the latter 0.009997). C5 held (py 1.035 against xz 1.127; xml 0.371 against 0.433). C6 held, against my 0.30: it closes 46% of the distance to xz on c and 40% on table. C7 held (with nothing changing c and table are harmed). **C8 failed**: the mean trust over lengths of 32 and more is above that over lengths 8 to 15 on py and c only; on the other four it is below. My doubt of section 3 is what the figures show: the trusts are small everywhere (0.004 to 1.15) and smallest at the levels used least, the long ones; on tiny Shakespeare no trust of the sender is above 0.18. The answering cell has not had the uses to learn how far a long agreement can be trusted. C9 (the variant) held: three places gain 0.004, 0.004, 0.014, 0.018, 0.024, 0.014 over one, under a quarter of the gain of one over none on the four streams where that gain is large (and 40% of it on the two prose streams, where it is small).

**Where the system now stands against xz, going on** (second cell, three places): below it on prose by 0.375 and 0.383, on py by 0.106 and on xml by 0.076; above it on c by 0.103 and on table by 0.165.

**My readings.** (1) The idea is general by the rule of `PLAN_2.md` on the going-on measure (five kinds of five lowered, none harmed) and not on the measure with nothing changing. (2) It is the largest single gain of the day on four of the six streams, for no cells. (3) It is known art (the match model of compressors that mix); what is this programme's own is only that its trust is learned by the answering cell's local rule like any other. (4) What is left on c and table is, I think, the speed of the trust: step 4.
