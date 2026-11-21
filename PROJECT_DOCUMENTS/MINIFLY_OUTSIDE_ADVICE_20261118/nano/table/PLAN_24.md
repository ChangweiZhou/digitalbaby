# Cells that count on what the answering cell says: a second level that counts (roadmap second revision, sections 15 and 20; depth in another form) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" and before it "继续大胆尝试，小心求证。" (my translations: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas." "Go on trying boldly, and verify with care."). **Source.** `claude/friendly-darwin-8mmhfm` at 83cb5ac5f; `gate.py` with the option `guess`, checked against numpy (`check24_output.txt`, sixteen cases; one case failed at first because the check's own data made the first cell's answer a tie that rounding decided, and the data of that case were changed, not the program). Nothing else was run with the option. Written before any run. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

1. The line `nano/layers` put answering cells on answering cells and found that depth adds nothing beyond one wide cell: every further cell adds to one sum, and "a cell on a cell is a wide cell taught less well" (section 14).
2. The line `nano/suite` found its largest general gains in senders that tell the answering cells something they could not form by adding trusts: what a cell counted last, what followed the last time these bytes stood here (section 20).
3. What an answering cell says is itself a fact of the moment that no cell holds: the system's own summary of everything its cells of up to six bytes know. Cells that are keyed on that summary would count what actually came when the system said so. That is a second level made of cells that count, not of cells that add; general-purpose compressors that mix contexts have had such a last stage for a long time.

## 2. The one change

Three more kinds of cell, keyed on **the byte the first answering cell holds likeliest**: (a) with how sure it is of it (four steps: at most one half, at most 0.8, at most 0.95, above); (b) with the last byte; (c) with the last two bytes. They count the byte that comes, as every cell does, and speak with a cell's message. They are heard by the second answering cell only, with a trust born at nothing and parts like any sender. The first answering cell is untouched.

*The control:* three kinds of the same shape and number keyed on the byte at the farthest delay in place of the first cell's answer.

*The setting,* the same on both texts: the rule of the step of `PLAN_22.md`, addition 1, with 0.03; every trust born at nothing; the smoothing as each text has had it.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The second answering cell with these cells is at least 0.005 nats below the second answering cell without them on each of six books (the two lives of `PLAN_15.md`; each book's last tenth right after its stream). *Refuted if* it is not 0.002 lower on at least four of the six.

| | Statement | Chance |
|---|---|---|
| Y1 | at least 0.005 lower on each of six books | 0.35 |
| Y2 | at least 0.002 lower on each | 0.60 |
| Y3 | never-shown sums without classes, five reported seeds, after a read line: the second cell's median at least 1 point higher (86.2 without) | 0.40 |
| Y4 | the control gives under a third of the gain on prose | 0.70 |
| Y5 | of the three kinds, the one keyed with the last two bytes earns the largest trust on prose | 0.50 |

## 4. Cost, what is not run

4 S + S squared + S cubed further cells (279,110 on prose, of which most will never count; 2,996 on sums); three more senders a byte. Under twenty CPU-seconds a life, forty a run on sums. Not run: the same cells keyed on the second cell's answer or on the two likeliest bytes; the larger prose systems; streams of other kinds (offered to `nano/suite` if it holds here).

## 5. Outcome (entered 20 November after all runs; `res/gate_*guess*.json`, the printed lines in `out24/`)

**Prose** (runs of 0 to 6 bytes; the second answering cell without the new cells, with them, and with the control; nats a byte on each book's last tenth right after its stream).

| | bryant | burgess | carroll | chesterton-ball | austen-emma | melville |
|---|---|---|---|---|---|---|
| without | 1.4726 | 1.3191 | 1.3979 | 1.6106 | 1.2082 | 1.4351 |
| **with cells keyed on the first cell's answer** | 1.4696 | 1.3095 | 1.3893 | 1.6119 | 1.2016 | 1.4297 |
| lower by | 0.0029 | **0.0096** | **0.0087** | -0.0013 | **0.0066** | **0.0054** |
| the control lower by | -0.0009 | 0.0038 | -0.0005 | -0.0033 | 0.0015 | 0.0022 |

The test text's last tenth right after its stream: 1.4983 without, 1.4956 with (the control 1.5014). At the end of the two lives: 1.5434 and 1.5379 without, 1.5451 and 1.5446 with.

**Sums** without classes, never-shown additions after a read line, second cell, five reported seeds, last reading: 88.6, 85.6, 86.2, 83.8, 87.2 (median 86.2) without; 89.1, 86.6, 85.6, 83.5, 87.0 (86.6) with.

The trusts the three kinds earned on prose (largest over the levels used): with how sure the first cell is 0.083; with the last byte 0.128; with the last two bytes 0.067. Of 279,110 possible cells 8,611 have counted.

**The hypothesis is not held and not refuted.** The second cell is lower by 0.005 or more on four books of six where all six were asked, and by 0.002 or more on five, where under four would refute; one book is 0.001 worse.

| | Statement | Outcome |
|---|---|---|
| Y1 | at least 0.005 lower on each of six books | failed (four of six) |
| Y2 | at least 0.002 lower on each | failed (five of six; one 0.0013 worse) |
| Y3 | sums: the second cell's median at least 1 point higher | failed (+0.4) |
| Y4 | the control under a third of the gain on prose | held over the six books (0.0005 a book against 0.0053), though on one book it gives 0.004 of 0.010 |
| Y5 | the kind keyed with the last two bytes earns the largest trust | failed (the kind keyed with the last byte) |

Readings (mine).
1. **A second level that counts gives something where a second level that adds gave nothing**, and it is small: half a hundredth of a nat on prose on average, uneven from book to book, nothing on sums. By the criterion of section 15 (streams of at least three kinds) it is not shown general; it is entered as a small gain on prose only.
2. It costs almost nothing (8,600 cells that have counted, three senders a byte), so it is worth one run on the streams of `nano/suite`, where senders that tell the answering cell a fact of the moment gained several times more than on prose. Offered to that line; not run here.
3. What is kept of the old text is unchanged or a little worse (0.002 to 0.007).
