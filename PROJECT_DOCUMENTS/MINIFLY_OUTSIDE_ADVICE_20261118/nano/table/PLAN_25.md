# How the loss and the store move with the bytes read: one life of six million bytes of prose (roadmap second revision, sections 15 and 20; a measurement) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。" (my translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas."). Earlier the same day he asked how the system's cost compares with the usual methods in order of growth, and what changes if the text is finite. **Source.** `claude/friendly-darwin-8mmhfm` at f4e1a4ba8; `gate.py` as it stands. Written before the program of this plan (`curve.py`) exists and before any run. Development level; no checker gate, seal or review is claimed.

## 1. Why

The bitter lesson is a statement about what happens as data and computation grow. Every figure of this programme is at one size: about a million bytes. Nothing says whether the system's loss keeps falling with more text, how fast, or what the store costs as it does; `growth.py` counted cells along one text and scored nothing. The measure is now the loss while going on (section 20), which can be read at every point of a life.

## 2. What is measured

One life: all the English prose in the repository that this line has used or that is unopened, read once in this order: the test text (tiny Shakespeare, all of it), `bryant-stories`, `burgess-busterbrown`, `carroll-alice`, `chesterton-thursday`, `chesterton-brown`, `austen-persuasion`, `chesterton-ball`, `austen-emma`, `melville-moby_dick`, `edgeworth-parents` (the last is opened here for the first time by this line; `nano/layers` opened it once): about 6.3 million bytes, mapped to the test text's 65 byte values as in `life.py`.

The system: runs of 0 to 6 bytes (nothing given by hand), the two answering cells, with no number set for a text: the rule of the step with 0.03 (`PLAN_22.md`, addition 1), every trust born at nothing, and the learned room for unmet followers (`PLAN_23.md`). The stream is read in blocks of 10,000 bytes; each block is answered with the system as it stands before the block and then learned (so the loss is that of going on, a block at a time).

Reported: the loss of each cell over every stretch of the life in which the bytes read double (100,000 to 200,000, and so on), and over each text; the cells that have counted and the counts above zero at each doubling.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| Z1 | on the stretches after the first 400,000 bytes the second cell's loss falls with every doubling of the bytes read, save where a new author begins | 0.50 |
| Z2 | from the stretch 400,000 to 800,000 to the stretch 3.2 to 6.3 million the second cell's loss falls by at least 0.10 nats | 0.45 |
| Z3 | the cells that have counted grow as the bytes read to a power between 0.55 and 0.75 over the last three doublings | 0.65 |
| Z4 | the second cell's gain over the first is larger on the last stretch than on the stretch 400,000 to 800,000 | 0.60 |

These are expectations of a measurement; there is no mechanism to hold or fail. The texts differ in difficulty (the books' own figures in `PLAN_15.md` run from 1.22 to 1.63), so the curve mixes more reading with other authors; the losses by text are given so that the two can be told apart as far as they can.

## 4. Cost

About five CPU-minutes and 2 GB. Not run: the same life in another order; a life of one author; streams of other kinds (`nano/suite` holds them, at one to two million bytes each).

## 5. Outcome (entered 20 November after the run; `res/curve.json`, `out25/curve.txt`)

One life of 6,261,576 bytes, read once; 142 CPU-seconds. Loss in nats a byte while going on (a block of 10,000 bytes answered, then learned).

| Bytes read | first cell | second cell | the second lower by | cells that have counted | counts above zero |
|---|---|---|---|---|---|
| 100,000 to 200,000 | 1.6014 | 1.5831 | 0.018 | 178,135 | 296,484 |
| 200,000 to 400,000 | 1.4884 | 1.4587 | 0.030 | 267,902 | 472,082 |
| 400,000 to 800,000 | 1.4783 | 1.4506 | 0.028 | 403,173 | 752,612 |
| 800,000 to 1,600,000 | 1.5079 | 1.4712 | 0.037 | 619,763 | 1,205,863 |
| 1,600,000 to 3,200,000 | 1.3793 | 1.3429 | 0.036 | 871,323 | 1,793,298 |
| 3,200,000 to 6,261,576 | 1.3737 | 1.3442 | 0.030 | 1,241,332 | 2,670,558 |

By text, in the order read (second cell): tiny Shakespeare 1.5026; bryant 1.5429; burgess 1.4011; carroll 1.4715; chesterton-thursday 1.4311; chesterton-brown 1.3843; austen-persuasion 1.2648; chesterton-ball 1.3484; austen-emma 1.2262; melville 1.4542; edgeworth-parents 1.3041.

| | Statement | Outcome |
|---|---|---|
| Z1 | after 400,000 bytes the second cell's loss falls with every doubling, save where a new author begins | not shown: it rises on the stretch that holds four new authors and stands still on the last, which holds the hardest of the later books; the stretches mix more reading with other texts and cannot settle it |
| Z2 | from the stretch 400,000 to 800,000 to the last stretch it falls by at least 0.10 | held by the figures (0.106), with the same reservation: the later books are easier ones |
| Z3 | cells grow as the bytes read to a power between 0.55 and 0.75 over the last three doublings | held at the edge: 0.62, 0.49 and 0.53 on the three, 0.55 over them together |
| Z4 | the second cell's gain is larger on the last stretch than on the stretch 400,000 to 800,000 | held, barely (0.030 against 0.028); it is 0.03 to 0.04 from 200,000 bytes on |

Readings (mine).
1. **The store of this system grows as about the square root of what it has read**: 1.24 million cells and 2.7 million counts after 6.3 million bytes, 0.43 counts for each byte read and falling. That is for runs of up to six bytes with nothing else that makes cells; the kinds that multiplied the store on the test text (the pool, classes) are not in it.
2. **What more reading is worth can be read cleanly only where the author stays the same.** Three books of one author read as the fourth, fifth and seventh texts of the life: 1.4311, 1.3843, 1.3484. Two of another, sixth and ninth: 1.2648, 1.2262. So within an author each further book is read about 0.04 nats better than the one before; across the whole life the figures are dominated by which book is being read (1.23 to 1.54).
3. **The second answering cell's gain does not shrink as the system reads on** (0.03 to 0.04 from 200,000 bytes to six million).
4. This is one order of one set of English books. It gives no law of growth, and I draw no figure for larger texts from it; what it shows is that at six times the text the system is still learning, at a store that grows much more slowly than the text.
