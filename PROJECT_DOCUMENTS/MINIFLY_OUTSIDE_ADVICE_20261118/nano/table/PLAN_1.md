# A table made of cells, on the text of the test (roadmap second revision, track L; added step) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Keep experimenting, let the evidence drive your experiments." **Source.** `claude/friendly-darwin-8mmhfm` at 3ad7218f; the measure is `score.py` of the reference line, copied unchanged from `claude/nano-ref-8mmhfm` at 9378f611 (every byte of the last tenth scored, nothing floored). Written before any run. Development level; one text; no checker gate.

## 1. The evidence this follows

The reference line's first figures on the last tenth of the text: a smoothed table of the last six bytes has a loss of 1.5165 nats a byte and names 54.8% of the next bytes; a small GPT taught by back-propagation for 2,000 steps has 1.906; nanoGPT's full setting is described at about 1.47. On a text of this size the test is therefore mostly a test of estimating the next byte from the last six or so. In `loop_small_case` a table was the thing a system of cells was not to be: on sums a table answers nothing it was not shown. On this text the table is the level to reach, so the question changes.

## 2. The question

**Can a system of cells be the table: a cell for each context that has occurred, strengths that are set by what follows, and nothing counted or decided outside the cells; and is it then at the level of the smoothed table?**

*The system.* For each k from 1 to K a cell exists for every run of k bytes that has occurred in the stream (the cell is made the first time the run occurs, with a connection from each of its k lines: a connection that involves k lines jointly); one further cell is always active. At a byte, the active cells are those of the last 1, 2, ..., K bytes, at most K + 1 of them. Each cell has a strength to each of the 65 branches of the answering cell. Two ways of fixing the strengths:

- **Counting.** When a cell is active and a byte arrives, its strength to that byte's branch grows by one. A cell knows its total and how many of its strengths are above zero. The answering cell takes the longest active cell's strengths less a fixed discount d, and gives the mass that the discount frees (d times the number of strengths above zero, over the total) to what the next shorter cell says, and so on down. With `kn=1` a cell that is not the longest uses, in place of its own counts, the number of different longer cells that have reported each byte to it (each longer cell tells its shorter neighbour the first time it meets a byte). All of this is what an interpolated Kneser-Ney table computes, with one discount; nothing in it is new except that it is stated as cells.
- **Mismatch.** The branches of the answering cell compete on the sum of the active cells' strengths (probabilities by exp and a sum to one); when the byte arrives every active cell changes its strengths by its step times the mismatch; a cell's step is eta over the square root of one plus the number of times it has been active. One reading of the stream.

*Cells that withdraw* (`minuse=T`): a cell that has been active fewer than T times over the stream is removed before the test.

The one number chosen (d, or eta) is chosen on the last twentieth of the stream with the cells built on the rest; then the cells are built on the whole stream and scored once on the last tenth.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* The counting cells with K = 6 are within 0.05 nats of the smoothed table (at most 1.567). *Refuted if* they are above 1.60.

| | Statement | Chance |
|---|---|---|
| W1 | counting, K = 6, `kn=1`: at most 1.567 nats | 0.70 |
| W2 | mismatch, K = 6, one reading: more than 0.05 nats worse than counting | 0.60 |
| W3 | `minuse=2` removes more than half of the cells and costs less than 0.03 nats | 0.50 |
| W4 | `kn=0` is at least 0.03 nats worse than `kn=1` | 0.60 |

## 4. Size, cost, what is not run

The number of cells is the number of different runs of 1 to K bytes in the stream; each has 65 strengths, most of them zero; both are reported. A run takes seconds (counting) to a few minutes (mismatch). Not run here: the table's cells together with cells taught by the relay (the next step if this holds); more than one reading for the counting form (it would change nothing); two-digit sums, where a table of this kind answers nothing new.
