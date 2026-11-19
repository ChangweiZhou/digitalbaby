# Addition as the text: the smallest system that adds sums it was never shown (19 November 2026)

This is the reviewer's task of 15:13 UTC to E57's worker. The text and the score are those of the reviewer's `yesno/addtext.py` (`claude/vibrant-cerf-sje726` at d6808869f, copied here as `ref_addtext.py`):

- lines "a+b=s\n", with a and b from 0 to 9;
- 20 of the 100 pairs kept out of learning;
- 60,000 lines of learning, with the next byte named and answered yes or no.

**Score**: the whole answer is right when every byte after '=' up to the end mark is right. It is read for sums shown (2,000 lines, no learning) and for sums never shown (2,000 lines, no learning).

**Runs**: seeds 1–3 (which pairs are held out) × orders 0–4. Order 0 is addtext.py's own stream. Every table is in `TABLES.md`, written by `tables.py` from `res/*.json`. Each run is one run; nothing is confirmed independently.

**Code**: `addsys.py` is my own. Its output units learn by rule Q (THE_TISSUE_AS_RUN section 3b: eta 1/16, z1, exploration 1/16). With the system `byte,trace` it reproduces addtext.py's figures exactly, checked at 6,000 lines.

## The cells

- **byte**: a cell for each byte value, firing when that byte is the present byte.
- **trace**: the reviewer's slow copies s_2, s_4, s_8 of the byte cells.
- **pair:D1-D2**: a cell for each pair of byte values at delays D1 and D2. It fires on the two together (a conjunction).
- **mag:D1-D2:G**:
  - **magnitude cell**: takes the byte itself as a graded input over two connections of strength 1, with delays D1 and D2, so its activity is the sum of the two byte values.
  - **value-tuned cells** (`tuned=1`): each fires only when the magnitude equals its value T. It is excited by threshold cell T and inhibited by threshold cell T + 1.
  - **gate G**: each tuned cell also needs the present byte (G = 1), or the present byte and the byte 1 back (G = 2).
  - **magall:K**: every such pair 1 ≤ D1 < D2 ≤ K, with both gates.
- Only the output units learn.

## Attempts: whole answer right, %, smallest / median / largest over 15 runs

| System | cells that fired | sums shown | sums never shown | what it is |
|---|---|---|---|---|
| A0 byte + trace (addtext.py's system) | 51 | 8.6 / 11.1 / 12.3 | 4.2 / 9.8 / 16.6 | the form of a line only |
| A1 byte + pair:1-3 + pair:2-4 | 233 | 61.6 / 71.5 / 80.2 | 0.0 / 9.6 / 16.0 | **a table of the shown pairs** |
| A5 byte + mag:1-3, mag:2-4, tuned, gate 1 | 356 | 87.2 / 95.0 / 97.9 | 68.8 / 85.7 / 94.8 | misses sums 13–17, second digit (see below) |
| A6 as A5 with gate 2 on both | 770 | 100 / 100 / 100 | 38.4 / 45.2 / 60.4 | **a table again**: at '=' the byte 1 back is the operand b |
| **A7** byte + mag:1-3:1 + mag:2-4:2, tuned | 601 | 98.8 / 100 / 100 | **94.3 / 100 / 100** | adds; a table of the 19 sums |
| A8 as A7 with both gates on both | 1,114 | 100 / 100 / 100 | 94.3 / 100 / 100 | the same as A7: no gate chosen |
| A9 magall:4 (delays not chosen) | 4,206 | 97.2 / 98.8 / 100 | 75.8 / 94.8 / 95.7 | adds, less surely |
| A10 magall:6 | 12,626 | 94.0 / 97.5 / 100 | 74.8 / 85.1 / 95.7 | adds, less surely |
| A11 A7 with strengths 1 ± 0.1 | 583 | 74.8 / 82.9 / 86.0 | 58.6 / 74.1 / 81.7 | equal strengths matter |
| A12 A7 with strengths 1 ± 0.01 | 601 | 98.8 / 100 / 100 | 94.3 / 100 / 100 | as A7 |

The single runs of A2–A5 on seed 1, order 0, with and without tuning and gating, are also in `TABLES.md`.

## What the rows say

1. **A7 adds.** In all 15 runs, every never-shown pair whose sum occurs among the shown pairs is right 20 times out of 20. The only miss is 9+9 on seed 1, whose sum 18 occurs in no shown pair. It does not hold a table of pairs; it holds a table of the 19 sums, in value-tuned cells gated by the present byte. The adding itself is done by the two magnitude cells.
2. **What makes it addition**: a cell that receives the byte as a graded value from two delays over connections of equal strength.
   - It works only because the digit bytes are consecutive codes, so the byte value is the digit plus 48.
   - With strengths 10% apart (A11), the sum and the pair are no longer one-to-one, and the score falls to a median of 74.1 on never-shown sums.
3. **Conjunctions over bytes alone are tables** (A1, A6). Without the present-byte gate the sum cells are confounded across positions in the line (A4: 61.6 / 52.3 on seed 1).
   - A5's misses come from such a confound: with '1' present, the magnitude 2 + 4 back has the same values when '1' begins a line (units digit + '='). A7's second gate removes it.
4. **The gate and the delays need not be chosen.** With both gates on the two magnitude cells, A8 is as good as A7: the table cells for unseen pairs never fired, so they carry no weight. With every delay pair up to 4 or 6 (A9, A10), never-shown sums fall to medians of 94.8 and 85.1. Cells that depend on one operand take part of the credit, which hurts sums with few shown pairs (0+1, 1+0, 7+9, 8+8).
5. **What a cell needs** in A7:
   - a graded byte input;
   - a sum over two connections with delays and equal strength;
   - a threshold with inhibition from the next threshold (tuning);
   - a conjunction with the present byte;
   - output units with rule Q.

   Nothing but the output units learns. Whether the equal strengths, the delays and the tuning can themselves be learned is not tested.

## Cost

Each run took 5 to 60 s of one thread, with at most three processes.

# Second task: equal strengths learned, a pool with withdrawal, subtraction (reviewer's message of 15:42 UTC)

**Stopped** by the reviewer at 16:14 UTC on Mr. Xu's word: the work is finished for now. All rows are in `TABLES2.md`, written by `tables2.py` from `res/*.json`. Each run is one run; nothing is confirmed independently.

**Program**: `addpool.py` is my own. It has a pool of magnitude cells that can learn its strengths and withdraw and remake cells. With A7's two cells it reproduces A7 bit for bit.

Whole answer right, %: smallest / median / largest over 15 runs (seeds 1–3 × orders 0–4).

## Item 1: the equal strengths learned by a local rule

**The rule** (`eq=drive`):

- Each connection keeps a running mean of its drive: r_i += (w_i x_i − r_i) / 4096.
- After each byte: w_i *= 1 + (1/1024)(r̄ / r_i − 1).
- The cell then scales w1 + w2 back to its total.

| Run | strengths at the start | rule | shown | never shown | largest \|w1 − w2\| / mean w (median) |
|---|---|---|---|---|---|
| B1 | 0.5–1.5 | none | 29.3 / 59.7 / 92.2 | 10.2 / 35.6 / 80.3 | 0.41 |
| B2 | 0.5–1.5 | eq=drive, not at the end mark (my error) | 73.6 / 92.6 / 100 | 61.5 / 84.4 / 100 | 0.033 |
| B3 | as B2, tau 32,768 | the same error | 73.6 / 93.4 / 100 | 61.5 / 81.3 / 100 | 0.033 |
| B4 | 0.5–1.5 | eq=drive at every byte | 73.6 / 92.5 / 100 | 61.5 / 86.1 / 100 | 0.0000 |
| B5 | 0.5–1.5, total 2 | eq=drive | 48.4 / 52.5 / 57.8 | 29.5 / 40.6 / 61.8 | 0.0000 |
| **B7** | 0.5–1.5, total 2, tuned bands centred | eq=drive | **100 / 100 / 100** | **94.3 / 100 / 100** | 0.0000 |
| B8 | as B7 | none | 34.0 / 74.0 / 100 | 10.2 / 50.4 / 100 | 0.48 |

- **B2 and B3 carry my error.** The rule did not run at the end mark, so the two delays saw different byte mixes, and the strengths stayed 1–3% apart.
- **B4**: the strengths become equal, but each cell keeps its random total. A total below 2 merges neighbouring sums into one tuned cell.
- **B5**: with strengths of almost exactly 1, the magnitude lies on the integer boundary, and floor() flips it between two tuned cells.
- **B7** centres each tuned band on its value (round). It is as good as A7: every held pair with a shown sum is right 20/20 in all 15 runs, starting from random strengths.
- **What the cell needs**:
  - a running mean of the drive at each of its two connections;
  - a multiplicative change toward equal mean drive;
  - a fixed total (2, chosen by me so that one tuned cell takes one unit of value);
  - tuned bands centred on their value.

  It uses no yes or no. It works because the two delayed inputs carry the same bytes in the long run.

## Item 2: a pool with withdrawal

**The pool**: 300 magnitude cells, delays drawn from 1 to 16 with gate 1 or 2 (240 kinds), strengths 1.

**Withdrawal**: every 1,000 lines, the 1/8 of cells older than 2,000 lines with the lowest keep score are withdrawn and remade with new random delays and gate (1,952 withdrawn in a run).

| Run | keep score | shown | never shown |
|---|---|---|---|
| C300n | no withdrawal | 24.9 / 47.8 / 75.0 | 0.0 / 22.5 / 43.2 |
| C300w | summed \|weight\| to the output units | 23.6 / 43.6 / 64.6 | 0.0 / 9.8 / 45.6 |
| C300wb | that × mean firings per tuned cell | 23.6 / 54.8 / 71.6 | 0.0 / 25.4 / 50.4 |
| C300wf | Σ over tuned cells of \|weight\| × own firings | 23.9 / 49.0 / 68.5 | 0.0 / 25.4 / 40.8 |

- **No rule finds the adding cells.** At the end the strongest kinds are table-like cells joining the byte 3 back with another (3-12-2, 3-4-2, 2-3-2, 1-3-2). 1-3-1 is present in some pools, with less weight.
- **Even shown sums are learned poorly.** With 300 cells active at every byte, rule Q's change per connection is eta/n with n ≈ 300, about 100 times slower than in A7 (n = 3). 60,000 lines is short for that. This is my explanation; it is not tested.
- **NOT_RUN**: pools of 100 and 1,000 cells.

## Item 3: subtraction beside addition

**NOT_RUN.** `addpool2.py` (gates on any byte positions, and an inhibitory second connection for a difference cell) was written and checked equal to `addpool.py` on A7 at 6,000 lines. No run of item 3 was made.

## Cost of the second task

Pool runs took 104–150 CPU-seconds each, fixed-pool runs 9–24 s. At most three processes of one thread each.
