# Ninth plan: a chain of two, and cells that hold (roadmap R3 on the two-digit text) (20 November 2026)

**Authorisation.** As for the sixth to eighth plans (Mr. Xu, 20 November 2026). **Source.** This branch at a42c64f9, with `chain=C` added to `hold.py` (numpy) and a separate kernel for it (`kern4.c`); the earlier paths give the same readings as before. Written before any scored run with a chain. Development level; a small case; no checker gate.

## 1. Where the two-digit text stands without the chain

With the compiled kernel the runs of the sixth plan were lengthened to 2,400,000 lines (development seed, eta1 = 1): no crossing 1.1% of never-shown sums (by place 79, 20, 9); one crossing with the reader's own strengths 5.0% (93, 38, 10), flat from the first reading on. The backward-pass instrument with unlimited crossings had 22% at 600,000 lines and was rising (eta1 = 1/4; not run longer). On the texts with a gap the same crossing learns everything (seventh plan).

*My reading of the difference.* On the gap texts a cell has to take in "the first byte of the line" and "the byte before `=`", each marked by something it can see when it happens. For the units digit of the first two-digit number it has to take in the *second* byte after the end mark, and nothing on its lines says "second": a cell that tells the second byte from the first would have to be taught by the holding cell that needs it, which is itself taught by a reader. That is two crossings; the forward program has one.

## 2. The question, the hypothesis, what would refute it

**If the two bytes before the present one are on lines as well (a chain of two delay cells where the earlier plans had nine), do cells that hold, taught over one crossing, learn two-digit sums?** With two delay lines both digits of a number are on a cell's lines when the byte after it arrives (the operator, or `=`), so a cell can take in on an event it sees, and what it holds is taught by its reader over one crossing. Three of the four digits still have to be held by cells (up to five bytes), which the full chain of nine never required.

*Hypothesis.* Yes: at least 50% of never-shown sums within 2,400,000 lines on the development seed.

*Refuted if* the system with the crossing stays at or below 15%.

Arms (256 cells, K = 16, eta1 = 1, development seed 1): `chain=2` with `cross=0`, with `cross=1 ret=exact`, with `cross=1 ret=grown`; and `chain=1 cross=1 ret=exact`. Score: whole answer right on 500 never-shown and 500 shown pairs, mean of the last three readings at 200,000-line intervals.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| N1 | `chain=2 cross=1 ret=exact` reaches at least 50% of never-shown sums | 0.50 |
| N2 | `chain=2 cross=0` stays at or below 20% | 0.60 |
| N3 | `chain=2 cross=1 ret=grown` reaches at least 50% | 0.45 |
| N4 | `chain=1 cross=1 ret=exact` stays below `chain=2` by at least 20 points | 0.50 |
| N5 | wherever at least 30% of shown sums are right, never-shown sums are within 10 points of them | 0.80 |

If N1 or N3 holds, the arms are run on the five reported seeds.

## 4. Cost, and what is not run

About nine CPU-minutes a run with the kernel. Not run here: letters between the numbers on this text (the next step if this holds); the backward-pass instrument with a chain; more than one crossing forward.

## 5. Entered on 20 November after the development runs (seed 1), before any reported seed

Never-shown sums, mean of the last three readings of 2,400,000 lines; sums shown; by place at the end (`res/hold_d9_*.json`):

| | never shown | shown | hundreds, tens, units |
|---|---|---|---|
| no chain, no crossing (sixth plan, lengthened) | 1.1 | 1.7 | 79, 20, 9 |
| no chain, one crossing | 5.0 | 5.1 | 93, 38, 10 |
| `chain=2`, no crossing | 35.0 | 41.7 | 94, 46, 94 |
| `chain=2`, one crossing, `ret=exact` | **72.4** | 74.4 | 98, 75, 100 |
| `chain=1`, one crossing, `ret=exact` | 63.6 (71 at the last reading, rising) | 68.5 | 99, 78, 93 |

N1 holds. N2 fails: with two delay lines the cells learn the units digit by their own path alone (94), and the tens digit is what the crossing adds (46 to 75). N4 fails: one delay line gives nearly as much (9 points less, and still rising). N5 holds. `ret=grown` is entered in the report.

**Reported seeds** (73129, 93683, 116089, 406364, 545353): `chain=2 cross=0`, `chain=2 cross=1 ret=exact`, `chain=1 cross=1 ret=exact`; eta1 = 1; **1,600,000 lines** (the development run with two delay lines was level from about 1,000,000 lines on; shortened here, before those runs, for cost), readings every 200,000.
