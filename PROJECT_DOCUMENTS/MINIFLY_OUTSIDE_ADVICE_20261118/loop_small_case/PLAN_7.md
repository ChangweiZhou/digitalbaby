# Seventh plan: for what does a cell learn to hold a byte? One digit across a gap of letters (roadmap R3) (20 November 2026)

**Authorisation.** As for the sixth plan (Mr. Xu, 20 November 2026: "Keep working, see how much you can push along the road map. Feel free to revise anything you like."). **Source.** This branch at f2990c46 (the sixth plan), with the two texts below added to `hold.py`; the changed file reproduces the earlier runs on both of its paths (single digits without crossing, 60,000 lines; two digits with the crossing and learned return strengths, 4,000 lines: identical readings). Written before any run on these texts. Development level; a small case; no checker gate.

## 1. Why a smaller text

The sixth plan's instrument at X = inf (back-propagation through the line, which is not something a cell has) was the first of its runs to finish: on two-digit sums these cells reach 16% of never-shown sums over the last three readings and 22% at 600,000 lines, still rising; by place 97, 75, 35. With the delay chain the same text was at 80% after 100,000 lines. So two-digit sums are slow for these cells under any credit at this setting, and the text cannot separate one form of credit from another at a cost I can pay here. (The sixth plan's K5 has failed. Its forward runs are reported as they come.)

The reading I gave in the fifth and sixth plans can be tested on a text that is small enough to be learned: **a cell learns to hold a byte, by its own path alone, only where the byte held is useful to the answering cell by itself; where it is useful only together with another byte, the cell that holds it is taught by the cell that reads it, which takes one crossing.**

## 2. The texts

Lines `a+<letters>b=c` with an end mark; a and b from 0 to 9; between the operator and the second number, 0 to 8 letters (each one of four letter values, drawn afresh for every line), the place where the first family has its letters. So `a` must be held across a gap of varying length, and the letters must be kept out.

- `g1`: c = a + b, one or two digits. The first byte of the answer depends on `a` alone (the larger `a`, the more often it is `1`).
- `m1`: c = (a + b) mod 10, one digit. Given `a` alone, every value of c is equally likely: `a` is useful only together with `b`.

80 of the 100 pairs are shown (drawn by the seed as before). The score is the whole answer right on the sums shown, at gaps of 0, 4 and 8 letters (letters fixed by the pair when scoring), since no learner so far answers single-digit sums it was not shown. 18 byte values.

## 3. Arms

256 cells, K = 16, as before. `cross=0` (own path only); `cross=1 ret=exact`; `cross=1 ret=learned`; the instrument at X = inf as a reference of what these cells can learn. eta1 at 1/4 and at 1 for the forward arms, 1/4 for the instrument. 240,000 lines (about 2.4 million bytes), readings every 40,000; mean of the last three readings. Development seed 1 first.

## 4. Hypothesis, what would refute it, predictions (mine, before any run)

*Holds if,* at the gap of 8 letters on `m1`, `cross=1` is at least 30 points above `cross=0`, and on `g1` `cross=0` is at least 20 points above `cross=0` on `m1`.

*Refuted if* on `m1` at the gap of 8 letters `cross=0` is within 10 points of `cross=1` (the crossing adds nothing), or if `cross=0` does as badly on `g1` as on `m1` (own-path credit does not teach holding even where the byte is useful by itself). If `cross=1` stays below 30% on `m1` at that gap while the instrument is at 60% or above, one crossing at the last step is not enough, and that is the finding.

Each figure is for the better of the two values of eta1.

| | Statement | Chance |
|---|---|---|
| M1 | `m1`, gap 8, `cross=0`: at most 40% | 0.65 |
| M2 | `m1`, gap 8, `cross=1 ret=exact`: at least 30 points above `cross=0` | 0.50 |
| M3 | `g1`, gap 8, `cross=0`: at least 20 points above `cross=0` on `m1` | 0.50 |
| M4 | the instrument at X = inf reaches at least 60% on `m1` at gap 8 | 0.60 |
| M5 | `cross=1 ret=learned` is within 15 points of `ret=exact` on `m1` at gap 8 | 0.45 |
| M6 | at gap 0, `cross=0` has at least 70% on both texts | 0.65 |

If the hypothesis holds on the development seed, the forward arms are run on the five reported seeds (73129, 93683, 116089, 406364, 545353) at the eta1 chosen, entered below before those runs.

## 5. Cost, and what is not run

A run is about six CPU-minutes; development 14 runs, about an hour and a half of CPU; five seeds, three arms, two texts: 30 runs, about three CPU-hours. Not run: two-digit numbers with a gap; gaps beyond 8; never-shown sums as a score (no learner here answers them on single digits); more than one crossing in the forward program.
