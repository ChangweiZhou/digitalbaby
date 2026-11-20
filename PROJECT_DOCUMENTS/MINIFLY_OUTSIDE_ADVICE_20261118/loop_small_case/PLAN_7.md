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

## 6. Entered on 20 November after the development runs of section 3 (seed 1), before the runs named here

Sums shown, mean of the last three readings over the three gaps; at the end by gap (0, 4, 8 letters); `res/hold_p7_*.json`.

| | `m1`, eta1 1/4 | `m1`, eta1 1 | `g1`, eta1 1/4 | `g1`, eta1 1 |
|---|---|---|---|---|
| `cross=0` | 13.2 (14, 19, 15) | 10.7 (12, 15, 5) | 12.5 (12, 14, 12) | 18.2 (15, 16, 16) |
| `cross=1 ret=exact` | 36.2 (40, 42, 39) | 49.3 (51, 52, 52) | 52.1 (59, 55, 55) | 78.2 (79, 81, 81) |
| `cross=1 ret=learned` | 12.6 (12, 11, 12) | 12.1 (11, 12, 10) | 26.0 (25, 25, 24) | running |
| instrument, X = inf, eta1 1/4 | 12.1 (11, 15, 14), with an overflow | | running | |

On this seed: M1 holds (15 at most). M2 holds (52 against 15 at the gap of 8). **M3 fails, and with it the first half of my reading:** without the crossing `g1` is learned no better than `m1`; own-path credit did not teach holding across letters even where the byte held is useful by itself. What the crossing gives is the same at every length of gap. M4 fails as run (the instrument overflowed at this step). M5 fails (12 against 52). M6 fails (12 to 15 at gap 0).

**Added now, each on the development seed, eta1 = 1, 240,000 lines unless said** (none foreseen in sections 3 and 4; no registered prediction):

- the instrument at X = inf with smaller steps (1/16 and 1/64), since at 1/4 it overflowed and is no reference;
- `cross=1 ret=exact` with eta1 = 4, and with 480,000 lines, since eta1 = 1 was the better of two and still rising;
- why `ret=learned` fails: `ret=exact` with the same fading of the strengths between cells (1e-5 a byte), to see whether the fading by itself is the cause; `ret=learned` with a tenth of the fading;
- two further forms of return strength that a cell could have: `ret=grown` (a separate return strength that starts at nothing and receives the changes of the forward strength; no fading) and `ret=sign` (the sign of the forward strength and one fixed size, as in the record's sweep).

The five reported seeds are run for `cross=0`, `cross=1 ret=exact` and the best of the other forms of return strength, on both texts, at eta1 = 1 (or the value entered here if these runs give a better one).

## 7. A correction of mine about the return strengths, and one more form (entered on 20 November before the runs of `ret=local` were read)

In the sixth plan I called `ret=learned` the candidate and `ret=exact` its reference, on the model of the answering cell's return connections. On working through who holds what, that was wrong as built:

- In `ret=exact` the cell that reads forms the returned value from its own quantities only: what it received, its two sensitivities, and its own strengths from the cell it reads. No strength of another cell is used. What is assumed is something else: that a cell can send back, to each cell it reads, a value that differs from connection to connection.
- `ret=learned` and `ret=grown`, as built, keep the separate return strength at the reading cell and feed it that cell's own changes. They assume the same and remove nothing; they only show how exact the return strength has to be.
- The form that corresponds to the answering cell's return connections is this, added now as **`ret=local`**: every reader sends the same two numbers to all the cells it reads (what it received times each of its two sensitivities); the return strengths are kept by the cell that is read, start at nothing, and change by those two numbers times that cell's own output one byte earlier. Nothing of the reader's strengths or traces is used, and no value differs from connection to connection.

Development seed, eta1 = 1, 240,000 lines, the follow-up runs so far: `ret=grown` 70.4 on `m1` (88, 86, 81 by gap at the end) and 49.3 on `g1` (36, 49, 48); `ret=sign` 11.7 and 17.6 (nothing learned); `ret=learned` on `g1` 44.9. The five reported seeds are run for `cross=0`, `cross=1 ret=exact` and, if it learns on the development seed, `cross=1 ret=local`; if it does not, `ret=grown` in its place.

## 8. Entered on 20 November after the follow-up runs (development seed 1), before any reported seed

From here on the runs use the compiled kernel (`fast=1`; `PLAN_8.md`, section 1). `res/hold_p7b_*`, `p7c_*`, `p7d_*`, `p7e_*`.

| `m1`, eta1 = 1 | 240,000 lines | 960,000 lines (last three readings; at the end by gap 0, 4, 8) |
|---|---|---|
| `cross=0` | 10.7 | 22.6 (21, 28, 32) |
| `cross=1 ret=exact` | 49.3 | **98.1** (100, 100, 100) |
| `cross=1 ret=grown` | 70.4 | **99.6** (99, 100, 100) |
| `cross=1 ret=local` | 11.8 | 10.6 (12, 8, 8) |
| `cross=1 ret=sign` | 11.7 | not run |
| `cross=1 ret=exact`, the strengths between cells fading at 1e-5 a byte | 12.4 | not run |
| `cross=1 ret=learned`, fading 1e-6 | 33.1 | not run |
| `cross=1 ret=exact`, eta1 = 4 | 59.6 | not run |
| instrument, eta1 = 1: X = 1L / 1 / 2 / inf | 35.7 / 36.7 / 86.9 / 79.3 | not run |
| instrument X = inf, eta1 = 1/64, 1/16, 1/4 | 9.4, 12.1, 12.1 | not run |

`g1`, eta1 = 1: `cross=1 ret=exact` 78.2 at 240,000 lines and 95.3 at 960,000 (98, 99, 99); `ret=local` 18.3 at 240,000; the other long runs are entered in the report.

Readings (mine). (1) The crossing teaches holding across the gap, and the answer is as good after 8 letters as after none. (2) `ret=learned` failed because the fading it applies to the strengths between cells stops learning even with exact return strengths. (3) `ret=local` does not learn: a reader that takes in the second number holds it for one byte before the answer is named, so at the moment of the mismatch its sensitivity is near zero, and a return strength built from the product at that moment alone gets nothing; `ret=grown`, which uses the reader's trace, works. (4) With the same step the backward-pass instrument learns this text too, faster with two or more crossings (87% and 79% at 240,000 lines against 36% with one); my earlier instrument runs had steps too small for the length of run. M4 is therefore met at eta1 = 1 (79 to 87) and was not at the steps the plan named.

**Setting for the five reported seeds** (73129, 93683, 116089, 406364, 545353): both texts; `cross=0`, `cross=1 ret=exact`, `cross=1 ret=grown`; eta1 = 1; **960,000 lines** (the plan's 240,000 was too short for any arm to finish learning; amended here before those runs); the strengths saved for `look_hold.py`.

## 9. A fairness run for the arm without the crossing, and what it changes (entered 20 November, before the reported seeds of that arm at the larger step were run)

The arm without the crossing had been run at 960,000 lines only with eta1 = 1. On the development seed with other steps (`res/hold_p7f_*`): `m1`: 14.6 (eta1 = 1/4), 22.6 (eta1 = 1), **47.4** (eta1 = 4; 54 at the last reading, still rising; 51, 54, 58 by gap); `g1`: 16.4, 29.2, 27.2. So a cell's own path alone does teach some holding when the step is large, and "nothing" was too strong a word for it in sections 6 and 8; the crossing arms are at 98 to 100 on the same seed. The five reported seeds are therefore run for `cross=0` at eta1 = 4 as well as at eta1 = 1, and the report gives both.
