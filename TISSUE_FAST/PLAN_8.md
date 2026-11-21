# Plan 8: between the two regimes (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "继续做。" **Program:** `TISSUE_EVENT/` at the commit of this plan (unchanged since plan 7).

**Why.** Plan 7: at totals 3/4 activity is driven by the input but thins out (coincidence form with C_E = C_I = 3/4: at level to d = 7, 0.103 at d = 8); at 1.5 and above activity reaches every distance but no longer carries the byte. The values in between were not tried.

**Arms** (both forms; on plan 6's F32 of its form, as in plan 7):

| Arm | C_E, C_I |
|---|---|
| B10 | 1.0, 1.0 |
| B125 | 1.25, 1.25 |
| B075i | 0.75, 1.125 |
| B10i | 1.0, 1.5 |

**H8.** In some arm of some form, S(d,0) in A2 is at least 0.1103 at d = 8 or beyond, with S(1,0) at least 0.1103 and at most 1,536 active cells at every stretch end. **Refuted if** none.

| No. | Prediction | Chance |
|---|---|---|
| J1 | H8 holds | 0.3 |
| J2 | In each form, S(8,0) rises from B075 (plan 7) to B10 | 0.4 |
| J3 | B125 behaves as plan 7's B15 (S(4,0) below 0.2) in both forms | 0.5 |
| J4 | Inhibition above excitation (B075i, B10i) carries farther than the equal arm of the same C_E in some form | 0.4 |

**Arms run:** 4 × 2 = 8 tissues of 16,384 bytes. **Cost:** under 30 minutes on 4 cores. **Not run:** other seeds; all-or-none outputs; STDP.

## Outcome (entered 21 November 2026; eight runs, exit 0; `p8_tables.py`)

### first

| Arm | S(d,0) for d = 1 ... 11 | Active share at reading steps, d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte | H8 met |
|---|---|---|---|---|---|---|---|
| F32 (plan 6) | 1.000 0.971 0.800 0.499 0.250 0.179 0.022 0.022 0.022 0.022 0.022 | 0.17 0.16 0.10 0.05 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 6 | 7.5 | 2 / 190 | 0.60 | no |
| B075 (plan 7) | 1.000 0.949 0.762 0.508 0.121 0.124 0.022 0.022 0.022 0.022 0.022 | 0.17 0.16 0.10 0.04 0.01 0.01 0.00 0.00 0.00 0.00 0.00 | 6 | 7.3 | 10 / 267 | 0.57 | no |
| B10 | 1.000 0.947 0.716 0.524 0.430 0.262 0.202 0.171 0.078 0.071 0.053 | 0.17 0.17 0.15 0.13 0.11 0.08 0.07 0.05 0.06 0.08 0.08 | 8 | 5.8 | 6 / 837 | 1.53 | yes |
| B125 | 0.933 0.661 0.524 0.412 0.283 0.191 0.145 0.119 0.088 0.069 0.054 | 0.36 0.31 0.36 0.40 0.43 0.50 0.59 0.62 0.61 0.62 0.64 | 8 | 3.7 | 150 / 1516 | 4.00 | yes |
| B075i | 0.993 0.927 0.834 0.493 0.271 0.070 0.054 0.022 0.022 0.022 0.022 | 0.17 0.15 0.08 0.04 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 5 | 7.2 | 0 / 296 | 0.58 | no |
| B10i | 1.000 0.939 0.769 0.441 0.264 0.164 0.091 0.069 0.048 0.047 0.043 | 0.20 0.16 0.15 0.13 0.13 0.12 0.14 0.13 0.15 0.12 0.10 | 6 | 5.6 | 10 / 1061 | 1.73 | no |
| B15 (plan 7) | 0.755 0.429 0.263 0.151 0.103 0.081 0.067 0.062 0.059 0.047 0.033 | 0.47 0.39 0.41 0.45 0.45 0.43 0.53 0.47 0.49 0.47 0.54 | 4 | 3.7 | 527 / 1252 | 4.54 | no |

### coincidence

| Arm | S(d,0) for d = 1 ... 11 | Active share at reading steps, d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte | H8 met |
|---|---|---|---|---|---|---|---|
| F32 (plan 6) | 1.000 0.982 0.925 0.618 0.321 0.099 0.031 0.022 0.022 0.022 0.022 | 0.19 0.18 0.14 0.06 0.02 0.01 0.00 0.00 0.00 0.00 0.00 | 5 | 7.1 | 10 / 162 | 1.12 | no |
| B075 (plan 7) | 0.990 0.967 0.886 0.660 0.283 0.149 0.160 0.103 0.022 0.022 0.022 | 0.17 0.17 0.14 0.09 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 7 | 6.9 | 10 / 175 | 1.15 | no |
| B10 | 1.000 0.955 0.702 0.565 0.310 0.223 0.151 0.085 0.078 0.060 0.041 | 0.21 0.22 0.21 0.23 0.25 0.25 0.29 0.31 0.29 0.33 0.36 | 7 | 5.1 | 10 / 492 | 2.48 | no |
| B125 | 0.874 0.528 0.109 0.031 0.031 0.027 0.029 0.024 0.020 0.020 0.022 | 0.26 0.27 0.27 0.22 0.20 0.18 0.19 0.20 0.20 0.18 0.18 | 2 | 4.7 | 38 / 1504 | 2.76 | no |
| B075i | 1.000 0.916 0.858 0.598 0.316 0.133 0.028 0.022 0.022 0.022 0.022 | 0.16 0.16 0.14 0.08 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 6 | 6.8 | 9 / 204 | 1.10 | no |
| B10i | 0.995 0.925 0.704 0.486 0.317 0.214 0.156 0.139 0.092 0.055 0.050 | 0.22 0.22 0.23 0.23 0.25 0.25 0.26 0.26 0.24 0.22 0.24 | 8 | 5.5 | 0 / 1056 | 2.70 | yes |
| B15 (plan 7) | 0.738 0.384 0.105 0.030 0.026 0.026 0.026 0.026 0.029 0.028 0.026 | 0.28 0.25 0.17 0.11 0.11 0.11 0.11 0.11 0.11 0.10 0.10 | 2 | 5.0 | 48 / 1376 | 2.22 | no |


The readings, as in every plan here, are of E63's records, in which an output below 0.01 is stored as 0: a cell whose output stays below 0.01 counts as silent and carries nothing to the score.

| | Outcome |
|---|---|
| H8 | **Held.** First form B10 (C_E = C_I = 1): S(8,0) 0.171, at level to d = 8, with at most 837 cells active at a stretch end; B125 also to d = 8 (0.119) at up to 1,516 active. Coincidence form B10i (C_E 1, C_I 1.5): S(8,0) 0.139, at most 1,056 active |
| J1 | Held |
| J2 | Failed: in the coincidence form S(8,0) is 0.085 at B10 against 0.103 at B075 |
| J3 | Failed: in the first form B125 has S(4,0) 0.412 |
| J4 | Held: coincidence form B10i reaches d = 8 against B10's 7 |

**Reading (development level, one seed).** Between the regime that thins out (totals 3/4) and the one that runs on its own (1.5), totals near 1 with inhibition at least equal to excitation keep activity at every distance (shares 0.05 to 0.36) while it still follows the input: the byte is above level 8 connections from the lines, with traces above chance to 11 (first form B10: 0.078, 0.071, 0.053 at d = 9 to 11). Ten connections at level are not yet reached; one seed only.
