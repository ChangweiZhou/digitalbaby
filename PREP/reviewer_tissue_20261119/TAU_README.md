# A spread of time constants of the potential: how many bytes back the tissue carries (19 November 2026)

**What this measures.** The reviewer's task of 19 November (06:12 UTC) asks how many bytes back the tissue carries when its cells have different time constants. The tissue is the reviewer's `tissue.c` with one change, made in `tissue_tau.c`: a time constant of the potential for each cell, v ← v + (A − v)/τ_i.

- **Setting the time constants.** Each τ_i is drawn at birth by `tis_tau.py`, from Random(seed + 700000), one draw per cell in cell order. This is separate from the tissue's connection generator.
- **τ = 2.** The definition's expression v/2 + A/2 is kept, so with every τ = 2 the program reproduces where.py's earlier results exactly (`M_2_73129` against `where/M_73129.json`: every S equal).
- **Bases.**
  - **M**: rule fixed, gain=2 rho=1 gmax=4.
  - **U**: rule fixed, no control.
- **Seeds and window.** 73129, 93683 and 116089, over 16,384 bytes, read in window A2.
- **Settings of τ.**
  1. `2`: every cell τ = 2, the present definition.
  2. `q16`: τ = 16 for a random quarter of the cells (Random.sample of 512 of the 2,048), 2 for the rest.
  3. `u2-16`: τ from {2, 4, 8, 16}, with equal chance.
  4. `u2-64`: τ from {2, 4, ..., 64}, with equal chance.
  5. `u2-64e`: as (4), but a cell whose neighbourhood holds an input line keeps τ = 2.
  6. `u2-256`: τ from {2, 4, ..., 256}. This is the follow-on run, made because (4) and (5) reach lag 3 in only one or two seeds.

**Programs and records.**

- **Program.** `where_tau.py` is where.py on `tissue_tau.c`, with the divergence of two copies after 512 bytes, measured as in diag.py.
- **Tables.** `where_tau_table.py` writes `where_tau/TABLE_<base>_<setting>.md`. Each table gives every distance and lags 0 to 4, read two ways:
  - **(a)** the readings tool as it is;
  - **(b)** no floor, with each cell divided by its own standard deviation.

  It also gives the pooled whole tissue, the mean output by distance, the mean τ by distance, and the divergence. One table was made for each base and setting.
- **Marks.**
  - `*` means above C0(lag) + 0.02, where C0 is what the present byte alone gives.
  - `+` means above C<L(lag) + 0.02, where C<L is what every nearer byte gives (`where/CLT.json`).
  - For lag 1 the two levels are the same. For lags 3 and 4, C<L is 0.18 to 0.19: the bytes three and four back follow largely from the nearer bytes.

## Summary: how many bytes back, where, on all three seeds

All readings below are under (b), against C<L(lag) + 0.02.

- **Every cell τ = 2.** The tissue carries one byte back, and only far out:
  - lag 1 at d = 6 to 10 on M and d = 9 to 11 on U, in all three seeds;
  - the present byte above 0.5 out to d = 5 or 6 (M) and d = 9 (U);
  - S(4, 0) 0.78 to 0.96.
- **A spread of time constants** brings the earlier bytes nearer the input lines and adds a second byte back on all three seeds.
  - **q16 and u2-16:**
    - lag 1 at d = 2 to 8 (2 to 9 on U with q16);
    - lag 2 at d = 5, 6, 8 (q16, M), 4 to 8 (u2-16, M), 8 to 9 (q16, U) and 5 to 9 (u2-16, U).
  - **u2-64:**
    - lag 1 at d = 1 to 4 (M) and 1 to 6 (U);
    - lag 2 at d = 2 to 4 (M) and 2 to 6 (U);
    - lag 3 in one or two seeds only.
  - **u2-256**, the only setting that reaches three bytes back on all three seeds:
    - lag 1 at d = 1 to 4 (M) and 1 to 5 (U);
    - lag 2 at d = 1 to 4 (M) and 1 to 5 (U);
    - lag 3 at d = 1 and 2, on both bases, at 0.21 to 0.29 against C<3 + 0.02 = 0.20 to 0.21;
    - lag 4 at d = 2 on U only, at 0.20 to 0.24 against C<4 + 0.02 = 0.20.
  - The pooled whole tissue carries lags 1 and 2 above C<L in every setting with a spread, on all three seeds, but lag 3 in at most one seed.
- **What the present byte loses.** The longer the time constants, the less of the present byte survives at the middle distances.
  - S(1, 0) stays at 0.96 to 1.00 in every setting.
  - S(2, 0) falls from 0.87 to 0.97 (τ = 2) to 0.71 to 0.78 (u2-256).
  - S(4, 0) falls:

    | Base | τ = 2 | q16 | u2-16 | u2-64e | u2-64 | u2-256 |
    |---|---|---|---|---|---|---|
    | M | 0.78–0.84 | 0.47–0.59 | 0.38–0.41 | 0.29–0.42 | 0.17–0.25 | 0.12–0.16 |
    | U | 0.87–0.96 | 0.74–0.79 | 0.59–0.69 | 0.50–0.59 | 0.33–0.39 | 0.18–0.29 |

  - The far cells (d ≥ 7) lose the present byte almost entirely once τ reaches 64: S(d, 0) below 0.04 on M (u2-64).
- **Keeping τ = 2 near the input lines** (u2-64e) raises the present byte against u2-64, but not to its level with τ = 2: S(2, 0) is 0.88 to 0.95 against 0.81 to 0.89, and S(4, 0) 0.29 to 0.42 (M) and 0.50 to 0.59 (U) against 0.17 to 0.25 and 0.33 to 0.39. It moves lags 1 and 2 about one distance further from the lines: lag 1 at d = 2 to 5 (M) and 2 to 7 (U), and lag 2 at d = 3 to 4 (M) and 3 to 7 (U), on all three seeds.
- **Stability.** No setting makes the tissue depend on its past: the largest output difference of two copies after 512 bytes is 0 in every run, and at most 5e-10 after 1 to 64 bytes.

## Not explained here

On M with u2-64, seed 93683, the cells at d = 6 to 10 have a mean output of 5e-3, with 11 to 18% of their outputs above 0.01. Yet they score at or below chance for every lag (0.014 to 0.016). The other two seeds do not show this. The same seed carries lags 3 and 4 above C<L at d = 4 to 8 on U with u2-64 and u2-64e, which the other seeds do not.

## Limits

- One window of 4,096 bytes per tissue, 16,384 bytes long.
- Rule fixed only: no rule of strength change.
- The time constants are drawn independently of where a cell is, except in u2-64e.
- The margin of 0.02 is the step-1 brief's margin, not a test.
