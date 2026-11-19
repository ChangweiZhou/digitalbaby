# Time constants of the potential on the tissue that carries the present byte far (19 November 2026)

**What this measures.** The reviewer's task of 19 November (06:24 UTC) asks whether the tissue that carries the present byte far can also carry the bytes before it, once its cells have a spread of time constants.

- **Tissue.** The reviewer's `tissue.c` at 2b0db54b, copied into this folder, plus `tissue_tau.c` and `tis_tau.py`. These add a time constant of the potential for each cell, v ← v + (A − v)/τ_i, drawn at birth from Random(seed + 700000). With every τ = 2 the program computes what tissue.c computes: setting (1) on seed 73129 gives the reviewer's 0.929 / 0.722 / 0.379.
- **Base B.** `rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1`, no gain.
- **Seeds and length.** 73129, 93683 and 116089; 16,384 bytes, then 65,536 for the best setting.
- **Settings.** The six settings of TAU_README.md in `PREP/reviewer_tissue_20261119`: `2`, `q16`, `u2-16`, `u2-64`, `u2-64e`, `u2-256`.
- **Extra runs.**
  - `tau_f=16` for u2-64 and u2-256, which lose the present byte at d = 4.
  - The follow-on run **BG**: base B with the bounded ratio control (`gain=2 rho=1 gmax=4`), for settings `2` and `u2-64e`.
- **Tables.** `where_tau/TABLE_<label>.md` cover every distance and lags 0 to 4, read two ways:
  - **(a)** the readings tool as it is, with its floor of 0.01;
  - **(b)** no floor, with each cell divided by its own standard deviation.

  They also give the pooled whole tissue, the mean output by distance, the mean τ by distance, and the divergence of two copies after 512 bytes. The json files hold connections a cell (`conns_per_cell`) and the largest output by distance.
- **Marks.** `+` means above C<L(lag) + 0.02, where C<L is what every nearer byte gives. Window A2 uses `where/CLT.json`; the last window of 65,536 bytes uses `where/CLT_65536.json`.

## What the runs measured

| Run | S(1, 0) / S(2, 0) / S(4, 0), tool as it is | Present byte at d = 8 to 11, tool as it is | Lag 1 above C<L + 0.02, all seeds | Lag 2 above C<L + 0.02, all seeds |
|---|---|---|---|---|
| B, τ = 2 | 0.93–0.98 / 0.69–0.78 / 0.38–0.48 | 0.13–0.21 at d = 8; 0.03–0.07 at d = 10; chance at d = 11 | nowhere | nowhere |
| B, q16 | 0.93–0.98 / 0.69–0.78 / 0.41–0.53 | ≤ 0.15 at d = 8 | d = 3–11 (b); the tool at d = 4 | in one seed only |
| B, u2-16 | 0.93–0.98 / 0.73–0.75 / 0.42–0.48 | ≤ 0.08 at d = 8 | d = 3–11 (b); the tool at d = 2–4 | d = 9–10 (b) |
| B, u2-64 | 0.93–0.96 / 0.63–0.67 / 0.31–0.43 | ≤ 0.05 at d = 8; chance from d = 9 | d = 1–9 (b); the tool at d = 1–6 | d = 2–9 (b); the tool at d = 2–6 |
| B, u2-64e | 0.95–0.98 / 0.70–0.83 / 0.40–0.48 | ≤ 0.09 at d = 8 | d = 2–10 (b); the tool at d = 2–4 | d = 3–9 (b); the tool at d = 3–4 |
| B, u2-256 | 0.89–0.95 / 0.58–0.67 / 0.22–0.29 | ≤ 0.05 at d = 8; chance from d = 9 | d = 1–8 (b); the tool at d = 1–6 | d = 1–6 (b); the tool at d = 1–5 (lag 3 at d = 1) |
| B, u2-64, tau_f = 16 | 0.92–0.96 / 0.62–0.68 / 0.29–0.36 | ≤ 0.08 at d = 8 | d = 1–9 (b) | d = 2–9 (b) |
| B, u2-256, tau_f = 16 | 0.89–0.95 / 0.59–0.62 / 0.25–0.29 | ≤ 0.06 at d = 8 | d = 1–6 (b) | d = 1–6 (b) |
| B, u2-64e, 65,536 bytes | 0.95–0.97 / 0.70–0.84 / 0.42–0.49 | 0.10–0.12 at d = 8; 0.02–0.05 at d = 9 to 11 | d = 2–10 (b); the tool at d = 2–8 | d = 3–10 (b); the tool at d = 3–4 |
| B, τ = 2, 65,536 bytes | 0.93–0.98 / 0.69–0.79 / 0.37–0.49 | 0.14–0.28 at d = 8; 0.08–0.18 at d = 10–11 | nowhere | nowhere |
| BG, τ = 2 | 0.93–0.97 / 0.62–0.65 / 0.27–0.36 | 0.13–0.23 at d = 8; 0.09–0.17 at d = 10–11 | in one seed only | nowhere |
| BG, u2-64e | 0.93–0.97 / 0.68–0.75 / 0.29–0.43 | 0.10–0.15 at d = 8; 0.04–0.12 at d = 10–11 | d = 2–8 (b and the tool alike) | d = 4 |

- **Divergence and connections.** In every run, two copies differ by 0 after 512 bytes. Base B keeps 4.8 connections a cell. BG keeps 2.9 to 3.0, and its distances reach 14 to 19 connections.
- **What tau_f = 16 changes.** Slower traces of the timing rule change the present byte by at most 0.07 at d = 4, and change lags 1 and 2 little.
- **What the longer life does to base B.** At τ = 2 over 65,536 bytes the tool sees the present byte at the far cells better than over 16,384 (0.14 to 0.28 at d = 8 against 0.13 to 0.21). It still carries no byte before.
- **What BG does to amplitude.** In BG the mean output does not fall with distance: 1.4e-2 to 3.2e-2 from d = 1 to 12. That is why (a) and (b) agree there, and why the tool sees the present byte at the far cells.

## The answer to the question

No setting meets every part on all three seeds.

- **B with u2-64e** comes closest within base B, and holds over 65,536 bytes.
  - The tool as it is sees the present byte at d = 1, 2 and 4 on all three seeds: 0.95 to 0.98, 0.70 to 0.84 and 0.40 to 0.49.
  - The tool also sees lag 1 above C<L + 0.02 at d = 2 to 4 (and to d = 8 over 65,536 bytes) and lag 2 at d = 3 to 4.
  - It does not see the present byte at the far cells: 0.10 to 0.12 at d = 8, and 0.02 to 0.05 at d = 9 to 11.
- **BG with u2-64e** (the follow-on run) adds the far cells:
  - present byte 0.93 to 0.97 / 0.68 to 0.75 / 0.29 to 0.43 at d = 1 / 2 / 4;
  - 0.10 to 0.15 at d = 8 and 0.04 to 0.12 at d = 10 and 11, by the tool as it is;
  - lag 1 above C<L + 0.02 by the tool at d = 2 to 8 on all three seeds;
  - lag 2 above C<L + 0.02 at d = 4 on all three seeds.

  Its weak point is seed 93683, where the present byte at d = 10 to 11 is 0.040 (chance 0.022). Its full table is `where_tau/TABLE_BG_u2-64e.md`.

## Limits

- One window of 4,096 bytes per tissue, 16,384 bytes long (65,536 for B with u2-64e).
- Rule timing with no_make, so connections are only lost, never made.
- The margin of 0.02 is the step-1 brief's margin, not a test.
- "The tool sees the present byte" here means a score above chance; the brief's rule of success is not applied.
