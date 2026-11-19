# Replication of the reviewer's step-1 row on three seeds, with two draws of the time constants (19 November 2026)

**What this checks.** The reviewer's request of 08:04 UTC asked for an independent replication of one row on three seeds. The tissue is the reviewer's folder at b2ab1bb7, copied here unchanged. Its `tissue.c` holds a time constant for each cell, drawn in its `tis.py` by default_rng(seed + 20261119).

- **The row.** `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`, with `tau` set to u2-64, u2-128 or u2-256.
- **The two draws of the time constants.**
  - **R_ / W_:** the reviewer's own draw, as a direct replication.
  - **O_ / WO_:** my own draw, as an independent check. It uses Random(seed + 700000), one draw per cell in cell order, from {2, 4, ..., top} with equal chance. This is option `taudraw=own` in `where_tau.py`, which overwrites the drawn time constants before the tissue runs.
- **Programs.**
  - `where_tau.py` reads one window, as before.
  - `where_win.py` reads several windows of 4,096 bytes; here bytes 0, 20,480, 40,960 and 61,440 of 65,536.
  - `nine.py` and `nine_win.py` check the nine entries of roadmap step 1 per seed: lags 0, 1 and 2 at d = 1, 2 and 4, read by the tool as it is. Each entry is checked against two levels:
    - **level A:** lag 0 above 0.1103, and lag L above C0(L) + 0.02;
    - **level B:** lag L above C<L(L) + 0.02, using C<L of the same window.
- **Outputs.** `where_tau/NINE_16k.txt`, `where_tau/NINE_65k_4windows.txt` and `where_tau/TABLE_*.md`.

## What the runs measured

**Replication at 16,384 bytes (window A2).** My runs give the reviewer's figures to three decimals:

- seed 73129 for u2-64, u2-128 and u2-256;
- seeds 93683 and 116089 for u2-64.

| Setting | Draw | Nine entries above level A, seeds 73129 / 93683 / 116089 | Above level B | Divergence after 512 bytes | Connections a cell |
|---|---|---|---|---|---|
| u2-64 | reviewer | 8 / 9 / 9 (S(1,2) = 0.062 in seed 73129) | 8 / 9 / 9 | 0 | 3.55–3.84 |
| u2-64 | own | 9 / 9 / 9 | 8 / 8 / 8 (S(1,2) 0.090–0.102) | 0; 1.9e-9 in seed 73129 | 3.53–4.23 |
| u2-128 | reviewer | 9 / 9 / 9 | 8 / 9 / 9 (S(1,2) = 0.102 in seed 73129) | 0 | 3.62–4.21 |
| u2-128 | own | 9 / 9 / 9 | 9 / 9 / 9 | 0 | 4.05–4.31 |
| u2-256 | reviewer | 9 / 9 / 9 | 9 / 9 / 9 | 0 | 3.76–4.28 |
| u2-256 | own | 9 / 9 / 9 | 9 / 9 / 9 | 0 | 3.92–4.83 |

- **Far cells (d = 8 to 11).** In every one of these tissues every cell answers, and the largest output is 1.00. The tool reads the present byte at 0.02 to 0.13 there. It is above 0.1103 in one of the 18 tissues: d = 8 of the reviewer's draw of u2-64 on seed 73129, at 0.127.
- **Cells nearest the output lines** (the largest distance with at least 8 cells, d = 13 to 18): 0.02 to 0.06 for lags 0 to 2.

**u2-256 over 65,536 bytes, in four windows.** The reviewer's draw was run on three seeds, and my own draw on seed 73129.

- **The nine entries.** All are above level A in every window of every run. They are above level B in every window, except S(4,2) = 0.098 in the first window (bytes 0 to 4,095) of seed 116089.
- **Last window, reviewer's draw**, as S(d=1 2 4) for lag 0 / lag 1 / lag 2:

  | Seed | Nine entries |
  |---|---|
  | 73129 | 0.981 0.718 0.376 / 0.330 0.409 0.351 / 0.148 0.224 0.245 |
  | 93683 | 0.978 0.702 0.324 / 0.377 0.480 0.377 / 0.199 0.305 0.349 |
  | 116089 | 0.994 0.691 0.304 / 0.350 0.482 0.346 / 0.172 0.275 0.259 |

  My own draw on seed 73129 gives 0.971 0.705 0.214 / 0.337 0.497 0.283 / 0.175 0.258 0.225.
- **Divergence after 512 bytes:** 0 in all four runs.
- **Connections.** Connections a cell rise from 3.5–3.8 in the first window to 3.9–4.6 in the last, with the largest at 14 to 24. Most connections are made and withdrawn before the second window. Cumulative made and withdrawn by the last window:

  | Seed | Made | Withdrawn |
  |---|---|---|
  | 73129 | 10,643 | 18,561 |
  | 93683 | 17,666 | 24,168 |
  | 116089 | 20,470 | 27,158 |
  | 73129, own draw | 17,962 | 25,139 |

- **Far cells, last window.** Every cell answers, and the largest output is 1.00. The tool reads lag 0 at 0.03 to 0.11 and is above 0.1103 nowhere; the largest is 0.109, at d = 8 in seed 116089. Lag 1 reads 0.04 to 0.15 and lag 2 0.05 to 0.16. In the first window (bytes 0 to 4,095) the far cells are at chance (0.02 to 0.04).
- **Nearest the output (d = 14), last window:** 0.02 to 0.09.

## Plainly

- **The nine entries.** On all three seeds, u2-128 and u2-256 put all nine entries of step 1 above their levels by the tool as it is, under both levels. One exception is S(1,2) under level B with the reviewer's draw of u2-128 on seed 73129. Two copies do not diverge.
- **Over the longer life.** Over 65,536 bytes, u2-256 keeps all nine entries above both levels from the second window on, on all three seeds and with both draws.
- **The far cells and the cells near the output.** They all answer, but the tool does not read the present byte there above 0.1103. The one exception is d = 8 in one tissue at 16,384 bytes; in the last window at 65,536 bytes there is none.

## Limits

- One window (16,384 bytes) or four windows (65,536 bytes) per tissue.
- My own draw at 65,536 bytes was run on one seed only.
- The margin of 0.02 is the step-1 brief's margin, not a test.
