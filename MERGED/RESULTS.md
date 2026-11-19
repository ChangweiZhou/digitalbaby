# One tissue with the four parts, and its life (65,536 bytes, seeds 73129, 93683, 116089)

**The program.** `tissue.c` and `tis.py` in this folder start from the reviewer's at 75ea0812, which already holds part (a). Three parts are added as options, each off by default:

| Part | Option | Source |
|---|---|---|
| (b) making by a long mean | `mk_mean`, `mk_tau`, `mk_share`, `mk_take` | E52's worker, `PREP/tissue5/tissue_mk.c` at 9ff7d03e |
| (c) a time constant of the potential for each cell | `tau=u2-64e` and the other draws | the helper, `PREP/reviewer_tissue_2b0db54b/tissue_tau.c` and `tis_tau.py` at eb21cee5 |
| (d) a delay of its own for every connection between cells | `dmax=N` (prev2.py's `delayrand` draw) | E57's worker, `TISSUE_PREV2/tissue.c` and `prev2.py` at d14a3558 |

E57's `TISSUE_DELAY` was not on its branch when this was written.

**One choice is new here.** A connection made later takes the delay of its (cell, neighbourhood member) pair. Every pair's delay is drawn at birth by numpy `default_rng(20261120)`, and a starting connection's pair keeps the delay prev2.py gives it.

The authors' folders are copied unchanged under `src/`.

## Checks: each part alone, step for step against its author's own program (`equiv.py`, logs `equiv_*.log`)

Each check runs seed 73129 for 16,384 bytes. Both programs run the same options from birth, and the outputs of every cell after each of the last 512 bytes are compared, together with the connections at the end.

| Part | Options | Largest difference of outputs | Connections | Made / given up (both programs) |
|---|---|---|---|---|
| base (a) | `rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 we=1 wi=1/4` | 0 | identical | 0 / 9,777 |
| (b) | `rule=timing norm=1 lines_learn=1 eta_s=8 mk_mean=1 a_make=1/4096 start_w=1/8 make_period=1024 by_use=0` | 0 | identical (4.02 a cell over 2,048) | 17,206 / 24,849 |
| (c) | `rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 tau=u2-64e` | 0 | identical | 0 / 5,028 |
| (d) | `rule=fixed gain=2 rho=1 gmax=4 dmax=32` | 0 | identical | 0 / 0 |

The base row equals the reviewer's figures for B1 on three seeds, which are reproduced in `claude/e63-norm-timing` at af080bae.

## The tissues

The options shared by all four tissues are: `rule=timing norm=1 lines_learn=1 eta_s=1 we=1 wi=1/4 mk_mean=1 a_make=1/4096 start_w=1/8 make_period=1024 by_use=0`.

| Tissue | Parts | Added options |
|---|---|---|
| T1 | (a) + (b) | none |
| T2 | (a) + (b) + (c) | `tau=u2-64e` |
| T3 | (a) + (b) + (d) | `dmax=16` |
| T4 | (a) + (b) + (c) + (d) | `tau=u2-64e dmax=16` |

Each life runs 65,536 bytes and is read in four windows: the last 4,096 bytes before 16,384, 32,768, 49,152 and 65,536. The readings come from `ntd.py`.

- **Every distance:** the readings tool as it is and with no floor, at lags 0 to 4.
- **The level of lag L** is C0(L) + 0.02 in that window.
- **Divergence:** a memory copy of the tissue 4,608 bytes before the end of the window, with 1e-9 added to one cell's potential.
- **Cells near the output lines:** E55's worker's `reach.py` at 5ddfbaad, unchanged. Its reader predicts the next byte from 495 to 519 cells: the 256 drivers of the output lines and their neighbourhoods, at 8 to 17 connections from the input lines. It is fitted on positions 0 to 49,151 and scored on 49,152 to 65,535.

The full table of every window is in `T_TABLE.md`, made by `ttable.py`. The reader's table is below, made by `rtable.py`.

### Last window (bytes 61,440 to 65,535)

| T | seed | window end | S(d,0) tool d = 1 / 2 / 4 | lag 1 above level: distances (best S) | lag 2 above level: distances (best S) | largest output d = 8 / 11 | connections a cell (largest) | made / given up | cells differing after 64 / 512 bytes | ms a byte |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 73129 | 65,536 | 0.949 / 0.645 / 0.374 | none | none | 0.57 / 0.49 | 3.11 (15) | 253 / 9,945 | 0 / 0 | 5.4 |
| T1 | 93683 | 65,536 | 0.916 / 0.623 / 0.421 | none | none | 0.53 / 0.46 | 3.35 (12) | 802 / 10,036 | 0 / 0 | 5.8 |
| T1 | 116089 | 65,536 | 0.932 / 0.680 / 0.289 | none | none | 0.54 / 0.34 | 3.46 (17) | 445 / 9,443 | 0 / 0 | 6.2 |
| T2 | 73129 | 65,536 | 0.933 / 0.703 / 0.383 | 2–11 (0.283) | 2–13 (0.124) | 0.43 / 0.21 | 3.67 (18) | 873 / 9,467 | 0 / 0 | 7.0 |
| T2 | 93683 | 65,536 | 0.938 / 0.717 / 0.408 | 2–10 (0.298) | 2–10 (0.161) | 0.31 / 0.24 | 3.99 (23) | 625 / 8,578 | 0 / 0 | 7.6 |
| T2 | 116089 | 65,536 | 0.937 / 0.651 / 0.394 | 2–9 (0.246) | 2–8 (0.140) | 0.46 / 0.25 | 3.93 (13) | 408 / 8,483 | 0 / 0 | 7.4 |
| T3 | 73129 | 65,536 | 0.914 / 0.598 / 0.267 | 2–9 (0.388) | 3–11 (0.223) | 0.67 / 0.55 | 3.32 (12) | 365 / 9,643 | 0 / 0 | 9.0 |
| T3 | 93683 | 65,536 | 0.923 / 0.580 / 0.174 | 2–8 (0.307) | 2–12 (0.230) | 0.61 / 0.50 | 3.61 (15) | 480 / 9,189 | 0 / 0 | 9.5 |
| T3 | 116089 | 65,536 | 0.968 / 0.640 / 0.189 | 1–8 (0.268) | 2–10 (0.159) | 0.65 / 0.57 | 3.60 (23) | 1,278 / 10,015 | 0 / 0 | 9.6 |
| T4 | 73129 | 65,536 | 0.934 / 0.677 / 0.193 | 1–8 (0.518) | 2–11 (0.310) | 0.45 / 0.27 | 3.70 (14) | 295 / 8,834 | 0 / 0 | 10.3 |
| T4 | 93683 | 65,536 | 0.915 / 0.577 / 0.190 | 1–12 (0.415) | 2–12 (0.295) | 0.37 / 0.26 | 3.91 (14) | 1,040 / 9,150 | 0 / 0 | 10.9 |
| T4 | 116089 | 65,536 | 0.965 / 0.685 / 0.205 | 1–6 (0.407) | 2–10 (0.269) | 0.46 / 0.32 | 3.86 (15) | 222 / 8,433 | 0 / 0 | 10.9 |

### The reader of the next byte (share right, positions 49,152 to 65,535)

| T | seed | cells near the output lines (distances) | near the output lines | all 1,984 cells | one-hot of the present byte | one-hots of the present and previous byte | table of the last 3 bytes |
|---|---|---|---|---|---|---|---|
| T1 | 73129 | 495 (d 8–13) | 0.0837 | 0.1140 | 0.1059 | 0.1296 | 0.2512 |
| T1 | 93683 | 519 (d 8–13) | 0.0784 | 0.1133 | 0.1035 | 0.1245 | 0.2533 |
| T1 | 116089 | 495 (d 8–13) | 0.0715 | 0.1130 | 0.1077 | 0.1268 | 0.2536 |
| T2 | 73129 | 495 (d 8–15) | 0.1301 | 0.2034 | 0.1059 | 0.1296 | 0.2512 |
| T2 | 93683 | 519 (d 8–15) | 0.1919 | 0.2294 | 0.1035 | 0.1245 | 0.2533 |
| T2 | 116089 | 495 (d 8–15) | 0.1046 | 0.2268 | 0.1077 | 0.1268 | 0.2536 |
| T3 | 73129 | 495 (d 8–17) | 0.1087 | 0.1702 | 0.1059 | 0.1296 | 0.2512 |
| T3 | 93683 | 519 (d 8–14) | 0.1365 | 0.2259 | 0.1035 | 0.1245 | 0.2533 |
| T3 | 116089 | 495 (d 8–15) | 0.1016 | 0.1711 | 0.1077 | 0.1268 | 0.2536 |
| T4 | 73129 | 495 (d 8–14) | 0.1213 | 0.2223 | 0.1059 | 0.1296 | 0.2512 |
| T4 | 93683 | 519 (d 8–16) | 0.1691 | 0.2301 | 0.1035 | 0.1245 | 0.2533 |
| T4 | 116089 | 495 (d 8–14) | 0.1506 | 0.2414 | 0.1077 | 0.1268 | 0.2536 |

## Answer, as measured

All four tissues share these properties, on three seeds:
- They read the present byte at d = 1, 2 and 4 by the tool. S(1,0) is 0.88 to 0.97, S(2,0) is 0.55 to 0.73 and S(4,0) is 0.16 to 0.42, all above 0.1103.
- They make and withdraw their own connections at a small number a cell: 3.07 to 3.99 connections a cell on average, at most 23 on any cell.
- They do not depend on their past: no cell differs between two copies after 64 or 512 bytes, in any window.
- They settle over the life: across the four windows, S(d,0) at any distance moves by at most 0.12, and connections a cell by at most 0.1.

They differ as follows:

- **T1** carries neither the byte one back nor the byte two back above its level at any distance. Few connections are made: 26 to 802 in 65,536 bytes, against about 9,800 given up, most of these early.
  - Near the output lines, the largest output is above 1/2 at d = 8 (0.52 to 0.57). But the reader of those cells is below the one-hot of the present byte: 0.072 to 0.084, against 0.104 to 0.108.
- **T2** carries the bytes one and two back above their levels from d = 2 to d = 8–13 on all three seeds. The best S is 0.24 to 0.34 at lag 1 and 0.11 to 0.17 at lag 2.
  - Near the output lines the reader gives 0.105 to 0.192. That is above the one-hot of the present byte on two seeds (73129, 93683) and above the one-hots of the present and previous bytes on the same two; on 73129 the margin is 0.0005.
  - It lacks amplitude there: the largest output at d = 8 is 0.30 to 0.46, so no cell there reaches 1/2.
- **T3** carries the bytes one and two back on all three seeds, at d = 1–2 to 6–9 for lag 1 (best 0.26 to 0.39) and at d = 1–3 to 9–12 for lag 2 (best 0.14 to 0.23).
  - It keeps the most amplitude near the output end: the largest output is 0.61 to 0.67 at d = 8 and 0.50 to 0.57 at d = 11.
  - Its S(4,0) is the lowest of the four with T4, at 0.17 to 0.27.
  - Near the output lines the reader gives 0.102 to 0.137. That is above the present byte alone on two seeds, and above the present and previous bytes on one (93683).
- **T4** carries the byte one back most strongly (best S 0.40 to 0.54) and the byte two back at 0.235 to 0.32, on all three seeds.
  - Near the output lines the reader gives 0.121 to 0.169. That is above the present byte alone on all three seeds and above the present and previous bytes on two (93683, 116089), the most of the four.
  - It loses amplitude near the output end: the largest output at d = 8 is 0.37 to 0.46.

**What every tissue still lacks near the output lines.** No reader of the cells near the output lines reaches the table of the last three bytes, which scores 0.251 to 0.254. Only T1 and T3 have cells there whose output exceeds 1/2. No tissue gives a byte: the emission was not read in these runs.
