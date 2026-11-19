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

## Part (e) and T5 to T8 (the reviewer's message of 07:32 UTC, which reached this session at 08:53)

**The program, rebuilt.** It is rebuilt on the reviewer's tissue.c at e7be1222. That version holds part (e): `prune_only`, the gain `gain=3` (two_sided 3) and making on the long mean `mk_long`. Parts (b), (c) and (d) are re-added unchanged. The first build, on which T1 to T4 ran, is kept in `v1/`.

**The checks, rerun on the rebuilt program** (`equiv_*.log`; the first build's checks are in `equiv_v1/`):
- **base B1, (d) and (e)** give outputs and connections identical to their authors' programs. Part (e) was run as `rule=timing prune_only=1 norm=1 lines_learn=1 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`: 3.24 connections a cell, 1,252 made, 10,482 given up.
- **(b) and (c)** give identical connections and identical counts of made and given up. Their outputs, however, now differ from their authors' programs at rounding level:
  - (c): largest difference 2.8e-14, in 2 cells;
  - (b): largest difference 1.4e-5, in 27 cells.
- On the first build both were exact. The likely cause is that the compiled code of e7be1222 orders its floating-point operations differently; it is not a change of rule, but this was not proved.

**The tissues.** All four use part (e) as above, with totals 3/4 and 1/4 (the defaults):

| Tissue | Parts | Added options |
|---|---|---|
| T5 | (e) with making | none |
| T6 | T5 + (c) | `tau=u2-64e` |
| T7 | T5 + (d) | `dmax=16` |
| T8 | T5 + (c) + (d) | `tau=u2-64e dmax=16` |

The readings are as for T1 to T4. Every window of T1 to T8 is in `T_TABLE.md`.

### Last window (bytes 61,440 to 65,535), T5 to T8

| T | seed | window end | S(d,0) tool d = 1 / 2 / 4 | S(d,0) tool d = 8 / 11 | lag 1 above level: distances (best S) | lag 2 above level: distances (best S) | largest output d = 8 / 11 | connections a cell (largest) | made / given up | cells differing after 64 / 512 bytes | ms a byte |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T5 | 73129 | 65,536 | 0.975 / 0.812 / 0.642 | 0.403 / 0.318 | 7–14 (0.205) | 9–10 (0.093) | 1.00 / 1.00 | 3.34 (21) | 1,471 / 10,718 | 0 / 0 | 8.8 |
| T5 | 93683 | 65,536 | 0.983 / 0.860 / 0.565 | 0.276 / 0.128 | 9–12 (0.131) | none | 1.00 / 1.00 | 3.62 (16) | 1,741 / 10,427 | 0 / 0 | 9.3 |
| T5 | 116089 | 65,536 | 0.999 / 0.785 / 0.564 | 0.145 / 0.095 | 6 (0.095) | none | 1.00 / 1.00 | 3.52 (17) | 1,000 / 9,886 | 0 / 0 | 9.0 |
| T6 | 73129 | 65,536 | 0.977 / 0.813 / 0.591 | 0.102 / 0.056 | 2–11 (0.346) | 3–14 (0.190) | 1.00 / 1.00 | 3.63 (16) | 3,605 / 12,271 | 0 / 0 | 9.7 |
| T6 | 93683 | 65,536 | 0.991 / 0.869 / 0.407 | 0.113 / 0.084 | 2–11 (0.339) | 2–12 (0.209) | 1.00 / 1.00 | 4.04 (12) | 6,461 / 14,318 | 0 / 0 | 10.1 |
| T6 | 116089 | 65,536 | 1.000 / 0.855 / 0.537 | 0.137 / 0.055 | 1–10 (0.292) | 2–11 (0.179) | 1.00 / 1.00 | 3.71 (14) | 3,472 / 11,987 | 0 / 0 | 9.6 |
| T7 | 73129 | 65,536 | 0.922 / 0.698 / 0.320 | 0.083 / 0.045 | 1–9 (0.434) | 1–12 (0.312) | 1.00 / 1.00 | 3.52 (10) | 768 / 9,657 | 0 / 0 | 12.7 |
| T7 | 93683 | 65,536 | 0.947 / 0.688 / 0.245 | 0.068 / 0.059 | 1–9 (0.363) | 1–11 (0.232) | 1.00 / 1.00 | 3.71 (11) | 1,024 / 9,537 | 0 / 0 | 13.1 |
| T7 | 116089 | 65,536 | 0.962 / 0.711 / 0.227 | 0.062 / 0.049 | 1–8 (0.361) | 1–10 (0.227) | 1.00 / 1.00 | 3.68 (11) | 1,035 / 9,597 | 0 / 0 | 13.3 |
| T8 | 73129 | 65,536 | 0.922 / 0.691 / 0.187 | 0.070 / 0.034 | 1–6 (0.484) | 1–8 (0.325) | 1.00 / 1.00 | 3.49 (10) | 1,103 / 10,053 | 0 / 0 | 13.0 |
| T8 | 93683 | 65,536 | 0.901 / 0.620 / 0.103 | 0.053 / 0.026 | 1–6 (0.384) | 1–8 (0.273) | 1.00 / 1.00 | 3.76 (13) | 986 / 9,397 | 0 / 0 | 13.4 |
| T8 | 116089 | 65,536 | 0.967 / 0.724 / 0.157 | 0.048 / 0.035 | 1–6 (0.439) | 1–7 (0.267) | 1.00 / 1.00 | 3.67 (11) | 1,066 / 9,654 | 0 / 0 | 13.0 |

### The reader of the next byte, T5 to T8

| T | seed | cells near the output lines (distances) | near the output lines | all 1,984 cells | one-hot of the present byte | one-hots of the present and previous byte | table of the last 3 bytes |
|---|---|---|---|---|---|---|---|
| T5 | 73129 | 495 (d 8–16) | 0.1079 | 0.1467 | 0.1059 | 0.1296 | 0.2512 |
| T5 | 93683 | 519 (d 8–17) | 0.0782 | 0.1465 | 0.1035 | 0.1245 | 0.2533 |
| T5 | 116089 | 495 (d 8–19) | 0.1124 | 0.1328 | 0.1077 | 0.1268 | 0.2536 |
| T6 | 73129 | 495 (d 0–16) | 0.0970 | 0.1967 | 0.1059 | 0.1296 | 0.2512 |
| T6 | 93683 | 519 (d 0–17) | 0.0947 | 0.2020 | 0.1035 | 0.1245 | 0.2533 |
| T6 | 116089 | 495 (d 8–17) | 0.1346 | 0.2218 | 0.1077 | 0.1268 | 0.2536 |
| T7 | 73129 | 495 (d 8–17) | 0.1058 | 0.2006 | 0.1059 | 0.1296 | 0.2512 |
| T7 | 93683 | 519 (d 8–17) | 0.1275 | 0.2166 | 0.1035 | 0.1245 | 0.2533 |
| T7 | 116089 | 495 (d 8–16) | 0.1063 | 0.2062 | 0.1077 | 0.1268 | 0.2536 |
| T8 | 73129 | 495 (d 8–17) | 0.0731 | 0.2119 | 0.1059 | 0.1296 | 0.2512 |
| T8 | 93683 | 519 (d 7–18) | 0.0667 | 0.2025 | 0.1035 | 0.1245 | 0.2533 |
| T8 | 116089 | 495 (d 8–18) | 0.0749 | 0.2285 | 0.1077 | 0.1268 | 0.2536 |

### Answer for T5 to T8, as measured

All four share these properties, on three seeds:
- The largest output is 1.00 at d = 8 and d = 11. Every T5 to T8 tissue keeps full amplitude to the farthest cells, unlike T1 to T4.
- They make and withdraw their own connections, at 3.3 to 4.04 a cell (largest 21).
- No cell differs between two copies after 64 or 512 bytes, in any window.

They differ as follows:

- **T5** reads the present byte best at every distance:
  - S(d,0) is 0.98 to 1.00 at d = 1, 0.79 to 0.86 at d = 2, 0.56 to 0.64 at d = 4, and 0.15 to 0.40 at d = 8.
  - It carries the byte one back above its level only at the far cells, and only on two seeds (d = 7 to 14 on 73129, d = 7 to 13 on 93683; best 0.13 to 0.21). The byte two back is above its level only at d = 9 and 10 in one window.
- **T6** keeps T5's reading at d = 1 to 4 (S(4,0) 0.40 to 0.59).
  - It carries both earlier bytes above their levels from d = 1–3 out to d = 9–14 on all three seeds (lag 1 best 0.27 to 0.35; lag 2 best 0.14 to 0.21).
  - It loses part of T5's present byte at the far cells: d = 8 is 0.10 to 0.14.
  - On two seeds a few cells near the output lines end with no connection (distance 0 in the reach set).
- **T7** carries the earlier bytes more strongly and nearer the input lines (lag 1 at d = 1 to 7–9, best 0.35 to 0.44; lag 2 at d = 1 to 10–12, best 0.22 to 0.32).
  - It gives up the present byte from d = 4 on: S(4,0) 0.23 to 0.35, and d = 8 at most 0.08.
- **T8** carries the byte one back most strongly (best 0.38 to 0.50), at d = 1 to 6 only.
  - S(4,0) is 0.10 to 0.19, below the level of 0.1103 on seed 93683.

**Near the output lines, T5 to T8** (reader of the next byte, last quarter):
- **T5** gives 0.078 to 0.112, about the present byte alone (0.104 to 0.108). It is below the present and previous bytes on every seed.
- **T6** gives 0.095 to 0.135. It is above the present and previous bytes only on 116089.
- **T7** gives 0.106 to 0.128. It is above the present and previous bytes only on 93683.
- **T8** gives 0.067 to 0.075, below the present byte alone on every seed.
- **All 1,984 cells** give 0.133 to 0.147 for T5 and 0.197 to 0.229 for T6 to T8. No reader reaches the table of the last three bytes (0.251 to 0.254).

So full amplitude near the output lines (T5 to T8), against at most 0.67 in T1 to T4, does not make those cells carry more of what the next byte depends on. The best readers near the output lines remain T4 (0.121 to 0.169) and T2 on one seed (0.192).

## Overall answer (T1 to T8, three seeds)

Every tissue reads the present byte at d = 1, 2 and 4 above the level, except T8 at d = 4 on seed 93683. Every tissue makes and withdraws its own connections at about 3 to 4 a cell, and none depends on its past.

The bytes one and two back are carried above their levels by T2, T3, T4, T6, T7 and T8 on all three seeds. T5 carries them only weakly, and T1 not at all.

Only T5 to T8 keep full amplitude (largest output 1.00) out to the cells near the output lines. Of these, T6 also keeps the present byte at d = 4 (S(4,0) 0.40 to 0.59) and carries both earlier bytes.

What every tissue still lacks near the output lines: a reader of those cells that does better than a table of the last three bytes. The emission itself was not read in these runs.

## Long axons (the reviewer's message of 10:32 UTC): T5L to T8L

**The program, third build.** It is the reviewer's tissue.c and tis.py at aa0a4341, compiled with `-ffp-contract=off`. That version holds part (e), a time constant for every cell (`tau=u2-N`, by the reviewer's draw) and the long axons (`long=K-R`, as in THE_TISSUE_AS_RUN_20261119.md, section 3a). Parts (b) and (d) are re-added, and the helper's draw of time constants is kept as `tau=h:SPEC`. The second build is kept in `v2/`.

**Checks** (`equiv_*.log`; the second build's checks are in `equiv_v2/`). Each compares the merged program with the reviewer's own on seed 73129, over 16,384 bytes.

| Setting | Outputs of every cell, last 512 bytes | Connections | Made / given up | Connections a cell (over 2,048) |
|---|---|---|---|---|
| The reviewer's L8_256 row: `… a_make=1/16384 start_w=1/64 tau=u2-256 long=8-1/2` | identical (difference 0.0) | identical | 898 / 11,657 | 2.50 |
| T5's options with `long=8-1/2` | identical | identical | 473 / 10,976 | 2.62 |

- Base B1 and the delays (d) are also identical to their authors' programs.
- Part (b) differs at rounding level (1.4e-5), because its author's build does not use `-ffp-contract=off`.

**The tissues.** All use T5's options with `long=8-1/2`: `rule=timing prune_only=1 norm=1 lines_learn=1 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`, with totals 3/4 and 1/4.

| Tissue | Added options |
|---|---|
| T5L | none |
| T6L64 | `tau=u2-64` |
| T6L128 | `tau=u2-128` |
| T6L256 | `tau=u2-256` |
| T7L | `dmax=16` |
| T8L | `tau=u2-64 dmax=16` |

The time constants use the reviewer's draw. Each tissue runs three seeds for 65,536 bytes, read in four windows; the first window ends at 16,384, and all four are in `ntd/*.json`.

**The cells near the output lines, two ways:**
- **near18:** the drivers, with every cell within 1/8 of a driver (495 to 519 cells).
- **nearL:** the drivers, with every member of their enlarged neighbourhoods (569 to 595 cells). This set adds the long-axon cells within 1/2 of a driver.

**The next-byte reader.** `reach2.py` is reach.py with the second set added.

### S near the output lines and the whole tissue (last window), `L_TABLE.md`

| tissue | seed | near18 (cells): S lag 0 / 1 / 2 / 3 | nearL (cells): S lag 0 / 1 / 2 / 3 | levels lag 0 / 1 / 2 / 3 | whole tissue S(d,0) d = 1 / 2 / 4 | largest distance | connections a cell (largest) | made / given up | cells differing after 64 / 512 bytes | ms a byte |
|---|---|---|---|---|---|---|---|---|---|---|
| T5L | 73129 | (495) 0.259 / 0.042 / 0.038 / 0.070 | (577) 0.288 / 0.047 / 0.039 / 0.072 | 0.110 / 0.102 / 0.073 / 0.181 | 0.972 / 0.723 / 0.324 | 8 | 2.67 (15) | 545 / 11,118 | 0 / 0 | 8.0 |
| T5L | 93683 | (519) 0.237 / 0.098 / 0.041 / 0.053 | (595) 0.277 / 0.099 / 0.039 / 0.055 | 0.110 / 0.100 / 0.081 / 0.186 | 0.909 / 0.418 / 0.288 | 8 | 2.71 (12) | 643 / 11,131 | 0 / 0 | 8.0 |
| T5L | 116089 | (495) 0.283 / 0.046 / 0.034 / 0.063 | (569) 0.283 / 0.044 / 0.038 / 0.057 | 0.110 / 0.091 / 0.074 / 0.178 | 0.927 / 0.687 / 0.331 | 7 | 2.69 (11) | 397 / 10,923 | 0 / 0 | 8.2 |
| T6L64 | 73129 | (495) 0.122 / 0.096 / 0.051 / 0.053 | (577) 0.137 / 0.093 / 0.061 / 0.054 | 0.110 / 0.102 / 0.073 / 0.181 | 0.967 / 0.586 / 0.172 | 9 | 3.30 (12) | 767 / 10,093 | 0 / 0 | 9.1 |
| T6L64 | 93683 | (519) 0.133 / 0.151 / 0.119 / 0.100 | (595) 0.135 / 0.157 / 0.123 / 0.102 | 0.110 / 0.100 / 0.081 / 0.186 | 0.839 / 0.539 / 0.157 | 8 | 3.06 (13) | 1,412 / 11,221 | 0 / 0 | 8.8 |
| T6L64 | 116089 | (495) 0.219 / 0.195 / 0.118 / 0.068 | (569) 0.244 / 0.225 / 0.125 / 0.095 | 0.110 / 0.091 / 0.074 / 0.178 | 0.957 / 0.642 / 0.272 | 9 | 2.87 (10) | 1,254 / 11,431 | 0 / 0 | 8.3 |
| T6L128 | 73129 | (495) 0.166 / 0.156 / 0.110 / 0.099 | (577) 0.187 / 0.163 / 0.115 / 0.095 | 0.110 / 0.102 / 0.073 / 0.181 | 0.981 / 0.641 / 0.195 | 8 | 2.97 (12) | 575 / 10,560 | 0 / 0 | 8.7 |
| T6L128 | 93683 | (519) 0.180 / 0.238 / 0.214 / 0.167 | (595) 0.185 / 0.249 / 0.217 / 0.156 | 0.110 / 0.100 / 0.081 / 0.186 | 0.824 / 0.501 / 0.283 | 8 | 2.82 (12) | 4,379 / 14,663 | 0 / 0 | 7.9 |
| T6L128 | 116089 | (495) 0.221 / 0.191 / 0.117 / 0.098 | (569) 0.226 / 0.204 / 0.126 / 0.099 | 0.110 / 0.091 / 0.074 / 0.178 | 0.939 / 0.579 / 0.263 | 10 | 2.67 (12) | 1,226 / 11,795 | 0 / 0 | 8.2 |
| T6L256 | 73129 | (495) 0.165 / 0.172 / 0.147 / 0.144 | (577) 0.181 / 0.176 / 0.143 / 0.139 | 0.110 / 0.102 / 0.073 / 0.181 | 0.934 / 0.467 / 0.193 | 9 | 2.58 (8) | 723 / 11,480 | 0 / 0 | 7.6 |
| T6L256 | 93683 | (519) 0.169 / 0.170 / 0.151 / 0.141 | (595) 0.173 / 0.186 / 0.162 / 0.144 | 0.110 / 0.100 / 0.081 / 0.186 | 0.864 / 0.499 / 0.226 | 9 | 2.70 (17) | 4,956 / 15,466 | 0 / 0 | 7.4 |
| T6L256 | 116089 | (495) 0.180 / 0.173 / 0.140 / 0.112 | (569) 0.233 / 0.218 / 0.153 / 0.126 | 0.110 / 0.091 / 0.074 / 0.178 | 0.969 / 0.621 / 0.250 | 10 | 2.64 (10) | 1,851 / 12,485 | 0 / 0 | 7.5 |
| T7L | 73129 | (495) 0.180 / 0.215 / 0.130 / 0.094 | (577) 0.191 / 0.227 / 0.124 / 0.094 | 0.110 / 0.102 / 0.073 / 0.181 | 0.852 / 0.530 / 0.202 | 6 | 3.32 (8) | 13 / 9,297 | 0 / 0 | 12.7 |
| T7L | 93683 | (519) 0.191 / 0.221 / 0.151 / 0.124 | (595) 0.188 / 0.221 / 0.159 / 0.115 | 0.110 / 0.100 / 0.081 / 0.186 | 0.731 / 0.350 / 0.219 | 7 | 2.95 (8) | 105 / 10,118 | 0 / 0 | 12.4 |
| T7L | 116089 | (495) 0.114 / 0.135 / 0.075 / 0.071 | (569) 0.124 / 0.139 / 0.080 / 0.074 | 0.110 / 0.091 / 0.074 / 0.178 | 0.838 / 0.439 / 0.157 | 7 | 3.16 (8) | 34 / 9,628 | 0 / 0 | 12.7 |
| T8L | 73129 | (495) 0.119 / 0.135 / 0.143 / 0.125 | (577) 0.129 / 0.149 / 0.155 / 0.128 | 0.110 / 0.102 / 0.073 / 0.181 | 0.902 / 0.418 / 0.151 | 7 | 3.19 (8) | 37 / 9,587 | 0 / 0 | 12.7 |
| T8L | 93683 | (519) 0.080 / 0.137 / 0.148 / 0.138 | (595) 0.094 / 0.150 / 0.149 / 0.139 | 0.110 / 0.100 / 0.081 / 0.186 | 0.750 / 0.358 / 0.088 | 8 | 3.11 (8) | 113 / 9,820 | 0 / 0 | 12.4 |
| T8L | 116089 | (495) 0.078 / 0.145 / 0.161 / 0.126 | (569) 0.105 / 0.194 / 0.190 / 0.143 | 0.110 / 0.091 / 0.074 / 0.178 | 0.901 / 0.499 / 0.108 | 8 | 2.91 (9) | 73 / 10,169 | 0 / 0 | 11.6 |

### The reader of the next byte (last quarter), `L_READ.md`

| tissue | seed | near: drivers + enlarged neighbourhoods (cells) | near: drivers + cells within 1/8 (cells) | all 1,984 cells | one-hot present | one-hots present + previous | 3-byte table |
|---|---|---|---|---|---|---|---|
| T5L | 73129 | 0.0986 (577) | 0.0954 (495) | 0.1313 | 0.1059 | 0.1296 | 0.2512 |
| T5L | 93683 | 0.0756 (595) | 0.0771 (519) | 0.1157 | 0.1035 | 0.1245 | 0.2533 |
| T5L | 116089 | 0.0915 (569) | 0.0867 (495) | 0.1328 | 0.1077 | 0.1268 | 0.2536 |
| T6L64 | 73129 | 0.1244 (577) | 0.1182 (495) | 0.1966 | 0.1059 | 0.1296 | 0.2512 |
| T6L64 | 93683 | 0.1205 (595) | 0.1176 (519) | 0.2000 | 0.1035 | 0.1245 | 0.2533 |
| T6L64 | 116089 | 0.1669 (569) | 0.1610 (495) | 0.2203 | 0.1077 | 0.1268 | 0.2536 |
| T6L128 | 73129 | 0.1851 (577) | 0.1796 (495) | 0.2205 | 0.1059 | 0.1296 | 0.2512 |
| T6L128 | 93683 | 0.1456 (595) | 0.1389 (519) | 0.2302 | 0.1035 | 0.1245 | 0.2533 |
| T6L128 | 116089 | 0.1648 (569) | 0.1602 (495) | 0.2388 | 0.1077 | 0.1268 | 0.2536 |
| T6L256 | 73129 | 0.1790 (577) | 0.1782 (495) | 0.2208 | 0.1059 | 0.1296 | 0.2512 |
| T6L256 | 93683 | 0.1224 (595) | 0.1183 (519) | 0.2179 | 0.1035 | 0.1245 | 0.2533 |
| T6L256 | 116089 | 0.1578 (569) | 0.1585 (495) | 0.2418 | 0.1077 | 0.1268 | 0.2536 |
| T7L | 73129 | 0.1578 (577) | 0.1570 (495) | 0.1796 | 0.1059 | 0.1296 | 0.2512 |
| T7L | 93683 | 0.1412 (595) | 0.1388 (519) | 0.1735 | 0.1035 | 0.1245 | 0.2533 |
| T7L | 116089 | 0.1224 (569) | 0.1198 (495) | 0.1613 | 0.1077 | 0.1268 | 0.2536 |
| T8L | 73129 | 0.1619 (577) | 0.1559 (495) | 0.2155 | 0.1059 | 0.1296 | 0.2512 |
| T8L | 93683 | 0.1274 (595) | 0.1251 (519) | 0.2115 | 0.1035 | 0.1245 | 0.2533 |
| T8L | 116089 | 0.1838 (569) | 0.1769 (495) | 0.2245 | 0.1077 | 0.1268 | 0.2536 |

### Answer (item 4), as measured

Two tissues keep the present byte near the output lines above its level while carrying the bytes one and two back above theirs, on all three seeds and in both near sets:

| Tissue | Present byte near the output lines | Lag 1 | Lag 2 | Next-byte reader near the output lines |
|---|---|---|---|---|
| T6L128 | 0.166 to 0.226 | 0.156 to 0.249 | 0.110 to 0.217 | 0.139 to 0.185 |
| T6L256 | 0.165 to 0.233 | 0.170 to 0.218 | 0.140 to 0.162 | 0.118 to 0.179 |

- **T6L128 is the stronger.** Its near-output reader is above the one-hots of the present and previous bytes on all three seeds. T6L256's reader falls just below them on 93683 (0.122 against 0.125).
- Lag 3 is below its level near the output lines in every tissue.
- Neither tissue depends on its past (0 cells differ after 64 and 512 bytes). Each holds 2.6 to 3.0 connections a cell (largest 12 and 17) and makes 575 to 4,956 connections in the life.
- **The other four:**
  - T5L carries no earlier byte.
  - T6L64 misses lags 1 and 2 on 73129.
  - T7L misses lag 2 on 116089 (0.075, at its level).
  - T8L loses the present byte near the output lines on 93683 and 116089.
- **The two near sets differ little.** In every row the reader on the drivers' enlarged neighbourhoods and the reader on the cells within 1/8 differ by at most 0.007, so the long-axon cells that reach the drivers add little to what those drivers' neighbours tell.
- **What is still lacking:** no near-output reader reaches the table of the last three bytes (0.251 to 0.254). The emission was not read.
