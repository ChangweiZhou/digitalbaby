# The previous byte at the reading step (19 November 2026)

The reviewer asked for this in a message to E57's worker at 05:49 UTC on 19 November. The material is the reviewer's folder `reviewer_tissue_20261119` on `claude/vibrant-cerf-sje726` at d81a85c7, copied here as `TISSUE_PREV/`.

What I changed in the copy:

- **`tissue.c`**, additions marked `TISSUE_PREV`, each off unless set:
  - a time constant of the potential for each cell;
  - rho for each cell;
  - a delay in steps for each connection, with a history of outputs;
  - the input line on for the first n steps of a byte only;
  - `tis_run_phases6`, which records outputs after steps 1, 2, 4, 8, 16 and 32.
- **`tis.py`**: the six new fields appended to its structure.
- **The reviewer's `drive.py` does not know the new fields.** It is not used here; `prev.py` builds every tissue through `tis.py`.
- **New files**: `prev.py` (one tissue: item 1, scores, divergence), `prev_tables.py` (writes `PREV_TABLES.md`), `cand_queue.sh` and `seed_queue.sh` (the runs).

All figures are from one run per setting, seed 73129 unless said otherwise, window A2. Nothing is confirmed independently.

## Measurements

- **Item 1**: the share of a cell's output variance explained by the previous byte, using diag.py's η² on raw outputs. It is read after steps 1, 2, 4, 8, 16 and 32 of the byte, by distance d = 1 to 11.
- **Table**:
  - S(d, L) for d = 1, 2, 4 and L = 0, 1, 2, from readings.py on the reading step, as drive.py computes it;
  - c0 at lags 1 and 2;
  - the share of cells neither silent nor constant;
  - the divergence after 512 bytes, by diag.py's method.
- **The base** reproduces the reviewer's figures:

| Lag | d = 1 | d = 2 | d = 4 |
|---|---|---|---|
| S0 | 0.982 | 0.811 | 0.243 |
| S1 | 0.071 | 0.063 | 0.048 |
| S2 | 0.049 | 0.046 | 0.032 |

  The base's c0 is 0.070 at lag 1 and 0.056 at lag 2.

## 1. Where the previous byte is, in the base (`rule=fixed gain=2 rho=1`)

Median share explained by the previous byte, after steps 1 / 2 / 4 / 8 / 16 / 32 of the following byte:

| d | cells | step 1 | step 2 | step 4 | step 8 | step 16 | step 32 |
|---|---|---|---|---|---|---|---|
| 1 | 102 | 0.579 | 0.255 | 0.116 | 0.063 | 0.043 | 0.038 |
| 2 | 215 | 0.851 | 0.794 | 0.559 | 0.234 | 0.083 | 0.042 |
| 3 | 240 | 0.617 | 0.618 | 0.594 | 0.385 | 0.130 | 0.048 |
| 4 | 220 | 0.344 | 0.346 | 0.347 | 0.307 | 0.138 | 0.042 |
| 5 | 199 | 0.147 | 0.143 | 0.132 | 0.122 | 0.074 | 0.032 |
| 6 to 11 | | 0.028 or less at every step | | | | | |

- The previous byte is present at d = 1 to 5 only during the first steps of the next byte: at d = 1 until step 2, at d = 2 and 3 until about step 4 to 8, and at d = 4 until step 8.
- By step 32 the present byte has replaced it. The present byte's median at step 32 is 0.988 / 0.853 / 0.342 at d = 1 / 2 / 4.
- Unlike in the tissue with no control, the previous byte does not reach d ≥ 6 at any step.

## 2. The candidates, each alone (table in `PREV_TABLES.md`; bold there marks a score at or above its level)

| Candidate | S0 at d = 1 / 2 / 4 | S1 at d = 1 / 2 / 4 | S2 at d = 1 / 2 / 4 | Reaches the lag-1 level at |
|---|---|---|---|---|
| base | 0.982 / 0.811 / 0.243 | 0.071 / 0.063 / 0.048 | 0.049 / 0.046 / 0.032 | — |
| (a) rho 2, 1/4 or 1/2 of the cells | 0.291 / 0.039 / 0.019 and 0.270 / 0.058 / 0.023 | at most 0.031 | at most 0.028 | — |
| (b) potential time constant 8, 1/4 of the cells | 0.984 / 0.743 / 0.148 | 0.088 / **0.097** / 0.071 | 0.053 / 0.048 / 0.044 | d = 2 |
| (b) time constant 16, 1/4 | 0.984 / 0.621 / 0.125 | **0.142 / 0.159 / 0.105** | 0.072 / **0.081** / 0.057 | d = 1, 2, 4 |
| (b) time constant 32 on 1/4; 8, 16, 32 on 1/2 | S0 at d = 1 falls to 0.161–0.339, except time constant 8 on 1/2 (0.965 / 0.511 / 0.055) | at most 0.078 | at most 0.040 | — |
| (c) delay 8 steps, 1/4 of the connections between cells | 0.984 / 0.799 / 0.235 | **0.094 / 0.145 / 0.141** | 0.052 / 0.059 / 0.060 | d = 1, 2, 4 |
| (c) delay 16, 1/4 | 0.989 / 0.782 / 0.172 | **0.121 / 0.194 / 0.135** | 0.066 / **0.087 / 0.092** | d = 1, 2, 4 |
| (c) **delay 32, 1/4** | 0.988 / 0.716 / 0.120 | **0.260 / 0.512 / 0.139** | **0.099 / 0.192 / 0.124** | d = 1, 2, 4, and every lag |
| (c) delay 8, 1/2 | 0.989 / 0.794 / 0.163 | **0.106 / 0.218 / 0.156** | 0.059 / **0.079 / 0.102** | d = 1, 2, 4 |
| (c) delay 16, 1/2 | 0.989 / 0.767 / 0.112 | **0.187 / 0.369 / 0.135** | **0.082 / 0.140 / 0.118** | d = 1, 2, 4, and every lag |
| (c) delay 32, 1/2 | 0.991 / 0.595 / 0.049 | **0.334 / 0.670 / 0.091** | **0.165 / 0.335 / 0.087** | d = 1, 2, 4; S0 at d = 4 is below its level |
| (d) input line on for the first 8 steps (the input end, not a cell) | 0.157 / 0.196 / 0.132 | 0.052 / 0.055 / 0.046 | 0.031 / 0.032 / 0.023 | — |

In every delay row, every score at lag 0 is above its level, except d = 4 with delay 32 on half the connections (0.049). In delay 32 on a quarter, all nine scores are above their levels.

**What the failures have in common:**

- rho 2, time constant 32 on a quarter, and time constants 16 and 32 on a half all send the median gain of the ratio control to 3,414–4,096 (the cap), against 1.45 in the base.
- Mean outputs at step 32 are then 0.2 to 0.6, and the present byte's η² at d = 1 falls to about 0.06.
- Time constant 8 on half the cells has a median gain of 69 and loses S0 at d = 4.

**Delay 32 on a quarter, item 1.** The previous byte's median at step 32 is 0.062 / 0.197 / 0.143 / 0.079 at d = 1 / 2 / 3 / 4, against 0.038 / 0.042 / 0.048 / 0.042 in the base. Its upper decile at d = 2 is 0.498.

**Divergence after 512 bytes.** It is 1.0 in the base, in the delay rows (except delay 16 on half, 0), and in time constant 16 on a quarter. It is 0 in the runs whose gain ran away. A tissue whose state differs by 1.0 after 512 bytes can still carry the previous byte above its level.

## A second seed (93683): the base and three of the candidates

The levels above were computed for seed 73129, so these rows are compared with this seed's own base. This seed's c0 is 0.076 at lag 1 and 0.054 at lag 2.

| Tissue | S0 at d = 1 / 2 / 4 | S1 at d = 1 / 2 / 4 | S2 at d = 1 / 2 / 4 |
|---|---|---|---|
| base | 1.000 / 0.807 / 0.195 | 0.074 / 0.088 / 0.073 | 0.059 / 0.050 / 0.044 |
| delay 8, 1/4 | 1.000 / 0.815 / 0.171 | 0.084 / 0.145 / 0.116 | 0.064 / 0.081 / 0.067 |
| delay 32, 1/4 | 0.997 / 0.736 / 0.096 | 0.277 / 0.446 / 0.136 | 0.101 / 0.229 / 0.114 |
| time constant 16, 1/4 | 0.883 / 0.219 / 0.026 | 0.082 / 0.053 / 0.027 | 0.055 / 0.040 / 0.021 |

- **Delay 32 on a quarter** raises S1 and S2 at every d on this seed too. S0 at d = 4 falls from 0.195 to 0.096.
- **Time constant 16 on a quarter** fails on this seed: its median gain is 16.4 and its mean output at d = 4 is 0.25.

## What the numbers say

- **Delays work on both seeds.** A share of the connections between cells carrying a delay of 8 to 32 steps brings the previous byte, and with longer delays the byte two back, to the reading step at d = 1, 2 and 4. The present byte stays above its level except with the longest delay on half the connections. This is a quantity of one connection: its delay.
- **The slower potential worked on one seed only.** It reached the levels on seed 73129 at time constant 16 on a quarter of the cells, but failed on seed 93683.
- **The slower potential, and rho above 1, can make the ratio control run away.** That is a property of how the candidate combines with the control, not of the candidate alone.
- **Switching the input line off after 8 steps** removes the present byte at d = 1 (S0 0.157, share 0.029) and does not bring the previous byte.

## Other work in this session

In `TISSUE_EVENT/evt_core.c`, the sizes of the event arrays (made and given up) are now `size_t`, as the reviewer asked. They overflowed an int in `realloc` near 250,000 bytes of the coincidence tissue. Part (a) was rerun afterwards: the example and the 512-byte tissue pass, and their receipts were rewritten.

## Cost

The 20 runs took 0.43 CPU-hours (`prev/*.json`), with at most three processes of one thread each.
