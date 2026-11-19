# Bytes one to four back at the reading step, on bases that cannot depend on their past (19 November 2026)

The reviewer asked for this in two messages to E57's worker, at 06:06 and 06:14 UTC on 19 November. The material is the reviewer's folder `reviewer_tissue_20261119` at d4250a84 on `claude/vibrant-cerf-sje726`, copied here as `TISSUE_PREV2/`.

What I changed in the copy:

- **`tissue.c` and `tis.py`**: the additions of `TISSUE_PREV` are reapplied, marked `TISSUE_PREV2`. They are delays for each connection with a history of outputs, a time constant and rho for each cell, and the input line on for the first steps of a byte; each is off unless set.
- **`prev2.py`** (new): one tissue, its scores, and its divergence. With `full=1` it also reads every distance at lags 0 to 4 in two ways: the tool as it is, and no floor with each cell divided by its own standard deviation. The second way is the method of the helper's `where.py` at 4bda7aa1, which is kept here as `where_helper_4bda7aa1.py`.
- **`prev2_tables.py`** (new) writes `PREV2_TABLES.md`, which holds every figure below.
- **`q1.sh`, `q2.sh`, `q3.sh`**: the runs.

All figures are from one run per setting. Nothing is confirmed independently.

## Terms

- **M**: `rule=fixed gain=2 rho=1 gmax=4`, the ratio control with the gain bounded at 4.
- **U**: `rule=fixed`, no gain.
- **A delayed connection** carries the sender's output of 1 + D steps before; an undelayed one carries that of the step before.
- **Levels**:
  - Seed 73129 (the reviewer's): lag 0: 0.1103; lag 1: 0.090; lag 2: 0.076.
  - For the three-seed readings at lags 1 to 4: c0(L) + 0.02 of each seed. This is the rule that gives 0.090 and 0.076 from seed 73129's c0 of 0.070 and 0.056; it is my reading of how those levels were set.
  - On seed 73129, c0 at lag 3 is 0.162: the stream's structure makes the byte three back partly predictable from the present byte.
- **Divergence**: two copies of the tissue, the second with its most active cell's potential changed by 1e-9; the largest difference of outputs 64 and 512 bytes later.
- **It is 0 after 64 and after 512 bytes in every run of this folder (31 runs).** None of these tissues depends on its past.

## Step 1: a share of the connections between cells delayed (seed 73129, the tool as it is)

Bold marks a score at or above its level.

| Tissue | S0 at d = 1 / 2 / 4 | S1 at d = 1 / 2 / 4 | S2 at d = 1 / 2 / 4 |
|---|---|---|---|
| M base | **0.979 / 0.826 / 0.306** | 0.068 / 0.056 / 0.040 | 0.050 / 0.039 / 0.030 |
| M, delay 16 on 1/4 | **0.981 / 0.837 / 0.227** | 0.067 / **0.116 / 0.092** | 0.050 / 0.043 / 0.034 |
| M, delay 32 on 1/4 | **0.985 / 0.835 / 0.161** | **0.145 / 0.651 / 0.148** | 0.055 / **0.105** / 0.075 |
| M, delay 16 on 1/2 | **0.984 / 0.840 / 0.161** | 0.074 / **0.320 / 0.126** | 0.050 / 0.057 / 0.054 |
| M, delay 32 on 1/2 | **1.000 / 0.752** / 0.062 | **0.214 / 0.794 / 0.135** | 0.066 / **0.302 / 0.094** |
| U base | **0.979 / 0.826** / 0.022 | 0.066 / 0.056 / 0.022 | 0.053 / 0.042 / 0.022 |
| U, delay 32 on 1/4 | **0.984 / 0.849** / 0.022 | 0.086 / **0.584** / 0.022 | 0.053 / 0.060 / 0.022 |
| U, delay 32 on 1/2 | **1.000 / 0.734** / 0.022 | **0.102 / 0.780** / 0.022 | 0.055 / 0.066 / 0.022 |

The rows not shown change little:

- M with delay 8 (on 1/4 or 1/2) does not reach the lag-1 level at any d.
- U with delay 8 or 16 equals the U base within 0.003 (largest difference 0.0026).

So, under the tool as it is, only delays near a byte carry the previous byte on these bases, and in U only at d ≤ 2.

## Step 2: every connection between cells with its own delay (seed 73129, the tool as it is)

| Tissue | S0 at d = 1 / 2 / 4 | S1 at d = 1 / 2 / 4 | S2 at d = 1 / 2 / 4 |
|---|---|---|---|
| M, delay proportional to the distance of the two cells, dmax 8 | **0.978 / 0.831 / 0.166** | 0.067 / 0.084 / **0.156** | 0.049 / 0.042 / 0.035 |
| M, dmax 16 | **0.993 / 0.837** / 0.027 | 0.073 / **0.286 / 0.205** | 0.050 / 0.051 / 0.073 |
| M, dmax 32 | **0.988 / 0.823** / 0.052 | **0.111 / 0.680** / 0.045 | 0.055 / **0.211 / 0.160** |
| **M, delay drawn from 0 to 32** | **0.988 / 0.850** / 0.050 | **0.104 / 0.584 / 0.114** | 0.053 / **0.153 / 0.128** |
| U, dmax 32 | **0.984 / 0.827** / 0.022 | 0.073 / **0.566** / 0.022 | 0.051 / 0.061 / 0.022 |
| U, delay drawn from 0 to 32 | **0.986 / 0.830** / 0.022 | 0.073 / **0.350** / 0.022 | 0.053 / 0.054 / 0.022 |

U with dmax 8 or 16 equals the U base within 0.003 (largest difference 0.0026).

## Step 3: three seeds, every distance, lags 0 to 4 (M; `PREV2_TABLES.md` gives every row)

**Read with no floor and each cell standardised.** For each lag: the best score over distances 1 to 11 (and where it falls), then the distances at which the score is at or above the seed's level.

**M base:**

| Seed | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 4 |
|---|---|---|---|---|---|
| 73129 | 0.995; d 1–9 | 0.196 at d 8; d 6–10 | 0.075; none | 0.156 at d 1; none | 0.081; none |
| 93683 | 1.000; d 1–9 | 0.210; d 5–11 | 0.083; d 9–11 | 0.154; none | 0.088; none |
| 116089 | 1.000; d 1–9 | 0.261; d 5–11 | 0.099; d 9–11 | 0.155; none | 0.081; none |

**M, delay drawn from 0 to 32 on every connection between cells:**

| Seed | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 4 |
|---|---|---|---|---|---|
| 73129 | 0.999; d 1–4 | 0.715 at d 3; d 1–5 | 0.488 at d 4; d 2–8 | 0.326 at d 4; d 3–6 | 0.269 at d 6; d 2–9 |
| 93683 | 1.000; d 1–4 | 0.752 at d 3; d 1–6 | 0.553 at d 4; d 2–8 | 0.419 at d 4; d 2–6 | 0.334 at d 5; d 2–8 |
| 116089 | 1.000; d 1–4 | 0.680 at d 2; d 1–5 | 0.473 at d 4; d 2–7 | 0.296 at d 5; d 3–6 | 0.217 at d 6; d 2–8 |

**M, delay 32 on 1/4 of the connections between cells:**

| Seed | Lag 0 | Lag 1 | Lag 2 | Lag 3 | Lag 4 |
|---|---|---|---|---|---|
| 73129 | 0.997; d 1–8 | 0.659 at d 2; d 1–9 | 0.304 at d 3; d 2–9 | 0.196 at d 7; d 6–8 | 0.158 at d 8; d 2–9 |
| 93683 | 1.000; d 1–8 | 0.629 at d 2; d 1–9 | 0.368 at d 4; d 2–9 | 0.268 at d 6; d 2–8 | 0.190 at d 8; d 2–9 |
| 116089 | 1.000; d 1–7 | 0.721 at d 2; d 1–9 | 0.306 at d 4; d 2–9 | 0.208 at d 6; d 6–8 | 0.146 at d 8; d 2, 6–9 |

**The pool of all non-feedback cells** (a sample of 128), at lags 0 to 4:

| Tissue | 73129 | 93683 | 116089 |
|---|---|---|---|
| M random 0–32 | 0.400 / 0.355 / 0.284 / 0.270 / 0.232 | 0.402 / 0.466 / 0.410 / 0.318 / 0.273 | 0.498 / 0.300 / 0.238 / 0.230 / 0.198 |
| M base | 0.623 / 0.161 / 0.081 / 0.113 / 0.082 | 0.714 / 0.155 / 0.090 / 0.156 / 0.084 | 0.702 / 0.204 / 0.073 / 0.129 / 0.077 |

**With the tool as it is (floor 0.01)**, the delays' effect at lags 1 and 2 is the one shown in steps 1 and 2. At lags 3 and 4, at d = 1, 2 and 4, the tool's scores stay near c0 of the lag, except at d = 2. There, M random 0–32 gives 0.124–0.164 at lag 3 and 0.088–0.147 at lag 4 on the three seeds, against c0 of 0.153–0.162 and 0.073–0.087.

## What the numbers say

1. **On bases that cannot depend on their past, delays of about one byte carry the earlier bytes.**
   - In M, a delay drawn from 0 to 32 steps for every connection between cells carries the byte one back above its level at d = 1, 2, 4 with the tool as it is, and the byte two back at d = 2, 4.
   - Read without the floor, it carries bytes one to four back above their levels, on all three seeds: lag 4 at d = 2 to 8 or 9.
   - The base carries only the byte one back (at d = 5 to 11), and the byte two back only at d = 9 to 11 on two of the three seeds.
2. **The cost is the present byte far from the input.**
   - With random delays, S0 under the tool falls at d = 4: 0.050 / 0.032 / 0.033 on the three seeds, against 0.306 / 0.309 / 0.260 in the base.
   - Without the floor, lag 0 is above its level only to d = 4, against d = 9 in the base.
   - Delay 32 on a quarter keeps S0 at d = 4 above its level on all three seeds (0.161 / 0.175 / 0.151) and lag 0 to d = 7 or 8. It reaches lag 4 at d = 2 to 9 as well.
3. **U, with no gain, carries the earlier bytes under the tool only at d ≤ 2.** Its amplitude at greater distances is below the tool's floor.
4. **On the unbounded base of `TISSUE_PREV`**, delays of 8 to 16 steps had larger effects under the tool. Under the bound (M), delay 8 has none and delay 16 a smaller one.

## Cost

31 runs took 0.96 CPU-hours (`prev2/*.json`), with at most three processes of one thread each.
