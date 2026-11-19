# Delays in the tissue that carries the present byte far (19 November 2026)

The reviewer asked for this in a message to E57's worker at 06:42 UTC on 19 November. The material is the reviewer's folder `reviewer_tissue_20261119` at 6d805e04 on `claude/vibrant-cerf-sje726`, copied here as `TISSUE_DELAY/`.

What I changed in the copy:

- **`tissue.c`**, additions marked `TISSUE_DELAY`: a delay for each connection, with histories of outputs and of fast traces.
  - Under the rule TIMING, a connection compares the receiver's activity with the sender's activity **as it arrives**: the delayed output, and the sender's fast trace as it was D steps before. The trace of the delayed output is that delayed trace, and it is what the receiving end has. The file says so.
- **`tis.py`**: four fields appended to the structure.
- **`delay.py`** (new): one tissue and every reading.
- **`delay_tables.py`** (new) writes `DELAY_TABLES.md`, the summary and every row.
- **`q.sh`, `q4.sh`**: the runs.

All figures are from one run per setting. Nothing is confirmed independently.

## Terms

- **B**: `rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1`. **B1**: B with `we=1 wi=1/4`.
- **A delay of D** on a connection between cells: it carries the sender's output of 1 + D steps before; no delay means the step before. Connections from lines are not delayed.
- **The delay settings**:
  - random 0–32 or 0–16 on every connection between cells;
  - 32 on a quarter of them;
  - proportional to the distance of the two cells' coordinates (radius 1/8 = 32 steps);
  - *mix*: random 0–32 on every connection, then 32 on a quarter of them. I made this run after step 2, as the next run the rows pointed to.
- **Window**: the last 4,096 bytes of the run. For 65,536 bytes, the last 4,096 bytes of each quarter.
- **Readings**: every distance d, the pool of all non-feedback cells, and the cells near the output lines, at lags 0 to 4.
  - **(a)**: the readings tool as it is.
  - **(b)**: no floor, each cell divided by its own standard deviation (`where.py`'s method).
  - **Near cells** are reach.py's set: the 256 drivers and their neighbourhoods, 495 to 519 cells. They lie at d = 7 to 12 on every seed.
- **Levels**: lag 0: 0.1103. Lags 1 to 4: c0(L) + 0.02 of the window. That is the rule that gives the reviewer's 0.090 and 0.076 at lags 1 and 2 on seed 73129; it is my reading of how those levels were set. The lag-3 level is 0.17 to 0.19, because the stream makes the byte three back partly predictable from the present byte.
- **Divergence**: two copies, the second with its most active cell's potential changed by 1e-9 at byte NB − 4,608; the largest output difference 64 and 512 bytes later. **It is 0 in all 36 runs.**

## Answer to the question

**Is there a setting in which, on three seeds, the tool as it is reads the present byte at d = 1, 2, 4, and the bytes one to three back are above their levels somewhere in the tissue and among the cells near the output lines?**

**No setting tried meets all of it.** The parts:

- **Present byte at d = 1, 2, 4 by the tool, on all three seeds**: met by every setting except delays proportional to distance. In that setting S0 at d = 4 falls to 0.072 (B, 73129) and 0.097 (B1, 116089).
- **Bytes one to three back above their levels somewhere in the tissue, on three seeds**: met by B and B1 with random 0–32 and with delays proportional to distance (by both readings), and by mix. With random 0–16 and with 32 on a quarter, the byte three back is above its level somewhere on at most two seeds.
- **Among the near cells**: no run has the byte three back above its level. The byte one back is above its level among near cells in two settings:
  - B, 32 on a quarter, on all three seeds, by (b);
  - B1, 32 on a quarter, on seeds 73129 and 93683, by both readings.
  
  The byte two back is above its level among near cells in B and B1 with 32 on a quarter (three seeds, by (b)), and in some runs with random delays.
- **The closest setting is B with 32 on a quarter**, on all three seeds:

| Seed | S0 by (a) at d = 1 / 2 / 4 | Bytes one / two back above level among near cells by (b) | Byte three back among near cells |
|---|---|---|---|
| 73129 | 0.927 / 0.720 / 0.410 | 0.118 / 0.118 | 0.090 |
| 93683 | 0.974 / 0.695 / 0.382 | 0.171 / 0.154 | 0.172, level 0.173 |
| 116089 | 0.984 / 0.706 / 0.238 | 0.146 / 0.121 | 0.087 |

  But in this setting the byte three back is above its level somewhere in the tissue on seed 93683 only.

## Step 2: the rows (last window, 16,384 bytes)

Each entry gives (a) S0 at d = 1 / 2 / 4; then, by (b), the best over distances at lags 1 / 2 / 3; then the near cells at lags 1 / 2 / 3. Seeds in the order 73129, 93683, 116089.

- **B, no delay**: S0 0.929/0.722/0.379, 0.975/0.782/0.482, 0.981/0.694/0.471. Best at lags 1 and 2 is 0.061 to 0.076 and 0.048 to 0.054. **No byte before the present one is carried.** About 5 connections a cell.
- **B, random 0–32**: S0 0.927/0.683/0.196, 0.973/0.655/0.193, 0.983/0.750/0.165.
  - best: 0.546/0.316/0.262, 0.494/0.342/0.250, 0.483/0.346/0.281
  - near: 0.065/0.086/0.116, 0.038/0.045/0.095, 0.068/0.081/0.120
- **B, random 0–16**: S0 0.928/0.743/0.350, 0.975/0.762/0.346, 0.984/0.726/0.343.
  - best: 0.396/0.188/0.155, 0.382/0.201/0.159, 0.394/0.187/0.157
  - near: 0.107/0.167/0.133, 0.073/0.154/0.147, 0.076/0.168/0.135
- **B, 32 on 1/4**: S0 0.927/0.720/0.410, 0.974/0.695/0.382, 0.984/0.706/0.238.
  - best: 0.467/0.155/0.156, 0.438/0.279/0.192, 0.507/0.186/0.157
  - near: 0.118/0.118/0.090, 0.171/0.154/0.172, 0.146/0.121/0.087
- **B, proportional to distance**: S0 0.924/0.629/0.072, 0.973/0.632/0.110, 0.983/0.689/0.115.
  - best: 0.606/0.435/0.345, 0.475/0.416/0.351, 0.579/0.369/0.256
  - near: 0.040 to 0.068
- **B, mix**: S0 0.927/0.664/0.185, 0.977/0.608/0.124, 0.983/0.624/0.126.
  - best: 0.656/0.384/0.297, 0.581/0.403/0.301, 0.578/0.396/0.256
  - near: 0.044/0.045/0.077, 0.035/0.052/0.072, 0.092/0.089/0.100
- **B1** behaves like B with 2.9 to 3.4 connections a cell. With random 0–32 on seed 93683, the near cells reach the bytes one and two back: 0.129 and 0.141. Every B1 row is in `DELAY_TABLES.md`.

## Step 4: B with random 0–32 for 65,536 bytes, four windows

The four windows end at bytes 16,383, 32,767, 49,151 and 65,535.

| Seed | S0 at d = 1 / 2 / 4, range over the windows | Best at lags 1 / 2 / 3 | Near cells at lags 1 / 2 / 3 | Connections a cell |
|---|---|---|---|---|
| 73129 | 0.921–0.939 / 0.669–0.683 / 0.171–0.196 | 0.546–0.574 / 0.313–0.339 / 0.262–0.277 | 0.065–0.079 / 0.086–0.095 / 0.116–0.141 | 4.8 → 3.8 |
| 93683 | 0.973–0.979 / 0.655–0.690 / 0.193–0.221 | 0.493–0.503 / 0.339–0.386 / 0.255–0.271 | 0.036–0.060 / 0.045–0.081 / 0.093–0.121 | 5.2 → 4.4 |
| 116089 | 0.955–0.983 / 0.735–0.757 / 0.156–0.165 | 0.475–0.487 / 0.346–0.376 / 0.281–0.326 | 0.068–0.084 / 0.081–0.099 / 0.120–0.161 | 4.9 → 4.1 |

- The readings hold over 65,536 bytes: no trend of loss.
- Among the near cells, the byte two back is above its level in every window on seeds 73129 and 116089, and in the last two windows on seed 93683.
- The bytes one and three back stay below their levels among near cells.
- Divergence is 0 on all three seeds.

## What the numbers say

1. **In the timing tissue, delays bring the earlier bytes without making the tissue depend on its past.** With no delay, no earlier byte is carried. With random delays from 0 to 32 on every connection between cells, bytes one to three back are above their levels somewhere on all three seeds. Meanwhile the tool still reads the present byte at d = 1, 2 and 4.
2. **The near cells lie at d = 7 to 12 and carry less.** Delays that reach far (random 0–32, proportional to distance, mix) put the earlier bytes at the middle distances. A delay of 32 on a quarter of the connections serves the near cells best: bytes one and two back on three seeds. None of the 36 runs brings the byte three back to the near cells.

## Cost

36 runs took 4.05 CPU-hours (`delay/*.json`), with at most three processes of one thread each. Each run of 65,536 bytes took 1,523 to 1,653 CPU-seconds.
