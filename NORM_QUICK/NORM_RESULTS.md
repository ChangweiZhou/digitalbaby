# Conserved total incoming strength with the timing rule: replication and extension (16,384 bytes unless stated)

**Source.** This folder copies `reviewer_tissue_20261119` from the reviewer's branch at 476afc59, unchanged. `where.py` and `readings_tool.py` are copied from the helper's `claude/prep-step1-full` at 4bda7aa1.

**The programs added here:**
- `ntd.py` runs one tissue and reads it at the end of each stretch of its life. The window is the last 4,096 bytes before each end. The readings are:
  - **Divergence:** a second tissue is made equal to the first in every state by a memory copy, 4,608 bytes before the end. One cell's potential is then changed by 1e-9, as diag.py does. The run counts the cells whose output differs after 1, 16, 64, 256 and 512 bytes.
  - **By distance**, for d = 1 to the largest, and for the pool of all cells:
    - the number of cells;
    - the mean and largest output;
    - the share of outputs above 0.01;
    - the median share of variance explained by the present byte and by the previous byte;
    - S(d, lag) for lags 0 to 4, (a) by the tool as it is and (b) with no floor and each cell divided by its own standard deviation, as `where.py` does;
    - C0.
  - **Connections:** the number per cell (mean, smallest, largest); how many connections hold more than half of their cell's excitatory total; how many were given up and made.
- `summ.py` prints compact rows from those files.
- The results are in `ntd/*.json` and `logs/`.

**The tissue.** Every run uses `rule=timing norm=1 lines_learn=1` with no gain control. Item 1 adds `no_make=1`.

**Check against the reviewer.** Seed 73129, eta_s 1, reproduces the reviewer's S(d,0) of 0.929 / 0.722 / 0.379 exactly. The reviewer's 4.76 connections a cell is taken over all 2,048 cells; this file reports 4.92 over the 1,984 ordinary cells.

## Item 1: three seeds, window A2

Each cell gives the range over seeds 73129, 93683 and 116089.

| eta_s | we / wi | S(d,0) tool, d = 1 / 2 / 4 | S(d,0) d = 8–11: tool / no floor | Largest output, d = 8 / d = 11 | Connections a cell | Cells differing after 16–512 bytes |
|---|---|---|---|---|---|---|
| 1/8 | 3/4 / 1/4 | 0.93–0.96 / 0.68–0.77 / 0.37–0.56 | 0.022–0.053 / 0.23–0.53 | 0.004–0.015 / 0.000 | 6.2–6.6 | 0 |
| 1/4 | 3/4 / 1/4 | 0.92–0.98 / 0.63–0.79 / 0.38–0.50 | ≤ 0.099 / 0.21–0.39 | 0.012–0.045 / 0.000 | 5.7–6.3 | 0 |
| 1/2 | 3/4 / 1/4 | 0.93–0.98 / 0.74–0.81 / 0.45–0.52 | ≤ 0.176 / 0.24–0.41 | 0.050–0.062 / ≤ 0.002 | 5.2–5.7 | 0 |
| 1 | 3/4 / 1/4 | 0.93–0.98 / 0.69–0.78 / 0.38–0.48 | 0.020–0.207 / 0.17–0.28 | 0.065–0.076 / 0.003–0.010 | 4.9–5.2 | 0 |
| 2 | 3/4 / 1/4 | 0.93–0.98 / 0.66–0.81 / 0.42–0.52 | 0.035–0.210 / 0.17–0.26 | 0.072–0.079 / 0.021–0.023 | 4.0–4.7 | 0 |
| 8 | 3/4 / 1/4 | 0.81–0.84 / 0.53–0.56 / 0.28–0.32 | 0.111–0.173 / 0.12–0.18 | 0.070–0.079 / 0.024–0.030 | 3.6–3.8 | 0 |
| 1 | 7/8 / 1/8 | 0.93–0.97 / 0.69–0.71 / 0.40–0.41 | 0.074–0.124 / 0.09–0.13 | 0.21–0.22 / 0.12–0.13 | 3.5–3.6 | 0 |

At lags 1 and 2, S(d, lag) with no floor is at or below C0(lag) + 0.02 at every distance in every row. A single run (eta_s 1, seed 116089) shows one differing cell after one byte.

## Runs opened by item 1 (not asked: a larger excitatory total kept more amplitude far out)

| eta_s | we / wi | Seeds | S(d,0) tool, d = 1 / 4 / 8 / 11 | Largest output, d = 8 / d = 11 | Present byte, d = 8 / 11 (median share of variance) | Connections a cell | Cells differing |
|---|---|---|---|---|---|---|---|
| 1 | 15/16 / 1/16 | three | 0.94–0.98 / 0.32–0.37 / 0.10–0.14 / 0.08–0.11 | 0.35–0.36 / 0.22–0.24 | 0.63–0.97 / 0.64–0.97 | 3.3–3.6 | 0 |
| 1 | 1 / 0 | three | 0.91–0.95 / 0.28–0.34 / 0.12–0.19 / 0.08–0.11 | **0.57–0.58 / 0.42–0.46** | 0.71–0.97 / 0.64–1.00 | 1.6–1.8 | 0 |
| 8 | 7/8 / 1/8 | three | 0.36–0.42 / 0.10–0.13 / 0.06–0.07 / 0.05 | 0.21–0.23 / 0.12–0.13 | 0.27–0.68 / 0.25–0.41 | 3.4–3.5 | 0 (one cell after one byte, one seed) |

## Item 3: the life to 262,144 bytes (eta_s 8, the highest S(d,0) at d = 8 to 11 in item 1)

| Seed | End of window (byte) | S(d,0) tool, d = 1 / 2 / 4 / 8 / 11 | Largest output, d = 8 / 11 | Connections a cell | Given up so far | Cells differing |
|---|---|---|---|---|---|---|
| 73129 | 16,384 | 0.813 / 0.539 / 0.279 / 0.152 / 0.131 | 0.079 / 0.030 | 3.57 | 8,787 | 0 |
| 73129 | 65,536 | 0.809 / 0.500 / 0.256 / 0.141 / 0.131 | 0.079 / 0.031 | 3.38 | 9,161 | 0 |
| 73129 | 262,144 | 0.805 / 0.509 / 0.284 / 0.156 / 0.132 | 0.079 / 0.031 | 3.34 | 9,240 | 0 |
| 93683 | 16,384 | 0.822 / 0.527 / 0.315 / 0.173 / 0.135 | 0.073 / 0.024 | 3.84 | 8,252 | 0 |
| 93683 | 65,536 | 0.799 / 0.516 / 0.319 / 0.194 / 0.153 | 0.079 / 0.030 | 3.55 | 8,827 | 0 |
| 93683 | 262,144 | 0.790 / 0.493 / 0.305 / 0.160 / 0.157 | 0.073 / 0.030 | 3.49 | 8,945 | 0 |
| 116089 | 16,384 | 0.844 / 0.561 / 0.287 / 0.122 / 0.111 | 0.070 / 0.030 | 3.62 | 8,699 | 0 |
| 116089 | 65,536 | 0.828 / 0.569 / 0.298 / 0.142 / 0.129 | 0.069 / 0.030 | 3.40 | 9,120 | 0 |
| 116089 | 262,144 | 0.828 / 0.556 / 0.298 / 0.126 / 0.115 | 0.070 / 0.030 | 3.37 | 9,180 | 0 |

The tissue settles. Between bytes 16,384 and 262,144:
- S(d,0) changes by at most 0.04 at any distance;
- connections fall by 0.2 to 0.35 a cell;
- at most 693 more connections are given up;
- no cell ever differs between the two copies.

Each life took 5.0 to 5.2 ms a byte.

## Item 4: connections made (no_make = 0), seed 73129

Two settings were run: eta_s 1 and eta_s 8. Both use we / wi = 3/4 / 1/4 and windows ending at bytes 8,192, 12,288 and 16,384.

| eta_s | End of window | Connections a cell (smallest–largest) | Made / given up so far | S(d,0) tool, d = 1 / 2 / 4 / 8 | Largest output, d = 8 | Cells differing after 1 / 16 / 64 / 256 / 512 bytes | ms a byte |
|---|---|---|---|---|---|---|---|
| 1 | 8,192 | 27.4 (8–81) | 4,007,762 / 3,969,297 | 0.290 / 0.157 / 0.075 / 0.022 | 0.000 | 0 / 0 / 1 / 1,755 / 1,776 | 25.8 |
| 1 | 12,288 | 27.8 (9–80) | 6,262,640 / 6,223,444 | 0.296 / 0.166 / 0.067 / 0.022 | 0.000 | 1 / 0 / 1,349 / 1,814 / 1,757 | |
| 1 | 16,384 | 27.3 (9–79) | 8,518,548 / 8,480,332 | 0.281 / 0.152 / 0.075 / 0.022 | 0.000 | 1 / 0 / 32 / 1,798 / 1,777 | |
| 8 | 8,192 | 23.7 (8–79) | 4,005,543 / 3,974,438 | 0.235 / 0.108 / 0.080 / 0.022 | 0.006 | 0 / 0 / 132 / 1,606 / 1,681 | 22.1 |
| 8 | 12,288 | 23.5 (8–79) | 6,259,078 / 6,228,254 | 0.228 / 0.106 / 0.071 / 0.022 | 0.003 | 0 / 16 / 1,063 / 1,672 / 1,747 | |
| 8 | 16,384 | 23.4 (8–78) | 8,512,607 / 8,482,026 | 0.227 / 0.112 / 0.075 / 0.022 | 0.004 | 0 / 1 / 32 / 1,638 / 1,725 | |

**Why it is slow.** The number of connections a cell does not grow over the life: it stays at 23 to 28 from byte 8,192 to byte 16,384. Instead, connections turn over. About 520 are made and about 520 given up per byte, which is about one a cell every 128 steps.

- Every cell below its target makes a connection every P_s steps.
- After the normalisation, a new connection's strength falls below 1/128, and the connection is given up when it reaches age P_s.

The cost is 22 to 26 ms a byte, against 5 to 7 with no makings. It comes from about 5 times as many connections a cell (23 to 28, against 3.3 to 6.6) and from the making and giving up at every period.

**What making connections does.** With connections made, the tissue depends on its own past: 1,600 to 1,800 of 1,984 cells differ after 256 to 512 bytes. The present byte reaches d = 4 at most (S(4,0) 0.07 to 0.08). Beyond that, the largest output at d = 8 is at most 0.006.

## Answer to the question, as measured

Does conserving a cell's total incoming strength, with the timing rule, carry the present byte at usable amplitude to the cells near the output lines (d = 8 to 11), on three seeds, without the tissue depending on its past? And at how many connections a cell?

- **Without depending on its past:** yes. With no connections made, two copies stay identical after 16 to 512 bytes in every run, on all three seeds, through 262,144 bytes.
- **The present byte at d = 8 to 11:** carried weakly. S(d,0) is at most 0.21 by the tool, and 0.11 to 0.17 at eta_s 8 on every seed, against 0.022 chance and C0 = 1.000. The share of a cell's variance that the present byte explains there is high (median 0.54 to 1.00). So the far cells vary with the byte, but each separates few byte values.
- **Usable amplitude:** with we / wi = 3/4 / 1/4, no. The largest output is 0.07 to 0.08 at d = 8 and 0.03 at d = 11, where emission needs more than 1/2.
  - With we = 1 and wi = 0 (no inhibition), the largest output is 0.57 at d = 8 and 0.42 to 0.46 at d = 11. That is the only setting near the emission level, and it holds 1.6 to 1.8 connections a cell.
- **Connections a cell:** 3.3 to 6.6 with no connections made, depending on eta_s and we; 1.6 to 1.8 at we = 1.
- **With connections made:** 23 to 28 a cell, in constant turnover. The tissue then depends on its past (1,600 to 1,800 cells differ), and the byte does not pass d = 4.
- **The previous byte:** not carried above C0 + 0.02 anywhere.
