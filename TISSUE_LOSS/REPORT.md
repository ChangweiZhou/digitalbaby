# Where, between the input lines and the cells four connections away, is the information about the present byte lost, and why (19 November 2026)

The reviewer asked for this in a message to E57's worker at 05:42 UTC on 19 November. The material is the reviewer's folder `reviewer_tissue_20261119` on `claude/vibrant-cerf-sje726` at 0b0b6f8e, copied here as `TISSUE_LOSS/`.

What I changed in the copy:

- **`tissue.c`**: added recording of the outputs after steps 4, 8, 16 and 32 of a byte; added the split of a cell's arrival by kind of source (`tis_run_phases`, `arrival_parts`); and added one switch for the change tried at the end (`tis_set_no_theta`).
- **`drive.py`, `deps/` and `out/` (the reviewer's results)** are unchanged.
- **`measure.py`** (new) builds each tissue exactly as `drive.py` does and makes measurements 1 to 5.
- **`loss_tables.py`** (new) writes `LOSS_TABLES.md`, which holds every figure below.

All figures are from one seed (73129), one run per setting, window A2 (bytes 12,288 to 16,383). Nothing is confirmed independently.

## Terms and method

- **The three tissues**:
  - **U**: unchanging, no gain, a* = 1/8 (default).
  - **G**: unchanging, gain on, a* = 1/32.
  - **C**: coincidence rule, gain on, a* = 1/32, floor 1/128.
- **d** is the distance from the lines of the bytes present in A2, along the connections present at the end of the window, as `drive.py` defines it.
- **Output** is the raw output at the reading step (step 32 of a byte) unless a step is named. Nothing below 0.01 is set to zero here, except in the scores S.
- **η²** of a cell is its between-byte variance divided by its total variance, over the 4,096 positions. It is computed for the present byte and for the previous byte; I report the median and the upper decile (90th percentile) over the cells at each d.
- **S(d, L)** is the score of the brief's section 7.3, computed by the reviewer's `readings.py`. These runs reproduce the reviewer's S(d, 0):

| Tissue | S(1,0) | S(2,0) | S(4,0) |
|---|---|---|---|
| U | 0.979 | 0.826 | 0.022 |
| G | 0.731 | 0.145 | 0.038 |
| C | 0.764 | 0.126 | 0.039 |

## Measurements (full tables in `LOSS_TABLES.md`)

**1 and 2. Mean output, share of positions above 0.01, and η² at d = 1 … 8**

| | d = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| **U** cells | 102 | 215 | 240 | 220 | 199 | 191 | 178 | 173 |
| U mean output | 0.0046 | 0.0009 | 0.0002 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| U share > 0.01 | 0.059 | 0.034 | 0 | 0 | 0 | 0 | 0 | 0 |
| U η² present, median | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.992 | 0.964 |
| U η² previous, median | 0.038 | 0.037 | 0.033 | 0.030 | 0.031 | 0.031 | 0.038 | 0.062 |
| **G** mean output | 0.029 | 0.030 | 0.031 | 0.031 | 0.032 | 0.031 | 0.030 | 0.030 |
| G share > 0.01 | 0.081 | 0.067 | 0.057 | 0.057 | 0.054 | 0.049 | 0.048 | 0.046 |
| G η² present, median (upper decile) | 0.228 (0.624) | 0.090 (0.154) | 0.056 (0.112) | 0.038 (0.112) | 0.026 (0.065) | 0.019 (0.029) | 0.015 (0.022) | 0.012 (0.018) |
| G η² previous, median | 0.020 | 0.016 | 0.016 | 0.016 | 0.018 | 0.020 | 0.021 | 0.019 |
| **C** cells | 237 | 121 | 229 | 234 | 217 | 208 | 204 | 202 |
| C mean output | 0.029 | 0.031 | 0.031 | 0.031 | 0.031 | 0.031 | 0.031 | 0.031 |
| C share > 0.01 | 0.071 | 0.055 | 0.052 | 0.050 | 0.050 | 0.050 | 0.050 | 0.045 |
| C η² present, median (upper decile) | 0.609 (0.767) | 0.060 (0.460) | 0.023 (0.206) | 0.017 (0.057) | 0.013 (0.028) | 0.012 (0.016) | 0.012 (0.016) | 0.011 (0.014) |
| C η² previous, median | 0.030 | 0.014 | 0.013 | 0.013 | 0.012 | 0.012 | 0.012 | 0.011 |

**3. η² of the present byte, median, read after steps 4 / 8 / 16 / 32 of the byte**

| d | U | G | C |
|---|---|---|---|
| 1 | 0.993 / 1.000 / 1.000 / 1.000 | 0.219 / 0.290 / 0.309 / 0.228 | 0.410 / 0.489 / 0.603 / 0.609 |
| 2 | 0.742 / 0.992 / 1.000 / 1.000 | 0.016 / 0.025 / 0.059 / 0.090 | 0.019 / 0.033 / 0.049 / 0.060 |
| 4 | 0.043 / 0.282 / 0.969 / 1.000 | 0.014 / 0.015 / 0.020 / 0.038 | 0.014 / 0.015 / 0.017 / 0.017 |
| 8 | 0.048 / 0.048 / 0.087 / 0.964 | 0.009 / 0.009 / 0.009 / 0.012 | 0.011 / 0.011 / 0.011 / 0.011 |

In U, the previous byte's η² at d = 4 is 0.999 / 0.756 / 0.067 / 0.030 over the same steps. So the present byte reaches distance d with a delay of a few steps for each connection, and replaces the previous byte there.

**4. The same tissue run twice from byte 8,192, the second copy with v changed by 1e-9**

Two variants were run:

- **One cell**: the most active cell at d = 1. In U the outputs differ by 5.6e-15 after 1 byte and by 0 from 16 bytes on. In G and C the difference is 0 at every checkpoint: that cell's change never reached its output.
- **Every cell** (every cell that is not a feedback cell). Largest output difference after n bytes, with the number of cells that differ:

| n | U | G | C |
|---|---|---|---|
| 1 | 4.8e-11 (1,772 cells) | 4.0e-12 (4 cells) | 1.9e-07 (90 cells) |
| 4 | 7.8e-15 | 1.8e-43 | 1.0 (112 cells) |
| 16 | 0 | 9.1e-15 | 0.75 |
| 64 | 0 | 4.2e-07 (620 cells) | 1.0 (317 cells) |
| 256 | 0 | 0.088 (8 cells) | 0.84 (41 cells) |

After 256 bytes, the internal quantities have also come apart:

| Tissue | Largest gain difference | Largest threshold difference |
|---|---|---|
| G | 660 | 0.41 |
| C | 2,200 | 1.6 (strengths differ by up to 1.0) |

**5. Share of the absolute arrival at the reading step, mean over the cells at d**

Each entry gives the share from nearer / same-distance / farther cells. No cell at d ≥ 2 has an input line.

| d | U | G | C |
|---|---|---|---|
| 2 | 0.686 / 0.247 / 0.067 | 0.277 / 0.384 / 0.340 | 0.174 / 0.321 / 0.505 |
| 3 | 0.633 / 0.309 / 0.058 | 0.322 / 0.448 / 0.230 | 0.269 / 0.497 / 0.235 |
| 4 | 0.590 / 0.337 / 0.073 | 0.311 / 0.439 / 0.250 | 0.268 / 0.516 / 0.216 |

**Also measured, in G at the end of the run:**

- **Thresholds**: the cells with a threshold above 0 are 100 of 102 at d = 1, 215 of 215 at d = 2, 218 of 220 at d = 4, and 172 of 173 at d = 8.
- **Gains**: the median gain at d = 1 to 8 is 2.3 to 4.1, and the upper decile is 5.4 to 14.2. One to three cells at each d sit at the cap of 4,096.

## Which of (i) to (iv) accounts for the loss, from these numbers

- **(iii), the control removes the byte's part within the 32 steps of a byte: not supported.** In G and C, η² at d ≥ 2 rises through the byte (G at d = 2: 0.016 → 0.090) rather than falling. The one exception is G at d = 1, which falls from 0.309 at step 16 to 0.228 at step 32.
- **In U the byte is not lost.**
  - η² of the present byte is about 1.0 at every distance up to 7, and 0.964 at d = 8.
  - What falls is the amplitude: the mean output drops about fivefold per connection (0.0046, 0.0009, 0.0002, 0.0001), so from d = 3 no position is above 0.01.
  - S(4, 0) is at chance only because the readings set outputs below 0.01 to zero. This is (iv): a loss in the reading, not in the tissue.
  - U also forgets a change of its state completely within 16 bytes, so its activity is a function of the stream.
- **(i), the tissue's own activity is added at each connection, and (ii), the activity depends on the tissue's past:** both hold in G and C.
  - Under gain, 68 to 83% of the arrival at d = 2 to 4 comes from same-distance and farther cells, against 31 to 41% in U. Those cells' η² is close to the previous-byte level.
  - G's and C's state depends on their past: C's within 4 bytes, G's within 64 to 256.
- **But (i) and (ii) do not account for G's loss.** The change below removes most of G's loss while leaving (i) unchanged and making (ii) stronger.
- **What the numbers point to in G is (iv), the threshold under gain.**
  - In G, nearly every cell's threshold has risen above 0. The gain and the threshold both integrate the same error (a* minus x), so the pair can drift along a line that leaves the output unchanged. Item 4's gain and threshold differences, with almost no output differences, show such a drift.
  - With the threshold above 0, a cell's output is only the part of its potential above the threshold. Positions above 0.01 fall to 5 to 8%, and the graded, byte-dependent part of the potential below the threshold is cut off.

## The one change: the threshold held at 0, so a cell with the gain on controls its activity by its gain alone

This uses one quantity of the cell, its threshold, which is no longer adapted. Measured S(d, L) for d = 1, 2, 4 and L = 0, 1, 2:

| Tissue | Change | S(1,0) | S(1,1) | S(1,2) | S(2,0) | S(2,1) | S(2,2) | S(4,0) | S(4,1) | S(4,2) |
|---|---|---|---|---|---|---|---|---|---|---|
| U | without | 0.979 | 0.066 | 0.053 | 0.826 | 0.056 | 0.042 | 0.022 | 0.022 | 0.022 |
| U | with | identical (U's thresholds stay at 0 without the change) | | | | | | | | |
| G | without | 0.731 | 0.045 | 0.036 | 0.145 | 0.023 | 0.026 | 0.038 | 0.022 | 0.029 |
| G | **with** | **0.982** | 0.072 | 0.046 | **0.774** | 0.054 | 0.032 | **0.257** | 0.061 | 0.036 |
| C | without | 0.764 | 0.057 | 0.042 | 0.126 | 0.033 | 0.029 | 0.039 | 0.027 | 0.025 |
| C | with | 0.181 | 0.029 | 0.021 | 0.032 | 0.022 | 0.019 | 0.021 | 0.023 | 0.019 |

**G with the change:**

- η² of the present byte, median at d = 1 / 2 / 4 / 8: 0.931 / 0.770 / 0.302 / 0.014, against 0.228 / 0.090 / 0.038 / 0.012.
- Mean output is still about 0.030 at every d.
- The share of positions above 0.01 is now 0.45 to 0.78 at d = 1, 2, 4 and 8, against 0.05 to 0.08.
- The arrival split is unchanged: 28% from nearer cells at d = 2.
- The tissue depends on its past more than before. With every cell nudged, outputs differ by 0.13 after 16 bytes and 1.0 after 256.
- The byte is still lost by d = 8 (η² 0.014).

**C with the change is worse.** Making a connection requires a threshold of 0, so holding the threshold at 0 also changes the making of connections. C's connections rise from 19.4 to 79.9 a cell, and the tissue is a different one. The change is not separated from that effect here.

## Cost and files

- **Cost**: the six final runs of `measure.py` took 819 CPU-seconds (0.23 CPU-hours; `out/measure_*.json`).
- **Earlier runs**: the three tissues were also measured twice before. Item 4 first nudged a cell that went silent at the next step, and then was extended with the slow quantities. Those runs' costs were not kept separately; they were of the same size, about 0.2 CPU-hours each time.
- **Processes**: at most three, with one thread each.

`TISSUE_LOSS/`: `REPORT.md`, `LOSS_TABLES.md`, `measure.py`, `loss_tables.py`, `tissue.c` (the reviewer's, with additions marked), the reviewer's `drive.py`, `cmp_fixed.py`, `deps/`, `batch*.txt` and `out/` (the reviewer's results), `out/measure_*.json` (these runs), and `logs/`.
