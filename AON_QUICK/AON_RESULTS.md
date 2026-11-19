# The all-or-none test (seed 73129 unless stated, 16,384 bytes, window A2 = the last 4,096 bytes)

**Source.** This folder copies `reviewer_tissue_20261119` from the reviewer's branch at 539f92bd. Two files are changed and three added:
- `tissue.c` gains three options:
  - `allornone=1`: x = 1 when v − θ > 0, else 0;
  - `eta_t=..`: η_θ of formula 9;
  - `tauv1=1`: v = A.
- `tis.py` gains two options:
  - `noinh=1`: no inhibitory cell;
  - `theta0=..`: the starting threshold of every ordinary cell.
- `drive.py` is changed only so that it passes the new fields.
- `aon.py` is new. It is diag.py's readings plus:
  - the share of positions at which a cell is active, by distance;
  - S(d, 0..2) at d = 1, 2, 4 and the three largest distances;
  - divergence as a count of cells. Two copies are compared after a change of 1e-9 in one cell's v, and after one active cell's output is set to 0 for one step.
- The other new files are `logs/` and `aon/*.json`.

**Check that nothing else changed.** With `allornone=0`, `diag.py fixed rule=fixed` reproduces the reviewer's `diag/fixed.json` exactly.

**Reading the tables.** "present" and "previous" are the medians, over the cells at a distance, of the share of a cell's raw output variance that the present byte (or the previous byte) explains. Chance for S is 1/46 = 0.0217, and C0(0) = 1.000.

## Item 1: rule=fixed, all-or-none, no gain

| a* | Cells sending a step | ms a byte | Active share, d = 1 / 4 / 8 / 11 | Present, d = 1 / 2 / 4 / 8 / 11 | Previous, d = 1 / 8 | S(d,0), d = 1 / 2 / 4 / 9–11 | Cells differing after 1/16/64/256/512 bytes (both perturbations) |
|---|---|---|---|---|---|---|---|
| 1/8 | 243 | 1.06 | 0.112 / 0.123 / 0.121 / 0.125 | 0.014 / 0.012 / 0.012 / 0.011 / 0.009 | 0.017 / 0.009 | 0.066 / 0.028 / 0.027 / 0.024–0.027 | 0 at every count |
| 1/32 | 61 | 1.02 | 0.024 / 0.031 / 0.031 / 0.032 | 0.019 / 0.021 / 0.016 / 0.012 / 0.011 | 0.015 / 0.012 | 0.030 / 0.027 / 0.026 / 0.020–0.026 | 0 |
| 1/128 | 15 | 1.05 | 0.005 / 0.008 / 0.007 / 0.008 | 0.013 / 0.015 / 0.014 / 0.011 / 0.013 | 0.012 / 0.011 | 0.034 / 0.033 / 0.030 / 0.026–0.027 | 0 |

S(d,1) and S(d,2) lie between 0.019 and 0.031 in every row.

## Item 2 (activity did not die out with distance; run as the next obvious test): a* = 1/32 unless stated

| Setting | Cells a step | Active share, d = 1 / 8 | Present, d = 1 / 2 / 4 / 8 | S(d,0), d = 1 / 2 / 4 | Divergence (one output set to 0) |
|---|---|---|---|---|---|
| η_θ = 1/32 | 61 | 0.018 / 0.033 | 0.012 / 0.012 / 0.011 / 0.010 | 0.023 / 0.022 / 0.021 | 8, 0, 33, 59, 0 |
| η_θ = 1/8 | 61 | 0.028 / 0.032 | 0.011 / 0.010 / 0.011 / 0.011 | 0.030 / 0.018 / 0.019 | 0 |
| v = A | 60 | 0.015 / 0.033 | 0.020 / 0.025 / 0.034 / 0.019 | 0.037 / 0.029 / 0.029 | 0 |
| v = A, a* = 1/8 | 237 | 0.094 / 0.123 | 0.029 / 0.028 / 0.030 / 0.026 | 0.077 / 0.036 / 0.030 | 0 |

## A threshold held fixed (η_θ = 0, θ = theta0; not asked; run because item 2 left the threshold control as the open cause)

| Setting | Cells a step | Active share, d = 1 / 2 / 4 / 8 | Always active, d = 1 / 4 | Present, d = 1 / 2 | S(d,0), d = 1 / 2 / 4 |
|---|---|---|---|---|---|
| θ = 1/16 (one net input of 1/8 fires) | 1,835 | 0.806 / 0.871 / 0.955 / 0.942 | 80 of 102 / 210 of 220 | 1.000 / 0.435 | 0.198 / 0.085 / 0.022 |
| θ = 1/16, v = A | 1,834 | the same | 81 / 210 | 1.000 / 1.000 | 0.198 / 0.091 / 0.022 |
| θ = 3/16 (two net inputs needed) | 0 | 0 everywhere | 0 | — | 0.022 everywhere |
| θ = 3/16, v = A | 0 | 0 everywhere | 0 | — | 0.022 everywhere |

## Item 4: v = A, a* = 1/8 (the homeostatic setting with the highest S(1,0) above)

| Tissue | Cells a step | ms a byte | Connections a cell | Present, d = 1 / 2 / 4 / 8 | S(d,0), d = 1 / 2 / 4 / largest three | Divergence (one output set to 0) |
|---|---|---|---|---|---|---|
| fixed, seed 73129 | 237 | 0.98 | 7.8 | 0.029 / 0.028 / 0.030 / 0.026 | 0.077 / 0.036 / 0.030 / 0.019–0.025 | 0 |
| fixed, seed 93683 | 240 | 1.08 | 7.8 | 0.021 / 0.018 / 0.018 / 0.021 | 0.046 / 0.029 / 0.026 / 0.020–0.025 | 0 |
| fixed, seed 116089 | 240 | 1.11 | 7.8 | 0.018 / 0.016 / 0.019 / 0.022 | 0.062 / 0.027 / 0.028 / 0.024–0.030 | 0 |
| coincidence, 73129 | 245 | 3.82 | 10.1 | 0.014 / 0.011 / 0.011 / 0.011 | 0.048 / 0.024 / 0.018 / 0.021–0.023 (d = 13–15) | 330, 675, 428, 196, 707 |
| timing, 73129 | 184 | 15.78 | 27.6 | 0.032 / 0.012 / 0.011 / 0.011 | 0.070 / 0.027 / 0.023 / 0.017–0.022 (d = 11–13) | 0, 386, 337, 271, 294 |

## What the runs measured, against the question

The question is whether an all-or-none tissue carries the present and the previous byte to cells 8 to 11 connections from the input lines, at full amplitude, without depending on its own past. **In no setting run does it.**

- **Under the threshold control (formula 9):**
  - Every distance from 1 to 11 is active at about a*, at full amplitude (x = 1).
  - The tissue of rule=fixed does not depend on its own past: no cell differs after either perturbation, in 8 of the 9 such runs (the exception is η_θ = 1/32: up to 59 cells differ, and none after 512 bytes).
  - But the active cells do not carry the byte, even at one connection from the lines. The present byte explains at most 0.034 of a cell's variance (median), and S(1,0) is at most 0.077 against C0 = 1.000.
  - What a cell fires on is therefore set mostly by the cycle of its own threshold, not by what arrives.
- **With the threshold held fixed:**
  - At θ = 1/16, cells at one connection carry the present byte (it explains 1.000). But activity fills the tissue and locks on: most cells beyond two connections are always active, and S(4,0) is at chance.
  - At θ = 3/16, which is above one connection's strength of 1/8, nothing is ever active.
  - With every strength at 1/8, the threshold either passes any single input or needs two, and the measured tissue either fills or stays silent.
- **The coincidence and timing rules** make the all-or-none tissue depend on its own past: 196 to 707 cells and 271 to 386 cells differ. They carry nothing beyond d = 1, where S(1,0) is 0.048 and 0.070.
