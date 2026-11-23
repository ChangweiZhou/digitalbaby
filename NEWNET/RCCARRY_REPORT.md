# RCCARRY: do rule cells form the carry? (23 November 2026)

Helper branch `claude/newnet-rccarry-sei232` (from `claude/magical-albattani-sei232` @ d0f0a9da). Development level, one machine,
OMP_NUM_THREADS=1, **seed 1 only**. No gate, seal or outside review is claimed. Plan and criterion written and committed before the
runs: `RCCARRY_PLAN.md`. Code (new files only): `net13rcc.py` (net13 subclass; `rc_match` / `rc_pairs` widen only the rule cells'
input set; `rc_ro=1` the one variation), `t24_add_rcc.py` (t24_add.py importing the net by module name, plus a read-only carry
diagnostic). Check: net13rcc with defaults and t24_add_rcc net=net13 are byte-identical to `t24_add.py net=13` (300 lines, L=2).
Logs and results: `rccarry_runs/` (`small.jsonl`, `full.jsonl`, `*.log`, `jobs.txt`, `run.sh`, `summ.py`).

Task (t24): fmt=col writes each column as `a b s` (carry not written); colc as `c a b s`. Whole lines right in phase N1 (prequential,
learning on; every pair new), old length (1-4 digits) / new length (5-8). Free: last free test, learning off, own answers fed back,
8 digits. s bytes: right share of s bytes at new lengths with carry-in 0 / 1.

## Full size (struct K8, L=16, N0 = N1 = 8000, seed 1)

| run | fmt | lines old / **new** | free 8 | s bytes new, carry 0 / 1 | time |
|---|---|---|---|---|---|
| storage (ADD_REPORT.md) | col | 0.799 / 0.376 | 0.30 | - | - |
| net12 lagro=1 relay=1 (ADD_REPORT.md) | col | 0.466 / 0.031 | 0.00 | - | 1628 s |
| **net13** (rc=64, k=3, tau=8, vig=0.4) | col | 0.462 / **0.031** | 0.00 | 0.734 / 0.501 | 1294 s |
| net13, rule cells also read match cells (rc_match=4) | col | 0.467 / **0.033** | 0.00 | 0.738 / 0.508 | 1302 s |
| net13, rule cells also read pair cells (rc_pairs=32768) | col | PAIRS_FULL |
| storage (ADD_REPORT.md) | colc | 0.967 / 0.867 | 0.90 | - | - |
| net12 lagro=1 relay=1 (ADD_REPORT.md) | colc | 0.919 / 0.739 | 0.48 | - | 1831 s |
| **net13** | colc | 0.926 / **0.759** | 0.50 | 0.988 / 0.951 | 1668 s |

## Small (struct K8, L=4, N0 = N1 = 4000, seed 1)

| run | fmt | lines old / new | free 8 | s bytes new, carry 0 / 1 |
|---|---|---|---|---|
| net12 lagro=1 relay=1 | col | 0.393 / 0.016 | 0.00 | 0.702 / 0.399 |
| net13 | col | 0.402 / 0.019 | 0.00 | 0.705 / 0.403 |
| net13 rc_match=4 | col | 0.408 / 0.015 | 0.00 | 0.710 / 0.405 |
| net13 rc_pairs=32768 | col | 0.404 / 0.014 | 0.00 | 0.706 / 0.401 |
| net13 rc_ro=1 (the variation) | col | 0.413 / 0.017 | 0.00 | 0.701 / 0.405 |
| net12 lagro=1 relay=1 | colc | 0.862 / 0.536 | 0.48 | 0.969 / 0.909 |
| net13 | colc | 0.889 / 0.621 | 0.54 | 0.986 / 0.913 |

## Rule cells against the true carry (phase N1, columns i >= 1; state at the moment s_i is predicted)

max |corr| = the largest |Pearson corr| of one rule cell's normalised activity with the carry into column i; ">0.2" = number of the 64
cells above 0.2; purity = the winner at s_(i-1) used as a carry code (each winner's majority on the first half, scored on the second
half) against the base rate; persist = share of s bytes at which the s_(i-1) winner is still active; linear probe = least squares from
the 64 activities to the carry, fit on the first half, scored on the second (a diagnostic, not part of the net). Every cell's corr is
in the jsonl (`carry_diag.*.all_corr`).

| run | n | max abs corr (P1 / P1 new) | >0.2 | purity / base | persist | linear probe | recruitments |
|---|---|---|---|---|---|---|---|
| full net13 col | 35,692 | 0.151 / 0.168 | 0 | 0.585 / 0.511 | 1.00 | 0.628 | 6,570 |
| full rc_match=4 col | 35,692 | 0.186 / 0.187 | 0 | 0.619 / 0.511 | 1.00 | 0.696 | 28,467 |
| full rc_pairs col | PAIRS_DIAG |
| small net13 col (rc_ro=1: identical cells) | 17,869 | 0.145 / 0.157 | 0 | 0.590 / 0.510 | 1.00 | 0.621 | 4,070 |
| small rc_match=4 col | 17,869 | 0.183 / 0.194 | 0 | 0.613 / 0.510 | 1.00 | 0.686 | 14,068 |
| small rc_pairs col | 17,869 | 0.082 / 0.069 | 0 | 0.519 / 0.510 | 1.00 | 0.580 | 55,357 |
| full net13 colc (carry written) | 35,692 | 0.665 / 0.638 | 20 | 0.554 / 0.511 | 0.015 | 0.997 | 49 |

Top cells, full net13 col (P1): cell 6 +0.151, 36 +0.126, 48 +0.120, 25 -0.113, 30 -0.099. In colc the carry is a byte in the window
(c_i at lag 2 when s_i is predicted), so the rule cells that see it correlate with it; there they do not form it, they read it.

## What it shows (seed 1)
1. **The hypothesis fails.** Full-size fmt=col new-length lines: net13 0.031, with match inputs 0.033, with pair inputs PAIRS_NEW;
   needed >= 0.30. Exactly the net12 value (0.031); storage 0.376. Carry-1 s bytes at new lengths stay at chance-like 0.50 against
   0.73 for carry 0.
2. **No rule cell is a carry cell.** The largest |corr| with the carry is 0.15-0.19 (full); no cell above 0.2 in phase N1. The
   s_(i-1) winner predicts the carry held-out at 0.59-0.62 against a base rate 0.51. A linear read of all 64 cells reaches
   0.63-0.70: some carry information is spread through the window-clusters, as it must be (the carry is a function of bytes in the
   window), but no cell stands for it.
3. **Persistence is not what is missing.** The cell that won when s_(i-1) was read is still active when s_i is predicted at every
   s byte (1.00): tau = 8 and k = 3 keep the last three winners on. The cells persist; they persist the wrong thing - a cluster of the
   8-byte window by overlap (vigilance 0.4: >= 4 shared (lag, byte) pairs), not the one bit that is a function of three bytes jointly.
   Wider inputs do not change this: match cells add a little (max corr 0.19, probe 0.70); pair cells (28 active of 36 inputs) dilute
   the clustering further (small: 0.07, purity at base rate) while recruiting 55k times.
4. **The variation (rc_ro=1, rule cells also feed the direct readout) changes nothing**: small new-length lines 0.017 against
   0.019; the cells are identical (they depend on the stream only). By the plan's rule (gain >= 0.02 needed), its full run is NOT_RUN.
5. **colc is not worse**: net13 new-length lines 0.759 against net12 0.739 (full), 0.621 against 0.536 (small); free test at 8 digits
   0.50 against 0.48. One seed.

## Limits / NOT_RUN
- Seed 1 only; seeds 2-3 not run because no arm passed (plan rule).
- rc_ro=1 at full size: NOT_RUN (failed the small gain rule). rc_match / rc_pairs with colc: not run.
- No other rule-cell settings (rc count, vigilance, tau, k) were tried; the plan allowed one variation and the check showed persistence
  was not the gap.
- The diagnostics (purity, linear probe) are fitted read-outs for description only; they are not claims about what the net uses.

## Verdict
Rule cells, as net13 has them (competitive clustering of the lag window, persistent for a column), **do not form a carry-like state and
do not generalise column addition to new lengths** when the carry is not written (0.031-0.033 against the 0.30 criterion; no cell with
|corr| > 0.2 to the carry). Widening only their input set (match or pair cells) does not change this, and neither does letting them
feed the direct readout. With the carry written, net13 is not worse than net12 (0.759 vs 0.739). The carry is not a context that
recurs as a pattern of the window; it is a computed bit, and a recruit-by-overlap state cannot find it. Learning such a hidden quantity
locally remains open.
