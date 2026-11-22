# Has a rule been learned? First encounters in one life (t22_rules.py, 22 November 2026)

Development level, one machine, seed 1 only; no gate, seal or outside review is claimed. Logs and results: `rules_runs/`.
(The net12 runs were started while the lag-cell design was still named net11.py; their result lines say `net 11`. It is `net12.py`.
The sheet net11 was restored.)

Stream: lines `w=W\n`, 3 symbols, copy (lines 0-12,000), rev (12,000-18,000), copy again (18,000-24,000), all with "=" (no cue);
93 symbols, the alphabet grows from 8 by one every 250 lines. Only answer bytes of items met for the first time under the rule in force
are scored, before they are read (learning on). Layers: a - storage could know it; b - the symbol has been an answer at least 5 times;
n - 1-4 times; c - never.

Right share (number of bytes) by phase and layer:

| | P0 copy b | P0 copy c | P1 rev a | P1 rev b | P1 rev c | P2 copy b | P2 copy c |
|---|---|---|---|---|---|---|---|
| storage (back-off n-gram, 8 bytes) | 0.05 | 0.05 | 0.74 | 0.00 | 0.00 | 0.07 | 0.00 |
| N1 (net10) | 0.18 | 0.00 | 0.71 | 0.06 | 0.00 | 0.13 | 0.00 |
| K8 (net10) | 0.16 | 0.00 | 0.70 | 0.03 | 0.00 | 0.12 | 0.00 |
| net12 K8 + lag cells, direct readout | 0.92 | 0.02 | 0.73 | 0.30 | 0.00 | 0.75 | 0.00 |
| net12 K8 + lag cells, direct readout + gated relay | **0.99** | **0.84** | 0.74 | **0.77** | **0.71** | **0.91** | **0.79** |

(bytes: P0 b 30,448, c 55; P1 a 8,396, b 9,401, c 24; P2 b 17,608, c 14; c is small.)

After a change of rule (right share in windows of 250 lines):

| | copy -> rev, first 12 windows | rev -> copy (return), first 12 windows |
|---|---|---|
| storage | 0.14 ... 0.37 | 0.08 ... 0.09 |
| K8 | 0.11 ... 0.38 | 0.11 ... 0.12 |
| net12 direct | 0.33 ... 0.53 | 0.58 ... 0.72 |
| net12 direct + relay | 0.34 (7 windows) then 0.48, 0.61, 0.80, 0.89, 0.95 | 0.64, 0.67, ... 0.98, 1.00, 1.00 |

What it shows (seed 1):
1. net10 (K8, N1) is storage: on new combinations of known symbols it is barely above the n-gram table (0.16-0.18 against 0.05), and
   on new symbols it is at 0. The diagnosis from t21 holds on this measure.
2. Lag cells with a factorised readout pass layer b (0.92) but not c (0.02): a strength for each (lag, byte) covers new combinations,
   not symbols never given as answers. The gated relay passes c too (0.71-0.84 on 14-55 bytes).
3. Without a cue the change of rule is slow: the relay stays at 0.34 for about 1,750 lines of rev, then switches. On the return to copy
   it starts at 0.64 (part of copy was kept) and is at 1.00 after about 2,500 lines.
Limits: one seed; layer c is small; the topographic map of the relay is a fixed anatomical prior that helps only rules that repeat
earlier bytes; a learned mapping (successor) and the keeping of other tasks are measured in t23.
