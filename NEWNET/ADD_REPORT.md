# Column addition in one life (t24_add.py, 23 November 2026)

Development level, seed 1, one machine; no gate, seal or outside review is claimed. Results: `add_runs/res.jsonl` (two runs were
killed by a container restart and run again; the killed logs are kept as `*_killed_by_restart.log`).
Numbers lowest digit first, one column a time: col = `a b s` (the carry not written), colc = `c a b s` (the carry written first).
N0 = 8,000 lines with 1-4 digits, then N1 = 8,000 lines with 1-8 (5-8 new). Every pair is new: all scores are first encounters.

| | col: lines right, old length / new length | col: free, 8 digits | colc: lines right, old / new | colc: free, 8 digits |
|---|---|---|---|---|
| storage (back-off n-gram) | 0.80 / 0.38 | 0.30 | 0.97 / 0.87 | 0.90 |
| K8 (net10) | 0.44 / 0.03 | 0.00 | 0.90 / 0.68 | 0.46 |
| net12 K8 + lag cells + relay | 0.47 / 0.03 | 0.00 | 0.92 / 0.74 | 0.48 |

(lines right: phase N1, prequential, learning on; free: the last free test, learning off, its own answer bytes fed back.)

What it shows: with the carry written, every answer byte is a function of a few recent bytes at fixed lags, and the nets learn it
and carry it to new lengths (0.68-0.74), but less well than exact counting (0.87). Without the carry written they fail at new lengths
(0.03); the table does better (0.38) because it holds 5-byte contexts exactly. Neither net infers an intermediate quantity (the carry).
Learning such a hidden variable locally is open.
