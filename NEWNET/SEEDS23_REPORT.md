# Seeds 2 and 3 of the t23 / t24 / t22 seed-1 results (23 November 2026)

This is development-level work on one machine (4 cores, OMP_NUM_THREADS=1, 4 processes at once). No gate, seal or outside review is claimed.
- The code is at commit 3fdc5279 (from claude/magical-albattani-sei232). No .py file in NEWNET/ was changed. The only file added is `seeds23_runs/tables.py`, which builds the tables.
- The queue is `seeds23_runs/run.sh`, with one log per run in `seeds23_runs/`. All 16 runs finished and no log contains an error.
  - Each net run took 1,065-1,348 s (t23 1,214-1,348; t24 1,065-1,186; t22 827-832).
  - Each storage run took 2-6 s.
- The seed-1 lines come from `life_rules_runs/res.jsonl` (t23, cue=1), `add_runs/res.jsonl` (t24) and `rules_runs/res.jsonl` (t22 relay=2).
- The seed-2/3 lines come from `seeds23_runs/res_t23.jsonl`, `res_t24.jsonl` and `res_t22.jsonl`.
- The tables below are the output of `python3 seeds23_runs/tables.py`, also kept as `seeds23_runs/tables.md`. Spread is the population sd over seeds.
- t24 `col` was run on seeds 2 and 3 for storage only, as asked. K8 and net12 `col` have seed 1 only.
- P0 copy c for net12 at seed 1 is 0.725. It prints as 0.72 here; LIFE_RULES_REPORT.md rounds it to 0.73.

## Runs

```
python3 t23_life_rules.py net=store cue=1 seed=S
python3 t23_life_rules.py net=10 struct=K8 cue=1 seed=S
python3 t23_life_rules.py net=12 struct=K8 lagro=1 relay=1 cue=1 seed=S
python3 t22_rules.py net=12 struct=K8 lagro=1 relay=2 seed=S
python3 t24_add.py net=store fmt=colc seed=S ; python3 t24_add.py net=store fmt=col seed=S
python3 t24_add.py net=10 struct=K8 fmt=colc seed=S
python3 t24_add.py net=12 struct=K8 lagro=1 relay=1 fmt=colc seed=S        S = 2, 3
```

## t23 (cue=1)

| net | seed | bits total | P0 copy b / c | P1 rev b | P3 succ b | P4 copy b | P5 rev b | text 17k -> 29k | succ 23k -> 29k | copy / rev at 29k |
|---|---|---|---|---|---|---|---|---|---|---|
| storage | 1 | 1,984,668 | 0.06 / 0.07 | 0.00 | 0.02 | 0.00 | 0.00 | 0.37 -> 0.37 | 0.13 -> 0.10 | 0.15 / 0.71 |
| storage | 2 | 1,987,427 | 0.05 / 0.12 | 0.00 | 0.02 | 0.00 | 0.00 | 0.38 -> 0.37 | 0.08 -> 0.10 | 0.10 / 0.70 |
| storage | 3 | 1,989,122 | 0.06 / 0.05 | 0.00 | 0.01 | 0.00 | 0.00 | 0.38 -> 0.37 | 0.07 -> 0.10 | 0.07 / 0.73 |
| K8 (net10) | 1 | 1,149,490 | 0.14 / 0.00 | 0.01 | 0.13 | 0.10 | 0.01 | 0.36 -> 0.09 | 0.11 -> 0.00 | 0.00 / 0.35 |
| K8 (net10) | 2 | 1,144,630 | 0.14 / 0.00 | 0.00 | 0.13 | 0.11 | 0.01 | 0.36 -> 0.12 | 0.13 -> 0.01 | 0.00 / 0.35 |
| K8 (net10) | 3 | 1,154,974 | 0.15 / 0.00 | 0.00 | 0.12 | 0.11 | 0.01 | 0.35 -> 0.07 | 0.11 -> 0.00 | 0.00 / 0.34 |
| net12 relay | 1 | 828,527 | 0.98 / 0.72 | 0.80 | 0.51 | 0.99 | 1.00 | 0.35 -> 0.24 | 0.98 -> 0.07 | 1.00 / 1.00 |
| net12 relay | 2 | 829,414 | 0.98 / 0.72 | 0.80 | 0.51 | 0.99 | 1.00 | 0.37 -> 0.26 | 0.95 -> 0.03 | 1.00 / 1.00 |
| net12 relay | 3 | 831,508 | 0.98 / 0.72 | 0.79 | 0.50 | 0.99 | 1.00 | 0.35 -> 0.23 | 0.98 -> 0.04 | 1.00 / 1.00 |

Mean ± spread (population sd) over seeds:

| net | n | bits total | P0 copy b | P0 copy c | P1 rev b | P3 succ b | P4 copy b | P5 rev b | text 17k | text 29k | succ 23k | succ 29k | copy 29k | rev 29k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| storage | 3 | 1,987,072 ± 1,836 | 0.06 ± 0.00 | 0.08 ± 0.03 | 0.00 ± 0.00 | 0.02 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.37 ± 0.00 | 0.37 ± 0.00 | 0.10 ± 0.03 | 0.10 ± 0.00 | 0.11 ± 0.03 | 0.71 ± 0.01 |
| K8 (net10) | 3 | 1,149,698 ± 4,225 | 0.14 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.12 ± 0.00 | 0.11 ± 0.00 | 0.01 ± 0.00 | 0.35 ± 0.01 | 0.09 ± 0.02 | 0.12 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.35 ± 0.01 |
| net12 relay | 3 | 829,816 ± 1,250 | 0.98 ± 0.00 | 0.72 ± 0.00 | 0.80 ± 0.00 | 0.50 ± 0.00 | 0.99 ± 0.00 | 1.00 ± 0.00 | 0.36 ± 0.01 | 0.24 ± 0.01 | 0.97 ± 0.02 | 0.04 ± 0.02 | 1.00 ± 0.00 | 1.00 ± 0.00 |

## t24

| net | fmt | seed | lines right P1 old | P1 new | free 8 digits (last test, line) |
|---|---|---|---|---|---|
| storage | col | 1 | 0.80 | 0.38 | 0.30 (16000) |
| storage | col | 2 | 0.80 | 0.37 | 0.38 (16000) |
| storage | col | 3 | 0.81 | 0.36 | 0.42 (16000) |
| K8 (net10) | col | 1 | 0.44 | 0.03 | 0.00 (16000) |
| net12 relay | col | 1 | 0.47 | 0.03 | 0.00 (16000) |
| storage | colc | 1 | 0.97 | 0.87 | 0.90 (16000) |
| storage | colc | 2 | 0.96 | 0.87 | 0.84 (16000) |
| storage | colc | 3 | 0.96 | 0.87 | 0.86 (16000) |
| K8 (net10) | colc | 1 | 0.90 | 0.68 | 0.46 (16000) |
| K8 (net10) | colc | 2 | 0.90 | 0.70 | 0.34 (16000) |
| K8 (net10) | colc | 3 | 0.88 | 0.61 | 0.52 (16000) |
| net12 relay | colc | 1 | 0.92 | 0.74 | 0.48 (16000) |
| net12 relay | colc | 2 | 0.92 | 0.75 | 0.36 (16000) |
| net12 relay | colc | 3 | 0.91 | 0.70 | 0.60 (16000) |

| net | fmt | n | P1 old | P1 new | free 8 |
|---|---|---|---|---|---|
| storage | col | 3 | 0.80 ± 0.00 | 0.37 ± 0.01 | 0.37 ± 0.05 |
| K8 (net10) | col | 1 | 0.44 | 0.03 | 0.00 |
| net12 relay | col | 1 | 0.47 | 0.03 | 0.00 |
| storage | colc | 3 | 0.96 ± 0.00 | 0.87 ± 0.00 | 0.87 ± 0.02 |
| K8 (net10) | colc | 3 | 0.89 ± 0.01 | 0.66 ± 0.04 | 0.44 ± 0.07 |
| net12 relay | colc | 3 | 0.92 ± 0.00 | 0.73 ± 0.02 | 0.48 ± 0.10 |

## t22 net12 relay=2 (right share, bytes in brackets)

| seed | 0b | 0c | 0n | 1a | 1b | 1c | 1n | 2a | 2b | 2c | 2n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.98 (30448) | 0.73 (55) | 0.81 (220) | 0.86 (8396) | 0.83 (9401) | 0.42 (24) | 0.61 (77) | 1.00 (19) | 0.81 (17608) | 0.43 (14) | 0.39 (56) |
| 2 | 0.98 (30438) | 0.76 (55) | 0.83 (218) | 0.86 (8391) | 0.82 (9371) | 0.46 (24) | 0.67 (79) | 1.00 (28) | 0.82 (17605) | 0.21 (14) | 0.45 (56) |
| 3 | 0.98 (30506) | 0.71 (55) | 0.81 (219) | 0.86 (8352) | 0.83 (9423) | 0.50 (24) | 0.68 (72) | 1.00 (26) | 0.81 (17616) | 0.00 (14) | 0.45 (56) |
| mean ± sd | 0.98 ± 0.00 | 0.73 ± 0.02 | 0.82 ± 0.01 | 0.86 ± 0.00 | 0.83 ± 0.00 | 0.46 ± 0.03 | 0.65 ± 0.03 | 1.00 ± 0.00 | 0.82 ± 0.00 | 0.21 ± 0.18 | 0.43 ± 0.02 |

After a change of rule (250-line windows):

- seed 1: copy->rev [0.34, 0.34, 0.38, 0.44, 0.5, 0.64, 0.79, 0.92, 0.94, 0.97, 0.99, 0.99]; rev->copy [0.35, 0.35, 0.38, 0.38, 0.45, 0.54, 0.66, 0.79, 0.87, 0.93, 0.95, 0.95]
- seed 2: copy->rev [0.33, 0.35, 0.38, 0.42, 0.48, 0.63, 0.82, 0.9, 0.94, 0.96, 0.98, 0.99]; rev->copy [0.36, 0.35, 0.37, 0.39, 0.43, 0.55, 0.67, 0.83, 0.89, 0.93, 0.94, 0.97]
- seed 3: copy->rev [0.33, 0.36, 0.37, 0.42, 0.49, 0.63, 0.8, 0.93, 0.94, 0.96, 0.99, 0.99]; rev->copy [0.35, 0.36, 0.36, 0.38, 0.43, 0.55, 0.67, 0.81, 0.86, 0.92, 0.95, 0.96]

## Which seed-1 findings hold (3 seeds)

**t23 (one life with rules and a text)**
1. *Total bits.* The order and ratio hold. net12 relay uses 829,816 ± 1,250 bits, K8 uses 1,149,698 ± 4,225 and storage 1,987,072 ± 1,836. net12 needs 42% of storage's bits and 72% of K8's, as at seed 1.
2. *Rules met for the first time.* For net12 relay, the share right on first encounters hardly moves across seeds (sd ≤ 0.005 in every phase):
   - P0 copy: 0.98 on layer b and 0.72 on layer c (40 bytes).
   - rev: 0.80. succ: 0.50. copy again: 0.99. rev again: 1.00.
   - K8 and storage stay at 0.00-0.15 on every phase, as at seed 1.
3. *Kept rules.* Probes at the end of the life give copy and rev 1.00 / 1.00 for net12 at every seed. This holds.
4. *The learned mapping (succ) is forgotten.* It holds at every seed: 0.97 ± 0.02 right after the succ phase, 0.04 ± 0.02 at the end.
5. *The text is partly lost.* It holds:
   - net12 keeps 0.36 -> 0.24 (about 67%).
   - K8 keeps 0.35 -> 0.09 ± 0.02 (about 25%).
   - storage keeps 0.37 -> 0.37.

**t24 (column addition)**

6. *Carry written (colc).* At new lengths, the nets are below exact counting at every seed:
   - P1 new: net12 0.73 ± 0.02, K8 0.66 ± 0.04, storage 0.87 ± 0.00.
   - The last free test at 8 digits spreads more across seeds: net12 0.48 ± 0.10 (0.36-0.60), K8 0.44 ± 0.07 (0.34-0.52), storage 0.87 ± 0.02.
   - net12 is at or above K8 on every seed and measure. On the free test the gap (0.02-0.08) is smaller than the spread across seeds.
7. *Carry not written (col).* Storage holds (P1 new 0.37 ± 0.01, free 8 digits 0.37 ± 0.05). The nets' failure at new lengths (0.03) was not repeated on seeds 2-3; it rests on seed 1 only.

**t22 relay=2 (the learned code relay)**

8. *Layer table.* The table repeats closely on seeds 2 and 3 for layers a, b and n:
   - 0b 0.98. 1a 0.86. 1b 0.83. 2b 0.82. 0n 0.82. 1n 0.65. 2n 0.43.
   - Layer c (14-55 bytes) is lower and noisy: 0c 0.73 ± 0.02, 1c 0.46 ± 0.03 and 2c 0.21 ± 0.18 (0.43, 0.21, 0.00 on 14 bytes).
   - Compared with relay=1 in RULES_SEEDS_REPORT.md (1c 0.71, 2c 0.81, 1b 0.77, 2b 0.91), relay=2 is higher on 1a and 1b but lower on layer c and on 2b. This holds at all three seeds.
9. *After a change of rule.* The windows repeat at every seed:
   - copy->rev rises from 0.33-0.34 to 0.99 within 12 windows.
   - On the return to copy it starts at 0.35-0.36 (nothing of copy carried over, unlike relay=1's 0.64) and reaches 0.95-0.97.

**Limits.** Three seeds of one code version. Layer c is a few dozen bytes. t24 col for the nets, cue=0 and the ctxg/xdg variants were not run here (NOT_RUN).
