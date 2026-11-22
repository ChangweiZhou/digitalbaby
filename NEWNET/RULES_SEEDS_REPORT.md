# Rules learned at first encounter: seeds 2 and 3, and one new option (t22_rules.py, 22 November 2026)

This is development-level work on one machine (4 cores, OMP_NUM_THREADS=1, 4 processes at once). No gate, seal or outside review is claimed.
- No existing .py file was changed.
- The seed-1 lines come from `rules_runs/res.jsonl`. The net12 runs there say `net 11` because the file was renamed; it is the same code as net12.py with default options.
- Seeds 2 and 3, the ctxg variant and the storage baseline (seeds 1-3) are in `rules_runs_seeds/res.jsonl`, with one log per run in `rules_runs_seeds/`.
- The queue is `rules_runs_seeds/run_seeds.sh`. The tables below come from `rules_runs_seeds/tables.py`, whose full output is in `rules_runs_seeds/tables.md`.
- Each net run took 810-976 s. The storage runs took 4 s each.

Stream and scoring are as in RULES_REPORT.md:
- Phases: copy for lines 0-12,000 (P0), rev for 18,000 (P1), copy again for 24,000 (P2). There is no cue.
- Only the answer bytes of first encounters are scored.
- Layers: a, storage could know it; b, a new combination of symbols that were answers at least 5 times; n, 1-4 times; c, the symbol was never an answer before.

The stream depends on the seed (rng seed+4000), so the byte counts differ a little between seeds.

## Runs

```
python3 t22_rules.py net=store seed=S                                                     S = 1,2,3
python3 t22_rules.py net=10 struct=K8 seed=S                                              S = 2,3
python3 t22_rules.py net=12 struct=K8 lagro=1 seed=S                                      S = 2,3
python3 t22_rules.py net=12 struct=K8 lagro=1 relay=1 seed=S                              S = 2,3
python3 t22_rules.py net=12 struct=K8 lagro=1 relay=1 ctxg=16 ctxg_err=1.0 seed=S         S = 1,2,3
```
All runs used `out=rules_runs_seeds/res.jsonl`. All 12 finished, and no log contains an error.

## Right share by phase and layer, per run (byte counts in brackets)

| run | seed | P0 b | P0 c | P1 a | P1 b | P1 c | P2 b | P2 c |
|---|---|---|---|---|---|---|---|---|
| storage | 1 | 0.05 (30448) | 0.06 (55) | 0.74 (8396) | 0.00 (9401) | 0.00 (24) | 0.07 (17608) | 0.00 (14) |
| storage | 2 | 0.05 (30438) | 0.07 (55) | 0.74 (8391) | 0.01 (9371) | 0.00 (24) | 0.07 (17605) | 0.07 (14) |
| storage | 3 | 0.05 (30506) | 0.04 (55) | 0.74 (8352) | 0.01 (9423) | 0.00 (24) | 0.07 (17616) | 0.14 (14) |
| K8 (net10) | 1 | 0.16 (30448) | 0.00 (55) | 0.70 (8396) | 0.03 (9401) | 0.00 (24) | 0.12 (17608) | 0.00 (14) |
| K8 (net10) | 2 | 0.16 (30438) | 0.00 (55) | 0.70 (8391) | 0.03 (9371) | 0.00 (24) | 0.11 (17605) | 0.00 (14) |
| K8 (net10) | 3 | 0.16 (30506) | 0.00 (55) | 0.71 (8352) | 0.05 (9423) | 0.00 (24) | 0.11 (17616) | 0.00 (14) |
| net12 direct | 1 | 0.92 (30448) | 0.02 (55) | 0.73 (8396) | 0.30 (9401) | 0.00 (24) | 0.75 (17608) | 0.00 (14) |
| net12 direct | 2 | 0.92 (30438) | 0.02 (55) | 0.74 (8391) | 0.30 (9371) | 0.00 (24) | 0.75 (17605) | 0.00 (14) |
| net12 direct | 3 | 0.92 (30506) | 0.02 (55) | 0.74 (8352) | 0.31 (9423) | 0.00 (24) | 0.75 (17616) | 0.00 (14) |
| net12 direct + relay | 1 | 0.99 (30448) | 0.84 (55) | 0.73 (8396) | 0.77 (9401) | 0.71 (24) | 0.91 (17608) | 0.79 (14) |
| net12 direct + relay | 2 | 0.99 (30438) | 0.82 (55) | 0.73 (8391) | 0.77 (9371) | 0.62 (24) | 0.91 (17605) | 0.86 (14) |
| net12 direct + relay | 3 | 0.99 (30506) | 0.84 (55) | 0.73 (8352) | 0.77 (9423) | 0.79 (24) | 0.91 (17616) | 0.79 (14) |
| net12 direct + relay + ctxg=16, ctxg_err=1 | 1 | 0.98 (30448) | 0.80 (55) | 0.82 (8396) | 0.79 (9401) | 0.67 (24) | 0.99 (17608) | 0.93 (14) |
| net12 direct + relay + ctxg=16, ctxg_err=1 | 2 | 0.98 (30438) | 0.80 (55) | 0.81 (8391) | 0.76 (9371) | 0.62 (24) | 0.99 (17605) | 0.93 (14) |
| net12 direct + relay + ctxg=16, ctxg_err=1 | 3 | 0.98 (30506) | 0.80 (55) | 0.94 (8352) | 0.84 (9423) | 0.88 (24) | 1.00 (17616) | 0.93 (14) |

The n layers and P2 a (11-19 bytes) are in `rules_runs_seeds/tables.md` and the jsonl files.

## Mean ± sd over seeds 1-3 (sd with n-1)

| run | P0 b | P0 c | P1 a | P1 b | P1 c | P2 b | P2 c |
|---|---|---|---|---|---|---|---|
| storage | 0.05 ± 0.00 | 0.05 ± 0.02 | 0.74 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.07 ± 0.00 | 0.07 ± 0.07 |
| K8 (net10) | 0.16 ± 0.00 | 0.00 ± 0.00 | 0.70 ± 0.00 | 0.04 ± 0.01 | 0.00 ± 0.00 | 0.11 ± 0.00 | 0.00 ± 0.00 |
| net12 direct | 0.92 ± 0.00 | 0.02 ± 0.00 | 0.74 ± 0.00 | 0.30 ± 0.01 | 0.00 ± 0.00 | 0.75 ± 0.00 | 0.00 ± 0.00 |
| net12 direct + relay | 0.99 ± 0.00 | 0.83 ± 0.01 | 0.73 ± 0.00 | 0.77 ± 0.00 | 0.71 ± 0.08 | 0.91 ± 0.00 | 0.81 ± 0.04 |
| direct + relay + ctxg=16, ctxg_err=1 | 0.98 ± 0.00 | 0.80 ± 0.00 | 0.86 ± 0.07 | 0.80 ± 0.04 | 0.72 ± 0.13 | 0.99 ± 0.01 | 0.93 ± 0.00 |

Layer c is small: 55, 24 and 14 bytes per phase. Its spread is in steps of 1/24 or 1/14 of a byte share.

## After a change of rule (right share in windows of 250 lines, first 12 windows = 3,000 lines)

copy -> rev (P1):

| run | seed | windows 1-12 |
|---|---|---|
| storage | 1 / 2 / 3 | 0.14 ... 0.37 / 0.13 ... 0.38 / 0.13 ... 0.36 |
| K8 | 1 | 0.11 0.20 0.26 0.31 0.32 0.36 0.37 0.37 0.37 0.37 0.37 0.38 |
| K8 | 2 | 0.13 0.20 0.26 0.30 0.35 0.38 0.38 0.37 0.38 0.36 0.38 0.37 |
| K8 | 3 | 0.13 0.19 0.26 0.33 0.35 0.37 0.37 0.39 0.38 0.40 0.38 0.39 |
| net12 direct | 1 | 0.33 0.35 0.38 0.39 0.43 0.46 0.48 0.48 0.49 0.52 0.51 0.53 |
| net12 direct | 2 | 0.34 0.36 0.39 0.42 0.44 0.45 0.49 0.47 0.50 0.53 0.51 0.52 |
| net12 direct | 3 | 0.31 0.35 0.39 0.41 0.45 0.46 0.47 0.52 0.50 0.50 0.55 0.54 |
| direct + relay | 1 | 0.34 0.34 0.35 0.34 0.34 0.35 0.38 0.48 0.61 0.80 0.89 0.95 |
| direct + relay | 2 | 0.34 0.34 0.34 0.35 0.34 0.35 0.38 0.45 0.62 0.77 0.90 0.95 |
| direct + relay | 3 | 0.34 0.34 0.34 0.35 0.35 0.35 0.38 0.46 0.61 0.79 0.90 0.95 |
| + ctxg_err | 1 | 0.34 0.34 0.36 0.37 0.40 0.47 0.65 0.71 0.86 0.92 0.95 0.98 |
| + ctxg_err | 2 | 0.34 0.34 0.36 0.38 0.43 0.48 0.66 0.71 0.79 0.84 0.89 0.90 |
| + ctxg_err | 3 | 0.37 0.53 0.68 0.81 0.80 0.94 0.92 0.94 0.95 0.96 0.97 0.90 |

rev -> copy, the return (P2):

| run | seed | windows 1-12 |
|---|---|---|
| storage | 1-3 | 0.07 ... 0.11 |
| K8 | 1-3 | 0.09 ... 0.14 |
| net12 direct | 1 | 0.58 0.59 0.62 0.61 0.63 0.67 0.67 0.70 0.71 0.71 0.72 0.72 |
| net12 direct | 2 | 0.59 0.60 0.61 0.62 0.64 0.63 0.68 0.69 0.72 0.72 0.74 0.76 |
| net12 direct | 3 | 0.58 0.61 0.61 0.64 0.63 0.65 0.66 0.71 0.71 0.72 0.75 0.75 |
| direct + relay | 1 | 0.64 0.67 0.69 0.71 0.71 0.78 0.82 0.89 0.94 0.98 1.00 1.00 |
| direct + relay | 2 | 0.63 0.67 0.68 0.70 0.71 0.76 0.83 0.90 0.95 0.97 0.99 1.00 |
| direct + relay | 3 | 0.62 0.68 0.69 0.69 0.72 0.77 0.80 0.90 0.94 0.98 0.99 1.00 |
| + ctxg_err | 1 | 0.84 0.95 0.99 1.00 1.00 0.99 0.99 1.00 1.00 1.00 1.00 1.00 |
| + ctxg_err | 2 | 0.95 0.95 0.99 0.99 1.00 1.00 0.99 0.99 1.00 0.99 1.00 1.00 |
| + ctxg_err | 3 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |

First window with a right share of at least 0.9, read from the tables above (seeds 1 / 2 / 3; "-" = not within 12 windows):

| | copy -> rev | rev -> copy |
|---|---|---|
| direct + relay | 12 / 11 / 11 | 9 / 8 / 8 |
| direct + relay + ctxg=16, ctxg_err=1 | 10 / 12 / 6 | 2 / 1 / 1 |
| net12 direct | - / - / - | - / - / - |

## The ctxg_err variant against direct + relay

- **Return to copy: faster on all three seeds.** It starts at 0.84 / 0.95 / 1.00 against 0.62-0.64 for direct + relay, and reaches 0.9 within 1-2 windows (250-500 lines) instead of 8-9 windows (2,000-2,250 lines).
- **Switch to rev: faster on two seeds, about the same on one.**
  - Seeds 1 and 2: the first 4-5 windows match relay (0.34-0.43). It then rises earlier: 0.65-0.66 in window 7 against 0.38. It reaches 0.9 in window 10 on seed 1. On seed 2 it reaches 0.9 only in window 12, a window later than relay (11), and its window-12 value is lower (0.90 against 0.95).
  - Seed 3: it switches early, at 0.53 in window 2 and 0.94 in window 6.
  - Its windows also fluctuate more: seed 3 falls from 0.97 to 0.90 at window 12.
- **Phase totals:** P1 a is higher (0.86 ± 0.07 against 0.73) and P2 b is higher (0.99 against 0.91). P0 c is a little lower (0.80 against 0.83 ± 0.01). P1 b (0.80 ± 0.04 against 0.77) and P1 c (0.72 ± 0.13 against 0.71 ± 0.08) are about the same. On P1 c the spread is larger than the difference.

## Do the seed-1 findings hold on seeds 2 and 3?

(i) **K8 near storage on layer b, 0 on c: holds.**
- K8 on b: P0 0.16 ± 0.00, P1 0.04 ± 0.01, P2 0.11 ± 0.00. Storage: 0.05, 0.00, 0.07.
- K8 on c: 0.00 on every seed and phase.
- Storage gets 1-2 of the 14 P2 c bytes on seeds 2 and 3 (0.07, 0.14). K8 does not.

(ii) **Direct readout passes b, not c: holds.**
- On b: P0 0.92, P2 0.75 on every seed. P1 b is 0.30-0.31; that phase is the slow switch.
- On c: 0.02 (one of 55 bytes) in P0 and 0.00 in P1 and P2, on all three seeds.

(iii) **Relay passes b and c: holds.**
- On b: P0 0.99, P1 0.77, P2 0.91 on every seed.
- On c: P0 0.82-0.84, P1 0.62-0.79, P2 0.79-0.86, on 14-55 bytes.

(iv) **Without a cue the switch to rev is slow: holds for K8, direct and relay.**
- Relay sits at 0.34-0.35 for 6 windows (1,500 lines) on every seed and reaches 0.9 after 2,750-3,000 lines.
- Direct readout is at 0.52-0.54 after 3,000 lines.
- K8 and storage stay at or below 0.40.
- The ctxg_err option shortens this on seeds 1 and 3, but not on seed 2, where it reaches 0.9 one window later than relay.

## Limits

- 3 seeds only.
- Layer c has 14-55 bytes per phase.
- ctxg=16 / ctxg_err=1.0 was tried at one setting only; no other ctxg values, and no ctxg without ctxg_err, were run.
- The relay's topographic map is a fixed prior, as stated in RULES_REPORT.md.
- One life per run; the keeping of other tasks is not measured here.
