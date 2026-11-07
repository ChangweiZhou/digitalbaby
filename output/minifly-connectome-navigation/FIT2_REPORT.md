# MiniFly closed-loop gain refit

## Outcome

**3/3 seeds met the revised combined-loop gates.** Native combined convergence reached 79/96 in every seed, five goals below the clamped-heading score of 84/96, with a 57–58-goal lead over the equally fitted rewires. Native four-second median compass error was 12.01°–12.08°. The prior 128-update fit reached 55–56/96 combined goals. [Corrected first-round report](FIT_REPORT.md)

The fitted circuit consists of the EPG, PEN, Δ7, PFL3 and FC2 projections. It has eight role-pair gains and seven adapter or dynamics scalars, 15 free parameters in all. The other 98 extracted role pairs are absent from the simulator.

## Five historical probes

Stationary heading and the dark-turn count require bump retention and error within 20°. Dark-turn degrees give the median absolute heading error. Clamped steering counts correct initial directions. Clamped convergence gives the number within 20° and median terminal goal error. The combined loop gives the number within 20° and median compass error at four seconds.

| Arm and seed | Stationary heading | Dark turns | Clamped steering | Clamped convergence | Combined loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native hand-set (all seeds) | 16/16 | 8.56° · 81/96 | 94/96 | 75/96 · 12.70° | 42/96 · 21.08° |
| Rewired hand-set (all seeds) | 1/16 | 84.81° · 11/96 | 49/96 | 14/96 · 58.19° | 11/96 · 84.75° |
| Native fitted 197 | 16/16 | 9.75° · 79/96 | 96/96 | 84/96 · 5.86° | 79/96 · 12.01° |
| Rewired fitted 197 | 0/16 | 62.95° · 0/96 | 74/96 | 24/96 · 43.24° | 22/96 · 33.44° |
| Native fitted 281 | 16/16 | 9.77° · 79/96 | 96/96 | 84/96 · 5.97° | 79/96 · 12.06° |
| Rewired fitted 281 | 16/16 | 61.75° · 0/96 | 74/96 | 23/96 · 45.21° | 21/96 · 30.33° |
| Native fitted 389 | 16/16 | 9.39° · 79/96 | 96/96 | 84/96 · 5.84° | 79/96 · 12.08° |
| Rewired fitted 389 | 16/16 | 61.75° · 0/96 | 74/96 | 23/96 · 44.51° | 22/96 · 29.53° |

## Revised gates

| Seed | Clamped minus combined ≤10 | Compass <15° | Lead over rewired ≥30 | Dark-turn change vs 8.56° hand-set |
| --- | ---: | ---: | ---: | ---: |
| 197 | 5 (pass) | 12.01° (pass) | +57 (pass) | +1.19° |
| 281 | 5 (pass) | 12.06° (pass) | +58 (pass) | +1.21° |
| 389 | 5 (pass) | 12.08° (pass) | +57 (pass) | +0.83° |

The dark-turn tolerance allows up to 3° of regression. Larger regressions are reported as trade-offs. The clamped and combined panels have different offsets. Their score difference is an empirical reference for this round.

## Steering misses at four seconds

Each angle column counts misses out of 16 episodes with that signed initial goal offset. The panel spans ±30°, ±65° and ±120°, so it can reveal asymmetry and differences across the sampled angles. The panel has no near-zero or near-180° offsets.

| Parameters | Wiring | Seed | Misses | −120° | −65° | −30° | +30° | +65° | +120° | Stable wrong heading | Still approaching | Moving away | Unresolved |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hand-set | native | all | 21 | 6 | 2 | 4 | 4 | 3 | 2 | 11 | 6 | 4 | 0 |
| Hand-set | rewired | all | 82 | 13 | 12 | 14 | 14 | 14 | 15 | 28 | 19 | 35 | 0 |
| 128 updates | native | 197 | 23 | 9 | 2 | 1 | 2 | 2 | 7 | 2 | 20 | 1 | 0 |
| 128 updates | rewired | 197 | 78 | 16 | 14 | 11 | 8 | 13 | 16 | 14 | 30 | 34 | 0 |
| 128 updates | native | 281 | 21 | 9 | 2 | 1 | 1 | 2 | 6 | 2 | 19 | 0 | 0 |
| 128 updates | rewired | 281 | 81 | 16 | 14 | 14 | 8 | 13 | 16 | 12 | 32 | 37 | 0 |
| 128 updates | native | 389 | 22 | 9 | 2 | 1 | 2 | 2 | 6 | 2 | 19 | 1 | 0 |
| 128 updates | rewired | 389 | 78 | 16 | 14 | 12 | 11 | 10 | 15 | 21 | 24 | 33 | 0 |
| 2,048 updates | native | 197 | 12 | 4 | 2 | 2 | 2 | 1 | 1 | 5 | 4 | 3 | 0 |
| 2,048 updates | rewired | 197 | 72 | 16 | 14 | 6 | 8 | 12 | 16 | 13 | 41 | 18 | 0 |
| 2,048 updates | native | 281 | 12 | 4 | 2 | 2 | 2 | 1 | 1 | 5 | 4 | 3 | 0 |
| 2,048 updates | rewired | 281 | 73 | 16 | 14 | 6 | 8 | 13 | 16 | 9 | 46 | 17 | 1 |
| 2,048 updates | native | 389 | 12 | 4 | 2 | 2 | 2 | 1 | 1 | 5 | 4 | 3 | 0 |
| 2,048 updates | rewired | 389 | 73 | 16 | 14 | 6 | 8 | 13 | 16 | 8 | 46 | 19 | 0 |

All three native seeds missed the same 12 episodes. The native refit missed 24 negative-offset and 12 positive-offset episodes across three seeds. The most misses occurred at −120° (12/48). Of 36 misses, 15 had a stable wrong heading, 12 were still approaching, 9 were moving away, and 0 were unresolved. Every missed episode ended with a negative signed residual (goal minus heading). All four slow misses per seed began at −120°.

The rewired refit missed 108 negative-offset and 110 positive-offset episodes across three seeds. The most misses occurred at −120° and +120° (48/48 per listed angle). Of 218 misses, 30 had a stable wrong heading, 133 were still approaching, 54 were moving away, and 1 were unresolved.

A stable wrong heading requires no more than 1° absolute-error span over the final 20 steps and a final yaw command no larger than 0.05° per step. Approaching and moving away require a fitted late slope beyond ±1°/s and a net late change above 1° in the same direction. These labels describe the observed four-second behavior. The matched fitting procedure shows where the two wirings differ after fitting. Each topology has its own fitted adapter values, so this comparison cannot assign a cluster to wiring independently of those fitted adapters.

[Every missed episode with initial angle, signed final error and late trend](FIT2_STEERING.csv) · [Machine-readable diagnosis](FIT2_STEERING.json)

## Training trajectory

Values are RMS degrees on all 384 training episodes at the saved iterate. Each cell gives heading RMS / goal RMS. The heading term covers all 80 steps. The goal term covers steps 61–80.

| Update | 197 native | 197 rewired | 281 native | 281 rewired | 389 native | 389 rewired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 42.59 / 47.80 | 78.64 / 95.93 | 34.59 / 47.45 | 72.45 / 98.40 | 38.82 / 44.08 | 80.16 / 94.65 |
| 128 | 12.40 / 21.68 | 21.29 / 69.95 | 11.94 / 21.66 | 21.76 / 69.48 | 12.91 / 21.83 | 72.12 / 88.58 |
| 256 | 11.82 / 18.46 | 21.81 / 68.07 | 11.90 / 18.83 | 21.41 / 68.44 | 11.82 / 18.66 | 25.78 / 81.85 |
| 384 | 12.17 / 16.93 | 22.05 / 67.81 | 12.29 / 17.18 | 21.59 / 68.20 | 12.08 / 17.24 | 21.06 / 68.86 |
| 512 | 12.19 / 16.15 | 22.22 / 67.59 | 12.12 / 16.32 | 21.51 / 68.13 | 12.25 / 16.33 | 20.81 / 68.71 |
| 640 | 12.28 / 15.83 | 22.48 / 67.42 | 12.30 / 15.89 | 21.71 / 68.03 | 12.25 / 16.08 | 20.77 / 68.64 |
| 768 | 12.29 / 15.63 | 22.23 / 67.46 | 12.28 / 15.68 | 21.45 / 68.08 | 12.25 / 15.88 | 20.83 / 68.56 |
| 896 | 12.32 / 15.48 | 22.29 / 67.39 | 12.32 / 15.50 | 21.60 / 68.00 | 12.29 / 15.68 | 20.95 / 68.46 |
| 1024 | 12.30 / 15.39 | 22.70 / 67.22 | 12.32 / 15.41 | 22.10 / 67.81 | 12.29 / 15.54 | 21.23 / 68.31 |
| 1152 | 12.21 / 15.43 | 22.60 / 67.23 | 12.19 / 15.46 | 21.97 / 67.84 | 12.18 / 15.56 | 21.31 / 68.25 |
| 1280 | 12.25 / 15.36 | 22.62 / 67.21 | 12.23 / 15.40 | 22.03 / 67.82 | 12.23 / 15.48 | 21.38 / 68.20 |
| 1408 | 12.28 / 15.31 | 22.63 / 67.19 | 12.30 / 15.31 | 22.01 / 67.81 | 12.27 / 15.40 | 21.44 / 68.15 |
| 1536 | 12.38 / 15.22 | 22.66 / 67.17 | 12.39 / 15.23 | 22.00 / 67.81 | 12.33 / 15.32 | 21.49 / 68.11 |
| 1664 | 12.29 / 15.27 | 22.63 / 67.17 | 12.29 / 15.29 | 21.98 / 67.81 | 12.28 / 15.34 | 21.54 / 68.08 |
| 1792 | 12.30 / 15.25 | 22.67 / 67.15 | 12.30 / 15.27 | 22.02 / 67.80 | 12.30 / 15.32 | 21.58 / 68.05 |
| 1920 | 12.34 / 15.22 | 22.67 / 67.14 | 12.34 / 15.23 | 22.00 / 67.80 | 12.33 / 15.27 | 21.59 / 68.03 |
| 2048 | 12.32 / 15.22 | 22.81 / 67.08 | 12.32 / 15.24 | 22.16 / 67.74 | 12.30 / 15.29 | 21.75 / 67.97 |

### Late goal-loss change

Native goal RMS finished at 15.22–15.29°. It changed by at most 0.04° in the final 512 updates. The training loss was nearly flat by the end of this schedule.

| Seed | Wiring | Goal RMS at 1,536 | Goal RMS at 2,048 | Change |
| --- | --- | ---: | ---: | ---: |
| 197 | native | 15.22° | 15.22° | +0.00° |
| 197 | rewired | 67.17° | 67.08° | -0.08° |
| 281 | native | 15.23° | 15.24° | +0.01° |
| 281 | rewired | 67.81° | 67.74° | -0.07° |
| 389 | native | 15.32° | 15.29° | -0.03° |
| 389 | rewired | 68.11° | 67.97° | -0.13° |

## Three cycles

1. Verified signed-degree, subtype and side invariants and legacy rollout parity. Traced the archived hand-set and 128-update fits on the original clamped panel.
2. Reproduced the original trajectory and batch order, checked the new loss gradients against central differences (maximum absolute difference 1.4e-08), and replayed two Adam updates exactly.
3. Fit all three native and rewired pairs for 2,048 updates with Adam. The learning rate started at 0.025 and halved after each 512-update block. Every fit restarted from the original seed-specific initialization. Final parameters and training curves were saved before any refit panel was scored.

[Lock](FIT2_LOCK.json) · [Gain vectors and scores](FIT2_RESULTS.json) · [Training curves](FIT2_CURVES.json)
