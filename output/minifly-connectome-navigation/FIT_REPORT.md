# MiniFly connectome gain fitting

## Outcome

**Qualification failed.** Native-fitted reached 55–56/96 goals versus 13–15/96 for equally fitted rewires. The native four-second compass medians spanned 10.12°–11.25°. The native lead was 41–42 goals across matched seeds. All three missed the 72-goal threshold and regressed on dark turns.

## Five locked probes

Stationary counts require a retained bump and error within 20°. Dark turns report median absolute heading error and the count satisfying the same joint criterion. Clamped steering counts correct initial directions. Clamped convergence reports goals within 20° and median terminal goal error. Combined reports goals within 20° and median compass error at four seconds.

| Arm and seed | Stationary heading | Dark turns | Clamped steering | Clamped convergence | Combined loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native hand-set (all three seeds) | 16/16 | 8.56° · 81/96 | 94/96 | 75/96 · 12.70° | 42/96 · 21.08° |
| Rewired hand-set (all three seeds) | 1/16 | 84.81° · 11/96 | 49/96 | 14/96 · 58.19° | 11/96 · 84.75° |
| Native fitted 197 | 16/16 | 12.05° · 79/96 | 95/96 | 73/96 · 13.35° | 55/96 · 11.25° |
| Rewired fitted 197 | 0/16 | 57.30° · 0/96 | 49/96 | 18/96 · 59.70° | 13/96 · 28.96° |
| Native fitted 281 | 16/16 | 11.75° · 78/96 | 95/96 | 75/96 · 13.00° | 56/96 · 10.12° |
| Rewired fitted 281 | 16/16 | 58.68° · 3/96 | 50/96 | 15/96 · 58.33° | 15/96 · 38.41° |
| Native fitted 389 | 16/16 | 12.38° · 79/96 | 95/96 | 74/96 · 12.86° | 56/96 · 10.74° |
| Rewired fitted 389 | 0/16 | 52.81° · 0/96 | 52/96 | 18/96 · 64.27° | 14/96 · 64.57° |

The deterministic hand-set arms share a row across seeds. Both rows reproduce their earlier qualification. The earlier 54.5° rewired turn figure used native hand-set parameters. The 84.81° row here uses the rewired arm’s own hand-set parameters.

## Gates by seed

| Seed | Native goals ≥72 | Native compass <15° | Native lead ≥30 | Native isolated regressions | Rewired isolated regressions | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 197 | Fail | Pass | Pass (+42) | dark-turn qualified count, convergence, dark-turn median | stationary, dark-turn qualified count | Fail |
| 281 | Fail | Pass | Pass (+41) | dark-turn qualified count, dark-turn median | dark-turn qualified count | Fail |
| 389 | Fail | Pass | Pass (+42) | dark-turn qualified count, convergence, dark-turn median | stationary, dark-turn qualified count | Fail |

Rewired-fitted reached both native combined thresholds on zero seeds.

## Three cycles

1. Verified signed degree, subtype, and side invariants for both rewires. On separate development episodes the Torch and NumPy hand-set trajectories agreed within 5e-14 radians.
2. Checked all 15 gradients on full 80-step training episodes. The largest absolute difference from central differences was 3.2e-08, and two-step optimizer replay was bitwise identical.
3. Completed the six matched fits. Saved the final parameters and curves before scoring the historical panels. The training inputs were disjoint from those panels by starting heading. The panels had already been published in the hand-set qualification.

## Fit and scope

Adam differentiated through the full 80-step combined loop. Each fit used the same 384 training episodes and 128 updates of 48 episodes. Each paired seed used identical initial gains and minibatch order across the two wirings. The loss combined the mean squared wrapped compass error throughout the episode and the mean squared wrapped terminal goal error. The initial landmark cue was the only compass cue.

Only eight of the 106 extracted role pairs contribute to this historical simulator. The remaining 98 gains are recorded at unity. Seven adapter or dynamics scalars bring the fitted count to 15. Within each projection the historical signed weights and normalization were preserved, including the fixed Delta7 inhibitory assignment in the prior simulator. Steering uses the same 64-phase normalization with the original statistics recomputed for the continuous path weights. Several gains appear as products or can partly trade off through normalization, so the individual numbers cannot establish unique biological strengths.

| Seed | Arm | Training heading MSE start → end | Training terminal-goal MSE start → end |
| --- | --- | ---: | ---: |
| 197 | native | 0.5526 → 0.0464 | 0.7470 → 0.1430 |
| 197 | rewired | 1.8840 → 0.2263 | 2.7029 → 2.3273 |
| 281 | native | 0.3644 → 0.0413 | 0.7062 → 0.1368 |
| 281 | rewired | 1.5988 → 0.3158 | 3.0180 → 2.2792 |
| 389 | native | 0.4589 → 0.0452 | 0.6251 → 0.1392 |
| 389 | rewired | 1.9572 → 1.1426 | 2.7640 → 2.2141 |

Losses are in radians squared. [Lock](FIT_LOCK.json) records the frozen protocol. [Results and gain vectors](FIT_RESULTS.json) contain all scores and the three cycle receipts. [Fitting curves](FIT_CURVES.json) contain every update and full-training loss endpoints.
