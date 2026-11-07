# MiniFly official gain-fit confirmation

## Outcome

**3/3 fitting seeds met the revised combined-loop gates on the fresh panels.** The six gain vectors came from the 2,048-update fit and were frozen before scoring. The panels use one newly sampled heading per angular sector and preserve the original yaw schedules, goal offsets, episode lengths and sample sizes. Both wiring arms and both hand-set references received the same fresh panels.

## Five probes

| Arm | Seed | Stationary heading | Dark turns | Clamped steering | Clamped convergence | Combined loop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Native hand-set | all | 16/16 | 8.67° · 80/96 | 94/96 | 75/96 · 12.01° | 45/96 · 18.68° |
| Rewired hand-set | all | 2/16 | 84.07° · 15/96 | 53/96 | 14/96 · 58.68° | 10/96 · 81.50° |
| Native fitted | 197 | 16/16 | 8.91° · 82/96 | 96/96 | 84/96 · 5.56° | 80/96 · 9.80° |
| Rewired fitted | 197 | 0/16 | 62.57° · 0/96 | 76/96 | 26/96 · 39.55° | 22/96 · 31.47° |
| Native fitted | 281 | 16/16 | 8.93° · 82/96 | 96/96 | 84/96 · 5.61° | 80/96 · 9.91° |
| Rewired fitted | 281 | 16/16 | 60.99° · 0/96 | 76/96 | 27/96 · 42.27° | 20/96 · 31.03° |
| Native fitted | 389 | 16/16 | 8.67° · 82/96 | 96/96 | 84/96 · 5.59° | 80/96 · 9.83° |
| Rewired fitted | 389 | 16/16 | 60.94° · 0/96 | 77/96 | 26/96 · 40.09° | 24/96 · 29.47° |

Dark turns report median heading error and qualified count. Clamped convergence reports successes within 20° and median terminal goal error. Combined reports successes within 20° and median terminal compass error.

## Gates

| Seed | Native gap ≤10 | Native compass <15° | Native lead ≥30 | Dark-turn delta vs fresh hand-set | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 197 | 4 | 9.80° | +58 | +0.24° | Pass |
| 281 | 4 | 9.91° | +60 | +0.26° | Pass |
| 389 | 4 | 9.83° | +56 | -0.00° | Pass |

A dark-turn regression above 3° is a reportable trade-off. The clamped and combined panels use different offset sets, so the gap is an empirical comparison.

## Clamped steering diagnosis

The bins show misses out of 16 episodes. The four-second trend classifier uses the locked criteria from the 2,048-update exercise. Every miss has signed final error and late trend in the result file.

| Arm | Seed | −120° | −65° | −30° | +30° | +65° | +120° | Negative final residual | Stable wrong | Approaching | Moving away | Unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native hand-set | all | 4 | 2 | 4 | 4 | 3 | 4 | 14/21 | 11 | 6 | 4 | 0 |
| Rewired hand-set | all | 13 | 12 | 13 | 13 | 15 | 16 | 45/82 | 29 | 18 | 34 | 1 |
| Native fitted | 197 | 3 | 2 | 1 | 2 | 3 | 1 | 12/12 | 5 | 3 | 4 | 0 |
| Rewired fitted | 197 | 16 | 14 | 5 | 7 | 12 | 16 | 35/70 | 7 | 44 | 19 | 0 |
| Native fitted | 281 | 3 | 2 | 1 | 2 | 3 | 1 | 12/12 | 5 | 3 | 4 | 0 |
| Rewired fitted | 281 | 16 | 14 | 6 | 6 | 11 | 16 | 36/69 | 7 | 45 | 17 | 0 |
| Native fitted | 389 | 3 | 2 | 1 | 2 | 3 | 1 | 12/12 | 5 | 3 | 4 | 0 |
| Rewired fitted | 389 | 16 | 14 | 6 | 6 | 12 | 16 | 36/70 | 7 | 46 | 17 | 0 |

[Frozen protocol and panels](FIT_OFFICIAL_LOCK.json) · [Scores and individual steering misses](FIT_OFFICIAL_RESULTS.json) · [2,048-update fitting report](FIT2_REPORT.md)
