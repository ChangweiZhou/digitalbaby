# V26 results

**Status: completed.** Source SHA256 `159ed6992e37e99d591f82e37aaca06703541f5509f30ccb817baad89e5c1f7b`.

Predeclared prediction: afferent geometry and sparsification explain more of the
structured-transfer gap than temporal complexity. Intervals are descriptive paired
Student 95% intervals over seed differences, unadjusted for multiplicity.

## Reference points (V25, six seeds)

| Reference | Value |
|---|---:|
| structured order blind ceiling | 75.00% |
| structured observable | 79.15% |
| structured minimal blended | 64.88% |
| structured ordinary | 56.00% |
| association minimal base | 72.69% |
| association minimal next recurrence | 86.43% |
| association e3 local base | 68.14% |
| context minimal blended | 56.67% |
| context e3 blended | 56.84% |

## Block A gate

| Candidate | structured Δpp [CI95] | survives S1 | assoc Δpp | one-shot Δpp | gate | salience |
|---|---|---|---|---|---|---|
| A0B1 | -4.34 [-6.06, -2.61] | no | -3.62 [-6.41, -0.82] | +0.26 [-0.59, +1.11] | fails | - |
| A0B2 | -0.29 [-3.49, +2.90] | no | -7.15 [-9.09, -5.22] | +1.27 [+0.71, +1.83] | inconclusive | - |
| A1B0 | +4.48 [-9.75, +18.71] | no | -1.75 [-4.18, +0.68] | -0.06 [-0.67, +0.56] | inconclusive | - |
| A1B1 | -0.08 [-14.33, +14.16] | no | -2.63 [-4.89, -0.37] | +0.05 [-0.69, +0.78] | inconclusive | - |
| A2B0 | -5.64 [-20.65, +9.37] | no | -2.06 [-3.95, -0.18] | +0.11 [-0.49, +0.70] | inconclusive | task_specific_salience_exploit |
| A2B1 | -5.79 [-18.11, +6.54] | no | -4.97 [-8.58, -1.36] | +0.11 [-0.43, +0.66] | inconclusive | untested |
| A2B2 | -7.09 [-19.02, +4.84] | no | -10.48 [-13.02, -7.94] | +1.14 [+0.63, +1.66] | inconclusive | untested |

## Block B gate

_Block verdict: **representation_benefit_present**_

| Arm | new cohort Δpp | old cohort Δpp | one-shot Δpp | gate |
|---|---|---|---|---|
| R1_live | +7.70 [+1.92, +13.48] | +2.48 [-3.34, +8.31] | -1.41 [-1.84, -0.97] | pareto_improvement |
| R2_frozen_address | +4.38 [+2.33, +6.43] | -0.38 [-6.24, +5.48] | -0.65 [-0.92, -0.37] | benefit_without_compatibility |
| R3_stable_scaffold | +7.03 [+4.28, +9.78] | -0.08 [-4.52, +4.36] | -0.87 [-1.38, -0.36] | pareto_improvement |

## Interpretation limits

- If the Block-B verdict is `no_representation_benefit`, the local rule or the
  shift task failed to create a representation-learning problem. Do NOT read the
  retention columns as evidence about compatibility in that case.
- A salience candidate labelled `task_specific_salience_exploit` won only because
  the generator gives the informative coordinate 2.17x the nuisance variance. That
  is not a representation result.
- Slow writes appear only as a labelled comparator. V25 showed +1.8 pp final costs
  -8.5 pp prequential and -20 pp next-recurrence; ordinary writes are the substrate.
- Hard and blended views share one learned state and are not independent runs.
- These are small synthetic streams. No scaling law is claimed or measurable here.
