# V27 results

**Source SHA256** `d340bf12327458e7e2cb6b14c23c5b5a2376737d5d05c89810d490ba2cc40472`

Predeclared: competitive selection lies above the dense frontier, and
random-subset plasticity recovers part but not all of it. If the budget-matched
dense control or the random-subset control matches competitive, the mechanism is
an update-budget effect and is retired as the novelty gate was.

## Is there headroom for adaptation at all?

_Status: **usable**_

- frozen anchor new-cohort: 46.17%
- best adaptation gain: +8.46 [+3.36, +13.56] (A1_dense_LL_r3)

Adaptation can buy at least 2.0 pp; frontier verdicts are interpretable.

## Does competitive plasticity move the frontier?

| Arm | above dense frontier | vs budget-matched | vs random-subset | vs bottom-k | gate |
|---|---|---|---|---|---|
| A2_competitive_FL_r0.1 | -1.19 [-2.51, +0.13] | +0.34 [-0.56, +1.23] | +0.29 [-0.40, +0.99] | +0.46 [-0.05, +0.98] | **update_budget_effect** |
| A2_competitive_FL_r0.25 | -0.32 [-4.48, +3.84] | +0.55 [-1.76, +2.85] | +0.97 [-0.47, +2.40] | +0.67 [-1.14, +2.48] | **update_budget_effect** |
| A2_competitive_FL_r0.5 | -0.22 [-2.93, +2.49] | +1.85 [+0.45, +3.25] | +1.56 [+0.38, +2.74] | +1.43 [-0.43, +3.30] | **partial** |
| A2_competitive_FL_r1 | -1.44 [-2.58, -0.30] | +0.21 [-1.95, +2.37] | +0.21 [-1.64, +2.06] | +1.60 [-0.17, +3.37] | **update_budget_effect** |
| A2_competitive_FL_r2 | -0.14 [-2.35, +2.07] | +0.46 [-1.65, +2.58] | +0.59 [-1.43, +2.61] | +3.16 [+0.15, +6.16] | **partial** |
| A2_competitive_FL_r3 | -1.71 [-2.62, -0.80] | +0.88 [-1.94, +3.71] | +0.08 [-2.61, +2.78] | +3.03 [+0.11, +5.95] | **partial** |
| A2_competitive_LL_r0.1 | -1.32 [-7.35, +4.72] | +0.76 [-2.37, +3.88] | -0.63 [-3.73, +2.47] | -0.34 [-3.95, +3.28] | **update_budget_effect** |
| A2_competitive_LL_r0.25 | -3.71 [-8.55, +1.14] | +1.18 [-3.10, +5.45] | +1.89 [-1.34, +5.13] | +1.09 [-2.11, +4.30] | **update_budget_effect** |
| A2_competitive_LL_r0.5 | +0.15 [-4.78, +5.08] | +2.86 [+0.61, +5.11] | +0.93 [-1.04, +2.89] | +2.69 [-0.73, +6.12] | **sparse_plasticity_effect** |
| A2_competitive_LL_r1 | +1.39 [-2.16, +4.93] | +2.15 [-0.91, +5.20] | +2.53 [+0.10, +4.95] | +4.84 [+1.12, +8.56] | **partial** |
| A2_competitive_LL_r2 | +0.46 [-4.35, +5.27] | +2.36 [-0.24, +4.95] | +1.05 [-0.84, +2.94] | +4.71 [+2.43, +7.00] | **partial** |
| A2_competitive_LL_r3 | +3.11 [-1.67, +7.89] | +3.03 [-0.70, +6.76] | +2.65 [-0.84, +6.15] | +4.12 [+0.11, +8.14] | **partial** |

## C1: does variance salience survive falsification?

_Verdict: **task_specific_salience_exploit**_

| Task variant | variance8 vs random8 (blended) |
|---|---|
| none | -10.08 [-25.59, +5.43] |
| permuted | +1.16 [-13.11, +15.42] |
| variance_matched | -8.33 [-21.21, +4.54] |
| distributed | +0.00 [-4.67, +4.67] |

## Interpretation limits

- A `no_headroom` status voids every frontier verdict below it. If the frozen
  anchor already solves the new cohort, the curve is a pure loss curve and no
  mechanism can be evaluated against it. Redesign the task first.
- `sparse_plasticity_effect` means competitive beats dense but not the
  sparsity-matched controls: the benefit is updating fewer cells, not
  choosing which. Anti-competitive bottom-k matching it is decisive.
- `update_budget_effect` means competitive plasticity is matched by simply
  updating fewer or weaker cells. That retires the mechanism; it does not
  mean sparse plasticity is useless, only that competition adds nothing.
- `controls_missing` means the deciding arm did not complete. It is not a pass.
- Frontier offsets are never extrapolated beyond the measured dense range.
- The A0/FF anchor has no adaptation, so every plasticity rule collapses to it.
  A nonzero contrast there would indicate a bug, not a finding.
- Six seeds, one small synthetic stream, unadjusted intervals. No scaling law.
