# E44, job L2, the sweep's long lives: tables (interim_world1)

Written by `smallworld/l2_sweep_report.py tables interim_world1` from the score files under `exp45/e44/L2/sweep/scores/` (the sweep's lives) and `exp45/e44/L2/sweep/main/` (the main worker's lives under back-propagation, read from `claude/e44-l2-first-long-lives` at 56e77d86). Every figure is the scorer's (`smallworld/l2_score.py`, unchanged). A difference is the sweep's figure minus back-propagation's. Intervals are two-sided 90% intervals (Student's t over worlds).

Lives scored: `sweep` worlds 1, 2; `naive` worlds 1, 2; `sweep+replay` worlds 1; `replay` worlds 1.

## The births

The two rules start from different births: each life starts from E41's half-width birth (H = 256, U = 128, taught fraction 0.50) made under its own rule. A: per cent right on E41's held-out panel of add89 and sub89; R: per cent right on E41's 2,000 held-out reduction pairs (the receipts' final rows). Worlds 1 and 4 use seed 73129, world 2 seed 93683, world 3 seed 116089.

| Seed | Rule of the birth | A | R | Updates |
|---|---|---|---|---|
| 73129 | back-propagation | 94.50 | 94.65 | 40960 |
| 73129 | the sweep | 95.83 | 91.95 | 40960 |
| 93683 | back-propagation | 96.17 | 94.25 | 40960 |
| 93683 | the sweep | 93.50 | 92.55 | 40960 |
| 116089 | back-propagation | 96.17 | 93.00 | 40960 |
| 116089 | the sweep | 92.83 | 90.90 | 40960 |

## The quantities of each life (D2 to D7)

Late level: the mean score on reduction items over the last quarter of the visits (D4), beside the blind predictor's and the level by kind of visit. Score trend: the slope of the score on ln(visit + 1) over the returns of the second half (D5); beside it the slope over all visits of that half (not used). First-visit level: D6, the median over the families met. Trend of learning: the slope of D6 on ln n (D7), beside the coefficient of ln n in the joint fit with the logarithm of the taught pairs.

| Visits | Arm | World | Families | Late level | Blind | First visits | Continuations | Returns | Score trend | Returns in it | Slope, all visits | Median first-visit level | Trend of learning | Joint fit, ln n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | `sweep` | 1 | 17 | 0.589 | 0.014 | 0.496 | 0.761 | 0.566 | -0.0549 | 103 | -0.0155 | 0.653 | -0.0147 | -0.0188 |
| 256 | `sweep` | 2 | 13 | 0.512 | 0.015 | 0.287 | 0.645 | 0.482 | -0.1232 | 95 | -0.0680 | 0.440 | -0.0337 | -0.0482 |
| 256 | `naive` | 1 | 17 | 0.582 | 0.014 | 0.587 | 0.738 | 0.554 | -0.0281 | 103 | +0.0673 | 0.653 | -0.0062 | -0.0151 |
| 256 | `naive` | 2 | 13 | 0.696 | 0.015 | 0.485 | 0.825 | 0.667 | -0.0154 | 95 | +0.0450 | 0.685 | +0.0143 | +0.0011 |
| 256 | `sweep+replay` | 1 | 17 | 0.665 | 0.014 | 0.558 | 0.723 | 0.664 | +0.0188 | 103 | +0.0427 | 0.615 | +0.0575 | +0.0536 |
| 256 | `replay` | 1 | 17 | 0.577 | 0.014 | 0.480 | 0.638 | 0.574 | -0.0675 | 103 | +0.0013 | 0.393 | -0.0230 | -0.0300 |
| 1024 | `sweep` | 1 | 52 | 0.650 | 0.015 | 0.677 | 0.814 | 0.630 | +0.0633 | 436 | +0.0657 | 0.773 | +0.0822 | +0.0831 |
| 1024 | `sweep` | 2 | 52 | 0.628 | 0.016 | 0.495 | 0.776 | 0.614 | +0.0979 | 430 | +0.1117 | 0.707 | +0.0696 | +0.0695 |
| 1024 | `naive` | 1 | 52 | 0.669 | 0.015 | 0.676 | 0.814 | 0.652 | -0.0094 | 436 | -0.0075 | 0.757 | +0.1064 | +0.1102 |
| 1024 | `naive` | 2 | 52 | 0.756 | 0.016 | 0.675 | 0.886 | 0.742 | +0.0418 | 430 | +0.0481 | 0.841 | +0.1108 | +0.1086 |
| 1024 | `sweep+replay` | 1 | 52 | 0.784 | 0.015 | 0.679 | 0.801 | 0.786 | +0.0459 | 436 | +0.0494 | 0.694 | +0.1126 | +0.1135 |
| 1024 | `replay` | 1 | 52 | 0.730 | 0.015 | 0.651 | 0.769 | 0.728 | +0.0790 | 436 | +0.0867 | 0.595 | +0.1035 | +0.1057 |

## Kept at return (D8), kept on the list (D9) and relearning at return (D10)

D8: the share right on the reduction items of the first update of a later visit, pooled over the visits of an absence class (right / items). D9: the family's share right on its 200 list items divided by its share at the end of its most recent visit; mean and median over the points, and the points left out because the earlier share was 0. D10: the family's figure over the first 16 updates and over the last 64 updates of the visit, beside the last 64 updates of the family's previous visit, averaged over the visits.

**Up to 256 visits.**

| Arm | World | Absence (visits) | Visits | D8 kept at return | D10 first 16 updates | D10 last 64 | D10 previous visit's last 64 | D9 points | D9 left out | D9 mean | D9 median | List share right |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 | 1 visit | 68 | 0.788 (216/274) | 0.796 | 0.810 | 0.790 | 25 | 0 | 0.015 | 0.007 | 0.010 |
| `sweep` | 1 | 2 to 4 | 86 | 0.009 (3/346) | 0.093 | 0.739 | 0.723 | 51 | 0 | 0.018 | 0.000 | 0.015 |
| `sweep` | 1 | 5 to 32 | 66 | 0.012 (3/244) | 0.087 | 0.706 | 0.723 | 172 | 0 | 0.020 | 0.000 | 0.015 |
| `sweep` | 1 | 33 or more | 19 | 0.013 (1/79) | 0.061 | 0.692 | 0.674 | 11 | 0 | 0.000 | 0.000 | 0.000 |
| `sweep` | 1 | 5 or more | 85 | 0.012 (4/323) | 0.081 | 0.703 | 0.712 | 183 | 0 | 0.019 | 0.000 | 0.014 |
| `sweep` | 2 | 1 visit | 70 | 0.598 (147/246) | 0.637 | 0.674 | 0.651 | 22 | 0 | 0.053 | 0.007 | 0.034 |
| `sweep` | 2 | 2 to 4 | 90 | 0.065 (21/322) | 0.175 | 0.661 | 0.654 | 50 | 0 | 0.068 | 0.007 | 0.042 |
| `sweep` | 2 | 5 to 32 | 65 | 0.031 (8/254) | 0.104 | 0.608 | 0.618 | 138 | 0 | 0.024 | 0.000 | 0.016 |
| `sweep` | 2 | 33 or more | 18 | 0.000 (0/60) | 0.072 | 0.594 | 0.642 | 7 | 0 | 0.065 | 0.029 | 0.050 |
| `sweep` | 2 | 5 or more | 83 | 0.025 (8/314) | 0.097 | 0.605 | 0.624 | 145 | 0 | 0.026 | 0.000 | 0.018 |
| `naive` | 1 | 1 visit | 68 | 0.785 (215/274) | 0.782 | 0.799 | 0.781 | 25 | 0 | 0.037 | 0.011 | 0.028 |
| `naive` | 1 | 2 to 4 | 86 | 0.029 (10/346) | 0.133 | 0.707 | 0.692 | 51 | 0 | 0.033 | 0.011 | 0.024 |
| `naive` | 1 | 5 to 32 | 66 | 0.016 (4/244) | 0.090 | 0.650 | 0.661 | 172 | 0 | 0.024 | 0.000 | 0.016 |
| `naive` | 1 | 33 or more | 19 | 0.038 (3/79) | 0.055 | 0.603 | 0.572 | 11 | 0 | 0.002 | 0.000 | 0.001 |
| `naive` | 1 | 5 or more | 85 | 0.022 (7/323) | 0.082 | 0.640 | 0.641 | 183 | 0 | 0.023 | 0.000 | 0.015 |
| `naive` | 2 | 1 visit | 70 | 0.752 (185/246) | 0.782 | 0.801 | 0.782 | 22 | 0 | 0.048 | 0.021 | 0.038 |
| `naive` | 2 | 2 to 4 | 90 | 0.037 (12/322) | 0.204 | 0.801 | 0.782 | 50 | 0 | 0.039 | 0.000 | 0.033 |
| `naive` | 2 | 5 to 32 | 65 | 0.012 (3/254) | 0.116 | 0.748 | 0.756 | 138 | 0 | 0.012 | 0.000 | 0.010 |
| `naive` | 2 | 33 or more | 18 | 0.000 (0/60) | 0.102 | 0.764 | 0.785 | 7 | 0 | 0.041 | 0.051 | 0.033 |
| `naive` | 2 | 5 or more | 83 | 0.010 (3/314) | 0.113 | 0.751 | 0.762 | 145 | 0 | 0.014 | 0.000 | 0.011 |
| `sweep+replay` | 1 | 1 visit | 68 | 0.785 (215/274) | 0.774 | 0.773 | 0.755 | 25 | 0 | 0.725 | 0.800 | 0.491 |
| `sweep+replay` | 1 | 2 to 4 | 86 | 0.436 (151/346) | 0.507 | 0.678 | 0.662 | 51 | 0 | 0.584 | 0.634 | 0.397 |
| `sweep+replay` | 1 | 5 to 32 | 66 | 0.447 (109/244) | 0.507 | 0.679 | 0.671 | 172 | 0 | 0.577 | 0.601 | 0.387 |
| `sweep+replay` | 1 | 33 or more | 19 | 0.481 (38/79) | 0.478 | 0.698 | 0.664 | 11 | 0 | 0.490 | 0.540 | 0.326 |
| `sweep+replay` | 1 | 5 or more | 85 | 0.455 (147/323) | 0.500 | 0.683 | 0.669 | 183 | 0 | 0.572 | 0.600 | 0.383 |
| `replay` | 1 | 1 visit | 68 | 0.763 (209/274) | 0.738 | 0.755 | 0.740 | 25 | 0 | 0.750 | 0.765 | 0.471 |
| `replay` | 1 | 2 to 4 | 86 | 0.483 (167/346) | 0.523 | 0.638 | 0.625 | 51 | 0 | 0.678 | 0.725 | 0.438 |
| `replay` | 1 | 5 to 32 | 66 | 0.426 (104/244) | 0.468 | 0.601 | 0.594 | 172 | 0 | 0.639 | 0.674 | 0.388 |
| `replay` | 1 | 33 or more | 19 | 0.456 (36/79) | 0.409 | 0.587 | 0.539 | 11 | 0 | 0.619 | 0.618 | 0.339 |
| `replay` | 1 | 5 or more | 85 | 0.433 (140/323) | 0.455 | 0.598 | 0.582 | 183 | 0 | 0.638 | 0.672 | 0.385 |

**Up to 1,024 visits.**

| Arm | World | Absence (visits) | Visits | D8 kept at return | D10 first 16 updates | D10 last 64 | D10 previous visit's last 64 | D9 points | D9 left out | D9 mean | D9 median | List share right |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 | 1 visit | 151 | 0.786 (448/570) | 0.791 | 0.816 | 0.782 | 113 | 0 | 0.022 | 0.005 | 0.016 |
| `sweep` | 1 | 2 to 4 | 253 | 0.009 (9/983) | 0.105 | 0.797 | 0.778 | 263 | 0 | 0.011 | 0.000 | 0.009 |
| `sweep` | 1 | 5 to 32 | 305 | 0.012 (14/1147) | 0.078 | 0.763 | 0.773 | 1481 | 0 | 0.017 | 0.000 | 0.013 |
| `sweep` | 1 | 33 or more | 263 | 0.018 (18/1021) | 0.079 | 0.758 | 0.765 | 2021 | 0 | 0.014 | 0.000 | 0.011 |
| `sweep` | 1 | 5 or more | 568 | 0.015 (32/2168) | 0.078 | 0.760 | 0.769 | 3502 | 0 | 0.015 | 0.000 | 0.012 |
| `sweep` | 2 | 1 visit | 158 | 0.684 (415/607) | 0.685 | 0.720 | 0.692 | 108 | 0 | 0.033 | 0.003 | 0.023 |
| `sweep` | 2 | 2 to 4 | 264 | 0.036 (37/1016) | 0.131 | 0.715 | 0.700 | 253 | 0 | 0.033 | 0.000 | 0.024 |
| `sweep` | 2 | 5 to 32 | 294 | 0.020 (23/1123) | 0.090 | 0.704 | 0.699 | 1455 | 0 | 0.019 | 0.000 | 0.014 |
| `sweep` | 2 | 33 or more | 256 | 0.020 (19/936) | 0.079 | 0.687 | 0.696 | 1755 | 0 | 0.018 | 0.000 | 0.014 |
| `sweep` | 2 | 5 or more | 550 | 0.020 (42/2059) | 0.085 | 0.696 | 0.697 | 3210 | 0 | 0.019 | 0.000 | 0.014 |
| `naive` | 1 | 1 visit | 151 | 0.807 (460/570) | 0.782 | 0.810 | 0.779 | 113 | 0 | 0.022 | 0.000 | 0.017 |
| `naive` | 1 | 2 to 4 | 253 | 0.026 (26/983) | 0.142 | 0.781 | 0.754 | 263 | 0 | 0.017 | 0.000 | 0.014 |
| `naive` | 1 | 5 to 32 | 305 | 0.016 (18/1147) | 0.094 | 0.750 | 0.767 | 1481 | 0 | 0.014 | 0.000 | 0.011 |
| `naive` | 1 | 33 or more | 263 | 0.015 (15/1021) | 0.095 | 0.743 | 0.751 | 2021 | 0 | 0.012 | 0.000 | 0.010 |
| `naive` | 1 | 5 or more | 568 | 0.015 (33/2168) | 0.095 | 0.747 | 0.759 | 3502 | 0 | 0.013 | 0.000 | 0.010 |
| `naive` | 2 | 1 visit | 158 | 0.832 (505/607) | 0.827 | 0.844 | 0.823 | 108 | 0 | 0.021 | 0.000 | 0.018 |
| `naive` | 2 | 2 to 4 | 264 | 0.018 (18/1016) | 0.160 | 0.851 | 0.834 | 253 | 0 | 0.017 | 0.000 | 0.015 |
| `naive` | 2 | 5 to 32 | 294 | 0.010 (11/1123) | 0.114 | 0.833 | 0.833 | 1455 | 0 | 0.010 | 0.000 | 0.008 |
| `naive` | 2 | 33 or more | 256 | 0.007 (7/936) | 0.107 | 0.836 | 0.847 | 1755 | 0 | 0.008 | 0.000 | 0.007 |
| `naive` | 2 | 5 or more | 550 | 0.009 (18/2059) | 0.111 | 0.834 | 0.840 | 3210 | 0 | 0.009 | 0.000 | 0.008 |
| `sweep+replay` | 1 | 1 visit | 151 | 0.772 (440/570) | 0.785 | 0.792 | 0.771 | 113 | 0 | 0.753 | 0.804 | 0.560 |
| `sweep+replay` | 1 | 2 to 4 | 253 | 0.581 (571/983) | 0.633 | 0.761 | 0.749 | 263 | 0 | 0.669 | 0.735 | 0.501 |
| `sweep+replay` | 1 | 5 to 32 | 305 | 0.498 (571/1147) | 0.596 | 0.767 | 0.763 | 1481 | 0 | 0.627 | 0.665 | 0.473 |
| `sweep+replay` | 1 | 33 or more | 263 | 0.509 (520/1021) | 0.610 | 0.780 | 0.779 | 2021 | 0 | 0.643 | 0.684 | 0.489 |
| `sweep+replay` | 1 | 5 or more | 568 | 0.503 (1091/2168) | 0.602 | 0.773 | 0.770 | 3502 | 0 | 0.636 | 0.676 | 0.482 |
| `replay` | 1 | 1 visit | 151 | 0.739 (421/570) | 0.730 | 0.758 | 0.735 | 113 | 0 | 0.759 | 0.783 | 0.530 |
| `replay` | 1 | 2 to 4 | 253 | 0.527 (518/983) | 0.581 | 0.701 | 0.688 | 263 | 0 | 0.688 | 0.735 | 0.489 |
| `replay` | 1 | 5 to 32 | 305 | 0.465 (533/1147) | 0.540 | 0.703 | 0.703 | 1481 | 0 | 0.642 | 0.670 | 0.453 |
| `replay` | 1 | 33 or more | 263 | 0.481 (491/1021) | 0.536 | 0.709 | 0.700 | 2021 | 0 | 0.642 | 0.662 | 0.461 |
| `replay` | 1 | 5 or more | 568 | 0.472 (1024/2168) | 0.538 | 0.706 | 0.701 | 3502 | 0 | 0.642 | 0.665 | 0.458 |

## What the wrong answers are (D14)

Pooled over the worlds scored for the arm. Stream: the reduction items of the first update of a later visit. List: the list's rows of D9. Classes tried in the scorer's order; the chance line is the never-met modulus (family 55's) tried in the place of the family taught last.

**Up to 256 visits, stream.**

| Arm | Absence | Answers | right | no reduction | modulus taught last | earlier modulus | 89 | an operand | never-met (control) | other two digits | not two digits | Chance line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 visit | 520 | 0.698 | 0.000 | 0.000 | 0.019 | 0.000 | 0.012 | 0.019 | 0.252 | 0.000 | 0.019 |
| `sweep` | 2 to 4 | 668 | 0.036 | 0.096 | 0.274 | 0.057 | 0.022 | 0.015 | 0.021 | 0.479 | 0.000 | 0.024 |
| `sweep` | 5 to 32 | 498 | 0.022 | 0.090 | 0.277 | 0.060 | 0.020 | 0.032 | 0.010 | 0.486 | 0.002 | 0.010 |
| `sweep` | 33 or more | 139 | 0.007 | 0.129 | 0.338 | 0.101 | 0.000 | 0.007 | 0.007 | 0.410 | 0.000 | 0.007 |
| `naive` | 1 visit | 520 | 0.769 | 0.002 | 0.000 | 0.015 | 0.000 | 0.004 | 0.010 | 0.200 | 0.000 | 0.010 |
| `naive` | 2 to 4 | 668 | 0.033 | 0.058 | 0.322 | 0.057 | 0.027 | 0.018 | 0.004 | 0.481 | 0.000 | 0.007 |
| `naive` | 5 to 32 | 498 | 0.014 | 0.070 | 0.353 | 0.080 | 0.008 | 0.028 | 0.006 | 0.440 | 0.000 | 0.006 |
| `naive` | 33 or more | 139 | 0.022 | 0.165 | 0.353 | 0.079 | 0.022 | 0.036 | 0.014 | 0.309 | 0.000 | 0.014 |
| `sweep+replay` | 1 visit | 274 | 0.785 | 0.007 | 0.000 | 0.029 | 0.000 | 0.000 | 0.004 | 0.175 | 0.000 | 0.004 |
| `sweep+replay` | 2 to 4 | 346 | 0.436 | 0.017 | 0.029 | 0.061 | 0.000 | 0.009 | 0.026 | 0.422 | 0.000 | 0.026 |
| `sweep+replay` | 5 to 32 | 244 | 0.447 | 0.029 | 0.025 | 0.070 | 0.008 | 0.008 | 0.008 | 0.406 | 0.000 | 0.008 |
| `sweep+replay` | 33 or more | 79 | 0.481 | 0.013 | 0.000 | 0.152 | 0.000 | 0.013 | 0.013 | 0.329 | 0.000 | 0.013 |
| `replay` | 1 visit | 274 | 0.763 | 0.000 | 0.000 | 0.026 | 0.000 | 0.004 | 0.007 | 0.201 | 0.000 | 0.007 |
| `replay` | 2 to 4 | 346 | 0.483 | 0.006 | 0.012 | 0.064 | 0.009 | 0.009 | 0.020 | 0.399 | 0.000 | 0.020 |
| `replay` | 5 to 32 | 244 | 0.426 | 0.020 | 0.033 | 0.078 | 0.000 | 0.016 | 0.008 | 0.418 | 0.000 | 0.008 |
| `replay` | 33 or more | 79 | 0.456 | 0.013 | 0.013 | 0.101 | 0.000 | 0.000 | 0.038 | 0.380 | 0.000 | 0.038 |

**Up to 256 visits, list.**

| Arm | Absence | Answers | right | no reduction | modulus taught last | earlier modulus | 89 | an operand | never-met (control) | other two digits | not two digits | Chance line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 visit | 9400 | 0.021 | 0.086 | 0.284 | 0.079 | 0.019 | 0.023 | 0.019 | 0.467 | 0.002 | 0.020 |
| `sweep` | 2 to 4 | 20200 | 0.028 | 0.079 | 0.292 | 0.078 | 0.017 | 0.019 | 0.019 | 0.466 | 0.002 | 0.020 |
| `sweep` | 5 to 32 | 62000 | 0.016 | 0.101 | 0.311 | 0.093 | 0.012 | 0.015 | 0.015 | 0.436 | 0.002 | 0.015 |
| `sweep` | 33 or more | 3600 | 0.019 | 0.200 | 0.370 | 0.086 | 0.002 | 0.010 | 0.019 | 0.294 | 0.000 | 0.019 |
| `naive` | 1 visit | 9400 | 0.033 | 0.062 | 0.340 | 0.069 | 0.011 | 0.018 | 0.010 | 0.456 | 0.000 | 0.011 |
| `naive` | 2 to 4 | 20200 | 0.028 | 0.073 | 0.347 | 0.069 | 0.012 | 0.025 | 0.014 | 0.432 | 0.000 | 0.014 |
| `naive` | 5 to 32 | 62000 | 0.013 | 0.091 | 0.405 | 0.068 | 0.007 | 0.017 | 0.010 | 0.388 | 0.000 | 0.010 |
| `naive` | 33 or more | 3600 | 0.014 | 0.176 | 0.471 | 0.078 | 0.003 | 0.009 | 0.005 | 0.244 | 0.000 | 0.006 |
| `sweep+replay` | 1 visit | 5000 | 0.491 | 0.019 | 0.014 | 0.076 | 0.005 | 0.011 | 0.006 | 0.377 | 0.000 | 0.006 |
| `sweep+replay` | 2 to 4 | 10200 | 0.397 | 0.021 | 0.017 | 0.088 | 0.006 | 0.013 | 0.009 | 0.449 | 0.000 | 0.009 |
| `sweep+replay` | 5 to 32 | 34400 | 0.387 | 0.021 | 0.030 | 0.129 | 0.005 | 0.012 | 0.014 | 0.403 | 0.000 | 0.014 |
| `sweep+replay` | 33 or more | 2200 | 0.326 | 0.020 | 0.005 | 0.138 | 0.003 | 0.011 | 0.004 | 0.493 | 0.000 | 0.005 |
| `replay` | 1 visit | 5000 | 0.471 | 0.010 | 0.008 | 0.087 | 0.001 | 0.012 | 0.006 | 0.406 | 0.000 | 0.006 |
| `replay` | 2 to 4 | 10200 | 0.438 | 0.015 | 0.010 | 0.086 | 0.002 | 0.017 | 0.014 | 0.418 | 0.000 | 0.015 |
| `replay` | 5 to 32 | 34400 | 0.388 | 0.012 | 0.018 | 0.105 | 0.002 | 0.018 | 0.015 | 0.441 | 0.000 | 0.015 |
| `replay` | 33 or more | 2200 | 0.339 | 0.005 | 0.004 | 0.133 | 0.000 | 0.015 | 0.015 | 0.490 | 0.000 | 0.015 |

**Up to 1,024 visits, stream.**

| Arm | Absence | Answers | right | no reduction | modulus taught last | earlier modulus | 89 | an operand | never-met (control) | other two digits | not two digits | Chance line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 visit | 1177 | 0.733 | 0.002 | 0.000 | 0.065 | 0.001 | 0.008 | 0.011 | 0.181 | 0.000 | 0.011 |
| `sweep` | 2 to 4 | 1999 | 0.023 | 0.091 | 0.395 | 0.158 | 0.013 | 0.010 | 0.013 | 0.296 | 0.002 | 0.014 |
| `sweep` | 5 to 32 | 2270 | 0.016 | 0.109 | 0.406 | 0.193 | 0.010 | 0.013 | 0.006 | 0.244 | 0.002 | 0.007 |
| `sweep` | 33 or more | 1957 | 0.019 | 0.121 | 0.418 | 0.237 | 0.008 | 0.010 | 0.004 | 0.176 | 0.008 | 0.005 |
| `naive` | 1 visit | 1177 | 0.820 | 0.003 | 0.000 | 0.045 | 0.001 | 0.004 | 0.004 | 0.122 | 0.000 | 0.004 |
| `naive` | 2 to 4 | 1999 | 0.022 | 0.091 | 0.477 | 0.130 | 0.014 | 0.011 | 0.004 | 0.252 | 0.000 | 0.006 |
| `naive` | 5 to 32 | 2270 | 0.013 | 0.115 | 0.514 | 0.149 | 0.005 | 0.011 | 0.005 | 0.189 | 0.000 | 0.006 |
| `naive` | 33 or more | 1957 | 0.011 | 0.129 | 0.515 | 0.183 | 0.011 | 0.013 | 0.005 | 0.134 | 0.000 | 0.007 |
| `sweep+replay` | 1 visit | 570 | 0.772 | 0.004 | 0.000 | 0.084 | 0.000 | 0.004 | 0.002 | 0.135 | 0.000 | 0.002 |
| `sweep+replay` | 2 to 4 | 983 | 0.581 | 0.012 | 0.015 | 0.137 | 0.003 | 0.005 | 0.012 | 0.234 | 0.000 | 0.012 |
| `sweep+replay` | 5 to 32 | 1147 | 0.498 | 0.026 | 0.015 | 0.243 | 0.003 | 0.004 | 0.006 | 0.204 | 0.000 | 0.006 |
| `sweep+replay` | 33 or more | 1021 | 0.509 | 0.026 | 0.008 | 0.284 | 0.005 | 0.004 | 0.009 | 0.155 | 0.000 | 0.010 |
| `replay` | 1 visit | 570 | 0.739 | 0.002 | 0.000 | 0.089 | 0.005 | 0.005 | 0.007 | 0.153 | 0.000 | 0.007 |
| `replay` | 2 to 4 | 983 | 0.527 | 0.015 | 0.012 | 0.164 | 0.011 | 0.005 | 0.012 | 0.252 | 0.001 | 0.012 |
| `replay` | 5 to 32 | 1147 | 0.465 | 0.027 | 0.017 | 0.255 | 0.015 | 0.005 | 0.013 | 0.203 | 0.000 | 0.013 |
| `replay` | 33 or more | 1021 | 0.481 | 0.023 | 0.013 | 0.285 | 0.011 | 0.006 | 0.010 | 0.172 | 0.000 | 0.010 |

**Up to 1,024 visits, list.**

| Arm | Absence | Answers | right | no reduction | modulus taught last | earlier modulus | 89 | an operand | never-met (control) | other two digits | not two digits | Chance line |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 visit | 44200 | 0.019 | 0.095 | 0.393 | 0.214 | 0.012 | 0.011 | 0.008 | 0.245 | 0.003 | 0.010 |
| `sweep` | 2 to 4 | 103200 | 0.016 | 0.099 | 0.405 | 0.208 | 0.012 | 0.010 | 0.008 | 0.238 | 0.003 | 0.010 |
| `sweep` | 5 to 32 | 587200 | 0.014 | 0.110 | 0.408 | 0.228 | 0.010 | 0.008 | 0.005 | 0.215 | 0.003 | 0.007 |
| `sweep` | 33 or more | 755200 | 0.012 | 0.110 | 0.423 | 0.288 | 0.007 | 0.005 | 0.004 | 0.147 | 0.004 | 0.006 |
| `naive` | 1 visit | 44200 | 0.017 | 0.103 | 0.499 | 0.160 | 0.010 | 0.008 | 0.005 | 0.197 | 0.000 | 0.007 |
| `naive` | 2 to 4 | 103200 | 0.014 | 0.112 | 0.521 | 0.148 | 0.008 | 0.009 | 0.007 | 0.180 | 0.000 | 0.008 |
| `naive` | 5 to 32 | 587200 | 0.010 | 0.126 | 0.528 | 0.164 | 0.007 | 0.006 | 0.005 | 0.154 | 0.000 | 0.006 |
| `naive` | 33 or more | 755200 | 0.009 | 0.126 | 0.553 | 0.206 | 0.006 | 0.004 | 0.004 | 0.092 | 0.000 | 0.006 |
| `sweep+replay` | 1 visit | 22600 | 0.560 | 0.016 | 0.012 | 0.210 | 0.005 | 0.005 | 0.007 | 0.184 | 0.000 | 0.007 |
| `sweep+replay` | 2 to 4 | 52600 | 0.501 | 0.024 | 0.015 | 0.233 | 0.007 | 0.006 | 0.006 | 0.208 | 0.000 | 0.007 |
| `sweep+replay` | 5 to 32 | 296200 | 0.473 | 0.022 | 0.015 | 0.284 | 0.006 | 0.006 | 0.009 | 0.185 | 0.000 | 0.010 |
| `sweep+replay` | 33 or more | 404200 | 0.489 | 0.019 | 0.015 | 0.333 | 0.006 | 0.004 | 0.007 | 0.126 | 0.000 | 0.011 |
| `replay` | 1 visit | 22600 | 0.530 | 0.015 | 0.011 | 0.229 | 0.007 | 0.006 | 0.006 | 0.196 | 0.000 | 0.007 |
| `replay` | 2 to 4 | 52600 | 0.489 | 0.023 | 0.011 | 0.258 | 0.008 | 0.007 | 0.009 | 0.193 | 0.000 | 0.010 |
| `replay` | 5 to 32 | 296200 | 0.453 | 0.021 | 0.013 | 0.298 | 0.008 | 0.006 | 0.009 | 0.192 | 0.000 | 0.010 |
| `replay` | 33 or more | 404200 | 0.461 | 0.021 | 0.015 | 0.355 | 0.010 | 0.004 | 0.010 | 0.123 | 0.000 | 0.013 |

## Kept at return by the number of earlier visits to the family (D15)

Pooled over the worlds scored for the arm. In each cell: kept at return (right / items; returns), and after the semicolon the mean share right over the first 16 updates of the return.

**Up to 256 visits.**

| Arm | Absence (visits) | 1 earlier | 2 to 3 earlier | 4 to 7 earlier | 8 to 15 earlier | 16 or more earlier |
|---|---|---|---|---|---|---|
| `sweep` | 2 to 4 | 0.000 (0/37; 11); 0.092 | 0.000 (0/99; 24); 0.069 | 0.043 (5/115; 36); 0.104 | 0.024 (4/168; 44); 0.131 | 0.060 (15/249; 61); 0.189 |
| `sweep` | 5 to 32 | 0.000 (0/20; 6); 0.144 | 0.000 (0/65; 18); 0.089 | 0.008 (1/125; 29); 0.087 | 0.014 (2/141; 38); 0.064 | 0.054 (8/147; 40); 0.128 |
| `sweep` | 33 or more | 0.000 (0/4; 3); 0.030 | 0.000 (0/13; 3); 0.016 | 0.000 (0/32; 9); 0.039 | 0.000 (0/26; 8); 0.036 | 0.016 (1/64; 14); 0.120 |
| `naive` | 2 to 4 | 0.000 (0/37; 11); 0.113 | 0.010 (1/99; 24); 0.086 | 0.026 (3/115; 36); 0.130 | 0.048 (8/168; 44); 0.172 | 0.040 (10/249; 61); 0.234 |
| `naive` | 5 to 32 | 0.000 (0/20; 6); 0.110 | 0.031 (2/65; 18); 0.068 | 0.024 (3/125; 29); 0.090 | 0.007 (1/141; 38); 0.098 | 0.007 (1/147; 40); 0.132 |
| `naive` | 33 or more | 0.000 (0/4; 3); 0.005 | 0.231 (3/13; 3); 0.071 | 0.000 (0/32; 9); 0.023 | 0.000 (0/26; 8); 0.073 | 0.000 (0/64; 14); 0.133 |
| `sweep+replay` | 2 to 4 | 0.087 (2/23; 6); 0.173 | 0.333 (24/72; 16); 0.361 | 0.321 (17/53; 16); 0.449 | 0.487 (37/76; 18); 0.537 | 0.582 (71/122; 30); 0.665 |
| `sweep+replay` | 5 to 32 | 0.000 (0/11; 4); 0.345 | 0.241 (7/29; 8); 0.325 | 0.410 (32/78; 17); 0.456 | 0.480 (36/75; 22); 0.551 | 0.667 (34/51; 15); 0.640 |
| `sweep+replay` | 33 or more | 0.000 (0/3; 2); 0.241 | 0.429 (3/7; 2); 0.290 | 0.409 (9/22; 5); 0.404 | 0.455 (5/11; 2); 0.475 | 0.583 (21/36; 8); 0.631 |
| `replay` | 2 to 4 | 0.043 (1/23; 6); 0.152 | 0.361 (26/72; 16); 0.379 | 0.396 (21/53; 16); 0.427 | 0.368 (28/76; 18); 0.498 | 0.746 (91/122; 30); 0.741 |
| `replay` | 5 to 32 | 0.091 (1/11; 4); 0.382 | 0.172 (5/29; 8); 0.257 | 0.321 (25/78; 17); 0.352 | 0.507 (38/75; 22); 0.506 | 0.686 (35/51; 15); 0.678 |
| `replay` | 33 or more | 0.000 (0/3; 2); 0.097 | 0.286 (2/7; 2); 0.248 | 0.364 (8/22; 5); 0.274 | 0.364 (4/11; 2); 0.304 | 0.611 (22/36; 8); 0.638 |

**Up to 1,024 visits.**

| Arm | Absence (visits) | 1 earlier | 2 to 3 earlier | 4 to 7 earlier | 8 to 15 earlier | 16 or more earlier |
|---|---|---|---|---|---|---|
| `sweep` | 2 to 4 | 0.000 (0/107; 28); 0.092 | 0.005 (1/205; 53); 0.096 | 0.018 (5/273; 81); 0.098 | 0.023 (12/524; 135); 0.115 | 0.031 (28/890; 220); 0.136 |
| `sweep` | 5 to 32 | 0.018 (2/114; 31); 0.089 | 0.008 (2/259; 71); 0.080 | 0.020 (9/456; 119); 0.080 | 0.016 (8/506; 135); 0.084 | 0.017 (16/935; 243); 0.086 |
| `sweep` | 33 or more | 0.000 (0/100; 30); 0.067 | 0.034 (7/208; 55); 0.095 | 0.011 (4/373; 95); 0.056 | 0.011 (5/455; 130); 0.068 | 0.026 (21/821; 209); 0.094 |
| `naive` | 2 to 4 | 0.019 (2/107; 28); 0.113 | 0.015 (3/205; 53); 0.134 | 0.022 (6/273; 81); 0.126 | 0.029 (15/524; 135); 0.149 | 0.020 (18/890; 220); 0.170 |
| `naive` | 5 to 32 | 0.000 (0/114; 31); 0.093 | 0.015 (4/259; 71); 0.087 | 0.018 (8/456; 119); 0.105 | 0.014 (7/506; 135); 0.108 | 0.011 (10/935; 243); 0.108 |
| `naive` | 33 or more | 0.000 (0/100; 30); 0.074 | 0.019 (4/208; 55); 0.126 | 0.005 (2/373; 95); 0.078 | 0.007 (3/455; 130); 0.092 | 0.016 (13/821; 209); 0.114 |
| `sweep+replay` | 2 to 4 | 0.196 (9/46; 12); 0.348 | 0.336 (37/110; 28); 0.435 | 0.418 (59/141; 40); 0.557 | 0.641 (182/284; 75); 0.659 | 0.706 (284/402; 98); 0.735 |
| `sweep+replay` | 5 to 32 | 0.081 (5/62; 18); 0.314 | 0.189 (21/111; 31); 0.428 | 0.394 (87/221; 60); 0.518 | 0.527 (144/273; 73); 0.625 | 0.654 (314/480; 123); 0.700 |
| `sweep+replay` | 33 or more | 0.058 (3/52; 15); 0.299 | 0.206 (20/97; 26); 0.456 | 0.427 (82/192; 48); 0.571 | 0.535 (129/241; 65); 0.618 | 0.651 (286/439; 109); 0.701 |
| `replay` | 2 to 4 | 0.130 (6/46; 12); 0.247 | 0.373 (41/110; 28); 0.434 | 0.461 (65/141; 40); 0.504 | 0.504 (143/284; 75); 0.562 | 0.654 (263/402; 98); 0.709 |
| `replay` | 5 to 32 | 0.081 (5/62; 18); 0.334 | 0.252 (28/111; 31); 0.348 | 0.321 (71/221; 60); 0.449 | 0.549 (150/273; 73); 0.579 | 0.581 (279/480; 123); 0.640 |
| `replay` | 33 or more | 0.154 (8/52; 15); 0.281 | 0.340 (33/97; 26); 0.426 | 0.349 (67/192; 48); 0.477 | 0.519 (125/241; 65); 0.530 | 0.588 (258/439; 109); 0.626 |

## Stuck or still learning (D13)

Over the worlds scored for the arm: the mean and 90% interval of the score trend (D5) and of the trend of learning (D7). *Stuck*: neither interval lies above zero. *Declining*: an interval wholly below zero.

| Visits | Arm | Worlds | Score trend |  | Trend of learning |  | Statement |
|---|---|---|---|---|---|---|---|
| 256 | `sweep` | 1, 2 | -0.0891 [-0.3048, +0.1266] | not found | -0.0242 [-0.0843, +0.0359] | not found | stuck |
| 256 | `naive` | 1, 2 | -0.0217 [-0.0617, +0.0182] | not found | +0.0040 [-0.0605, +0.0686] | not found | stuck |
| 1024 | `sweep` | 1, 2 | +0.0806 [-0.0287, +0.1899] | not found | +0.0759 [+0.0362, +0.1156] | improving | still learning |
| 1024 | `naive` | 1, 2 | +0.0162 [-0.1456, +0.1779] | not found | +0.1086 [+0.0947, +0.1225] | improving | still learning |

## Between the rules (D11, D12)

Per world: the sweep's figure, back-propagation's, and the difference. Over the worlds: the mean difference with its 90% interval, and *higher*, *lower* or *not distinguished*.

**`sweep` against `naive`** (worlds 1, 2).

| Visits | Quantity | World 1 | World 2 | World 3 | World 4 | Mean difference | 90% interval |  |
|---|---|---|---|---|---|---|---|---|
| 256 | late level (D4) | 0.5892 − 0.5816 = +0.0076 | 0.5121 − 0.6963 = -0.1842 | — | — | -0.0883 | [-0.6939, +0.5173] | not distinguished |
| 256 | kept at return, absences of 5 to 32 visits (D8) | 0.0123 − 0.0164 = -0.0041 | 0.0315 − 0.0118 = +0.0197 | — | — | +0.0078 | [-0.0673, +0.0829] | not distinguished |
| 256 | kept at return, absences of 33 visits or more (D8) | 0.0127 − 0.0380 = -0.0253 | 0.0000 − 0.0000 = +0.0000 | — | — | -0.0127 | [-0.0926, +0.0673] | not distinguished |
| 256 | score trend (D5) | -0.0549 − -0.0281 = -0.0269 | -0.1232 − -0.0154 = -0.1078 | — | — | -0.0674 | [-0.3230, +0.1883] | not distinguished |
| 256 | trend of learning (D7) | -0.0147 − -0.0062 = -0.0085 | -0.0337 − 0.0143 = -0.0480 | — | — | -0.0283 | [-0.1529, +0.0964] | not distinguished |
| 1024 | late level (D4) | 0.6499 − 0.6690 = -0.0191 | 0.6278 − 0.7557 = -0.1279 | — | — | -0.0735 | [-0.4172, +0.2702] | not distinguished |
| 1024 | kept at return, absences of 5 to 32 visits (D8) | 0.0122 − 0.0157 = -0.0035 | 0.0205 − 0.0098 = +0.0107 | — | — | +0.0036 | [-0.0411, +0.0483] | not distinguished |
| 1024 | kept at return, absences of 33 visits or more (D8) | 0.0176 − 0.0147 = +0.0029 | 0.0203 − 0.0075 = +0.0128 | — | — | +0.0079 | [-0.0233, +0.0391] | not distinguished |
| 1024 | score trend (D5) | 0.0633 − -0.0094 = +0.0727 | 0.0979 − 0.0418 = +0.0561 | — | — | +0.0644 | [+0.0120, +0.1168] | higher |
| 1024 | trend of learning (D7) | 0.0822 − 0.1064 = -0.0242 | 0.0696 − 0.1108 = -0.0412 | — | — | -0.0327 | [-0.0863, +0.0209] | not distinguished |

Closed (D12): **open at 1,024 visits** (late-level difference: not distinguished at 256 visits, not distinguished at 1,024). The fixed sentence: over 1,024 visits, with no store, the sweep is **level** in late level, and its score trend over the second half is **steeper** than back-propagation's. At 256 visits the sweep is level in late level. **Only 2 of the four worlds are scored; the sentence is provisional.**

**`sweep+replay` against `replay`** (worlds 1).

| Visits | Quantity | World 1 | World 2 | World 3 | World 4 | Mean difference | 90% interval |  |
|---|---|---|---|---|---|---|---|---|
| 256 | late level (D4) | 0.6653 − 0.5773 = +0.0880 | — | — | — | +0.0880 | — | not read |
| 256 | kept at return, absences of 5 to 32 visits (D8) | 0.4467 − 0.4262 = +0.0205 | — | — | — | +0.0205 | — | not read |
| 256 | kept at return, absences of 33 visits or more (D8) | 0.4810 − 0.4557 = +0.0253 | — | — | — | +0.0253 | — | not read |
| 256 | score trend (D5) | 0.0188 − -0.0675 = +0.0863 | — | — | — | +0.0863 | — | not read |
| 256 | trend of learning (D7) | 0.0575 − -0.0230 = +0.0805 | — | — | — | +0.0805 | — | not read |
| 1024 | late level (D4) | 0.7844 − 0.7304 = +0.0540 | — | — | — | +0.0540 | — | not read |
| 1024 | kept at return, absences of 5 to 32 visits (D8) | 0.4978 − 0.4647 = +0.0331 | — | — | — | +0.0331 | — | not read |
| 1024 | kept at return, absences of 33 visits or more (D8) | 0.5093 − 0.4809 = +0.0284 | — | — | — | +0.0284 | — | not read |
| 1024 | score trend (D5) | 0.0459 − 0.0790 = -0.0331 | — | — | — | -0.0331 | — | not read |
| 1024 | trend of learning (D7) | 0.1126 − 0.1035 = +0.0091 | — | — | — | +0.0091 | — | not read |

Closed (D12): **open at 1,024 visits** (late-level difference: not read at 256 visits, not read at 1,024). The fixed sentence: over 1,024 visits, with the store, the sweep is **not read** in late level, and its score trend over the second half is **not read** than back-propagation's. At 256 visits the sweep is not read in late level. **Only 1 of the four worlds are scored; the sentence is provisional.**

**`sweep+replay` against `sweep`** (worlds 1).

| Visits | Quantity | World 1 | World 2 | World 3 | World 4 | Mean difference | 90% interval |  |
|---|---|---|---|---|---|---|---|---|
| 256 | late level (D4) | 0.6653 − 0.5892 = +0.0761 | — | — | — | +0.0761 | — | not read |
| 256 | kept at return, absences of 5 to 32 visits (D8) | 0.4467 − 0.0123 = +0.4344 | — | — | — | +0.4344 | — | not read |
| 256 | kept at return, absences of 33 visits or more (D8) | 0.4810 − 0.0127 = +0.4684 | — | — | — | +0.4684 | — | not read |
| 256 | score trend (D5) | 0.0188 − -0.0549 = +0.0737 | — | — | — | +0.0737 | — | not read |
| 256 | trend of learning (D7) | 0.0575 − -0.0147 = +0.0722 | — | — | — | +0.0722 | — | not read |
| 1024 | late level (D4) | 0.7844 − 0.6499 = +0.1345 | — | — | — | +0.1345 | — | not read |
| 1024 | kept at return, absences of 5 to 32 visits (D8) | 0.4978 − 0.0122 = +0.4856 | — | — | — | +0.4856 | — | not read |
| 1024 | kept at return, absences of 33 visits or more (D8) | 0.5093 − 0.0176 = +0.4917 | — | — | — | +0.4917 | — | not read |
| 1024 | score trend (D5) | 0.0459 − 0.0633 = -0.0174 | — | — | — | -0.0174 | — | not read |
| 1024 | trend of learning (D7) | 0.1126 − 0.0822 = +0.0304 | — | — | — | +0.0304 | — | not read |

## The curves of score by age

Mean score on reduction items over blocks of 64 visits, for the four arms in each world (the per-visit curves are in `curves_w<world>.csv`). The last column of each pair marks the blocks in which the sweep's mean is at or above back-propagation's.

**World 1.**

| Visits | `sweep` | `naive` | `sweep+replay` | `replay` | `sweep` ≥ `naive` | `sweep+replay` ≥ `replay` |
|---|---|---|---|---|---|---|
| 0 to 63 | 0.772 | 0.782 | 0.732 | 0.748 |  |  |
| 64 to 127 | 0.572 | 0.544 | 0.596 | 0.552 | yes | yes |
| 128 to 191 | 0.599 | 0.563 | 0.648 | 0.571 | yes | yes |
| 192 to 255 | 0.589 | 0.582 | 0.665 | 0.577 | yes | yes |
| 256 to 319 | 0.646 | 0.643 | 0.724 | 0.623 | yes | yes |
| 320 to 383 | 0.629 | 0.622 | 0.730 | 0.634 | yes | yes |
| 384 to 447 | 0.619 | 0.624 | 0.726 | 0.641 |  | yes |
| 448 to 511 | 0.604 | 0.668 | 0.758 | 0.685 |  | yes |
| 512 to 575 | 0.599 | 0.662 | 0.762 | 0.694 |  | yes |
| 576 to 639 | 0.659 | 0.667 | 0.761 | 0.684 |  | yes |
| 640 to 703 | 0.636 | 0.680 | 0.770 | 0.709 |  | yes |
| 704 to 767 | 0.679 | 0.697 | 0.775 | 0.709 |  | yes |
| 768 to 831 | 0.639 | 0.685 | 0.775 | 0.721 |  | yes |
| 832 to 895 | 0.628 | 0.695 | 0.796 | 0.742 |  | yes |
| 896 to 959 | 0.684 | 0.656 | 0.769 | 0.731 | yes | yes |
| 960 to 1023 | 0.649 | 0.640 | 0.797 | 0.728 | yes | yes |

**World 2.**

| Visits | `sweep` | `naive` | `sweep` ≥ `naive` |
|---|---|---|---|
| 0 to 63 | 0.620 | 0.717 |  |
| 64 to 127 | 0.571 | 0.668 |  |
| 128 to 191 | 0.527 | 0.686 |  |
| 192 to 255 | 0.512 | 0.696 |  |
| 256 to 319 | 0.511 | 0.712 |  |
| 320 to 383 | 0.494 | 0.663 |  |
| 384 to 447 | 0.633 | 0.747 |  |
| 448 to 511 | 0.623 | 0.761 |  |
| 512 to 575 | 0.614 | 0.724 |  |
| 576 to 639 | 0.600 | 0.747 |  |
| 640 to 703 | 0.535 | 0.752 |  |
| 704 to 767 | 0.619 | 0.767 |  |
| 768 to 831 | 0.566 | 0.750 |  |
| 832 to 895 | 0.633 | 0.751 |  |
| 896 to 959 | 0.638 | 0.745 |  |
| 960 to 1023 | 0.675 | 0.777 |  |

## The sweep's lives: cost and ledger

From the receipts under `exp45/e44/L2/sweep/lives/`. Share right at birth: the birth family's 2,000 reduction items answered before visit 0. Stored numbers beyond the weights (317,457) and the optimizer's moments (634,914): the store of `sweep+replay`, 55,209 items of 23 bytes (1,269,828 bytes).

| Arm | World | Birth seed | Share right at birth (R items) | CPU-hours | Resumes | Store capacity (items) | Store maximum | Replayed an update | Event file SHA-256 |
|---|---|---|---|---|---|---|---|---|---|
| `sweep` | 1 | 73129 | 0.919 | 3.78 | 2 | 0 | 0 | 0 | 781fa06ad2ad |
| `sweep` | 2 | 93683 | 0.925 | 3.77 | 2 | 0 | 0 | 0 | 0fa41db3094f |
| `sweep+replay` | 1 | 73129 | 0.919 | 3.80 | 2 | 55209 | 55209 | 24 | b7c2a8c3fba6 |
