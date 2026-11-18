# E44, job L2, release 1: the trial's tables (world 1, 1,024 visits, `naive` and `replay`)

Written by `smallworld/l2_report.py trial` from the two lives' receipts, event files and list files; every figure is in `E44_L2_TRIAL.json` beside this file.

## The two lives

| Arm | Life | Commit | CPU-hours | of which the list | CPU-seconds a visit | Peak memory, MiB | Resumed | Store: items at the end / capacity | Replayed items an update | Decay |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | `lw_naive_w1_v1024` | c0657575 | 3.42 | 0.08 | 11.65 | 770 | 1 | 0 / 0 | 0 | 0.25 |
| replay | `lw_replay_w1_v1024_rho0.5` | c0657575 | 3.44 | 0.08 | 11.73 | 817 | 1 | 55,209 / 55,209 | 24 | 0.25 |

Host of both: {"hostname": "vm", "platform": "Linux-6.18.44-fc-v595-x86_64-with-glibc2.39", "machine": "x86_64", "cores": 4, "python": "3.13.18", "torch": "2.15.0+cpu", "numpy": "2.5.3", "torch_threads": 1}; threads {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "EXP45_WINDOW": "10", "EXP45_HIDDEN": "512", "OPENBLAS_NUM_THREADS": "1"}.

Commands:

- `python3 sw_run.py life --resume /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_naive_w1_v1024_state_v768.pt --out /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives`
- `python3 sw_run.py life --resume /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives/lw_replay_w1_v1024_rho0.5_state_v768.pt --out /home/user/digitalbaby/MOD97_LOCAL_RULE_20261106/exp45/e44/L2/lives`

## Checks on the files

| Arm | List items digest equals the audit's (52 families) | Row keys equal the reviewer's reference (part K; 5,031 keys, 1,238,400 items) | Items in the reviewer's order | Scorer equals part J at 256 visits | at 1,024 visits |
|---|---|---|---|---|---|
| naive | True | True | True | True | True |
| replay | True | True | True | True | True |

## The stop conditions

| Condition | Holds | Figures |
|---|---|---|
| a loss that is not finite | False | naive: mean loss of the first visit 0.2926, of the last 0.0312, largest 0.4741; replay: mean loss of the first visit 0.3097, of the last 0.1044, largest 0.3849 |
| the buffer below capacity after visit 64 | False | the store in the state after 64 visits: {'items': 55209, 'bytes': 938638, 'offered': 786432}; at the end 55,209 items; first replayed update 1 |
| score by age on reduction pairs below 0.05 over visits 512 to 1,023 in both arms | False | naive 0.673; replay 0.715 |
| a forecast for release 1 above 75 CPU-hours | False | 48.0 CPU-hours (before the trial: 48.2) |

**Any stop condition holds: False.**

## The first-visit level of `replay` beside `naive`'s (change 8)

| Arm | Families met | Median first-visit level (D6), all families | without the first family | over the families met in the first 256 visits |
|---|---|---|---|---|
| naive | 52 | 0.757 | 0.755 | 0.653 |
| replay | 52 | 0.595 | 0.592 | 0.393 |

`replay`'s median is not below 0.20.

## Score by age on reduction items (D2), in blocks of 64 visits

Every visit is in `E44_L2_TRIAL_score_by_age.csv`. "Returns" are visits to a family met before after an absence of two or more visits.

| Visits | First / continuation / return | naive: all | naive: returns | naive: blind | replay: all | replay: returns | replay: blind |
|---|---|---|---|---|---|---|---|
| 0 to 63 | 6 / 44 / 14 | 0.782 | 0.754 | 0.014 | 0.748 | 0.747 | 0.014 |
| 64 to 127 | 3 / 7 / 54 | 0.544 | 0.534 | 0.015 | 0.552 | 0.552 | 0.015 |
| 128 to 191 | 4 / 8 / 52 | 0.563 | 0.575 | 0.015 | 0.571 | 0.590 | 0.015 |
| 192 to 255 | 4 / 9 / 51 | 0.582 | 0.554 | 0.014 | 0.577 | 0.574 | 0.014 |
| 256 to 319 | 6 / 7 / 51 | 0.643 | 0.638 | 0.016 | 0.623 | 0.636 | 0.016 |
| 320 to 383 | 4 / 6 / 54 | 0.622 | 0.604 | 0.015 | 0.634 | 0.637 | 0.015 |
| 384 to 447 | 4 / 7 / 53 | 0.624 | 0.596 | 0.014 | 0.641 | 0.641 | 0.014 |
| 448 to 511 | 1 / 7 / 56 | 0.668 | 0.637 | 0.014 | 0.685 | 0.673 | 0.014 |
| 512 to 575 | 4 / 12 / 48 | 0.662 | 0.643 | 0.014 | 0.694 | 0.693 | 0.014 |
| 576 to 639 | 4 / 3 / 57 | 0.667 | 0.656 | 0.015 | 0.684 | 0.686 | 0.015 |
| 640 to 703 | 2 / 5 / 57 | 0.680 | 0.674 | 0.014 | 0.709 | 0.716 | 0.014 |
| 704 to 767 | 3 / 10 / 51 | 0.697 | 0.668 | 0.015 | 0.709 | 0.705 | 0.015 |
| 768 to 831 | 3 / 12 / 49 | 0.685 | 0.656 | 0.015 | 0.721 | 0.715 | 0.015 |
| 832 to 895 | 3 / 2 / 59 | 0.695 | 0.687 | 0.014 | 0.742 | 0.745 | 0.014 |
| 896 to 959 | 1 / 7 / 56 | 0.656 | 0.633 | 0.015 | 0.731 | 0.727 | 0.015 |
| 960 to 1023 | 0 / 5 / 59 | 0.640 | 0.631 | 0.015 | 0.728 | 0.725 | 0.015 |

## The quantities D4 to D10, D14 and D15

| Life | Visits read | Families met | First / continuation / return visits | Late level (D4) | Blind predictor's | Late level: returns | continuations | first visits | Score trend, returns (D5) | … all visits (not used) | Trend of learning (D7) | … ln n in the joint fit | Median first-visit level (D6) | … without the first family |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive, 256 | 256 | 17 | 17 / 68 / 171 | 0.582 | 0.014 | 0.554 | 0.738 | 0.587 | -0.0281 | +0.0673 | -0.0062 | -0.0151 | 0.653 | 0.551 |
| replay, 256 | 256 | 17 | 17 / 68 / 171 | 0.577 | 0.014 | 0.574 | 0.638 | 0.480 | -0.0675 | +0.0013 | -0.0230 | -0.0300 | 0.393 | 0.390 |
| naive, 1024 | 1024 | 52 | 52 / 151 / 821 | 0.669 | 0.015 | 0.652 | 0.814 | 0.676 | -0.0094 | -0.0075 | +0.1064 | +0.1102 | 0.757 | 0.755 |
| replay, 1024 | 1024 | 52 | 52 / 151 / 821 | 0.730 | 0.015 | 0.728 | 0.769 | 0.651 | +0.0790 | +0.0867 | +0.1035 | +0.1057 | 0.595 | 0.592 |

**Kept at return, from the stream (D8), and relearning at return (D10).** Share right on the reduction items of the visit's first update (answered before any update of the visit); the family's figure over the first 16 updates and over the last 64 updates of the visit; the last 64 updates of the family's previous visit. The class "5 or more" pools the two classes above it (recommendation 4).

| Life | Absence | Updates since | Decay constants | Visits | Items | Right | Kept (D8) | First 16 updates | Last 64 updates | Previous visit, last 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| naive, 256 | 1 visit | 256 to 256 | 0.06 to 0.06 | 68 | 274 | 215 | 0.785 | 0.782 | 0.799 | 0.781 |
| naive, 256 | 2 to 4 | 512 to 1,024 | 0.13 to 0.26 | 86 | 346 | 10 | 0.029 | 0.133 | 0.707 | 0.692 |
| naive, 256 | 5 to 32 | 1,280 to 8,192 | 0.32 to 2.05 | 66 | 244 | 4 | 0.016 | 0.090 | 0.650 | 0.661 |
| naive, 256 | 33 or more | 8,448+ | 2.11+ | 19 | 79 | 3 | 0.038 | 0.055 | 0.603 | 0.572 |
| naive, 256 | 5 or more | 1,280+ | 0.32+ | 85 | 323 | 7 | 0.022 | 0.082 | 0.640 | 0.641 |
| replay, 256 | 1 visit | 256 to 256 | 0.06 to 0.06 | 68 | 274 | 209 | 0.763 | 0.738 | 0.755 | 0.740 |
| replay, 256 | 2 to 4 | 512 to 1,024 | 0.13 to 0.26 | 86 | 346 | 167 | 0.483 | 0.523 | 0.638 | 0.625 |
| replay, 256 | 5 to 32 | 1,280 to 8,192 | 0.32 to 2.05 | 66 | 244 | 104 | 0.426 | 0.468 | 0.601 | 0.594 |
| replay, 256 | 33 or more | 8,448+ | 2.11+ | 19 | 79 | 36 | 0.456 | 0.409 | 0.587 | 0.539 |
| replay, 256 | 5 or more | 1,280+ | 0.32+ | 85 | 323 | 140 | 0.433 | 0.455 | 0.598 | 0.582 |
| naive, 1024 | 1 visit | 256 to 256 | 0.06 to 0.06 | 151 | 570 | 460 | 0.807 | 0.782 | 0.810 | 0.779 |
| naive, 1024 | 2 to 4 | 512 to 1,024 | 0.13 to 0.26 | 253 | 983 | 26 | 0.026 | 0.142 | 0.781 | 0.754 |
| naive, 1024 | 5 to 32 | 1,280 to 8,192 | 0.32 to 2.05 | 305 | 1147 | 18 | 0.016 | 0.094 | 0.750 | 0.767 |
| naive, 1024 | 33 or more | 8,448+ | 2.11+ | 263 | 1021 | 15 | 0.015 | 0.095 | 0.743 | 0.751 |
| naive, 1024 | 5 or more | 1,280+ | 0.32+ | 568 | 2168 | 33 | 0.015 | 0.095 | 0.747 | 0.759 |
| replay, 1024 | 1 visit | 256 to 256 | 0.06 to 0.06 | 151 | 570 | 421 | 0.739 | 0.730 | 0.758 | 0.735 |
| replay, 1024 | 2 to 4 | 512 to 1,024 | 0.13 to 0.26 | 253 | 983 | 518 | 0.527 | 0.581 | 0.701 | 0.688 |
| replay, 1024 | 5 to 32 | 1,280 to 8,192 | 0.32 to 2.05 | 305 | 1147 | 533 | 0.465 | 0.540 | 0.703 | 0.703 |
| replay, 1024 | 33 or more | 8,448+ | 2.11+ | 263 | 1021 | 491 | 0.481 | 0.536 | 0.709 | 0.700 |
| replay, 1024 | 5 or more | 1,280+ | 0.32+ | 568 | 2168 | 1024 | 0.472 | 0.538 | 0.706 | 0.701 |

**Kept on the list (D9).** A family's share right on its 200 list items at the end of visit v divided by its share at the end of its most recent visit ℓ < v, by v − ℓ; a family whose share at ℓ is 0 is left out and counted. Beside it the mean share right itself.

| Life | Visits since the family's last visit | Points | Left out | Mean kept | Median kept | Mean share right |
|---|---|---|---|---|---|---|
| naive, 256 | 1 visit | 25 | 0 | 0.037 | 0.011 | 0.028 |
| naive, 256 | 2 to 4 | 51 | 0 | 0.033 | 0.011 | 0.024 |
| naive, 256 | 5 to 32 | 172 | 0 | 0.024 | 0.000 | 0.016 |
| naive, 256 | 33 or more | 11 | 0 | 0.002 | 0.000 | 0.001 |
| naive, 256 | 5 or more | 183 | 0 | 0.023 | 0.000 | 0.015 |
| replay, 256 | 1 visit | 25 | 0 | 0.750 | 0.765 | 0.471 |
| replay, 256 | 2 to 4 | 51 | 0 | 0.678 | 0.725 | 0.438 |
| replay, 256 | 5 to 32 | 172 | 0 | 0.639 | 0.674 | 0.388 |
| replay, 256 | 33 or more | 11 | 0 | 0.619 | 0.618 | 0.339 |
| replay, 256 | 5 or more | 183 | 0 | 0.638 | 0.672 | 0.385 |
| naive, 1024 | 1 visit | 113 | 0 | 0.022 | 0.000 | 0.017 |
| naive, 1024 | 2 to 4 | 263 | 0 | 0.017 | 0.000 | 0.014 |
| naive, 1024 | 5 to 32 | 1481 | 0 | 0.014 | 0.000 | 0.011 |
| naive, 1024 | 33 or more | 2021 | 0 | 0.012 | 0.000 | 0.010 |
| naive, 1024 | 5 or more | 3502 | 0 | 0.013 | 0.000 | 0.010 |
| replay, 1024 | 1 visit | 113 | 0 | 0.759 | 0.783 | 0.530 |
| replay, 1024 | 2 to 4 | 263 | 0 | 0.688 | 0.735 | 0.489 |
| replay, 1024 | 5 to 32 | 1481 | 0 | 0.642 | 0.670 | 0.453 |
| replay, 1024 | 33 or more | 2021 | 0 | 0.642 | 0.662 | 0.461 |
| replay, 1024 | 5 or more | 3502 | 0 | 0.642 | 0.665 | 0.458 |

**Does a memory become more lasting as a family is met again? (D15.)** Kept at return (D8) and the family's figure over the first 16 updates of the visit (D10), by the number of earlier visits to the family within each class of absence. Each cell: kept (items) / first 16 updates (visits).

| Life | Absence | 1 earlier | 2 to 3 earlier | 4 to 7 earlier | 8 to 15 earlier | 16 or more earlier |
|---|---|---|---|---|---|---|
| naive, 256 | 1 visit | 0.545 (11) / 0.625 (4) | 0.750 (20) / 0.765 (6) | 0.947 (38) / 0.870 (8) | 0.819 (83) / 0.796 (20) | 0.738 (122) / 0.772 (30) |
| naive, 256 | 2 to 4 | 0.000 (23) / 0.097 (6) | 0.014 (72) / 0.107 (16) | 0.000 (53) / 0.097 (16) | 0.053 (76) / 0.144 (18) | 0.041 (122) / 0.166 (30) |
| naive, 256 | 5 to 32 | 0.000 (11) / 0.106 (4) | 0.034 (29) / 0.068 (8) | 0.013 (78) / 0.079 (17) | 0.013 (75) / 0.094 (22) | 0.020 (51) / 0.105 (15) |
| naive, 256 | 33 or more | 0.000 (3) / 0.007 (2) | 0.429 (7) / 0.107 (2) | 0.000 (22) / 0.009 (5) | 0.000 (11) / 0.021 (2) | 0.000 (36) / 0.091 (8) |
| replay, 256 | 1 visit | 0.455 (11) / 0.501 (4) | 0.650 (20) / 0.624 (6) | 0.789 (38) / 0.760 (8) | 0.759 (83) / 0.743 (20) | 0.803 (122) / 0.783 (30) |
| replay, 256 | 2 to 4 | 0.043 (23) / 0.152 (6) | 0.361 (72) / 0.379 (16) | 0.396 (53) / 0.427 (16) | 0.368 (76) / 0.498 (18) | 0.746 (122) / 0.741 (30) |
| replay, 256 | 5 to 32 | 0.091 (11) / 0.382 (4) | 0.172 (29) / 0.257 (8) | 0.321 (78) / 0.352 (17) | 0.507 (75) / 0.506 (22) | 0.686 (51) / 0.678 (15) |
| replay, 256 | 33 or more | 0.000 (3) / 0.097 (2) | 0.286 (7) / 0.248 (2) | 0.364 (22) / 0.274 (5) | 0.364 (11) / 0.304 (2) | 0.611 (36) / 0.638 (8) |
| naive, 1024 | 1 visit | 0.667 (18) / 0.638 (7) | 0.756 (41) / 0.732 (12) | 0.897 (117) / 0.822 (30) | 0.805 (154) / 0.782 (39) | 0.783 (240) / 0.789 (63) |
| naive, 1024 | 2 to 4 | 0.022 (46) / 0.132 (12) | 0.027 (110) / 0.128 (28) | 0.007 (141) / 0.108 (40) | 0.035 (284) / 0.146 (75) | 0.027 (402) / 0.157 (98) |
| naive, 1024 | 5 to 32 | 0.000 (62) / 0.068 (18) | 0.009 (111) / 0.050 (31) | 0.018 (221) / 0.098 (60) | 0.018 (273) / 0.106 (73) | 0.017 (480) / 0.101 (123) |
| naive, 1024 | 33 or more | 0.000 (52) / 0.055 (15) | 0.031 (97) / 0.109 (26) | 0.000 (192) / 0.074 (48) | 0.008 (241) / 0.085 (65) | 0.023 (439) / 0.112 (109) |
| replay, 1024 | 1 visit | 0.500 (18) / 0.551 (7) | 0.683 (41) / 0.677 (12) | 0.726 (117) / 0.732 (30) | 0.747 (154) / 0.717 (39) | 0.767 (240) / 0.768 (63) |
| replay, 1024 | 2 to 4 | 0.130 (46) / 0.247 (12) | 0.373 (110) / 0.434 (28) | 0.461 (141) / 0.504 (40) | 0.504 (284) / 0.562 (75) | 0.654 (402) / 0.709 (98) |
| replay, 1024 | 5 to 32 | 0.081 (62) / 0.334 (18) | 0.252 (111) / 0.348 (31) | 0.321 (221) / 0.449 (60) | 0.549 (273) / 0.579 (73) | 0.581 (480) / 0.640 (123) |
| replay, 1024 | 33 or more | 0.154 (52) / 0.281 (15) | 0.340 (97) / 0.426 (26) | 0.349 (192) / 0.477 (48) | 0.519 (241) / 0.530 (65) | 0.588 (439) / 0.626 (109) |

**What the wrong answers are (D14).** Shares of all answers, classes tried in the order of the columns. "Stream": the reduction items of the first update of a visit to a family met before (the family taught last is the family of the visit before; at an absence of 1 it is the item's own family, so that class is empty). "List": a family's 200 list items at the end of a visit to another family (the family taught last is that visit's). "Chance": the modulus of family 55, which no life is scheduled to meet, tried in the place of the family taught last.

| Life | Stream | Absence | Answers | right | no reduction | last family's modulus | an earlier family's | 89 | an operand | never-met modulus | another two-digit number | not two digits and end | Chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive, 256 | Stream | 1 visit | 274 | 0.785 | 0.000 | 0.000 | 0.026 | 0.000 | 0.004 | 0.004 | 0.182 | 0.000 | 0.004 |
| naive, 256 | Stream | 2 to 4 | 346 | 0.029 | 0.052 | 0.208 | 0.055 | 0.049 | 0.017 | 0.000 | 0.590 | 0.000 | 0.000 |
| naive, 256 | Stream | 5 to 32 | 244 | 0.016 | 0.074 | 0.311 | 0.102 | 0.004 | 0.033 | 0.000 | 0.459 | 0.000 | 0.000 |
| naive, 256 | Stream | 33 or more | 79 | 0.038 | 0.190 | 0.291 | 0.089 | 0.038 | 0.051 | 0.025 | 0.278 | 0.000 | 0.025 |
| replay, 256 | Stream | 1 visit | 274 | 0.763 | 0.000 | 0.000 | 0.026 | 0.000 | 0.004 | 0.007 | 0.201 | 0.000 | 0.007 |
| replay, 256 | Stream | 2 to 4 | 346 | 0.483 | 0.006 | 0.012 | 0.064 | 0.009 | 0.009 | 0.020 | 0.399 | 0.000 | 0.020 |
| replay, 256 | Stream | 5 to 32 | 244 | 0.426 | 0.020 | 0.033 | 0.078 | 0.000 | 0.016 | 0.008 | 0.418 | 0.000 | 0.008 |
| replay, 256 | Stream | 33 or more | 79 | 0.456 | 0.013 | 0.013 | 0.101 | 0.000 | 0.000 | 0.038 | 0.380 | 0.000 | 0.038 |
| naive, 1024 | Stream | 1 visit | 570 | 0.807 | 0.002 | 0.000 | 0.056 | 0.000 | 0.007 | 0.002 | 0.126 | 0.000 | 0.002 |
| naive, 1024 | Stream | 2 to 4 | 983 | 0.026 | 0.081 | 0.420 | 0.145 | 0.024 | 0.012 | 0.001 | 0.289 | 0.000 | 0.002 |
| naive, 1024 | Stream | 5 to 32 | 1147 | 0.016 | 0.114 | 0.483 | 0.175 | 0.005 | 0.012 | 0.005 | 0.189 | 0.000 | 0.007 |
| naive, 1024 | Stream | 33 or more | 1021 | 0.015 | 0.135 | 0.479 | 0.194 | 0.016 | 0.015 | 0.005 | 0.142 | 0.000 | 0.008 |
| replay, 1024 | Stream | 1 visit | 570 | 0.739 | 0.002 | 0.000 | 0.089 | 0.005 | 0.005 | 0.007 | 0.153 | 0.000 | 0.007 |
| replay, 1024 | Stream | 2 to 4 | 983 | 0.527 | 0.015 | 0.012 | 0.164 | 0.011 | 0.005 | 0.012 | 0.252 | 0.001 | 0.012 |
| replay, 1024 | Stream | 5 to 32 | 1147 | 0.465 | 0.027 | 0.017 | 0.255 | 0.015 | 0.005 | 0.013 | 0.203 | 0.000 | 0.013 |
| replay, 1024 | Stream | 33 or more | 1021 | 0.481 | 0.023 | 0.013 | 0.285 | 0.011 | 0.006 | 0.010 | 0.172 | 0.000 | 0.010 |

| Life | List | Absence | Answers | right | no reduction | last family's modulus | an earlier family's | 89 | an operand | never-met modulus | another two-digit number | not two digits and end | Chance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive, 256 | List | 1 visit | 5000 | 0.028 | 0.060 | 0.297 | 0.063 | 0.021 | 0.016 | 0.004 | 0.510 | 0.000 | 0.004 |
| naive, 256 | List | 2 to 4 | 10200 | 0.024 | 0.082 | 0.292 | 0.078 | 0.018 | 0.017 | 0.006 | 0.484 | 0.000 | 0.006 |
| naive, 256 | List | 5 to 32 | 34400 | 0.016 | 0.096 | 0.368 | 0.066 | 0.007 | 0.015 | 0.006 | 0.427 | 0.000 | 0.006 |
| naive, 256 | List | 33 or more | 2200 | 0.001 | 0.241 | 0.521 | 0.044 | 0.005 | 0.001 | 0.006 | 0.181 | 0.000 | 0.006 |
| replay, 256 | List | 1 visit | 5000 | 0.471 | 0.010 | 0.008 | 0.087 | 0.001 | 0.012 | 0.006 | 0.406 | 0.000 | 0.006 |
| replay, 256 | List | 2 to 4 | 10200 | 0.438 | 0.015 | 0.010 | 0.086 | 0.002 | 0.017 | 0.014 | 0.418 | 0.000 | 0.015 |
| replay, 256 | List | 5 to 32 | 34400 | 0.388 | 0.012 | 0.018 | 0.105 | 0.002 | 0.018 | 0.015 | 0.441 | 0.000 | 0.015 |
| replay, 256 | List | 33 or more | 2200 | 0.339 | 0.005 | 0.004 | 0.133 | 0.000 | 0.015 | 0.015 | 0.490 | 0.000 | 0.015 |
| naive, 1024 | List | 1 visit | 22600 | 0.017 | 0.100 | 0.479 | 0.160 | 0.011 | 0.009 | 0.004 | 0.219 | 0.000 | 0.007 |
| naive, 1024 | List | 2 to 4 | 52600 | 0.014 | 0.120 | 0.493 | 0.160 | 0.010 | 0.008 | 0.004 | 0.192 | 0.000 | 0.006 |
| naive, 1024 | List | 5 to 32 | 296200 | 0.011 | 0.128 | 0.501 | 0.177 | 0.008 | 0.006 | 0.003 | 0.166 | 0.000 | 0.006 |
| naive, 1024 | List | 33 or more | 404200 | 0.010 | 0.134 | 0.533 | 0.211 | 0.008 | 0.004 | 0.002 | 0.099 | 0.000 | 0.006 |
| replay, 1024 | List | 1 visit | 22600 | 0.530 | 0.015 | 0.011 | 0.229 | 0.007 | 0.006 | 0.006 | 0.196 | 0.000 | 0.007 |
| replay, 1024 | List | 2 to 4 | 52600 | 0.489 | 0.023 | 0.011 | 0.258 | 0.008 | 0.007 | 0.009 | 0.193 | 0.000 | 0.010 |
| replay, 1024 | List | 5 to 32 | 296200 | 0.453 | 0.021 | 0.013 | 0.298 | 0.008 | 0.006 | 0.009 | 0.192 | 0.000 | 0.010 |
| replay, 1024 | List | 33 or more | 404200 | 0.461 | 0.021 | 0.015 | 0.355 | 0.010 | 0.004 | 0.010 | 0.123 | 0.000 | 0.013 |

