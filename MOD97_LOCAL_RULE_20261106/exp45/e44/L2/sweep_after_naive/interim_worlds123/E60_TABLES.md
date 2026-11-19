# E60: the long life under the sweep from back-propagation's birth: tables

Written by `smallworld/l2_san_report.py final`. Lives of `sweep-after-naive` scored: worlds 1, 2, 3. The figures of `naive` and `sweep` are the score files of `738e2d61`. Each difference is computed from unrounded per-world values; each figure is rounded once. Intervals: two-sided 90%, Student's t over the worlds of the reading (`l2_score.interval90`), read with `l2_score.verdict`.

**Limit beside every reading over worlds 1 to 4:** Worlds 1 and 4 start from the same births (seed 73129), so that the four values are not those of four independent learners.

## Levels

Early level: the mean of D2 over visits 0 to 63 (from the score file at V = 1,024). Late level: D4 at V = 256 and 1,024.

| Arm | Age | World 1 | World 2 | World 3 | World 4 |
|---|---|---|---|---|---|
| `naive` | over the first 64 visits | 0.782 | 0.717 | 0.633 | 0.759 |
| `naive` | at 256 visits | 0.582 | 0.696 | 0.692 | 0.685 |
| `naive` | at 1,024 visits | 0.669 | 0.756 | 0.742 | 0.732 |
| `sweep-after-naive` | over the first 64 visits | 0.689 | 0.574 | 0.521 | — |
| `sweep-after-naive` | at 256 visits | 0.486 | 0.546 | 0.633 | — |
| `sweep-after-naive` | at 1,024 visits | 0.631 | 0.632 | 0.706 | — |
| `sweep` | over the first 64 visits | 0.772 | 0.620 | 0.639 | 0.713 |
| `sweep` | at 256 visits | 0.589 | 0.512 | 0.611 | 0.599 |
| `sweep` | at 1,024 visits | 0.650 | 0.628 | 0.682 | 0.672 |

## The nine cells (section 7 of the brief)

Each cell: the four per-world differences, then each reading's mean, 90% interval and word. Worlds 1 and 4 start from the same births (seed 73129), so that the four values are not those of four independent learners.

| Pair | Age | World 1 | World 2 | World 3 | World 4 | Over worlds 1 to 4 | Over worlds 1 to 3 |
|---|---|---|---|---|---|---|---|
| R: `sweep-after-naive` − `naive` (same birth; rules differ) | over the first 64 visits | -0.0936 | -0.1431 | -0.1115 | — | not read | -0.116 [-0.158, -0.074], lower |
| R: `sweep-after-naive` − `naive` (same birth; rules differ) | at 256 visits | -0.0951 | -0.1505 | -0.0591 | — | not read | -0.102 [-0.179, -0.024], lower |
| R: `sweep-after-naive` − `naive` (same birth; rules differ) | at 1,024 visits | -0.0381 | -0.1238 | -0.0355 | — | not read | -0.066 [-0.150, +0.019], not distinguished |
| B: `sweep` − `sweep-after-naive` (same rule; births differ) | over the first 64 visits | +0.0836 | +0.0453 | +0.1179 | — | not read | +0.082 [+0.021, +0.143], higher |
| B: `sweep` − `sweep-after-naive` (same rule; births differ) | at 256 visits | +0.1028 | -0.0337 | -0.0219 | — | not read | +0.016 [-0.112, +0.143], not distinguished |
| B: `sweep` − `sweep-after-naive` (same rule; births differ) | at 1,024 visits | +0.0191 | -0.0042 | -0.0235 | — | not read | -0.003 [-0.039, +0.033], not distinguished |
| RB: `sweep` − `naive` (both differ) | over the first 64 visits | -0.0100 | -0.0977 | +0.0064 | -0.0461 | -0.037 [-0.091, +0.017], not distinguished | -0.034 [-0.128, +0.061], not distinguished |
| RB: `sweep` − `naive` (both differ) | at 256 visits | +0.0076 | -0.1842 | -0.0810 | -0.0853 | -0.086 [-0.178, +0.007], not distinguished | -0.086 [-0.248, +0.076], not distinguished |
| RB: `sweep` − `naive` (both differ) | at 1,024 visits | -0.0191 | -0.1279 | -0.0590 | -0.0592 | -0.066 [-0.120, -0.013], lower | -0.069 [-0.162, +0.024], not distinguished |

RB = R + B in every world and age: the largest deviation is 0.0 (rounding of floating point).

## Pair R: the score trend, the change, and D13

| Quantity | World 1 | World 2 | World 3 | World 4 | Over worlds 1 to 4 | Over worlds 1 to 3 |
|---|---|---|---|---|---|---|
| difference of D5 (score trend) at 1,024 visits | +0.0286 | +0.0265 | -0.0092 | — | not read | +0.0153 [-0.0204, +0.0511], not distinguished |
| change: R at 1,024 visits less R over the first 64 | +0.0555 | +0.0193 | +0.0760 | — | not read | +0.0503 [+0.0019, +0.0986], higher |

D13 for `sweep-after-naive` at 1,024 visits: *still learning* if the interval of D5 or of D7 lies wholly above zero, *stuck* otherwise; *declining* for an interval wholly below zero.

| Reading | Score trend (D5) | Trend of learning (D7) | Statement | Declining |
|---|---|---|---|---|
| over worlds 1 to 4 | not read | not read | not read | none |
| over worlds 1 to 3 | +0.0370 [-0.0087, +0.0828], not distinguished | +0.1439 [+0.1145, +0.1733], higher | still learning | none |

## The three fixed sentences

**Over worlds 1 to 4** (Worlds 1 and 4 start from the same births (seed 73129), so that the four values are not those of four independent learners.):

1. From the same birth, the life under the sweep is not read against the life under back-propagation over the first 64 visits (not read), not read against it at 256 visits (not read) and not read against it at 1,024 visits (not read).
2. In 3 of the four worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits, in 0 it changed sign, and in 0 it is not nearer to zero; over these worlds the change (the difference at 1,024 visits less the difference over the first 64 visits) is not read (not read). By world: world 1: nearer to zero; world 2: nearer to zero; world 3: nearer to zero; world 4: not read. Overtaking is not said in this reading (it needs pair R *lower* over the first 64 visits and *higher* at 1,024 visits). The deficit of `sweep-after-naive` became smaller in world 1. The deficit of `sweep-after-naive` became smaller in world 2. The deficit of `sweep-after-naive` became smaller in world 3.
3. Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, -0.066 [-0.120, -0.013], lower), the part with the birth held equal is R (not read) and the part with the rule held equal is B (not read). Nothing is assigned: R is not read, not lower as RB (needed for the rule); B is not read, not lower as RB (needed for the birth).

**Over worlds 1 to 3**:

1. From the same birth, the life under the sweep is lower than the life under back-propagation over the first 64 visits (-0.116 [-0.158, -0.074], lower), lower than it at 256 visits (-0.102 [-0.179, -0.024], lower) and not distinguished from it at 1,024 visits (-0.066 [-0.150, +0.019], not distinguished).
2. In 3 of the three worlds the difference `sweep-after-naive` − `naive` is nearer to zero at 1,024 visits than over the first 64 visits, in 0 it changed sign, and in 0 it is not nearer to zero; over these worlds the change (the difference at 1,024 visits less the difference over the first 64 visits) is higher (+0.0503 [+0.0019, +0.0986], higher). By world: world 1: nearer to zero; world 2: nearer to zero; world 3: nearer to zero. Overtaking is not said in this reading (it needs pair R *lower* over the first 64 visits and *higher* at 1,024 visits). The deficit of `sweep-after-naive` became smaller in world 1. The deficit of `sweep-after-naive` became smaller in world 2. The deficit of `sweep-after-naive` became smaller in world 3.
3. Of the difference between the arms `sweep` and `naive` at 1,024 visits (RB, -0.069 [-0.162, +0.024], not distinguished), the part with the birth held equal is R (-0.066 [-0.150, +0.019], not distinguished) and the part with the rule held equal is B (-0.003 [-0.039, +0.033], not distinguished). Nothing is assigned: RB is not distinguished.

## Scores at birth

The receipts' final rows (E41): A and R in per cent.

| Seed | Rule of the birth | A | R |
|---|---|---|---|
| 73129 | B | 94.50 | 94.65 |
| 73129 | S | 95.83 | 91.95 |
| 93683 | B | 96.17 | 94.25 |
| 93683 | S | 93.50 | 92.55 |
| 116089 | B | 96.17 | 93.00 |
| 116089 | S | 92.83 | 90.90 |

## The new lives against `naive`

Per-visit stream digests equal to those of `naive`'s receipt of the same world; the whole-life stream digest; the receipt's `learner` object equal key for key; resumptions; CPU-hours.

| World | Visits with equal stream digests | Whole stream equal | Learner equal | Resumes | CPU-hours |
|---|---|---|---|---|---|
| 1 | 1024 of 1024 | True | True | 0 | 3.41 |
| 2 | 1024 of 1024 | True | True | 0 | 3.41 |
| 3 | 1024 of 1024 | True | True | 0 | 3.41 |

## The curves of score by age

Mean of D2 over blocks of 64 visits (per visit in `curves_w<world>.csv`).

**World 1.**

| Visits | `naive` | `sweep-after-naive` | `sweep` |
|---|---|---|---|
| 0 to 63 | 0.782 | 0.689 | 0.772 |
| 64 to 127 | 0.544 | 0.440 | 0.572 |
| 128 to 191 | 0.563 | 0.474 | 0.599 |
| 192 to 255 | 0.582 | 0.486 | 0.589 |
| 256 to 319 | 0.643 | 0.555 | 0.646 |
| 320 to 383 | 0.622 | 0.528 | 0.629 |
| 384 to 447 | 0.624 | 0.549 | 0.619 |
| 448 to 511 | 0.668 | 0.569 | 0.604 |
| 512 to 575 | 0.662 | 0.602 | 0.599 |
| 576 to 639 | 0.667 | 0.645 | 0.659 |
| 640 to 703 | 0.680 | 0.611 | 0.636 |
| 704 to 767 | 0.697 | 0.658 | 0.679 |
| 768 to 831 | 0.685 | 0.622 | 0.639 |
| 832 to 895 | 0.695 | 0.653 | 0.628 |
| 896 to 959 | 0.656 | 0.626 | 0.684 |
| 960 to 1023 | 0.640 | 0.622 | 0.649 |

**World 2.**

| Visits | `naive` | `sweep-after-naive` | `sweep` |
|---|---|---|---|
| 0 to 63 | 0.717 | 0.574 | 0.620 |
| 64 to 127 | 0.668 | 0.553 | 0.571 |
| 128 to 191 | 0.686 | 0.540 | 0.527 |
| 192 to 255 | 0.696 | 0.546 | 0.512 |
| 256 to 319 | 0.712 | 0.533 | 0.511 |
| 320 to 383 | 0.663 | 0.582 | 0.494 |
| 384 to 447 | 0.747 | 0.619 | 0.633 |
| 448 to 511 | 0.761 | 0.612 | 0.623 |
| 512 to 575 | 0.724 | 0.583 | 0.614 |
| 576 to 639 | 0.747 | 0.623 | 0.600 |
| 640 to 703 | 0.752 | 0.607 | 0.535 |
| 704 to 767 | 0.767 | 0.648 | 0.619 |
| 768 to 831 | 0.750 | 0.590 | 0.566 |
| 832 to 895 | 0.751 | 0.652 | 0.633 |
| 896 to 959 | 0.745 | 0.621 | 0.638 |
| 960 to 1023 | 0.777 | 0.666 | 0.675 |

**World 3.**

| Visits | `naive` | `sweep-after-naive` | `sweep` |
|---|---|---|---|
| 0 to 63 | 0.633 | 0.521 | 0.639 |
| 64 to 127 | 0.662 | 0.588 | 0.624 |
| 128 to 191 | 0.711 | 0.614 | 0.627 |
| 192 to 255 | 0.692 | 0.633 | 0.611 |
| 256 to 319 | 0.629 | 0.589 | 0.600 |
| 320 to 383 | 0.702 | 0.631 | 0.646 |
| 384 to 447 | 0.713 | 0.594 | 0.653 |
| 448 to 511 | 0.697 | 0.626 | 0.660 |
| 512 to 575 | 0.720 | 0.703 | 0.699 |
| 576 to 639 | 0.730 | 0.676 | 0.671 |
| 640 to 703 | 0.726 | 0.707 | 0.689 |
| 704 to 767 | 0.705 | 0.676 | 0.662 |
| 768 to 831 | 0.764 | 0.738 | 0.727 |
| 832 to 895 | 0.744 | 0.702 | 0.677 |
| 896 to 959 | 0.725 | 0.667 | 0.649 |
| 960 to 1023 | 0.733 | 0.717 | 0.677 |
