# E44, job L2, the sweep's long lives: early level and the change of the difference

Written by `smallworld/l2_sweep_early.py` from the score files under `exp45/e44/L2/sweep/scores/` and `exp45/e44/L2/sweep/main/` (the scorer `smallworld/l2_score.py`, unchanged). Early level: the mean score on reduction items (D2) over visits 0 to 63 of a life. Late level: D4. A difference is the sweep's figure minus back-propagation's; the change is the late difference minus the early difference. Intervals: two-sided 90%, Student's t over the four worlds (D11). The reviewer's addition of 17 November.

| Pair | Late window at | World | Early level (visits 0 to 63) | Late level (D4) | Change of the difference |
|---|---|---|---|---|---|
| `sweep` − `naive` | 256 | 1 | 0.772 − 0.782 = -0.0100 | 0.589 − 0.582 = +0.0076 | +0.0176 |
| `sweep` − `naive` | 256 | 2 | 0.620 − 0.717 = -0.0977 | 0.512 − 0.696 = -0.1842 | -0.0865 |
| `sweep` − `naive` | 256 | 3 | 0.639 − 0.633 = +0.0064 | 0.611 − 0.692 = -0.0810 | -0.0874 |
| `sweep` − `naive` | 256 | 4 | 0.713 − 0.759 = -0.0461 | 0.599 − 0.685 = -0.0853 | -0.0392 |
| `sweep` − `naive` | 256 | mean [90%] | -0.0369 [-0.0912, +0.0174] not distinguished | -0.0857 [-0.1779, +0.0065] not distinguished | -0.0488 [-0.1073, +0.0097] **not changed** |
| `sweep` − `naive` | 1024 | 1 | 0.772 − 0.782 = -0.0100 | 0.650 − 0.669 = -0.0191 | -0.0091 |
| `sweep` − `naive` | 1024 | 2 | 0.620 − 0.717 = -0.0977 | 0.628 − 0.756 = -0.1279 | -0.0302 |
| `sweep` − `naive` | 1024 | 3 | 0.639 − 0.633 = +0.0064 | 0.682 − 0.742 = -0.0590 | -0.0654 |
| `sweep` − `naive` | 1024 | 4 | 0.713 − 0.759 = -0.0461 | 0.672 − 0.732 = -0.0592 | -0.0131 |
| `sweep` − `naive` | 1024 | mean [90%] | -0.0369 [-0.0912, +0.0174] not distinguished | -0.0663 [-0.1195, -0.0131] lower | -0.0294 [-0.0597, +0.0008] **not changed** |
| `sweep+replay` − `replay` | 256 | 1 | 0.732 − 0.748 = -0.0162 | 0.665 − 0.577 = +0.0880 | +0.1041 |
| `sweep+replay` − `replay` | 256 | 2 | 0.597 − 0.698 = -0.1010 | 0.525 − 0.755 = -0.2302 | -0.1292 |
| `sweep+replay` − `replay` | 256 | 3 | 0.624 − 0.616 = +0.0085 | 0.588 − 0.707 = -0.1192 | -0.1277 |
| `sweep+replay` − `replay` | 256 | 4 | 0.673 − 0.734 = -0.0614 | 0.669 − 0.659 = +0.0100 | +0.0714 |
| `sweep+replay` − `replay` | 256 | mean [90%] | -0.0425 [-0.0997, +0.0146] not distinguished | -0.0628 [-0.2282, +0.1025] not distinguished | -0.0203 [-0.1680, +0.1274] **not changed** |
| `sweep+replay` − `replay` | 1024 | 1 | 0.732 − 0.748 = -0.0162 | 0.784 − 0.730 = +0.0540 | +0.0701 |
| `sweep+replay` − `replay` | 1024 | 2 | 0.597 − 0.698 = -0.1010 | 0.574 − 0.812 = -0.2378 | -0.1367 |
| `sweep+replay` − `replay` | 1024 | 3 | 0.624 − 0.616 = +0.0085 | 0.789 − 0.835 = -0.0462 | -0.0547 |
| `sweep+replay` − `replay` | 1024 | 4 | 0.673 − 0.734 = -0.0614 | 0.769 − 0.709 = +0.0605 | +0.1219 |
| `sweep+replay` − `replay` | 1024 | mean [90%] | -0.0425 [-0.0997, +0.0146] not distinguished | -0.0424 [-0.2061, +0.1213] not distinguished | +0.0002 [-0.1382, +0.1385] **not changed** |

**`sweep` against `naive`, the third clause:** from the first 64 visits to the late window of 1,024 visits the difference has **not changed** (change -0.0294 [-0.0597, +0.0008]); early difference -0.0369 (not distinguished), late difference -0.0663 (behind).

**`sweep+replay` against `replay`, the third clause:** from the first 64 visits to the late window of 1,024 visits the difference has **not changed** (change +0.0002 [-0.1382, +0.1385]); early difference -0.0425 (not distinguished), late difference -0.0424 (level).
