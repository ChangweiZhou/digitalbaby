# E44 job F1: the calibration's tables

Machine record `E44_F1.json`; receipts, event files and lists beside it. Levels are shares right over asked items; the **family's figure** is the mean of add and sub on asked reduction pairs; **first-visit level** reads the last quarter of the family's first visit (16, 32, 64 updates at visits of 64, 128, 256). Families are numbered as in the file (0 to 7); a calibration life takes them in turn, so family f is the (f + 1)-th taken.

## The gate (under `naive`): the median over families 1 to 7 of the first-visit level on reduction pairs

| Visit length | World 1 | World 2 | World 3 | World 4 | World 5 | World 6 | World 7 | World 8 | Worlds in 0.20–0.90 | above | below | CPU-s a visit | Forecast, 1,024 visits | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.059 | 0.112 | 0.068 | 0.065 | 0.068 | 0.053 | 0.084 | 0.133 | 0 | 0 | 8 | 3.4 | 0.97 h | no |
| 128 | 0.169 | 0.434 | 0.169 | 0.322 | 0.213 | 0.096 | 0.195 | 0.301 | 4 | 0 | 4 | 6.8 | 1.94 h | no |
| 256 | 0.368 | 0.566 | 0.301 | 0.457 | 0.491 | 0.368 | 0.436 | 0.636 | 8 | 0 | 0 | 13.5 | 3.85 h | yes |

Adopted visit length: **256**.

## The same median in every set

| Set | Arm | Visit length | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8 | Mean | Trend over n = 2..8: mean slope, worlds positive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.059 | 0.112 | 0.068 | 0.065 | 0.068 | 0.053 | 0.084 | 0.133 | 0.080 | -0.065, 2 of 8 |
| gate | sweep | 64 | 0.186 | 0.355 | 0.122 | 0.182 | 0.240 | 0.112 | 0.152 | 0.233 | 0.198 | -0.038, 2 of 8 |
| gate | naive | 128 | 0.169 | 0.434 | 0.169 | 0.322 | 0.213 | 0.096 | 0.195 | 0.301 | 0.237 | -0.091, 1 of 8 |
| gate | sweep | 128 | 0.452 | 0.458 | 0.396 | 0.416 | 0.420 | 0.345 | 0.422 | 0.441 | 0.419 | -0.063, 3 of 8 |
| gate | naive | 256 | 0.368 | 0.566 | 0.301 | 0.457 | 0.491 | 0.368 | 0.436 | 0.636 | 0.453 | -0.101, 2 of 8 |
| gate | sweep | 256 | 0.669 | 0.656 | 0.538 | 0.561 | 0.737 | 0.537 | 0.622 | 0.595 | 0.614 | -0.039, 2 of 8 |
| san | sweep-after-naive | 64 | 0.026 | 0.031 | 0.036 | 0.036 | 0.042 | 0.059 | 0.029 | 0.031 | 0.036 | -0.047, 2 of 8 |
| san | sweep-after-naive | 128 | 0.040 | 0.047 | 0.017 | 0.067 | 0.085 | 0.054 | 0.051 | 0.045 | 0.051 | -0.045, 3 of 8 |
| san | sweep-after-naive | 256 | 0.157 | 0.275 | 0.117 | 0.187 | 0.182 | 0.200 | 0.212 | 0.240 | 0.196 | -0.029, 4 of 8 |
| wd0 | naive | 64 | 0.043 | 0.098 | 0.071 | 0.076 | 0.058 | 0.053 | 0.057 | 0.124 | 0.073 | -0.062, 3 of 8 |
| wd0 | naive | 128 | 0.169 | 0.360 | 0.135 | 0.289 | 0.232 | 0.095 | 0.178 | 0.289 | 0.218 | -0.098, 1 of 8 |
| wd0 | naive | 256 | 0.440 | 0.612 | 0.289 | 0.457 | 0.534 | 0.319 | 0.433 | 0.680 | 0.471 | -0.132, 2 of 8 |

## First-visit levels by window, pooled over worlds (mean over the 8 worlds of families 1 to 7)

| Set | Arm | Visit length | Window | Reduction (figure) | Without reduction | Max | Min |
|---|---|---|---|---|---|---|---|
| gate | naive | 64 | first16 | 0.023 | 0.144 | 0.562 | 0.609 |
| gate | naive | 64 | whole | 0.082 | 0.193 | 0.770 | 0.797 |
| gate | naive | 64 | last quarter | 0.136 | 0.264 | 0.894 | 0.913 |
| gate | sweep | 64 | first16 | 0.042 | 0.286 | 0.730 | 0.718 |
| gate | sweep | 64 | whole | 0.137 | 0.406 | 0.850 | 0.847 |
| gate | sweep | 64 | last quarter | 0.230 | 0.523 | 0.918 | 0.919 |
| gate | naive | 128 | first16 | 0.037 | 0.184 | 0.596 | 0.619 |
| gate | naive | 128 | whole | 0.184 | 0.343 | 0.857 | 0.877 |
| gate | naive | 128 | last quarter | 0.301 | 0.474 | 0.939 | 0.945 |
| gate | sweep | 128 | first16 | 0.061 | 0.335 | 0.735 | 0.728 |
| gate | sweep | 128 | whole | 0.275 | 0.554 | 0.900 | 0.901 |
| gate | sweep | 128 | last quarter | 0.415 | 0.685 | 0.952 | 0.949 |
| gate | naive | 256 | first16 | 0.037 | 0.198 | 0.580 | 0.598 |
| gate | naive | 256 | whole | 0.327 | 0.483 | 0.903 | 0.913 |
| gate | naive | 256 | last quarter | 0.471 | 0.612 | 0.952 | 0.956 |
| gate | sweep | 256 | first16 | 0.063 | 0.342 | 0.721 | 0.736 |
| gate | sweep | 256 | whole | 0.434 | 0.654 | 0.926 | 0.929 |
| gate | sweep | 256 | last quarter | 0.588 | 0.749 | 0.956 | 0.960 |
| san | sweep-after-naive | 64 | first16 | 0.015 | 0.077 | 0.440 | 0.468 |
| san | sweep-after-naive | 64 | whole | 0.028 | 0.089 | 0.599 | 0.621 |
| san | sweep-after-naive | 64 | last quarter | 0.045 | 0.112 | 0.725 | 0.729 |
| san | sweep-after-naive | 128 | first16 | 0.020 | 0.107 | 0.504 | 0.530 |
| san | sweep-after-naive | 128 | whole | 0.061 | 0.144 | 0.746 | 0.766 |
| san | sweep-after-naive | 128 | last quarter | 0.096 | 0.193 | 0.856 | 0.860 |
| san | sweep-after-naive | 256 | first16 | 0.020 | 0.153 | 0.489 | 0.541 |
| san | sweep-after-naive | 256 | whole | 0.151 | 0.264 | 0.824 | 0.843 |
| san | sweep-after-naive | 256 | last quarter | 0.239 | 0.349 | 0.903 | 0.910 |
| wd0 | naive | 64 | first16 | 0.022 | 0.145 | 0.567 | 0.605 |
| wd0 | naive | 64 | whole | 0.084 | 0.200 | 0.776 | 0.799 |
| wd0 | naive | 64 | last quarter | 0.139 | 0.274 | 0.901 | 0.917 |
| wd0 | naive | 128 | first16 | 0.032 | 0.184 | 0.594 | 0.614 |
| wd0 | naive | 128 | whole | 0.187 | 0.367 | 0.864 | 0.880 |
| wd0 | naive | 128 | last quarter | 0.303 | 0.513 | 0.947 | 0.949 |
| wd0 | naive | 256 | first16 | 0.035 | 0.209 | 0.589 | 0.595 |
| wd0 | naive | 256 | whole | 0.332 | 0.520 | 0.910 | 0.917 |
| wd0 | naive | 256 | last quarter | 0.479 | 0.664 | 0.958 | 0.961 |

## The family's figure (first-visit level, last quarter) against its taught pairs and its place in turn, pooled over worlds

Families 1 to 7 of every world (56 a row; the count of each cell in brackets); the joint fit is the least-squares fit of the figure on ln n and ln(taught pairs) together, over n = 2 to 8.

| Set | Arm | Visit length | 264 or fewer pairs | 265 to 688 pairs | 689 or more pairs | Joint fit: per ln n | per ln(taught pairs) |
|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.066 (14) | 0.141 (29) | 0.201 (13) | -0.068 | 0.099 |
| gate | sweep | 64 | 0.160 (14) | 0.264 (29) | 0.231 (13) | -0.040 | 0.062 |
| gate | naive | 128 | 0.200 (14) | 0.328 (29) | 0.348 (13) | -0.095 | 0.119 |
| gate | sweep | 128 | 0.347 (14) | 0.446 (29) | 0.417 (13) | -0.066 | 0.078 |
| gate | naive | 256 | 0.378 (14) | 0.496 (29) | 0.515 (13) | -0.105 | 0.110 |
| gate | sweep | 256 | 0.507 (14) | 0.619 (29) | 0.606 (13) | -0.042 | 0.085 |
| san | sweep-after-naive | 64 | 0.037 (14) | 0.038 (29) | 0.071 (13) | -0.047 | 0.023 |
| san | sweep-after-naive | 128 | 0.057 (14) | 0.098 (29) | 0.135 (13) | -0.047 | 0.063 |
| san | sweep-after-naive | 256 | 0.156 (14) | 0.258 (29) | 0.283 (13) | -0.032 | 0.095 |
| wd0 | naive | 64 | 0.065 (14) | 0.145 (29) | 0.206 (13) | -0.065 | 0.108 |
| wd0 | naive | 128 | 0.213 (14) | 0.330 (29) | 0.341 (13) | -0.102 | 0.109 |
| wd0 | naive | 256 | 0.383 (14) | 0.505 (29) | 0.523 (13) | -0.136 | 0.114 |

| Set | Arm | Visit length | n = 1 (family 0) | n = 2 (family 1) | n = 3 (family 2) | n = 4 (family 3) | n = 5 (family 4) | n = 6 (family 5) | n = 7 (family 6) | n = 8 (family 7) |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.243 | 0.197 | 0.194 | 0.092 | 0.100 | 0.112 | 0.193 | 0.065 |
| gate | sweep | 64 | 0.447 | 0.216 | 0.367 | 0.159 | 0.196 | 0.220 | 0.300 | 0.154 |
| gate | naive | 128 | 0.454 | 0.377 | 0.430 | 0.188 | 0.229 | 0.268 | 0.454 | 0.158 |
| gate | sweep | 128 | 0.561 | 0.426 | 0.581 | 0.295 | 0.395 | 0.398 | 0.493 | 0.315 |
| gate | naive | 256 | 0.616 | 0.536 | 0.641 | 0.376 | 0.354 | 0.442 | 0.616 | 0.332 |
| gate | sweep | 256 | 0.650 | 0.562 | 0.730 | 0.540 | 0.542 | 0.588 | 0.645 | 0.507 |
| san | sweep-after-naive | 64 | 0.095 | 0.097 | 0.063 | 0.039 | 0.020 | 0.033 | 0.040 | 0.026 |
| san | sweep-after-naive | 128 | 0.163 | 0.150 | 0.139 | 0.043 | 0.060 | 0.073 | 0.164 | 0.046 |
| san | sweep-after-naive | 256 | 0.328 | 0.293 | 0.284 | 0.154 | 0.158 | 0.249 | 0.358 | 0.175 |
| wd0 | naive | 64 | 0.249 | 0.196 | 0.211 | 0.079 | 0.092 | 0.118 | 0.218 | 0.061 |
| wd0 | naive | 128 | 0.464 | 0.387 | 0.419 | 0.209 | 0.234 | 0.279 | 0.440 | 0.156 |
| wd0 | naive | 256 | 0.635 | 0.551 | 0.681 | 0.392 | 0.368 | 0.435 | 0.613 | 0.311 |

## Score by age (share right over all asked items of the visit, mean over worlds) and the blind predictor's

| Set | Arm | Visit length | Visit 0 | Visit 1 | Visit 2 | Visit 3 | Visit 4 | Visit 5 | Visit 6 | Visit 7 | Blind, mean over visits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.434 | 0.413 | 0.449 | 0.428 | 0.445 | 0.463 | 0.522 | 0.503 | 0.018 |
| gate | sweep | 64 | 0.614 | 0.502 | 0.584 | 0.525 | 0.551 | 0.558 | 0.618 | 0.587 | 0.018 |
| gate | naive | 128 | 0.555 | 0.539 | 0.571 | 0.531 | 0.546 | 0.560 | 0.642 | 0.568 | 0.018 |
| gate | sweep | 128 | 0.693 | 0.613 | 0.699 | 0.625 | 0.643 | 0.662 | 0.706 | 0.657 | 0.018 |
| gate | naive | 256 | 0.671 | 0.653 | 0.680 | 0.623 | 0.620 | 0.644 | 0.728 | 0.643 | 0.019 |
| gate | sweep | 256 | 0.755 | 0.694 | 0.781 | 0.717 | 0.723 | 0.742 | 0.764 | 0.729 | 0.019 |
| san | sweep-after-naive | 64 | 0.315 | 0.271 | 0.326 | 0.306 | 0.317 | 0.342 | 0.372 | 0.405 | 0.018 |
| san | sweep-after-naive | 128 | 0.398 | 0.398 | 0.413 | 0.388 | 0.409 | 0.435 | 0.487 | 0.474 | 0.018 |
| san | sweep-after-naive | 256 | 0.502 | 0.499 | 0.497 | 0.476 | 0.494 | 0.533 | 0.592 | 0.551 | 0.019 |
| wd0 | naive | 64 | 0.435 | 0.416 | 0.454 | 0.427 | 0.451 | 0.468 | 0.528 | 0.508 | 0.018 |
| wd0 | naive | 128 | 0.558 | 0.547 | 0.576 | 0.538 | 0.552 | 0.577 | 0.652 | 0.582 | 0.018 |
| wd0 | naive | 256 | 0.677 | 0.663 | 0.693 | 0.638 | 0.629 | 0.663 | 0.740 | 0.661 | 0.019 |

## What is kept (the list of item 5): share at a later visit divided by the share at the end of the family's own visit, with decay alone beside it

Median over the families of the eight worlds (families 0 to 6 at one visit since, fewer as the distance grows); the families with a share of 0 at the end of their own visit are left out (count in brackets). Decay alone: the same list answered by the weights at the end of the family's own visit, every parameter but the byte table multiplied by (1 − lr × decay)^u.

| Set | Arm | Visit length | Visits since: 1 | 2 | 3 | 4 | 7 | Decay alone: 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.168 (56) | 0.064 (48) | 0.180 (40) | 0.127 (32) | 0.167 (8) | 1.000 | 0.983 | 0.943 | 0.835 | 0.650 |
| gate | sweep | 64 | 0.085 (56) | 0.031 (48) | 0.000 (40) | 0.071 (32) | 0.015 (8) | 0.959 | 0.901 | 0.844 | 0.757 | 0.590 |
| gate | naive | 128 | 0.054 (56) | 0.022 (48) | 0.030 (40) | 0.078 (32) | 0.056 (8) | 0.935 | 0.849 | 0.696 | 0.576 | 0.212 |
| gate | sweep | 128 | 0.022 (56) | 0.000 (48) | 0.009 (40) | 0.016 (32) | 0.022 (8) | 0.922 | 0.778 | 0.643 | 0.472 | 0.122 |
| gate | naive | 256 | 0.021 (56) | 0.015 (48) | 0.010 (40) | 0.019 (32) | 0.016 (8) | 0.891 | 0.651 | 0.418 | 0.227 | 0.014 |
| gate | sweep | 256 | 0.003 (56) | 0.000 (48) | 0.000 (40) | 0.012 (32) | 0.007 (8) | 0.821 | 0.479 | 0.242 | 0.086 | 0.010 |
| san | sweep-after-naive | 64 | 0.500 (56) | 0.375 (48) | 0.292 (40) | 0.453 (32) | 0.148 (8) | 1.000 | 1.000 | 1.000 | 0.938 | 0.683 |
| san | sweep-after-naive | 128 | 0.197 (56) | 0.103 (48) | 0.138 (40) | 0.211 (32) | 0.121 (8) | 0.938 | 0.902 | 0.750 | 0.598 | 0.254 |
| san | sweep-after-naive | 256 | 0.071 (56) | 0.032 (48) | 0.030 (40) | 0.062 (32) | 0.071 (8) | 0.848 | 0.617 | 0.414 | 0.221 | 0.031 |
| wd0 | naive | 64 | 0.162 (55, 1 out) | 0.077 (47, 1 out) | 0.118 (39, 1 out) | 0.190 (31, 1 out) | 0.193 (8) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wd0 | naive | 128 | 0.075 (56) | 0.026 (48) | 0.034 (40) | 0.081 (32) | 0.061 (8) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wd0 | naive | 256 | 0.031 (56) | 0.017 (48) | 0.008 (40) | 0.038 (32) | 0.026 (8) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**The birth family** (mod 89, E41's R items, 2,000): share right before visit 0 and at the end of visits 0, 1, 3 and 7 (mean over worlds), with decay alone of the born weights beside it.

| Set | Arm | Visit length | Before | After visit 0 | 1 | 3 | 7 | Decay alone after visit 0 | 1 | 3 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.947 | 0.195 | 0.116 | 0.062 | 0.076 | 0.935 | 0.920 | 0.875 | 0.702 |
| gate | sweep | 64 | 0.919 | 0.096 | 0.124 | 0.015 | 0.011 | 0.913 | 0.898 | 0.843 | 0.578 |
| gate | naive | 128 | 0.947 | 0.134 | 0.060 | 0.030 | 0.040 | 0.920 | 0.875 | 0.702 | 0.212 |
| gate | sweep | 128 | 0.919 | 0.078 | 0.087 | 0.005 | 0.005 | 0.898 | 0.843 | 0.578 | 0.099 |
| gate | naive | 256 | 0.947 | 0.084 | 0.050 | 0.018 | 0.029 | 0.875 | 0.702 | 0.212 | 0.000 |
| gate | sweep | 256 | 0.919 | 0.059 | 0.074 | 0.004 | 0.003 | 0.843 | 0.578 | 0.099 | 0.001 |
| san | sweep-after-naive | 64 | 0.947 | 0.209 | 0.147 | 0.080 | 0.069 | 0.935 | 0.920 | 0.875 | 0.702 |
| san | sweep-after-naive | 128 | 0.947 | 0.179 | 0.094 | 0.050 | 0.057 | 0.920 | 0.875 | 0.702 | 0.212 |
| san | sweep-after-naive | 256 | 0.947 | 0.111 | 0.066 | 0.025 | 0.037 | 0.875 | 0.702 | 0.212 | 0.000 |
| wd0 | naive | 64 | 0.947 | 0.196 | 0.118 | 0.068 | 0.085 | 0.947 | 0.947 | 0.947 | 0.947 |
| wd0 | naive | 128 | 0.947 | 0.138 | 0.068 | 0.035 | 0.050 | 0.947 | 0.947 | 0.947 | 0.947 |
| wd0 | naive | 256 | 0.947 | 0.099 | 0.045 | 0.021 | 0.036 | 0.947 | 0.947 | 0.947 | 0.947 |

## The second lap (at the adopted length): each family's return

Pooled share right on the family's asked reduction items (add and sub) in the return's first update and its first 16 updates; the family's figure over the last quarter of the return beside that of its first visit; its list share at the end of its own visit, at the end of the first lap (visit 7) and at the end of the return. Means over the eight worlds.

| Arm | Family | First update | First 16 updates | Last quarter (return) | Last quarter (first visit) | List: own visit | end of lap 1 | end of return |
|---|---|---|---|---|---|---|---|---|
| naive | 0 | 0.014 | 0.102 | 0.672 | 0.616 | 0.587 | 0.021 | 0.636 |
| naive | 1 | 0.000 | 0.161 | 0.633 | 0.536 | 0.534 | 0.057 | 0.621 |
| naive | 2 | 0.075 | 0.119 | 0.736 | 0.641 | 0.611 | 0.032 | 0.705 |
| naive | 3 | 0.000 | 0.066 | 0.500 | 0.376 | 0.360 | 0.034 | 0.475 |
| naive | 4 | 0.000 | 0.055 | 0.437 | 0.354 | 0.370 | 0.005 | 0.431 |
| naive | 5 | 0.000 | 0.056 | 0.562 | 0.442 | 0.437 | 0.021 | 0.547 |
| naive | 6 | 0.000 | 0.071 | 0.692 | 0.616 | 0.573 | 0.034 | 0.661 |
| naive | 7 | 0.000 | 0.049 | 0.434 | 0.332 | 0.298 | 0.298 | 0.399 |
| sweep | 0 | 0.000 | 0.043 | 0.710 | 0.650 | 0.597 | 0.006 | 0.657 |
| sweep | 1 | 0.000 | 0.109 | 0.671 | 0.562 | 0.493 | 0.019 | 0.623 |
| sweep | 2 | 0.050 | 0.127 | 0.763 | 0.730 | 0.689 | 0.004 | 0.722 |
| sweep | 3 | 0.000 | 0.074 | 0.638 | 0.540 | 0.479 | 0.016 | 0.580 |
| sweep | 4 | 0.000 | 0.073 | 0.589 | 0.542 | 0.501 | 0.015 | 0.546 |
| sweep | 5 | 0.042 | 0.070 | 0.653 | 0.588 | 0.545 | 0.022 | 0.627 |
| sweep | 6 | 0.021 | 0.098 | 0.724 | 0.645 | 0.570 | 0.039 | 0.648 |
| sweep | 7 | 0.000 | 0.059 | 0.581 | 0.507 | 0.466 | 0.466 | 0.539 |

## CPU-seconds per visit (updates with their asking; the list apart)

| Set | Arm | Visit length | Per visit, mean | List per visit, mean | Forecast, 1,024 visits | Peak memory |
|---|---|---|---|---|---|---|
| gate | naive | 64 | 3.40 | 0.99 | 0.97 h | 509 MiB |
| gate | sweep | 64 | 3.54 | 0.97 | 1.01 h | 509 MiB |
| gate | naive | 128 | 6.81 | 1.01 | 1.94 h | 509 MiB |
| gate | sweep | 128 | 7.26 | 1.00 | 2.07 h | 508 MiB |
| gate | naive | 256 | 13.54 | 0.99 | 3.85 h | 510 MiB |
| gate | sweep | 256 | 14.89 | 1.02 | 4.23 h | 508 MiB |
| san | sweep-after-naive | 64 | 3.51 | 0.94 | 1.00 h | 508 MiB |
| san | sweep-after-naive | 128 | 7.04 | 0.95 | 2.00 h | 509 MiB |
| san | sweep-after-naive | 256 | 14.17 | 0.98 | 4.03 h | 508 MiB |
| wd0 | naive | 64 | 3.33 | 0.99 | 0.95 h | 507 MiB |
| wd0 | naive | 128 | 6.52 | 0.96 | 1.85 h | 508 MiB |
| wd0 | naive | 256 | 12.90 | 0.95 | 3.67 h | 511 MiB |
