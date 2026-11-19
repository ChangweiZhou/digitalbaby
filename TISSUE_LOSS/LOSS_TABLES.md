# Where the present byte is lost: tables (written by loss_tables.py from out/measure_*.json)

Seed 73129, 16,384 bytes, window A2 (bytes 12,288 to 16,383), outputs at the reading step unless said, raw (nothing below 0.01 set to 0). d: distance from the input lines of the bytes present in A2 along the connections present at the end of the window.

## unchanging, no gain

CPU 19 s; connections a cell 7.8; S(d, 0) at d = 1 / 2 / 4: 0.979 / 0.826 / 0.022

| d | cells | 1. mean output | 1. share > 0.01 | 2. eta2 present, median | upper decile | 2. eta2 previous, median | upper decile | 3. eta2 present median at step 4 / 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 102 | 0.0046 | 0.059 | 1.000 | 1.000 | 0.038 | 0.115 | 0.993 / 1.000 / 1.000 / 1.000 |
| 2 | 215 | 0.0009 | 0.034 | 1.000 | 1.000 | 0.037 | 0.074 | 0.742 / 0.992 / 1.000 / 1.000 |
| 3 | 240 | 0.0002 | 0.000 | 1.000 | 1.000 | 0.033 | 0.066 | 0.125 / 0.861 / 0.999 / 1.000 |
| 4 | 220 | 5.4e-05 | 0.000 | 1.000 | 1.000 | 0.030 | 0.057 | 0.043 / 0.282 / 0.969 / 1.000 |
| 5 | 199 | 1.2e-05 | 0.000 | 1.000 | 1.000 | 0.031 | 0.061 | 0.044 / 0.074 / 0.834 / 1.000 |
| 6 | 191 | 2.4e-06 | 0.000 | 0.999 | 1.000 | 0.031 | 0.056 | 0.044 / 0.047 / 0.539 / 0.999 |
| 7 | 178 | 6e-07 | 0.000 | 0.992 | 0.998 | 0.038 | 0.070 | 0.043 / 0.043 / 0.205 / 0.992 |
| 8 | 173 | 1.1e-07 | 0.000 | 0.964 | 0.992 | 0.062 | 0.188 | 0.048 / 0.048 / 0.087 / 0.964 |

4. Potential v changed by 1e-9 at byte 8,192: (one cell) cell 422, the most active at d = 1 (output 0.125); (every cell) every cell that is not a feedback cell. Largest difference of outputs, and of gains and thresholds, after n bytes:

| n | one cell: outputs | cells differing | every cell: outputs | cells differing | gains | thresholds | strengths |
|---|---|---|---|---|---|---|---|
| 1 | 5.6e-15 | 1758 | 4.8e-11 | 1772 | 0.000 | 2e-12 | 0.000 |
| 4 | 1.6e-21 | 1279 | 7.8e-15 | 1796 | 0.000 | 0.000 | 0.000 |
| 16 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| 64 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| 256 | 0.000 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |

5. Share of the absolute arrival at the reading step, mean over the cells at d (lines / nearer / same distance / farther / not reached):

| d | input lines | nearer | same distance | farther | not reached |
|---|---|---|---|---|---|
| 2 | 0.000 | 0.686 | 0.247 | 0.067 | 0.000 |
| 3 | 0.000 | 0.633 | 0.309 | 0.058 | 0.000 |
| 4 | 0.000 | 0.590 | 0.337 | 0.073 | 0.000 |

## unchanging, gain, a*=1/32

CPU 22 s; connections a cell 7.8; S(d, 0) at d = 1 / 2 / 4: 0.731 / 0.145 / 0.038

| d | cells | 1. mean output | 1. share > 0.01 | 2. eta2 present, median | upper decile | 2. eta2 previous, median | upper decile | 3. eta2 present median at step 4 / 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 102 | 0.0289 | 0.081 | 0.228 | 0.624 | 0.020 | 0.039 | 0.219 / 0.290 / 0.309 / 0.228 |
| 2 | 215 | 0.0303 | 0.067 | 0.090 | 0.154 | 0.016 | 0.023 | 0.016 / 0.025 / 0.059 / 0.090 |
| 3 | 240 | 0.0307 | 0.057 | 0.056 | 0.112 | 0.016 | 0.019 | 0.013 / 0.015 / 0.027 / 0.056 |
| 4 | 220 | 0.0314 | 0.057 | 0.038 | 0.112 | 0.016 | 0.019 | 0.014 / 0.015 / 0.020 / 0.038 |
| 5 | 199 | 0.0315 | 0.054 | 0.026 | 0.065 | 0.018 | 0.021 | 0.014 / 0.014 / 0.018 / 0.026 |
| 6 | 191 | 0.0311 | 0.049 | 0.019 | 0.029 | 0.020 | 0.023 | 0.012 / 0.013 / 0.015 / 0.019 |
| 7 | 178 | 0.0302 | 0.048 | 0.015 | 0.022 | 0.021 | 0.024 | 0.012 / 0.012 / 0.013 / 0.015 |
| 8 | 173 | 0.0304 | 0.046 | 0.012 | 0.018 | 0.019 | 0.022 | 0.009 / 0.009 / 0.009 / 0.012 |

4. Potential v changed by 1e-9 at byte 8,192: (one cell) cell 1518, the most active at d = 1 (output 0.126); (every cell) every cell that is not a feedback cell. Largest difference of outputs, and of gains and thresholds, after n bytes:

| n | one cell: outputs | cells differing | every cell: outputs | cells differing | gains | thresholds | strengths |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0 | 4e-12 | 4 | 1.7e-08 | 7.3e-11 | 0.000 |
| 4 | 0.000 | 0 | 1.8e-43 | 2 | 1.8e-08 | 7.3e-11 | 0.000 |
| 16 | 0.000 | 0 | 9.1e-15 | 3 | 1.2e-08 | 6.8e-11 | 0.000 |
| 64 | 0.000 | 0 | 4.2e-07 | 620 | 3.2e-05 | 3.2e-08 | 0.000 |
| 256 | 0.000 | 0 | 0.088 | 8 | 658.208 | 0.414 | 0.000 |

5. Share of the absolute arrival at the reading step, mean over the cells at d (lines / nearer / same distance / farther / not reached):

| d | input lines | nearer | same distance | farther | not reached |
|---|---|---|---|---|---|
| 2 | 0.000 | 0.277 | 0.384 | 0.340 | 0.000 |
| 3 | 0.000 | 0.322 | 0.448 | 0.230 | 0.000 |
| 4 | 0.000 | 0.311 | 0.439 | 0.250 | 0.000 |

## coincidence, gain, a*=1/32, floor 1/128

CPU 189 s; connections a cell 19.4; S(d, 0) at d = 1 / 2 / 4: 0.764 / 0.126 / 0.039

| d | cells | 1. mean output | 1. share > 0.01 | 2. eta2 present, median | upper decile | 2. eta2 previous, median | upper decile | 3. eta2 present median at step 4 / 8 / 16 / 32 |
|---|---|---|---|---|---|---|---|---|
| 1 | 237 | 0.0291 | 0.071 | 0.609 | 0.767 | 0.030 | 0.069 | 0.410 / 0.489 / 0.603 / 0.609 |
| 2 | 121 | 0.0309 | 0.055 | 0.060 | 0.460 | 0.014 | 0.053 | 0.019 / 0.033 / 0.049 / 0.060 |
| 3 | 229 | 0.0314 | 0.052 | 0.023 | 0.206 | 0.013 | 0.025 | 0.014 / 0.018 / 0.023 / 0.023 |
| 4 | 234 | 0.0312 | 0.050 | 0.017 | 0.057 | 0.013 | 0.017 | 0.014 / 0.015 / 0.017 / 0.017 |
| 5 | 217 | 0.0311 | 0.050 | 0.013 | 0.028 | 0.012 | 0.015 | 0.013 / 0.012 / 0.013 / 0.013 |
| 6 | 208 | 0.0311 | 0.050 | 0.012 | 0.016 | 0.012 | 0.015 | 0.011 / 0.012 / 0.012 / 0.012 |
| 7 | 204 | 0.0313 | 0.050 | 0.012 | 0.016 | 0.012 | 0.015 | 0.011 / 0.011 / 0.012 / 0.012 |
| 8 | 202 | 0.0310 | 0.045 | 0.011 | 0.014 | 0.011 | 0.014 | 0.011 / 0.011 / 0.011 / 0.011 |

4. Potential v changed by 1e-9 at byte 8,192: (one cell) cell 2041, the most active at d = 1 (output 0.620); (every cell) every cell that is not a feedback cell. Largest difference of outputs, and of gains and thresholds, after n bytes:

| n | one cell: outputs | cells differing | every cell: outputs | cells differing | gains | thresholds | strengths |
|---|---|---|---|---|---|---|---|
| 1 | 0.000 | 0 | 1.9e-07 | 90 | 1.3e-08 | 2.2e-09 | 4.2e-09 |
| 4 | 0.000 | 0 | 1.000 | 112 | 285.154 | 0.100 | 1.000 |
| 16 | 0.000 | 0 | 0.747 | 11 | 1718.463 | 0.598 | 1.000 |
| 64 | 0.000 | 0 | 1.000 | 317 | 1983.752 | 0.728 | 1.000 |
| 256 | 0.000 | 0 | 0.836 | 41 | 2184.955 | 1.610 | 1.000 |

5. Share of the absolute arrival at the reading step, mean over the cells at d (lines / nearer / same distance / farther / not reached):

| d | input lines | nearer | same distance | farther | not reached |
|---|---|---|---|---|---|
| 2 | 0.000 | 0.174 | 0.321 | 0.505 | 0.000 |
| 3 | 0.000 | 0.269 | 0.497 | 0.235 | 0.000 |
| 4 | 0.000 | 0.268 | 0.516 | 0.216 | 0.000 |

## The one change: the threshold held at 0 (a cell with the gain on controls its activity by its gain alone)

| Tissue | change | S(1,0) | S(1,1) | S(1,2) | S(2,0) | S(2,1) | S(2,2) | S(4,0) | S(4,1) | S(4,2) | eta2 present median d = 1 / 2 / 4 / 8 | mean output d = 1 / 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unchanging, no gain | without | 0.979 | 0.066 | 0.053 | 0.826 | 0.056 | 0.042 | 0.022 | 0.022 | 0.022 | 1.000 / 1.000 / 1.000 / 0.964 | 0.0046 / 5.4e-05 |
| unchanging, no gain | with | 0.979 | 0.066 | 0.053 | 0.826 | 0.056 | 0.042 | 0.022 | 0.022 | 0.022 | 1.000 / 1.000 / 1.000 / 0.964 | 0.0046 / 5.4e-05 |
| unchanging, gain, a*=1/32 | without | 0.731 | 0.045 | 0.036 | 0.145 | 0.023 | 0.026 | 0.038 | 0.022 | 0.029 | 0.228 / 0.090 / 0.038 / 0.012 | 0.0289 / 0.0314 |
| unchanging, gain, a*=1/32 | with | 0.982 | 0.072 | 0.046 | 0.774 | 0.054 | 0.032 | 0.257 | 0.061 | 0.036 | 0.931 / 0.770 / 0.302 / 0.014 | 0.0297 / 0.0309 |
| coincidence, gain, a*=1/32, floor 1/128 | without | 0.764 | 0.057 | 0.042 | 0.126 | 0.033 | 0.029 | 0.039 | 0.027 | 0.025 | 0.609 / 0.060 / 0.017 / 0.011 | 0.0291 / 0.0312 |
| coincidence, gain, a*=1/32, floor 1/128 | with | 0.181 | 0.029 | 0.021 | 0.032 | 0.022 | 0.019 | 0.021 | 0.023 | 0.019 | 0.019 / 0.016 / 0.011 / 0.011 | 0.0259 / 0.0304 |
