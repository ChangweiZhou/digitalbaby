# The depth check on positions where the earlier bytes tell nothing (21 November 2026)

Development level; readings of the committed records of plans 8 and 9 (`letters_read.py`); no new run. Positions: the present byte is a distractor letter (drawn independently and uniformly from A-Z) and the byte before it is one too, 1,218 to 1,293 positions of window A2. On them the previous byte alone scores at chance (0.022 to 0.039). Level: the largest score of 220 label shuffles (20 at each distance) of the same classifier on the same cells and positions.

| P8_coincidence_B10i | 1293 | 0.022 | 0.992 0.946 0.824 0.635 0.441 0.282 0.189 0.172 0.123 0.082 0.060 | 0.040 / 0.062 | 10 |
| P8_first_B10 | 1293 | 0.022 | 1.000 0.960 0.821 0.581 0.450 0.273 0.217 0.183 0.134 0.113 0.067 | 0.039 / 0.060 | 11 |
| P8_first_B10i | 1293 | 0.022 | 1.000 0.959 0.845 0.500 0.281 0.169 0.114 0.105 0.078 0.058 0.060 | 0.039 / 0.058 | 11 |
| P9_coincidence_B10i_cap12 | 1293 | 0.022 | 0.999 0.959 0.801 0.540 0.455 0.342 0.222 0.139 0.087 0.080 0.067 | 0.040 / 0.056 | 11 |
| P9_coincidence_B10i_cap4 | 1293 | 0.022 | 1.000 0.923 0.844 0.681 0.531 0.306 0.193 0.156 0.151 0.111 0.095 | 0.039 / 0.054 | 11 |
| P9_coincidence_B10i_cap6 | 1293 | 0.022 | 1.000 0.923 0.829 0.605 0.509 0.400 0.187 0.181 0.170 0.125 0.062 | 0.039 / 0.054 | 11 |
| P9_coincidence_B10i_cap8 | 1293 | 0.022 | 1.000 0.983 0.879 0.664 0.500 0.393 0.218 0.121 0.104 0.091 0.085 | 0.039 / 0.054 | 11 |
| P9_coincidence_B10i_s116089 | 1218 | 0.039 | 1.000 0.906 0.636 0.488 0.393 0.341 0.307 0.253 0.221 0.206 0.144 | 0.039 / 0.054 | 11 |
| P9_coincidence_B10i_s93683 | 1273 | 0.035 | 0.978 0.920 0.838 0.587 0.544 0.381 0.301 0.272 0.215 0.196 0.214 | 0.039 / 0.058 | 11 |
| P9_first_B10_cap12 | 1293 | 0.022 | 0.984 0.969 0.884 0.600 0.463 0.345 0.227 0.186 0.159 0.111 0.085 | 0.039 / 0.054 | 11 |
| P9_first_B10_cap4 | 1293 | 0.022 | 1.000 0.923 0.852 0.653 0.390 0.281 0.173 0.142 0.139 0.112 0.121 | 0.040 / 0.056 | 11 |
| P9_first_B10_cap6 | 1293 | 0.022 | 0.981 0.899 0.764 0.460 0.390 0.317 0.161 0.148 0.143 0.061 0.065 | 0.040 / 0.058 | 11 |
| P9_first_B10_cap8 | 1293 | 0.022 | 1.000 0.883 0.787 0.532 0.558 0.359 0.238 0.216 0.149 0.152 0.094 | 0.039 / 0.054 | 11 |
| P9_first_B10_s116089 | 1218 | 0.039 | 1.000 0.980 0.824 0.758 0.478 0.330 0.279 0.218 0.185 0.142 0.095 | 0.039 / 0.056 | 11 |
| P9_first_B10_s93683 | 1273 | 0.035 | 1.000 0.977 0.876 0.623 0.413 0.301 0.222 0.171 0.163 0.143 0.136 | 0.040 / 0.056 | 11 |

**Result.** In all 15 runs (three seeds, both forms, four caps) the present letter is decodable above the shuffle level at d = 10, and in 14 of 15 at d = 11 (the farthest distance in the tissue). The scores at d = 10 to 11 are 0.06 to 0.21 against a level of 0.054 to 0.062 and a null mean of 0.039 to 0.040. Since no earlier byte predicts these letters, the score there cannot come from the text's statistics: the present byte itself reaches the far side of the tissue.

**What this does not show.** Whether downstream cells can use a signal this weak (the scores at d >= 9 are a quarter or less of those at d = 1); that the signal is relayed connection by connection rather than over several paths (a clamp of the near cells was not run); anything beyond seeds 73129, 93683 and 116089, 16,384 bytes, and these settings. It is a check by the same session that ran the experiments, not an outside review.
