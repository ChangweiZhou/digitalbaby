# Plan 9: plan 8's best on two more seeds, and a cap on connections at totals 1 (21 November 2026)

Written before its runs. Development level. **Authorisation:** owner, 21 November 2026 ("继续做"; "要的就是快速迭代"). **Program:** `TISSUE_EVENT/` at this commit; the second run of every tissue without records is now off by default (`--verify-core` turns it on), which halves the time; nothing else changed.

**Arms** (all with turnover, trial threshold 1, making period 32,768, withdrawal by use, 16,384 bytes; first form C_E = C_I = 1, trial 2,048; coincidence form C_E 1, C_I 1.5, trial 512):
- **Seeds:** plan 8's first-form B10 and coincidence-form B10i on seeds 93683 and 116089 (4 tissues).
- **Cap:** the same two settings on seed 73129 with `--cap` 4, 6, 8, 12 (incoming connections, lines included) (8 tissues).

**H9a.** On both new seeds both settings are at level at d = 8. **H9b.** Some cap carries the byte farther than no cap (plan 8: d = 8 in both forms). Refuted case by case.

| No. | Prediction | Chance |
|---|---|---|
| K1 | H9a holds in all four cases | 0.4 |
| K2 | H9b holds in some form | 0.35 |
| K3 | Cap 4 is worse than cap 8 in both forms | 0.6 |

**Cost:** 12 tissues at about 40 s each, 3 rounds on 4 cores: about 3 minutes.

## Outcome (entered 21 November 2026; twelve runs in 142 s of wall time on 4 cores; `tab.py`)

| Run | S(d,0), d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte (with records) |
|---|---|---|---|---|---|
| P8_first_B10 | 1.000 0.947 0.716 0.524 0.430 0.262 0.202 0.171 0.078 0.071 0.053 | 8 | 5.8 | 6 / 837 | 1.73 |
| P9_first_B10_s116089 | 1.000 0.973 0.853 0.695 0.425 0.315 0.268 0.209 0.134 0.092 0.068 | 9 | 5.6 | 34 / 729 | 1.62 |
| P9_first_B10_s93683 | 1.000 0.944 0.837 0.530 0.306 0.216 0.158 0.119 0.115 0.101 0.085 | 9 | 5.3 | 1 / 514 | 1.27 |
| P9_first_B10_cap12 | 0.986 0.928 0.762 0.542 0.384 0.240 0.194 0.163 0.116 0.081 0.050 | 9 | 5.6 | 13 / 837 | 1.72 |
| P9_first_B10_cap4 | 1.000 0.824 0.705 0.509 0.346 0.245 0.135 0.112 0.114 0.076 0.078 | 9 | 4.7 | 0 / 605 | 1.32 |
| P9_first_B10_cap6 | 0.989 0.808 0.669 0.416 0.332 0.267 0.131 0.114 0.104 0.041 0.040 | 8 | 5.3 | 0 / 750 | 1.29 |
| P9_first_B10_cap8 | 1.000 0.830 0.697 0.525 0.479 0.311 0.187 0.159 0.111 0.099 0.050 | 9 | 5.6 | 1 / 839 | 1.42 |
| P8_coincidence_B10i | 0.995 0.925 0.704 0.486 0.317 0.214 0.156 0.139 0.092 0.055 0.050 | 8 | 5.5 | 0 / 1056 | 2.91 |
| P9_coincidence_B10i_s116089 | 1.000 0.931 0.649 0.555 0.381 0.359 0.325 0.283 0.218 0.175 0.108 | 10 | 5.4 | 40 / 798 | 2.93 |
| P9_coincidence_B10i_s93683 | 0.980 0.890 0.792 0.503 0.395 0.295 0.229 0.218 0.168 0.141 0.131 | 11 | 5.3 | 1 / 761 | 2.90 |
| P9_coincidence_B10i_cap12 | 0.995 0.942 0.701 0.483 0.375 0.264 0.184 0.112 0.092 0.075 0.057 | 8 | 5.3 | 0 / 1122 | 3.10 |
| P9_coincidence_B10i_cap4 | 0.978 0.847 0.744 0.537 0.412 0.282 0.164 0.138 0.109 0.071 0.056 | 8 | 4.6 | 0 / 1096 | 2.70 |
| P9_coincidence_B10i_cap6 | 0.978 0.891 0.754 0.552 0.441 0.329 0.195 0.146 0.115 0.090 0.043 | 9 | 4.9 | 0 / 767 | 2.49 |
| P9_coincidence_B10i_cap8 | 1.000 0.948 0.807 0.560 0.403 0.257 0.161 0.095 0.074 0.060 0.057 | 7 | 5.4 | 10 / 1077 | 2.73 |


| | Outcome |
|---|---|
| H9a | **Held in all four cases**, and farther than on seed 73129: first form at level to d = 9 on both new seeds; coincidence form to d = 10 (116089) and d = 11 (93683; S(11,0) 0.131) |
| H9b | Held in both forms, by one connection: first form caps 4, 8 and 12 reach d = 9 against 8; coincidence form cap 6 reaches 9 against 8 |
| K1 | Held |
| K2 | Held |
| K3 | Failed: in the coincidence form cap 4 (d = 8) is not worse than cap 8 (d = 7) |

**Limits.** One run a setting; the farthest d varies by 1 to 3 between seeds, more than between caps, so the caps' effect is within the spread of seeds. The level 0.1103 was set against the present byte alone; the bytes before it predict the present byte to about 0.2 on their own (`checks/`: a reading of plan 8's window by a helper), so scores near the level at d >= 9 must still be read on the positions where the previous byte tells nothing (the distractor letters) before they count. That reading is the next step.
