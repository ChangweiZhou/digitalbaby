# Headroom check summary: main

Ladder (held bits/symbol): uniform 8.006, unigram 4.445, bigram 3.538, ppmc2 2.986, ppmc3 2.667, ppmc4 2.606, ppmc5 2.616

| Reading | held | acc | train (class bits) |
|---|---:|---:|---:|
| T1 | 3.463 | 0.312 | 3.390 |
| T2 | 2.963 | 0.426 | 2.003 |
| T3 | 2.699 | 0.470 | 1.818 |
| T4 | 2.625 | 0.482 | 2.123 |

Differences (line-bootstrap SD): T3-T2 -0.263 (0.016), T1-T2 +0.500 (0.018), T4-PPM4 +0.018 (0.019)

Qualification: {'T4_at_most_2.90': True, 'T3_not_cut_off': True, 'T4_not_cut_off': True}
Parity: {'applies': True, 'T2_within_0.005_of_v153': True, 'ladder_within_0.001_of_v153': True}; code parity True
Thresholds: beat counting <= 2.556; room <= 2.863
**Verdict: ROOM_BUT_NOT_COUNTING**
