# Headroom check summary: smoke

Ladder (held bits/symbol): uniform 8.006, unigram 4.543, bigram 3.913, ppmc2 3.828, ppmc3 3.850, ppmc4 3.882, ppmc5 3.892

| Reading | held | acc | train (class bits) |
|---|---:|---:|---:|
| T1 | 3.968 | 0.250 | 3.392 |
| T2 | 3.930 | 0.262 | 1.445 |
| T3 | 3.929 | 0.263 | 3.019 |
| T4 | 4.002 | 0.258 | 2.692 |

Differences (line-bootstrap SD): T3-T2 -0.001 (0.035), T1-T2 +0.038 (0.031), T4-PPM4 +0.121 (0.076)

Qualification: {'T4_at_most_2.90': False, 'T3_not_cut_off': True, 'T4_not_cut_off': True}
Parity: {'applies': False, 'T2_within_0.005_of_v153': False, 'ladder_within_0.001_of_v153': False}; code parity True
Thresholds: beat counting <= 2.556; room <= 2.863
**Verdict: REFERENCE_NOT_READY**
