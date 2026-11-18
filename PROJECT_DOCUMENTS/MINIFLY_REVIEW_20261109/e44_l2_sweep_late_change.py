"""The reviewer's own addition to the verification of the sweep's long lives (18 November 2026).

The late levels (D4) below are the four-decimal values that e44_l2_sweep_verify.py prints for each life
at 256 and at 1,024 visits (worlds 1 to 4).  This script forms, for each pair of arms, the difference at
each length, its change from 256 to 1,024 visits by world, and the interval of that change in the way
D11 forms an interval (mean of the four worlds +- t(0.95, 3) * sd / 2).  It is a description that was not
fixed beforehand.  Run: python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_l2_sweep_late_change.py
"""
import math

T = 2.353363
LATE = {"naive": [(0.5816, 0.6690), (0.6963, 0.7557), (0.6917, 0.7415), (0.6845, 0.7315)],
        "sweep": [(0.5892, 0.6499), (0.5121, 0.6278), (0.6107, 0.6825), (0.5993, 0.6723)],
        "replay": [(0.5773, 0.7304), (0.7553, 0.8117), (0.7070, 0.8347), (0.6593, 0.7086)],
        "sweep+replay": [(0.6653, 0.7844), (0.5251, 0.5739), (0.5879, 0.7885), (0.6693, 0.7691)]}


def interval(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    h = T * sd / math.sqrt(n)
    return m, m - h, m + h


for a, b in (("sweep", "naive"), ("sweep+replay", "replay")):
    d256 = [LATE[a][w][0] - LATE[b][w][0] for w in range(4)]
    d1024 = [LATE[a][w][1] - LATE[b][w][1] for w in range(4)]
    ch = [d1024[w] - d256[w] for w in range(4)]
    print(f"{a} - {b}: difference at 256 visits", ["%+.4f" % v for v in d256], "at 1,024", ["%+.4f" % v for v in d1024])
    print("   change from 256 to 1,024 visits by world", ["%+.4f" % v for v in ch], "mean %+.4f [%+.4f, %+.4f]" % interval(ch))
for a in LATE:
    g = [LATE[a][w][1] - LATE[a][w][0] for w in range(4)]
    print(f"{a}: late level at 1,024 visits minus at 256, by world", ["%+.4f" % v for v in g], "all above zero:", all(v > 0 for v in g))
