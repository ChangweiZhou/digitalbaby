import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from collections import Counter
SEEDS = (73129, 93683, 116089)
def split_nested(t, seed, f):
    taught, rest = N.split(t, 0.10, seed)
    pairs = taught + rest
    k = round(f * len(pairs))
    return pairs[:k], pairs[k:]
for seed in SEEDS:
    for t in N.F4:
        assert split_nested(t, seed, 0.10) == N.split(t, 0.10, seed)
        a, b, c = (set(split_nested(t, seed, f)[0]) for f in (0.05, 0.10, 0.20))
        assert a < b < c and (len(a), len(b), len(c)) == (344, 689, 1378)
    out = []
    for t in N.F4[:2]:
        tr, he = split_nested(t, seed, 0.05)
        ca = Counter(x for x, y in tr); cb = Counter(y for x, y in tr)
        ma = {v for v in range(83) if ca[v] == 0}; mb = {v for v in range(83) if cb[v] == 0}
        red = [(x, y) for x, y in he if N.needs_reduction(t[0], x, y)]
        unc = sum(1 for x, y in red if x in ma or y in mb)
        common = [(x, y) for x, y in split_nested(t, seed, 0.20)[1] if N.needs_reduction(t[0], x, y)]
        unc_c = sum(1 for x, y in common if x in ma or y in mb)
        out.append(f"{t[0]}: values absent at 0.05 first {len(ma)} second {len(mb)}; reduction pairs untaught at 0.05 with an absent value {unc} of {len(red)}; reduction pairs untaught at 0.20 (common to the three fractions) {len(common)}, with a value absent at 0.05: {unc_c}")
    print(seed, " | ".join(out))
print("nested split equals exp11_cell.split at 0.10 on all seeds and operations; taught sets are nested 344 < 689 < 1378")
