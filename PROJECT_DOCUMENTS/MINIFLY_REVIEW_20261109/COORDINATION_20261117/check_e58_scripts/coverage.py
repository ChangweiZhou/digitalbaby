import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from collections import Counter
SEEDS = (73129, 93683, 116089)
for f in (0.05, 0.10, 0.20):
    for seed in SEEDS:
        out = []
        for t in N.F4:
            tr, he = N.split(t, f, seed)
            ca = Counter(a for a, b in tr); cb = Counter(b for a, b in tr)
            ma = [v for v in range(83) if ca[v] == 0]; mb = [v for v in range(83) if cb[v] == 0]
            n_unc = sum(1 for a, b in he if a in ma or b in mb)
            n_unc_red = sum(1 for a, b in he if (a in ma or b in mb) and t[0] in ("add", "sub") and N.needs_reduction(t[0], a, b))
            n_red = sum(1 for a, b in he if t[0] in ("add", "sub") and N.needs_reduction(t[0], a, b))
            out.append(f"{t[0]}: absent first {len(ma)}, second {len(mb)}, fewest {min(min(ca[v] for v in range(83)), min(cb[v] for v in range(83)))}; untaught pairs with an absent value {n_unc}" + (f" ({n_unc_red} of {n_red} reduction pairs)" if n_red else ""))
        print(f, seed, " | ".join(out))
