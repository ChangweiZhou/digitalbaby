import random
from address import ARMS, ROWS, codes, parent
from pathlib import Path
import numpy as np

rng = random.Random(198)
prefixes = [b"", b"x", b"xx", b"x"*32, b"xy"*20] + [
    rng.randbytes(n) for n in range(33) for _ in range(24)
]
changed = 0
for p in prefixes:
    c,q = codes(p)
    expected, coef = parent.bag(p, ROWS)
    assert np.array_equal(c["SHORT"][0], expected)
    drops = []
    for arm in ARMS:
        rows, f = c[arm]
        assert len(set(rows)) == len(expected) and np.array_equal(f,coef)
        assert np.array_equal(rows,codes(b"ignored"+p)[0][arm][0]) if len(p)>=16 else True
        if arm != "SHORT":
            drops.append(set(expected)-set(rows))
    assert drops[0] == drops[1] and len(drops[0]) == q
    changed += q > 0
assert changed
parent.write(Path("CYCLE2_TEST.json"),dict(status="PASS",prefixes=len(prefixes),nonzero_replacement=changed))
