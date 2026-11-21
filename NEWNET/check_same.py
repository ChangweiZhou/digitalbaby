"""Check that net5.py gives identical cells, strengths, sources, reliabilities and holds to a reference copy (argv[1]) over 2,000 bytes, and time both."""
import sys, time, importlib.util, numpy as np
def load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A = load(sys.argv[1], "ref").Tissue(n_grow=8)
B = load("/home/user/digitalbaby/NEWNET/net5.py", "new").Tissue(n_grow=8)
rng = np.random.default_rng(5); mid = list(rng.choice(32, 10, replace=False)); s = []
while len(s) < 2000: s += [mid[0] if rng.random() < .5 else mid[1]] + mid[2:]
ta = tb = 0
for x in s[:2000]:
    t = time.time(); ca, _ = A.step(x); ta += time.time() - t
    t = time.time(); cb, _ = B.step(x); tb += time.time() - t
    assert (ca == cb).all() and (A.rw == B.rw).all() and (A.rsrc == B.rsrc).all() and (A.rel == B.rel).all() and (A.hold == B.hold).all()
print(f"identical over 2000 bytes; ms a byte: ref {ta/2:.2f}, new {tb/2:.2f}")
