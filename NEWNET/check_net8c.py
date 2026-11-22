"""net8c (C) against net8 (numpy), step by step, in net9's configuration of the slow stage (N=1024, gran=8192, tau_max=150,
center=1000, out_var=1000, eta_o=0.1, fb=none). For each case: 3,000 bytes with learning, with every 500 bytes a state saved,
40 bytes read with learning (as a rehearsal) and the state put back (as net9's get_state/set_state); then 500 bytes with
learning off. At every step: the answer (argmax of O) must be identical; h, y, m, z, O, vy, A are compared (largest absolute
difference over all steps). fbm=yesno: the answer given (gave) is the tissue's own argmax, as in net9 with eps=0.
Also the time a byte of each, learning on. Usage: OMP_NUM_THREADS=1 python3 check_net8c.py [n=3000]"""
import sys, time, numpy as np
from net8 import Tissue as A8
from net8c import Tissue as C8
n = int(dict(x.split("=") for x in sys.argv[1:]).get("n", 3000))
here = "/home/user/digitalbaby/"
SH = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8).astype(int)
kw = dict(N=1024, tau_max=150, center=1000, fb="none", gran=8192, out_var=1000, eta_o=0.1)
KEYS = ("h", "y", "m", "z", "O", "vy", "A")
STATE = ("h", "y", "z", "O", "byte", "prev_byte", "yprev", "hprev_byte", "m")


def stream(kind, V, L):
    if kind == "text": return SH[:L] % V
    rng = np.random.default_rng(1001); G = 8                                       # t13's A stream: start, G middle bytes, end by branch
    pool = np.arange(97, 123) if V == 128 else np.arange(V)
    u = rng.choice(pool, G + 4, replace=False); mid = list(u[:G]); st, en = u[G:G + 2], u[G + 2:]; s = []
    while len(s) < L: b = rng.integers(2); s += [st[b]] + mid + [en[b]]
    return np.array(s[:L])


def save(t): return {k: (getattr(t, k).copy() if hasattr(getattr(t, k), "copy") else getattr(t, k)) for k in STATE}


def put(t, st):
    for k, v in st.items(): setattr(t, k, v)


def run(kind, V, fbm):
    a, b = A8(seed=1, n_values=V, fbm=fbm, **kw), C8(seed=1, n_values=V, fbm=fbm, **kw)
    assert b.c
    s = stream(kind, V, n + 600); diff = {k: 0.0 for k in KEYS}; bad = None; ta = tb = 0.0; steps = 0
    def one(x, learn, timed):
        nonlocal ta, tb
        t0 = time.perf_counter(); a.step(int(x), learn); t1 = time.perf_counter(); b.step(int(x), learn); t2 = time.perf_counter()
        if timed: ta += t1 - t0; tb += t2 - t1
        ga, gb = a.answer(), b.answer()
        if fbm == "yesno": a.gave, b.gave = ga, gb
        for k in KEYS: diff[k] = max(diff[k], float(np.abs(getattr(a, k) - getattr(b, k)).max()))
        return ga == gb
    for t in range(n):
        if t and t % 500 == 0:                                                         # a rehearsal: 40 bytes learned, then the state put back
            ka, kb = save(a), save(b); a.byte = b.byte = None; ga, gb = a.gave, b.gave; a.gave = b.gave = None
            for x in s[n + (t // 500) * 7: n + (t // 500) * 7 + 40]:
                if not one(x, True, False) and bad is None: bad = ("rehearsal", t)
            put(a, ka); put(b, kb); a.gave, b.gave = ga, gb
        if not one(s[t], True, True) and bad is None: bad = ("learning", t)
        steps += 1
    for t in range(500):
        if not one(s[n + 100 + t], False, False) and bad is None: bad = ("learning off", t)
    exact = all(v == 0 for v in diff.values())
    print(f"{kind:4s} V={V:3d} fbm={fbm:5s}: answers {'identical at every step' if bad is None else 'DIFFER first at %s byte %d' % bad};"
          f" states {'exactly equal' if exact else 'largest |diff| ' + ' '.join(f'{k} {v:.1e}' for k, v in diff.items())};"
          f" {'within 1e-9' if max(diff.values()) <= 1e-9 else 'NOT within 1e-9'}; ms a byte (learning) numpy {1e3*ta/steps:.2f}, C {1e3*tb/steps:.3f}", flush=True)
    return bad is None and max(diff.values()) <= 1e-9


ok = True
for kind in ("text", "ctx"):
    for fbm in ("teach", "yesno"):
        for V in (32, 64, 128): ok &= run(kind, V, fbm)
print("fallback (fb=kp, no expansion): C step used =", C8(seed=1, n_values=32, N=64, fb="kp").c)
print("ALL PASS" if ok else "SOME CASE FAILED")
