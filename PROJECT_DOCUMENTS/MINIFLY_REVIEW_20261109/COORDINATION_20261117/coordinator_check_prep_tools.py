"""Coordinator's checks of the three delivered tools against what is already verified. Read-only."""
import sys, json, hashlib, numpy as np
from pathlib import Path
from random import Random
P, V = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(P))
import life_stream as LS, readings as RD, reserve as RS
# 1. the life's first 262,144 bytes against the stream of my own verification script (decision 73)
src = (V / "e63_stage_a_verify.py").read_text(); ns = {"__name__": "x", "sys": sys}
exec(src[src.index("SYM ="):src.index("ST = stream")], {"Random": Random, "np": np}, ns)
for seed in (73129, 93683, 116089):
    L = LS.life(seed); mine = ns["stream"](seed, 262144) if False else None
import types
g = {"Random": Random, "np": np}; exec(src[src.index("SYM ="):src.index("ST = stream")], g)
for seed in (73129, 93683, 116089):
    L = LS.life(seed); mine = g["stream"](seed, 393216)
    print(f"seed {seed}: the life's first 393,216 bytes equal my own stream: {bool((L['bytes'][:393216] == mine).all())}; life length {len(L['bytes']):,}")
    by, du, cl, ph = L["bytes"], L["due"], L["cls"], L["phase"]
    # what is due in KNOWABLE: is it fixed by the three bytes before it?
    ctx = {}
    idx = np.nonzero(ph == 1)[0]
    for i in idx.tolist():
        ctx.setdefault(bytes(by[i - 3:i].tolist()), set()).add(int(by[i]))
    amb = sum(1 for i in idx[du[idx] > 0].tolist() if len(ctx[bytes(by[i - 3:i].tolist())]) > 1)
    notdue_fixed = sum(1 for i in idx[du[idx] == 0].tolist() if len(ctx[bytes(by[i - 3:i].tolist())]) == 1)
    print(f"   KNOWABLE: due positions {int((du[idx] > 0).sum()):,}, of which the three bytes before do not fix the byte: {amb}; positions not marked due whose byte the three bytes before do fix (within this phase): {notdue_fixed:,} of {int((du[idx] == 0).sum()):,}")
    # the reserve: my own split against reserve.py's division, and the life never shows a reserve pair
    D = json.loads((P / "run2" / f"reserve_{seed}.json").read_text()) if (P / "run2" / f"reserve_{seed}.json").exists() else None
    for op, m, code in LS.DEV:
        pairs = [(a, b) for a in range(m) for b in range(m)]; Random(seed * 1000 + 50 + m * 7 + code[0]).shuffle(pairs)
        res = set(pairs[3960:])
        if D:
            got = set(tuple(p) for part in D["division"][op].values() for p in part); sizes = [len(v) for v in D["division"][op].values()]
            print(f"   {op}: reserve.py's three parts together equal the second half of my own split: {got == res}; part sizes {sizes}")
# 2. readings.py on all three tissues of stage A against the receipts, and on the pair code of check 9(e)
ST = g["stream"](73129, 16384); worst = 0.0; nsh = 0
for tis in ("A_first_73129", "A_coincidence_73129", "A_fixed_73129"):
    rec = json.loads((V / "runs" / tis / "receipt.json").read_text())
    for w in ("A1", "A2"):
        z = np.load(V / "runs" / tis / f"{w}.npz"); r = RD.read_window(z["x"], z["bytes"], z["distance"], 73129, stream=ST, start=int(z["positions"][0])); m = rec["measures"][w]
        for d in (1, 2, 4):
            a, b = r["shares"][d], m["shares"][str(d)]; nsh += any(a[k] != b[k] for k in ("n", "informative", "silent", "constant"))
            for L in (0, 1, 2, 4, 8): worst = max(worst, abs(r["scores"][d]["by_lag"][L]["S"] - m["scores"][str(d)]["by_lag"][str(L)]["S"]))
        for L in (0, 1, 2, 4, 8): worst = max(worst, abs(r["c0"][L]["S"] - m["C0"][str(L)]))
        for L in (1, 2, 4, 8): worst = max(worst, abs(r["c_lt"][L]["S"] - m["C_lt"][str(L)]))
print(f"readings.py on the three tissues, both windows: shares differing {nsh} of 18; largest |difference| from the receipts over 90 S, 10 C0 and 8 C<L (each twice or thrice): {worst:.2e}")
w0, wl = 12288, 4096; idx = np.arange(wl); keys = []
for t in range(w0, w0 + wl):
    j = t - 1; n = 0
    while j >= 0 and ST[j] != 10: j -= 1; n += 1
    keys.append((int(ST[t - 1]), 0 if ST[t - 1] == 10 else n))
vals = sorted(set(keys)); F = RD._onehot(keys, vals)
S, se, k = RD.classify(F, ST[w0:w0 + wl].astype(np.int64), idx, wl)
print(f"readings.classify on a one-hot code of (byte one back, bytes since the last end mark) in A2, scoring the present byte: {S:.4f} (the worker's check 9(e): 0.0903; the brief's first figure: 0.0946)")
