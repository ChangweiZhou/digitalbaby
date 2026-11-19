import sys, json, numpy as np
from pathlib import Path
from random import Random
from collections import Counter, defaultdict
P, V = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(P))
import readings as RD, life_stream as LS
src = (V / "e63_stage_a_verify.py").read_text(); g = {"Random": Random, "np": np}; exec(src[src.index("SYM ="):src.index("ST = stream")], g)
ST = g["stream"](73129, 16384)
print("readings.py against the receipts of stage A: every score that differs by more than 1e-9")
for tis in ("A_first_73129", "A_coincidence_73129", "A_fixed_73129"):
    rec = json.loads((V / "runs" / tis / "receipt.json").read_text())
    for w in ("A1", "A2"):
        z = np.load(V / "runs" / tis / f"{w}.npz"); r = RD.read_window(z["x"], z["bytes"], z["distance"], 73129, stream=ST, start=int(z["positions"][0])); m = rec["measures"][w]
        for d in (1, 2, 4):
            for L in (0, 1, 2, 4, 8):
                a, b = r["scores"][d]["by_lag"][L]["S"], m["scores"][str(d)]["by_lag"][str(L)]["S"]
                if abs(a - b) > 1e-9: print(f"   {tis} {w} S(d={d}, L={L}): readings.py {a:.4f}, receipt {b:.4f}, difference {a - b:+.4f}")
        for L in (0, 1, 2, 4, 8):
            if abs(r["c0"][L]["S"] - m["C0"][str(L)]) > 1e-9: print(f"   {tis} {w} C0(L={L}): {r['c0'][L]['S']:.4f} against {m['C0'][str(L)]:.4f}")
        for L in (1, 2, 4, 8):
            if abs(r["c_lt"][L]["S"] - m["C_lt"][str(L)]) > 1e-9: print(f"   {tis} {w} C<L(L={L}): {r['c_lt'][L]['S']:.4f} against {m['C_lt'][str(L)]:.4f}")
# KNOWABLE: positions not marked due whose byte is fixed by the three bytes before, counting only contexts met at least 20 times
L = LS.life(73129); by, du, ph, cl = L["bytes"], L["due"], L["phase"], L["cls"]
idx = np.nonzero(ph == 1)[0]; ctx = defaultdict(Counter)
for i in idx.tolist(): ctx[bytes(by[i - 3:i].tolist())][int(by[i])] += 1
tally = Counter()
for i in idx[du[idx] == 0].tolist():
    c = ctx[bytes(by[i - 3:i].tolist())]
    if sum(c.values()) >= 20 and len(c) == 1: tally[LS.CLASSES[int(cl[i])]] += 1
print("KNOWABLE, seed 73129: positions not marked due whose byte follows with certainty from the three bytes before (contexts met at least 20 times), by class:", dict(tally))
