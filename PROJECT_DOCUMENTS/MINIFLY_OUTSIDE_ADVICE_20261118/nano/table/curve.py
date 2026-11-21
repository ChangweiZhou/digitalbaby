#!/usr/bin/env python3
"""PLAN_25.md: one life of about six million bytes of prose, read once, the loss while going on (a block of 10,000 bytes at a time) and the store, at every doubling.
    python3 curve.py"""
import json, os, sys, time
import numpy as np
import gate, score
from gate import G, Sys2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..", ".."); S = 65; BLOCK = 10000
BOOKS = [("minifly_v148_pretrain_probe_dev_20261025", "bryant-stories"), ("minifly_v148_pretrain_probe_dev_20261025", "burgess-busterbrown"), ("minifly_v148_pretrain_probe_dev_20261025", "carroll-alice"),
         ("minifly_v153_bits_dev_v1_20261025", "chesterton-thursday"), ("minifly_v149_input_ceiling_dev_20261025", "chesterton-brown"), ("minifly_v153_bits_dev_v1_20261025", "austen-persuasion"),
         ("minifly_v155_keys_dev_v1_20261026", "chesterton-ball"), ("minifly_v155_keys_dev_v1_20261026", "austen-emma"), ("minifly_v155_keys_dev_v1_20261026", "melville-moby_dick"), ("minifly_v149_input_ceiling_dev_20261025", "edgeworth-parents")]
def load():
    xa, cut, _ = score.load(); vals = score.byte_values(); tab = np.full(256, 1, np.int64); tab[vals] = np.arange(S); parts = [("tiny-shakespeare", xa)]
    for folder, b in BOOKS:
        raw = np.frombuffer(open(os.path.join(ROOT, folder, "data", b + ".txt"), "rb").read(), np.uint8); raw = raw[raw != 13]; parts.append((b, tab[raw]))
    x = np.concatenate([p[1] for p in parts]); bounds = np.cumsum([0] + [len(p[1]) for p in parts]); return x, [p[0] for p in parts], bounds
if __name__ == "__main__":
    t0 = time.process_time(); Sys2.GATE = 0; Sys2.RULE = 2; Sys2.ALPHA = 0.03; Sys2.ROOM = 1; G.Sys = Sys2; x, names, bounds = load(); s = G.prose_sys(x, dict(mode="none")); n = len(x); D = s.D
    L1 = np.zeros(n); L2 = np.zeros(n); sizes = []
    for lo in range(D, n, BLOCK):
        hi = min(n, lo + BLOCK); ctx = G.ctx_of(x, lo, hi, D); y = x[lo:hi]
        s.cell = 1; P = s.say(ctx); L1[lo:hi] = -np.log(P[np.arange(hi - lo), y]); s.cell = 2; P = s.say(ctx); L2[lo:hi] = -np.log(P[np.arange(hi - lo), y]); s.learn(ctx, y)
        sizes.append((hi, int((s.N[:s.top] > 0).sum()), int((s.C[:s.top] > 0).sum())))
    out = dict(bytes=n, texts=[dict(name=nm, lo=int(a), hi=int(b), first=float(L1[max(a, D):b].mean()), second=float(L2[max(a, D):b].mean())) for nm, a, b in zip(names, bounds[:-1], bounds[1:])], stretches=[], cpu=None)
    a = 100000
    while a < n:
        b = min(2 * a, n); cells = next(c for h, c, k in sizes if h >= b); counts = next(k for h, c, k in sizes if h >= b)
        out["stretches"].append(dict(lo=a, hi=b, first=float(L1[a:b].mean()), second=float(L2[a:b].mean()), cells_at_hi=cells, counts_above_zero_at_hi=counts)); a = b
    out["first_100000"] = dict(first=float(L1[D:100000].mean()), second=float(L2[D:100000].mean())); out["parts_used"] = int((s.Gu > 0).sum()); out["cpu"] = round(time.process_time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "res", "curve.json"), "w"))
    print("bytes", n, "| cpu", out["cpu"]); print("stretch (bytes read) | first cell | second cell | gain | cells | counts above zero")
    for r in out["stretches"]: print(f"  {r['lo']:>8} to {r['hi']:>8} | {r['first']:.4f} | {r['second']:.4f} | {r['first'] - r['second']:.4f} | {r['cells_at_hi']} | {r['counts_above_zero_at_hi']}")
    print("by text (in the order read):"); [print(f"  {t['name']:22s} {t['hi'] - t['lo']:>8} bytes | first {t['first']:.4f} | second {t['second']:.4f}") for t in out["texts"]]
