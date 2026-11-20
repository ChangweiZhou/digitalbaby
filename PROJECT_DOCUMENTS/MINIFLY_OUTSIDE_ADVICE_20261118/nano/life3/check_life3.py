#!/usr/bin/env python3
"""The compiled kernels against the numpy references on reduced lives of three tasks with fifteen byte values: every strength, count and signal
compared after the life, and every reading.  python3 check_life3.py > check_life3_output.txt"""
import numpy as np, life3
def one(arm, plan, every, **k):
    o = {}; st = {}
    for fast in (0, 1):
        kw = dict(label="check", seed="1", arm=arm, plan=plan, every=str(every), fast=str(fast), nowrite=1, **{a: str(b) for a, b in k.items()}); o[fast] = life3.run(kw); s = life3.run.last
        if arm == "S":
            q = s.s
            if fast: st[fast] = dict(T=q.T.copy(), Tu=q.Tu.copy(), size=q.size())
            else: st[fast] = dict(T=q.T.copy(), Tu=q.Tu.copy(), size=q.size())
        else:
            st[fast] = {a: np.array(getattr(s, a), float).copy() for a in ("W1", "b1", "W2", "b2", "R")}
            if arm == "K": st[fast].update({a: np.array(getattr(s, a), float).copy() for a in ("seen", "A", "C", "n", "nl")}); st[fast]["born"] = s.born[:s.nl].astype(float).copy()
            if arm in "OR": st[fast]["cells"] = np.array(s.cells.size(), float)
    d = max(float(np.abs(np.asarray(st[0][a], float) - np.asarray(st[1][a], float)).max()) for a in st[0] if a != "size"); keys = [a for a in o[0]["rows"][0] if a.split("_")[0] in ("add", "sub", "max")]
    same = all(r0[a] == r1[a] for r0, r1 in zip(o[0]["rows"], o[1]["rows"]) for a in keys) and (arm != "S" or st[0]["size"] == st[1]["size"])
    last = o[1]["rows"][-1]; extra = f"; cells joined {last['joined']} (at the three phase ends: {[r['joined'] for r in o[1]['rows'] if r['lines'] == max(q['lines'] for q in o[1]['rows'] if q['phase'] == r['phase'])]})" if arm == "K" else ""
    print(f"arm {arm} {k or ''} plan {plan}: largest difference of any strength, count or signal {d:.1e}; all {len(o[0]['rows'])} readings the same: {same}; last reading add/sub/max {last['add']}/{last['sub']}/{last['max']}{extra}", "PASSED" if d < 1e-9 and same else "FAILED", flush=True)
one("P", "add:1500,sub:800,max:800", 400)
one("O", "add:1000,sub:600,max:600", 400)
one("R", "add:1000,sub:600,max:600", 400)
one("K", "add:1600,sub:1600,max:1600", 400)
one("K", "add:1200,sub:1200,max:1200,sat:1200", 600, H=300, young=2000)
one("S", "add:150,sub:100,max:100", 50)
one("S", "add:150,sub:100,max:100", 50, m=3, shrink=0, eta=0.008, e=0.1)
