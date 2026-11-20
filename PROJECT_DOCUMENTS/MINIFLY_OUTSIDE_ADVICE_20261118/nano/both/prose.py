#!/usr/bin/env python3
"""nano/both, PLAN_6.md: the relay system's trust as a given sender on prose, along the stream and on the last tenth.
nano/table/relay.py's arm 1 unchanged (imported): T2's cells that count, one answering cell, the relay system of nano/loop as a given sender.
    python3 prose.py [seed=1] [pieces=20]"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); TAB = os.path.join(HERE, "..", "table"); sys.path.insert(0, TAB); sys.path.insert(1, os.path.join(HERE, "..", "loop")); os.chdir(TAB)
import score, words, relay
kw = dict(a.split("=") for a in sys.argv[1:]); seed = int(kw.get("seed", 1)); pieces = int(kw.get("pieces", 20)); t0 = time.process_time(); S = relay.S
x, cut, _ = score.load(); keys = words.word_keys(x); X, xlev, alone = relay.opinions(0, 1024, seed, 512); s = relay.SysX(x, keys, 6, X, xlev); rows = []; edges = np.linspace(6, cut, pieces + 1).astype(int)
for lo, hi in zip(edges[:-1], edges[1:]):
    P = s.go(lo, hi, True, True); y = x[lo:hi]; Xs = X[lo:hi].astype(np.float64); lv = xlev[lo:hi]; n = hi - lo; g = Xs[np.arange(n), y] - (P * Xs).sum(1); rr = Xs.argmax(1) == y; wr = P.argmax(1) == y; py = P[np.arange(n), y]
    row = dict(bytes_read=int(hi), trust=s.T[-1, :10].tolist(), by_bin=[])
    for b in range(10):
        m = lv == b; row["by_bin"].append(dict(bytes=int(m.sum()), relay_first_choice_right=float(100 * rr[m].mean()) if m.any() else None, whole_answer_right=float(100 * wr[m].mean()) if m.any() else None,
                                               mean_of_what_moves_the_trust=float(g[m].mean()) if m.any() else None, whole_answer_share_of_the_true_byte=float(py[m].mean()) if m.any() else None, relay_share_of_the_true_byte=float(np.exp(Xs[m, y[m]]).mean()) if m.any() else None))
    rows.append(row)
def tenth(): return score.score(s.go(cut, len(x), False, True))["nats"]
keep = s.T[-1].copy(); res = dict(seed=seed, relay_alone=alone, rows=rows, last_tenth=dict(as_learned=tenth(), trust_as_learned=keep[:10].tolist())); lt = res["last_tenth"]
lvt = xlev[cut:]; lt["bytes_by_bin"] = np.bincount(lvt, minlength=10).tolist(); lt["relay_first_choice_right_by_bin"] = [float(100 * (X[cut:][lvt == b].argmax(1) == x[cut:][lvt == b]).mean()) if (lvt == b).any() else None for b in range(10)]
s.T[-1, :10] = 0; lt["relay_silent"] = tenth(); s.T[-1] = keep
for name, bins in (("top_bin", [9]), ("top_three_bins", [7, 8, 9]), ("all_bins", list(range(10)))):
    for v in (0.5, 1.0, 2.0): s.T[-1, bins] = v; lt[f"{name}_at_{v}"] = tenth(); s.T[-1] = keep
res["cpu"] = round(time.process_time() - t0, 1); json.dump(res, open(os.path.join(HERE, "res", f"s6_prose_seed{seed}.json"), "w"))
print("relay alone", alone, "| cpu", res["cpu"]); print("last tenth:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in lt.items() if not isinstance(v, list)})
print("bytes by bin on the last tenth", lt["bytes_by_bin"], "| relay right by bin", [None if v is None else round(v, 1) for v in lt["relay_first_choice_right_by_bin"]])
for r in rows[::max(1, pieces // 5)] + rows[-1:]: print(r["bytes_read"], "trust", [round(v, 3) for v in r["trust"]])
r = rows[-1]; print("last piece of the stream by bin: bytes, relay right, whole answer right, mean of what moves the trust, whole answer's share of the true byte, relay's share")
for b, q in enumerate(r["by_bin"]): print(" ", b, q["bytes"], *[None if q[k] is None else round(q[k], 3) for k in ("relay_first_choice_right", "whole_answer_right", "mean_of_what_moves_the_trust", "whole_answer_share_of_the_true_byte", "relay_share_of_the_true_byte")])
