#!/usr/bin/env python3
"""Step 2: the compiled kernel against the numpy program on the test text (the first 3,000 bytes of the stream, 64 cells), every arm, both forms
of the answering cell; then, if a copy of nano/loop/loop.py is given, multi.py with every line read against that program's numpy reference.
    python3 check_text.py [path to a folder holding nano/loop's loop.py]"""
import sys, time
import numpy as np
import textm

def cmp(kw):
    t = time.process_time(); oa = textm.run(dict(kw, label="chk", nowrite="1", fast="0")); a = textm.run.last; ta = time.process_time() - t
    t = time.process_time(); ob = textm.run(dict(kw, label="chk", nowrite="1", fast="1")); b = textm.run.last; tb = time.process_time() - t; worst = 0.0
    for k in ("W1", "b1", "W2", "b2", "R", "PW", "M", "PA", "PB"):
        if getattr(a, k, None) is None: continue
        x, y = np.asarray(getattr(a, k), float), np.asarray(getattr(b, k), float); worst = max(worst, np.abs(x - y).max() / (np.abs(x).max() + 1e-300))
    ra, rb = oa["rows"][-1], ob["rows"][-1]
    same = (ra["made_lines"], ra["made_pairs"], ra["stream_right"], ra["right"]) == (rb["made_lines"], rb["made_pairs"], rb["stream_right"], rb["right"]) and abs(ra["nats"] - rb["nats"]) < 1e-9 and abs(ra["stream_nats"] - rb["stream_nats"]) < 1e-9
    return worst, same, ta, tb, (ra["made_lines"], ra["made_pairs"])

if __name__ == "__main__":
    bad = 0; base = dict(seed="3", H="64", bytes="3000", passes="2", rewire="64")
    arms = [("64 lines", dict(nline="64")), ("32 lines + 32 pairs", dict(nline="32", npair="32")), ("64 pairs", dict(nline="0", npair="64")),
            ("32 lines", dict(nline="32")), ("16 lines + 16 pairs", dict(nline="16", npair="16")), ("32 pairs", dict(nline="0", npair="32")),
            ("32 + 32, pairs tried among the active", dict(nline="32", npair="32", pdraw="active")), ("32 + 32, neighbouring delays, among the active", dict(nline="32", npair="32", pdraw="active", adj="1")),
            ("64 pairs, among the active", dict(nline="0", npair="64", pdraw="active")), ("every line read, no rewiring", dict(rewire="0"))]
    for ans, extra in (("P", dict(eta1="1", eta2="1", rscale="0.3")), ("LS", dict(eta1="1/4", eta2="1/16"))):
        for title, kw in arms:
            w, same, ta, tb, made = cmp(dict(base, ans=ans, **extra, **kw)); bad += (w > 1e-6) or not same
            print(f"{ans:2s} {title:46s} largest difference over largest strength {w:.1e}  scores, losses and turnover equal: {same}  made {made}  numpy {ta:4.1f} s, compiled {tb:4.1f} s")
    if len(sys.argv) > 1:                             # nano/loop's own numpy reference, every line read
        sys.path.insert(0, sys.argv[1]); sys.dont_write_bytecode = True; import loop
        x, cut, S, pad = loop.load()
        for soft, e1, e2, rs in ((1, 1.0, 1.0, 0.3), (0, 0.25, 1 / 16, 1.0)):
            a = loop.Sys(3, 64, 10, S, pad, eta1=e1, eta2=e2, rscale=rs, soft=soft); a.read(x[:2999], x[1:3000], fast=False)
            for fast in ("0", "1"):
                textm.run(dict(label="chk", nowrite="1", seed="3", H="64", bytes="3000", passes="1", ans="P" if soft else "LS", eta1=str(e1), eta2=str(e2), rscale=str(rs), fast=fast)); b = textm.run.last
                w = max(np.abs(a.W1t.T - b.W1).max(), np.abs(a.b1 - b.b1).max(), np.abs(a.W2 - b.W2).max(), np.abs(a.b2 - b.b2).max(), np.abs(a.Rt.T - b.R).max()); bad += w > 1e-6 or a.right != b.right
                print(f"nano/loop loop.py (numpy, {'P' if soft else 'LS'}) against multi.py, every line read, fast={fast}: largest difference {w:.1e}; bytes named right {a.right} and {b.right}")
    print("CHECK", "FAILED" if bad else "PASSED")
