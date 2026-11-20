#!/usr/bin/env python3
"""PLAN_3.md: replacement cell by cell. As merge.py, with one change: an R-cell whose X-cell is in a class of a kept pair is silent only if
the class foretells that cell's counts at least as well as the cell foretells them itself (each of its counts from its others); else it goes on speaking beside its class cell.
A cell born after the last finding, whose X-cell is in a class, is silent (it has no record of its own yet). Silent cells go on counting.
    python3 merge3.py check
    python3 merge3.py sums|dev|life  key=value ...      (the arguments of merge.py; rep is set to 2)"""
import json, os, sys
import numpy as np
import merge
HERE = os.path.dirname(os.path.abspath(__file__))

def judge(rows, pooled, S, e):
    """For member cells (rows: their counts) of class cells (pooled: the class's counts at the same z, the member's own included):
    what the class without the member foretells of the member's counts, and what the member's own other counts foretell of each of them."""
    rows = rows.astype(np.float64); pooled = pooled.astype(np.float64); n = rows.sum(1, keepdims=True); N = pooled.sum(1, keepdims=True)
    jc = (rows * (np.log(pooled - rows + e) - np.log(N - n + S * e))).sum(1)
    jo = (rows * (np.log(np.maximum(rows - 1, 0) + e) - np.log(n - 1 + S * e))).sum(1)
    return jc, jo

class Sys3(merge.Sys):
    def __init__(s, *a, **k):
        super().__init__(*a, **k); s.loud = np.zeros(s.nbase, bool); s.judged = np.zeros(s.nbase, bool); s.judge_log = []
    def find(s):
        super().find(); s.loud[:] = False; s.judged[:] = False; quiet = np.zeros(s.nbase, bool); rec = []
        for sl in s.slots:
            if sl is None: continue
            ids_, cod = s.r_cells(sl["k"]); xk, zk = s.split(sl["k"], sl["X"], cod); c = sl["clsmap"][xk]; ok = c >= 0; key = c[ok].astype(np.int64) * sl["ZS"] + zk[ok]
            j = np.searchsorted(sl["keys"], key); ii = ids_[ok]; q = np.zeros(len(ii), bool)
            for a in range(0, len(ii), 100000):
                jc, jo = judge(s.C[ii[a:a + 100000]], s.C[sl["cids"][j[a:a + 100000]]], s.S, s.eJ); q[a:a + 100000] = jc >= jo
            s.judged[ii] = True; quiet[ii[q]] = True; rec.append(dict(R=list(s.kinds[sl["k"]]), X=list(sl["X"]), member_cells=int(len(ii)), silent=int(q.sum())))
        s.loud = s.judged & ~quiet; s.judge_log.append(dict(bytes_read=int(s.read), judged=int(s.judged.sum()), silent=int((s.judged & quiet).sum()), by_pair=rec))
    def ids(s, ctx, create=False):
        b = s.base_ids(ctx)
        if not s.P: return np.ascontiguousarray(b)
        cl = s.class_ids(ctx, create)
        for p, sl in enumerate(s.slots):
            if sl is None: continue
            r = np.flatnonzero(cl[:, p] >= 0); v = b[r, sl["k"]]; pos = v >= 0; sil = np.zeros(len(r), bool); sil[pos] = ~s.loud[v[pos]]; b[r, sl["k"]] = np.where(sil, -2 - v, v)
        return np.ascontiguousarray(np.hstack([b, cl]))
    def withdrawn(s):
        """The R-cells that are silent: judged at the last finding and found to be foretold by their class, or born since with their X-cell in a class."""
        out = []
        for sl in s.slots:
            if sl is None: continue
            ids_, cod = s.r_cells(sl["k"]); xk, _ = s.split(sl["k"], sl["X"], cod); i = ids_[sl["clsmap"][xk] >= 0]; out.append(i[~s.loud[i]])
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

def check():
    rng = np.random.default_rng(3); S, e = 14, 0.02; ok = True; import math
    rows = rng.integers(0, 4, (200, S)) * (rng.random((200, S)) < 0.3); rows[rows.sum(1) == 0, 0] = 1; other = rng.integers(0, 6, (200, S)) * (rng.random((200, S)) < 0.4); pooled = rows + other
    jc, jo = judge(rows.astype(np.float32), pooled.astype(np.float32), S, e); worst = 0.0
    for i in range(200):
        n = int(rows[i].sum()); N = int(pooled[i].sum()); a = sum(int(rows[i, v]) * (math.log(pooled[i, v] - rows[i, v] + e) - math.log(N - n + S * e)) for v in range(S) if rows[i, v])
        b = sum(int(rows[i, v]) * (math.log(rows[i, v] - 1 + e) - math.log(n - 1 + S * e)) for v in range(S) if rows[i, v]); worst = max(worst, abs(a - jc[i]), abs(b - jo[i]))
    ok &= worst < 1e-9; print(f"the judgment of 200 member cells: numpy against plain Python, largest difference {worst:.1e}", "PASSED" if worst < 1e-9 else "FAILED")
    merge.Sys = Sys3; out, s = merge.run_sums(dict(seed="1", lines="3000", P="8", rep="2")); D = 9
    rs = np.random.default_rng(9); x = rs.integers(0, 14, 2000); x[::10] = merge.END; ctx = merge.ctx_of(x, D, len(x), D); idd = s.ids(ctx); bad = 0; nsil = 0
    for p, sl in enumerate(s.slots):
        if sl is None: continue
        col = idd[:, sl["k"]]; act = idd[:, s.KB + p] >= 0; base = s.base_ids(ctx)[:, sl["k"]]
        for t in np.flatnonzero(act):
            want_silent = not s.loud[base[t]]; nsil += col[t] < -1
            if want_silent and col[t] != -2 - base[t]: bad += 1
    for t in range(len(ctx)):                                                    # a cell is silent only where some class cell stands
        for k in range(s.KB):
            if idd[t, k] < -1 and not any(sl is not None and sl["k"] == k and idd[t, s.KB + p] >= 0 for p, sl in enumerate(s.slots)): bad += 1
    ok &= bad == 0 and nsil > 0; print(f"silent addresses on 1,991 contexts of a small sums system: {int(nsil)} silent, {bad} against the rule", "PASSED" if bad == 0 and nsil > 0 else "FAILED")
    return ok

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    merge.Sys = Sys3; kw = dict(a.split("=") for a in sys.argv[2:]); kw["rep"] = "2"; out, s = dict(sums=merge.run_sums, dev=merge.run_dev, life=merge.run_life)[sys.argv[1]](kw); out["silent_cells_at_findings"] = s.judge_log
    tag = kw["tag"]; json.dump(out, open(os.path.join(HERE, "res", tag + ".json"), "w")); j = s.judge_log[-1] if s.judge_log else {}
    if sys.argv[1] == "sums": print(tag, "| last3", round(out["last3"], 1), "last", out["last"], "place", [round(v, 1) for v in out["place_last"]], "|", " ".join(f"{r['lines'] // 1000}k:{r['add']:.1f}" for r in out["rows"]), "| cells", out["sizes"]["cells"], "not silent", out["sizes"]["cells_with_merged_withdrawn"], "| judged", j.get("judged"), "silent", j.get("silent"), "| cpu", out["cpu"], flush=True)
    else: print(tag, "|", {k: round(v["nats"], 4) for k, v in out["scores"].items()} if "scores" in out else round(out["nats"], 4), "| cells", (out.get("sizes") or out)["cells"], "not silent", (out.get("sizes") or out)["cells_with_merged_withdrawn"], "| judged", j.get("judged"), "silent", j.get("silent"), "| cpu", out["cpu"], flush=True)
