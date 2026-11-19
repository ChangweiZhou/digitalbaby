"""Diagnosis of a finished pool (addval.py arch=P, saved with save=1): for every never-shown pair answered wrongly, at the first wrong
answer byte: which pool cells were active; each active cell classed by its behaviour over all 100 pairs at that place in the line:
  sum cell   its firing at that place depends only on a + b (it fires for every pair with its total and no other), or
  table cell it fires for some but not all pairs of a total (it answers to particular pairs);
and the answering cell's evidence (its weight toward the value named minus its weight toward the right value) from the sum cells, the
table cells, and the rest (delay one-hots, constant). Also, for the right value: whether a sum cell for the right total at that place was
active, and its connection to the right value against the strongest table cell's.
    python3 diagpool.py <saved .npz> <seed>
"""
import json, sys
import numpy as np
z = np.load(sys.argv[1]); seed = int(sys.argv[2])
PI, PJ, PK, PV, PT, W, D, base = z["PI"], z["PJ"], z["PK"], z["PV"], z["PT"], z["W"], int(z["D"]), int(z["base"])
held = [tuple(x) for x in z["held"]]; N = len(PI)
SYMS = sorted(set(b"0123456789+=\n")); SI = {v: k for k, v in enumerate(SYMS)}; NS = len(SYMS); NLc = ord("\n"); EQ = ord("=")
line = lambda a, b: list(f"{a}+{b}={a + b}\n".encode())
pairs = [(a, b) for a in range(10) for b in range(10)]


def run(a, b):
    """per answer place k (0 = byte after '='): (delays state, features, pool activity)"""
    d = [NLc] * (D + 1); L = line(a, b); q = L.index(EQ); out = []
    for x in [NLc] + L[:-1]:
        d = [x] + d[:-1]
    d = [NLc] * (D + 1)
    for i, x in enumerate([NLc] + L[:-1]):
        d = [x] + d[:-1]
        if i - 1 >= q:
            dv = np.array(d); act = (dv[PI] + dv[PJ] == PT) & (dv[PK] == PV)
            f = np.zeros(base + N + 1)
            for k in range(D + 1): f[k * NS + SI[d[k]]] = 1.0
            f[base:base + N] = act; f[-1] = 1.0
            out.append((i - 1 - q, act, f, L[i]))
    return out


# activity of every pool cell at each answer place over all pairs
acts = {}
for a, b in pairs:
    for k, act, f, nxt in run(a, b): acts.setdefault(k, {})[(a, b)] = act
cls = {}
for k in acts:
    pk = [p for p in pairs if p in acts[k]]                          # pairs whose line has this answer place
    A = np.array([acts[k][p] for p in pk]); sums = np.array([a + b for a, b in pk])
    c = np.full(N, "silent", dtype=object)
    for j in range(N):
        fire = A[:, j]
        if not fire.any(): continue
        tot = set(sums[fire].tolist())
        c[j] = "sum" if all(fire[sums == s].all() for s in tot) else "table"
    cls[k] = c
res = []
for a, b in held:
    for k, act, f, nxt in run(a, b):
        sc = W @ f; nm = int(sc.argmax()); right = SI[nxt]
        if nm == right: continue
        c = cls[k]; idx = np.flatnonzero(act)
        diff = W[nm] - W[right]                                      # evidence for the wrong value over the right one
        ev = {t: float((diff[base + idx[c[idx] == t]]).sum()) for t in ("sum", "table")}
        ev["rest"] = float(diff[:base].dot(f[:base]) + diff[-1])
        sum_right = [j for j in idx if c[j] == "sum"]
        res.append(dict(pair=[int(a), int(b)], place=k, named=chr(SYMS[nm]), right=chr(nxt), active=dict(sum=int((c[idx] == "sum").sum()), table=int((c[idx] == "table").sum())),
                        evidence_wrong_minus_right=ev,
                        sum_cells_to_right=float(W[right, base + np.array(sum_right)].sum()) if sum_right else 0.0,
                        table_cells_to_named=float(W[nm, base + idx[c[idx] == "table"]].sum()) if (c[idx] == "table").any() else 0.0))
        break
kept = {t: int(sum((cls[k] == t).sum() for k in cls)) for t in ("sum", "table", "silent")}
strength = np.linalg.norm(W[:, base:base + N], axis=0)
mean_str = {t: float(np.mean([strength[j] for k in cls for j in np.flatnonzero(cls[k] == t)])) for t in ("sum", "table")}
summary = dict(wrong=len(res), of=len(held),
               mean_evidence={t: float(np.mean([r["evidence_wrong_minus_right"][t] for r in res])) for t in ("sum", "table", "rest")} if res else None,
               table_evidence_largest=sum(1 for r in res if max(r["evidence_wrong_minus_right"], key=r["evidence_wrong_minus_right"].get) == "table"),
               sum_evidence_largest=sum(1 for r in res if max(r["evidence_wrong_minus_right"], key=r["evidence_wrong_minus_right"].get) == "sum"),
               rest_evidence_largest=sum(1 for r in res if max(r["evidence_wrong_minus_right"], key=r["evidence_wrong_minus_right"].get) == "rest"),
               no_sum_cell_active=sum(1 for r in res if r["active"]["sum"] == 0),
               cell_classes_by_place_summed=kept, mean_connection_strength=mean_str)
json.dump(dict(summary=summary, cases=res), open(sys.argv[1].replace(".npz", "_diag.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
for r in res[:8]: print(r)
