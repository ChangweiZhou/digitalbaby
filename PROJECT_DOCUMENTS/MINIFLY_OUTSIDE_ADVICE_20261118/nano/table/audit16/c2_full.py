#!/usr/bin/env python3
"""Full-size reproduction of PLAN_16's figures for the system with all kinds of cells, through the audit's memory-light driver (light.py; the author's kernel,
keys and addresses; validated bit for bit against the author's classes on a short text).  One job, under 3 GB.
  0. T2 through the light driver: must give the author's 1.4972798140342958 / 1.4820229952540094
  1. all kinds but the relay system: reported 1.4740423241935867 / 1.4444047536092852
  2. all kinds (relay opinions from c1_relay.py): reported 1.47071069298652 / 1.44206046189165
  for 1 and 2: state digests before/after frozen scoring, own losses, sums to one, the every-8th-byte subset (the bytes of the reference's 1.454), sizes."""
import os, sys, json, hashlib, time, gc, resource
W = os.path.dirname(os.path.abspath(__file__)); sys.dont_write_bytecode = True; sys.path.insert(0, W)
import numpy as np
import light as LT
import score
REP = "/home/user/digitalbaby/PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/table/res/"
def log(*a): print(*a, flush=True)
def sha(a): a = np.ascontiguousarray(a); return hashlib.sha256(memoryview(a).cast("B")).hexdigest()[:16]
def digest(s): d = {k: sha(getattr(s, k)) for k in ("C", "N", "U", "last", "T", "Gt", "half", "ids", "cb", "x")}; d["clock"] = float(s.clock); return d
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
x, cut, S = score.load(); n = len(x); y = x[cut:]; t00 = time.process_time(); ar = np.arange(len(y))
def loss(P, sl=slice(None)): return float(-np.log(P[ar, y])[sl].mean())
def sizes(s):
    nz = 0
    for a in range(0, len(s.C), 500000): nz += int((s.C[a:a + 500000] > 0).sum())
    birth = np.full_like(s.T, 0.25); birth[10:10 + s.born0] = 0.0
    return dict(cells_that_have_counted=int((s.N > 0).sum()), cells_addressed=int(len(s.N)), counts_above_zero=nz, parts_used=int((s.Gt != 0).sum()), parts=int(s.Gt.size), trusts_moved_from_birth=int((s.T != birth).sum()), trusts=int(s.T.size))
def report(name, s, P1, P2, rep):
    for c, P in (("first cell", P1), ("second cell", P2)):
        rs = P.sum(1); log(f"   {name}, {c}: rows {P.shape}; min prob {P.min():.2e}; row sums in [{rs.min():.15f}, {rs.max():.15f}]; smallest p(true byte) {P[ar, y].min():.2e}; "
                           f"own loss over all {len(y)} bytes {loss(P):.16f}; score.score {score.score(P, y)['nats']:.16f}; every 8th byte (13,943 bytes) {loss(P, slice(None, None, 8)):.4f}; the other seven eighths {float(-np.log(P[ar, y])[np.arange(len(y)) % 8 != 0].mean()):.4f}; named right {100 * float((P.argmax(1) == y).mean()):.2f}%")
    if rep: log("   reported:", rep["frozen"], "| equal to the last digit:", loss(P1) == rep["frozen"]["cell1"]["nats"], loss(P2) == rep["frozen"]["cell2"]["nats"], "| differences", loss(P1) - rep["frozen"]["cell1"]["nats"], loss(P2) - rep["frozen"]["cell2"]["nats"])
    l1 = -np.log(P1[ar, y]); l2 = -np.log(P2[ar, y]); d = l1 - l2; log(f"   {name}: gain of the second cell {d.mean():.4f}, standard error over bytes {d.std(ddof=1) / np.sqrt(len(d)):.4f}; per-byte loss sd {l2.std():.3f}; by tenths of the last tenth, second cell: {[round(float(v.mean()), 3) for v in np.array_split(l2, 10)]}; first cell: {[round(float(v.mean()), 3) for v in np.array_split(l1, 10)]}")
    return l1, l2

# ---------- 0. T2 ----------
s = LT.Light(x, ("companions", "relay", "since", "pool"), 0); s.go(6, cut, True); P1, P2 = s.go(cut, n, False, True)
log("0. T2 through the light driver:", loss(P1), loss(P2), "| the author's json:", json.load(open(REP + "gateall_text_gate0_withoutcompanions+relay+since+pool.json"))["frozen"], "| equal:", loss(P1) == 1.4972798140342958, loss(P2) == 1.4820229952540094, "| cpu", round(time.process_time() - t00, 1))
del s, P1, P2; gc.collect()

# ---------- 1. all kinds but the relay system ----------
t0 = time.process_time(); s = LT.Light(x, (), 0); log("built: kinds", s.K1, "| cells addressed", len(s.N), "of which table", s.ntab, "| ids", s.ids.shape, s.ids.dtype, "| cpu", round(time.process_time() - t0, 1), "RSS MB", rss())
def run(name, rep, save=None):
    t0 = time.process_time(); s.go(6, cut, True); sz = sizes(s); before = digest(s); P1, P2 = s.go(cut, n, False, True); after = digest(s)
    log(f"{name}: stream read once (positions 6..{cut - 1}), then frozen scoring of {cut}..{n - 1} | state unchanged by the frozen scoring: {before == after} | clock {s.clock} | cpu {round(time.process_time() - t0, 1)} RSS MB {rss()}")
    log("   size at the moment of frozen scoring:", sz); l1, l2 = report(name, s, P1, P2, rep)
    tag = name.replace(" ", "_"); np.save(os.path.join(W, tag + "_nll.npy"), np.stack([l1, l2]))
    json.dump(dict(T=sha(s.T), Gt=sha(s.Gt), P1_first_55556_rows=sha(P1[:55556]), P2_first_55556_rows=sha(P2[:55556]), P1_row0=sha(P1[:1]), P2_row0=sha(P2[:1]), N_sum=float(s.N.astype(np.float64).sum()), cells=int((s.N > 0).sum()), U_sum=float(s.U.astype(np.float64).sum()),
                   C_sum=float(sum(float(s.C[a:a + 500000].astype(np.float64).sum()) for a in range(0, len(s.C), 500000))), last_sum=float(s.last.sum()), loss1=loss(P1), loss2=loss(P2)), open(os.path.join(W, tag + "_digests.json"), "w"))
    return loss(P1), loss(P2)
run("all but relay", json.load(open(REP + "gateall_text_gate0_withoutrelay.json")), save="allbutrelay"); gc.collect()

# ---------- 2. all kinds ----------
s.C.fill(0); s.N.fill(0); s.U.fill(0); s.last.fill(0); s.T[:] = 0.25; s.T[10:157] = 0.0; s.Gt.fill(0); s.clock = 0.0
s.X = np.load(os.path.join(W, "relay_X.npy"), mmap_mode="r"); s.xlev = np.load(os.path.join(W, "relay_lev.npy")); assert s.X.shape == (n, S) and s.X.dtype == np.float32 and s.xlev.dtype == np.int32
rep = json.load(open(REP + "gateall_text_gate0.json")); f1, f2 = run("all kinds", rep); gc.collect()
def piece_nll(lo, hi):
    p1, p2 = s.go(lo, hi, False, True); yy = x[lo:hi]; q = np.arange(hi - lo); return -np.log(p1[q, yy]), -np.log(p2[q, yy])
# what the frozen clock is worth: the same frozen system, but the time goes on through the last tenth (companions fade as bytes pass), in pieces of 500 bytes
keep = s.clock; o1 = []; o2 = []
for a in range(cut, n, 500):
    s.clock = keep + (a - cut); l1, l2 = piece_nll(a, min(a + 500, n)); o1.append(l1); o2.append(l2)
s.clock = keep; log(f"   sensitivity (not the author's measure): nothing changing but the time going on through the last tenth, so that the companions fade: first cell {np.concatenate(o1).mean():.4f}, second cell {np.concatenate(o2).mean():.4f} (time standing still: {f1:.4f}, {f2:.4f})"); del o1, o2
# companions silenced at the moment of scoring (trusts and parts of the 7 companion kinds set to nothing), for the size of what they carry
kT = s.T.copy(); kG = s.Gt.copy(); s.T[157:164] = 0; s.Gt[:, 157:164] = 0; o = [piece_nll(a, min(a + 20000, n)) for a in range(cut, n, 20000)]; s.T[:] = kT; s.Gt[:] = kG
log(f"   sensitivity: companions silenced at scoring: {np.concatenate([q[0] for q in o]).mean():.4f}, {np.concatenate([q[1] for q in o]).mean():.4f}"); del o, kT, kG
# the relay system silenced at the moment of scoring
kT = s.T.copy(); kG = s.Gt.copy(); s.T[164] = 0; s.Gt[:, 164] = 0; o = [piece_nll(a, min(a + 20000, n)) for a in range(cut, n, 20000)]; s.T[:] = kT; s.Gt[:] = kG
log(f"   sensitivity: the relay system silenced at scoring: {np.concatenate([q[0] for q in o]).mean():.4f}, {np.concatenate([q[1] for q in o]).mean():.4f}"); del o, kT, kG
# the going-on measure and the sizes the program reports (it counts them after this pass)
G1, G2 = s.go(cut, n, True, True); log(f"   going on through the last tenth (a different measure): first cell {loss(G1):.16f}, second cell {loss(G2):.16f} | reported {rep['going_on']} | sizes after it: {sizes(s)} | reported cells {rep['cells']}, parts_used {rep['parts_used']}")
log("total cpu", round(time.process_time() - t00, 1), "peak RSS MB", rss())
