"""Coordinator's read-only measurement of where E63's program spends its time (no file of the repository is written).
Runs the first-form tissue of seed 73129 to byte 1,024 with the worker's program unchanged, then measures 48 bytes three ways."""
import sys, time, copy, hashlib, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import tissue_step1 as M
N, S = M.N, M.S
st, _ = M.stream(73129, 1200)
T = M.Tissue(73129, "first")
t0 = time.process_time()
marks = {}
for t in range(1024):
    T.run_byte(int(st[t]))
    if t + 1 in (64, 256, 512, 1024):
        marks[t + 1] = (time.process_time() - t0, int(T.alive[:T.n].sum()))
print("cpu s to byte, live connections:", marks, flush=True)
def digest(T):
    h = hashlib.sha256()
    for a in (T.snd[:T.n], T.rcv[:T.n], T.w[:T.n], T.e[:T.n], T.alive[:T.n], T.v, T.p, T.th, T.abar, T.klast):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr(T.rl.getstate()).encode())
    return h.hexdigest()[:16]
base = copy.deepcopy(T)
# --- (a) the program as it is, with counts of what is active
A = copy.deepcopy(base)
stats = dict(elig=0, elig_exhausted=0, active_cells=0, conn_active_sender=0, conn_active_receiver_prev=0, conn_both=0, conns=0, steps=0, by_phi={})
def hook(T, k, Av, m, v, x):
    pass
g0, m0 = A.given_up, A.made
c = time.process_time()
for t in range(1024, 1072):
    for _ in range(S):
        n = A.n; al = A.alive[:n]; snd = A.snd[:n]; rcv = A.rcv[:n]
        xp = A.x_prev
        z = A.z.copy(); z[:N] = xp
        if A.k % S == 0:
            z[M.LINE0:M.LINE0 + 256] = 0; z[M.LINE0 + int(st[t])] = 1
        sa = (z[snd] != 0) & al
        ra = (xp[rcv] != 0) & al & (snd < N)
        el = np.nonzero((~A.isfb) & (A.th == 0.0) & (A.abar < M.ASTAR) & ((A.k - A.klast) >= M.PS))[0]
        stats["elig"] += len(el)
        stats["elig_exhausted"] += sum(1 for i in el if len(A.srcset[int(i)]) >= len(A.members[int(i)]))
        stats["active_cells"] += int((xp != 0).sum())
        stats["conn_active_sender"] += int(sa.sum()); stats["conn_active_receiver_prev"] += int(ra.sum())
        stats["conn_both"] += int((sa & ra).sum()); stats["conns"] += int(al.sum()); stats["steps"] += 1
        ph = A.k % S
        if ph in (0, 1, 2, 4, 8, 16, 31):
            stats["by_phi"].setdefault(ph, []).append(int((xp != 0).sum()))
        A.step(int(st[t]))
tc = time.process_time() - c
print("counts over 48 bytes (per step means):", {k: (round(v / stats["steps"], 1) if not isinstance(v, dict) else {p: round(sum(q) / len(q), 1) for p, q in v.items()}) for k, v in stats.items()}, "given up", A.given_up - g0, "made", A.made - m0, flush=True)
# --- (b) the program as it is, timed without the counting
Bt = copy.deepcopy(base); c = time.process_time()
for t in range(1024, 1072): Bt.run_byte(int(st[t]))
tb = time.process_time() - c
# --- (c) the same, with the search for a candidate skipped at cells whose neighbourhood is already full (no other change)
C = copy.deepcopy(base)
full = [i for i in range(N) if len(C.srcset[i]) >= len(C.members[i])]
saved = {i: C.members[i] for i in full}
C.members = dict(C.members)
for i in full: C.members[i] = []
c = time.process_time()
for t in range(1024, 1072): C.run_byte(int(st[t]))
tcc = time.process_time() - c
for i in full: C.members[i] = saved[i]
Bt.members = Bt.members
print(json.dumps(dict(cells_full_at_1024=len(full), cpu_s_per_byte_as_is=tb / 48, cpu_s_per_byte_skip_full=tcc / 48,
                      digest_as_is=digest(Bt), digest_skip_full=digest(C), equal=digest(Bt) == digest(C),
                      given_up_in_window=Bt.given_up - base.given_up, made_as_is=Bt.made - base.made, made_skip=C.made - base.made,
                      numpy=np.__version__, python=sys.version.split()[0]), indent=1), flush=True)
# --- (d) time of the numpy part alone: one step with making and giving up switched off (rule 'fixed' path is not the same; time pieces)
D = copy.deepcopy(base); n = D.n
import timeit
snd, rcv, w, e = D.snd[:n], D.rcv[:n], D.w[:n], D.e[:n]; z = D.z; z[:N] = D.x_prev; zs = z[snd]
def arrival(): return np.bincount(rcv, weights=D.sig[snd] * w * zs, minlength=N)
Av = arrival(); m = Av - D.p
D._index(); ci = D.ci; r = rcv[ci]
def f7(): return M.f8_bound(w[ci] + M.f7_dw(D.sig[snd[ci]], m[r], D.x_prev[r], e[ci]))
def trace(): return M.f6_trace(e, zs)
def giveup(): return D.not_fbline & ((D.k - D.k0[:n]) >= M.PS) & (D.w[:n] < M.GIVE_UP)
def elig_py():
    el = np.nonzero((~D.isfb) & (D.th == 0.0) & (D.abar < M.ASTAR) & ((D.k - D.klast) >= M.PS))[0]
    for i in el:
        i = int(i); ss = D.srcset[i]; cand = [mm for mm in D.members[i] if mm not in ss]
    return len(el)
for name, f in (("gather z[snd]", lambda: z[snd]), ("arrival", arrival), ("formula 7 with bound", f7), ("trace", trace), ("give-up test", giveup), ("candidate search (python)", elig_py)):
    reps = 20; tt = timeit.timeit(f, number=reps) / reps
    print(f"   one step, {name}: {tt * 1e3:.3f} ms  -> {tt * 32:.4f} s a byte", flush=True)
print("eligible cells at this step:", elig_py(), "live connections", int(D.alive[:n].sum()))
