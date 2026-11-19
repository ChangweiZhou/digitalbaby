# The same output-end rules on the recorded inputs of a real tissue. The tissue is the merged program of claude/e63-merged-tissue
# (copied in tissue/, third build: the reviewer's program at aa0a4341, equal to it on seed 73129), run once; its outputs at the reading
# step are recorded for the cells near the output lines: the drivers and every member of a driver's neighbourhood (enlarged by long
# axons), as measure.py's set (a). Output unit v (byte value v) sees its own driver's neighbourhood within that set, the driver
# included, and a constant 1, with measure.py's running standardisation ('run': m += r (x - m), s2 += r ((x - m)^2 - s2),
# r = 1 / min(n, 1024)). Rules A, B, D as measure.py (step 0.25 / P, P = the unit's inputs + 1), and rule Y of yn.py with n = P.
#   python3 -I ynreal.py <label> rules=... [tissue options] seed=.. nbytes=..   -> res/<label>.json
import sys, os, json, time, ctypes
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tissue')); sys.path.insert(0, os.path.join(HERE, 'deps'))
import tis, life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
RULES = [r.split(':') for r in kw.pop('rules').split(',')]
RULES = [(r[0], float(eval(r[1])), float(eval(r[2]))) if r[0] != 'Y' else ('Y', r[1], (float(eval(r[2])), r[3], r[4], float(eval(r[5])))) for r in RULES]
T = tis.make(kw); NB = T['NB']; N = T['N']; ts = tis.ts; B = T['B']; b = T['b'].astype(np.int64); isfb = B['kind'] == ts.FBK
SEED = T['SEED']; L = LS.life(SEED); assert np.array_equal(L['bytes'][:NB], T['b'])
drivers = [int(x) for x in B['driver']]; near = set(drivers)
for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
near = np.array(sorted(i for i in near if not isfb[i])); K = len(near); colk = {c: k for k, c in enumerate(near)}
own = [sorted({colk[c] for c in [i] + [int(m) for m in B['members'][i] if m < N] if c in colk}) for i in drivers]
G_ = max(map(len, own)); IDX = np.full((256, G_), K, np.int64)
for v in range(256): IDX[v, :len(own[v])] = own[v]
P = np.array([len(o) + 1.0 for o in own])
R = np.zeros((NB, K), np.float32)
for lo in range(0, NB, 4096):
    hi = min(NB, lo + 4096); rec = np.zeros((hi - lo, N), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    R[lo:hi] = rec[:, near]
t_tissue = time.time() - t0
C = len(RULES); Wt = np.zeros((C, 256, G_ + 1)); rbar = np.zeros((C, 256)); rglob = np.zeros(C); nemit = np.zeros((C, 256))
rng = np.random.default_rng(SEED); right = np.zeros((C, NB), bool); seen = np.zeros(256, bool)
m1 = np.zeros(K); s1 = np.zeros(K)
for t in range(NB - 1):
    x = R[t].astype(np.float64); r = 1.0 / min(t + 1, 1024); seen[b[t]] = True
    m1 += r * (x - m1); s1 += r * ((x - m1) ** 2 - s1); sd = np.sqrt(s1)
    z = np.where(sd > 0, (x - m1) / np.where(sd > 0, sd, 1), 0.0)
    U = np.ones((256, G_ + 1)); U[:, :-1] = np.append(z, 0.0)[IDX]
    yy = b[t + 1]
    for c, (rule, eps, par) in enumerate(RULES):
        a = (Wt[c] * U).sum(1)
        if rule == 'Y' and eps == 'prop':
            pa = np.maximum(a, 0.0); pa[~seen] = 0.0; tot = pa.sum()
            if tot > 0: e = min(int(np.searchsorted(np.cumsum(pa), rng.random() * tot, side='right')), 255)
            else: sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        else:
            e = int(a.argmax())
        if (rule == 'Y' and eps == 'eps' and rng.random() < 1 / 16) or (rule != 'Y' and eps and rng.random() < eps):
            sv = np.flatnonzero(seen); e = int(sv[rng.integers(0, len(sv))])
        ok = e == yy; right[c, t + 1] = ok
        if rule == 'A':
            Wt[c, e] += (1.0 if ok else -par) * (0.25 / P[e]) * U[e]
        elif rule == 'B':
            Wt[c, e] += (float(ok) - rbar[c, e]) * (0.25 / P[e]) * U[e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
        elif rule == 'Y':
            up, base, decay, s0 = par
            st = s0 * 0.25 / P[e] / (np.sqrt(1.0 + nemit[c, e] / 16.0) if decay == 's' else 1.0); nemit[c, e] += 1
            if base == 'own': g = float(ok) - rbar[c, e]; rbar[c, e] += (float(ok) - rbar[c, e]) / 64
            elif base == 'glob': g = float(ok) - rglob[c]; rglob[c] += (float(ok) - rglob[c]) / 64
            else: g = 1.0 if ok else -1.0 / 8
            Wt[c, e] += g * st * U[e]
            if up and not ok:
                msk = np.ones(256, bool); msk[e] = False
                Wt[c, msk] += (up * s0 * 0.25 / P[msk])[:, None] * U[msk]
        else:
            tg = np.zeros(256); tg[yy] = 1; Wt[c] += (par * (tg - a) / (U * U).sum(1))[:, None] * U
refs = SE.references(L['bytes'][:NB])
wins = [(a_, min(a_ + 4096, NB)) for a_ in range(0, NB, 4096)]
wsh = lambda ok: [float(ok[max(a_, 1):e_].mean()) for a_, e_ in wins]
lv = {k: wsh(refs[k] == b) for k in ('ref_1', 'ref_2', 'ref_3')}
q0 = (3 * NB) // 4; res = {}
for c, (rule, eps, par) in enumerate(RULES):
    nm = (f'Y {eps} up={par[0]:g} base={par[1]} decay={par[2]} s0={par[3]:g}' if rule == 'Y' else None) or f'{rule}' + (f' eps={eps:g}' if rule != 'D' else '') + (f' kappa={par:g}' if rule == 'A' else '') + (f' eta={par:g}' if rule == 'D' else '')
    w = wsh(right[c])
    res[nm] = dict(windows=w, last_window=w[-1], last_quarter=float(right[c, q0:].mean()),
                   first_window_reaching={k: next((wins[i][1] for i in range(len(w)) if w[i] >= l[i]), None) for k, l in lv.items()})
out = dict(label=label, args=kw, seed=SEED, bytes=NB, cells=int(K), inputs_per_unit=dict(min=int(P.min() - 1), median=float(np.median(P) - 1), max=int(P.max() - 1)),
           rules=res, levels=lv, last_quarter_refs={k: float((refs[k] == b)[q0:].mean()) for k in ('ref_1', 'ref_2', 'ref_3')}, windows=wins,
           seconds=dict(tissue=t_tissue, all=time.time() - t0))
os.makedirs(os.path.join(HERE, 'res'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'res', f'{label}.json'), 'w'), indent=0)
print(f"== {label} (seed {SEED}, {NB} bytes, {K} cells, inputs a unit {out['inputs_per_unit']}; refs last quarter " + ' '.join(f"{k} {100 * v:.2f}" for k, v in out['last_quarter_refs'].items()) + f"; {time.time() - t0:.0f} s)")
for k, v in res.items():
    print(f"   {k:44s} last window {100 * v['last_window']:5.2f}  last quarter {100 * v['last_quarter']:5.2f}  first window reaching {v['first_window_reaching']}")
sys.stdout.flush()
