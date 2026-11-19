# Reviewer, 19 November 2026. The output end's rules on the activity of a real tissue, to see whether the yes-or-no rule Q of
# yesno/yq.py holds outside exact inputs. The tissue lives its whole life once (no output end inside it; the output end reads and
# does not act back); the outputs of all cells at every byte are kept in memory; then the rules learn online, one pass, to name the
# NEXT byte from the cells' outputs at the present byte. One unit a byte value.
# Two sets of inputs:   pool : every unit sees all cells near the output lines (the drivers and the members of their neighbourhoods);
#                       own  : unit v sees only the members of the neighbourhood of the v-th output-line cell (what one driver reaches).
# Inputs: a cell's output minus its running mean, over its running deviation (both at rate 1/1024, updated after use), clipped to
# [-8, 8]; a constant 1 appended; n = |U|^2. Rules (as yq.py): D told the right byte, every unit W[v] += eta (target - a_v) U / n;
# B the old yes-or-no rule, W[e] += (ok - rbar_e) (1/4) U / n; Q: the emitter W[e] += eta (ok - a_e) U / n, and with z1, on yes,
# the others W[v] += eta (0 - a_v) U / n. Emission of B and Q: the largest, with chance 1/16 a seen value at random.
# usage: outq.py <label> rule=.. [tissue options] nbytes=65536   -> outq/<label>.json
import sys, os, json, ctypes, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import tis
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yesno', 'deps')); import score_emissions as SE
ts = tis.ts
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:]); t0 = time.time()
T = tis.make(kw); t, L, b, N, NB = T['t'], T['L'], T['b'], T['N'], T['NB']
rec = np.zeros((NB, N), np.float32)
L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_void_p(rec.ctypes.data))
print(f"== {label}: life of {NB} bytes done in {time.time() - t0:.0f} s", flush=True)
B = T['B']; kind = np.asarray(B['kind']); isfb = kind == ts.FBK; outl = np.where(kind == ts.OUTE)[0]
near = np.zeros(N, bool)
for i in outl:
    near[i] = True
    for m in B['members'][int(i)]:
        if m < N: near[m] = True
near &= ~isfb; pool = np.where(near)[0]
own = [np.array([m for m in B['members'][int(i)] if m < N and not isfb[m]], int) for i in outl]
M = max(len(o) for o in own); idx = np.zeros((len(outl), M), int); msk = np.zeros((len(outl), M))
for v, o in enumerate(own): idx[v, :len(o)] = o; msk[v, :len(o)] = 1.0
bb = b.astype(np.int64); V = 256
assert len(outl) == V
RULES = [('D', 1 / 64, 1), ('B', 0.25, 0), ('Q', 1 / 4, 1), ('Q', 1 / 16, 1), ('Q', 1 / 64, 1), ('Q', 1 / 16, 0)]     # the first three lives (L8_128, 65,536 bytes) also ran D 1/16 and Q 1/256
names = ['D eta=1/64', 'B', 'Q 1/4 z1', 'Q 1/16 z1', 'Q 1/64 z1', 'Q 1/16 z0']
out = dict(label=label, args=kw, bytes=NB, sets={})
refs = SE.references(b[:NB]); q0 = 3 * NB // 4
out['last_quarter_refs'] = {k: float((refs[k] == bb)[q0:].mean()) for k in ('ref_1', 'ref_2', 'ref_3')}
print("   refs last quarter " + ' '.join(f"{k} {100 * v:.2f}" for k, v in out['last_quarter_refs'].items()), flush=True)
for sname in ('pool', 'own'):
    C = len(RULES); rng = np.random.default_rng(T['SEED']); seen = np.zeros(V, bool); right = np.zeros((C, NB), bool); rbar = np.zeros(V)
    mean = np.zeros(N); var = np.ones(N) * 1e-4
    if sname == 'pool': W = np.zeros((C, V, len(pool) + 1))
    else: W = np.zeros((C, V, M + 1))
    for tt in range(NB - 1):
        r = rec[tt].astype(np.float64); x = np.clip((r - mean) / np.sqrt(var + 1e-12), -8, 8); x[var < 1e-9] = 0.0
        mean += (r - mean) / 1024; var += ((r - mean) ** 2 - var) / 1024
        seen[bb[tt]] = True; yy = bb[tt + 1]; sv = np.flatnonzero(seen)
        if sname == 'pool':
            U = np.append(x[pool], 1.0); n = float(U @ U)
        else:
            U = np.hstack([x[idx] * msk, np.ones((V, 1))]); n = (U * U).sum(1)
        for c, (rule, eta, z) in enumerate(RULES):
            a = W[c] @ U if sname == 'pool' else (W[c] * U).sum(1)
            e = int(a.argmax())
            if rule != 'D' and rng.random() < 1 / 16: e = int(sv[rng.integers(0, len(sv))])
            ok = e == yy; right[c, tt + 1] = ok
            if rule == 'D':
                d = -a; d[yy] += 1.0; d *= eta
            elif rule == 'B':
                d = np.zeros(V); d[e] = (float(ok) - rbar[e]) * eta; rbar[e] += (float(ok) - rbar[e]) / 64
            else:
                d = np.zeros(V); d[e] = eta * (float(ok) - a[e])
                if ok and z: d = -eta * a; d[e] = eta * (1.0 - a[e])
            if sname == 'pool':
                nz = np.flatnonzero(d)
                if len(nz): W[c][nz] += np.outer(d[nz] / n, U)
            else: W[c] += (d / n)[:, None] * U
    res = {nm: dict(last_quarter=float(right[c, q0:].mean()), quarters=[float(right[c, i * NB // 4:(i + 1) * NB // 4].mean()) for i in range(4)], max_abs_weight=float(np.abs(W[c]).max())) for c, nm in enumerate(names)}
    out['sets'][sname] = dict(inputs=(len(pool) if sname == 'pool' else [int(msk.sum(1).min()), int(msk.sum(1).max())]), rules=res)
    print(f"   inputs '{sname}' ({out['sets'][sname]['inputs']} cells; {time.time() - t0:.0f} s)")
    for k, v in res.items(): print(f"      {k:12s} last quarter {100 * v['last_quarter']:5.2f}   quarters " + ' '.join(f"{100 * q:5.2f}" for q in v['quarters']) + f"   largest weight {v['max_abs_weight']:.2f}")
    sys.stdout.flush()
    os.makedirs('outq', exist_ok=True); json.dump(out, open(f'outq/{label}.json', 'w'))
