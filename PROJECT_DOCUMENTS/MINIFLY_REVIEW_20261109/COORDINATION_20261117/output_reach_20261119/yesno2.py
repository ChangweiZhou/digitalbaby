# Follow-up of yesno.py (same inputs, emission, windows and references; read that file's header). Two further rules from yes/no that use
# only what one driver has, because R1 to R3 stayed below 1% in yesno.py: with steps of +eta after yes and -eta after no an emitter's
# expected change in a context is eta (2p - 1), p the share of that context in which its byte comes next, so it gains only where p > 1/2.
#   A   after yes the emitter moves by +eta_eff * u, after no by -kappa * eta_eff * u (kappa 1/46 or 1/8); with probability eps the
#       emitter is a driver chosen at random among the bytes seen so far.
#   B   the emitter moves by eta_eff * (r - rbar_e) * u, r = 1 after yes and 0 after no, rbar_e the emitter's own running share of yes
#       over its emissions (rate 1/64, starting at 0); eps as in A.
# With strengths starting at 0, eta only scales the strengths in A and B, so one eta (1/4) is run. Input oh3 (three one-hots: the present
# byte and the two before) is added.
# usage: yesno2.py <label> input=U|M|oh1|oh2|oh3 seed=.. nbytes=..   -> yesno/<label>.json
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
inp = kw['input']; SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); W4 = 4096; CH = 4096
t0 = time.time()
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64)
CFG = [('A', p, k) for k in (1 / 46, 1 / 8) for p in (0.0, 1 / 16)] + [('B', p, 0.0) for p in (0.0, 1 / 16)]
C = len(CFG)
if inp in ('U', 'M'):
    import tis
    tkw = dict(rule='fixed', seed=str(SEED), nbytes=str(NB))
    if inp == 'M': tkw.update(gain='2', rho='1', gmax='4')
    T = tis.make(tkw); ts = tis.ts; B = T['B']; N = T['N']; isfb = B['kind'] == ts.FBK
    assert np.array_equal(T['b'], L['bytes'][:NB])
    drivers = list(B['driver'])
    near = set(drivers)
    for i in drivers: near.update(int(m) for m in B['members'][i] if m < N)
    near = np.array(sorted(i for i in near if not isfb[i])); col = {c: k for k, c in enumerate(near)}
    D = len(near) + 1
    mask = np.zeros((256, D)); mask[:, -1] = 1
    for v, i in enumerate(drivers):
        for c in [i] + [int(m) for m in B['members'][i] if m < N]:
            if c in col: mask[v, col[c]] = 1
    P = mask.sum(1)                                                  # inputs of each unit, the constant included
    m_run = np.zeros(len(near)); s2_run = np.zeros(len(near))
    def chunks():
        global m_run, s2_run
        n = 0
        for lo in range(0, NB, CH):
            hi = min(NB, lo + CH); rec = np.zeros((hi - lo, N), np.float32)
            T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
            X = rec[:, near].astype(np.float64); U_ = np.empty((hi - lo, D)); U_[:, -1] = 1
            for k in range(hi - lo):
                n += 1; r = 1.0 / min(n, 1024); x = X[k]
                m_run += r * (x - m_run); s2_run += r * ((x - m_run) ** 2 - s2_run)
                sd = np.sqrt(s2_run); U_[k, :-1] = np.where(sd > 0, (x - m_run) / np.where(sd > 0, sd, 1), 0.0)
            yield lo, U_
    info = dict(cells=int(len(near)), inputs_per_driver=dict(min=int(P.min()) - 1, median=float(np.median(P)) - 1, max=int(P.max()) - 1))
else:
    nh = {'oh1': 1, 'oh2': 2, 'oh3': 3}[inp]; D = 256 * nh + 1
    mask = np.ones((256, D)); P = np.full(256, nh + 1.0)
    def chunks():
        for lo in range(0, NB, CH):
            hi = min(NB, lo + CH); U_ = np.zeros((hi - lo, D)); U_[:, -1] = 1
            tt = np.arange(lo, hi); U_[tt - lo, b[tt]] = 1
            for h in range(1, nh):
                q = tt[tt >= h]; U_[q - lo, 256 * h + b[q - h]] = 1
            yield lo, U_
    info = dict(inputs=D - 1)
eta = np.full((C, 256), 0.25) / P[None, :]
eps = np.array([c[1] for c in CFG]); kap = np.array([c[2] for c in CFG]); isB = np.array([c[0] == 'B' for c in CFG]); rbar = np.zeros((C, 256))
Wt = np.zeros((C, 256, D)); rng = np.random.default_rng(SEED)
right = np.zeros((C, NB), bool)                                       # right[c, t + 1]: the emission at the reading step of t was right
seen = np.zeros(256, bool); cidx = np.arange(C)
for lo, U_ in chunks():
    for k in range(U_.shape[0]):
        t = lo + k; seen[b[t]] = True
        if t + 1 >= NB: break
        u = U_[k]; Um = mask * u                                      # 256 x D: each driver's own input
        A = Wt @ u                                                    # C x 256
        e = A.argmax(1)
        rnd = rng.random(C) < eps
        if rnd.any():
            sv = np.flatnonzero(seen); e[rnd] = sv[rng.integers(0, len(sv), rnd.sum())]
        y = b[t + 1]; ok = e == y; right[:, t + 1] = ok
        r = ok.astype(float)
        step = np.where(isB, r - rbar[cidx, e], np.where(ok, 1.0, -kap))
        Wt[cidx, e] += (step * eta[cidx, e])[:, None] * Um[e]
        rbar[cidx, e] += (r - rbar[cidx, e]) / 64
refs = SE.references(L['bytes'][:NB])
wins = [(a, min(a + W4, NB)) for a in range(0, NB, W4)]
def wshare(ok):
    return [float(ok[max(a, 1):e].mean()) for a, e in wins]
R = {}
for c, (rule, p, k) in enumerate(CFG):
    R[f'{rule} eps={p:g}' + (f' kappa=1/{round(1 / k)}' if rule == 'A' else '')] = wshare(right[c])
ref = {k: wshare(refs[k] == b) for k in ('ref_1', 'ref_2', 'ref_3')}
ref['most frequent byte of the window'] = [float(np.bincount(b[a:e], minlength=256).max() / (e - a)) for a, e in wins]
ref['end mark (byte 10)'] = [float((b[a:e] == 10).mean()) for a, e in wins]
out = dict(label=label, args=kw, bytes=NB, window=W4, info=info, windows=[[a, e] for a, e in wins], rules=R, references=ref, seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'yesno'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'yesno', f'{label}.json'), 'w'), indent=0)
print(f'== {label} ({NB} bytes, {time.time() - t0:.0f} s) {info}; last window share right, %:')
for k, v in list(R.items()) + list(ref.items()):
    print(f'   {k:40s} last {100 * v[-1]:6.2f}   max {100 * max(v):6.2f}   first {100 * v[0]:6.2f}')
sys.stdout.flush()
