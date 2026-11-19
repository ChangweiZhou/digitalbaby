# The output end alone, learning online from yes/no (roadmap step 2, a first measurement). The tissue is run as in reach.py and is not
# changed; its outputs at the reading step are the inputs of 256 linear units, one for each output line's driver.
#
# usage: yesno.py <label> input=U|M|oh1|oh2 seed=.. nbytes=..
#   U   rule=fixed (no control);   M   rule=fixed gain=2 rho=1 gmax=4;   the inputs of driver v are the set-(a) cells in driver v's own
#       neighbourhood (non-feedback cells within the radius; the driver itself included), each standardised by a running mean and
#       deviation the cell itself could hold: m += r (x - m), s2 += r ((x - m)^2 - s2), r = 1 / min(n, 1024), n the bytes so far
#       (no floor; a deviation of 0 gives an input of 0).
#   oh1 the one-hot present byte (256 inputs, shared by all drivers);  oh2 two one-hots, present and previous byte (512 inputs).
#   Every unit also has a constant input 1 (its threshold). Strengths start at 0.
# Emission: at the reading step of byte t the most active driver emits its byte (ties to the lowest value); every position has an
#   emission (no level of 1/2 for these linear units). Yes if the emitted byte equals the byte at t + 1, no otherwise.
# Rules (one pass, in order; eta_eff = eta / P, P the expected squared length of the unit's input: number of cell inputs + 1, or 2 for
#   oh1, 3 for oh2):
#   R1  after yes the emitting driver's strengths move by +eta_eff * u, after no by -eta_eff * u (u its own input); nothing else changes.
#   R2  as R1, but with probability eps the emitter is a driver chosen at random among those whose byte has occurred in the stream so far.
#   R3  as R2, and after a no every other driver's strengths move by +(eta_eff / 46) * u_j (its own input).
#   D   the supervised delta rule told the right byte: every driver moves by eta * (target - activity) * u / |u|^2 (normalised by the
#       squared length of its own input at that step, which the unit has; target 1 for the byte at t + 1, 0 otherwise); the ceiling of a
#       one-pass linear learner. (With eta_eff = eta / P instead, the rule overflowed on the cell inputs, whose running standardisation
#       gives large values in the first bytes.)
# Output: yesno/<label>.json with, for every rule, the share right in windows of 4,096 life positions, and the references of the scorer
#   (ref_1, ref_2, ref_3 online) and the share of the most frequent byte of each window and of the end mark.
import sys, os, json, time, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps', 'PREP'))
import life_stream as LS, score_emissions as SE
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
inp = kw['input']; SEED = int(kw.get('seed', 73129)); NB = int(kw.get('nbytes', 65536)); W4 = 4096; CH = 4096
t0 = time.time()
L = LS.life(SEED); b = L['bytes'][:NB].astype(np.int64)
ETAS = (1 / 16, 1 / 4, 1.0)
CFG = [('R1', 0.0, e) for e in ETAS] + [('R2', p, e) for p in (1 / 64, 1 / 16) for e in ETAS] + [('R3', p, e) for p in (0.0, 1 / 64, 1 / 16) for e in ETAS] + [('D', 0.0, e) for e in ETAS]
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
    D = 257 if inp == 'oh1' else 513
    mask = np.ones((256, D)); P = np.full(256, 2.0 if inp == 'oh1' else 3.0)
    def chunks():
        for lo in range(0, NB, CH):
            hi = min(NB, lo + CH); U_ = np.zeros((hi - lo, D)); U_[:, -1] = 1
            tt = np.arange(lo, hi); U_[tt - lo, b[tt]] = 1
            if inp == 'oh2':
                q = tt[tt > 0]; U_[q - lo, 256 + b[q - 1]] = 1
            yield lo, U_
    info = dict(inputs=D - 1)
eta = np.array([c[2] for c in CFG])[:, None] / P[None, :]            # C x 256
eps = np.array([c[1] for c in CFG]); r3 = np.array([c[0] == 'R3' for c in CFG]); dl = np.array([c[0] == 'D' for c in CFG])
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
        sg = np.where(ok, 1.0, -1.0)
        nd = ~dl
        Wt[cidx[nd], e[nd]] += (sg[nd] * eta[nd, e[nd]])[:, None] * Um[e[nd]]
        no3 = r3 & ~ok
        if no3.any():
            for c in np.flatnonzero(no3):
                add = (eta[c] / 46)[:, None] * Um; add[e[c]] = 0; Wt[c] += add
        if dl.any():
            nrm2 = (Um * Um).sum(1); nrm2[nrm2 == 0] = 1
            for c in np.flatnonzero(dl):
                tg = np.zeros(256); tg[y] = 1
                Wt[c] += (ETAS[c - C + 3] * (tg - A[c]) / nrm2)[:, None] * Um
refs = SE.references(L['bytes'][:NB])
wins = [(a, min(a + W4, NB)) for a in range(0, NB, W4)]
def wshare(ok):
    return [float(ok[max(a, 1):e].mean()) for a, e in wins]
R = {}
for c, (rule, p, et) in enumerate(CFG):
    R[f'{rule} eps={p:g} eta={et:g}'] = wshare(right[c])
ref = {k: wshare(refs[k] == b) for k in ('ref_1', 'ref_2', 'ref_3')}
ref['most frequent byte of the window'] = [float(np.bincount(b[a:e], minlength=256).max() / (e - a)) for a, e in wins]
ref['end mark (byte 10)'] = [float((b[a:e] == 10).mean()) for a, e in wins]
out = dict(label=label, args=kw, bytes=NB, window=W4, info=info, windows=[[a, e] for a, e in wins], rules=R, references=ref, seconds=time.time() - t0)
os.makedirs(os.path.join(HERE, 'yesno'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'yesno', f'{label}.json'), 'w'), indent=0)
print(f'== {label} ({NB} bytes, {time.time() - t0:.0f} s) {info}; last window share right, %:')
for k, v in list(R.items()) + list(ref.items()):
    print(f'   {k:40s} last {100 * v[-1]:6.2f}   max {100 * max(v):6.2f}   first {100 * v[0]:6.2f}')
sys.stdout.flush()
