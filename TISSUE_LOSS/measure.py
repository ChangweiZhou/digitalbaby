# Where, between the input lines and the cells four connections away, is the information about the present byte lost, and why.
# Measurements 1 to 5 of the reviewer's brief of 19 November 2026 (message to E57's worker, 05:42 UTC), on one tissue of tissue.c
# (the reviewer's program, copied here; added: recording at steps 4, 8, 16, 32 of a byte and the split of the arrival).
#
#   python3 -I measure.py <label> rule=first|coincidence|fixed [floor=..] [astar=..] [gain=1] [eta_g=..] [nbytes=16384] [seed=..]
#                                 [change=<name>]
#
# Writes out/measure_<label>.json. The tissue is built exactly as drive.py builds it (same birth, stream, options, generator seed).
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'deps', 'TISSUE_STEP1')); sys.path.insert(0, os.path.join(HERE, 'deps', 'TISSUE_EVENT'))
import tissue_step1 as ts  # noqa: E402
import readings  # noqa: E402

label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
rule = {'first': 0, 'coincidence': 1, 'fixed': 2}[kw.get('rule', 'first')]
floor = float(eval(kw.get('floor', '0'))); astar = float(eval(kw.get('astar', '1/8'))); by_use = int(kw.get('by_use', 0))
two = int(kw.get('gain', 0)); eta_g = float(eval(kw.get('eta_g', '1/4096'))); NB = int(kw.get('nbytes', 16384)); SEED = int(kw.get('seed', 73129))
CHANGE = kw.get('change')
SRC = os.path.join(HERE, 'tissue.c')
so = os.path.join(HERE, 'tissue.so')
if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(SRC):
    tmp = so + f'.{os.getpid()}'
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC', SRC, '-o', tmp, '-lm'], check=True); os.replace(tmp, so)
L = ctypes.CDLL(so)
L.tis_set_no_theta(1 if CHANGE == 'no_theta' else 0)
assert CHANGE in (None, 'no_theta')
N = ts.N; B = ts.birth(SEED)
maxd = max(len(m) for m in B['members'].values()) + 2
P = ctypes.c_void_p


class Tis(ctypes.Structure):
    _fields_ = [('maxd', ctypes.c_int), ('rule', ctypes.c_int), ('floor_x', ctypes.c_double), ('astar', ctypes.c_double), ('by_use', ctypes.c_int),
                ('two_sided', ctypes.c_int), ('eta_g', ctypes.c_double), ('gain', P),
                ('kind', P), ('sig', P), ('nmem', P), ('mem', P),
                ('deg', P), ('c_src', P), ('c_mi', P), ('c_w', P), ('c_e0', P), ('c_t0', P), ('c_k0', P), ('c_cu', P), ('c_nb', P), ('issrc', P),
                ('v', P), ('xp', P), ('p', P), ('th', P), ('abar', P), ('vt', P), ('E', P), ('klast', P), ('z', P),
                ('rng', ctypes.c_uint64), ('made', ctypes.c_int64), ('given', ctypes.c_int64), ('given_by_use', ctypes.c_int64), ('k', ctypes.c_int64),
                ('active_sum', ctypes.c_double), ('qpow', ctypes.c_double * 4097)]


def build():
    A = dict(kind=np.ascontiguousarray(B['kind'], np.int8), sig=np.ascontiguousarray(B['sig'], np.float64), nmem=np.zeros(N, np.int32), mem=np.zeros((N, maxd), np.int16),
             deg=np.zeros(N, np.int32), c_src=np.zeros((N, maxd), np.int16), c_mi=np.zeros((N, maxd), np.int16), c_w=np.zeros((N, maxd)), c_e0=np.zeros((N, maxd)),
             c_t0=np.zeros((N, maxd), np.int32), c_k0=np.zeros((N, maxd), np.int32), c_cu=np.zeros((N, maxd)), c_nb=np.zeros((N, maxd), np.int32), issrc=np.zeros((N, maxd), np.uint8),
             v=np.zeros(N), xp=np.zeros(N), p=np.zeros(N), th=np.zeros(N), abar=np.zeros(N), vt=np.full(N, astar), E=np.zeros(N), klast=np.zeros(N, np.int32), z=np.zeros(ts.NZ),
             gain=np.ones(N))
    for i in range(N):
        mem = B['members'][i]; A['nmem'][i] = len(mem); A['mem'][i, :len(mem)] = mem
        pos = {m: q for q, m in enumerate(mem)}
        for d, s in enumerate(B['src'][i]):
            A['c_src'][i, d] = s; A['c_mi'][i, d] = pos[s]; A['c_w'][i, d] = ts.START_W; A['issrc'][i, pos[s]] = 1
        A['deg'][i] = len(B['src'][i])
    t = Tis(); t.maxd = maxd; t.rule = rule; t.floor_x = floor; t.astar = astar; t.by_use = by_use; t.two_sided = two; t.eta_g = eta_g
    t.rng = 88172645463325252 + SEED
    for k, a in A.items():
        setattr(t, k, a.ctypes.data)
    L.tis_init(ctypes.byref(t))
    return t, A


def clone(t, A):
    A2 = {k: a.copy() for k, a in A.items()}
    t2 = Tis()
    ctypes.memmove(ctypes.byref(t2), ctypes.byref(t), ctypes.sizeof(Tis))
    for k, a in A2.items():
        setattr(t2, k, a.ctypes.data)
    return t2, A2


def distances(A, present):
    """drive.py's distances (from the present byte values' lines along present connections), vectorised; 0 = not reached."""
    lines = np.zeros(ts.NZ, bool); lines[[ts.LINE0 + int(v) for v in present]] = True
    deg, src = A['deg'], A['c_src'].astype(np.int64)
    valid = np.arange(maxd)[None, :] < deg[:, None]
    isfb = B['kind'] == ts.FBK
    dist = np.zeros(N, np.int16)
    fr = (~isfb) & (valid & lines[src]).any(1)
    dist[fr] = 1; d = 1
    iscell = valid & (src < N)
    while fr.any():
        frz = np.zeros(ts.NZ, bool); frz[:N] = fr
        nxt = (dist == 0) & (~isfb) & (iscell & frz[src]).any(1)
        d += 1; dist[nxt] = d; fr = nxt
    return dist


L.tis_run.argtypes = [ctypes.POINTER(Tis), P, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, P]
L.tis_run_phases.argtypes = [ctypes.POINTER(Tis), P, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, P, P, P, P]


def run_plain(t, b, lo, hi):
    dummy = np.zeros((1, N), np.float32)
    L.tis_run(ctypes.byref(t), b.ctypes.data, lo, hi, 0, 0, dummy.ctypes.data)


def eta2(X, lab):
    """Per cell: between-byte variance / total variance of X [positions, cells] for labels lab; NaN where the total variance is 0."""
    X = X.astype(np.float64)
    vals, inv = np.unique(lab, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    M = np.zeros((len(vals), X.shape[1]))
    np.add.at(M, inv, X)
    M /= cnt[:, None]
    mu = X.mean(0)
    between = (cnt[:, None] * (M - mu) ** 2).sum(0) / X.shape[0]
    total = X.var(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, between / total, np.nan)


def summarise(v):
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), median=float(np.median(v)) if len(v) else None, upper_decile=float(np.quantile(v, 0.9)) if len(v) else None)


b, _ = ts.stream(SEED, NB); b = np.ascontiguousarray(b, np.uint8)
lo, hi = NB - 4096, NB
t, A = build()
c0 = time.process_time()
# --- to byte 8,192; item 4: two copies from the same state, the second with one cell's potential changed by 1e-9
HALF = 8192
run_plain(t, b, 0, HALF)
dist_half = distances(A, sorted(set(b[HALF - 4096:HALF].tolist())))
xh = A['xp'].copy()
d1 = np.nonzero(dist_half == 1)[0]
cell = int(d1[np.argmax(xh[d1])])                    # the most active cell at d = 1 at byte 8,192


def item4_run(which):
    tA, AA = clone(t, A); tB, AB = clone(t, A)
    if which == 'one':
        AB['v'][cell] += 1e-9
    else:                                            # every cell that is not a feedback cell
        AB['v'][B['kind'] != ts.FBK] += 1e-9
    res = dict(after={})
    done = HALF
    for nb in (1, 4, 16, 64, 256):
        run_plain(tA, b, done, HALF + nb); run_plain(tB, b, done, HALF + nb); done = HALF + nb
        dx = np.abs(AA['xp'] - AB['xp'])
        res['after'][str(nb)] = dict(largest=float(dx.max()), by_distance={str(d): float(dx[dist_half == d].max()) if (dist_half == d).any() else None for d in range(1, 9)},
                                     cells_differing=int((dx > 0).sum()),
                                     state_largest={q: float(np.abs(AA[q] - AB[q]).max()) for q in ('v', 'th', 'p', 'gain', 'abar', 'c_w')},
                                     state_cells_differing={q: int((np.abs(AA[q] - AB[q]) > 0).reshape(N, -1).any(1).sum()) for q in ('v', 'th', 'p', 'gain', 'c_w')})
    return res


item4 = dict(cell=int(cell), cell_distance=int(dist_half[cell]), cell_output_at_8192=float(xh[cell]), one_cell=item4_run('one'),
             every_cell=item4_run('all'))
# --- on to the window A2, recording at steps 4, 8, 16, 32 of each byte, with the split of the reading step's arrival
run_plain(t, b, HALF, lo)
dist = distances(A, sorted(set(b[lo:hi].tolist())))
rec4 = np.zeros((4, hi - lo, N), np.float32)
ab = np.zeros((N, 5)); sg = np.zeros((N, 5))
L.tis_run_phases(ctypes.byref(t), b.ctypes.data, lo, hi, lo, hi, rec4.ctypes.data, dist.ctypes.data, ab.ctypes.data, sg.ctypes.data)
cpu = time.process_time() - c0
X = rec4[3]                                          # the reading step (step 32 of the byte)
lab0 = b[lo:hi].astype(np.int64); lab1 = b[lo - 1:hi - 1].astype(np.int64)
out = dict(label=label, args=kw, cpu_s=cpu, window=[lo, hi - 1], distance_counts={str(d): int((dist == d).sum()) for d in range(0, 12)},
           conns_per_cell=float(A['deg'].sum()) / N, gain_median=float(np.median(A['gain'])), item4=item4)
it1, it2, it3, it5 = {}, {}, {}, {}
e0 = eta2(X, lab0); e1 = eta2(X, lab1)
eph = [eta2(rec4[q], lab0) for q in range(4)]
eph1 = [eta2(rec4[q], lab1) for q in range(4)]
for d in range(1, 9):
    m = dist == d
    if not m.any():
        continue
    it1[str(d)] = dict(cells=int(m.sum()), mean_output=float(X[:, m].mean()), share_above_001=float((X[:, m] > 0.01).mean()))
    it2[str(d)] = dict(present=summarise(e0[m]), previous=summarise(e1[m]))
    it3[str(d)] = {f"step_{s}": dict(present=summarise(eph[q][m]), previous=summarise(eph1[q][m])) for q, s in enumerate((4, 8, 16, 32))}
    if d in (2, 3, 4):
        tot = ab[m].sum(1)
        ok = tot > 0
        shares = ab[m][ok] / tot[ok][:, None]
        names = ("input lines", "nearer", "same distance", "farther", "not reached")
        it5[str(d)] = dict(cells=int(m.sum()), cells_with_arrival=int(ok.sum()),
                           mean_share_of_abs_arrival={n: float(shares[:, q].mean()) for q, n in enumerate(names)},
                           pooled_share_of_abs_arrival={n: float(ab[m][:, q].sum() / ab[m].sum()) for q, n in enumerate(names)},
                           pooled_signed_arrival_per_position={n: float(sg[m][:, q].sum() / m.sum() / (hi - lo)) for q, n in enumerate(names)})
out.update(item1=it1, item2=it2, item3=it3, item5=it5)
# the scores of drive.py, for comparison with the reviewer's figures
x = np.where(X < 0.01, 0.0, X)
r = readings.read_window(x, b[lo:hi], dist, SEED, stream=b, start=lo, lags=(0, 1, 2))
out['A2'] = dict(share={str(d): r['shares'][d]['share'] if r['shares'][d]['n'] else None for d in (1, 2, 4)},
                 S={str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for d in (1, 2, 4)},
                 c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()})
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', f'measure_{label}.json'), 'w'), indent=1, default=float)
f = lambda v: 'nan' if v is None else f'{v:.3f}'
print(label, f"cpu {cpu:.0f}s", 'S0', [f(out['A2']['S'][d]['0']) for d in ('1', '2', '4')], flush=True)
