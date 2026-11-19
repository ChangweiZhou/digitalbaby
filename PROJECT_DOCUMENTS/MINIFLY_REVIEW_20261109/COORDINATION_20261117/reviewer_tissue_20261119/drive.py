# Driver of the reviewer's tissue.c: one tissue, seed 73129, 16,384 bytes, the two windows of stage A, the readings of the helper's tool.
# usage: drive.py <label> rule=first|coincidence|fixed [floor=..] [astar=..] [by_use=1] [gain=1] [eta_g=..] [nbytes=..]
import sys, os, time, json, ctypes, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FAST = os.environ.get('FASTDIR') or os.path.join(HERE, 'deps')
sys.path.insert(0, os.path.join(FAST, 'TISSUE_STEP1')); sys.path.insert(0, os.path.join(FAST, 'TISSUE_EVENT'))
import tissue_step1 as ts, readings
label = sys.argv[1]; kw = dict(a.split('=') for a in sys.argv[2:])
rule = {'first': 0, 'coincidence': 1, 'fixed': 2, 'timing': 3}[kw.get('rule', 'first')]
floor = float(eval(kw.get('floor', '0'))); astar = float(eval(kw.get('astar', '1/8'))); by_use = int(kw.get('by_use', 0))
two = int(kw.get('gain', 0)); eta_g = float(eval(kw.get('eta_g', '1/4096'))); NB = int(kw.get('nbytes', 16384)); SEED = int(kw.get('seed', 73129))
so = os.path.join(HERE, 'tissue.so')
if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(os.path.join(HERE, 'tissue.c')):
    tmp = so + f'.{os.getpid()}'
    subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC', os.path.join(HERE, 'tissue.c'), '-o', tmp, '-lm'], check=True); os.replace(tmp, so)
L = ctypes.CDLL(so)
N = ts.N; B = ts.birth(SEED)
maxd = max(len(m) for m in B['members'].values()) + 2
P = ctypes.c_void_p
class Tis(ctypes.Structure):
    _fields_ = [('maxd', ctypes.c_int), ('rule', ctypes.c_int), ('floor_x', ctypes.c_double), ('astar', ctypes.c_double), ('by_use', ctypes.c_int),
                ('two_sided', ctypes.c_int), ('eta_g', ctypes.c_double), ('gain', P), ('rho', ctypes.c_double), ('ibar', P), ('gmax', ctypes.c_double), ('F', P), ('tau_f', ctypes.c_double), ('eta_s', ctypes.c_double), ('norm', ctypes.c_double), ('we', ctypes.c_double), ('wi', ctypes.c_double), ('wcap', ctypes.c_double), ('pk', P), ('inh_learn', ctypes.c_double), ('rate_norm', ctypes.c_double), ('M', P), ('a_make', ctypes.c_double), ('start_w', ctypes.c_double), ('make_period', ctypes.c_double), ('no_make', ctypes.c_double), ('lines_learn', ctypes.c_double),
                ('kind', P), ('sig', P), ('nmem', P), ('mem', P),
                ('deg', P), ('c_src', P), ('c_mi', P), ('c_w', P), ('c_e0', P), ('c_t0', P), ('c_k0', P), ('c_cu', P), ('c_nb', P), ('issrc', P),
                ('v', P), ('xp', P), ('p', P), ('th', P), ('abar', P), ('vt', P), ('E', P), ('klast', P), ('z', P),
                ('rng', ctypes.c_uint64), ('made', ctypes.c_int64), ('given', ctypes.c_int64), ('given_by_use', ctypes.c_int64), ('k', ctypes.c_int64),
                ('active_sum', ctypes.c_double), ('qpow', ctypes.c_double * 4097)]
A = dict(kind=np.ascontiguousarray(B['kind'], np.int8), sig=np.ascontiguousarray(B['sig'], np.float64), nmem=np.zeros(N, np.int32), mem=np.zeros((N, maxd), np.int16),
         deg=np.zeros(N, np.int32), c_src=np.zeros((N, maxd), np.int16), c_mi=np.zeros((N, maxd), np.int16), c_w=np.zeros((N, maxd)), c_e0=np.zeros((N, maxd)),
         c_t0=np.zeros((N, maxd), np.int32), c_k0=np.zeros((N, maxd), np.int32), c_cu=np.zeros((N, maxd)), c_nb=np.zeros((N, maxd), np.int32), issrc=np.zeros((N, maxd), np.uint8),
         v=np.zeros(N), xp=np.zeros(N), p=np.zeros(N), th=np.zeros(N), abar=np.zeros(N), vt=np.full(N, astar), E=np.zeros(N), klast=np.zeros(N, np.int32), z=np.zeros(ts.NZ),
         gain=np.ones(N), ibar=np.zeros(N), F=np.zeros(ts.NZ), M=np.full(ts.NZ, 1 / 64), pk=np.zeros(N))
for i in range(N):
    mem = B['members'][i]; A['nmem'][i] = len(mem); A['mem'][i, :len(mem)] = mem
    pos = {m: q for q, m in enumerate(mem)}
    for d, s in enumerate(B['src'][i]):
        A['c_src'][i, d] = s; A['c_mi'][i, d] = pos[s]; A['c_w'][i, d] = ts.START_W; A['issrc'][i, pos[s]] = 1
    A['deg'][i] = len(B['src'][i])
t = Tis(); t.maxd = maxd; t.rule = rule; t.floor_x = floor; t.astar = astar; t.by_use = by_use; t.two_sided = two; t.eta_g = eta_g; t.rho = float(eval(kw.get('rho', '7/8'))); t.gmax = float(eval(kw.get('gmax', '4096'))); t.tau_f = float(eval(kw.get('tau_f', '4'))); t.eta_s = float(eval(kw.get('eta_s', '1'))); t.no_make = float(kw.get('no_make', 0)); t.lines_learn = float(kw.get('lines_learn', 0)); t.norm = float(kw.get('norm', 0)); t.a_make = float(eval(kw.get('a_make', '0'))); t.rate_norm = float(kw.get('rate_norm', 0)); t.inh_learn = float(kw.get('inh_learn', 0)); t.wcap = float(eval(kw.get('wcap', '0'))); t.start_w = float(eval(kw.get('start_w', '1/8'))); t.make_period = float(eval(kw.get('make_period', '128'))); t.we = float(eval(kw.get('we', '3/4'))); t.wi = float(eval(kw.get('wi', '1/4')))
t.rng = 88172645463325252 + SEED
for k, a in A.items(): setattr(t, k, a.ctypes.data)
L.tis_init(ctypes.byref(t))
b, _ = ts.stream(SEED, NB); b = np.ascontiguousarray(b, np.uint8)
WIN = {'A1': (2048, 6144), 'A2': (NB - 4096, NB)}
def distances(present):
    lines = set(ts.LINE0 + int(v) for v in present)
    deg = A['deg']; src = A['c_src']
    dist = np.zeros(N, np.int16); isfb = B['kind'] == ts.FBK
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)])
    dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist
out = dict(label=label, args=kw); pos = 0; t0 = time.process_time()
recs = {}; dists = {}
for w, (lo, hi) in WIN.items():
    rec = np.zeros((hi - lo, N), np.float32)
    L.tis_run(ctypes.byref(t), ctypes.c_void_p(b.ctypes.data), ctypes.c_int64(pos), ctypes.c_int64(hi), ctypes.c_int64(lo), ctypes.c_int64(hi), ctypes.c_void_p(rec.ctypes.data))
    recs[w] = rec; dists[w] = distances(sorted(set(b[lo:hi].tolist()))); pos = hi
cpu = time.process_time() - t0
nf = int((B['kind'] != ts.FBK).sum())
out.update(ms_per_byte=cpu / NB * 1e3, conns_per_cell=float(A['deg'].sum()) / N, made=int(t.made), given=int(t.given), given_by_use=int(t.given_by_use),
           active_per_step=t.active_sum / t.k, gain_median=float(np.median(A['gain'])), gain_p90=float(np.percentile(A['gain'], 90)), gain_at_max=int((A['gain'] >= t.gmax).sum()), th_zero=int((A['th'] == 0).sum()), abar_mean=float(A['abar'].mean()))
for w, (lo, hi) in WIN.items():
    x = np.where(recs[w] < 0.01, 0.0, recs[w])
    r = readings.read_window(x, b[lo:hi], dists[w], SEED, stream=b, start=lo, lags=(0, 1, 2))
    out[w] = dict(share={str(d): r['shares'][d]['share'] if r['shares'][d]['n'] else None for d in (1, 2, 4)}, n={str(d): r['shares'][d]['n'] for d in (1, 2, 4)},
                  S={str(d): {str(Lg): (r['scores'][d]['by_lag'][Lg] or {}).get('S') for Lg in (0, 1, 2)} for d in (1, 2, 4)},
                  c0={str(k): (v.get('S') if isinstance(v, dict) else v) for k, v in r['c0'].items()})
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
json.dump(out, open(os.path.join(HERE, 'out', label + '.json'), 'w'), default=float)
a2 = out['A2']; f = lambda v: 'nan' if v is None else f'{v:.3f}'
print(f"{label:26s} {out['ms_per_byte']:6.2f} ms/byte conns {out['conns_per_cell']:5.1f} active/step {out['active_per_step']:6.0f} gain {out['gain_median']:.2f} | share d1 d2 d4 "
      f"{f(a2['share']['1'])} {f(a2['share']['2'])} {f(a2['share']['4'])} | S0 {f(a2['S']['1']['0'])} {f(a2['S']['2']['0'])} {f(a2['S']['4']['0'])} | S1 {f(a2['S']['1']['1'])} {f(a2['S']['2']['1'])} {f(a2['S']['4']['1'])} "
      f"| S2 {f(a2['S']['1']['2'])} {f(a2['S']['2']['2'])} {f(a2['S']['4']['2'])} | c0 {f(a2['c0'].get('1'))} {f(a2['c0'].get('2'))}", flush=True)
