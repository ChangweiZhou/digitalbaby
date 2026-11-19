# tis_tau.py: tis.py with the time constant of each cell's potential (tissue_tau.c); option tau=2|q16|u2-16|u2-64|u2-64e.
# Set-up of the reviewer's tissue.c as a module (make(kw) gives one tissue). From drive.py: one tissue, seed 73129, 16,384 bytes, the two windows of stage A, the readings of the helper's tool.
# usage: drive.py <label> rule=first|coincidence|fixed [floor=..] [astar=..] [by_use=1] [gain=1] [eta_g=..] [nbytes=..]
import sys, os, time, json, ctypes, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FAST = os.environ.get('FASTDIR') or os.path.join(HERE, 'deps')
sys.path.insert(0, os.path.join(FAST, 'TISSUE_STEP1')); sys.path.insert(0, os.path.join(FAST, 'TISSUE_EVENT'))
import tissue_step1 as ts, readings

def tau_of_cells(spec, seed, B):
    """The time constant of every cell's potential, drawn at birth from a generator seeded by the seed (Random(seed + 700000)), one
    draw per cell in cell order, apart from the tissue's connection generator. spec: '2' (every cell 2); 'q16' (16 for a random quarter
    of the cells, Random.sample of N // 4, 2 for the rest); 'u2-16' (from 2, 4, 8, 16 with equal chance); 'u2-64' (from 2, 4, 8, 16,
    32, 64); 'u2-256' (from 2, 4, ..., 256; the follow-on run); 'u2-64e' (as u2-64, but a cell whose neighbourhood holds an input line keeps 2)."""
    from random import Random
    N = ts.N; r = Random(seed + 700000); tau = np.full(N, 2.0)
    if spec == '2':
        return tau
    if spec == 'q16':
        tau[r.sample(range(N), N // 4)] = 16.0
        return tau
    choices = {'u2-16': (2, 4, 8, 16), 'u2-64': (2, 4, 8, 16, 32, 64), 'u2-64e': (2, 4, 8, 16, 32, 64), 'u2-256': (2, 4, 8, 16, 32, 64, 128, 256)}[spec]
    for i in range(N):
        tau[i] = float(choices[r.randrange(len(choices))])
    if spec == 'u2-64e':
        for i in range(N):
            if any(ts.LINE0 <= m < ts.LINE0 + ts.NIN for m in B['members'][i]):
                tau[i] = 2.0
    return tau


def make(kw):
    rule = {'first': 0, 'coincidence': 1, 'fixed': 2, 'timing': 3}[kw.get('rule', 'first')]
    floor = float(eval(kw.get('floor', '0'))); astar = float(eval(kw.get('astar', '1/8'))); by_use = int(kw.get('by_use', 0))
    two = int(kw.get('gain', 0)); eta_g = float(eval(kw.get('eta_g', '1/4096'))); NB = int(kw.get('nbytes', 16384)); SEED = int(kw.get('seed', 73129))
    so = os.path.join(HERE, 'tissue_tau.so')
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(os.path.join(HERE, 'tissue_tau.c')):
        tmp = so + f'.{os.getpid()}'
        subprocess.run(['gcc', '-O3', '-march=native', '-shared', '-fPIC', os.path.join(HERE, 'tissue_tau.c'), '-o', tmp, '-lm'], check=True); os.replace(tmp, so)
    L = ctypes.CDLL(so)
    N = ts.N; B = ts.birth(SEED)
    maxd = max(len(m) for m in B['members'].values()) + 2
    P = ctypes.c_void_p
    class Tis(ctypes.Structure):
        _fields_ = [('maxd', ctypes.c_int), ('rule', ctypes.c_int), ('floor_x', ctypes.c_double), ('astar', ctypes.c_double), ('by_use', ctypes.c_int),
                    ('two_sided', ctypes.c_int), ('eta_g', ctypes.c_double), ('gain', P), ('rho', ctypes.c_double), ('ibar', P), ('gmax', ctypes.c_double), ('F', P), ('tau_f', ctypes.c_double), ('eta_s', ctypes.c_double),
                    ('kind', P), ('sig', P), ('nmem', P), ('mem', P),
                    ('deg', P), ('c_src', P), ('c_mi', P), ('c_w', P), ('c_e0', P), ('c_t0', P), ('c_k0', P), ('c_cu', P), ('c_nb', P), ('issrc', P),
                    ('v', P), ('xp', P), ('p', P), ('th', P), ('abar', P), ('vt', P), ('E', P), ('klast', P), ('z', P),
                    ('rng', ctypes.c_uint64), ('made', ctypes.c_int64), ('given', ctypes.c_int64), ('given_by_use', ctypes.c_int64), ('k', ctypes.c_int64),
                    ('active_sum', ctypes.c_double), ('qpow', ctypes.c_double * 4097), ('tauv', P)]
    A = dict(kind=np.ascontiguousarray(B['kind'], np.int8), sig=np.ascontiguousarray(B['sig'], np.float64), nmem=np.zeros(N, np.int32), mem=np.zeros((N, maxd), np.int16),
             deg=np.zeros(N, np.int32), c_src=np.zeros((N, maxd), np.int16), c_mi=np.zeros((N, maxd), np.int16), c_w=np.zeros((N, maxd)), c_e0=np.zeros((N, maxd)),
             c_t0=np.zeros((N, maxd), np.int32), c_k0=np.zeros((N, maxd), np.int32), c_cu=np.zeros((N, maxd)), c_nb=np.zeros((N, maxd), np.int32), issrc=np.zeros((N, maxd), np.uint8),
             v=np.zeros(N), xp=np.zeros(N), p=np.zeros(N), th=np.zeros(N), abar=np.zeros(N), vt=np.full(N, astar), E=np.zeros(N), klast=np.zeros(N, np.int32), z=np.zeros(ts.NZ),
             gain=np.ones(N), ibar=np.zeros(N), F=np.zeros(N))
    for i in range(N):
        mem = B['members'][i]; A['nmem'][i] = len(mem); A['mem'][i, :len(mem)] = mem
        pos = {m: q for q, m in enumerate(mem)}
        for d, s in enumerate(B['src'][i]):
            A['c_src'][i, d] = s; A['c_mi'][i, d] = pos[s]; A['c_w'][i, d] = ts.START_W; A['issrc'][i, pos[s]] = 1
        A['deg'][i] = len(B['src'][i])
    t = Tis(); t.maxd = maxd; t.rule = rule; t.floor_x = floor; t.astar = astar; t.by_use = by_use; t.two_sided = two; t.eta_g = eta_g; t.rho = float(eval(kw.get('rho', '7/8'))); t.gmax = float(eval(kw.get('gmax', '4096'))); t.tau_f = float(eval(kw.get('tau_f', '4'))); t.eta_s = float(eval(kw.get('eta_s', '1')))
    t.rng = 88172645463325252 + SEED
    for k, a in A.items(): setattr(t, k, a.ctypes.data)
    taus = tau_of_cells(kw.get('tau', '2'), SEED, B)
    A['tauv'] = np.ascontiguousarray(taus, np.float64); t.tauv = A['tauv'].ctypes.data
    L.tis_init(ctypes.byref(t))
    
    b, _ = ts.stream(SEED, NB); b = np.ascontiguousarray(b, np.uint8)
    return dict(t=t, A=A, B=B, L=L, b=b, N=N, NB=NB, SEED=SEED)
