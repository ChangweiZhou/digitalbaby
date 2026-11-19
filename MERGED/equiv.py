# Each part of MERGED/tissue.c alone against its author's own program, step for step: both run the same options from birth to byte NB,
# the outputs of every cell after each of the last 512 bytes are compared, with the connections at the end.
#   python3 -I equiv.py <part: base|mk|tau|dly> [nbytes=..]
import sys, os, importlib.util, ctypes, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
part = sys.argv[1]; extra = dict(a.split('=') for a in sys.argv[2:])
OPTS = {'base': ('base', 'tis', "rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 we=1 wi=1/4"),
        'mk': ('mk', 'tis5', "rule=timing norm=1 lines_learn=1 eta_s=8 mk_mean=1 a_make=1/4096 start_w=1/8 make_period=1024 by_use=0"),
        'tau': ('tau', 'tis_tau', "rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 tau=u2-64e"),
        'dly': ('dly', 'tis', "rule=fixed gain=2 rho=1 gmax=4 dmax=32")}
folder, modname, opts = OPTS[part]
kw = dict(a.split('=') for a in opts.split()); kw.update(extra); kw.setdefault('nbytes', '16384')


def load(path, name):
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(name + '_' + os.path.basename(path), os.path.join(path, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.path.pop(0)
    return m


def run(T, NB):
    rec = np.zeros((512, T['N']), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(NB - 512), ctypes.c_int64(NB), ctypes.c_void_p(rec.ctypes.data))
    return rec


src = os.path.join(HERE, 'src', folder)
theirs = load(src, modname)
kw_t = dict(kw)
if part == 'dly':                                     # TISSUE_PREV2's prev2.py set-up of cand=delayrand, copied
    kw_t.pop('dmax')
    T1 = theirs.make(kw_t); t, A, N = T1['t'], T1['A'], T1['N']
    rng = np.random.default_rng(20261119); maxd = A['c_src'].shape[1]
    valid = np.arange(maxd)[None, :] < A['deg'][:, None]; cc = np.argwhere(valid & (A['c_src'] < N))
    dl = np.zeros((N, maxd), np.int32); dl[cc[:, 0], cc[:, 1]] = rng.integers(0, int(kw['dmax']) + 1, len(cc))
    hl = int(dl.max()) + 2; zh = np.zeros((hl, N)); T1['keep'] = (dl, zh)
    t.c_del = dl.ctypes.data; t.zh = zh.ctypes.data; t.hlen = hl
else:
    T1 = theirs.make(kw_t)
mine = load(HERE, 'tis')
T2 = mine.make(dict(kw))
NB = int(kw['nbytes'])
r1 = run(T1, NB); r2 = run(T2, NB)
d = np.abs(r1.astype(np.float64) - r2)
same_conn = np.array_equal(T1['A']['deg'], T2['A']['deg']) and all(np.array_equal(T1['A']['c_src'][i, :T1['A']['deg'][i]], T2['A']['c_src'][i, :T2['A']['deg'][i]]) for i in range(N if part == 'dly' else T1['N']))
print(f"{part}: {opts} nbytes {NB}: largest difference of outputs over the last 512 bytes {d.max():.3e}; cells ever differing {int((d.max(0) > 0).sum())}; "
      f"connections identical {same_conn}; mean connections {T1['A']['deg'].mean():.3f} / {T2['A']['deg'].mean():.3f}; made {T1['t'].made} / {T2['t'].made}; given up {T1['t'].given} / {T2['t'].given}", flush=True)
