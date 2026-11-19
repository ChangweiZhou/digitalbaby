# Item 1 of the reviewer's message of 09:53 UTC, 19 November: the option long=K-R in this program, against the reviewer's.
#   python3 -I eqcheck.py <reference folder> [seed] [nbytes]
# Builds one tissue with the reviewer's tis.py and tissue.c (the folder given: reviewer_tissue_20261119 at 341741c26) and one with this
# folder's (long written here; tissue.c with the TISSUE_LONG delay additions, no delay set), both with the line of the reviewer's batch 40:
#   rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024
#   start_w=1/64 long=8-1/2
# Compares the long cells, every cell's members and starting sources, the outputs of every cell after every byte, and the connections
# at the end (sources, strengths). Writes qual/eqcheck_<seed>_<nbytes>.json.
import ctypes
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(sys.argv[1]); SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 73129; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
LINE = dict(rule='timing', prune_only='1', norm='1', lines_learn='1', we='3/4', wi='1/4', gain='3', rho='1', gmax='8', mk_long='1',
            a_make='1/16384', make_period='1024', start_w='1/64', long='8-1/2', seed=str(SEED), nbytes=str(NB))


def load(folder, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(folder, 'tis.py'))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def run(m):
    T = m.make(dict(LINE))
    rec = np.zeros((NB, T['N']), np.float32)
    T['L'].tis_run(ctypes.byref(T['t']), ctypes.c_void_p(T['b'].ctypes.data), ctypes.c_int64(0), ctypes.c_int64(NB), ctypes.c_int64(0), ctypes.c_int64(NB),
                   ctypes.c_void_p(rec.ctypes.data))
    return T, rec


ref = load(REF, 'tis_reference'); mine = load(HERE, 'tis_mine')
Tr, Rr = run(ref); Tm, Rm = run(mine)
Br, Bm = Tr['B'], Tm['B']
N = Tr['N']
res = dict(seed=SEED, nbytes=NB, line=LINE)
res['long_cells_equal'] = bool(np.array_equal(np.asarray(Br['longs']), np.asarray(Bm['longs'])))
res['long_cells'] = int(len(Bm['longs']))
res['members_equal'] = all(list(Br['members'][i]) == list(Bm['members'][i]) for i in range(N))
res['starting_sources_equal'] = all(list(Br['src'][i]) == list(Bm['src'][i]) for i in range(N))
res['outputs_identical'] = bool(np.array_equal(Rr, Rm))
res['largest_output_difference'] = float(np.abs(Rr.astype(np.float64) - Rm).max())
dr, dm = Tr['A']['deg'], Tm['A']['deg']
res['connections_a_cell_equal'] = bool(np.array_equal(dr, dm))
same_src = bool(np.array_equal(dr, dm)) and all(np.array_equal(Tr['A']['c_src'][i, :dr[i]], Tm['A']['c_src'][i, :dm[i]]) for i in range(N))
res['sources_at_end_equal'] = same_src
res['strengths_at_end_identical'] = bool(same_src and all(np.array_equal(Tr['A']['c_w'][i, :dr[i]], Tm['A']['c_w'][i, :dm[i]]) for i in range(N)))
res['connections_at_end'] = int(dm.sum())
res['reference_tissue_c_sha256'] = hashlib.sha256(open(os.path.join(REF, 'tissue.c'), 'rb').read()).hexdigest()
res['reference_tis_py_sha256'] = hashlib.sha256(open(os.path.join(REF, 'tis.py'), 'rb').read()).hexdigest()
res['tissue_c_sha256'] = hashlib.sha256(open(os.path.join(HERE, 'tissue.c'), 'rb').read()).hexdigest()
res['tis_py_sha256'] = hashlib.sha256(open(os.path.join(HERE, 'tis.py'), 'rb').read()).hexdigest()
res['pass'] = bool(res['long_cells_equal'] and res['members_equal'] and res['starting_sources_equal'] and res['outputs_identical']
                   and res['sources_at_end_equal'] and res['strengths_at_end_identical'])
os.makedirs(os.path.join(HERE, 'qual'), exist_ok=True)
json.dump(res, open(os.path.join(HERE, 'qual', f'eqcheck_{SEED}_{NB}.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('line',) and not k.endswith('sha256')}), flush=True)
