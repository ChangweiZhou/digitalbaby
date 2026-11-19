# One tissue of step 1's second round on the fast program (E57's worker's evt_core.c), run by the reviewer.
# usage: round.py <dir with evt.py> <rule> <floor> <label>
import sys, time, json, numpy as np
d, rule, floor, label = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
sys.path.insert(0, 'TISSUE_STEP1'); sys.path.insert(0, d)
import evt, tissue_step1 as ts, readings
SEED, NB = 73129, 16384
WIN = {"A1": (2048, 6144), "A2": (12288, 16384)}
b, _ = ts.stream(SEED, NB)
T = evt.EvTissue(SEED, rule, floor=floor, tol=1/1024, trace_sender=True, strength_byte=True)
rec = {w: np.zeros((hi - lo, ts.N), np.float32) for w, (lo, hi) in WIN.items()}
dist = {}
t0 = time.process_time(); pos = 0
for w, (lo, hi) in WIN.items():
    T.run_bytes(b[pos:lo])
    for i in range(lo, hi):
        T.run_bytes(b[i:i + 1]); rec[w][i - lo] = T.x
    dist[w] = T.distances(sorted(set(b[lo:hi].tolist())))
    pos = hi
cpu = time.process_time() - t0
snd = T.live()[0]
c = T.counters()
out = dict(label=label, rule=rule, floor=floor, cpu_s_per_byte=cpu / NB, conns_per_cell=len(snd) / ts.N, made=int(T.made), given_up=int(T.given_up),
           cells_updated_a_step=c['cell_updates'] / c['steps'], senders_a_step=c['senders_passing_on'] / c['steps'])
for w, (lo, hi) in WIN.items():
    x = np.where(rec[w] < 0.01, 0.0, rec[w])
    r = readings.read_window(x, b[lo:hi], dist[w], SEED, stream=b, start=lo, lags=(0, 1, 2))
    out[w] = dict(shares={str(k): v for k, v in r['shares'].items()},
                  S={str(dd): {str(L): (r['scores'][dd]['by_lag'][L] or {}).get('S') for L in (0, 1, 2)} for dd in (1, 2, 4)},
                  cells={str(dd): r['scores'][dd]['cells_at_d'] for dd in (1, 2, 4)},
                  c0={str(k): v for k, v in r['c0'].items()})
json.dump(out, open(f'out_{label}.json', 'w'), default=float)
print(label, 'done', f"{cpu / NB * 1e3:.2f} ms/byte", f"{out['conns_per_cell']:.1f} conns", flush=True)
