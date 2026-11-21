"""Cost and the tissue's state under one setting of its numbers (seed 73129, first form, switches on)."""
import sys, time, json, numpy as np
sys.path[:0] = ["TISSUE_STEP1", "TISSUE_EVENT"]
import tissue_step1 as ts
a = dict(arg.split("=") for arg in sys.argv[1:])
rule = a.get("rule", "first"); n = int(a.get("bytes", 16384)); S = int(a.get("S", 32)); cap = int(a.get("cap", 1 << 30))
tol = float(eval(a.get("tol", "1/1024"))); astar = float(eval(a.get("astar", "0.125")))
ts.S = S
import evt
evt.L.evt_set_cap(cap)
st, _ = ts.stream(73129, n)
T = evt.EvTissue(73129, rule, floor=1/128, tol=tol, trace_sender=True, strength_byte=True, astar=astar)
c = np.zeros(8); half = n // 2
T.run_bytes(st[:half]); evt.L.evt_counters(c); c0 = c.copy()
t = time.process_time(); em = T.run_bytes(st[half:]); cpu = time.process_time() - t
evt.L.evt_counters(c); d = (c - c0) / (n - half)
snd, rcv, w, k0 = T.live()
cells = int((snd < ts.YES_LINE).sum())
print(json.dumps(dict(S=S, cap=cap if cap < 1 << 30 else None, tol=tol, astar=astar, ms_a_byte=round(1000 * cpu / (n - half), 4),
      conns_a_cell=round(cells / ts.N, 1), senders_a_byte=round(d[1]), visits_a_byte=round(d[2]), cells_computed_a_byte=round(d[0]),
      emission_share=round(float((em >= 0).mean()), 4), active_cells_end=int((T.x > 0).sum()))))
