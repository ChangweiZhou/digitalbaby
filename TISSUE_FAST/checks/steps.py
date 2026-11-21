import sys, os, numpy as np
sys.path.insert(0, "TISSUE_STEP1"); sys.path.insert(0, "TISSUE_EVENT")
import tissue_step1 as ts, evt
rule = sys.argv[1]
st, _ = ts.stream(73129, 3072)
T = evt.EvTissue(73129, rule, floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True)
T.run_bytes(st[:2048])
S = ts.S
upd = np.zeros(S); snd = np.zeros(S)
o = np.zeros(16)
for b in st[2048:3072]:
    for s in range(S):
        evt.L.evt_counters(o); c0 = o.copy()
        evt.L.evt_step(int(b))
        evt.L.evt_counters(o); upd[s] += o[0] - c0[0]
a = upd / 1024
print("steps a byte:", S)
print("cell updates a byte, total: %.0f" % a.sum())
print("by step within the byte:", " ".join("%d" % v for v in a))
