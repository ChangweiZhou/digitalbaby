import sys, time, numpy as np
d, rule, n, ce, spike, fire = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), float(eval(sys.argv[6]))
sys.path[:0] = [d + "/TISSUE_STEP1", d + "/TISSUE_EVENT"]
import tissue_step1 as ts, evt
evt.L.evt_set_norm(1, ce, ce / 3); evt.L.evt_set_turnover(1, 32768, 2048 if rule == "first" else 512, 1.0); evt.L.evt_set_spike(0, 0.125); evt.L.evt_set_inh_adapt(1, float(sys.argv[7]), float(eval(sys.argv[8])), 4.0)
st, _ = ts.stream(73129, n)
T = evt.EvTissue(73129, rule, floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True, withdraw_by_use=True)
for blk in range(0, n, 1024):
    c = time.process_time(); em = T.run_bytes(st[blk:blk + 1024]); cpu = time.process_time() - c
    snd, rcv, w, k0 = T.live()
    dist = T.distances(sorted(set(st[blk:blk + 1024].tolist())))
    act = T.x > 0
    print(f"bytes {blk + 1024}: {cpu / 1.024:.3f} ms/B conns {(snd < ts.YES_LINE).sum() / ts.N:.1f} active {act.sum()} by d: " + " ".join(f"{k}:{act[dist == k].sum()}/{(dist == k).sum()}" for k in range(1, 12) if (dist == k).any()))
