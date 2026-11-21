import sys, time, numpy as np
d, rule, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path[:0] = [d + "/TISSUE_STEP1", d + "/TISSUE_EVENT"]
import tissue_step1 as ts, evt
st, _ = ts.stream(73129, n)
T = evt.EvTissue(73129, rule, floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True, withdraw_by_use=True)
c = time.process_time(); em = T.run_bytes(st); cpu = time.process_time() - c
snd, rcv, w, e, k0, ub = T.live_full(); cells = T.cells()
np.savez(out, em=em, snd=snd, rcv=rcv, w=w, e=e, k0=k0, **{k: v for k, v in cells.items() if v is not None},
         rng=np.array(T.rng_state()[1], np.int64), cpu=cpu, made=T.made, gone=T.given_up, wu=T.withdrawn_by_use)
print(out, "%.4f ms a byte, %.1f connections a cell" % (1000 * cpu / n, (snd < ts.YES_LINE).sum() / ts.N))
