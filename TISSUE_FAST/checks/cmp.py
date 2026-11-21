"""Run one tissue with the program in <dir> and save emissions, connections, cells, generator state and CPU time."""
import sys, time, pickle, numpy as np
d, rule, n, mode, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
sys.path[:0] = [d + "/TISSUE_STEP1", d + "/TISSUE_EVENT"]
import tissue_step1 as ts, evt
st, _ = ts.stream(73129, n)
sw = dict(floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True) if mode == "on" else {}
T = evt.EvTissue(73129, rule, **sw)
c = time.process_time(); em = T.run_bytes(st); cpu = time.process_time() - c
snd, rcv, w, e, k0, ub = T.live_full()
cells = T.cells()
np.savez(out, em=em, snd=snd, rcv=rcv, w=w, e=e, k0=k0, **{k: v for k, v in cells.items() if v is not None},
         rng=np.array(T.rng_state()[1], np.int64), cpu=cpu, k=T.k, made=T.made, gone=T.given_up)
print(out, "cpu %.2f s, %.4f ms a byte" % (cpu, 1000 * cpu / n))
