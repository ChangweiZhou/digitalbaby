"""Timing only: CPU seconds per byte, by blocks, on seed 73129, for E63's program and the event-driven program.
usage: python3 bench.py <orig|evt> <rule> <nbytes> <block> [on]"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TISSUE_STEP1"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TISSUE_EVENT"))
import numpy as np
import tissue_step1 as ts
prog, rule, n, blk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
on = len(sys.argv) > 5 and sys.argv[5] == "on"
st, _ = ts.stream(73129, n)
t0 = time.process_time()
if prog == "orig":
    T = ts.Tissue(73129, rule)
    run = lambda b: [T.run_byte(int(x)) for x in b]
else:
    import evt
    sw = dict(floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True) if on else {}
    T = evt.EvTissue(73129, rule, **sw)
    run = T.run_bytes
init = time.process_time() - t0
blocks = []
for i in range(0, n, blk):
    c = time.process_time(); run(st[i:i + blk]); blocks.append((time.process_time() - c) / blk)
res = dict(prog=prog, rule=rule, on=on, nbytes=n, block=blk, init_s=round(init, 2),
           cpu_s_per_byte_by_block=[round(b, 6) for b in blocks], mean=round(sum(blocks) / len(blocks), 6))
print(json.dumps(res)); sys.stdout.flush()
