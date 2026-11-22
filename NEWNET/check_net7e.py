"""net7 against net7e (event-driven kernel), step by step: cells, cols, surprise and predicted share at every byte; at the end
rw, rsrc, gsrc, rel, use (and hold). Context settings as in net10 (L=8, C=128, S=64, lines_in=16, n_values=128, hold=0, n_grow=8),
M=16 and 32, ctx=2 with mesh=2 mesh_dag=1 mesh_base=0, ctx=2 without mesh, ctx=0; streams tinyshakespeare (from its start) and
check_net7.py's context stream. Then net10 (L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1) with ctx_e=0 against ctx_e=1: every answer
and the context compartments' outputs. Usage: OMP_NUM_THREADS=1 python3 check_net7e.py [T]"""
import sys, time, os, numpy as np
from net7 import Tissue as A
from net7e import Tissue as B
import net10
T = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
here = os.path.dirname(os.path.abspath(__file__)) + "/../"
SH = np.frombuffer(open(here + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8)[:T].astype(int)
rng = np.random.default_rng(3); u = rng.choice(32, 12, replace=False); CS = []
while len(CS) < T: CS += [u[8] if rng.random() < .5 else u[9]] + list(u[:8]) + [u[10]]
CS = np.array(CS[:T])
streams = [("tinyshakespeare", SH), ("context stream", CS)]
settings = [("ctx=2 mesh=2 mesh_dag=1 mesh_base=0", dict(ctx=2, mesh=2, mesh_dag=1, mesh_base=0)),
            ("ctx=2 no mesh", dict(ctx=2)), ("ctx=0", dict(ctx=0))]
common = dict(seed=301, L=8, C=128, S=64, lines_in=16, n_values=128, hold=0, n_grow=8)
allok = True
for M in (16, 32):
    for sname, s in streams:
        for name, kw in settings:
            a, b = A(M=M, **common, **kw), B(M=M, **common, **kw)
            bad = None; ta = tb = 0.0
            for t, x in enumerate(s):
                t0 = time.perf_counter(); ca, la = a.step(int(x)); ta += time.perf_counter() - t0
                t0 = time.perf_counter(); cb, lb = b.step(int(x)); tb += time.perf_counter() - t0
                if not ((ca == cb).all() and (la == lb).all() and (a.surprise == b.surprise).all() and a.predicted == b.predicted):
                    bad = t; break
            end = all((getattr(a, k) == getattr(b, k)).all() for k in ("rw", "rsrc", "gsrc", "rel", "use", "hold"))
            ok = bad is None and end; allok &= ok
            nw = int((a.rw > 0).sum())
            print(f"M={M} {sname}, {name}: {'every byte identical' if bad is None else 'DIFFERS at byte %d' % bad} over {len(s)} bytes; "
                  f"end state {'identical' if end else 'DIFFERS'} ({nw} slots in use); ms a byte net7 {1e3*ta/len(s):.2f}, net7e {1e3*tb/len(s):.2f}", flush=True)
for M in (16, 32):
    for sname, s in streams:
        na = net10.Tissue(L=8, M=M, joint=1, mesh=2, mesh_base=0, mesh_dag=1)
        nb = net10.Tissue(L=8, M=M, joint=1, mesh=2, mesh_base=0, mesh_dag=1, ctx_e=1)
        bad = None
        for t, x in enumerate(s):
            na.step(int(x)); nb.step(int(x))
            ga, gb = na.answer(), nb.answer()
            if ga != gb or not (na.Oc == nb.Oc).all() or not (na.xc == nb.xc).all(): bad = t; break
        end = (na.Ac == nb.Ac).all() and (na.ctx.rw == nb.ctx.rw).all()
        ok = bad is None and end; allok &= ok
        print(f"net10 L=8 M={M} joint=1 mesh=2 mesh_base=0 mesh_dag=1, {sname}: ctx_e=0 vs 1 "
              f"{'every answer identical' if bad is None else 'DIFFERS at byte %d' % bad} over {len(s)} bytes; end state {'identical' if end else 'DIFFERS'}", flush=True)
print("ALL IDENTICAL" if allok else "MISMATCH FOUND")
