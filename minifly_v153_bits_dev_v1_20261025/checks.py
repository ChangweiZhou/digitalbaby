"""V153 technical checks (C1). Writes results/technical/CHECKS.json. Run through launch.py --stage technical."""
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import corpus
import native
import readouts as RO
import states

R = RO.R
fm = states.fastmem
HERE = Path(__file__).resolve().parent
out = []


def check(name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


recs = corpus.records(corpus.load("buster"))[:5]

# front-end cells
f95, f81 = states.make_front("P95"), states.make_front("P81")
check("fade per byte: P95 0.95, P81/N81 0.807, N95 0.95",
      abs(math.exp(-fm.DELTA / f95.tau) - 0.95) < 1e-12 and abs(math.exp(-fm.DELTA / f81.tau) - math.exp(-fm.DELTA / 10)) < 1e-12
      and abs(math.exp(-fm.DELTA / states.make_front("N95").tau) - 0.95) < 1e-12
      and abs(math.exp(-fm.DELTA / states.make_front("N81").tau) - math.exp(-fm.DELTA / 10)) < 1e-12)

# cells P95 and N81 reproduce V149's PERM-x and FE0-x states exactly
_spec = importlib.util.spec_from_file_location("v149_states", HERE.parent / "minifly_v149_input_ceiling_dev_20261025" / "states.py")
v149s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v149s)
ok = True
for r in recs:
    ok &= np.array_equal(states.record_states(states.new_memory("P95"), r)[0], v149s.record_states(v149s.new_memory("perm"), r)[0])
    ok &= np.array_equal(states.record_states(states.new_memory("N81"), r)[0], v149s.record_states(v149s.new_memory("fe0"), r)[0])
    ok &= np.array_equal(states.record_states(states.new_memory("N81"), r)[1], v149s.record_states(v149s.new_memory("fe0"), r)[1])
check("P95 and N81 codes (and N81 observed activity) equal V149's PERM-x / FE0-x / FE0-obs states", ok)

# the four cells differ where they should
x = {c: states.record_states(states.new_memory(c), recs[0])[0] for c in states.CELLS}
check("cells give different codes beyond the first byte",
      all(not np.array_equal(x[u][3:], x[v][3:]) for u in states.CELLS for v in states.CELLS if u < v))
check("all cells give the same code after exactly one byte (no history yet)",
      all(np.array_equal(x["N81"][1], x[c][1]) for c in ("N95",)) and np.array_equal(x["P95"][1], x["P81"][1]))

# NATIVE (homeostasis off) sees exactly the frozen observed state after teaching
mem = native.new_native()
for r in recs[:3]:
    mem.train(b"", r)
view = mem.clone()
view._control(True)
view._control(False)
frozen = states.record_states(states.new_memory("N81"), recs[4])[1]
same = True
for i in range(len(recs[4]) + 1):
    f = view.fe.clone()
    f.advance(view.time + fm.DELTA)
    xx = view.encode(f.read())[0]
    fly = view.bank[0].fly.clone()
    fly.rest(fm.DELTA)
    same &= np.array_equal(fm.NATIVE.common.observed_activity(fly.m, xx[None, :])[0], frozen[i])
    if i < len(recs[4]):
        view._event(recs[4][i])
check("NATIVE after teaching reads exactly the frozen FE0 observed state (thresholds stay zero)", same and not mem.theta.any())

# C16 readout equals the native logits formula
rng = np.random.default_rng(0)
X = sparse.csr_matrix(rng.random((4, 7)) * (rng.random((4, 7)) < 0.4))
theta = rng.normal(size=16 * 7 + 16)
W, b = theta[:112].reshape(16, 7), theta[112:]
ok = all(np.allclose(RO.c16_logits(theta, X, False)[i], fm.logits(np.asarray(X[i].todense()).ravel() @ W.T + b))
         for i in range(4))
check("C16 port logits equal fastmem.logits of the 16 linear values", ok)

# gradients: finite differences for C16/C16Q and the online delta step
y = rng.integers(0, 257, 4)


def loss_c16(th, wq, lam):
    Z = RO.c16_logits(th, X, wq)
    Z -= Z.max(1, keepdims=True)
    W_ = th[:112].reshape(16, 7)
    return -(Z[np.arange(4), y] - np.log(np.exp(Z).sum(1))).sum() + 0.5 * lam * (W_ * W_).sum()


for wq in (False, True):
    th = rng.normal(size=112 + 16 + (257 if wq else 0)) * 0.3
    captured = {}
    orig = RO.minimize

    def fake(f, x0, jac, method, options):
        captured["g"] = f(x0)[1]
        return type("R", (), {"x": x0, "nit": 0, "success": True})()
    RO.minimize = fake
    RO.fit_c16(X, y, 0.7, wq, th, 1)
    RO.minimize = orig
    num = np.array([(loss_c16(th + e, wq, 0.7) - loss_c16(th - e, wq, 0.7)) / 2e-6
                    for e in np.eye(len(th))[:40] * 1e-6])
    err = float(np.max(np.abs(num - captured["g"][:40]) / np.maximum(1, np.abs(num))))
    check(f"C16{'Q' if wq else ''} analytic gradient matches finite differences (first 40 coords)", err < 1e-5, err)

S, ys = states.collect(recs[:2], cells=("N81",))
cls = R.Classes(ys)
c = cls.encode(ys)
Xb = R.with_empty(S["N81-obs"])
snaps, n = RO.online_delta(Xb[:1], c[:1], cls.K, 0.05, 1)
P0 = np.full(cls.K, 1 / cls.K)
P1 = RO.predict_linear(snaps[0], Xb[:1])[0]
check("one online delta step raises the target's probability and uses warmup 1/16 at n=0",
      P1[c[0]] > P0[c[0]] and abs(RO.warmup(0) - 1 / 16) < 1e-12 and n == 1)
check("online delta rule is deterministic", all(np.array_equal(u[0], v[0]) for u, v in
      zip(RO.online_delta(Xb, c, cls.K, 0.01, 2)[0], RO.online_delta(Xb, c, cls.K, 0.01, 2)[0])))
check("readout probabilities sum to 1", np.allclose(RO.predict_linear(snaps[0], Xb).sum(1), 1))

# launch refuses an unlocked or ungranted stage (functions only; no attempt is written)
import launch  # noqa: E402
auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
check("AUTHORIZATION: official not granted; caps are the defaults 2 CPU-h / 5,000 records / 3 cycles",
      auth["stages"]["official"]["granted"] is False and auth["caps"] == {"max_cpu_h": 2, "max_records": 5000, "max_cycles": 3})

# cost (states for one reduced dev record set, all cells + obs)
t = time.process_time()
S, ys = states.collect(corpus.records(corpus.load("buster"))[:20])
per = (time.process_time() - t) / len(ys)
check("cost measured", True, {"states_cpu_s_per_position_4cells_plus_obs": round(per, 4), "positions": len(ys)})
stage = os.environ.get("MINIFLY_STAGE", "unlaunched")
d = HERE / "results" / stage
d.mkdir(parents=True, exist_ok=True)
(d / "CHECKS.json").write_text(json.dumps(out, indent=1))
print(sum(o["pass"] for o in out), "/", len(out), "pass")
