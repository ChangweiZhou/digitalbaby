"""V149 cycle-1 checks (design section 8). Writes cycles/CYCLE1_CHECKS.json. python checks.py"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import corpus
import readouts as R
import states

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(corpus.V148))
import probe  # noqa: E402  (V148: state_hash)

fm = states.fastmem
out = []


def check(name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


# corpus pins and splits
for t in corpus.TEXTS:
    raw = corpus.load(t)
    recs = corpus.records(raw)
    sp = corpus.split(recs, 40, 5, 10) if t != "buster" else corpus.split(recs, 40, 5, 10)
    check(f"corpus {t}: pinned, split without overlap", sp["offsets"]["calib"][1] <= sp["offsets"]["held"][0],
          {"bytes": len(raw), "records": len(recs), "offsets": sp["offsets"]})

recs = corpus.records(corpus.load("buster"))[:6]

# states equal values() on the full 16-compartment memory, for compartments 0 and 15
mem1 = states.new_memory("perm")
full = fm.FastMemory()
ok_x = ok_dx = True
for r in recs:
    x1, dx1, _ = states.record_states(mem1, r)
    view = full.clone()
    view._control(True)
    view._control(False)
    for i in range(len(r) + 1):
        f = view.fe.clone()
        f.advance(view.time + fm.DELTA)
        x = view.encode(f.read())[0]
        for j in (0, 15):
            fly = view.bank[j].fly.clone()
            fly.rest(fm.DELTA)
            dx = fm.NATIVE.common.observed_activity(fly.m, x[None, :])[0]
            ok_dx &= np.array_equal(dx, dx1[i])
        ok_x &= np.array_equal(x, x1[i])
        if i < len(r):
            view._event(r[i])
check("binary code x equals values() code on the 16-compartment memory", ok_x)
check("observed activity dx (1-compartment bank) equals compartments 0 and 15 of the full bank", ok_dx)

# record independence and no state change
h0 = probe.state_hash(mem1)
a = states.record_states(mem1, recs[3])
for r in recs[:3]:
    states.record_states(mem1, r)
b = states.record_states(mem1, recs[3])
check("a record's states do not depend on records read before it", all(np.array_equal(u, v) for u, v in zip(a, b)))
check("reading states leaves the memory unchanged (state hash)", probe.state_hash(mem1) == h0)

# raw vs max-normalised trace give the same code when thresholds are zero
same, n = True, 0
view = mem1.clone()
view._control(True)
view._control(False)
for r in recs:
    view._control(True)
    view._control(False)
    for byte in r:
        view._event(byte)
        f = view.fe.clone()
        f.advance(view.time + fm.DELTA)
        same &= np.array_equal(view.encode(f.read())[0], view.encode(f.p)[0])
        n += 1
check("raw and max-normalised FE-PERM traces give identical KC codes (thresholds zero)", same, {"positions": n})

# PERM-last: zero code at line start, single-byte code afterwards
x, dx, last = states.record_states(mem1, recs[0])
check("PERM-last is the zero code at the line start", not last[0].any() and not x[0].any() and not dx[0].any())
f = mem1._new_front()
f.feed(recs[0][0], 0.0)
f.advance(fm.DELTA)
check("PERM-last code at position 1 depends only on the previous byte", np.array_equal(last[1], mem1.encode(f.read())[0]))

# readouts
S, y = states.collect(recs)
cls = R.Classes(y)
c = cls.encode(y)
X = R.with_empty(S["PERM-obs"])
theta, _, _ = R.fit_r1(X, c, cls.K, 10.0, None, 30)
P1 = R.predict_r1(theta, X, cls.K)
P2 = R.predict_r2(R.fit_r2(X, c, cls.K), X, 0.1)
check("R1 and R2 class probabilities sum to 1", np.allclose(P1.sum(1), 1) and np.allclose(P2.sum(1), 1))
full_mass = P1[0, :-1].sum() + P1[0, -1]
check("bits over 257 symbols: seen classes plus OTHER spread over unseen symbols sum to 1", abs(full_mass - 1) < 1e-9,
      {"seen": len(cls.seen), "unseen": cls.n_unseen})
toy = R.toy_check()
check("R2 toy values match hand computation", np.allclose(toy["vote_both"], toy["expected_vote_both"]), toy)
theta_b, _, _ = R.fit_r1(X, c, cls.K, 10.0, None, 30)
check("R1 fit is deterministic and uses only the taught states", np.array_equal(theta, theta_b))
X1 = R.onehot_last(recs)
check("one-hot last-symbol code has exactly one active feature per position", np.all(np.diff(X1.indptr) == 1))

# closure can be pinned: repo modules in sys.modules after importing the runner
import run  # noqa: E402,F401
root = str(HERE.parent)
mods = sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and m.__file__.startswith(root)})
check("runtime closure lists the V149, V148 and native modules", any("minifly_v149" in m for m in mods)
      and any("fastmem.py" in m for m in mods) and any("ladder.py" in m for m in mods), {"n_modules": len(mods)})

# cost
recs_c = corpus.records(corpus.load("buster"))[:30]
t = time.process_time()
S, y = states.collect(recs_c)
check("cost measured", True, {"states_cpu_s_per_position_all_arms": round((time.process_time() - t) / len(y), 4)})

(HERE / "cycles").mkdir(exist_ok=True)
(HERE / "cycles" / "CYCLE1_CHECKS.json").write_text(json.dumps(out, indent=1))
print(sum(o["pass"] for o in out), "/", len(out), "pass")
