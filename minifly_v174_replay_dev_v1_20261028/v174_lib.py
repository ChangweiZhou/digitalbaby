"""V174 library (design sections 1-2): V172's native arm (DELTA writer with warmup, homeostasis optional), extended with a
write-inhibited observation; V172's generator under salt V174; label-free slot draws; NEW-phase forks.

A write-inhibited record runs the same prompt events, prediction, clock, homeostasis update (if on) and warmup counter
as a written one, and calls the native signed teacher with write=False (a non-plastic native event).
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SET = json.loads((HERE / "SETTINGS.json").read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L2 = load("v172_lib", ROOT / "minifly_v172_native_e1_dev_v1_20261027" / "v172_lib.py")
EOS = L2.EOS


def rng(world, stream):
    return np.random.Generator(np.random.PCG64(int.from_bytes(
        hashlib.sha256(f"{SET['data']['salt']}|{world}|{stream}".encode()).digest()[:16], "big")))


def dataset(world):
    """V172's (= V154's) generator under salt V174; only OLD, the selected NEW and the pass orders are returned."""
    m = SET["data"]["operand_max"] + 1
    orbits = [(i, j) for i in range(m) for j in range(i, m)]
    order = rng(world, "split").permutation(len(orbits))
    orient = rng(world, "orientation").integers(0, 2, len(orbits))
    rows = []
    for k in order:
        i, j = orbits[int(k)]
        a, b = (j, i) if orient[k] else (i, j)
        rows.append({"id": f"{world}:{int(k)}", "prompt": f"{a}+{b}=".encode(), "answer": str(a + b).encode()})
    n_old, n_pool, n_new = SET["data"]["OLD"], SET["data"]["NEW_pool"], SET["data"]["NEW"]
    old, pool = rows[:n_old], rows[n_old:n_old + n_pool]
    new = sorted(pool, key=lambda r: (hashlib.sha256(f"{SET['data']['salt']}|{world}|NEW|{r['id']}".encode()).hexdigest(),
                                      r["id"]))[:n_new]
    d = SET["dose"]
    return {"OLD": old, "NEW": new,
            "old_orders": [rng(world, f"OLD|pass|{p}").permutation(n_old).tolist() for p in range(1, d["OLD_orders"] + 1)],
            "new_orders": [rng(world, f"NEW|pass|{p}").permutation(n_new).tolist() for p in range(1, d["NEW_passes"] + 1)]}


class Arm(L2.Arm):
    def train(self, prompt, answer, write=True):
        self._prompt(bytes(prompt))
        sites = []
        for y in (*bytes(answer), EOS):
            v = self.values()
            p, lp = L2.read(v)
            c = self.coef(v, p, y)
            assert np.isfinite(c).all() and np.abs(c).max() <= 1
            self._tick()
            x = self._code()
            if self.homeo:
                self._homeostasis(x)
            for j, s in enumerate(self.bank):
                s.pending_t, s.pending_x = self.time, x
                s.teach_signed(0., float(c[j]) if write else 0., self.time, write=write)
            if y != EOS:
                self.fe.feed(y, self.time)
            sites.append({"y": int(y), "lp": float(lp[y]), "argmax": int(p.argmax())})
            self.n += 1
        self.records += 1
        return sites


def make(homeo):
    return Arm("DELTA", homeo)


def old_phase(world, homeo, d):
    m, hist = make(homeo), []
    for p in range(1, SET["dose"]["OLD_passes"] + 1):
        for i in d["old_orders"][(p - 1) % SET["dose"]["OLD_orders"]]:
            m.train(d["OLD"][i]["prompt"], d["OLD"][i]["answer"])
            hist.append(d["OLD"][i])
    return m, hist


def fork(m0, hist0, d, world, replay, passes=None):
    """NEW phase from the OLD checkpoint: each NEW record is followed by one slot record drawn uniformly (fixed seed)
    from all earlier presentations; REPLAY writes it, BLOCKED observes it write-inhibited."""
    m, hist, g = m0.clone(), list(hist0), rng(world, "slot")
    draws = []
    for p in range(1, (passes or SET["dose"]["NEW_passes"]) + 1):
        for i in d["new_orders"][p - 1]:
            r = d["NEW"][i]
            m.train(r["prompt"], r["answer"])
            hist.append(r)
            k = int(g.integers(0, len(hist)))
            s = hist[k]
            m.train(s["prompt"], s["answer"], write=replay)
            hist.append(s)
            draws.append(s["id"])
    return m, draws


def values_digest(model):
    return hashlib.sha256(b"".join(s.fly.m.fast.tobytes() + s.fly.m.slow.tobytes() for s in model.bank)).hexdigest()
