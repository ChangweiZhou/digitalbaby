"""V172 library (design section 1-3): data (V154's generator under salt V172), the four native arms, panels.

The native bank is `minifly_cls_dev_20261017/fastmem.py` unchanged (FE-PERM 0.95 per byte = Codex's P_SLOW, fresh
adaptation, q = 0). Arms differ only in the signed-write coefficient (N2B or DELTA, both scaled by the owner's warmup)
and in one-sided KC homeostasis. Teaching happens at answer sites after the prediction, as in V154/V171.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "minifly_cls_dev_20261017"))
import fastmem  # noqa: E402

SET = json.loads((HERE / "SETTINGS.json").read_text())
C = fastmem.CODE
EOS = 256
DELTA = fastmem.DELTA
W = SET["writer"]


# ---------------------------------------------------------------- data
def rng(world, stream):
    return np.random.Generator(np.random.PCG64(int.from_bytes(
        hashlib.sha256(f"{SET['data']['salt']}|{world}|{stream}".encode()).digest()[:16], "big")))


def dataset(world):
    """V154's dataset() under salt V172; returns only OLD, the selected NEW and the pass orders (held rows unread)."""
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


# ---------------------------------------------------------------- arms
def warmup(n):
    return W["warmup_start"] + (1 - W["warmup_start"]) * min(n / W["warmup_targets"], 1)


def read(v):
    z = C @ ((v - v.mean()) / 4)
    z = z - z.max()
    lp = z - np.log(np.exp(z).sum())
    return np.exp(lp), lp


class Arm(fastmem.FastMemory):
    def __init__(self, writer, homeo, force_warmup=None):
        h = SET["homeo"]
        super().__init__(front="perm", teach="delta", fresh_adapt=True, homeo=homeo, homeo_rate=h["rate"],
                         homeo_c=h["c"], homeo_beta=h["beta"], delta_temp=W["delta_temp"])
        assert writer in ("N2B", "DELTA")
        self.writer, self.n, self.force = writer, 0, force_warmup

    def clone(self):
        o = super().clone()
        o.theta, o.freq = self.theta.copy(), self.freq.copy()
        return o

    def coef(self, v, p, y):
        w = self.force if self.force is not None else warmup(self.n)
        if self.writer == "N2B":
            e = p.copy()
            e[y] -= 1
            g = C.T @ e / 4
            return W["n2b_gain"] * w * (g - g.mean())
        return w * (np.tanh((v - v.mean()) / self.delta_temp) - C[y]) / 2

    def train(self, prompt, answer):
        """One teaching record: prompt, then each answer byte and EOS taught after its prediction."""
        self._prompt(bytes(prompt))
        sites = []
        for y in (*bytes(answer), EOS):
            v = self.values()
            p, lp = read(v)
            c = self.coef(v, p, y)
            assert np.isfinite(c).all() and np.abs(c).max() <= 1
            self._tick()
            x = self._code()
            if self.homeo:
                self._homeostasis(x)
            for j, s in enumerate(self.bank):
                s.pending_t, s.pending_x = self.time, x
                s.teach_signed(0., float(c[j]), self.time, write=True)
            if y != EOS:
                self.fe.feed(y, self.time)
            sites.append({"y": int(y), "lp": float(lp[y]), "argmax": int(p.argmax()), "coef_abs_max": float(np.abs(c).max())})
            self.n += 1
        self.records += 1
        return sites


def make(arm, force_warmup=None):
    a = SET["arms"][arm]
    return Arm(a["writer"], a["homeo"], force_warmup)


# ---------------------------------------------------------------- evaluation (clones; never writes)
def own_and_tf(model, prompt, answer):
    own = model.clone()
    own._prompt(bytes(prompt))
    out, ended = bytearray(), False
    for _ in range(SET["generation_cap"]):
        p, _ = read(own.values())
        y = int(p.argmax())
        if y == EOS:
            ended = True
            break
        out.append(y)
        own._event(y)
    tf = model.clone()
    tf._prompt(bytes(prompt))
    sites = []
    for y in (*bytes(answer), EOS):
        p, lp = read(tf.values())
        sites.append({"y": int(y), "lp": float(lp[y]), "argmax": int(p.argmax())})
        if y != EOS:
            tf._event(y)
    return {"output": out.decode("latin-1"), "EOS": ended, "correct": bool(ended and bytes(out) == bytes(answer)),
            "tf": sites}


def panel(model, items):
    rows = [{"id": r["id"], **own_and_tf(model, r["prompt"], r["answer"])} for r in items]
    pay = [s for r in rows for s in r["tf"] if s["y"] != EOS]
    eos = [s for r in rows for s in r["tf"] if s["y"] == EOS]
    n = sum(len(r["tf"]) for r in rows)
    return {"correct": sum(r["correct"] for r in rows), "n": len(rows),
            "bits_per_target": -sum(s["lp"] for r in rows for s in r["tf"]) / n / np.log(2),
            "tf_payload_acc": sum(s["argmax"] == s["y"] for s in pay) / len(pay),
            "tf_eos_acc": sum(s["argmax"] == s["y"] for s in eos) / len(eos),
            "empty_eos": sum(r["EOS"] and r["output"] == "" for r in rows), "cap": sum(not r["EOS"] for r in rows),
            "rows": rows}


def digest(model):
    st = [(s.fly.m.fast.tobytes(), s.fly.m.slow.tobytes(), s.fly.m.adapt.tobytes()) for s in model.bank]
    return hashlib.sha256(pickle.dumps((st, model.theta.tobytes(), model.freq.tobytes(), model.time, model.n),
                                       protocol=5)).hexdigest()
