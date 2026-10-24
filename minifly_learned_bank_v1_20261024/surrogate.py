"""Fast tuning surrogate (development worlds only; never used for the main run or for claims).

Same FE-PERM traces, same learned layer and three-factor rule, same canonical PN->KC top-k as lb1.Model's learned
bank; the native opponent stores are replaced by a linear readout on the KC code learned with the same residual
e = (tanh(m/T) - y)/2. Used to narrow eta0 / lam / hidden / passes before confirming on native stores.

python surrogate.py --worlds D2,D3 --eta0 0.001,0.003,0.01 --lam 0,1e-3 --hidden 128 --passes 120 --lr 0.05
"""
import argparse
import itertools
import json

import numpy as np

import data
import lb1

ALL = [(a, b) for a in range(10) for b in range(10)]


def run(world, task, arm, hidden, eta0, lam, passes, lr, wd_r=0.0, eta_ramp=1024, every=10, centre_tau=0.0):
    model = lb1.Model("LEARN" if arm in ("LEARN", "SHUF") else "FROZEN", world["model_seed"], hidden=hidden,
                      centre_tau=centre_tau)
    m, side_n, layer = model.m, model.side_n, model.layer
    traces = {p: lb1.trace_for(data.prompt(*p)) for p in ALL}
    taught = {p: data.answer(task, *p) for p in world["train_pairs"]}
    if arm == "SHUF":
        taught = data.shuffled_labels(task, world)
    W = np.zeros((len(m.kc_side), lb1.NBITS))
    held = world["held_pairs"]
    curve = []

    def code(p):
        a, _, v = layer.forward(traces[p])
        return lb1.topk_side(m, lb1.drive(m, v), side_n, False), a

    def held_acc():
        """(accuracy, balanced accuracy = mean per-answer recall) on held prompts."""
        hit = {}
        for p in held:
            x, _ = code(p)
            t = data.answer(task, *p)
            hit.setdefault(t, []).append(int(lb1.byte_of(x @ W) == t))
        allh = [h for v in hit.values() for h in v]
        return round(float(np.mean(allh)), 3), round(float(np.mean([np.mean(v) for v in hit.values()])), 3)

    def overlap():
        X = np.array([code(p)[0] for p in ALL])
        inter = X @ X.T
        n = X.sum(1)
        J = inter / (n[:, None] + n[None, :] - inter)
        return round(float(J[np.triu_indices(len(J), 1)].mean()), 3)

    def taught_acc():
        return float(np.mean([lb1.byte_of(code(p)[0] @ W) == taught[p] for p in world["train_pairs"]]))

    order = data.schedule(world, passes)
    n_pass = len(world["train_pairs"])
    for i, p in enumerate(order):
        x, a = code(p)
        y = lb1.bits_of(taught[p])
        mrg = x @ W
        e = (np.tanh(mrg / lb1.TEMP) - y) / 2
        k = lb1.strength(i)
        W -= lr * k * np.outer(x, e)
        if wd_r:
            W *= (1.0 - wd_r)
        if arm in ("LEARN", "SHUF"):
            eta = eta0 * min(1.0, i / eta_ramp)
            if eta > 0:
                layer.update(traces[p], a, e, eta, lam)
        if (i + 1) % (n_pass * every) == 0:
            curve.append((*held_acc(), round(taught_acc(), 3), overlap()))
    h, hb = held_acc()
    return {"held": h, "held_bal": hb, "taught": round(taught_acc(), 3), "overlap": overlap(), "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="D2,D3")
    ap.add_argument("--task", default="MOD")
    ap.add_argument("--arms", default="LEARN,FROZEN")
    ap.add_argument("--eta0", default="0.003")
    ap.add_argument("--lam", default="1e-3")
    ap.add_argument("--hidden", default="128")
    ap.add_argument("--passes", type=int, default=120)
    ap.add_argument("--lr", default="0.05")
    ap.add_argument("--wd_r", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--centre_tau", default="0")
    a = ap.parse_args()
    for w, arm, h, eta, lam, lr, ct in itertools.product(a.worlds.split(","), a.arms.split(","),
                                                     [int(v) for v in a.hidden.split(",")],
                                                     [float(v) for v in a.eta0.split(",")],
                                                     [float(v) for v in a.lam.split(",")],
                                                     [float(v) for v in a.lr.split(",")],
                                                     [float(v) for v in a.centre_tau.split(",")]):
        if arm == "FROZEN" and (eta != float(a.eta0.split(",")[0]) or lam != float(a.lam.split(",")[0])):
            continue
        r = run(data.make_world(w), a.task, arm, h, eta, lam, a.passes, lr, a.wd_r, every=a.every, centre_tau=ct)
        print(json.dumps({"world": w, "task": a.task, "arm": arm, "hidden": h, "eta0": eta, "lam": lam, "lr": lr,
                          "centre_tau": ct,
                          "wd_r": a.wd_r, **r}), flush=True)


if __name__ == "__main__":
    main()
