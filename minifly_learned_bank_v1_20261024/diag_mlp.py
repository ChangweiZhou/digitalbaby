"""Cycle-2 attribution diagnostic (evaluator-side; no native store; never a candidate).

Same FE-PERM traces and splits. A one-hidden-layer network p -> relu(W1 p + b1) -> linear readout to 8 answer bits.
  BP : exact gradient for both layers (tells whether the data are sufficient at all for this input code)
  DFA: W1 by the three-factor rule with fixed random feedback G (the candidate's rule), readout by the delta rule
Tasks: MOD (last digit of a+b), CARRY (a+b>=10 -> '1'/'0'), CMP (a>b -> '1'/'0'), COPY (a).
"""
import argparse
import json

import numpy as np

import data
import lb1

ALL = [(a, b) for a in range(10) for b in range(10)]


def ans(task, a, b):
    if task == "CARRY":
        return ord("1" if a + b >= 10 else "0")
    if task == "CMP":
        return ord("1" if a > b else "0")
    if task.startswith("MOD") and task != "MOD":
        return ord(str((a + b) % int(task[3:])))
    return data.answer(task, a, b)


def run(world, task, mode, hidden, lr, wd, epochs, seed=0):
    rng = np.random.default_rng(seed)
    P = {p: lb1.trace_for(data.prompt(*p)) for p in ALL}
    tr, he = world["train_pairs"], world["held_pairs"]
    if task == "CMP":   # order matters: split by ordered pairs, not orbits
        perm = np.random.default_rng(data.seed_of("cmp:" + world["name"])).permutation(100)
        he = [ALL[i] for i in perm[:26]]
        tr = [ALL[i] for i in perm[26:]]
    X = np.array([P[p] for p in tr])
    Y = np.array([lb1.bits_of(ans(task, *p)) for p in tr])
    Xh = np.array([P[p] for p in he])
    yh = [ans(task, *p) for p in he]
    W1 = rng.normal(0, 1 / np.sqrt(88), (hidden, 88))
    b1 = np.zeros(hidden)
    W2 = np.zeros((lb1.NBITS, hidden))
    G = rng.normal(0, 1 / np.sqrt(lb1.NBITS), (hidden, lb1.NBITS))
    for ep in range(epochs):
        for i in rng.permutation(len(X)):
            x, y = X[i], Y[i]
            a = W1 @ x + b1
            h = np.maximum(a, 0)
            m = W2 @ h
            e = (np.tanh(m / lb1.TEMP) - y) / 2
            if mode == "BP":
                g = (1 - np.tanh(m / lb1.TEMP) ** 2) / lb1.TEMP * e
                d = (W2.T @ g) * (a > 0)
            elif mode == "FROZEN":
                g = e
                d = np.zeros(hidden)
            else:
                g = e
                d = (G @ e) * (a > 0)
            W2 -= lr * (np.outer(g, h) + wd * W2)
            if mode != "FROZEN":
                W1 -= lr * (np.outer(d, x) + wd * W1)
                b1 -= lr * d
    def acc(Xs, ys):
        out = []
        for x, yy in zip(Xs, ys):
            out.append(lb1.byte_of(W2 @ np.maximum(W1 @ x + b1, 0)) == yy)
        return float(np.mean(out))
    return {"held": round(acc(Xh, yh), 3), "taught": round(acc(X, [ans(task, *p) for p in tr]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="D2,D3")
    ap.add_argument("--tasks", default="MOD,CARRY,CMP,COPY")
    ap.add_argument("--modes", default="BP,DFA")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    for t in a.tasks.split(","):
        for mode in a.modes.split(","):
            rows = [run(data.make_world(w), t, mode, a.hidden, a.lr, a.wd, a.epochs) for w in a.worlds.split(",")]
            print(json.dumps({"task": t, "mode": mode, "lr": a.lr, "wd": a.wd, "epochs": a.epochs,
                              "held": [r["held"] for r in rows], "taught": [r["taught"] for r in rows]}), flush=True)


if __name__ == "__main__":
    main()
