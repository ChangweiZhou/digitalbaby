"""Cycle-2 diagnostic: how large do native learned-bank margins get, and how large is the residual that drives the
layer once taught prompts are memorised? (Development world only; FROZEN arm; read-only probes.)

python diag_margins.py --world D2 --task PAR --passes 30 --every 5
"""
import argparse
import json

import numpy as np

import data
import lb1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="D2")
    ap.add_argument("--task", default="PAR")
    ap.add_argument("--arm", default="FROZEN")
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()
    world = data.make_world(a.world)
    model = lb1.Model(a.arm, world["model_seed"])
    order = data.schedule(world, a.passes)
    per = len(world["train_pairs"])
    for i, (x, y) in enumerate(order):
        model.train(data.prompt(x, y), data.answer(a.task, x, y), data.answer(a.task, x, y))
        if (i + 1) % (per * a.every) == 0:
            mL, eL, ok = [], [], []
            for p in world["train_pairs"]:
                t = data.answer(a.task, *p)
                r = model.probe(data.prompt(*p), t)
                mL.append(r["mL"])
                eL.append((np.tanh(r["mL"] / lb1.TEMP) - lb1.bits_of(t)) / 2)
                ok.append(r["L"] == t)
            mL, eL = np.abs(np.array(mL)), np.abs(np.array(eL))
            print(json.dumps({"pass": (i + 1) // per, "taught_L": round(float(np.mean(ok)), 3),
                              "abs_margin_by_bit_median": np.round(np.median(mL, 0), 3).tolist(),
                              "abs_residual_by_bit_mean": np.round(eL.mean(0), 3).tolist()}), flush=True)


if __name__ == "__main__":
    main()
