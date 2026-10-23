"""Diagnostic: why does the fast memory lose addition when multiplication is learned (G4)?

Memory only (no slow core): 24 passes of the 80 taught addition records, then 24 passes of the same pairs as
products. Measures same-pair key overlap (`a+b=` vs `a*b=`, first answer position) and taught accuracy of both
tasks, with one-sided homeostasis on or off.
"""
import argparse
import json
import random
import time

import numpy as np

import data
import fastmem


def keys(mem, recs):
    out = []
    for p, _ in recs:
        v = mem.clone()
        v._prompt(p)
        out.append(v._code())
    return np.array(out)


def main(world, homeo, passes):
    t0 = time.process_time()
    d = data.addition(world)
    mem = fastmem.FastMemory(homeo=homeo)
    rng = random.Random(world)
    res = dict(world=world, homeo=homeo, passes=passes)
    for phase, recs in (("addition", d["taught"]), ("multiplication", d["mul_taught"])):
        for _ in range(passes):
            order = list(range(len(recs)))
            rng.shuffle(order)
            for i in order:
                mem.train(*recs[i])
        ka, km = keys(mem, d["taught"]), keys(mem, d["mul_taught"])
        res[phase] = dict(
            add_taught=float(np.mean([mem.generate(p)[0] == a for p, a in d["taught"]])),
            mul_taught=float(np.mean([mem.generate(p)[0] == a for p, a in d["mul_taught"]])),
            same_pair_key_overlap=float(((ka * km).sum(1) / ((ka + km) > 0).sum(1)).mean()))
    res["cpu_s"] = time.process_time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=data.WORLDS[0])
    ap.add_argument("--homeo", type=int, default=1)
    ap.add_argument("--passes", type=int, default=24)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    r = main(a.world, bool(a.homeo), a.passes)
    with open(a.save, "w") as f:
        json.dump(r, f)
    print(json.dumps(r))
