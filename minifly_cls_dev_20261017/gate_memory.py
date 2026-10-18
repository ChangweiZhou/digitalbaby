"""G1 (and G2 at the memory level): can the native fast memory store taught facts through the byte stream, and keep
them over rest?

One run = one world x one memory configuration. Taught records are presented in shuffled passes; exact own-output
accuracy is scored on the taught set during training, then after 1 hour and after 1 day of rest. HELD addition facts
are scored too (a lookup memory is not expected to answer them).
"""
import argparse
import json
import random
import time

import data
import fastmem


def score(mem, recs):
    out = [mem.generate(p) for p, _ in recs]
    return (sum(g == a for (g, _), (_, a) in zip(out, recs)) / len(recs),
            [g.decode("latin1") for g, _ in out], sum(m for _, m in out) / len(out))


def run(world, front, homeo, fresh_adapt, teach, passes, task):
    t0 = time.process_time()
    d = data.addition(world)
    recs = d["taught"] if task == "addition" else data.order_contrasts()
    mem = fastmem.FastMemory(front=front, teach=teach, fresh_adapt=fresh_adapt, homeo=homeo)
    rng = random.Random(world)
    curve = []
    for ep in range(1, passes + 1):
        order = list(range(len(recs)))
        rng.shuffle(order)
        for i in order:
            mem.train(*recs[i])
        if ep in (1, 2, 4, 8, 12, 16, 24, 32) or ep == passes:
            curve.append((ep, score(mem, recs)[0]))
    taught_acc, taught_out, margin = score(mem, recs)
    result = dict(world=world, front=front, homeo=homeo, fresh_adapt=fresh_adapt, teach=teach, passes=passes,
                  task=task, records=mem.records, curve=curve, taught_final=taught_acc, taught_margin=margin,
                  taught_answers=taught_out)
    if task == "addition":
        result["held_final"] = score(mem, d["held"])[0]
    mem.rest(3600.)
    result["taught_after_1h"] = score(mem, recs)[0]
    mem.rest(86400. - 3600.)
    result["taught_after_1d"], result["taught_answers_after_1d"], _ = score(mem, recs)
    result["cpu_s"] = time.process_time() - t0
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--front", choices=("perm", "fe0"), default="perm")
    ap.add_argument("--homeo", type=int, default=1)
    ap.add_argument("--fresh-adapt", type=int, default=1)
    ap.add_argument("--teach", choices=("logged", "signed", "delta"), default="delta")
    ap.add_argument("--passes", type=int, default=24)
    ap.add_argument("--task", choices=("addition", "order"), default="addition")
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    r = run(a.world, a.front, bool(a.homeo), bool(a.fresh_adapt), a.teach, a.passes, a.task)
    with open(a.save, "w") as f:
        json.dump(r, f)
    print(json.dumps({k: r[k] for k in ("task", "front", "homeo", "fresh_adapt", "teach", "curve", "taught_final",
                                         "taught_after_1h", "taught_after_1d", "cpu_s")}))
