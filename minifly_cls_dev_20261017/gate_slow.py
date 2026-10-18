"""G3: does the slow core answer never-taught sums? One run = world x seed x learning rule x downscaling x ticks.

Full-batch replay of the 80 taught addition records (as the diagnosis' batch regime, so results are comparable with
its GRU reference: 6% unseen at wd 0, 44% at wd 1). HELD = 10 commutative orbits never shown in either order.
"""
import argparse
import json
import time

import data
import slowcore


def value(s):
    return int(s) if s.isdigit() and (s == "0" or not s.startswith("0")) else None


def run(world, seed, rule, wd, ticks, steps, log_every, feedback):
    t0 = time.process_time()
    d = data.addition(world)
    core = slowcore.SlowCore(seed, ticks=ticks, rule=rule, wd=wd, feedback=feedback)
    curve = []
    for step in range(1, steps + 1):
        loss = core.step(d["taught"])
        if step % log_every == 0 or step == steps:
            curve.append((step, core.accuracy(d["taught"]), core.accuracy(d["held"]), loss))
    held = {f"{a},{b}": core.generate(p)[0].decode("latin1") for (a, b), (p, _) in zip(d["held_pairs"], d["held"])}
    errs = [value(g) - (int(k.split(",")[0]) + int(k.split(",")[1])) for k, g in held.items()
            if value(g) is not None]
    wrong = [e for e in errs if e != 0]
    comm = [held[f"{a},{b}"] == held[f"{b},{a}"] for a, b in d["held_pairs"] if a < b]
    return dict(world=world, seed=seed, rule=rule, feedback=feedback, wd=wd, ticks=ticks, steps=steps, curve=curve,
                taught_final=curve[-1][1], held_final=curve[-1][2], held_best=max(c[2] for c in curve),
                held_answers=held, held_wrong=len([g for g in held.values()]) - errs.count(0),
                held_off_by_one=sum(abs(e) == 1 for e in wrong), commutative_agreement=sum(comm) / len(comm),
                probe_order=core.probe_order(), cpu_s=time.process_time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", choices=("bptt", "eprop"), default="bptt")
    ap.add_argument("--feedback", choices=("symmetric", "random"), default="symmetric")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    r = run(a.world, a.seed, a.rule, a.wd, a.ticks, a.steps, a.log_every, a.feedback)
    with open(a.save, "w") as f:
        json.dump(r, f)
    print(json.dumps({k: r[k] for k in ("rule", "wd", "ticks", "taught_final", "held_final", "held_best",
                                         "held_off_by_one", "commutative_agreement", "probe_order", "cpu_s")}))
