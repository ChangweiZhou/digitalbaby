"""G4: keep and reuse. The combined fly lives through addition, then multiplication (same 80 taught pairs; the 20
HELD pairs never shown in either operation). Each pass over the current task's taught records is followed by one
sleep. Replay arms: `reservoir` (old experience stays in the cache) vs `recent` (cache holds only the last 80
records, i.e. the current task: the no-replay control).
"""
import argparse
import json
import random
import time

import cls
import data


def run(world, seed, replay, rule, wd, ticks, passes, sleep_steps, mem_replay=0):
    t0 = time.process_time()
    d = data.addition(world)
    fly = cls.CLSFly(seed, slow_kwargs=dict(rule=rule, wd=wd, ticks=ticks), replay=replay,
                     cache_size=160 if replay == "reservoir" else 80, mem_replay=mem_replay)
    rng = random.Random(world * 7 + seed)
    log = []

    def evaluate(phase, p):
        row = dict(phase=phase, passes=p)
        for name in ("taught", "held", "mul_taught", "mul_held"):
            row[name] = fly.score(d[name])
        log.append(row)

    for phase, recs in (("addition", d["taught"]), ("multiplication", d["mul_taught"])):
        for p in range(1, passes + 1):
            order = list(range(len(recs)))
            rng.shuffle(order)
            for i in order:
                fly.wake(recs[i])
            fly.sleep(sleep_steps)
            if p in (passes // 2, passes):
                evaluate(phase, p)
    return dict(world=world, seed=seed, replay=replay, rule=rule, wd=wd, ticks=ticks, passes=passes,
                sleep_steps=sleep_steps, mem_replay=mem_replay, log=log, cpu_s=time.process_time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", choices=("reservoir", "recent"), default="reservoir")
    ap.add_argument("--rule", choices=("bptt", "eprop"), default="bptt")
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--passes", type=int, default=24)
    ap.add_argument("--sleep-steps", type=int, default=800)
    ap.add_argument("--mem-replay", type=int, default=0)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    r = run(a.world, a.seed, a.replay, a.rule, a.wd, a.ticks, a.passes, a.sleep_steps, a.mem_replay)
    with open(a.save, "w") as f:
        json.dump(r, f)
    last = {row["phase"]: row for row in r["log"]}
    print(json.dumps(dict(replay=r["replay"], rule=r["rule"], mem_replay=r["mem_replay"], cpu_s=round(r["cpu_s"]),
                          after_addition={k: last["addition"][k] for k in ("taught", "held")},
                          after_multiplication={k: last["multiplication"][k]
                                                for k in ("taught", "held", "mul_taught", "mul_held")})))
