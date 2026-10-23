"""Aggregate results/{memory,slow,cls,crosstask} into results/SUMMARY.json and print the gate tables."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def load(sub):
    return [json.loads(p.read_text()) for p in sorted((R / sub).glob("*.json"))]


def rng(xs):
    return [round(float(min(xs)), 4), round(float(max(xs)), 4)]


def memory():
    groups = defaultdict(list)
    for r in load("memory"):
        cfg = (r["task"], r["front"], r["homeo"], r["fresh_adapt"], r["teach"])
        groups[cfg].append(r)
    out = []
    for (task, front, homeo, fresh, teach), rs in sorted(groups.items()):
        out.append(dict(task=task, front=front, homeo=homeo, fresh_adapt=fresh, teach=teach, n=len(rs),
                        taught=rng([r["taught_final"] for r in rs]), after_1h=rng([r["taught_after_1h"] for r in rs]),
                        after_1d=rng([r["taught_after_1d"] for r in rs]),
                        held=rng([r.get("held_final", 0) for r in rs]),
                        curve_mean={str(e): float(np.mean([dict(r["curve"])[e] for r in rs])) for e, _ in rs[0]["curve"]}))
    return out


def slow():
    groups = defaultdict(list)
    for r in load("slow"):
        groups[(r["rule"], r["wd"], r["ticks"])].append(r)
    return [dict(rule=k[0], wd=k[1], ticks=k[2], n=len(rs),
                 taught=float(np.mean([r["taught_final"] for r in rs])),
                 held_mean=float(np.mean([r["held_final"] for r in rs])), held=rng([r["held_final"] for r in rs]),
                 held_best_mean=float(np.mean([r["held_best"] for r in rs])),
                 commutative_agreement=float(np.mean([r["commutative_agreement"] for r in rs])),
                 probe_order=float(np.mean([r["probe_order"] for r in rs])))
            for k, rs in sorted(groups.items())]


def combined():
    groups = defaultdict(list)
    for r in load("cls"):
        groups[(r["replay"], r.get("mem_replay", 0))].append(r)
    out = []
    for (replay, mem_replay), rs in sorted(groups.items()):
        row = dict(replay=replay, mem_replay=mem_replay, n=len(rs))
        for phase in ("addition", "multiplication"):
            last = [next(x for x in reversed(r["log"]) if x["phase"] == phase) for r in rs]
            row[phase] = {s: {k: rng([x[s][k] for x in last]) for k in ("combined", "mem", "slow")}
                          for s in ("taught", "held", "mul_taught", "mul_held")}
        out.append(row)
    return out


def main():
    doc = dict(schema="MINIFLY_CLS_DEV_SUMMARY_V1", memory_gate=memory(), slow_gate=slow(), combined_gate=combined(),
               crosstask=load("crosstask"))
    (R / "SUMMARY.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("G1/G2 memory gate (24 passes): taught / after 1 h / after 1 d")
    for m in doc["memory_gate"]:
        print(f"  {m['task']:8} front={m['front']:4} homeo={m['homeo']!s:5} fresh={m['fresh_adapt']!s:5} "
              f"teach={m['teach']:6} n={m['n']} {m['taught']} {m['after_1h']} {m['after_1d']}")
    print("G3 slow core: taught, unseen mean [range], best, comm, order")
    for s in doc["slow_gate"]:
        print(f"  {s['rule']:5} wd={s['wd']:g} ticks={s['ticks']} n={s['n']} {s['taught']:.2f} {s['held_mean']:.2f} "
              f"{s['held']} {s['held_best_mean']:.2f} {s['commutative_agreement']:.2f} {s['probe_order']:.2f}")
    print("G4 combined fly: after multiplication (combined/mem/slow ranges)")
    for c in doc["combined_gate"]:
        a, m = c["multiplication"]["taught"], c["multiplication"]["mul_taught"]
        print(f"  replay={c['replay']:9} mem_replay={c['mem_replay']:2} addition kept {a}  multiplication {m}")


if __name__ == "__main__":
    main()
