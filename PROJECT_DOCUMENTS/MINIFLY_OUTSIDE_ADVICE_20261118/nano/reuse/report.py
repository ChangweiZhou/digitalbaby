#!/usr/bin/env python3
"""Tables from res/*.json.   python3 report.py arm [arm ...]     an arm is the tag without its seed, for instance a_reuse"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
def need(rows, name, level):
    for r in rows:
        if r["lines"] > 0 and r["score"][name] >= level: return r["lines"]
    return None
def fmt(v): return ">max" if v is None else f"{v / 1000:g}k"
def med_need(vs, mx):
    w = sorted(mx * 10 if v is None else v for v in vs); m = w[len(w) // 2]; return None if m > mx else m
def main(arms):
    for arm in arms:
        outs = [(s, json.load(open(os.path.join(HERE, "res", f"{arm}_s{s}.json")))) for s in SEEDS if os.path.exists(os.path.join(HERE, "res", f"{arm}_s{s}.json"))]
        if not outs: print(arm, ": no files"); continue
        print(f"{arm}: seeds {[s for s, _ in outs]}; phases {outs[0][1]['phases']}; mode {outs[0][1]['mode']}")
        nph = len(outs[0][1]["phases"])
        for i, (name, lines) in enumerate(outs[0][1]["phases"]):
            per = [[r for r in o["rows"] if r["phase"] == i] for _, o in outs]; marks = [r["lines"] for r in per[0]]
            print(f"  phase {i}, {name}, by lines (median of seeds): " + " ".join(f"{m / 1000:g}k:{np.median([p[j]['score'][name] for p in per]):.1f}" for j, m in enumerate(marks)))
            ev = [[r["score"][name] for r in p if r["lines"] % 5000 == 0 and r["lines"] > 0] for p in per]
            print("    last three readings (every 5,000 lines), by seed: " + ", ".join(f"{np.mean(e[-3:]):.1f}" for e in ev) + f"; median {np.median([np.mean(e[-3:]) for e in ev]):.1f} | last: " + ", ".join(f"{e[-1]:.1f}" for e in ev) + f"; median {np.median([e[-1] for e in ev]):.1f}")
            print("    places at the last reading (median): " + ", ".join(f"{np.median([p[-1]['place'][name][q] for p in per]):.1f}" for q in range(len(per[0][-1]["place"][name]))))
            for lv in (70, 80, 90):
                nd = [need(p, name, lv) for p in per]; print(f"    lines to {lv}%: " + ", ".join(fmt(v) for v in nd) + f"; median {fmt(med_need(nd, lines))}")
            if i == nph - 1 and i > 0:
                for j in range(i):
                    nm = outs[0][1]["phases"][j][0] + f"@{j}"; print(f"    the earlier task {outs[0][1]['phases'][j][0]} at the end (median): {np.median([p[-1]['score'][nm] for p in per]):.1f}")
        print(f"  cells {int(np.median([o['sizes']['cells'] for _, o in outs])):,}; CPU s {np.median([o['cpu'] for _, o in outs]):.0f} (judging {np.median([o['cpu_judging'] for _, o in outs]):.0f})")
        if outs[0][1]["mode"] != "none":
            for s, o in outs:
                b0 = sum(l * len(o["rows"][0]["place"]) for _, l in o["phases"][:-1]) if nph > 1 else 0   # not exact bytes; findings are listed by bytes read
                ph = [r for r in o["rows"] if r["phase"] == nph - 1]; start = (o["rows"][[r["phase"] for r in o["rows"]].index(nph - 1)]["bytes_read"]) if nph > 1 else 0
                early = [f for f in o["finds"] if start <= f["bytes_read"] < start + 45000]; fin = o["finds"][-1]
                def show(f): return f"{f['above_chance']} above chance; largest: " + "; ".join(f"{tuple(k['store'])}->{tuple(k['places'])} {k['gain']:.0f}/{k['chance']:.0f}" for k in f["kept"][:5])
                print(f"    seed {s}: first blocks of the last phase: " + " || ".join(f"at byte {f['bytes_read'] - start}: {show(f)}" for f in early[:3]))
                print(f"       readings at the end by largest trust: " + "; ".join(f"{tuple(r['store'])}->{tuple(r['places'])} {r['largest_trust']:.2f}" for r in sorted(o["readings_at_end"], key=lambda r: -r["largest_trust"])[:6]))
        print()
if __name__ == "__main__": main(sys.argv[1:])
