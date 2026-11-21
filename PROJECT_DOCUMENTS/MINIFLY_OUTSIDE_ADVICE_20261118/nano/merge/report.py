#!/usr/bin/env python3
"""Tables from the result files of merge.py.
    python3 report.py dev                 the development grid on both tasks
    python3 report.py sums <prefix>...    for each prefix, res/<prefix>_seed*.json: each seed, the median, sizes, what the classes are
    python3 report.py life <tag>...       the life on prose, one row a run, and how the cells grow"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = lambda t: json.load(open(os.path.join(HERE, "res", t + ".json")))
SEEDS = (73129, 93683, 116089, 406364, 545353)

def units_pair(o):
    """The kept pair at the last finding whose X is delays 3 and 6 inside R = 3, 4, 6, 7 (the two units digits against the two tens digits where the tens digit is named), and the pair X = 4, 7."""
    out = {}
    for r in o["finds"][-1]["kept"]:
        if r["R"] == [3, 4, 6, 7] and r["X"] in ([3, 6], [4, 7]): out[tuple(r["X"])] = r["carry"]
    return out

def dev():
    print("development seed 1, sums at 40,000 lines (mean of the last three readings; last; tens digit at the last), and prose (last twentieth of the stream, nats a byte)")
    print("setting | sums last3, last | cells, without the silent | CPU s | prose nats | cells, without | CPU s")
    for t in ["none"] + [f"P{P}_eJ{e}{r}" for r in ("", "_rep1") for e in ("0.02", "0.5") for P in (8, 32)]:
        row = [t]
        for task in ("sums", "prose"):
            f = os.path.join(HERE, "res", f"dev_{task}_{t}.json")
            if not os.path.exists(f): row.append("not run"); continue
            o = json.load(open(f)); z = o["sizes"] if task == "sums" else o
            row.append((f"{o['last3']:.1f}, {o['last']:.1f}" if task == "sums" else f"{o['nats']:.4f}") + f" | {z['cells']:,}, {z['cells_with_merged_withdrawn']:,} | {o['cpu']:.0f}")
        print(" | ".join(row))

def sums(prefixes):
    for pre in prefixes:
        runs = [(sd, R(f"{pre}_seed{sd}")) for sd in SEEDS if os.path.exists(os.path.join(HERE, "res", f"{pre}_seed{sd}.json"))]
        if not runs: print(pre, "no runs"); continue
        l3 = [o["last3"] for _, o in runs]; la = [o["last"] for _, o in runs]; pl = np.median([o["place_last"] for _, o in runs], 0); z = [o["sizes"] for _, o in runs]
        print(f"{pre}: seeds {[s for s, _ in runs]}")
        print(f"  last three: {', '.join(f'{v:.1f}' for v in l3)}; median {np.median(l3):.1f} | last: {', '.join(f'{v:.1f}' for v in la)}; median {np.median(la):.1f} | hundreds, tens, units (last, median): {pl[0]:.1f}, {pl[1]:.1f}, {pl[2]:.1f}")
        marks = sorted({r["lines"] for _, o in runs for r in o["rows"]}); print("  by lines (median): " + " ".join(f"{m // 1000}k:{np.median([r['add'] for _, o in runs for r in o['rows'] if r['lines'] == m]):.1f}" for m in marks))
        print(f"  cells {np.median([q['cells'] for q in z]):,.0f} (base {np.median([q['base_cells'] for q in z]):,.0f}, class {np.median([q['class_cells'] for q in z]):,.0f}); without the R-cells of merged classes {np.median([q['cells_with_merged_withdrawn'] for q in z]):,.0f}; counts above zero {np.median([q['counts_above_zero'] for q in z]):,.0f}; CPU s {np.median([o['cpu'] for _, o in runs]):.0f} (search {np.median([o['cpu_finding'] for _, o in runs]):.0f})")
        if "with_merged_withdrawn" in runs[0][1]: print("  scored with the R-cells of merged classes silent: " + ", ".join(f"{o['with_merged_withdrawn']['add']:.1f}" for _, o in runs))
        if "finds" in runs[0][1]:
            for sd, o in runs: print(f"  seed {sd}: kept pairs at the last finding {len(o['finds'][-1]['kept'])}; inside R = (3, 4, 6, 7): " + "; ".join(f"X = {k}: {v['digit_members']} pairs of digits, lying with the majority of their class by the carry {v['purity']}%, classes (size, of them carrying) {v['classes']}" for k, v in units_pair(o).items()) + f" | share of all trust held by class kinds {o['trust'].get('share_of_trust_in_class_kinds')}")

def life(tags):
    keys = [k for k in R(tags[0])["scores"]]; print("nats a byte; in brackets: scored with the R-cells of merged classes silent, without having learned so")
    print("run | " + " | ".join(keys) + " | cells at the end, without the silent | CPU s (search)")
    for t in tags:
        o = R(t); sc = o["scores"]; f = lambda k: (f"{sc[k]['nats']:.4f}" + (f" ({sc[k]['nats_with_merged_withdrawn']:.4f})" if "nats_with_merged_withdrawn" in sc[k] else "")) if k in sc else "not scored"
        print(" | ".join([t] + [f(k) for k in keys] + [f"{o['sizes']['cells']:,}, {o['sizes']['cells_with_merged_withdrawn']:,}", f"{o['cpu']:.0f} ({o['cpu_finding']:.0f})"]))
    base = R(tags[0])["scores"]
    for t in tags[1:]: print(f"{t} less {tags[0]}: " + ", ".join(f"{k} {R(t)['scores'][k]['nats'] - base[k]['nats']:+.4f}" for k in keys if k in R(t)["scores"]))
    for t in tags:
        o = R(t); g = o["growth"]; b = np.array([q["bytes_read"] for q in g], float); sel = (b >= 125000) & (b <= 1000000)
        ex = lambda key: float(np.polyfit(np.log(b[sel]), np.log(np.array([q[key] for q in g], float)[sel]), 1)[0])
        print(f"{t}: cells grow as the bytes read to the power {ex('cells'):.2f} (base cells {ex('base_cells'):.2f}; without the R-cells of merged classes {ex('cells_with_merged_withdrawn'):.2f}), fitted from 125,000 to 1,000,000 bytes of the test text")
        print("   bytes read: cells / without: " + "; ".join(f"{q['bytes_read']:,}: {q['cells']:,} / {q['cells_with_merged_withdrawn']:,}" for q in g[::2] + g[-1:]))

if __name__ == "__main__":
    dict(dev=lambda a: dev(), sums=sums, life=life)[sys.argv[1]](sys.argv[2:])
