#!/usr/bin/env python3
"""Three-digit sums (gate2.py result files s3_cls_seed*).   python3 report4.py <prefix> <seeds> [merge_tag]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); prefix = sys.argv[1]; seeds = sys.argv[2].split(","); agg = {}
for sd in seeds:
    d = json.load(open(os.path.join(HERE, "res", f"{prefix}_seed{sd}.json"))); rows = d["rows"]; ev = [r for r in rows if r["on_every"]]; cells = list(rows[0]["add"]); line = f"  seed {sd} ({rows[-1]['lines']} lines, cpu {d['cpu']} s, cells {rows[-1]['cells']:,}, parts used {rows[-1]['second_parts_used'][0]:,}):"
    for c in cells:
        last = rows[-1]["add"][c]["empty"]; l3 = float(np.mean([r["add"][c]["empty"] for r in ev[-3:]])); agg.setdefault((c, "last"), []).append(last); agg.setdefault((c, "last3"), []).append(l3); agg.setdefault((c, "place"), []).append(rows[-1]["add"][c]["empty_place"])
        line += f" {c} {last:.1f} (last3 {l3:.1f}) places {' '.join('%.1f' % v for v in rows[-1]['add'][c]['empty_place'])} |"
        assert rows[-1]["add"][c]["empty"] == rows[-1]["add"][c]["after_add"]
    if len(sys.argv) > 3:
        m = json.load(open(os.path.join(HERE, "..", "merge", "res", f"{sys.argv[3]}_seed{sd}.json"))); line += f" every reading of my first cell equal to nano/merge's stored file: {[x['add'] for x in m['rows']] == [x['add']['first']['empty'] for x in rows]}"
    print(line); print("    by lines: " + " ".join(f"{r['lines'] // 1000}k:{r['add']['first']['empty']:.1f}/{r['add']['second']['empty']:.1f}/{r['add']['control']['empty']:.1f}" for r in rows))
for (c, k), v in agg.items():
    if k == "place": print(f"  {c:8s} places at the end, median: " + " ".join(f"{x:.1f}" for x in np.median(np.array(v), 0)))
    else: print(f"  {c:8s} {k:6s}: " + ", ".join(f"{x:.1f}" for x in v) + f"; median {np.median(v):.1f}")
for k in ("last", "last3"):
    for c in ("second", "control"): g = np.array(agg[(c, k)]) - np.array(agg[("first", k)]); print(f"  {c} less first, {k}: " + ", ".join(f"{x:+.1f}" for x in g) + f"; median {np.median(g):+.1f}; difference of medians {np.median(agg[(c, k)]) - np.median(agg[('first', k)]):+.1f}")
