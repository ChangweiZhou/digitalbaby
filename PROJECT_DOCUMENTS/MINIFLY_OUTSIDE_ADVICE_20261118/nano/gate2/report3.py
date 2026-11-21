#!/usr/bin/env python3
"""A life of two tasks (gate2.py result files).   python3 report3.py <prefix> <seed,seed,...>
Never-shown additions: after a read addition line / after a read subtraction line / with an empty past.  Never-shown subtractions on the clean pairs: after a subtraction line / after an addition line / empty past.
Kept (PLAN_3.md): additions after a subtraction line, mean of the last three readings of the second task, over additions after an addition line, mean of the last three readings of the first."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); prefix = sys.argv[1]; seeds = sys.argv[2].split(","); agg = {}
for sd in seeds:
    d = json.load(open(os.path.join(HERE, "res", f"{prefix}_seed{sd}.json"))); rows = d["rows"]; cells = list(rows[0]["add"]); print(f"{prefix} seed {sd}: {d['mode']}, plan {d['plan']}, {d['clean_subtractions']} clean subtractions, cpu {d.get('cpu')} s, {'ended' if 'second_parts' in d else 'NOT ENDED'}")
    for r in rows:
        if r["phase"] == 0 and not (r["lines"] >= d["plan"][0][1] - 2 * (5000 if d["plan"][0][1] <= 40000 else 50000)): continue
        print(f"  {r['name']} {r['lines']:>6}: add " + " | ".join(f"{c} {r['add'][c]['after_add']:.1f}/{r['add'][c]['after_sub']:.1f}/{r['add'][c]['empty']:.1f}" for c in cells)
              + " || sub clean " + " | ".join(f"{c} {r['sub_clean'][c]['after_sub']:.1f}/{r['sub_clean'][c]['after_add']:.1f}/{r['sub_clean'][c]['empty']:.1f}" for c in cells) + f" || cells {r['cells']:,}")
    p0 = [r for r in rows if r["phase"] == 0 and r["on_every"]][-3:]; p1 = [r for r in rows if r["phase"] == 1 and r["on_every"]][-3:]; p1all = [r for r in rows if r["phase"] == 1]
    for c in cells:
        m = lambda rs, what, key: float(np.mean([r[what][c][key] for r in rs])); b = m(p0, "add", "after_add"); e = m(p1, "add", "after_sub")
        v = dict(before=b, end=e, kept=100 * e / b, kept_after_add=100 * m(p1, "add", "after_add") / b, kept_empty=100 * m(p1, "add", "empty") / m(p0, "add", "empty"), end_after_add=m(p1, "add", "after_add"), end_empty=m(p1, "add", "empty"),
                 sub=m(p1, "sub_clean", "after_sub"), sub_after_add=m(p1, "sub_clean", "after_add"), sub_empty=m(p1, "sub_clean", "empty"), lowest=min(r["add"][c]["after_sub"] for r in p1all), lowest_at=min(p1all, key=lambda r: r["add"][c]["after_sub"])["lines"])
        for k, x in v.items(): agg.setdefault((c, k), []).append(x)
        print(f"  {c}: addition before {b:.1f}, at the end {e:.1f} (after an addition line {v['end_after_add']:.1f}, empty past {v['end_empty']:.1f}); kept {v['kept']:.1f}% ({v['kept_after_add']:.1f}%, {v['kept_empty']:.1f}%); lowest {v['lowest']:.1f} at {v['lowest_at']} lines; subtraction clean {v['sub']:.1f} (after an addition line {v['sub_after_add']:.1f}, empty {v['sub_empty']:.1f})")
print(f"{prefix}: over seeds {seeds}")
for (c, k), v in agg.items():
    if k != "lowest_at": print(f"  {c:8s} {k:15s}: " + ", ".join(f"{x:.1f}" for x in v) + f"; median {np.median(v):.1f}")
    else: print(f"  {c:8s} {k:15s}: {v}")
