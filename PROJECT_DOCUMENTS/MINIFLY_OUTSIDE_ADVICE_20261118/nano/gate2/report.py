#!/usr/bin/env python3
"""Tables from the result files of gate2.py.   python3 report.py s40_cls 73129,93683,...  [merge_tag]
Never-shown additions, whole answer right, %.  'after': after a read line (mean of five draws); 'empty': with an empty past.  Last reading and mean of the last three."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def load(tag): return json.load(open(os.path.join(HERE, "res", tag + ".json")))
def last3(rows, cell, key, what="add"):
    r = [x[what][cell][key] for x in rows if x["on_every"]]; return float(np.mean(r[-3:]))
def table(prefix, seeds, merge_tag=None):
    cols = {}; print(f"{prefix}: seeds {seeds}")
    for sd in seeds:
        d = load(f"{prefix}_seed{sd}"); rows = [r for r in d["rows"] if r["phase"] == 0]; r = rows[-1]; line = f"  seed {sd} ({r['lines']} lines, cpu {d['cpu']} s):"
        for c in r["add"]:
            for key in ("after_add", "empty"):
                cols.setdefault((c, key, "last"), []).append(r["add"][c][key]); cols.setdefault((c, key, "last3"), []).append(last3(rows, c, key))
            line += f" {c} after {r['add'][c]['after_add']:.1f} (last3 {last3(rows, c, 'after_add'):.1f}) empty {r['add'][c]['empty']:.1f} (last3 {last3(rows, c, 'empty'):.1f}) |"
        if merge_tag:
            m = json.load(open(os.path.join(HERE, "..", "merge", "res", f"{merge_tag}_seed{sd}.json"))); same = m["last"] == r["add"]["first"]["empty"] and [x["add"] for x in m["rows"]] == [x["add"]["first"]["empty"] for x in rows]
            line += f" nano/merge's stored file: last {m['last']}, every reading the same as my first cell: {same}"
        line += f" parts used {r['second_parts_used']}"; print(line)
    for (c, key, w), v in cols.items(): print(f"  {c:8s} {key:10s} {w:6s}: " + ", ".join(f"{x:.1f}" for x in v) + f"; median {np.median(v):.1f}, mean {np.mean(v):.1f}")
    for key in ("after_add", "empty"):
        for w in ("last", "last3"):
            f = np.array(cols[("first", key, w)])
            for c in ("second", "control", "second_half", "control_half"):
                if (c, key, w) in cols: g = np.array(cols[(c, key, w)]) - f; print(f"  {c} less first, {key}, {w}: " + ", ".join(f"{x:+.1f}" for x in g) + f"; median {np.median(g):+.1f}; difference of medians {np.median(cols[(c, key, w)]) - np.median(f):+.1f}")
if __name__ == "__main__": table(sys.argv[1], sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else None)
