#!/usr/bin/env python3
"""PLAN_2.md, addition 1: readings on either side of the findings of classes.   python3 report2d.py <tag> ...
Never-shown additions after a read line, %: all kinds speaking; only the class kinds; only the plain kinds.  Then the kept pairs at each finding against the finding before."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for tag in sys.argv[1:]:
    d = json.load(open(os.path.join(HERE, "res", tag + ".json"))); print(f"{tag}: seed {d['seed']}, cpu {d['cpu']} s")
    for r in d["rows"]:
        f = lambda key, c: r[key][c]["after_add"]
        print(f"  {r['lines']:>6}: all kinds first {f('add', 'first'):.1f} second {f('add', 'second'):.1f} control {f('add', 'control'):.1f} | class kinds alone: first {f('add_class_kinds_alone', 'first'):.1f} second {f('add_class_kinds_alone', 'second'):.1f} control {f('add_class_kinds_alone', 'control'):.1f}"
              f" | plain kinds alone: first {f('add_plain_kinds_alone', 'first'):.1f} second {f('add_plain_kinds_alone', 'second'):.1f} control {f('add_plain_kinds_alone', 'control'):.1f} | class cells {r['class_cells']:,}")
    prev = None
    for fd in d["findings"]:
        cur = {(tuple(k["R"]), tuple(k["X"])): k for k in fd["kept"]}
        if prev is not None:
            same = [p for p in cur if p in prev]; regrouped = [p for p in same if (cur[p]["classes"], cur[p]["members"]) != (prev[p]["classes"], prev[p]["members"])]
            print(f"  finding at {fd['bytes_read'] // 10:>7,} lines: {len(cur)} pairs kept, {len(same)} of them kept at the finding before ({len(cur) - len(same)} new, born with nothing); of the {len(same)}, {len(regrouped)} with another number of classes or members")
        prev = cur
