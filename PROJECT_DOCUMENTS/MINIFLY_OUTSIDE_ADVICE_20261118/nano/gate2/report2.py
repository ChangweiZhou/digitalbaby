#!/usr/bin/env python3
"""Readings along a run (gate2.py result files).   python3 report2.py <tag> [<tag> ...]
Never-shown additions, whole answer right, %: after a read line (five draws) / with an empty past; then by place after a read line."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for tag in sys.argv[1:]:
    d = json.load(open(os.path.join(HERE, "res", tag + ".json"))); rows = d["rows"]; cells = list(rows[0]["add"]); print(f"{tag}: seed {d['seed']}, {d['mode']}, plan {d['plan']}, cpu {d.get('cpu')} s, {'ended' if 'second_parts' in d else 'NOT ENDED'}")
    for r in rows:
        print(f"  {r['name']} {r['lines']:>6}: " + " | ".join(f"{c} {r['add'][c]['after_add']:.1f} / {r['add'][c]['empty']:.1f}" for c in cells) + " | places " + " ; ".join(f"{c} " + " ".join(f"{v:.1f}" for v in r['add'][c]['after_add_place']) for c in cells[:2])
              + f" | cells {r['cells']:,} class cells {r['class_cells']:,} parts used {r['second_parts_used'][0]:,}")
    ev = [r for r in rows if r["on_every"] and r["phase"] == 0]; r40 = next((r for r in rows if r["lines"] == 40000 and r["phase"] == 0), None)
    for c in cells:
        l3 = np.mean([r["add"][c]["after_add"] for r in ev[-3:]]); l3e = np.mean([r["add"][c]["empty"] for r in ev[-3:]])
        print(f"  {c}: last {ev[-1]['add'][c]['after_add']:.1f} / {ev[-1]['add'][c]['empty']:.1f}; mean of the last three {l3:.1f} / {l3e:.1f}" + (f"; at 40,000 lines {r40['add'][c]['after_add']:.1f} / {r40['add'][c]['empty']:.1f}; lowest reading after 40,000 lines {min(r['add'][c]['after_add'] for r in rows if r['lines'] >= 40000 and r['phase'] == 0):.1f}" if r40 else ""))
