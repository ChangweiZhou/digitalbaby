#!/usr/bin/env python3
"""Tables from res/*.json.   python3 report.py sums_<tag> life_<tag> ...   (file names under res, without .json)"""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for tag in sys.argv[1:]:
    r = json.load(open(os.path.join(HERE, "res", tag + ".json")))
    if r["task"] == "sums":
        c = r["cells"]; f = c["first"]; s2 = c.get("second", f); print(f"\n== {tag}: never-shown sums, whole answer, %; seed {r['seed']}; cells {r['sizes']['cells']}; CPU s {r['cpu']}")
        print(f"   {'cell':22s} after a read line: last3  last | empty past: last3  last | tens after | gain over first (last3 after) | share of second's gain | parts used / held")
        for n, v in c.items():
            g = v["last3_after"] - f["last3_after"]; g2 = s2["last3_after"] - f["last3_after"]
            print(f"   {n:22s} {v['last3_after']:6.1f} {v['last_after']:6.1f} | {v['last3_empty']:6.1f} {v['last_empty']:6.1f} | {v['place_after'][1]:6.1f} | {g:+6.1f} | {g / g2 if g2 else 0:5.2f} | {v['parts_used']} / {v['parts']}")
    else:
        sc = r["scores"]; names = r["names"]; books = [k for k in sc if k not in ("A", "A_at_the_end")]; i2 = names.index("second") if "second" in names else 0
        print(f"\n== {tag}: prose, nats a byte; kinds {r['kinds']}; cells {r['cells']}; CPU s {r['cpu']}"); print(f"   {'cell':22s} test text's last tenth | " + " ".join(b[:10] for b in books) + " | mean gain over first on the books | share of second's | test text at the end of the life (kept) | parts used / held")
        g2 = np.mean([sc[b]["nats"][0] - sc[b]["nats"][i2] for b in books])
        for i, n in enumerate(names):
            g = np.mean([sc[b]["nats"][0] - sc[b]["nats"][i] for b in books])
            print(f"   {n:22s} {sc['A']['nats'][i]:.4f} | " + " ".join(f"{sc[b]['nats'][i]:.4f}" for b in books) + f" | {g:+.4f} | {g / g2 if g2 else 0:5.2f} | {sc['A_at_the_end']['nats'][i]:.4f} | {r['parts_used'][i]} / {r['parts'][i]}")
