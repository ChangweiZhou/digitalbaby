#!/usr/bin/env python3
"""The rule of PLAN_2.md applied to res/*.json.   python3 verdict.py name base_tag[:cell] idea_tag[:cell] ...
An idea lowers a kind of stream if the loss is lower by at least 0.001 (prose: on both streams); kept if at least three of five kinds are lowered and no stream is harmed by more than 0.005."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ST = ("shakespeare", "book", "py", "c", "table", "xml"); KINDS = (("shakespeare", "book"), ("py",), ("c",), ("table",), ("xml",))
def get(s, sp):
    f = os.path.join(HERE, "res", f"{s}__{sp.split(':')[0]}.json")
    if not os.path.exists(f): return None
    d = json.load(open(f)); c = "cell" + (sp.split(":")[1] if ":" in sp else "1"); z = d["sizes_at_the_frozen_scoring"]; return dict(frozen=d["frozen"][c], going=d["going_on"][c], cells=z["cells"], numbers=z["counts_above_zero"] + z.get("parts_used", 0), cpu=d["cpu"])
def verdict(base, idea, m):
    dif = {s: (None if get(s, base) is None or get(s, idea) is None else get(s, idea)[m] - get(s, base)[m]) for s in ST}
    if any(v is None for v in dif.values()): return dif, None, None, "NOT COMPLETE"
    low = sum(all(dif[s] <= -0.001 for s in k) for k in KINDS); harm = max(dif.values()); return dif, low, harm, ("KEPT" if low >= 3 and harm <= 0.005 else "not kept")
if __name__ == "__main__":
    name, base, idea = sys.argv[1:4]
    for m, label in (("going", "going on"), ("frozen", "nothing changing")):
        dif, low, harm, v = verdict(base, idea, m); print(f"| {name} | {label} | " + " | ".join("-" if dif[s] is None else f"{get(s, idea)[m]:.4f} ({dif[s]:+.4f})" for s in ST) + f" | {low} of 5 | {'-' if harm is None else f'{harm:+.4f}'} | {v} |")
    b = [get(s, base) for s in ST]; i = [get(s, idea) for s in ST]
    print(f"| {name} | cells x; numbers stored x; CPU s | " + " | ".join("-" if q is None or p is None else f"{q['cells'] / p['cells']:.2f}; {q['numbers'] / p['numbers']:.2f}; {q['cpu']:.0f}" for p, q in zip(b, i)) + " | | | |")
