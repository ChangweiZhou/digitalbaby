#!/usr/bin/env python3
"""Tables of nano/suite from res/*.json.   python3 report.py base | rows tag1[:cell] tag2[:cell] ..."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ST = ("shakespeare", "book", "py", "c", "table", "xml")
def L(name): 
    f = os.path.join(HERE, "res", name + ".json"); return json.load(open(f)) if os.path.exists(f) else None
f4 = lambda v: "-" if v is None else f"{v:.4f}"
def base():
    print("| stream | values | bytes read, tenth | base, nothing changing | best Kneser-Ney table (k) | base, going on | xz | bzip2 | gzip | cells | counts above zero | cells met once | CPU s |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in ST:
        b = L(f"{s}__base"); r = L(f"ref__{s}"); k = min(r["tables"], key=lambda k: r["tables"][k]["kneser_ney_nats"]); c = r["compressors"]; z = b["sizes_at_the_frozen_scoring"]
        print(f"| {s} | {b['S']} | {b['cut']:,}, {b['length'] - b['cut']:,} | {f4(b['frozen']['cell1'])} | {f4(r['tables'][k]['kneser_ney_nats'])} ({k}) | {f4(b['going_on']['cell1'])} | {f4(c['xz']['nats_a_byte_of_the_last_tenth'])} | {f4(c['bzip2']['nats_a_byte_of_the_last_tenth'])} | {f4(c['gzip']['nats_a_byte_of_the_last_tenth'])} | {z['cells']:,} | {z['counts_above_zero']:,} | {z['cells_met_once']:,} | {b['cpu']} |")
def rows(specs, ref=None):
    """Each spec tag[:cell]; prints nothing changing and going on for each stream, and the difference from the first spec (or ref)."""
    get = lambda s, sp: (lambda d, c: None if d is None else (d["frozen"][c], d["going_on"][c], d["sizes_at_the_frozen_scoring"]["cells"], d["sizes_at_the_frozen_scoring"]["counts_above_zero"] + d["sizes_at_the_frozen_scoring"].get("parts_used", 0), d["cpu"]))(L(f"{s}__{sp.split(':')[0]}"), "cell" + (sp.split(":")[1] if ":" in sp else "1"))
    print("| system | measure | " + " | ".join(ST) + " |"); print("|---|---|" + "---|" * len(ST))
    for i, sp in enumerate(specs):
        v = [get(s, sp) for s in ST]; b = [get(s, specs[0]) for s in ST]
        for m, name in ((0, "nothing changing"), (1, "going on")):
            print(f"| {sp} | {name} | " + " | ".join("-" if q is None else f4(q[m]) + ("" if i == 0 or b0 is None else f" ({q[m] - b0[m]:+.4f})") for q, b0 in zip(v, b)) + " |")
        print(f"| {sp} | cells; numbers stored; CPU s | " + " | ".join("-" if q is None else f"{q[2]:,}; {q[3]:,}; {q[4]}" for q in v) + " |")
if __name__ == "__main__":
    if sys.argv[1] == "base": base()
    else: rows(sys.argv[2:])
