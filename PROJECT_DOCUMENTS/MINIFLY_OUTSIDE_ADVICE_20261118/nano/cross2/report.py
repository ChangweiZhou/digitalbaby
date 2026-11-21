#!/usr/bin/env python3
"""Tables from res/hold_<label>.json.   python3 report.py e1 [seeds ...]   |   python3 report.py rows <label> [<label> ...]"""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
def load(label): return json.load(open(os.path.join(HERE, "res", f"hold_{label}.json")))
def e1(label):
    o = load(label); rows = {r["lines"]: r for r in o["rows"]}; sh = [(r["lines"], r["shown"]) for r in o["rows"]]
    final = statistics.mean(rows[n]["shown"] for n in (800000, 880000, 960000)); to90 = next((n for n, v in sh if v >= 90), None)
    stay = next((n for i, (n, v) in enumerate(sh) if all(w >= 90 for _, w in sh[i:])), None); at240 = rows[240000]["shown"]
    return dict(final=final, to90=to90, stay=stay, at240=at240, cpu=o["cpu"], gaps=rows[960000]["shown_by_gap"])
if __name__ == "__main__":
    if sys.argv[1] == "e1":
        seeds = sys.argv[2:] or ["73129", "93683", "116089", "406364", "545353"]; f = lambda v: "never" if v is None else f"{v // 1000}k"
        for c in ("1", "2"):
            R = [e1(f"e1_c{c}_{s}") for s in seeds]; big = 10 ** 9
            print(f"cross={c}  final %: " + " ".join(f"{r['final']:.1f}" for r in R) + f" | median {statistics.median(r['final'] for r in R):.1f}, lowest {min(r['final'] for r in R):.1f}")
            m = statistics.median(r["to90"] or big for r in R); print("   lines to 90% (first reading): " + " ".join(f(r["to90"]) for r in R) + " | median " + ("never" if m == big else f"{m / 1000:.0f}k"))
            print("   lines from which every reading is at least 90%: " + " ".join(f(r["stay"]) for r in R))
            print("   at 240,000 lines %: " + " ".join(f"{r['at240']:.1f}" for r in R) + " | at the end by gap 0, 4, 8: " + "; ".join(" ".join(f"{v:.0f}" for v in r["gaps"].values()) for r in R))
            print("   CPU s: " + " ".join(str(r["cpu"]) for r in R))
    else:
        for label in sys.argv[2:]:
            o = load(label); print(label, o["args"], f"cpu {o['cpu']} s; last3 shown {o['last3_shown']:.1f} never {o['last3_never']:.1f}")
            for r in o["rows"]:
                print(f"   {r['lines']:8d} shown {r['shown']:5.1f} never {r['never']:5.1f} place {r['place']} bytes {r['bytes_right']:.1f}" + (" | shown by gap " + " ".join(f"{v:.0f}" for v in r["shown_by_gap"].values()) if "shown_by_gap" in r else "")
                      + (" | never by gap " + " ".join(f"{v:.0f}" for v in r["never_by_gap"].values()) if "never_by_gap" in r else ""))
