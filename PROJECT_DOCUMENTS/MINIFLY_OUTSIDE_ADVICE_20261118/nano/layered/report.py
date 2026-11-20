#!/usr/bin/env python3
"""Tables of this line's plans from res/*.json.   python3 report.py > report.txt"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
def load(name):
    p = os.path.join(HERE, "res", "lay_" + name + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def fmt(v): return "    NOT_RUN" if v is None else f"{v:11.1f}"
def row(title, vals, tail=""):
    have = [v for v in vals if v is not None]; print(f"{title:66s}" + "".join(fmt(v) for v in vals) + (f"   median {st.median(have):5.1f}" if len(have) == len(vals) else "   (incomplete)") + tail)
def strs(title, vals): print(f"{title:66s}" + "".join("%11s" % (v if v is not None else "NOT_RUN") for v in vals))
def block(title, pat, key="last3_shown", gapkey="shown_by_gap", places=False):
    ds = [load(pat % s) for s in SEEDS]
    if not any(ds): return
    row(title, [d and d[key] for d in ds])
    if gapkey and all(ds) and gapkey in ds[0]["rows"][-1]:
        gs = list(ds[0]["rows"][-1][gapkey]); strs("   last reading by gap " + "/".join(gs), ["/".join(f"{d['rows'][-1][gapkey][g]:.0f}" for g in gs) for d in ds])
        print(f"{'':66s}   median by gap at the last reading: " + ", ".join(f"{g}: {st.median(d['rows'][-1][gapkey][g] for d in ds):.0f}" for g in gs))
    if places and all(ds): strs("   by place at the end (first/second/third answer byte)", ["/".join(str(x) for x in d["rows"][-1].get("place_never_at_gap", d["rows"][-1]["place"])) for d in ds])
    if all(ds): strs("   numbers that learn (forward) / CPU seconds", [f"{d['numbers']['forward_total']}/{d['cpu']}" for d in ds])

print("Columns: seeds", *SEEDS, "\n")
print("FIRST PLAN (E1). Sums shown, whole answer right, %, mean over gaps 0, 4, 8 and the last three readings of 960,000 lines")
for tx in ("m1", "g1"):
    block(f"{tx}: layered, chosen setting", f"r1_{tx}_main_%d"); block(f"{tx}: layered, cells A never change (etaA=0)", f"r1_{tx}_a0_%d")
print("\nSECOND PLAN (E2). Two-digit sums, two delay lines, 1,600,000 lines, mean of the last three readings")
block("d2: layered, chosen setting, never-shown sums", "r2_main_%d", key="last3_never", gapkey=None, places=True); block("d2: the same runs, sums shown", "r2_main_%d", key="last3_shown", gapkey=None)
block("d2: layered, cells A never change, never-shown sums", "r2_a0_%d", key="last3_never", gapkey=None, places=True)
print("\nTHIRD PLAN (E3). Two-digit numbers with 0 to 8 letters between them, two delay lines, 2,400,000 lines, never-shown sums, last three readings")
for name, t in (("main", "layered, chosen setting"), ("a0", "layered, cells A never change")):
    ds = [load(f"r3_{name}_%d" % s) for s in SEEDS]
    if any(ds):
        row(f"g2: {t}, mean over gaps", [d and d["last3_never"] for d in ds]); row(f"g2: {t}, at the gap of 8 letters", [d and d["last3_never_by_gap"]["8"] for d in ds])
        block(f"g2: {t} (again, with detail)", f"r3_{name}_%d", key="last3_never", gapkey="never_by_gap", places=True)
print("\nFOURTH PLAN (E4). A second gate on what a cell A gives out, at equal numbers of strengths")
for pat, t in (("r4_og_%d", "with the output gate"), ("r4_one_%d", "one gate, the same numbers")):
    ds = [load(pat % s) for s in SEEDS]
    if any(ds):
        k = "last3_never" if ds[0] is None or ds[0]["args"]["text"] in ("d2", "g2") else "last3_shown"
        block(t, pat, key=k, gapkey="never_by_gap" if k == "last3_never" else "shown_by_gap", places=True)
        if any(d and "last3_never_by_gap" in d for d in ds): row(f"   {t}: at the gap of 8 letters", [d and d["last3_never_by_gap"]["8"] for d in ds])
