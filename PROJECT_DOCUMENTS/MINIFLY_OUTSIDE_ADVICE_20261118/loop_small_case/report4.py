#!/usr/bin/env python3
"""Tables of the fourth, fifth and sixth plans from res/*.json.   python3 report4.py > report4.txt"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
def load(name):
    p = os.path.join(HERE, "res", name + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def fmt(v): return "  NOT_RUN" if v is None else f"{v:9.1f}"
def row(title, vals):
    have = [v for v in vals if v is not None]; print(f"{title:58s}" + "".join(fmt(v) for v in vals) + (f"   median {st.median(have):5.1f}" if len(have) == len(vals) else "   (incomplete)"))
def ph(d, phase): return [r for r in d["rows"] if r["phase"] == phase]
def first_reach(rows, level): return next((r["lines"] for r in rows if r["add"] >= level), None)

print("FOURTH PLAN (cells2.py). Whole answer right on 500 never-shown pairs, %. Columns: seeds", *SEEDS)
print("\nA and B. Never-shown addition, mean of the last three readings of 400,000 lines")
for title, pat in (("every cell reads all 140 lines (one step)", "c2_r_plain_%d"), ("32 lines a cell, fixed", "c2_r_fan32_%d"), ("32 lines a cell, rewiring", "c2_r_fan32rw_%d"),
                   ("64 lines a cell, rewiring", "c2_r_fan64rw_%d"), ("one sign for each cell", "c2_r_dale_%d"), ("two steps, 128 + 128", "c2_r_two128_%d"), ("two steps, 256 + 256", "c2_r_two256_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_add_phase0"] for d in ds])
    if any(ds): print(f"{'':58s}" + "".join(("  NOT_RUN" if d is None else "%9s" % "/".join(str(x) for x in d["by_place_add"])) for d in ds) + "   by place at the end (hundreds/tens/units)")
print("\nC. Addition (400,000 lines), then subtraction (200,000 lines), then addition again (40,000 lines)")
K = [load("c2_r_keep_%d" % s) for s in SEEDS]
if all(K):
    row("addition after its own phase (last three readings)", [d["last3_add_phase0"] for d in K])
    row("addition after subtraction", [ph(d, 1)[-1]["add"] for d in K]); row("subtraction after its phase", [ph(d, 1)[-1]["sub"] for d in K])
    row("addition 2,000 lines into subtraction", [next(r["add"] for r in ph(d, 1) if r["lines"] == 2000) for d in K])
    row("addition, the answering cell put back", [d["where"]["addition_with_the_answering_cell_put_back"] for d in K])
    row("addition, the cells in between put back", [d["where"]["addition_with_the_cells_in_between_put_back"] for d in K])
    row("share of the loss undone by the answering cell put back, %", [100 * (d["where"]["addition_with_the_answering_cell_put_back"] - ph(d, 1)[-1]["add"]) / (ph(d, 0)[-1]["add"] - ph(d, 1)[-1]["add"]) for d in K])
    row("change of the cells' strengths over their size", [d["where"]["change_of_W1_over_its_size"] for d in K]); row("change of the answering cell's strengths over their size", [d["where"]["change_of_W2_over_its_size"] for d in K])
    f1 = [first_reach(ph(d, 0), 70) for d in K]; f2 = [first_reach(ph(d, 2), 70) for d in K]
    row("lines to 70% at first", f1); row("lines to 70% when taught again", f2)
    if all(f1) and all(f2): row("ratio", [b / a for a, b in zip(f1, f2)])
    row("addition 2,000 lines into teaching again", [next(r["add"] for r in ph(d, 2) if r["lines"] == 2000) for d in K]); row("subtraction after addition is taught again", [ph(d, 2)[-1]["sub"] for d in K])
for title, pat in (("one addition line in 32: addition at the end", "c2_r_reh32_%d"), ("new cells (128) and one line in 32: addition at the end", "c2_r_res128_reh32_%d"), ("new cells (128), no rehearsal: addition at the end", "c2_r_res128_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and ph(d, 1)[-1]["add"] for d in ds]); row("   subtraction at the end", [d and ph(d, 1)[-1]["sub"] for d in ds])

print("\n\nFIFTH PLAN (hold.py, no delay chain, no crossing). Single-digit sums shown, 240,000 lines, mean of the last three readings")
for title, pat in (("cells learn", "hold_r1_learn_%d"), ("cells never change", "hold_r1_frozen_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_shown"] for d in ds])
print("\nSIXTH PLAN (hold.py cross=1). Two-digit sums never shown, 600,000 lines, mean of the last three readings")
for title, pat in (("no crossing", "hold_r2_cross0_%d"), ("one crossing, return strengths exact", "hold_r2_exact_%d"), ("one crossing, return strengths learned", "hold_r2_learned_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_never"] for d in ds])
    if any(ds): row("   sums shown", [d and d["last3_shown"] for d in ds]); print(f"{'':58s}" + "".join(("  NOT_RUN" if d is None else "%9s" % "/".join(str(x) for x in d["rows"][-1]["place"])) for d in ds) + "   by place at the end")
