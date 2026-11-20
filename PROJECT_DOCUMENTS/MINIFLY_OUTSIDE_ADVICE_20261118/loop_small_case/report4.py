#!/usr/bin/env python3
"""Tables of the fourth, fifth and sixth plans from res/*.json.   python3 report4.py > report4.txt"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = [73129, 93683, 116089, 406364, 545353]
def load(name):
    p = os.path.join(HERE, "res", name + ".json"); return json.load(open(p)) if os.path.exists(p) else None
def fmt(v): return "    NOT_RUN" if v is None else f"{v:11.1f}"
def row(title, vals):
    have = [v for v in vals if v is not None]; print(f"{title:58s}" + "".join(fmt(v) for v in vals) + (f"   median {st.median(have):5.1f}" if len(have) == len(vals) else "   (incomplete)"))
def ph(d, phase): return [r for r in d["rows"] if r["phase"] == phase]
def first_reach(rows, level): return next((r["lines"] for r in rows if r["add"] >= level), None)

print("FOURTH PLAN (cells2.py). Whole answer right on 500 never-shown pairs, %. Columns: seeds", *SEEDS)
print("\nA and B. Never-shown addition, mean of the last three readings of 400,000 lines")
for title, pat in (("every cell reads all 140 lines (one step)", "c2_r_plain_%d"), ("32 lines a cell, fixed", "c2_r_fan32_%d"), ("32 lines a cell, rewiring", "c2_r_fan32rw_%d"),
                   ("64 lines a cell, rewiring", "c2_r_fan64rw_%d"), ("one sign for each cell", "c2_r_dale_%d"), ("two steps, 128 + 128", "c2_r_two128_%d"), ("two steps, 256 + 256", "c2_r_two256_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_add_phase0"] for d in ds])
    if any(ds): print(f"{'':58s}" + "".join(("    NOT_RUN" if d is None else "%11s" % "/".join(str(x) for x in d["by_place_add"])) for d in ds) + "   by place at the end (hundreds/tens/units)")
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
print("\n\nSEVENTH PLAN (hold.py, a gap of 0 to 8 letters). Sums shown, whole answer right, %, mean of the last three readings of 960,000 lines (mean over gaps 0, 4, 8)")
for tx, name in (("m1", "c = (a + b) mod 10"), ("g1", "c = a + b")):
    print(f"  text {tx}: {name}")
    for title, arm in (("no crossing, eta1 = 1", "c0"), ("no crossing, eta1 = 4", "c0e4"), ("one crossing, the reader's own strengths", "exact"), ("one crossing, return strengths grown from nothing", "grown")):
        ds = [load(f"hold_r7_{tx}_{arm}_{s}") for s in SEEDS]; row("    " + title, [d and d["last3_shown"] for d in ds])
        if all(ds):
            for g in ("0", "4", "8"): row(f"       at the end, gap of {g} letters", [d["rows"][-1]["shown_by_gap"][g] for d in ds])

print("\n\nNINTH PLAN (hold.py with a short chain of delay lines). Two-digit sums never shown, %, mean of the last three readings of 1,600,000 lines")
for title, pat in (("the present byte and the two before it, no crossing", "hold_r9_ch2_c0_%d"), ("the present byte and the two before it, one crossing", "hold_r9_ch2_exact_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_never"] for d in ds])
    if all(ds):
        row("   sums shown", [d["last3_shown"] for d in ds]); row("   never shown, at the last reading", [d["rows"][-1]["never"] for d in ds])
        print(f"{'':58s}" + "".join("%11s" % "/".join(str(x) for x in d["rows"][-1]["place"]) for d in ds) + "   by place at the end (hundreds/tens/units)")

print("\n\nEIGHTH PLAN (cells2.py clip=1/4, the active-cell kernel). Never-shown addition, mean of the last three readings of 400,000 lines")
for title, pat in (("logistic cells, 256 (dense compiled kernel)", "c2_r_plain_%d"), ("silent cells, 256", "c2_r8_H256_%d"), ("silent cells, 512", "c2_r8_H512_%d")):
    ds = [load(pat % s) for s in SEEDS]; row(title, [d and d["last3_add_phase0"] for d in ds])
    if all(ds):
        row("   CPU seconds of the run", [d["cpu"] for d in ds])
        if "active_cells_a_byte" in ds[0]: row("   cells active at a byte", [d["active_cells_a_byte"] for d in ds])

print("\n\nTHE COMPILED KERNEL BESIDE THE NUMPY PROGRAM, same seed and options, full runs (never-shown addition, last three readings; numpy / compiled)")
import glob
for f in sorted(glob.glob(os.path.join(HERE, "res", "c2_np_r_*.json"))):
    a = json.load(open(f)); b = load(os.path.basename(f)[:-5].replace("c2_np_r_", "c2_r_"))
    print(f"   {os.path.basename(f)[8:-5]:22s} {a['last3_add_phase0']:6.1f} / " + (f"{b['last3_add_phase0']:6.1f}   CPU {a['cpu']} s / {b['cpu']} s" if b else "NOT_RUN"))
