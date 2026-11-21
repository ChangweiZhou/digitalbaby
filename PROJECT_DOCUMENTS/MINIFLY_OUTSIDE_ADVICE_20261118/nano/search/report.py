#!/usr/bin/env python3
"""Tables of PLAN_2.md from res/c_*.json.   python3 report.py > report_2.txt"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SEEDS = (73129, 93683, 116089, 406364, 545353)
def L(tag):
    f = os.path.join(HERE, "res", tag + ".json"); return json.load(open(f)) if os.path.exists(f) else None
def med(v): return round(float(np.median(v)), 1) if v else None
def sz(rs): return "cells %s, counts above zero %s, displaced %s, tried %s, CPU-s %s (medians)" % tuple(int(np.median([r["size"][k] if k != "cpu" else r["cpu"] for r in rs])) for k in ("cells", "counts_above_zero", "displaced", "kinds_tried", "cpu"))
def sums_row(name, pat, phase_rows=lambda r: [x for x in r["rows"] if x["phase"] == 0]):
    rs = [L(pat % s) for s in SEEDS]; rs = [r for r in rs if r]
    if not rs: print(f"{name}: NOT RUN"); return
    l3 = [round(float(np.mean([x["add"]["whole"] for x in phase_rows(r)][-3:])), 1) for r in rs]; la = [phase_rows(r)[-1]["add"]["whole"] for r in rs]; pl = np.median([phase_rows(r)[-1]["add"]["place"] for r in rs], 0)
    print(f"{name} ({len(rs)} seeds): last three {l3}; median {med(l3)} | last reading {la}; median {med(la)} | by place, last reading, median {[float(v) for v in pl]}\n      {sz(rs)}")
print("== Two-digit sums, never-shown pairs, own bytes, whole answer right, % (seeds 73129, 93683, 116089, 406364, 545353)")
sums_row("40,000 lines, search (nrep 16)", "c_s2_life_n16_%d"); sums_row("40,000 lines, control (same 128 random kinds, never changed)", "c_s2_40k_n0_%d"); sums_row("40,000 lines, nrep 8", "c_s2_40k_n8_%d"); sums_row("40,000 lines, nrep 32", "c_s2_40k_n32_%d")
sums_row("400,000 lines, search (nrep 16)", "c_s2_400k_n16_%d"); sums_row("400,000 lines, control", "c_s2_400k_n0_%d")
for tag in ("c_s2_400k_n16_%d", "c_s2_400k_n0_%d"):
    rs = [L(tag % s) for s in SEEDS]; rs = [r for r in rs if r]
    if rs: print("   ", tag, "median by lines:", {n: med([x["add"]["whole"] for r in rs for x in r["rows"] if x["lines"] == n]) for n in (10000, 20000, 40000, 100000, 200000, 300000, 400000)})
print("\n== The life add:40000, sub:20000 (search, nrep 16)")
rs = [L("c_s2_life_n16_%d" % s) for s in SEEDS]; rs = [r for r in rs if r]
if rs:
    b = [[x for x in r["rows"] if x["phase"] == 0][-1]["add"]["whole"] for r in rs]; a = [r["rows"][-1]["add"]["whole"] for r in rs]; sc = [r["rows"][-1]["sub_clean"] for r in rs]; sa = [r["rows"][-1]["sub_all"] for r in rs]
    print(f"addition before {b}; median {med(b)} | after {a}; median {med(a)} | after as % of before {[round(100 * y / x, 1) if x else None for x, y in zip(b, a)]}; median {med([100 * y / x for x, y in zip(b, a) if x])}")
    print(f"subtraction at the end: clean score {sc}; median {med(sc)} (pairs: {[r['rows'][-1]['n_clean'] for r in rs]}) | all 500 held-out pairs {sa}; median {med(sa)}\n      {sz(rs)}")
print("\n== Three-digit sums (aaa+bbb=cccc), never-shown pairs, own bytes, whole answer, %")
sums_row("40,000 lines, search", "c_s3_40k_n16_%d"); sums_row("40,000 lines, control", "c_s3_40k_n0_%d"); sums_row("400,000 lines, search", "c_s3_400k_n16_%d")
rs = [L("c_s3_400k_n16_%d" % s) for s in SEEDS]; rs = [r for r in rs if r]
if rs: print("    median by lines:", {n: med([x["add"]["whole"] for r in rs for x in r["rows"] if x["lines"] == n]) for n in (10000, 20000, 40000, 100000, 200000, 300000, 400000)})
print("\n== Prose: tiny Shakespeare's stream, then three books; each book's last tenth right after its stream, nats per byte (other lines: T0 1.5023, 1.3645, 1.4456; T2 1.4802, 1.3162, 1.4109)")
for name, tag in (("search (nrep 16)", "c_p_n16"), ("control (128 random kinds never changed)", "c_p_n0"), ("nrep 8", "c_p_n8"), ("nrep 32", "c_p_n32"), ("six hand-given runs of 1..6 bytes, this program, no search (orientation)", "c_p_runs6")):
    r = L(tag)
    if not r: print(f"{name}: NOT RUN"); continue
    print(f"{name}: " + ", ".join(f"{k.split('-')[0]} {v['nats']:.4f} ({v['right']:.1f}% right; before reading it {v['before_reading_it']:.4f})" for k, v in r["books"].items()) + f" | last tenth of tiny Shakespeare at the end {r['A_last_tenth_at_end']['nats']:.4f}\n      cells {r['size']['cells']}, counts above zero {r['size']['counts_above_zero']}, displaced {r['size']['displaced']}, tried {r['size']['kinds_tried']}, CPU-s {r['cpu']}")
print("\n== The kinds that earned most on the answer at the end (lines: dN = the byte N places back; gR.H = the guess of layer R of the byte H ahead; earnings in nats summed over the kind's life)")
for name, tag in [("two-digit sums 40,000+20,000 lines, seed %d" % s, "c_s2_life_n16_%d" % s) for s in SEEDS] + [("two-digit sums 400,000 lines, seed %d" % s, "c_s2_400k_n16_%d" % s) for s in SEEDS] + [("three-digit sums 400,000 lines, seed %d" % s, "c_s3_400k_n16_%d" % s) for s in SEEDS[:3]] + [("prose, end of the life", "c_p_n16")]:
    r = L(tag)
    if r: print(f"{name}: " + "; ".join(f"[{k['lines']}] {k['earned_answer']:.0f} (age {k['age']})" for k in r["kinds"][:10]))
r = L("c_p_n16")
if r:
    print("prose, after tiny Shakespeare's stream: " + "; ".join(f"[{k['lines']}] {k['earned_answer']:.0f}" for k in r["kinds_after_A"][:12]))
    for nm, rr in (("prose", r),) + tuple(("sums seed %d" % s, L("c_s2_life_n16_%d" % s)) for s in SEEDS if L("c_s2_life_n16_%d" % s)):
        ks = rr["kinds"]; n = len(ks); g = [k for k in ks if "g" in k["lines"]]; top = ks[:20]
        print(f"   {nm}: of {n} kinds alive at the end {len(g)} are keyed on a guess; of the 20 that earned most: {sum('g' in k['lines'] for k in top)} keyed on a guess, mean number of lines {np.mean([len(k['lines'].split()) for k in top]):.1f}, furthest place back used {max(int(l[1:]) for k in top for l in k['lines'].split() if l[0] == 'd')}")
