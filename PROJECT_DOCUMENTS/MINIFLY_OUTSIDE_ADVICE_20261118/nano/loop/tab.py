#!/usr/bin/env python3
"""tab.py: the tables of REPORT.md from the result files in res/.   python3 tab.py > tab_output.txt"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); L2 = np.log(2); SEEDS = [73129, 93683, 116089, 406364, 545353]
def get(label):
    f = os.path.join(HERE, "res", label + ".json"); return json.load(open(f)) if os.path.exists(f) else None
def row(label, reading=None):
    o = get(label)
    if o is None: return None
    rs = [r for r in o["rows"] if reading is None or r["passes"] == reading]
    if not rs or (reading is None and "return_and_forward_cosine" not in o): return None
    return rs[-1]
def line(name, labels, reading=None, key="nats"):
    rs = [(sd, row(l, reading)) for sd, l in labels]; rs = [(sd, r) for sd, r in rs if r is not None]
    if not rs: return f"| {name} | not run | | | | |"
    n = np.array([r["nats"] for _, r in rs]); a = np.array([r["right"] for _, r in rs]); st = np.array([r["stream_nats"] for _, r in rs])
    return f"| {name} | {n.mean():.4f} | {n.mean() / L2:.4f} | {a.mean():.2f} | {st.mean():.4f} | {len(rs)}: " + ", ".join(f"{x:.4f}" for x in n) + " (" + ", ".join(f"{x:.1f}" for x in a) + ") |"
HEAD = "| | loss, nats | loss, bits | named right, % | the stream as read, nats | seeds: loss of each (named right of each) |\n|---|---|---|---|---|---|"
def tables():
    t = json.load(open(os.path.join(HERE, "res", "tables.json"))); print("| unsmoothed table of the last k bytes | loss, nats (crude) | loss, bits (crude) | named right, % | contexts seen in the stream |\n|---|---|---|---|---|")
    for k, v in t["tables"].items(): print(f"| k = {k} | {v['nats']:.4f} | {v['bits']:.4f} | {v['right']:.2f} | {v['contexts']:,} |")
    print(f"\nAlways the most frequent byte: {t['most_frequent_right']:.2f}% right. The byte frequencies of the stream alone: {t['frequencies_nats']:.4f} nats ({t['frequencies_nats'] / L2:.4f} bits). All 65 byte values alike: {t['uniform_nats']:.4f} nats ({t['uniform_nats'] / L2:.4f} bits).\n")
if __name__ == "__main__":
    print("## Tables\n"); tables()
    print("## L1 (PLAN_1.md): the two answering cells, 8 readings, D = 10, reported seeds\n"); print(HEAD)
    for H in (256, 1024):
        for arm, nm in (("LS", "least squares (loss crude)"), ("P", "branches compete")):
            for rs, bn in ((0, "born at nothing"), (1, "born at the earlier size")):
                print(line(f"{H} cells, {nm}, return strengths {bn}", [(sd, f"c_{arm}_H{H}_r{rs}_s{sd}") for sd in SEEDS]))
    print("\n## L2 a (PLAN_2.md): cells; arm P born at nothing, D = 10, seeds 73129, 93683, 116089\n")
    for rd in (8, 2):
        print(f"After {rd} readings:\n"); print(HEAD)
        for H in (256, 1024, 4096, 16384):
            print(line(f"{H} cells", [(sd, f"c_P_H{H}_r0_s{sd}" if H <= 1024 else f"a_H{H}_s{sd}") for sd in SEEDS[:3]], rd))
        print()
    print("Pre-check on development seed 1 (1,024 cells, 4 readings): " + "; ".join(f"eta1 = {e}: {row(l, 4)['nats']:.4f} nats, {row(l, 4)['right']:.2f}%" for e, l in (("1", "s3_P_f1o4"), ("4", "pre_H1024_e4"), ("16", "pre_H1024_e16")) if row(l, 4)))
    print("\n## L2 b (PLAN_3.md): readings; arm P born at nothing, D = 10\n"); print(HEAD)
    for H, sds in ((1024, SEEDS[:3]), (4096, SEEDS[:1])):
        for rd in (1, 2, 4, 8, 16, 32):
            print(line(f"{H} cells, {rd} readings", [(sd, f"b_H1024_s{sd}" if H == 1024 else f"a_H4096_s{sd}") for sd in sds], rd))
    print("\nReading 8 by the two kernels (kernp.c in the first plan's runs; kernt.c here), nats: " + "; ".join(f"seed {sd}: {row(f'c_P_H1024_r0_s{sd}', 8)['nats']:.4f} and {row(f'b_H1024_s{sd}', 8)['nats']:.4f}" for sd in SEEDS[:3] if row(f"c_P_H1024_r0_s{sd}", 8) and row(f"b_H1024_s{sd}", 8)))
    print("\n## L2 c (PLAN_4.md): delay lines; arm P born at nothing, 1,024 cells, 8 readings\n"); print(HEAD)
    for D in (4, 10, 16, 32):
        print(line(f"D = {D}", [(sd, f"c_P_H1024_r0_s{sd}" if D == 10 else f"d_D{D}_s{sd}") for sd in SEEDS[:3]], 8))
