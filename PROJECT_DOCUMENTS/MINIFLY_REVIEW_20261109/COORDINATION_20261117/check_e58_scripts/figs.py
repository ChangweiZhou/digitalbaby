import json, statistics as st, sys
S = sys.argv[1]
SEEDS = (73129, 93683, 116089)
def L(n): return json.load(open(S + n + ".json"))
def red(row): return row["scores"]["summary"]["reduction_score"]
def rows(rec): return {r["updates"]: r for r in rec["rows"]}
f1 = lambda v: f"{v:.1f}"
E40 = {(k, r, s): L(f"e40_{k}_{r}_{s}") for k in ("lived", "fresh") for r in "SB" for s in SEEDS}
E48 = {(r, s, k): L(f"e48_{r}_{s}_r{k}") for r in "SB" for s in SEEDS for k in (4, 16, 64)}
print("== E40 lived reduction score by row")
R40 = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        print(r, s, "row0", f1(red(rw[0])), "|", ", ".join(f1(red(rw[u])) for u in R40))
print("== E40: largest move between neighbouring rows, before row 1024 (rows 64..1024) and from 1024 on")
mb = ma = 0
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        v = [red(rw[u]) for u in R40]
        b = max(abs(v[i + 1] - v[i]) for i in range(0, 4))     # 64-128,128-256,256-512,512-1024
        a = max(abs(v[i + 1] - v[i]) for i in range(4, 7))     # 1024-2048, 2048-4096, 4096-8192
        print(r, s, "before", f1(b), "after", f1(a), "| max row", max(R40, key=lambda u: red(rw[u])), f1(max(v)),
              "| sd rows>=1024", f"{st.pstdev(v[4:]):.2f} (pop) {st.stdev(v[4:]):.2f} (sample)", "| rise 256->max later", )
        mb = max(mb, b); ma = max(ma, a)
print("max before", f1(mb), "max after", f1(ma))
print("== E48 rank 16: rows")
R48 = (0, 16, 32, 64, 128, 256, 512, 1024)
for r in "SB":
    for s in SEEDS:
        rw = rows(E48[(r, s, 16)]); e = rows(E40[("lived", r, s)])
        print(r, s, ", ".join(f"{u}:{f1(red(rw[u]))}" for u in R48), "| rise 512->1024", f"{red(rw[1024]) - red(rw[512]):.2f}",
              "| at 64: term", f1(red(rw[64])), "E40", f1(red(e[64])), "below" if red(rw[64]) < red(e[64]) else "NOT BELOW")
print("== the hint: term r16 at 1024 minus E40 at 1024; minus E40's highest row")
for r in "SB":
    for s in SEEDS:
        rw = rows(E48[(r, s, 16)]); e = rows(E40[("lived", r, s)])
        hi = max(red(e[u]) for u in e)
        hi_pos = max(e, key=lambda u: red(e[u]))
        print(r, s, f"{red(rw[1024]) - red(e[1024]):.2f}", "| vs highest", f"{red(rw[1024]) - hi:.2f}", "(E40 highest", f1(hi), "at row", hi_pos, ")",
              "| vs E40 8192:", f"{red(rw[1024]) - red(e[8192]):.2f}")
print("== rank at row 1024")
for r in "SB":
    for s in SEEDS:
        print(r, s, ", ".join(f1(red(rows(E48[(r, s, k)])[1024])) for k in (4, 16, 64)))
print("== earlier panels: E40 lived, mean of the four life-family panels (add97 sub97 add73 sub73) and six; loss within 256")
pct = lambda d: 100.0 * d["exact"] / d["n"]
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        def mean(u, names): return sum(pct(rw[u]["scores"]["earlier_k12"][n]) for n in names) / len(names)
        life = ("add97", "sub97", "add73", "sub73"); six = ("add89", "sub89") + life; first = ("add89", "sub89")
        print(r, s, "life4:", " ".join(f"{u}:{mean(u, life):.1f}" for u in rw), "| loss by 256:", f"{mean(0, life) - mean(256, life):.1f}",
              "min over rows<=256:", f"{mean(0, life) - min(mean(u, life) for u in rw if u <= 256):.1f}")
        print("     six:", " ".join(f"{u}:{mean(u, six):.1f}" for u in rw), "| loss by 256:", f"{mean(0, six) - mean(256, six):.1f}")
        print("     first2:", " ".join(f"{u}:{mean(u, first):.1f}" for u in rw), "| loss by 256:", f"{mean(0, first) - mean(256, first):.1f}")
print("== fresh at 8192")
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("fresh", r, s)])
        print(r, s, f"{red(rw[8192]):.2f}", "max over rows", f"{max(red(v) for v in rw.values()):.2f}")
print("== n of reduction pairs")
for s in SEEDS:
    rw = rows(E40[("lived", "S", s)])
    print(s, {n: rw[0]["scores"]["held_plain_k0"][n]["reduction"]["n"] for n in ("add83", "sub83")})
print("== cost, seed 73129")
for key in (("lived", "S", 73129), ("lived", "B", 73129)):
    r = E40[key]; print("E40", key, "cpu_updates", f"{r['cpu_updates']:.0f}", "cpu_scoring", f"{r['cpu_scoring']:.0f}", "rows", len(r["rows"]), "per row", f"{r['cpu_scoring']/len(r['rows']):.1f}", "per update", f"{r['cpu_updates']/8192:.4f}")
for key in (("S", 73129, 16), ("B", 73129, 16)):
    r = E48[key]; print("E48", key, "cpu_updates", f"{r['cpu_updates']:.0f}", "cpu_scoring", f"{r['cpu_scoring']:.0f}", "rows", len(r["rows"]), "per update", f"{r['cpu_updates']/1024:.4f}")
    rr = r["rows"]; prev = 0
    print("   scoring per row:", [round(x["cpu_scoring_so_far"] - (rr[i-1]["cpu_scoring_so_far"] if i else 0), 1) for i, x in enumerate(rr)])
for s in SEEDS:
    for rl in "SB":
        a = E40[("lived", rl, s)]; b = E48[(rl, s, 16)]
        print(s, rl, "E40 upd", f"{a['cpu_updates']:.0f}", "score", f"{a['cpu_scoring']:.0f}", "| E48 upd", f"{b['cpu_updates']:.0f}", "score", f"{b['cpu_scoring']:.0f}")
