"""Point (e): the inhibitory total on 15 seeds (drive8.py's out/*.json; build -ffp-contract=off). By E52's worker. Standard library only.
    python3 -I table_e.py       # writes TABLE_E.md
Rows: one a tissue and seed, window A2: S(d, 0) at d = 1, 2, 4, 8, 10 and near the output lines; connections a cell; made in the last
quarter; excitatory loops at the end; divergence after 512 bytes. Then per tissue (N0, N256) and wi (1/4, 3/16, 1/8): on how many seeds
the present byte is above 0.1103 at d = 8, d = 10 and near the output lines; on how many all nine entries of step 1 are above the
seed's own levels (0.1103; c0(1) + 0.02; c0(2) + 0.02); the smallest, median and largest of S(d, 0) at each d and near the output lines,
and of S(near, 1)."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = ("1", "2", "4", "8", "10")
WI = {"4": "1/4", "16_3": "3/16", "8": "1/8"}
f = lambda v: "–" if v is None else f"{v:.3f}"
R = {}
for p in sorted((HERE / "out").glob("N*_wi*_s*.json")):
    r = json.loads(p.read_text())
    tis, wi, s = r["label"].split("_wi")[0], r["label"].split("_wi")[1].rsplit("_s", 1)[0], int(r["label"].rsplit("_s", 1)[1])
    R.setdefault((tis, wi), {})[s] = r
L = []
for tis in ("N0", "N256"):
    for wi in ("4", "16_3", "8"):
        D = R.get((tis, wi), {})
        if not D:
            continue
        L += [f"\n## {tis}, wi = {WI[wi]} ({len(D)} seeds)\n", "| seed | S(d,0) d1 / 2 / 4 / 8 / 10 | near lag 0 / 1 / 2 | all nine | conns mean / max | made last quarter | exc loops (cells) | div 512 |", "|" + "---|" * 8]
        nine = {}
        for s in sorted(D):
            r = D[s]; a = r["A2"]; c0 = a["c0"]
            lev = {"0": 0.1103, "1": (c0.get("1") or 0) + 0.02, "2": (c0.get("2") or 0) + 0.02}
            nine[s] = all(a["S"][d][g] is not None and a["S"][d][g] > lev[g] for d in ("1", "2", "4") for g in ("0", "1", "2"))
            lp = r["loops_end"]["excitatory"]
            L.append(f"| {s} | " + " / ".join(f(a['S'][d]['0']) for d in DS) + " | " + " / ".join(f(a['near']['S'][g]) for g in ("0", "1", "2"))
                     + f" | {'yes' if nine[s] else 'no'} | {r['conns_end']['mean']:.1f} / {r['conns_end']['max']} | {r['made_last_quarter']} | {lp['loops']} ({lp['cells_in_loops']}) | {r['divergence']['512']:.0e} |")
        ab = lambda fn: sum(1 for s in D if (fn(D[s]) or 0) > 0.1103)
        L.append(f"\n{tis}, wi = {WI[wi]}: present byte above 0.1103 at d = 8 on {ab(lambda r: r['A2']['S']['8']['0'])}, at d = 10 on {ab(lambda r: r['A2']['S']['10']['0'])}, "
                 f"near the output lines on {ab(lambda r: r['A2']['near']['S']['0'])} of {len(D)}; all nine entries of step 1 on {sum(nine.values())}; divergence 0 on {sum(D[s]['divergence']['512'] == 0 for s in D)}.\n")
        L.append("| entry | smallest | median | largest |\n|---|---|---|---|")
        for nm, fn in [(f"S({d},0)", (lambda d: lambda r: r["A2"]["S"][d]["0"])(d)) for d in DS] + [("S(near,0)", lambda r: r["A2"]["near"]["S"]["0"]), ("S(near,1)", lambda r: r["A2"]["near"]["S"]["1"])]:
            v = [fn(D[s]) for s in D if fn(D[s]) is not None]
            L.append(f"| {nm} | {min(v):.3f} | {st.median(v):.3f} | {max(v):.3f} |")
fl = HERE / "logs" / "FAILED.txt"
L.append("\nBuild: tissue_t8.c with -ffp-contract=off. Failed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".")
(HERE / "TABLE_E.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
