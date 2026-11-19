"""Long axons on 15 seeds (drive9.py's out/L*.json), by E52's worker. Standard library only.
    python3 -I table_L.py       # writes TABLE_L.md
Per setting: the nine entries of step 1 (L = 0, 1, 2 at d = 1, 2, 4 against the seed's own levels: 0.1103, c0(1) + 0.02, c0(2) + 0.02):
on how many seeds all nine, and the smallest margin (score less level) over all seeds and entries; near the output lines (both sets) at
lags 0, 1, 2: on how many seeds above the level, smallest / median / largest; on how many seeds lags 0, 1 and 2 are all above near the
output lines; connections a cell; made in the last quarter; divergence after 512 bytes. Then one row a seed."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET = {"1": "wi=1/4 tau=u2-256", "2": "wi=1/4 tau=u2-128", "3": "wi=1/4 tau=u2-256 longfast=1", "4": "wi=1/4 tau=u2-128 longfast=1",
       "5": "wi=3/16 tau=u2-128 longfast=1", "6": "wi=3/16 tau=u2-256"}
G = ("0", "1", "2")
R = {}
for p in sorted((HERE / "out").glob("L*_s*.json")):
    r = json.loads(p.read_text())
    n, s = r["label"][1:].split("_s")
    R.setdefault(n, {})[int(s)] = r


def lev(r):
    c0 = r["A2"]["c0"]
    return {"0": 0.1103, "1": (c0.get("1") or 0) + 0.02, "2": (c0.get("2") or 0) + 0.02}


L = ["| setting | seeds | nine of nine | smallest margin of the nine | near: lag 0 / 1 / 2 above on | near: all three on | near8: lag 0 / 1 / 2 above on | near8: all three on | conns mean (median over seeds) | made last quarter min / med / max | divergence 0 on |",
     "|" + "---|" * 11]
detail = []
for n in ("3", "4", "1", "2", "5", "6"):
    D = R.get(n, {})
    if not D:
        continue
    nine = 0; marg = []
    cnt = {k: {g: 0 for g in G} for k in ("near", "near8")}; allthree = {"near": 0, "near8": 0}
    vals = {k: {g: [] for g in G} for k in ("near", "near8")}
    for s, r in D.items():
        a = r["A2"]; lv = lev(r)
        m = [a["S"][d][g] - lv[g] for d in ("1", "2", "4") for g in G if a["S"][d][g] is not None]
        marg.append(min(m)); nine += min(m) > 0
        for k in ("near", "near8"):
            ok = True
            for g in G:
                v = a[k]["S"][g]; vals[k][g].append(v)
                if v is not None and v > lv[g]:
                    cnt[k][g] += 1
                else:
                    ok = False
            allthree[k] += ok
    ml = sorted(r["made_last_quarter"] for r in D.values())
    L.append(f"| ({n}) {SET[n]} | {len(D)} | {nine} | {min(marg):+.3f} | " + " / ".join(str(cnt['near'][g]) for g in G) + f" | {allthree['near']} | "
             + " / ".join(str(cnt['near8'][g]) for g in G) + f" | {allthree['near8']} | {st.median(r['conns_end']['mean'] for r in D.values()):.1f} | "
             + f"{ml[0]} / {st.median(ml):.0f} / {ml[-1]} | {sum(r['divergence']['512'] == 0 for r in D.values())} |")
    detail += [f"\n### ({n}) {SET[n]}\n", "| entry | smallest | median | largest |", "|---|---|---|---|"]
    for k in ("near", "near8"):
        for g in G:
            v = [x for x in vals[k][g] if x is not None]
            detail.append(f"| {k} lag {g} | {min(v):.3f} | {st.median(v):.3f} | {max(v):.3f} |")
    detail += ["\n| seed | nine: smallest margin | near lag 0 / 1 / 2 | near8 lag 0 / 1 / 2 | levels lag 1 / 2 | conns mean / max | made last quarter | div 512 |", "|" + "---|" * 8]
    for s in sorted(D):
        r = D[s]; a = r["A2"]; lv = lev(r)
        m = min(a["S"][d][g] - lv[g] for d in ("1", "2", "4") for g in G if a["S"][d][g] is not None)
        mk = lambda v, g: ("–" if v is None else f"{v:.3f}" + ("**" if v > lv[g] else ""))
        detail.append(f"| {s} | {m:+.3f} | " + " / ".join(mk(a['near']['S'][g], g) for g in G) + " | " + " / ".join(mk(a['near8']['S'][g], g) for g in G)
                      + f" | {lv['1']:.3f} / {lv['2']:.3f} | {r['conns_end']['mean']:.1f} / {r['conns_end']['max']} | {r['made_last_quarter']} | {r['divergence']['512']:.0e} |")
fl = HERE / "logs" / "FAILED.txt"
out = "\n".join(L + detail) + "\n\n** : above the seed's own level. Build: the reviewer's tissue.c at e633b205, -ffp-contract=off. Failed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ".\n"
(HERE / "TABLE_L.md").write_text(out)
print(out)
