# Copy of PREP/tissue4/grid_table4.py, for the runs of this folder.
"""Table of the drive4.py runs (out/*.json). Standard library only.

    python3 grid_table4.py [prefix ...]     # writes MAKE_TABLE.md (rows whose label starts with a prefix given, or all)

One row a tissue, last window (A2 for 16,384 bytes; Q1..Q4 rows for longer lives): connections a cell at the window's end (mean and
largest, non-feedback cells); made, withdrawn, withdrawn by use (whole run); share neither silent nor constant and S(d, 0) at d = 1, 2, 4,
8, 10; S(d, 1) at d = 1, 2, 4; c0 at lags 1, 2; ms a byte. ** marks a score above the level (lag 0: 0.1103; lag 1: 0.090).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = ("1", "2", "4", "8", "10")


def f(v, n=3):
    return "–" if v is None else f"{v:.{n}f}"


def m(v, lev):
    return f(v, 4) + ("**" if v is not None and v > lev else "")


def main(prefixes):
    rows = [json.loads(p.read_text()) for p in sorted((HERE / "out").glob("*.json"))]
    rows = [r for r in rows if not prefixes or any(r["label"].startswith(p) for p in prefixes)]
    L = ["| Tissue | window | conns a cell mean / max | made | withdrawn (by use) | share d1/2/4/8/10 | S(d,0) d1 / 2 / 4 / 8 / 10 | S(d,1) d1 / 2 / 4 | c0 lag 1 / 2 | ms a byte | all five S(d,0) above |",
         "|" + "---|" * 11]
    allfive = []
    for r in rows:
        for w in r.get("windows", ["A2"])[-1:] if len(r.get("windows", [])) <= 2 else r["windows"]:
            a = r[w]; e = a["at_end"]
            ok = all(a["S"][d]["0"] is not None and a["S"][d]["0"] > 0.1103 for d in DS)
            if ok and w == r["windows"][-1]:
                allfive.append(r["label"])
            L.append(f"| {r['label']} | {w} | {e['mean']:.1f} / {e['max']} | {e['made']} | {e['given']} ({e['given_by_use']}) | "
                     + " / ".join(f(a["share"][d], 2) for d in DS) + " | " + " / ".join(m(a["S"][d]["0"], 0.1103) for d in DS) + " | "
                     + " / ".join(m(a["S"][d]["1"], 0.090) for d in ("1", "2", "4")) + " | " + f(a["c0"].get("1"), 4) + " / " + f(a["c0"].get("2"), 4)
                     + f" | {r['ms_per_byte']:.1f} | {'yes' if ok else 'no'} |")
    failed = HERE / "logs" / "FAILED.txt"
    if failed.exists():
        L.append("\n**Failed runs**: " + ", ".join(x for x in failed.read_text().split() if x != "FAILED"))
    L.append(f"\n** : above the level (lag 0: 0.1103; lag 1: 0.090). Tissues: {len(rows)}. S(d, 0) above 0.1103 at all of d = 1, 2, 4, 8, 10 in the last window: "
             f"{len(allfive)}: " + ", ".join(allfive))
    (HERE / "MAKE_TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return allfive


if __name__ == "__main__":
    main(sys.argv[1:])
