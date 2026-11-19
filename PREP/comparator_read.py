"""Read the comparator's records (comparator_runs/comparator_<seed>_<N>.npz) by kind of due position, in windows of 16,384 positions.

The kinds are the labels of life_stream.py: the due positions SIGN, EQ, END (phase KNOWABLE), RECALL by gap 1, 2, 4, 8 (phase DELAY), ANSWER1
and ANSWER2 (the table phases); and, in phase FIRST, where nothing is due, the first family's two answer digits (classes R1 and R2). The
share is the share of positions of the kind whose given byte equals the byte shown (right at the first attempt).

Windows: each phase is cut into consecutive windows of 16,384 positions from its start; the *last window* of a phase is its last 16,384
positions (or the whole phase if it is shorter). This program was written because the shared scorer (PREP/score_emissions.py) was not yet
published; it uses only the labels of life(seed).

    python3 comparator_read.py              # all seeds with a whole-life record; writes comparator_runs/COMPARATOR_READ.json and prints the table
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import life_stream as LS          # noqa: E402

W = 16384


def kinds(L):
    ph, cls, due, gap = L["phase"], L["cls"], L["due"], L["gap"]
    k = {"R1 (first family, phase FIRST)": (ph == 0) & (cls == 9), "R2 (first family, phase FIRST)": (ph == 0) & (cls == 10),
         "SIGN": due == 1, "EQ": due == 2, "END": due == 3}
    for g in (1, 2, 4, 8):
        k[f"RECALL gap {g}"] = (due == 4) & (gap == g)
    k["ANSWER1"] = due == 5; k["ANSWER2"] = due == 6
    k["ANSWER1, key first asked"] = (due == 5) & L["first"]; k["ANSWER2, key first asked"] = (due == 6) & L["first"]
    return k


def read(path):
    d = np.load(path); seed = int(Path(path).stem.split("_")[1])
    L = LS.life(seed); n = len(d["given"])
    assert n == len(L["bytes"]) and np.array_equal(d["shown"], L["bytes"]), "the record is not of this life"
    right = d["given"] == d["shown"]
    K = kinds(L); out = {}
    for name, s, e in L["spans"]:
        if name == "QUIET":
            continue
        wins = [(a, min(a + W, e)) for a in range(s, e, W)] + [(max(s, e - W), e)]
        rows = []
        for a, b in wins:
            row = dict(start=a, end=b)
            for kn, m in K.items():
                mm = m[a:b]; c = int(mm.sum())
                if c:
                    row[kn] = dict(n=c, right=int(right[a:b][mm].sum()), share=float(right[a:b][mm].mean()))
            rows.append(row)
        out[f"{name} {s}-{e}"] = dict(windows=rows[:-1], last_window=rows[-1])
    return seed, dict(record=Path(path).name, bytes=n, right_share_all=float(right.mean()), phases=out)


def main():
    res = {}
    for p in sorted(glob.glob(str(HERE / "comparator_runs" / "comparator_*_7*.npz"))):
        seed, r = read(p); res[seed] = r
    (HERE / "comparator_runs" / "COMPARATOR_READ.json").write_text(json.dumps(res, indent=1))
    seeds = sorted(res)
    print("Share right at the first attempt in the last window (last 16,384 positions) of each phase; n = number of positions of the kind in that window.")
    print("| phase | kind | " + " | ".join(f"seed {s}" for s in seeds) + " |")
    print("|---|---|" + "---|" * len(seeds))
    first = res[seeds[0]]["phases"]
    for ph in first:
        for kn in first[ph]["last_window"]:
            if kn in ("start", "end"):
                continue
            cells = []
            for s in seeds:
                key = [k for k in res[s]["phases"] if k.split()[0] == ph.split()[0]][0]
                v = res[s]["phases"][key]["last_window"].get(kn)
                cells.append(f"{100 * v['share']:.1f}% (n {v['n']})" if v else "-")
            print(f"| {ph.split()[0]} | {kn} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
