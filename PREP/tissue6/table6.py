"""Table of the drive6.py runs (out/*.json), by E52's worker. Standard library only.
    python3 -I table6.py      # writes TABLE6.md
One row a window (A2 for 16,384 bytes; Q1..Q4 for longer lives): connections a cell at the window's end (mean / largest); made and
withdrawn by then; S(d, 0) at d = 1, 2, 4, 8, 10 (** above 0.1103); S(d, 1) at d = 1, 2, 4 (** above 0.090); cells at d = 8 and 10 with
any output / all cells at that distance; with a largest output above 1/2; the divergence of two copies after 1 / 16 / 512 bytes (read
before the last window only) and the check of the copy; ms a byte."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
DS = ("1", "2", "4", "8", "10")
f = lambda v, n=3: "–" if v is None else f"{v:.{n}f}"
m = lambda v, l: f(v, 3) + ("**" if v is not None and v > l else "")
L = ["| Tissue | window | conns mean / max | made | withdrawn | S(d,0) d1 / 2 / 4 / 8 / 10 | S(d,1) d1 / 2 / 4 | any output d8, d10 | above 1/2 d8, d10 | divergence 1 / 16 / 512 (copy) | ms a byte |",
     "|" + "---|" * 11]
for p in sorted((HERE / "out").glob("*.json")):
    r = json.loads(p.read_text())
    for w in r["windows"] if len(r["windows"]) > 2 else r["windows"][-1:]:
        a = r[w]; e = a["at_end"]; dv = r["divergence"] if w == r["windows"][-1] else None
        L.append(f"| {r['label']} | {w} | {e['mean']:.1f} / {e['max']} | {e['made']} | {e['given']} | " + " / ".join(m(a['S'][d]['0'], 0.1103) for d in DS)
                 + " | " + " / ".join(m(a['S'][d]['1'], 0.090) for d in ("1", "2", "4"))
                 + f" | {a['cells_any_output']['8']}/{a['cells']['8']}, {a['cells_any_output']['10']}/{a['cells']['10']} | {a['cells_above_half']['8']}, {a['cells_above_half']['10']} | "
                 + (f"{dv['1']:.0e} / {dv['16']:.0e} / {dv['512']:.0e} ({r['copy_check_max_diff']:.0e})" if dv else "") + f" | {r['ms_per_byte']:.1f} |")
fl = HERE / "logs" / "FAILED.txt"
L.append("\nFailed runs: " + (fl.read_text().strip() if fl.exists() else "none") + ". ** : above the level (lag 0: 0.1103; lag 1: 0.090).")
(HERE / "TABLE6.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
