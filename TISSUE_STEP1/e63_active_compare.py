"""E63, decision 66: the copy against the original at byte 2,048, and the table of timings, from runs_active/. Read-only.

    python3 e63_active_compare.py [--prune]   # writes runs_active/COMPARE.json and runs_active/COMPARE.md

--prune deletes a variant's final.npz once it has been found equal to the original's (its SHA-256 is kept in COMPARE.json), for want of
disk; the originals' saved states are kept. A comparison already recorded for a pruned variant is carried over from COMPARE.json.

Compared, for each variant against the original of the same tissue: the connections alive at byte 2,048 (sender, receiver, step of
making); the emissions (bytes 0 to 2,047 for abc, 1,024 to 2,047 for the variants started at byte 1,024); every array of the saved
state (strengths, traces, states, thresholds, averages, and the fifth tissue's use, trace average and counts), with ==; and the state of
the generator.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs_active"
TISSUES = ("fixed", "first", "coincidence", "first_f11written", "first_withdrawbyuse")
VARIANTS = ("abc", "a", "b", "c", "abcm")


def conns(d):
    a = d["alive"]
    return sorted(zip(d["snd"][a].tolist(), d["rcv"][a].tolist(), d["k0"][a].tolist()))


def main(prune=False):
    old = json.loads((OUT / "COMPARE.json").read_text()) if (OUT / "COMPARE.json").exists() else {}
    res = {}
    L = ["| Tissue | Variant | CPU-s a byte, bytes 1,024-2,047 | Ratio to the original | Live connections (mean) | Cells with non-zero output a step (mean) | "
         "Fifth tissue's lines, CPU-s a byte (use update; withdrawal test) | Equal to the original at byte 2,048 |", "|" + "---|" * 8]
    for t in TISSUES:
        po = OUT / t / "orig"
        if not (po / "receipt.json").exists():
            continue
        ro = json.loads((po / "receipt.json").read_text()); fo = np.load(po / "final.npz")
        L.append(f"| {t} | original | {ro['cpu_s_per_byte_1024_2047']:.4f} | 1 | {ro['mean_live_connections_1024_2047']:.0f} | "
                 f"{ro['mean_nonzero_cells_per_step_1024_2047']:.1f} | – | – |")
        res[t] = dict(orig=ro)
        for v in VARIANTS:
            pv = OUT / t / v
            if not (pv / "receipt.json").exists():
                continue
            rv = json.loads((pv / "receipt.json").read_text())
            s0 = rv["start_byte"]
            if not (pv / "final.npz").exists():                  # pruned earlier: carry the recorded comparison over
                prev = old.get(t, {}).get(v)
                assert prev and prev["comparison"]["equal"], f"no final state and no recorded equal comparison: {pv}"
                res[t][v] = prev; cmp = prev["comparison"]; ok = True
            else:
                fv = np.load(pv / "final.npz")
                cmp = None
            if cmp is None:
              keys = sorted(k for k in fo.files if k not in ("acc", "emis", "rl_state"))
              cmp = dict(same_keys=sorted(fo.files) == sorted(fv.files), connections=conns(fo) == conns(fv),
                       emissions=bool(np.array_equal(fo["emis"][s0:], fv["emis"][s0:])),
                       generator=bool(np.array_equal(fo["rl_state"], fv["rl_state"])),
                       arrays={k: bool(np.array_equal(fo[k], fv[k])) for k in keys},
                       final_sha256=hashlib.sha256((pv / "final.npz").read_bytes()).hexdigest(),
                       original_final_sha256=hashlib.sha256((po / "final.npz").read_bytes()).hexdigest())
              ok = cmp["same_keys"] and cmp["connections"] and cmp["emissions"] and cmp["generator"] and all(cmp["arrays"].values())
              cmp["equal"] = bool(ok)
              res[t][v] = dict(receipt=rv, comparison=cmp)
              if prune and ok:
                  (pv / "final.npz").unlink()
            tu = rv.get("fifth_tissue_lines_cpu_s_per_byte_1024_2047")
            tu_s = f"{tu[0]:.4f}; {tu[1]:.4f}" if tu and t == "first_withdrawbyuse" else "–"
            L.append(f"| {t} | {v} (from byte {s0}) | {rv['cpu_s_per_byte_1024_2047']:.4f} | "
                     f"{rv['cpu_s_per_byte_1024_2047'] / ro['cpu_s_per_byte_1024_2047']:.3f} | {rv['mean_live_connections_1024_2047']:.0f} | "
                     f"{rv['mean_nonzero_cells_per_step_1024_2047']:.1f} | {tu_s} | {'yes' if ok else 'NO: ' + ', '.join(k for k, x in cmp['arrays'].items() if not x)} |")
    (OUT / "COMPARE.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "COMPARE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main("--prune" in sys.argv)
