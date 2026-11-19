"""Test of readings.py on the two windows of the coincidence tissue of E63's stage A (seed 73129).

    python3 -I PREP/test_readings.py <folder holding A1.npz, A2.npz and receipt.json> <check_step1_brief.py>

The windows A1.npz and A2.npz and the receipt are TISSUE_STEP1/runs/A_coincidence_73129/ on origin/claude/e63-tissue-step1; the stream
(16,384 bytes of seed 73129) is regenerated with the functions of check_step1_brief.py (the reviewer's code of the stream, read as text:
only its definitions before part A's first print are executed) and its SHA-256 compared with the receipt's. Every reading of
readings.read_window is compared with the receipt's `measures` (written by the E63 worker's step1_read.py) and printed beside the figure
of STAGE_A_REPORT.md, which shows them to four decimals.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readings as RD                 # noqa: E402

D, CHK = Path(sys.argv[1]), Path(sys.argv[2])
rec = json.loads((D / "receipt.json").read_text())
src = CHK.read_text()
ns = {}
exec(compile(src[:src.index('print("=" * 110)')], str(CHK), "exec"), ns)
st, _, _ = ns["stream"](73129, rec["stream"]["length"])
sha = hashlib.sha256(bytes(st)).hexdigest()
print(f"stream: {len(st)} bytes, SHA-256 {sha}; the receipt's {rec['stream']['sha256']}; equal: {sha == rec['stream']['sha256']}")
worst = dict(shares=0, S=0.0, se=0.0, c0=0.0, c_lt=0.0)
for w in ("A1", "A2"):
    z = np.load(D / f"{w}.npz")
    start = int(z["positions"][0])
    assert (z["positions"] == np.arange(start, start + len(z["positions"]))).all()
    r = RD.read_window(z["x"], z["bytes"], z["distance"], 73129, stream=st, start=start)
    m = rec["measures"][w]
    print(f"\nwindow {w}: positions {start} to {start + len(z['positions']) - 1}")
    for d in (1, 2, 4):
        a, b = r["shares"][d], m["shares"][str(d)]
        eq = all(a[k] == b[k] for k in ("n", "informative", "silent", "constant"))
        worst["shares"] += 0 if eq else 1
        print(f"  share d={d}: {a['share']:.4f} of {a['n']} (silent {a['silent']}, constant {a['constant']}); receipt {b['share']:.4f} of {b['n']}; equal: {eq}")
    for d in (1, 2, 4):
        row, mr = r["scores"][d], m["scores"][str(d)]
        line = []
        for L in (0, 1, 2, 4, 8):
            a, b = row["by_lag"][L], mr["by_lag"][str(L)]
            dS, dse = abs(a["S"] - b["S"]), abs(a["se"] - b["se"])
            worst["S"] = max(worst["S"], dS); worst["se"] = max(worst["se"], dse)
            line.append(f"L={L} {a['S']:.4f} ± {a['se']:.3f} (|ΔS| {dS:.1e}, |Δse| {dse:.1e})")
        print(f"  S(d={d}), {row['sampled']} of {row['cells_at_d']} cells (receipt {mr['sampled']} of {mr['cells_at_d']}): " + "; ".join(line))
    for key, name in (("c0", "C0"), ("c_lt", "C<L")):
        mk = {"c0": "C0", "c_lt": "C_lt"}[key]
        have = m.get(mk) or m.get(name)
        parts = []
        for L, v in r[key].items():
            ref = None
            if isinstance(have, dict):
                h = have.get(str(L))
                ref = h["S"] if isinstance(h, dict) else h
            if ref is not None:
                worst[key] = max(worst[key], abs(v["S"] - ref))
            parts.append(f"L={L} {v['S']:.4f}" + ("" if ref is None else f" (receipt {ref:.4f})"))
        print(f"  {name}: " + "; ".join(parts))
print("\nlargest differences from the receipt: " + json.dumps(worst))

# the figures as STAGE_A_REPORT.md prints them (tables "2. coincidence, E10, window A1/A2"), compared to four decimals
rep = (D / "STAGE_A_REPORT.md").read_text() if (D / "STAGE_A_REPORT.md").exists() else None
if rep:
    bad, n = [], 0
    for w, head in (("A1", "#### 2. coincidence, E10, window A1"), ("A2", "#### 2. coincidence, E10, window A2")):
        sec = rep.split(head)[1].split("####")[0]
        z = np.load(D / f"{w}.npz")
        r = RD.read_window(z["x"], z["bytes"], z["distance"], 73129, stream=st, start=int(z["positions"][0]))
        for line in sec.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c[0].startswith("S(d = "):
                d = int(c[0][6]); vals = [x.split(" ± ") for x in c[1:6]]
                mine = [(f"{r['scores'][d]['by_lag'][L]['S']:.4f}", f"{r['scores'][d]['by_lag'][L]['se']:.3f}") for L in (0, 1, 2, 4, 8)]
            elif c[0] in ("C0(L)", "C<L(L)"):
                key = "c0" if c[0] == "C0(L)" else "c_lt"
                vals = [[x] for x in c[1:6]]
                mine = [((f"{r[key][L]['S']:.4f}",) if L in r[key] else ("—",)) for L in (0, 1, 2, 4, 8)]
            elif line.startswith("Shares neither silent"):
                for part in line.split(":", 1)[1].split(","):
                    dd, rest = part.split("=")[1].split(":")
                    sh, of = rest.strip().split(" of ")
                    a = r["shares"][int(dd)]
                    n += 1
                    if f"{a['share']:.4f}" != sh or str(a["n"]) != of:
                        bad.append((w, "share", dd))
                continue
            else:
                continue
            for v, m_ in zip(vals, mine):
                n += 1
                if tuple(v) != tuple(m_[:len(v)]):
                    bad.append((w, c[0], v, m_))
    print(f"STAGE_A_REPORT.md, coincidence tissue, windows A1 and A2: {n} printed figures compared, {len(bad)} differ" + (f": {bad}" if bad else ""))
