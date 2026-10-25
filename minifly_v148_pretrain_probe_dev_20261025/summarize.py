"""V148 summary: tables from a RESULTS.json and, for the main run, scoring of the registered predictions as
operationalised in LOCK.json.  python summarize.py results/<tag>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x):
    return "–" if x is None else f"{x:.3f}"


def main():
    d = Path(sys.argv[1])
    r = json.loads((d / "RESULTS.json").read_text())
    if not r.get("complete"):
        raise SystemExit("RESULTS incomplete")
    lines = [f"# V148 summary: {r['tag']} ({r['text']})", "",
             "| Taught KB | Pass | MiniFly held bits/byte | T | held acc | taught first 5 KB | taught last 5 KB | "
             "never-seen mass | renorm-seen held (diag.) | unigram | bigram | PPM-2 | PPM-3 | PPM-4 | PPM-5 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in r["checkpoints"]:
        m, L = c["minifly"], c["ladder"]
        lines.append(f"| {c['kb']:g} | {c.get('pass', 1)} | {f(m['held_bpb'])} | {m['T']:.2f} | {f(m['held_acc'])} | "
                     f"{f(m.get('taught_first5k_bpb'))} | {f(m.get('taught_last5k_bpb'))} | "
                     f"{f(m.get('held_mass_on_unseen_symbols'))} | {f(m.get('held_bpb_renorm_seen'))} | "
                     + " | ".join(f(L[k]['held_bpb']) for k in ("unigram", "bigram", "ppmc2", "ppmc3", "ppmc4", "ppmc5"))
                     + " |")
    gens = next((c["minifly"]["generations"] for c in r["checkpoints"] if "generations" in c["minifly"]), [])
    if gens:
        own = [g["own"] for g in gens]
        printable = sum(ch.isascii() and (ch.isprintable() or ch == " ") for s in own for ch in s) / max(1, sum(map(len, own)))
        lines += ["", f"Own generations: {len(gens)}; stopped at own EOS: {sum(g['stop'] == 'EOS' for g in gens)}; "
                  f"mean length {sum(map(len, own)) / len(own):.1f} bytes; printable-ASCII fraction {printable:.3f}.",
                  "", "First five (prefix → own continuation | true continuation):"]
        for g in gens[:5]:
            lines.append(f"* `{g['prefix']!r}` → `{g['own'][:50]!r}` | `{g['true'][:50]!r}`")
    out = {"tag": r["tag"], "text": r["text"]}
    lock_path = HERE / "LOCK.json"
    if r["tag"] == "main" and lock_path.exists():
        lock = json.loads(lock_path.read_text())
        P = lock["predictions"]
        top = max(r["prefixes_kb"])
        lo = min(k for k in r["prefixes_kb"] if k > 0)
        cp = {(c["kb"], c.get("pass", 1)): c for c in r["checkpoints"]}
        e = cp[(top, 1)]
        mf, L = e["minifly"]["held_bpb"], e["ladder"]
        best_hi = min(L[k]["held_bpb"] for k in ("ppmc3", "ppmc4", "ppmc5"))
        if mf < best_hi:
            p1 = "WEAKENED (better than predicted: beats PPM order 3 or above)"
        elif mf <= L["bigram"]["held_bpb"]:
            p1 = "AS PREDICTED (between PPM-3 and bigram)"
        else:
            p1 = "NOT AS PREDICTED (worse than bigram)"
        g_last = mf - e["minifly"]["taught_last5k_bpb"]
        g_first = mf - e["minifly"]["taught_first5k_bpb"]
        p2 = ("AS PREDICTED" if min(g_last, g_first) >= P["P2_gap_bits"] else
              "NOT AS PREDICTED (little memorised)" if g_last < P["P2_gap_bits"] else
              "NOT AS PREDICTED (early text erased)")
        d_mf = cp[(lo, 1)]["minifly"]["held_bpb"] - mf
        d_pp = {k: cp[(lo, 1)]["ladder"][k]["held_bpb"] - L[k]["held_bpb"] for k in ("ppmc2", "ppmc3", "ppmc4", "ppmc5")}
        p3 = "AS PREDICTED" if d_mf < min(d_pp.values()) else "WEAKENED (MiniFly curve as steep as a PPM rung or steeper)"
        out["predictions"] = {"P1": {"result": p1, "minifly": mf, "bigram": L["bigram"]["held_bpb"],
                                     "ppmc3": L["ppmc3"]["held_bpb"], "best_ppm_order3plus": best_hi},
                              "P2": {"result": p2, "held_minus_taught_last5k": g_last,
                                     "held_minus_taught_first5k": g_first, "threshold": P["P2_gap_bits"]},
                              "P3": {"result": p3, "minifly_improvement": d_mf,
                                     **{k + "_improvement": v for k, v in d_pp.items()}, "range_kb": [lo, top]}}
        lines += ["", "## Registered predictions (design section 7; operationalised in LOCK.json before the run)"]
        for k, v in out["predictions"].items():
            lines.append(f"* {k}: {v['result']} — " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}"
                                                                 for a, b in v.items() if a != "result"))
    (d / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (d / "SUMMARY.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
