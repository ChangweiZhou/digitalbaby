"""E63 stage A: every table of STAGE_A_REPORT.md, made from the run receipts (section 11.2 and 11.4 of the brief).

    python3 step1_tables.py [runs folder]      # prints Markdown to standard output

Nothing here is measured anew: the figures are read from runs/A_*/receipt.json (written by tissue_step1.py and step1_read.py).
"""
import json
import sys
from pathlib import Path

LEVEL = 0.1103          # section 11.3 as corrected after check 9 (e): 0.0903 + 0.02
TISSUES = [("first", "A_first_73129", "1. first form, E10"),
           ("coincidence", "A_coincidence_73129", "2. coincidence, E10"),
           ("fixed", "A_fixed_73129", "3. fixed")]
F11 = "A_first_f11written_73129"
DS, LAGS = ("1", "2", "4"), ("0", "1", "2", "4", "8")


def load(root, d):
    p = root / d / "receipt.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def stays(rec):
    """(share >= 1/2, S(d,0,A2) >= LEVEL) for d = 2 and 4 in A2."""
    m = rec["measures"]["A2"]
    out = {}
    for d in ("2", "4"):
        sh = m["shares"][d]["share"]
        s = m["scores"][d]["by_lag"]["0"]
        S = s["S"] if s else None
        out[d] = (sh, S, sh is not None and sh >= 0.5, S is not None and S >= LEVEL)
    return out


def outcome(recs):
    plastic = [recs[k] for k in ("first", "coincidence")]
    if any(r is None or "measures" not in r for r in plastic):
        return None, "not yet decidable: tissue 1 or 2 not read"
    st = [stays(r) for r in plastic]
    if any(all(s[d][2] and s[d][3] for d in ("2", "4")) for s in st):
        return 1, "Cells at two and four connections stay informative"
    if all(any(not s[d][2] for d in ("2", "4")) for s in st):
        return 2, "They are silent or constant"
    return 3, "Anything else"


def main(root):
    root = Path(root)
    recs = {k: load(root, d) for k, d, _ in TISSUES}
    names = {k: n for k, _, n in TISSUES}
    P = print
    oc, words = outcome(recs)
    P(f"## Outcome (section 11.3, level {LEVEL})\n\nOutcome {oc}: {words}.\n")
    P("| Tissue | d | share neither silent nor constant, A2 | S(d, 0, A2) | share ≥ 1/2 | S ≥ level |\n|---|---|---|---|---|---|")
    for k, r in recs.items():
        if r and "measures" in r:
            for d, (sh, S, a, b) in stays(r).items():
                P(f"| {names[k]} | {d} | {f(sh)} | {f(S)} | {'yes' if a else 'no'} | {'yes' if b else 'no'} |")
    P()

    # tissue 4
    r4 = load(root, F11)
    if r4:
        P("## Tissue 4 (formula 11 as written, 512 bytes): connections after operation 7\n")
        P("| Step | connections | starting remaining | given up | made | input-line connections | from feedback cells |\n|---|---|---|---|---|---|---|")
        for c in r4["counts"]:
            lines = sum(v["n"] for kk, v in c["by_kind"].items() if "input line" in kk)
            P(f"| {c['step']} | {c['connections']} | {c['starting_remaining']} | {c['given_up_total']} | {c['made_total']} | {lines} | {c['from_feedback_cells']['n']} |")
        c129 = [c for c in r4["counts"] if c["step"] == 129][0]
        P(f"\nX0 (fewer than 300 of 15,872 starting connections remain after step 129): {c129['starting_remaining']} remain, "
          f"{'held' if c129['starting_remaining'] < 300 else 'not held'}.\n")
        P("By kind after step 129:\n\n| Kind | n | mean strength |\n|---|---|---|")
        for kk, v in c129["by_kind"].items():
            P(f"| {kk} | {v['n']} | {f(v['mean_w'])} |")
        P()

    # connections
    P("## Connections (section 11.2, item 1; every 2,048 bytes shown, every 128 bytes in the receipts)\n")
    for k, r in recs.items():
        if not r or "counts" not in r:
            continue
        P(f"### {names[k]}\n\n| Step | byte | connections | mean a cell | starting remaining | given up | made | input-line conns | line values in use | from feedback cells (w min–max) | inh → inner | exc → inner mean w |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in r["counts"]:
            if c["step"] in (127, 129, 1024) or (c["step"] + 1) % (2048 * 32) == 0:
                lines = sum(v["n"] for kk, v in c["by_kind"].items() if "input line" in kk)
                fb = c["from_feedback_cells"]
                P(f"| {c['step']} | {(c['step'] + 1) // 32 - 1 if (c['step'] + 1) % 32 == 0 else '—'} | {c['connections']} | {f(c['per_cell_mean'], 2)} | "
                  f"{c['starting_remaining']} | {c['given_up_total']} | {c['made_total']} | {lines} | "
                  f"{c['line_values_in_use']} | {fb['n']} ({f(fb['w_min'])}–{f(fb['w_max'])}) | {c['by_kind']['inner <- inhibitory cell']['n']} | {f(c['exc_into_inner_mean_w'])} |")
        P()

    # activity
    P("## Activity per stretch of 1,024 bytes (section 11.2, item 2): share neither silent nor constant (mean output, sd)\n")
    for k, r in recs.items():
        if not r or "stretches" not in r:
            continue
        P(f"### {names[k]}\n\n| Bytes | d = 1 | d = 2 | d = 4 |\n|---|---|---|---|")
        for s in r["stretches"]:
            cells = []
            for d in DS:
                x = s["d"].get(d)
                cells.append("—" if not x or not x["n"] else
                             f"{x['n']}: {f(x['informative_share'], 3)} ({x['mean_output']:.3g}, {x['sd_output']:.3g})")
            P(f"| {s['bytes'][0]}–{s['bytes'][1]} | " + " | ".join(cells) + " |")
        P()

    # scores
    P("## Scores S(d, L) (section 11.2, item 3), beside C0(L) and C<L(L)\n")
    for k, r in recs.items():
        if not r or "measures" not in r:
            continue
        for wn in ("A1", "A2"):
            m = r["measures"][wn]
            P(f"### {names[k]}, window {wn} (bytes {m['window'][0]}–{m['window'][1]})\n\n| | " + " | ".join(f"L = {L}" for L in LAGS) + " |\n|---|" + "---|" * len(LAGS))
            for d in DS:
                row = m["scores"][d]
                P(f"| S(d = {d}), {row['sampled']} of {row['cells_at_d']} cells | " +
                  " | ".join(f(row["by_lag"][L]["S"]) + (f" ± {row['by_lag'][L]['se']:.3f}" if row["by_lag"][L] else "") if row["by_lag"][L] else "—" for L in LAGS) + " |")
            P("| C0(L) | " + " | ".join(f(m["C0"][L]) for L in LAGS) + " |")
            P("| C<L(L) | — | " + " | ".join(f(m["C_lt"][L]) for L in LAGS[1:]) + " |")
            P(f"\nShares neither silent nor constant in {wn}: " + ", ".join(f"d = {d}: {f(m['shares'][d]['share'])} of {m['shares'][d]['n']}" for d in DS) + "\n")

    # cost and emissions
    P("## Cost and emissions per stretch (section 11.2, items 4 and 5)\n\n| Bytes | " +
      " | ".join(f"{names[k]}: s/byte, conns a cell, emitted" for k in recs if recs[k]) + " |\n|---|" + "---|" * sum(1 for k in recs if recs[k]))
    n = max(len(r["cpu_by_stretch"]) for r in recs.values() if r)
    for i in range(n):
        row, b = [], None
        for k, r in recs.items():
            if not r:
                continue
            if i < len(r["cpu_by_stretch"]):
                c = r["cpu_by_stretch"][i]; b = c["bytes"]
                row.append(f"{c['per_byte']:.4f}, {c['mean_connections']:.2f}, {c['emitted_share']:.4f}")
            else:
                row.append("—")
        P(f"| {b[0]}–{b[1]} | " + " | ".join(row) + " |")
    P()

    # X2: the starting connections from feedback cells, from the final saved state (a connection given up never returns with
    # birth step 0, so alive at the end means alive at every count)
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tissue_step1 as T
    FB = np.array(T.birth(73129)["FB"])
    P("## Starting connections from feedback cells at the end (X2)\n\n| Tissue | starting ones alive | strengths |\n|---|---|---|")
    for k, d, nm in TISSUES + [("f11", F11, "4. formula 11 as written")]:
        p = root / d / "final_state.npz"
        if not p.exists():
            continue
        z = np.load(p)
        a = z["alive"].astype(bool) & np.isin(z["snd"], FB) & (z["k0"] == 0)
        P(f"| {nm} | {int(a.sum())} of 423 | {', '.join(f'{x:.4f}' for x in np.unique(z['w'][a])) or '—'} |")
    P()

    # last 4096 and forecast
    P("## Cost over the last 4,096 bytes and the forecast for stage B round 1 (section 11.4)\n\n| Tissue | s/byte, bytes 12,288–16,383 | mean connections a cell |\n|---|---|---|")
    last = {}
    for k, r in recs.items():
        if r and len(r["cpu_by_stretch"]) >= 16:
            cs = r["cpu_by_stretch"][-4:]
            last[k] = (sum(c["cpu_s"] for c in cs) / 4096, sum(c["mean_connections"] for c in cs) / 4)
            P(f"| {names[k]} | {last[k][0]:.4f} | {last[k][1]:.2f} |")
    if "fixed" in last and ("first" in last or "coincidence" in last):
        pl = max(last[k][0] for k in ("first", "coincidence") if k in last) * 1.25
        h = (6 * pl + 3 * last["fixed"][0]) * 262144 / 3600
        P(f"\nBrief's rule: plastic cost = max(first, coincidence) × 1.25 = {pl:.4f} s/byte; "
          f"262,144 × (6 × {pl:.4f} + 3 × {last['fixed'][0]:.4f}) = {h:.1f} CPU-hours.")
    P()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runs")
