"""Part (b): tables of the new program's runs beside E63's stage A (TISSUE_STEP1/runs, read only). Writes PART_B_TABLES.md.

    python3 b_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "TISSUE_STEP1" / "runs"
NEW = HERE / "runs"
TISS = (("first", "A_first_73129", "B_first_on"), ("coincidence", "A_coincidence_73129", "B_coincidence_on"),
        ("fixed", "A_fixed_73129", "B_fixed_on"))
SINGLE = (("floor 1/128", "S_first_floor"), ("tolerance 1/1024", "S_first_tol"), ("one trace a sender", "S_first_trace"),
          ("strengths once a byte", "S_first_strength"))


def load(p):
    return json.loads((p / "receipt.json").read_text())


def f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def old_cost(r, lo, hi):
    rows = [c for c in r["cpu_by_stretch"] if lo <= c["bytes"][0] and c["bytes"][1] < hi]
    return sum(c["cpu_s"] for c in rows) / sum(c["bytes"][1] - c["bytes"][0] + 1 for c in rows)


def conns_end(r):
    return r["counts"][-1]["connections"] / 2048


def stretch_row(r, q):
    s = r["stretches"][q]["d"]
    return " / ".join(f(s[d]["informative_share"], 3) for d in ("1", "2", "4"))


def main():
    out = []
    P = out.append
    P("# Part (b): tables (written by b_tables.py from the receipts)\n")
    P("E63 = E63's program, stage A receipts on `origin/claude/e63-tissue-step1` (TISSUE_STEP1/runs). New = this program with floor 1/128, "
      "tolerance 1/1024, one trace a sender, strengths once a byte. Seed 73129, 16,384 bytes. CPU-seconds a byte with E63's records "
      "and readings; for the new program also the core alone (the same tissue run again without records). Cells sending: senders "
      "passing on a change, a step (the new program counts it; E63's program does not, so —). Connection visits a byte: for E63's "
      "program 32 times its connections at the end (it visits every connection at every step, by construction, not counted); for the "
      "new program counted by the core.\n")
    P("| Tissue | Program | CPU-s a byte (records) | CPU-s a byte (core alone) | Connections a cell, end | Cells sending a step | Connection visits a byte | Cell updates a byte | Emission share |")
    P("|---|---|---|---|---|---|---|---|---|")
    for t, od, nd in TISS:
        o, n = load(OLD / od), load(NEW / nd)
        P(f"| {t} | E63 | {f(o['cpu_total_s'] / o['stream']['length'], 4)} | — | {f(conns_end(o), 2)} | — | {32 * o['counts'][-1]['connections']:,} (by construction) | — | {f(o['emissions']['share'], 4)} |")
        P(f"| {t} | new | {f(n['cpu_per_byte_with_records'], 5)} | {f(n['cpu_per_byte_core_only'], 5)} | {f(n['connections_a_cell_end'], 2)} | {f(n['mean_cells_sending_a_step'], 1)} | {n['connection_visits_a_byte']:,.0f} | {n['cell_updates_a_byte']:,.0f} | {f(n['emissions']['share'], 4)} |")
    P("\n## Readings of section 11.2 of the brief of step 1\n")
    P("### Connections (item 1): after operation 7 of steps 127, 129 and 1,024, and at the end of bytes 4,095, 8,191, 12,287 and 16,383\n")
    P("| Tissue | Program | " + " | ".join(["127", "129", "1024", "b 4095", "b 8191", "b 12287", "b 16383"]) + " | given up, end | made, end | starting kept, end |")
    P("|---|---|" + "---|" * 10)
    for t, od, nd in TISS:
        for name, r in (("E63", load(OLD / od)), ("new", load(NEW / nd))):
            by = {c["step"]: c for c in r["counts"]}
            steps = [127, 129, 1024] + [32 * b + 31 for b in (4095, 8191, 12287, 16383)]
            cells = [f"{by[s]['connections']:,}" if s in by else "—" for s in steps]
            last = r["counts"][-1]
            P(f"| {t} | {name} | " + " | ".join(cells) + f" | {last['given_up_total']:,} | {last['made_total']:,} | {last['starting_remaining']:,} |")
    P("\n### Activity (item 2): share of the cells at d = 1 / 2 / 4 neither silent nor constant, by stretch of 1,024 bytes\n")
    P("| Tissue | Program | " + " | ".join(f"{q * 1024}" for q in range(16)) + " |")
    P("|---|---|" + "---|" * 16)
    for t, od, nd in TISS:
        for name, r in (("E63", load(OLD / od)), ("new", load(NEW / nd))):
            P(f"| {t} | {name} | " + " | ".join(stretch_row(r, q) for q in range(16)) + " |")
    P("\n### Scores (item 3): S(d, L) in windows A1 and A2, and the shares of section 7.2 in the window\n")
    P("| Tissue | Program | Window | share d=1 / 2 / 4 | S(1,0) | S(2,0) | S(4,0) | S(2,1) | S(2,2) | S(2,4) | S(2,8) | C0(0) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, od, nd in TISS:
        for name, r in (("E63", load(OLD / od)), ("new", load(NEW / nd))):
            for wn in ("A1", "A2"):
                m = r["measures"][wn]
                sh = " / ".join(f(m["shares"][d]["share"], 3) for d in ("1", "2", "4"))

                def S(d, L):
                    v = m["scores"][d]["by_lag"].get(str(L)) if m["scores"].get(d) else None
                    return f(v["S"], 4) if v else "—"
                P(f"| {t} | {name} | {wn} | {sh} | {S('1', 0)} | {S('2', 0)} | {S('4', 0)} | {S('2', 1)} | {S('2', 2)} | {S('2', 4)} | {S('2', 8)} | {f(m['C0']['0'], 4)} |")
    P("\n### Cost by stretch (item 4), CPU-seconds a byte with records\n")
    P("| Tissue | Program | " + " | ".join(f"{q * 1024}" for q in range(16)) + " |")
    P("|---|---|" + "---|" * 16)
    for t, od, nd in TISS:
        for name, r in (("E63", load(OLD / od)), ("new", load(NEW / nd))):
            P(f"| {t} | {name} | " + " | ".join(f(c["per_byte"], 4 if name == 'E63' else 5) for c in r["cpu_by_stretch"]) + " |")
    P("\n## Each switch alone, first-form tissue, 4,096 bytes\n")
    P("The other three switches off. Scores need window A1 (bytes 2,048 to 6,143), which a run of 4,096 bytes does not complete, so "
      "none are given; the activity readings of the four stretches and the connections are given. E63's row is the first 4,096 bytes "
      "of its stage A run of the first-form tissue.\n")
    P("| Setting | CPU-s a byte (records) | CPU-s a byte (core alone) | Connections a cell, end | Cells sending a step | Connection visits a byte | Cell updates a byte | given up | made | d=1/2/4 informative, stretch 0 | stretch 1 | stretch 2 | stretch 3 |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    o = load(OLD / "A_first_73129")
    c4 = {c["step"]: c for c in o["counts"]}[32 * 4095 + 31]
    P(f"| E63 (all off) | {f(old_cost(o, 0, 4096), 4)} | — | {f(c4['connections'] / 2048, 2)} | — | {32 * c4['connections']:,} (by construction) | — | "
      f"{c4['given_up_total']:,} | {c4['made_total']:,} | " + " | ".join(stretch_row(o, q) for q in range(4)) + " |")
    for lab, nd in SINGLE:
        n = load(NEW / nd)
        last = n["counts"][-1]
        P(f"| {lab} | {f(n['cpu_per_byte_with_records'], 5)} | {f(n['cpu_per_byte_core_only'], 5)} | {f(n['connections_a_cell_end'], 2)} | "
          f"{f(n['mean_cells_sending_a_step'], 1)} | {n['connection_visits_a_byte']:,.0f} | {n['cell_updates_a_byte']:,.0f} | {last['given_up_total']:,} | "
          f"{last['made_total']:,} | " + " | ".join(stretch_row(n, q) for q in range(4)) + " |")
    n = load(NEW / "B_first_on")
    last = {c["step"]: c for c in n["counts"]}[32 * 4095 + 31]
    P(f"| all four on (first 4,096 bytes of the run of 16,384) | {f(old_cost(n, 0, 4096), 5)} | — | {f(last['connections'] / 2048, 2)} | — | — | — | "
      f"{last['given_up_total']:,} | {last['made_total']:,} | " + " | ".join(stretch_row(n, q) for q in range(4)) + " |")
    (HERE / "PART_B_TABLES.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
