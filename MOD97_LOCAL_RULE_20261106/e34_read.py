"""E34 reader (brief OPUS_TASK_E34_ENTROPY_WEIGHTED_ERROR_20261115.md, revision 2, "Reading the files"). Read-only; trains nothing.

    python3 e34_read.py reproduce     S-all's final file 75.33 / 2.0 and E26 arm S's 98.33 / 96.0 (12 / 40 letters) -> exp45/e34/E34_READ_REPRO.json
    python3 e34_read.py read          every saved row of every complete full arm (T050, T100, T200; seed 73129)    -> exp45/e34/E34_READ.json
    python3 e34_read.py tables        endpoints, marks, outcome row, the four facts, weight diagnostics            -> exp45/e34/E34_READ_TABLES.md

panels, measure, load_net and row_files are imported from e32_read unchanged.

Addendum 1 (stage 2): read takes an optional seed (default 73129; behaviour of `tables` for 73129 unchanged).
    python3 e34_read.py read <seed>   seed 73129 also reads B050; seeds 93683 and 116089 read T050   -> E34_READ.json / E34_READ_<seed>.json
    python3 e34_read.py tables_a1     per-seed table, the two facts, the B050 reading                -> exp45/e34/E34_ADDENDUM1_TABLES.md
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e32_read as R32               # noqa: E402  (imports e29_gates -> exp20_reference: environment gate)
from e32_read import panels, measure, load_net, row_files  # noqa: E402

T, X16, E, arch26 = R32.T, R32.X16, R32.E, R32.arch26
OUT = HERE / "exp45" / "e34"
SEED = 73129
ARMS = ("T050", "T100", "T200")
TAU = {"T050": 0.5, "T100": 1.0, "T200": 2.0}
ROWS, END_ROWS = R32.ROWS, R32.END_ROWS
FINE = ("distractor", "operand", "code", "operator", "equals", "answer_digit", "eos")
KEYS = ("acc12", "acc20", "acc40", "keepers12", "keepers20", "keepers40", "kw12", "kw20", "kw40")
BASELINE_ACC12, BALL_ACC12, BALL_ACC40 = 59.6, 92.8, 82.0
PROJ = {"T050": dict(ratio="0.0014 to 0.0019", shares=dict(distractor=6, operand=26, code=28, answer_digit=39)),
        "T100": dict(ratio="0.039 to 0.049", shares=dict(distractor=30, operand=42, code=17, answer_digit=11)),
        "T200": dict(ratio="0.20 to 0.23", shares=dict(distractor=47, operand=40, code=9, answer_digit=4))}


A1_SEEDS = (73129, 93683, 116089)


def set_seed(s_):
    global SEED
    SEED = s_


def read_path(seed=None):
    s_ = SEED if seed is None else seed
    return OUT / ("E34_READ.json" if s_ == 73129 else f"E34_READ_{s_}.json")


def read_arms():
    return ARMS + ("B050",) if SEED == 73129 else ("T050",)


def receipt(arm, seed=None):
    return OUT / f"e34_dev_{arm}_{SEED if seed is None else seed}.json"


def reproduce():
    arch26.install()
    want = [("E28 S-all final", HERE / "exp45" / "e28" / "e28_dev_S-all_73129.json", "e28_dev_S-all_73129.pt", 75.33, 2.0),
            ("E26 S final", HERE / "exp45" / "e26" / "e26_dev_S_73129.json", "e26_dev_S_73129.pt", 98.33, 96.0)]
    out, ok = dict(rows=[]), True
    pk = panels(SEED)
    for name, rp, f, a12, a40 in want:
        rec = json.loads(rp.read_text())
        assert rec["weights"] == f
        m, sha = load_net(rp.parent / f, rec["weights_sha256"], SEED)
        r = measure(m, SEED, pk)
        same = r["acc12"] == a12 and r["acc40"] == a40
        ok &= same
        out["rows"].append(dict(name=name, file=str((rp.parent / f).relative_to(HERE)), sha256=sha, want=dict(acc12=a12, acc40=a40),
                                reader={k: r[k] for k in KEYS}, exact=same))
        print(name, "exact" if same else "DIFFERS", r["acc12"], r["acc20"], r["acc40"], flush=True)
    out.update(passed=ok, python=sys.version.split()[0], torch=T.__version__, rule="accuracy at 12 and 40 letters equal to the brief's figures as rounded")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E34_READ_REPRO.json").write_text(json.dumps(out, indent=1))
    print("reproduction", "PASSED" if ok else "FAILED")
    return ok


def read():
    rp = OUT / "E34_READ_REPRO.json"
    assert rp.exists() and json.loads(rp.read_text())["passed"] is True, "reproduction has not passed"
    arch26.install()
    path = read_path()
    done = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    have = {(r["arm"], r["row"], r["sha256"]) for r in done["rows"]}
    pk = panels(SEED)
    for arm in read_arms():
        if not receipt(arm).exists() or not json.loads(receipt(arm).read_text()).get("complete"):
            continue
        rec, files = row_files(receipt(arm))
        for row in ROWS:
            f, want = files[row]
            if (arm, row, want) in have:
                continue
            t0 = time.process_time()
            m, sha = load_net(f, want, SEED)
            r = dict(arm=arm, seed=SEED, row=row, file=str(f.relative_to(HERE)), sha256=sha, **measure(m, SEED, pk))
            r["cpu_s"] = round(time.process_time() - t0, 2)
            done["rows"].append(r)
            path.write_text(json.dumps(done, indent=1))
            print(f"{arm} {row}: acc {r['acc12']:.1f} {r['acc20']:.1f} {r['acc40']:.1f} | keepers {r['keepers12']} {r['keepers20']} {r['keepers40']}"
                  f" | kw {r['kw12']} {r['kw20']} {r['kw40']}", flush=True)
    done.update(python=sys.version.split()[0], torch=T.__version__)
    path.write_text(json.dumps(done, indent=1))


# ---------------------------------------------------------------- marks, outcome, facts
def endpoints(rows):
    out = {}
    for arm in ARMS:
        rs = {r["row"]: r for r in rows if r["arm"] == arm}
        if not all(k in rs for k in END_ROWS):
            continue
        out[arm] = dict(mean={k: sum(rs[s][k] for s in END_ROWS) / 3 for k in KEYS}, final={k: rs[24576][k] for k in KEYS})
    return out


def mark(e):
    if e["acc12"] < 70:
        return "not learned in range"
    if e["acc40"] >= 60:
        return "extrapolates"
    if e["acc40"] <= 20:
        return "fails"
    return "intermediate"


def outcome(M):
    learned = [a for a in ARMS if M[a] != "not learned in range"]
    if any(M[a] == "extrapolates" for a in ARMS):
        return "1", "arms that learned in range and extrapolate: " + ", ".join(a for a in ARMS if M[a] == "extrapolates")
    if any(M[a] == "intermediate" for a in ARMS):
        return "2", "intermediate: " + ", ".join(a for a in ARMS if M[a] == "intermediate")
    if learned:
        return "3", "learned in range and fail: " + ", ".join(learned)
    return "4", "no arm learned in range"


def tables():
    rows = json.loads((OUT / "E34_READ.json").read_text())["rows"]
    Ep = endpoints(rows)
    assert set(Ep) == set(ARMS), f"endpoints for {sorted(Ep)} only"
    M = {a: mark(Ep[a]["mean"]) for a in ARMS}
    row, detail = outcome(M)
    recs = {a: json.loads(receipt(a).read_text()) for a in ARMS}
    L = ["# E34 reader tables (generated by `e34_read.py tables` from `E34_READ.json` and the receipts)", "",
         "Seed 73129. Endpoint = mean of rows 16,384, 20,480, 24,576; the final row (24,576) is in brackets.", "",
         "| Arm | tau | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |", "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        m_, f_ = Ep[a]["mean"], Ep[a]["final"]
        L.append(f"| {a} | {TAU[a]} | {m_['acc12']:.2f} ({f_['acc12']:.2f}) | {m_['acc20']:.2f} ({f_['acc20']:.2f}) | {m_['acc40']:.2f} ({f_['acc40']:.2f}) | "
                 f"{m_['keepers12']:.1f} / {m_['keepers20']:.1f} / {m_['keepers40']:.1f} ({f_['keepers12']} / {f_['keepers20']} / {f_['keepers40']}) | "
                 f"{m_['kw12']:.1f} / {m_['kw20']:.1f} / {m_['kw40']:.1f} ({f_['kw12']} / {f_['kw20']} / {f_['kw40']}) | {M[a]} |")
    L += ["", "## By fine role, last block of the birth (updates 24,065-24,576)", "",
          "Loss (nats per position, unweighted CE) / mean entropy (nats) / mean weight / share of the weighted error (%).", "",
          "| Arm | " + " | ".join(FINE) + " |", "|---" * (len(FINE) + 1) + "|"]
    end = {}
    for a in ARMS:
        last = recs[a]["role_curve"][-1]; assert last["block_end"] == 24576
        wb = last["weights"]["by_fine_role"]; lb = last["by_fine_role"]
        end[a] = {n: dict(loss=lb[n]["mean"], entropy=wb[n]["mean_entropy"], weight=wb[n]["mean_weight"], share=100 * wb[n]["share_weighted_error"]) for n in FINE}
        L.append(f"| {a} | " + " | ".join(f"{e['loss']:.3f} / {e['entropy']:.3f} / {e['weight']:.4g} / {e['share']:.1f}" for e in end[a].values()) + " |")
    L += ["", "## Distractor-to-answer weight ratio (mean weight of a distractor position / of an answer position: digits and EOS)", "",
          "First eight blocks (updates 1 to 4,096) and the block ending at each saved row.", "",
          "| Arm | " + " | ".join(f"block to {b}" for b in range(512, 4097, 512)) + " | " + " | ".join(f"at {r}" for r in ROWS) + " | projection |",
          "|---" * (8 + len(ROWS) + 2) + "|"]
    ratios = {}
    for a in ARMS:
        rc = {b["block_end"]: b["weights"]["distractor_to_answer_weight_ratio"] for b in recs[a]["role_curve"]}
        ratios[a] = rc
        L.append(f"| {a} | " + " | ".join(f"{rc[b]:.4g}" for b in range(512, 4097, 512)) + " | " + " | ".join(f"{rc[r]:.4g}" for r in ROWS) + f" | {PROJ[a]['ratio']} |")
    L += ["", "## Share of the weighted error by role (%), first eight blocks and at the end, beside the projections (row 24,576 of S-all)", "",
          "| Arm | Block end | distractor | operand | code | operator | equals | answer digit | eos |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for b in recs[a]["role_curve"]:
            if b["block_end"] <= 4096 or b["block_end"] == 24576:
                wb = b["weights"]["by_fine_role"]
                L.append(f"| {a} | {b['block_end']} | " + " | ".join(f"{100 * wb[n]['share_weighted_error']:.1f}" for n in FINE) + " |")
        p = PROJ[a]["shares"]
        L.append(f"| {a} | projection | {p['distractor']} | {p['operand']} | {p['code']} | | | {p['answer_digit']} | |")
    acc40 = [Ep[a]["mean"]["acc40"] for a in ARMS]
    no_rise = not (acc40[0] < acc40[1] < acc40[2])
    L += ["", f"## Outcome row: {row} ({detail})", "", "## Facts beside the row", "",
          f"- Dose order: accuracy at 40 letters for tau 0.5 / 1 / 2: {' / '.join(f'{v:.2f}' for v in acc40)}; "
          f"{'does not rise' if no_rise else 'rises'} with tau.",
          "- Range: accuracy at 12 letters " + ", ".join(f"{a} {Ep[a]['mean']['acc12']:.2f}" for a in ARMS) + f" (baseline S-all {BASELINE_ACC12}, B-all {BALL_ACC12}).",
          "- Distance to the reference (arms that extrapolate): " + (", ".join(
              f"{a} {Ep[a]['mean']['acc40']:.2f}, {'within' if abs(Ep[a]['mean']['acc40'] - BALL_ACC40) <= 15 else 'not within'} 15 points of B-all's {BALL_ACC40}"
              for a in ARMS if M[a] == "extrapolates") or "none extrapolates"),
          "- Realised weights: the two tables above."]
    if row == "4":
        L += ["", "Answer digits over the birth (outcome 4):", ""]
        for a in ARMS:
            ws = [b["weights"]["by_fine_role"]["answer_digit"] for b in recs[a]["role_curve"]]
            tw = sum(w["sum_weight"] for w in ws); tc = sum(w["positions"] for w in ws)
            tot = sum(sum(v["sum_weighted_ce"] for v in b["weights"]["by_fine_role"].values()) for b in recs[a]["role_curve"])
            L.append(f"- {a}: mean weight {tw / tc:.4g}, share of the weighted error {100 * sum(w['sum_weighted_ce'] for w in ws) / tot:.1f}%")
    L += ["", "## Every saved row", "", "| Arm | Row | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for r in sorted((r for r in rows if r["arm"] == a), key=lambda r: r["row"]):
            L.append(f"| {a} | {r['row']} | {r['acc12']:.2f} | {r['acc20']:.2f} | {r['acc40']:.2f} | {r['keepers12']} / {r['keepers20']} / {r['keepers40']} | "
                     f"{r['kw12']} / {r['kw20']} / {r['kw40']} |")
    (OUT / "E34_READ_TABLES.md").write_text("\n".join(L) + "\n")
    rec = json.loads((OUT / "E34_READ.json").read_text())
    rec.update(endpoints=Ep, marks=M, outcome=dict(row=row, detail=detail), end_by_fine_role=end,
               distractor_to_answer_ratio={a: {str(k): v for k, v in r.items()} for a, r in ratios.items()}, acc40_does_not_rise_with_tau=no_rise)
    (OUT / "E34_READ.json").write_text(json.dumps(rec, indent=1))
    print("\n".join(L))


# ---------------------------------------------------------------- addendum 1
def first_row_70(rows_by_row):
    for r in ROWS:
        if r in rows_by_row and rows_by_row[r] >= 70:
            return r
    return None


def b050_reading(e):
    if e["acc12"] < 70:
        return "does not learn in range", "Under backpropagation the gain at this temperature prevents in-range learning on this seed."
    d = e["acc40"] - BALL_ACC40
    if abs(d) < 15:
        return ("learns in range; at 40 letters within 15 of 82.0",
                "The gain neither helps nor hurts backpropagation's extrapolation here. What T050 shows is the sweep being brought to backpropagation's level by the objective.")
    if d >= 15:
        return "learns in range; at least 15 above 82.0", "The gain helps backpropagation too: the benefit belongs to the objective, not to the rule."
    return "learns in range; at least 15 below 82.0", "The gain hurts backpropagation's extrapolation on this seed and not the sweep's."


def tables_a1():
    e32 = json.loads((HERE / "exp45" / "e32" / "E32_READ.json").read_text())["rows"]
    plain = {(a, s_): {r["row"]: r["acc12"] for r in e32 if r["arm"] == a and r["seed"] == s_} for a in ("floor", "ceiling") for s_ in A1_SEEDS}
    lines = [("T050", s_) for s_ in A1_SEEDS] + [("B050", 73129)]
    L = ["# E34 addendum 1 reader tables (generated by `e34_read.py tables_a1` from `E34_READ.json`, `E34_READ_<seed>.json` and `E32_READ.json`)", "",
         "Endpoint = mean of rows 16,384, 20,480, 24,576; the final row (24,576) is in brackets. Plain sweep = E28 S-all (`floor` in E32_READ.json); "
         "plain backpropagation = E28 B-all (`ceiling`).", "",
         "| Seed | Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |", "|---|---|---|---|---|---|---|---|"]
    E_, rows_all = {}, {}
    for a, s_ in lines:
        rows = [r for r in json.loads(read_path(s_).read_text())["rows"] if r["arm"] == a and r["seed"] == s_]
        rs = {r["row"]: r for r in rows}
        assert all(k in rs for k in ROWS), f"{a} {s_}: rows {sorted(rs)}"
        rows_all[(a, s_)] = rs
        m_ = {k: sum(rs[r][k] for r in END_ROWS) / 3 for k in KEYS}; f_ = {k: rs[24576][k] for k in KEYS}
        E_[(a, s_)] = dict(mean=m_, final=f_, mark=mark(m_))
        L.append(f"| {s_} | {a} | {m_['acc12']:.2f} ({f_['acc12']:.2f}) | {m_['acc20']:.2f} ({f_['acc20']:.2f}) | {m_['acc40']:.2f} ({f_['acc40']:.2f}) | "
                 f"{m_['keepers12']:.1f} / {m_['keepers20']:.1f} / {m_['keepers40']:.1f} ({f_['keepers12']} / {f_['keepers20']} / {f_['keepers40']}) | "
                 f"{m_['kw12']:.1f} / {m_['kw20']:.1f} / {m_['kw40']:.1f} ({f_['kw12']} / {f_['kw20']} / {f_['kw40']}) | {E_[(a, s_)]['mark']} |")
    L += ["", "## Fact 1: first saved row at which accuracy at 12 letters reaches 70", "",
          "| Seed | Arm | Arm's first row | Plain backpropagation | Plain sweep |", "|---|---|---|---|---|"]
    facts = {}
    for a, s_ in lines:
        fr = first_row_70({r: v["acc12"] for r, v in rows_all[(a, s_)].items()})
        fb, fs = first_row_70(plain[("ceiling", s_)]), first_row_70(plain[("floor", s_)])
        facts[f"{a}_{s_}"] = dict(first_row_70=fr, plain_backprop=fb, plain_sweep=fs)
        L.append(f"| {s_} | {a} | {fr or 'never'} | {fb or 'never'} | {fs or 'never'} |")
    L += ["", "## Fact 2: the three rows at 40 letters beside their mean", "", "| Seed | Arm | 16,384 | 20,480 | 24,576 | Mean |", "|---|---|---|---|---|---|"]
    for a, s_ in lines:
        rs = rows_all[(a, s_)]
        L.append(f"| {s_} | {a} | " + " | ".join(f"{rs[r]['acc40']:.2f}" for r in END_ROWS) + f" | {E_[(a, s_)]['mean']['acc40']:.2f} |")
    e = E_[("B050", 73129)]["mean"]
    case, reading = b050_reading(e)
    L += ["", "## B050 (seed 73129)", "",
          f"Accuracy at 12 letters {e['acc12']:.2f}; at 40 letters {e['acc40']:.2f}, {e['acc40'] - BALL_ACC40:+.2f} against plain backpropagation's {BALL_ACC40} "
          f"(T050: {E_[('T050', 73129)]['mean']['acc40']:.2f}). Accuracy at 12 letters at row 8,192: {rows_all[('B050', 73129)][8192]['acc12']:.2f}.", "",
          f"Case: **{case}**. Reading: {reading}"]
    both = all(E_[("T050", s_)]["mark"] == "extrapolates" for s_ in (93683, 116089))
    L += ["", f"T050 learns in range and extrapolates on both new seeds: {'yes' if both else 'no'} "
          f"({', '.join(f'{s_}: ' + E_[('T050', s_)]['mark'] for s_ in (93683, 116089))})."]
    (OUT / "E34_ADDENDUM1_TABLES.md").write_text("\n".join(L) + "\n")
    (OUT / "E34_ADDENDUM1.json").write_text(json.dumps(dict(endpoints={f"{a}_{s_}": v for (a, s_), v in E_.items()}, facts=facts,
                                                            b050=dict(case=case, reading=reading), t050_both_new_seeds=both), indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if len(sys.argv) > 2:
        set_seed(int(sys.argv[2]))
    if cmd == "tables_a1":
        tables_a1(); sys.exit(0)
    if cmd == "reproduce":
        sys.exit(0 if reproduce() else 1)
    if cmd == "read":
        read(); sys.exit(0)
    if cmd == "tables":
        tables(); sys.exit(0)
    raise SystemExit(__doc__)
