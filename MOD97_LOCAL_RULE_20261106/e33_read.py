"""E33 reader (brief OPUS_TASK_E33_WHICH_TARGETS_BREAK_EXTRAPOLATION_20261115.md, revision 2, "Reading the files"). Read-only; trains nothing.

    python3 e33_read.py reproduce     S-bal's final file 91.33 / 6.0 and E26 arm S's 98.33 / 96.0 (12 / 40 letters) -> exp45/e33/E33_READ_REPRO.json
    python3 e33_read.py read          every saved row of every arm present (seed 73129)                         -> exp45/e33/E33_READ.json
    python3 e33_read.py tables        endpoints, marks, the outcome row and the eight-arm table                 -> exp45/e33/E33_READ_TABLES.md

Addendum 1 (stage 2, seeds 93683 and 116089): every command takes an optional seed, default 73129 (behaviour for 73129 unchanged).
    python3 e33_read.py reproduce2            check 1: S-all 93683 final 79.67 / 2.33 (E30_UNSEEN.json); S-bal 93683 and 116089 finals reported -> exp45/e33/E33_READ_REPRO_STAGE2.json
    python3 e33_read.py read <seed>           every saved row of A+D and A+OS for that seed      -> exp45/e33/E33_READ_<seed>.json
    python3 e33_read.py tables <seed>         per-seed table and the row R1-R4 that applies       -> exp45/e33/E33_READ_TABLES_<seed>.md

panels, measure, load_net and row_files are imported from e32_read unchanged.
Endpoint: mean of rows 16,384, 20,480, 24,576. Marks: learned in range acc12 >= 70; extrapolates acc40 >= 60; fails acc40 <= 20; intermediate
between; an arm not learned in range is not read for extrapolation. Outcome rows 0a, 0b, 1, 2, 3, 4 checked in the brief's order.
"""
import itertools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e32_read as R32               # noqa: E402  (imports e29_gates -> exp20_reference: environment gate)
from e32_read import panels, measure, load_net, row_files  # noqa: E402

T, X16, E, arch26 = R32.T, R32.X16, R32.E, R32.arch26
OUT = HERE / "exp45" / "e33"
SEED = 73129
ARMS = ("A", "A+D", "A+O", "A+S", "A+DO", "A+DS", "A+OS", "A+DOS")
CLS = {a: (a[2:] if a != "A" else "") for a in ARMS}
ROWS, END_ROWS = R32.ROWS, R32.END_ROWS
FINE = ("distractor", "operand", "code", "operator", "equals", "answer_digit", "eos")


STAGE2_SEEDS = (93683, 116089)
STAGE2_ARMS = ("A+D", "A+OS")


def set_seed(s):
    global SEED
    SEED = s


def read_path():
    return OUT / ("E33_READ.json" if SEED == 73129 else f"E33_READ_{SEED}.json")


def receipt(arm, seed=None):
    return OUT / f"e33_dev_{arm}_{SEED if seed is None else seed}.json"


# ---------------------------------------------------------------- reproduce
def reproduce():
    arch26.install()
    want = [("E28 S-bal final", HERE / "exp45" / "e28" / "e28_dev_S-bal_73129.json", "e28_dev_S-bal_73129.pt", 91.33, 6.0),
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
                                reader={k: r[k] for k in ("acc12", "acc20", "acc40", "keepers12", "keepers20", "keepers40", "kw12", "kw20", "kw40")},
                                exact=same))
        print(name, "exact" if same else "DIFFERS", r["acc12"], r["acc20"], r["acc40"], flush=True)
    out.update(passed=ok, python=sys.version.split()[0], torch=T.__version__,
               rule="accuracy at 12 and 40 letters equal to the brief's figures as rounded to two places (E30_UNSEEN.json: 91.33/6.0 and 98.33/96.0)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E33_READ_REPRO.json").write_text(json.dumps(out, indent=1))
    print("reproduction", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- read
def read():
    rp = OUT / ("E33_READ_REPRO.json" if SEED == 73129 else "E33_READ_REPRO_STAGE2.json")
    assert rp.exists() and json.loads(rp.read_text())["passed"] is True, "reproduction has not passed"
    arch26.install()
    path = read_path()
    done = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    have = {(r["arm"], r["row"], r["sha256"]) for r in done["rows"]}
    pk = panels(SEED)
    for arm in ARMS:
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


# ---------------------------------------------------------------- marks and outcome
KEYS = ("acc12", "acc20", "acc40", "keepers12", "keepers20", "keepers40", "kw12", "kw20", "kw40")


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
    """The brief's outcome rows in order; returns (row, detail)."""
    if M["A"] != "extrapolates":
        return "0a", "A does not extrapolate"
    if M["A+DOS"] != "fails":
        return "0b", "A+DOS does not fail"
    fails = {a for a in ARMS if M[a] == "fails"}
    rest_extrap = lambda f: all(M[a] == "extrapolates" for a in ARMS if a not in f)  # noqa: E731
    for n in (1, 2, 3):
        for C in itertools.combinations("DOS", n):
            want = {a for a in ARMS if any(c in CLS[a] for c in C)}
            if fails == want and rest_extrap(want):
                return "1", "C = {" + ", ".join(C) + "}"
    for c in "DOS":
        o = [k for k in "DOS" if k != c]
        want = {a for a in ARMS if c in CLS[a] or all(k in CLS[a] for k in o)}
        if fails == want and rest_extrap(want):
            return "2", f"c = {c}"
    if all(M[f"A+{c}"] == "extrapolates" for c in "DOS") and M["A+DOS"] == "fails":
        pairs = {a: M[a] for a in ("A+DO", "A+DS", "A+OS")}
        return "3", "pairs: " + ", ".join(f"{a} {v}" for a, v in pairs.items())
    return "4", "mixed"


def class_means(Ep):
    out = {}
    for c in "DOS":
        w = [a for a in ARMS if c in CLS[a]]; wo = [a for a in ARMS if c not in CLS[a]]
        out[c] = {k: dict(with_=sum(Ep[a]["mean"][k] for a in w) / 4, without=sum(Ep[a]["mean"][k] for a in wo) / 4) for k in ("acc12", "acc20", "acc40")}
    return out


def end_losses(arm):
    rec = json.loads(receipt(arm).read_text())
    last = rec["role_curve"][-1]
    assert last["block_end"] == 24576
    return {n: last["by_fine_role"][n]["mean"] for n in FINE}, rec


def tables():
    if SEED != 73129:
        return tables_stage2()
    rows = json.loads((OUT / "E33_READ.json").read_text())["rows"]
    Ep = endpoints(rows)
    assert set(Ep) == set(ARMS), f"endpoints for {sorted(Ep)} only"
    M = {a: mark(Ep[a]["mean"]) for a in ARMS}
    row, detail = outcome(M)
    L = ["# E33 reader tables (generated by `e33_read.py tables` from `E33_READ.json` and the receipts)", "",
         "Seed 73129. Endpoint = mean of rows 16,384, 20,480, 24,576; the final row (24,576) is in brackets.", "",
         "| Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        m_, f_ = Ep[a]["mean"], Ep[a]["final"]
        L.append(f"| {a} | {m_['acc12']:.2f} ({f_['acc12']:.2f}) | {m_['acc20']:.2f} ({f_['acc20']:.2f}) | {m_['acc40']:.2f} ({f_['acc40']:.2f}) | "
                 f"{m_['keepers12']:.1f} / {m_['keepers20']:.1f} / {m_['keepers40']:.1f} ({f_['keepers12']} / {f_['keepers20']} / {f_['keepers40']}) | "
                 f"{m_['kw12']:.1f} / {m_['kw20']:.1f} / {m_['kw40']:.1f} ({f_['kw12']} / {f_['kw20']} / {f_['kw40']}) | {M[a]} |")
    L += ["", "## Loss by fine role, last block of the birth (updates 24,065-24,576; nats per position; every role, selected or not)", "",
          "| Arm | " + " | ".join(FINE) + " |", "|---" * (len(FINE) + 1) + "|"]
    losses = {}
    for a in ARMS:
        lo, _ = end_losses(a); losses[a] = lo
        L.append(f"| {a} | " + " | ".join(f"{lo[n]:.3f}" for n in FINE) + " |")
    L += ["", "## Every saved row", "", "| Arm | Row | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for r in sorted((r for r in rows if r["arm"] == a), key=lambda r: r["row"]):
            L.append(f"| {a} | {r['row']} | {r['acc12']:.2f} | {r['acc20']:.2f} | {r['acc40']:.2f} | {r['keepers12']} / {r['keepers20']} / {r['keepers40']} | "
                     f"{r['kw12']} / {r['kw20']} / {r['kw40']} |")
    L += ["", f"## Outcome row: {row} ({detail})", ""]
    cm = class_means(Ep)
    L += ["Class means over endpoints (the four arms with the class / the four without):", "", "| Class | Acc 12 with / without | Acc 20 with / without | Acc 40 with / without |", "|---|---|---|---|"]
    for c in "DOS":
        L.append(f"| {c} | " + " | ".join(f"{cm[c][k]['with_']:.2f} / {cm[c][k]['without']:.2f}" for k in ("acc12", "acc20", "acc40")) + " |")
    (OUT / "E33_READ_TABLES.md").write_text("\n".join(L) + "\n")
    rec = json.loads((OUT / "E33_READ.json").read_text())
    rec.update(endpoints=Ep, marks=M, outcome=dict(row=row, detail=detail), class_means=cm, end_losses_by_fine_role=losses)
    (OUT / "E33_READ.json").write_text(json.dumps(rec, indent=1))
    print("\n".join(L))


# ---------------------------------------------------------------- addendum 1: stage 2
def reproduce2():
    arch26.install()
    out, ok = dict(rows=[]), True
    for name, f, seed, want in (("E28 S-all 93683 final", "e28_dev_S-all_93683.pt", 93683, (79.67, 2.33)),
                                ("E28 S-bal 93683 final (A+DOS reference)", "e28_dev_S-bal_93683.pt", 93683, None),
                                ("E28 S-bal 116089 final (A+DOS reference)", "e28_dev_S-bal_116089.pt", 116089, None)):
        rp = HERE / "exp45" / "e28" / f.replace(".pt", ".json")
        rec = json.loads(rp.read_text())
        assert rec["weights"] == f
        m, sha = load_net(rp.parent / f, rec["weights_sha256"], seed)
        r = measure(m, seed, panels(seed))
        same = None if want is None else (r["acc12"] == want[0] and r["acc40"] == want[1])
        ok &= same is not False
        out["rows"].append(dict(name=name, file=str((rp.parent / f).relative_to(HERE)), seed=seed, sha256=sha,
                                want=None if want is None else dict(acc12=want[0], acc40=want[1]),
                                reader={k: r[k] for k in KEYS}, exact=same))
        print(name, {True: "exact", False: "DIFFERS", None: "reported"}[same], r["acc12"], r["acc20"], r["acc40"], flush=True)
    out.update(passed=ok, python=sys.version.split()[0], torch=T.__version__,
               rule="addendum 1 check 1: S-all 93683 acc12 and acc40 equal to E30_UNSEEN.json as rounded (79.67 / 2.33); S-bal finals reported, no figure to match")
    (OUT / "E33_READ_REPRO_STAGE2.json").write_text(json.dumps(out, indent=1))
    print("reproduction (stage 2)", "PASSED" if ok else "FAILED")
    return ok


def r_row(Ep):
    """Addendum 1 rows, checked in the order R4, R2, R3, R1 (note to addendum 1); returns (row, also_met)."""
    d, o = Ep["A+D"]["mean"], Ep["A+OS"]["mean"]
    if d["acc12"] < 70 or o["acc12"] < 70:
        return "R4", []
    r2 = o["acc40"] < 60
    r3 = d["acc40"] >= 60 or (o["acc40"] - d["acc40"]) < 40
    if r2:
        return "R2", (["R3"] if r3 else [])
    if r3:
        return "R3", []
    return "R1", []


def tables_stage2():
    rows = [r for r in json.loads(read_path().read_text())["rows"] if r["seed"] == SEED]
    Ep = endpoints(rows)
    assert set(STAGE2_ARMS) <= set(Ep), f"endpoints for {sorted(Ep)} only"
    M = {a: mark(Ep[a]["mean"]) for a in STAGE2_ARMS}
    row, also = r_row(Ep)
    L = [f"# E33 addendum 1 reader table, seed {SEED} (generated by `e33_read.py tables {SEED}` from `{read_path().name}` and the receipts)", "",
         "Endpoint = mean of rows 16,384, 20,480, 24,576; the final row (24,576) is in brackets.", "",
         "| Arm | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Mark |", "|---|---|---|---|---|---|---|"]
    for a in STAGE2_ARMS:
        m_, f_ = Ep[a]["mean"], Ep[a]["final"]
        L.append(f"| {a} | {m_['acc12']:.2f} ({f_['acc12']:.2f}) | {m_['acc20']:.2f} ({f_['acc20']:.2f}) | {m_['acc40']:.2f} ({f_['acc40']:.2f}) | "
                 f"{m_['keepers12']:.1f} / {m_['keepers20']:.1f} / {m_['keepers40']:.1f} ({f_['keepers12']} / {f_['keepers20']} / {f_['keepers40']}) | "
                 f"{m_['kw12']:.1f} / {m_['kw20']:.1f} / {m_['kw40']:.1f} ({f_['kw12']} / {f_['kw20']} / {f_['kw40']}) | {M[a]} |")
    gap = Ep["A+OS"]["mean"]["acc40"] - Ep["A+D"]["mean"]["acc40"]
    L += ["", f"A+OS minus A+D at 40 letters (endpoints): {gap:.2f} points.", "",
          f"## Row: {row}" + (f" (the condition of {', '.join(also)} is also met)" if also else ""), "",
          "## Every saved row", "", "| Arm | Row | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 |", "|---|---|---|---|---|---|---|"]
    for a in STAGE2_ARMS:
        for r in sorted((r for r in rows if r["arm"] == a), key=lambda r: r["row"]):
            L.append(f"| {a} | {r['row']} | {r['acc12']:.2f} | {r['acc20']:.2f} | {r['acc40']:.2f} | {r['keepers12']} / {r['keepers20']} / {r['keepers40']} | "
                     f"{r['kw12']} / {r['kw20']} / {r['kw40']} |")
    (OUT / f"E33_READ_TABLES_{SEED}.md").write_text("\n".join(L) + "\n")
    rec = json.loads(read_path().read_text())
    rec.update(endpoints=Ep, marks=M, row=dict(row=row, also_met=also, gap_acc40=gap))
    read_path().write_text(json.dumps(rec, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if len(sys.argv) > 2:
        set_seed(int(sys.argv[2]))
    if cmd == "reproduce2":
        sys.exit(0 if reproduce2() else 1)
    if cmd == "reproduce":
        sys.exit(0 if reproduce() else 1)
    if cmd == "read":
        read(); sys.exit(0)
    if cmd == "tables":
        tables(); sys.exit(0)
    raise SystemExit(__doc__)
