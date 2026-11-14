"""E32 reader (brief OPUS_TASK_E32_EVERY_BYTE_BIRTH_SWITCHES_20261114.md, revision 4, "Reading the files"). Read-only; trains nothing.

    python3 e32_read.py reproduce     E30's floor and ceiling rows exactly, and the brief's floor table within one unit -> exp45/e32/E32_READ_REPRO.json
    python3 e32_read.py read          every saved row of floor, ceiling and the six arms               -> exp45/e32/E32_READ.json
    python3 e32_read.py tables        endpoints, marks and the per-seed tables from E32_READ.json       -> exp45/e32/E32_READ_TABLES.md

Gate values come from e29_gates.gates(m, x) (asserts bit-identity with arch26.forward); scoring is exp16_integrated.score, exact.
Only the network's own tensors are loaded (K's feedback matrix is not); every file's sha256 is checked against its receipt.

Panels: exp11_cell.panel(t, held, seed, ks=(12, 40), n=300) for 12 and 40 letters (E30's call) and a separate ks=(20,) call for 20,
t = add89, held = exp11_cell.split(t, .5, seed)[1].
Keeper across k letters: over the steps whose input byte is one of the k letters, the product of the unit's update gate is >= 0.5, in at
least half the items. Keeps and writes: a keeper whose update gate is <= 0.5 at some step before the first letter (positions 1 .. first
letter - 1), in at least half the items. Gate shares: of the letter-step gate values on each panel, the % above 0.944 and above 0.983.
U: E30's (e30_gates.U_of on e29_gates.add89_panel(seed, 12), and on the plus-minus panel).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_gates as G29              # noqa: E402  (imports exp20_reference: the harness's environment gate)
import e30_gates as G30              # noqa: E402
import torch as T                    # noqa: E402

X16, E, X, arch26 = G29.X16, G29.E, G29.X, G29.arch26
T.set_num_threads(1)
OUT = HERE / "exp45" / "e32"
E28 = HERE / "exp45" / "e28"
SEEDS = G29.SEEDS
KS = (12, 20, 40)
ROWS = (4096, 8192, 12288, 16384, 20480, 24576)
END_ROWS = (16384, 20480, 24576)
ARMS = ("floor", "ceiling", "G", "H", "K", "P4a", "P4b", "P4c")
KEEP_BAR = {73129: 25, 93683: 30, 116089: 31}
FLOOR_TABLE = {   # the brief's "Why": row -> (keepers at 12, keepers at 40, accuracy at 12) for seeds 73129 / 93683 / 116089
    4096: ((50, 59, 62), (22, 0, 48), (16, 26, 17)),
    8192: ((22, 20, 30), (0, 0, 0), (57, 42, 48)),
    12288: ((5, 5, 2), (0, 0, 0), (61, 54, 51)),
    24576: ((0, 0, 1), (0, 0, 0), (75, 80, 71)),
}


def receipt(arm, seed):
    if arm == "floor":
        return E28 / f"e28_dev_S-all_{seed}.json"
    if arm == "ceiling":
        return OUT / "ceiling" / f"e28_dev_B-all_{seed}.json"
    return OUT / f"e32_dev_{arm}_{seed}.json"


def row_files(rp):
    rec = json.loads(rp.read_text())
    assert rec.get("complete") is True, f"{rp.name} is not complete"
    out = {}
    for k, v in rec["row_weights"].items():
        out[int(k)] = (rp.parent / v["file"], v["sha256"])
    assert sorted(out) == list(ROWS), f"{rp.name}: rows {sorted(out)}"
    return rec, out


def load_net(path, want, seed):
    sha = X16.sha_file(path)
    assert sha == want, f"{path.name}: sha256 {sha[:12]} is not its receipt's {want[:12]}"
    m = arch26.new_net(seed)
    sd = T.load(path)
    m.load_state_dict(sd)          # strict: the network's own tensors and nothing else
    m.eval()
    return m, sha


def panels(seed):
    t = G29.ADD89
    held = E.split(t, .5, seed)[1]
    pk = E.panel(t, held, seed, ks=(12, 40), n=300)
    pk20 = E.panel(t, held, seed, ks=(20,), n=300)
    return {12: pk[12], 20: pk20[20], 40: pk[40]}


def keepers(m, items):
    x, _, _ = X.batch_tensor(items)
    Z, chk = G29.gates(m, x)
    roles = [G29.roles_of(r) for r in x]
    letter = T.tensor([[r == "letter" for r in row] for row in roles])            # (N, L)
    first = letter.int().argmax(1)
    assert bool((first == first[0]).all()), "first letter at different positions"
    f = int(first[0])
    Zd = Z.double()
    keep = T.where(letter[..., None], Zd, T.ones_like(Zd)).prod(1)                 # (N, U)
    n = Z.shape[0]
    keeper = (keep >= .5).sum(0) >= n / 2
    writes = (Z[:, 1:f, :] <= .5).any(1).sum(0) >= n / 2
    zl = Z[letter]                                                                 # (letter steps, U)
    return dict(keepers=int(keeper.sum()), keeps_and_writes=int((keeper & writes).sum()),
                share_944=round(100 * float((zl > .944).double().mean()), 3), share_983=round(100 * float((zl > .983).double().mean()), 3),
                letters=int(letter[0].sum()), first_letter_position=f, arch26=chk)


def measure(m, seed, pk):
    r = {}
    for k in KS:
        s = X16.score(m, pk[k])
        r[f"acc{k}"] = round(100 * s["exact"] / s["n"], 2)
        r[f"count{k}"] = s
        kp = keepers(m, pk[k])
        r[f"keepers{k}"], r[f"kw{k}"] = kp["keepers"], kp["keeps_and_writes"]
        r[f"share944_{k}"], r[f"share983_{k}"] = kp["share_944"], kp["share_983"]
        r[f"letters{k}"] = kp["letters"]
    r["U_add89_k12"] = G30.U_of(m, G29.add89_panel(seed, 12))["U"]
    r["U_plus_minus"] = G30.U_of(m, G29.plus_minus_panel(seed))["U"]
    return r


def p4_norms(rec, row):
    for c in rec["curve"]:
        if c["step"] == row:
            n = (c.get("e32") or {}).get("P4_norms_on_last_batch_of_block")
            return n
    return None


# ---------------------------------------------------------------- step 1: reproduce
def reproduce():
    arch26.install()
    e30 = json.loads((HERE / "exp45" / "e30" / "E30_UNSEEN.json").read_text())
    out, ok = dict(e30_rows=[], floor_table=[]), True
    for r in e30["rows"]:
        if r["experiment"] != "E28" or r["arm"] not in ("S-all", "B-all"):
            continue
        path = E28 / r["file"]
        m, sha = load_net(path, r["sha256"], r["seed"])
        pk = E.panel(G29.ADD89, E.split(G29.ADD89, .5, r["seed"])[1], r["seed"], ks=(12, 40), n=300)
        acc = {k: round(100 * X16.score(m, pk[k])["exact"] / 300, 2) for k in (12, 40)}
        ua = G30.U_of(m, G29.add89_panel(r["seed"], 12))["U"]
        up = G30.U_of(m, G29.plus_minus_panel(r["seed"]))["U"]
        mine = dict(acc12=acc[12], acc40=acc[40], U_add89_k12=ua, U_plus_minus=up, H=round(acc[12] - acc[40], 2))
        same = all(mine[k] == r[k] for k in mine)
        ok &= same
        out["e30_rows"].append(dict(arm=r["arm"], seed=r["seed"], row=r["row"], file=r["file"], e30={k: r[k] for k in mine}, reader=mine, exact=same))
        print(f"E30 {r['arm']} {r['seed']} row {r['row']}: {'exact' if same else 'DIFFERS'} {mine}", flush=True)
    for row, (k12, k40, a12) in FLOOR_TABLE.items():
        for i, s in enumerate(SEEDS):
            _, files = row_files(receipt("floor", s))
            m, _ = load_net(*files[row], s)
            pk = panels(s)
            kp12, kp40 = keepers(m, pk[12]), keepers(m, pk[40])
            acc12 = 100 * X16.score(m, pk[12])["exact"] / 300
            w = dict(keepers12=k12[i], keepers40=k40[i], acc12=a12[i])
            g = dict(keepers12=kp12["keepers"], keepers40=kp40["keepers"], acc12=round(acc12, 2))
            within = abs(g["keepers12"] - w["keepers12"]) <= 1 and abs(g["keepers40"] - w["keepers40"]) <= 1 and abs(g["acc12"] - w["acc12"]) < 1
            ok &= within
            out["floor_table"].append(dict(seed=s, row=row, brief=w, reader=g, within_one_unit=within))
            print(f"floor {s} row {row}: brief {w} reader {g} {'ok' if within else 'OUTSIDE'}", flush=True)
    out.update(passed=ok, python=sys.version.split()[0], torch=T.__version__,
               rule="E30 rows: acc12, acc40, U (both panels), H equal as rounded in E30_UNSEEN.json; floor table: keepers within one unit, accuracy at 12 within the table's rounding")
    (OUT / "E32_READ_REPRO.json").write_text(json.dumps(out, indent=1))
    print("reproduction", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- step 2: read every row
def read():
    rp = OUT / "E32_READ_REPRO.json"
    assert rp.exists() and json.loads(rp.read_text())["passed"] is True, "reproduction has not passed"
    arch26.install()
    path = OUT / "E32_READ.json"
    done = json.loads(path.read_text()) if path.exists() else dict(rows=[])
    have = {(r["arm"], r["seed"], r["row"], r["sha256"]) for r in done["rows"]}
    for s in SEEDS:
        pk = panels(s)
        for arm in ARMS:
            rec, files = row_files(receipt(arm, s))
            for row in ROWS:
                f, want = files[row]
                if (arm, s, row, want) in have:
                    continue
                t0 = time.process_time()
                m, sha = load_net(f, want, s)
                r = dict(arm=arm, seed=s, row=row, file=str(f.relative_to(HERE)), sha256=sha, **measure(m, s, pk))
                if arm.startswith("P4"):
                    r["P4_norms"] = p4_norms(rec, row)
                r["cpu_s"] = round(time.process_time() - t0, 2)
                done["rows"].append(r)
                path.write_text(json.dumps(done, indent=1))
                print(f"{arm} {s} {row}: acc {r['acc12']:.1f} {r['acc20']:.1f} {r['acc40']:.1f} | keepers {r['keepers12']} {r['keepers20']} {r['keepers40']}"
                      f" | kw {r['kw12']} {r['kw20']} {r['kw40']} | U {r['U_add89_k12']:.2f} {r['U_plus_minus']:.2f}", flush=True)
    done.update(python=sys.version.split()[0], torch=T.__version__)
    path.write_text(json.dumps(done, indent=1))


# ---------------------------------------------------------------- step 3: endpoints, marks, tables
def endpoints(rows):
    E_ = {}
    for arm in ARMS:
        for s in SEEDS:
            rs = [r for r in rows if r["arm"] == arm and r["seed"] == s and r["row"] in END_ROWS]
            if len(rs) != 3:
                continue
            E_[(arm, s)] = {k: sum(r[k] for r in rs) / 3 for k in ("acc12", "acc20", "acc40", "kw12", "kw20", "kw40", "keepers12")}
    return E_


def two_of_three(flags):
    return sum(bool(f) for f in flags) >= 2


def marks(E_):
    M = {}
    for arm in ARMS[2:]:
        if not all((arm, s) in E_ for s in SEEDS):
            continue
        d = {}
        for k in (20, 40):
            up = [E_[(arm, s)][f"acc{k}"] - E_[("floor", s)][f"acc{k}"] for s in SEEDS]
            gap = [E_[("ceiling", s)][f"acc{k}"] - E_[(arm, s)][f"acc{k}"] for s in SEEDS]
            mv = two_of_three(u >= 30 for u in up)
            pa = (not mv) and two_of_three(u >= 15 for u in up)
            d[f"acc{k}"] = dict(points_above_floor=[round(u, 2) for u in up], moves=mv, partial=pa, neither=not (mv or pa),
                                closes=two_of_three(abs(g) <= 15 for g in gap), points_below_ceiling=[round(g, 2) for g in gap])
        for k in KS:
            kw = [E_[(arm, s)][f"kw{k}"] for s in SEEDS]
            d[f"has_keepers{k}"] = dict(keeps_and_writes=[round(v, 2) for v in kw], bar=[KEEP_BAR[s] for s in SEEDS],
                                        yes=two_of_three(v >= KEEP_BAR[s] for v, s in zip(kw, SEEDS)))
        dn = [E_[("floor", s)]["acc12"] - E_[(arm, s)]["acc12"] for s in SEEDS]
        d["harm"] = dict(points_below_floor_at12=[round(v, 2) for v in dn], yes=two_of_three(v >= 15 for v in dn))
        M[arm] = d
    return M


def tables():
    rows = json.loads((OUT / "E32_READ.json").read_text())["rows"]
    E_ = endpoints(rows)
    M = marks(E_)
    L = ["# E32 reader tables (generated by `e32_read.py tables` from `E32_READ.json`)", ""]
    for s in SEEDS:
        L += [f"## Seed {s}", "",
              "| Arm | Row | Acc 12 | Acc 20 | Acc 40 | Keepers 12 / 20 / 40 | Keeps-and-writes 12 / 20 / 40 | Gates >0.944 at 12 / 20 / 40 | Gates >0.983 at 12 / 20 / 40 | U add89 12 | U plus-minus |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for r in sorted((r for r in rows if r["arm"] == arm and r["seed"] == s), key=lambda r: r["row"]):
                L.append(f"| {arm} | {r['row']} | {r['acc12']:.2f} | {r['acc20']:.2f} | {r['acc40']:.2f} | {r['keepers12']} / {r['keepers20']} / {r['keepers40']} | "
                         f"{r['kw12']} / {r['kw20']} / {r['kw40']} | {r['share944_12']:.2f} / {r['share944_20']:.2f} / {r['share944_40']:.2f} | "
                         f"{r['share983_12']:.2f} / {r['share983_20']:.2f} / {r['share983_40']:.2f} | {r['U_add89_k12']:.2f} | {r['U_plus_minus']:.2f} |")
            if (arm, s) in E_:
                e = E_[(arm, s)]
                L.append(f"| **{arm} endpoint** | mean 16,384-24,576 | {e['acc12']:.2f} | {e['acc20']:.2f} | {e['acc40']:.2f} | {e['keepers12']:.2f} / - / - | "
                         f"{e['kw12']:.2f} / {e['kw20']:.2f} / {e['kw40']:.2f} | | | | |")
        L.append("")
    p4 = [r for r in rows if r["arm"].startswith("P4")]
    if p4:
        L += ["## P4 gradient norms at each saved row (last batch of the block; joint norm over the update-gate rows)", "",
              "| Arm | Seed | Row | Task norm | Term norm at lam 1 | Term norm at lam |", "|---|---|---|---|---|---|"]
        for r in sorted(p4, key=lambda r: (r["arm"], r["seed"], r["row"])):
            n = r.get("P4_norms")
            L.append(f"| {r['arm']} | {r['seed']} | {r['row']} | " + (f"{n['task_norm']:.6g} | {n['term_norm_at_lam1']:.6g} | {n['term_norm_at_lam']:.6g} |" if n else "(term off) | | |"))
        L.append("")
    L += ["## Marks (endpoints: mean of rows 16,384, 20,480, 24,576; two of three seeds)", "",
          "| Arm | Acc 40: above floor | moves / partial | closes | Acc 20: above floor | moves / partial | closes | Has keepers 12 / 20 / 40 | Harm at 12 (points below floor) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, d in M.items():
        def mp(x): return "moves" if x["moves"] else ("partial" if x["partial"] else "neither")
        a40, a20 = d["acc40"], d["acc20"]
        hk = " / ".join(f"{'yes' if d[f'has_keepers{k}']['yes'] else 'no'} ({', '.join(f'{v:.1f}' for v in d[f'has_keepers{k}']['keeps_and_writes'])})" for k in KS)
        L.append(f"| {arm} | {', '.join(f'{v:+.1f}' for v in a40['points_above_floor'])} | {mp(a40)} | {'yes' if a40['closes'] else 'no'} | "
                 f"{', '.join(f'{v:+.1f}' for v in a20['points_above_floor'])} | {mp(a20)} | {'yes' if a20['closes'] else 'no'} | {hk} | "
                 f"{', '.join(f'{v:+.1f}' for v in d['harm']['points_below_floor_at12'])} ({'harm' if d['harm']['yes'] else 'no harm'}) |")
    (OUT / "E32_READ_TABLES.md").write_text("\n".join(L) + "\n")
    rec = json.loads((OUT / "E32_READ.json").read_text())
    rec["endpoints"] = {f"{a}_{s}": v for (a, s), v in E_.items()}
    rec["marks"] = M
    (OUT / "E32_READ.json").write_text(json.dumps(rec, indent=1))
    print("\n".join(L[-(len(M) + 3):]))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "reproduce":
        sys.exit(0 if reproduce() else 1)
    if cmd == "read":
        read(); sys.exit(0)
    if cmd == "tables":
        tables(); sys.exit(0)
    raise SystemExit(__doc__)
