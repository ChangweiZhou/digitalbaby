"""E43 checks (the brief's checks 1 to 7) and score 5 (decay alone, no run).

    python3 e43_checks.py spec                 # check 3: the specification file, its self-checks on this host, panel and stream digests
    python3 e43_checks.py decay <S|B> <seed>   # score 5: the birth with every parameter but the byte table times (1 - 0.00025)^t, score 1
    python3 e43_checks.py check1               # conflict arms on seed 73129 against E38's zero / none row hashes
    python3 e43_checks.py after                # checks 2, 5, 6 from the 24 receipts (and 7's forecast from the dry runs)
Receipts go to exp45/e43/checks/.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp43_unrelated as U          # noqa: E402

E38, C, R, X16, E, X, arch26, T, TK = U.E38, U.C, U.R, U.X16, U.E, U.X, U.arch26, U.T, U.TK
K = HERE.parent / "PROJECT_DOCUMENTS" / "COORD_CHECKS_20261116"
D = U.OUT43; CK = D / "checks"
SEEDS = (73129, 93683, 116089)
E38_PREFIX = {("zero", "S"): ("f3819049", "95f5de74", "e2d170be", "ef1ac53c"), ("none", "S"): ("5afe5e0a", "166dd91e", "08e44428", "23f6ddaa"),
              ("zero", "B"): ("35f0d6c9", "a5cad080", "676c2fa0", "ce901c57"), ("none", "B"): ("5abf61c9", "7789d013", "8b9c6649", "a423a30b")}
AUDIT_DIGESTS = {73129: ("3583cd943452ff06", "d3410fc2a8850bc1"), 93683: ("348392b8f4017512", "edbbcc1f0dc94926"),
                 116089: ("301fd99ddfa152f8", "fe35d695eace801d")}
# the audit's table (section 3): reduction score as a share of its start, sweep births then backprop births, seeds 73129, 93683, 116089
AUDIT_DECAY = {"S": {64: (99.6, 99.9, 99.7), 256: (95.2, 97.8, 98.1), 1024: (17.8, 35.3, 26.1), 2048: (0.4, 1.3, 0.3), 4096: (0.0, 0.0, 0.0)},
               "B": {64: (99.2, 99.7, 99.3), 256: (97.3, 96.7, 96.8), 1024: (39.3, 49.0, 46.0), 2048: (0.4, 3.1, 1.3), 4096: (0.0, 0.0, 0.0)}}
DECAY_T = (0, 64, 256, 1024, 2048, 4096)


def write(name, res, cmd, t0):
    CK.mkdir(parents=True, exist_ok=True); p = CK / f"e43_check_{name}.json"; R.refuse_existing(p)
    p.write_text(json.dumps(dict(experiment="E43", check=name, command=cmd, provenance=U.provenance(cmd), cpu_seconds=time.process_time() - t0, result=res), indent=1))
    print(name, "pass" if res.get("pass") else "FAIL" if "pass" in res else "", flush=True)


def spec(cmd):
    t0 = time.process_time()
    mine, src = HERE / "e43_task.py", K / "e43_unrelated_task_spec.py"
    identical = mine.read_bytes() == src.read_bytes()
    import os
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("E16_", "E11_"))}     # the harness's own variables, set at import, removed
    run = subprocess.run([sys.executable, "-I", str(src)], capture_output=True, text=True, env=clean)
    here = run.stdout.strip().splitlines(); mac = (K / "OUTPUT_E43_SPEC_MAC.txt").read_text().strip().splitlines()
    exact_lines = [0, 1, 2, 3, 4, 6]          # three seed lines, shared bytes, batch tensor, score
    lines = {i: dict(here=here[i], mac=mac[i], equal=here[i] == mac[i]) for i in exact_lines}
    loss_here = float(here[7].split()[2].rstrip(";")); loss_mac = float(mac[7].split()[2].rstrip(";"))
    dig = {}
    for s in SEEDS:
        tp = hashlib.sha256(repr(TK.panel(s, "taught")).encode()).hexdigest(); st = TK.digest(s, 4096)
        dig[s] = dict(taught_panel=tp, stream_4096=st, taught_equal=tp[:16] == AUDIT_DIGESTS[s][0], stream_equal=st[:16] == AUDIT_DIGESTS[s][1])
        print(s, "taught panel", tp[:16], "stream 4096", st[:16], flush=True)
    res = dict(file_identical=identical, returncode=run.returncode, stderr=run.stderr[-2000:], lines=lines, byte_table_line=dict(here=here[5], mac=mac[5]),
               first_batch_loss=dict(here=loss_here, mac=loss_mac, equal_two_decimals=round(loss_here, 2) == round(loss_mac, 2)), digests=dig)
    res["pass"] = identical and run.returncode == 0 and all(v["equal"] for v in lines.values()) and res["first_batch_loss"]["equal_two_decimals"] \
        and all(v["taught_equal"] and v["stream_equal"] for v in dig.values())
    print("\n".join(here), flush=True)
    write("3_spec", res, cmd, t0)


def decay(rule, seed, cmd):
    t0 = time.process_time(); arch26.install()
    bfile = C.birth_path(E38.E26DIR, rule, seed); m, _, _ = E38.admit(bfile, rule, seed)
    sd = T.load(bfile); s1 = U.first_family_items(); rows = {}
    for t in DECAY_T:
        f = (1 - 0.00025) ** t
        m.load_state_dict({k: (v if k.startswith("book.") else v * f) for k, v in sd.items()})
        rows[t] = dict(factor=f, score1=U.score1(m, s1))
        print(rule, seed, t, round(rows[t]["score1"]["reduction_score"], 2), flush=True)
    base = rows[0]["score1"]["reduction_score"]; si = SEEDS.index(seed)
    cmp = {t: dict(share=100 * rows[t]["score1"]["reduction_score"] / base, audit=AUDIT_DECAY[rule][t][si]) for t in DECAY_T[1:]}
    for v in cmp.values():
        v["within_3"] = abs(v["share"] - v["audit"]) <= 3
    write(f"score5_decay_{rule}_{seed}", dict(rows=rows, share_vs_audit=cmp, **{"pass": all(v["within_3"] for v in cmp.values())}), cmd, t0)


def check1(cmd):
    t0 = time.process_time(); res = {}
    for arm, e38arm in (("conflict-decay", "zero"), ("conflict", "none")):
        for rule in ("S", "B"):
            mine = {r["updates"]: r["tensors_sha256"] for r in json.loads((D / f"e43_{arm}_{rule}_73129.json").read_text())["rows"]}
            theirs = {r["updates"]: r["tensors_sha256"] for r in json.loads((E38.OUT38 / f"e38_{e38arm}_{rule}_73129.json").read_text())["rows"]}
            res[f"{arm}_{rule}"] = {u: dict(e43=mine[u], e38=theirs[u], equal=mine[u] == theirs[u], brief_prefix=p, prefix_equal=mine[u].startswith(p))
                                    for u, p in zip((64, 256, 1024, 4096), E38_PREFIX[e38arm, rule])}
    res["pass"] = all(v["equal"] and v["prefix_equal"] for k, d in res.items() for v in d.values())
    write("1_conflict_is_e38", res, cmd, t0)


def after(cmd):
    t0 = time.process_time()
    rec = {(a, r, s): json.loads((D / f"e43_{a}_{r}_{s}.json").read_text()) for a in U.ARMS43 for r in ("S", "B") for s in SEEDS}
    e38row0 = {r: json.loads((E38.OUT38 / f"e38_zero_{r}_73129.json").read_text())["rows"][0] for r in ("S", "B")}
    c2 = {}
    for (a, r, s), v in rec.items():
        row0 = v["rows"][0]
        brec = json.loads((HERE / v["settings"]["birth"]).with_suffix(".json").read_text())
        d = dict(row0_equals_birth=row0["tensors_sha256"] == v["settings"]["birth_tensors_sha256"] == brec["tensors_sha256"])
        if s == 73129:
            s1, e = row0["score1_first_family_all_pairs_k0"], e38row0[r]
            d["allpairs_equal_e38"] = all(s1[n]["reduction"]["exact"] == e["birth_allpairs_k0"][n]["wrap"]["exact"] and
                                          s1[n]["other"]["exact"] == e["birth_allpairs_k0"][n]["nowrap"]["exact"] for n in ("add89", "sub89"))
            d["heldout_equal_e38"] = row0["score2_first_family_heldout_k12"] == e["birth_heldout_k12"]
        c2[f"{a}_{r}_{s}"] = d
    write("2_row0", dict(runs=c2, **{"pass": all(all(x.values()) for x in c2.values())}), cmd, t0)
    c5 = {}
    for s in SEEDS:
        for stream, arms in (("conflict", ("conflict", "conflict-decay")), ("unrelated", ("unrelated", "unrelated-decay"))):
            runs = [rec[a, r, s] for a in arms for r in ("S", "B")]
            per = {u: len({json.dumps(next(x["data_digest"] for x in v["rows"] if x["updates"] == u)) for v in runs}) == 1 for u in U.ROWS}
            c5[f"{stream}_{s}"] = per
    write("5_same_batches", dict(rows=c5, **{"pass": all(all(v.values()) for v in c5.values())}), cmd, t0)
    c6 = {f"{a}_{r}_{s}": dict(decay_field=v["settings"]["decay"], groups=[g["weight_decay"] for g in v["settings"]["optimizer_groups_read_back"]],
                               groups_at_end=[g["weight_decay"] for g in v["complete"]["optimizer_groups_read_back"]]) for (a, r, s), v in rec.items()}
    ok6 = all(v["groups"] == v["groups_at_end"] == [v["decay_field"], 0.0] and (v["decay_field"] == 0.0) == (not k.split("_")[0].endswith("decay"))
              for k, v in c6.items())
    write("6_decay_groups", dict(runs=c6, **{"pass": ok6}), cmd, t0)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    {"spec": lambda: spec(cmd), "decay": lambda: decay(a[1], int(a[2]), cmd), "check1": lambda: check1(cmd), "after": lambda: after(cmd)}[a[0]]()
