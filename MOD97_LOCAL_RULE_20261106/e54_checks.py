"""E54: receipts of the checks that are read from receipts (brief revision 3, "Checks before the runs"). Nothing here trains a network.

    python3 e54_checks.py check1                      # the two dry runs at f = 0.50 against exp42_cells.py on this host and E42's receipts
    python3 e54_checks.py check6                      # a resumed run equals an uninterrupted one
    python3 e54_checks.py check2 <25|15|10>           # the batch digest at row 4,096 of the first run of a part
    python3 e54_checks.py check3 <S|B> <25|10>        # G against E41's receipt, every row to update 12,288 that the run has reached
    python3 e54_checks.py check3p4 <S|B>              # part 4 against E41's receipt, every row to update 40,960 that the run has reached
    python3 e54_checks.py check4                      # CPU rates from the dry runs and the forecast
    python3 e54_checks.py row256 <G|P4> <S|B> [25|10] # the first comparison of check 3, at row 256, from the printed line of the log

The rows of a run in progress are read from its receipt (written every 4,096 updates) or, if the saved state is newer, from the receipt
inside the saved state (written every 1,024 updates); the first comparison, at row 256, is made from the printed line of the log (`row256`). Every check writes exp45/e54/qual/<name>.json and exits 0 only if it passed.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp45" / "e54"
Q = OUT / "qual"
W = "origin/claude/gallant-dijkstra-i9zgsr"
SEED = 73129
ROWKEYS = ("loss", "A_counts", "R_counts", "taught_counts", "data_digest")
VOLATILE = ("wall", "cpu_updates", "cpu_scoring", "cpu_replacement")
DIGEST_4096 = {25: "a71b1ca10b24a36b", 15: "a6f5362d355c59fc", 10: "e120924e5af546f9"}


def show(ref, path):
    return json.loads(subprocess.run(["git", "show", f"{ref}:MOD97_LOCAL_RULE_20261106/{path}"], capture_output=True, check=True, cwd=HERE).stdout)


def write(name, res):
    res["written"] = time.strftime("%FT%TZ", time.gmtime())
    (Q / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(name, "PASS" if res["pass"] else "FAIL", flush=True)
    sys.exit(0 if res["pass"] else 1)


def live_rows(stem, folder=OUT):
    """The scored rows of a run, finished or in progress: the longer of the receipt's curve and the saved state's."""
    rp = folder / f"{stem}.json"; cp = folder / "ckpt" / f"{stem}.ckpt.pt"
    rec = json.loads(rp.read_text()) if rp.exists() else None
    if cp.exists():
        import torch
        st = torch.load(cp, weights_only=False)["rec"]
        if rec is None or len(st["curve"]) > len(rec["curve"]):
            return st, "saved state " + cp.name
    assert rec is not None, f"no receipt and no saved state for {stem}"
    return rec, "receipt " + rp.name


def check1():
    cases = [("F10", 2048, "exp45/e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json", "837dadc1", None),
             ("P8", 1024, "exp45/e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json", "1b78f306", [0, 0, 173, 49, 208])]
    res = dict(check=1, cases={})
    for cfg, dose, wpath, prefix, repl in cases:
        new = json.loads((Q / "check1_new" / f"e54_{cfg}_S_f50_dry{dose}_{SEED}.json").read_text())
        host = json.loads((Q / "check1_e42" / f"e42_{cfg}_S_dry{dose}_{SEED}.json").read_text())
        e42 = show(W, wpath)
        keys = ROWKEYS + ("length_counts", "replaced_since_previous_row")
        rows = lambda r: [{k: x[k] for k in keys} for x in r["curve"]]
        c = dict(new_receipt=f"qual/check1_new/e54_{cfg}_S_f50_dry{dose}_{SEED}.json", host_receipt=f"qual/check1_e42/e42_{cfg}_S_dry{dose}_{SEED}.json",
                 e42_receipt=f"{W}:{wpath}", tensors_sha256=dict(new=new["tensors_sha256"], host=host["tensors_sha256"], e42=e42["tensors_sha256"]),
                 tensors_equal=new["tensors_sha256"] == host["tensors_sha256"] == e42["tensors_sha256"] and new["tensors_sha256"].startswith(prefix),
                 mask_made_start_equal=new["structure_start_sha256"] == host["structure_start_sha256"] == e42["structure_start_sha256"],
                 mask_made_end_equal=new["structure_end_sha256"] == host["structure_end_sha256"] == e42["structure_end_sha256"],
                 replaced=[x["replaced_since_previous_row"] for x in new["curve"]],
                 rows=len(new["curve"]), rows_equal=rows(new) == rows(host) == rows(e42),
                 final_digest_equal=new["data_digest"] == host["data_digest"] == e42["data_digest"],
                 fraction=new["fraction"], length_panel_is_addition_panel=new["length_panel_is_addition_panel"])
        c["replaced_as_recorded"] = repl is None or c["replaced"] == repl
        c["pass"] = all(c[k] for k in ("tensors_equal", "mask_made_start_equal", "mask_made_end_equal", "rows_equal", "final_digest_equal", "replaced_as_recorded"))
        res["cases"][f"{cfg} S f50 dry{dose}"] = c
    res["compared_in_every_row"] = list(ROWKEYS) + ["length_counts", "replaced_since_previous_row"]
    res["pass"] = all(c["pass"] for c in res["cases"].values())
    write("check1", res)


def check6():
    name = f"e54_P8_S_f25_dry2048_{SEED}.json"
    a = json.loads((Q / "check6_whole" / name).read_text()); b = json.loads((Q / "check6_resumed" / name).read_text())
    strip = lambda c: [{k: v for k, v in r.items() if k not in VOLATILE} for r in c]
    res = dict(check=6, uninterrupted="qual/check6_whole/" + name, resumed="qual/check6_resumed/" + name,
               tensors_sha256=dict(uninterrupted=a["tensors_sha256"], resumed=b["tensors_sha256"]),
               tensors_equal=a["tensors_sha256"] == b["tensors_sha256"], digest_equal=a["data_digest"] == b["data_digest"],
               rows=len(a["curve"]), rows_equal_except_times=strip(a["curve"]) == strip(b["curve"]),
               resumed_at=[x["at_update"] for x in b["checkpointing"]["resumed"]], uninterrupted_resumed_at=[x["at_update"] for x in a["checkpointing"]["resumed"]],
               replaced=[x["replaced_since_previous_row"] for x in a["curve"]])
    res["pass"] = res["tensors_equal"] and res["digest_equal"] and res["rows_equal_except_times"] and res["resumed_at"] == [1024] \
        and res["uninterrupted_resumed_at"] == []
    write("check6", res)


def check2(fp):
    rec, src = live_rows(f"e54_G_S_f{fp}_{SEED}")
    row = next((r for r in rec["curve"] if r["step"] == 4096), None)
    assert row is not None, "row 4,096 has not been written yet"
    d = row["data_digest"]["sha256"]
    write(f"check2_f{fp}", dict(check=2, fraction=fp / 100, run=f"e54_G_S_f{fp}_{SEED}", read_from=src, row=4096, digest=d,
                                expected_prefix=DIGEST_4096[fp], **{"pass": d.startswith(DIGEST_4096[fp])}))


def compare(mine, ref, last):
    by = {r["step"]: r for r in ref["curve"]}
    n = 0; diff = []
    for r in mine["curve"]:
        if r["step"] > last:
            break
        n += 1
        for k in ROWKEYS:
            if r[k] != by[r["step"]][k]:
                diff.append(dict(step=r["step"], field=k, run=r[k], e41=by[r["step"]][k]))
    return n, diff


def check3(rule, fp):
    stem = f"e54_G_{rule}_f{fp}_{SEED}"
    rec, src = live_rows(stem)
    ref = json.loads((HERE / "exp45" / "e41" / f"e41_{rule}_h512_f{fp}_{SEED}.json").read_text())
    n, diff = compare(rec, ref, 12288)
    res = dict(check=3, run=stem, read_from=src, e41_receipt=f"exp45/e41/e41_{rule}_h512_f{fp}_{SEED}.json", fields=list(ROWKEYS),
               rows_compared=n, last_row_compared=rec["curve"][n - 1]["step"], differences=diff[:20])
    done = n == 49
    if done:
        h = rec.get("tensors_sha256_at", {}).get("12288")
        res.update(tensors_sha256_at_12288=h, e41_tensors_sha256=ref["tensors_sha256"], hash_equal=h == ref["tensors_sha256"])
    res["complete_to_12288"] = done
    res["pass"] = not diff and (not done or res["hash_equal"])
    write(f"check3_G_{rule}_f{fp}" + ("" if done else "_partial"), res)


def check3p4(rule):
    stem = f"e41_{rule}_h256_f10_dry122880_{SEED}"
    ref = json.loads((HERE / "exp45" / "e41" / f"e41_{rule}_h256_f10_{SEED}.json").read_text())
    rp = OUT / f"{stem}.json"
    res = dict(check=3, part=4, run=stem, e41_receipt=f"exp45/e41/e41_{rule}_h256_f10_{SEED}.json")
    if rp.exists():
        rec = json.loads(rp.read_text())
        n, diff = compare(rec, ref, 40960)
        res.update(read_from="receipt " + rp.name, fields=list(ROWKEYS), rows_compared=n, last_row_compared=rec["curve"][n - 1]["step"], differences=diff[:20])
        res["pass"] = not diff
        write(f"check3_part4_{rule}" + ("" if n == 161 else "_partial"), res)
    raise SystemExit("the receipt of part 4 has not been written yet")


def row256(kind, rule, fp=None):
    """The first comparison, at row 256, before any receipt or saved state exists: the printed line of the log (A, R and taught to two
    decimals) against E41's receipt."""
    if kind == "G":
        log, head, e41, name = f"e54_G_{rule}_f{fp}.log", f"e54 G {rule} f{fp} {SEED} 256:", f"e41_{rule}_h512_f{fp}_{SEED}.json", f"check3_G_{rule}_f{fp}_row256"
    else:
        log, head, e41, name = f"part4_{rule}.log", f"e41 {rule} h256 f10 {SEED} 256:", f"e41_{rule}_h256_f10_{SEED}.json", f"check3_part4_{rule}_row256"
    line = next(l for l in (OUT / "logs" / log).read_text().splitlines() if l.startswith(head))
    r = next(x for x in json.loads((HERE / "exp45" / "e41" / e41).read_text())["curve"] if x["step"] == 256)
    want = f"{head} A {r['A']:.2f} R {r['R']:.2f} taught {r['taught']:.2f} "
    write(name, dict(check=3, read_from=f"log line of row 256 in logs/{log}", e41_receipt=f"exp45/e41/{e41}", line=line, expected_start=want,
                     **{"pass": line.startswith(want)}))


def check4():
    """CPU-seconds an update and a scored row from the dry runs of checks 1, 3 and 6, and the forecast for 49,152 updates (193 scored rows)."""
    src = {"G S": Q / "check1_73129_runs" / f"e42_G_S_dry2048_{SEED}.json", "G B": Q / "check1_73129_runs" / f"e42_G_B_dry2048_{SEED}.json",
           "F10 S": Q / "check1_new" / f"e54_F10_S_f50_dry2048_{SEED}.json", "P8 S": Q / "check6_whole" / f"e54_P8_S_f25_dry2048_{SEED}.json"}
    rates = {}
    for k, p in src.items():
        r = json.loads(p.read_text()); n = r["dose"]
        rates[k] = dict(receipt=str(p.relative_to(OUT)), cpu_s_per_update=r["cpu_updates"] / n, cpu_s_per_scored_row=r["cpu_scoring"] / (n // 256 + 1),
                        cpu_s_per_event=r["cpu_replacement"] / (n // 256), processes_beside=r.get("processes_beside"))
        rates[k]["forecast_cpu_h_49152"] = (rates[k]["cpu_s_per_update"] * 49152 + rates[k]["cpu_s_per_scored_row"] * 193 + rates[k]["cpu_s_per_event"] * 192) / 3600
    # no dry run of F10 or P8 under back-propagation exists among the checks; E42's receipts give B/S ratios of 0.86/0.82 (F10) and 1.18/1.20 (P8)
    f = {k: v["forecast_cpu_h_49152"] for k, v in rates.items()}
    f["F10 B"] = f["F10 S"] * 0.86 / 0.82; f["P8 B"] = f["P8 S"] * 1.18 / 1.20
    part = sum(f.values())
    p4 = 122880 * (0.044 + 0.040) / 3600 + 2 * 481 * 1.5 / 3600
    res = dict(check=4, rates=rates, forecast_cpu_h_per_run=f, forecast_cpu_h_per_part_1_to_3=part,
               part4_cpu_h_from_e41_receipts=p4, checks_cpu_h=0.6, forecast_cpu_h_job=3 * part + p4 + 0.6, limit_cpu_h=51,
               note="F10 B and P8 B are the dry-run rate under the sweep times the ratio of E42's receipts of seed 73129; part 4 is at the rates of E41's receipts")
    res["pass"] = res["forecast_cpu_h_job"] <= 51
    write("check4", res)


if __name__ == "__main__":
    a = sys.argv[1:]
    dict(check1=check1, check6=check6, check2=lambda: check2(int(a[1])), check3=lambda: check3(a[1], int(a[2])),
         check3p4=lambda: check3p4(a[1]), check4=check4, row256=lambda: row256(a[1], a[2], int(a[3]) if len(a) > 3 else None))[a[0]]()
