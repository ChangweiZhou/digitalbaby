"""E26 stage 2: verify the six births and write their curves (E26_BIRTHS.json, E26_BIRTHS.md). Read-only on the births.
Run from MOD97_LOCAL_RULE_20261106: OMP_NUM_THREADS=1 python3 exp45/e26/verify_births.py"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; MOD = HERE.parent.parent
sys.path.insert(0, str(MOD))
import exp26_candidate as C
import arch26, torch as T
import exp20_reference as R
import exp16_integrated as X16

SEEDS = (73129, 93683, 116089); ARMS = ("S", "B"); OPS = ("add89", "sub89", "max89", "min89")
arch26.install()
pinned = {f: subprocess.run(["git", "show", f"360e1b8d:MOD97_LOCAL_RULE_20261106/{f}"], cwd=MOD, capture_output=True).stdout for f in C.SOURCES}
import hashlib
pinned = {f: hashlib.sha256(b).hexdigest() for f, b in pinned.items()}
out = dict(births={}, checks={})
for s in SEEDS:
    for a in ARMS:
        p = HERE / f"e26_dev_{a}_{s}.pt"; r = json.loads(p.with_suffix(".json").read_text())
        ck = {}
        try:
            C.admit_e26(p, a, s); ck["admit_e26"] = True
        except AssertionError as e:
            ck["admit_e26"] = f"FAILED: {e}"
        ck["complete"] = r["complete"] is True
        ck["optimizer_steps_24576"] = r["optimizer_steps"] == [24576]
        ck["weights_sha256_matches_file"] = X16.sha_file(p) == r["weights_sha256"]
        ck["tensors_sha256_matches_file"] = R.tensors_sha(T.load(p)) == r["tensors_sha256"]
        ck["sources_equal_360e1b8d"] = r["provenance"]["sources_sha256"] == pinned
        ck["six_rows_4096_to_24576"] = [c["step"] for c in r["curve"]] == list(range(4096, 24577, 4096))
        ck["finite_losses_every_row"] = all(c["finite_losses"] for c in r["curve"])
        ck["data_digest_last_row_equals_receipt"] = r["curve"][-1]["data_digest"] == r["data_digest"]
        ck["weight_decay_read_back"] = [g["weight_decay"] for g in r["optimizer_read_back"]["groups"]]
        out["checks"][f"{a}_{s}"] = ck
        out["births"][f"{a}_{s}"] = dict(
            cpu_total=r["cpu_total"], cpu_updates=r["cpu_updates"], cpu_scoring=r["cpu_scoring"], data_digest=r["data_digest"]["sha256"],
            curve=[dict(step=c["step"], loss=c["loss"], hard_k12=(c["sealed"]["add89"]["12"]["exact"] + c["sealed"]["sub89"]["12"]["exact"]) / 2,
                        sealed={o: {k: v["exact"] for k, v in c["sealed"][o].items()} for o in OPS}, book_shift=c["book_shift_by_role"]) for c in r["curve"]],
            silenced_k12={o: r["curve"][-1]["silenced_k12"][o]["exact"] for o in OPS})
    out["checks"][f"data_digest_S_equals_B_{s}"] = out["births"][f"S_{s}"]["data_digest"] == out["births"][f"B_{s}"]["data_digest"]
(HERE / "E26_BIRTHS.json").write_text(json.dumps(out, indent=1))
L = ["# E26 stage 2: the six births (full dose, 24,576 updates)", "", "Written by verify_births.py from the receipts. Exact answers out of 300.", "", "## Checks", ""]
for k, v in out["checks"].items():
    L.append(f"- {k}: {v}")
L += ["", "## Hard mean (add89 and sub89 at k=12) through the birth", "", "| step | " + " | ".join(f"{a} {s}" for s in SEEDS for a in ARMS) + " | S−B 73129 | S−B 93683 | S−B 116089 | S−B mean |", "|" + "---|" * 11]
for i in range(6):
    hs = {f"{a}_{s}": out["births"][f"{a}_{s}"]["curve"][i]["hard_k12"] for s in SEEDS for a in ARMS}
    d = [hs[f"S_{s}"] - hs[f"B_{s}"] for s in SEEDS]
    L.append(f"| {4096*(i+1)} | " + " | ".join(f"{hs[f'{a}_{s}']:.1f}" for s in SEEDS for a in ARMS) + " | " + " | ".join(f"{x:+.1f}" for x in d) + f" | {sum(d)/3:+.1f} |")
L += ["", "## Loss (training batch at the checkpoint)", "", "| step | " + " | ".join(f"{a} {s}" for s in SEEDS for a in ARMS) + " |", "|" + "---|" * 7]
for i in range(6):
    L.append(f"| {4096*(i+1)} | " + " | ".join(f"{out['births'][f'{a}_{s}']['curve'][i]['loss']:.4f}" for s in SEEDS for a in ARMS) + " |")
L += ["", "## End of birth, all operations and k (exact /300)", "", "| birth | " + " | ".join(f"{o} k{k}" for o in OPS for k in ("0", "4", "8", "12", "20")) + " |", "|" + "---|" * 21]
for s in SEEDS:
    for a in ARMS:
        se = out["births"][f"{a}_{s}"]["curve"][-1]["sealed"]
        L.append(f"| {a} {s} | " + " | ".join(str(se[o][k]) for o in OPS for k in ("0", "4", "8", "12", "20")) + " |")
L += ["", "## Silenced readout at k=12, end of birth", "", "| birth | " + " | ".join(OPS) + " |", "|---|---|---|---|---|"]
for s in SEEDS:
    for a in ARMS:
        L.append(f"| {a} {s} | " + " | ".join(str(out['births'][f'{a}_{s}']['silenced_k12'][o]) for o in OPS) + " |")
L += ["", "## CPU seconds", "", "| birth | total | updates | scoring |", "|---|---|---|---|"]
for s in SEEDS:
    for a in ARMS:
        b = out["births"][f"{a}_{s}"]; L.append(f"| {a} {s} | {b['cpu_total']:.0f} | {b['cpu_updates']:.0f} | {b['cpu_scoring']:.0f} |")
(HERE / "E26_BIRTHS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
