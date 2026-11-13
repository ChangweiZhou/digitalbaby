"""POSTHOC fail-closed check (added during the main run; NOT part of LOCK.json and never used by run.py).

Written after a cross-session review found three gaps in the locked pipeline:
  1. summarize.py fails open: a gate is None when its arm is absent, and None passes;
  2. summarize.py does not check the world count: 1 of 1 world reads "supported";
  3. LOCK.json hashes fastmem.py but not its runtime closure (bootstrap, minifly_a_v3 stores and the package).
The locked summarize.py stays as it is (editing a hashed file mid-run would also make run.py refuse the COPY phase).
This script reports alongside it; any disagreement between the two is itself a finding.

python posthoc_check.py results/main [--lock-commit 1ce1eadb]
Writes <results>/POSTHOC_VERDICT.json and cycles/POSTHOC_CLOSURE.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

MARGIN = 0.20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(commit, rel):
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(blob.stdout).hexdigest()


def closure(lock_commit):
    """Every repository file imported when lb1 is loaded, plus the native birth data, each verified against the
    git blob at the lock commit or against minifly_a_v3/SOURCE_LOCK.json (for the untracked extracted package)."""
    import lb1  # noqa: F401
    mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
            if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
    mods.discard(Path(__file__).resolve())          # this script is not part of the run
    mods |= {HERE / "run.py", HERE / "summarize.py", HERE / "main.sh"}
    mods = sorted(mods)
    a3 = ROOT / "minifly_a_v3"
    a3_lock = json.loads((a3 / "SOURCE_LOCK.json").read_text())["files"]
    data_files = [a3 / "package" / "FULL151_BIRTH_FINGERPRINT.json", a3 / "package" / "FULL151_CANONICAL_B.npz"]
    rows, ok = [], True
    for p in mods + data_files:
        rel = str(p.relative_to(ROOT))
        cur = sha(p)
        at_lock = git_blob_sha(lock_commit, rel)
        a3_rel = str(p.relative_to(a3)) if str(p).startswith(str(a3)) else None
        if at_lock is not None:
            status = "matches git blob at lock commit" if cur == at_lock else "CHANGED since lock commit"
        elif a3_rel is not None and a3_rel in a3_lock:
            status = "matches minifly_a_v3/SOURCE_LOCK.json" if cur == a3_lock[a3_rel] else "DIFFERS from SOURCE_LOCK"
        else:
            status = "UNVERIFIED"
        ok = ok and status.startswith("matches")
        rows.append({"file": rel, "sha256": cur, "status": status})
    return {"lock_commit": lock_commit, "all_verified": ok, "n_files": len(rows), "files": rows}


def balanced(outputs, world, task, k):
    hit = defaultdict(list)
    for a, b in world["held_pairs"]:
        out = outputs[f"{a}{b}"][k]
        if out is None:
            return None
        t = data.answer(task, a, b)
        hit[t].append(out == t)
    return float(np.mean([np.mean(v) for v in hit.values()]))


def plain(outputs, world, task, k):
    return float(np.mean([outputs[f"{a}{b}"][k] == data.answer(task, a, b) for a, b in world["held_pairs"]]))


def verdict(results_dir, lock):
    problems = []
    runs = {(r["task"], arm) for r in lock["runs"] for arm in r["arms"]}
    worlds_req = set(lock["main_worlds"])
    found = {}
    for f in sorted(Path(results_dir).glob("*_*_*.json")):
        if f.name in ("SUMMARY.json", "POSTHOC_VERDICT.json"):
            continue
        r = json.loads(f.read_text())
        found[(r["task"], r["arm"], r["world"])] = r
    worlds = {w for _, _, w in found}
    if worlds != worlds_req:
        problems.append(f"worlds present {sorted(worlds)} != locked main worlds {sorted(worlds_req)}")
    for w in sorted(worlds):
        for t, arm in sorted(runs):
            if (t, arm, w) not in found:
                problems.append(f"missing history {t} {arm} {w}")
    s = lock["settings"]
    for (t, arm, w), r in found.items():
        if (t, arm) not in runs:
            problems.append(f"unlocked history {t} {arm} {w}")
        want = {k: s[k] for k in ("hidden", "eta0", "lam")}
        if r["settings"] != want or r["passes"] != s["passes"]:
            problems.append(f"settings differ in {t} {arm} {w}")
        world = data.make_world(w)
        if r["train_records"] != s["passes"] * len(world["train_pairs"]):
            problems.append(f"record count {r['train_records']} wrong in {t} {arm} {w}")
        if r["checkpoints"][-1]["label"] != "final_rest1h":
            problems.append(f"final checkpoint missing in {t} {arm} {w}")
    per_world = {}
    for w in sorted(worlds):
        world = data.make_world(w)

        def get(t, arm):
            return found.get((t, arm, w))

        L, F, S, O = (get("PAR", a) for a in ("LEARN", "FROZEN", "SHUF", "ORACLE"))
        if not (L and F and S and O):
            per_world[w] = {"evaluated": False, "reason": "PAR arm missing"}
            continue
        fin = {a: r["checkpoints"][-1] for a, r in (("LEARN", L), ("FROZEN", F), ("SHUF", S), ("ORACLE", O))}
        bal = {a: balanced(c["outputs"], world, "PAR", 2) for a, c in fin.items()}
        mL0, mL1 = L["checkpoints"][0]["meters"]["L"], fin["LEARN"]["meters"]["L"]
        fz = [c["meters"]["L"] for c in F["checkpoints"]]
        copy = [get("COPY", a) for a in ("BASE", "FROZEN", "LEARN")]
        copy_sys = None if not all(copy) else [plain(c["checkpoints"][-1]["outputs"], world, "COPY", 0) for c in copy]
        v = {
            "evaluated": True, "balanced_held_L": {a: round(x, 4) for a, x in bal.items()},
            "learn_ge_frozen_plus_margin": bal["LEARN"] >= bal["FROZEN"] + MARGIN,
            "learn_ge_shuf_plus_margin": bal["LEARN"] >= bal["SHUF"] + MARGIN,
            "learn_gap_rises": mL1["held_transfer_gap"] > mL0["held_transfer_gap"],
            "learn_kta_rises": mL1["kta_all"] > mL0["kta_all"],
            "gate_oracle_ge_0.90": bal["ORACLE"] >= 0.90,
            "gate_frozen_codes_unchanged": all(m.get("train_code_jaccard_vs_prev", 1.0) == 1.0 for m in fz)
            and len({m["held_transfer_gap"] for m in fz}) == 1 and len({m["kta_all"] for m in fz}) == 1,
            "gate_copy_sys_held_ge_0.30": copy_sys is not None and all(x >= 0.30 for x in copy_sys),
            "copy_sys_held": None if copy_sys is None else [round(x, 4) for x in copy_sys],
        }
        v["gates_ok"] = v["gate_oracle_ge_0.90"] and v["gate_frozen_codes_unchanged"] and v["gate_copy_sys_held_ge_0.30"]
        v["supports_condition_1"] = v["gates_ok"] and all(v[k] for k in (
            "learn_ge_frozen_plus_margin", "learn_ge_shuf_plus_margin", "learn_gap_rises", "learn_kta_rises"))
        per_world[w] = v
    n_ok = sum(bool(v.get("supports_condition_1")) for v in per_world.values())
    if problems:
        overall = "INVALID_FOR_VERDICT"
    else:
        overall = "supported" if n_ok == 4 else "partial" if n_ok == 3 else "not supported"
    return {"overall": overall, "worlds_supporting": n_ok, "problems": problems, "per_world": per_world}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--lock-commit", default="1ce1eadb")
    a = ap.parse_args()
    lock = json.loads((HERE / "LOCK.json").read_text())
    res = Path(a.results)
    cl = closure(a.lock_commit)
    lock_ok = all(sha(HERE / rel) == d for rel, d in lock["code_sha256"].items())
    v = verdict(res, lock)
    summ = res / "SUMMARY.json"
    agree = None
    if summ.exists():
        s = json.loads(summ.read_text())
        diffs = []
        for w, pv in v["per_world"].items():
            sv = s.get("verdict_by_world", {}).get(w)
            if sv is not None and pv.get("evaluated") and sv["supports_condition_1"] != pv["supports_condition_1"]:
                diffs.append(f"{w}: locked summarize {sv['supports_condition_1']} vs posthoc {pv['supports_condition_1']}")
        if s.get("overall") != v["overall"]:
            diffs.append(f"overall: locked summarize '{s.get('overall')}' vs posthoc '{v['overall']}'")
        agree = {"agree": not diffs, "differences": diffs}
    out = {"label": "POSTHOC fail-closed check (not locked; see docstring)", "results": str(res),
           "locked_code_hashes_match": lock_ok, "runtime_closure_all_verified": cl["all_verified"],
           "closure_files": cl["n_files"], "verdict": v, "comparison_with_locked_summary": agree}
    (res / "POSTHOC_VERDICT.json").write_text(json.dumps(out, indent=1) + "\n")
    (HERE / "cycles" / "POSTHOC_CLOSURE.json").write_text(json.dumps(cl, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("locked_code_hashes_match", "runtime_closure_all_verified", "closure_files")}))
    print(json.dumps({"overall": v["overall"], "worlds_supporting": v["worlds_supporting"],
                      "problems": v["problems"][:10]}, indent=1))
    for w, pv in v["per_world"].items():
        print(w, json.dumps({k: pv[k] for k in pv if k in ("balanced_held_L", "supports_condition_1", "gates_ok")}))
    print("comparison with locked summarize.py:", json.dumps(agree))


if __name__ == "__main__":
    main()
