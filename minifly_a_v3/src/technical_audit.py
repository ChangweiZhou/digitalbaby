"""Compile TECHNICAL_AUDIT.json for the FINAL Package A V3-CLAUDE implementation (world 190000; no science).

Mechanism diagnostics are label-free (write counts, gate rates, residual/weight magnitudes); no probe
accuracy is computed or used.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import paths  # noqa: E402
import audit_a  # noqa: E402
import runner  # noqa: E402
import test_tamper  # noqa: E402
import audit_replay  # noqa: E402

RES = paths.ROOT / "results" / "technical_final"
# files a technical/science life actually executes or reads (audit, analysis and test tools are locked separately)
EXECUTED = ("src/paths.py", "src/stores.py", "src/systems.py", "src/runner.py", "src/lock.py", "src/calibrate_r.py",
            "src/drive_technical.py", "src/drive_science.py", "SPEC_LOCK.md", "ARM_ROSTER.json",
            "results/calibration/R_CALIBRATION.json", "package/MANIFEST.json")
PY = sys.executable


def sh(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return {"cmd": " ".join(map(str, cmd)), "returncode": p.returncode,
            "stdout_tail": p.stdout.strip().splitlines()[-1:] if p.stdout.strip() else [],
            "stderr_tail": p.stderr.strip().splitlines()[-2:] if p.stderr.strip() else []}


def diag(arm: str, r: dict) -> dict:
    W = r["mechanism_events"]["W"]
    out = {}
    led = [x for x in W if x[0] in ("w", "s")]
    for bank in sorted({x[2] for x in led}):
        rows = [x for x in led if x[2] == bank]
        out[f"bank{bank}_writes_W"] = sum(x[4] for x in rows)
        out[f"bank{bank}_alpha_L1_W"] = round(sum(x[5] for x in rows), 6)
    if arm in audit_a.R_ARMS:
        rr = [x for x in W if x[0] == "R"]
        permitted = [x for x in rr if x[4]]
        out["private_gate_rate_permitted_W"] = sum(x[3] for x in permitted) / len(permitted)
        out["novelty_median"] = statistics.median(x[2] for x in rr)
        if arm in audit_a.SIGNED:
            s = [abs(v) for x in rr for v in x[6][1:]]
            out["signed_b"] = sorted({x[6][0] for x in rr})
            out["abs_s_mean"] = statistics.fmean(s)
            out["abs_s_zero_count"] = sum(1 for v in s if v == 0.0)
    if arm in audit_a.Z_ARMS:
        z = [x for x in W if x[0] == "Z"]
        out["z_events_W"] = len(z)
        out["native_slow_L1_W"] = round(sum(x[4] for x in z), 6)
        out["gated_slow_L1_W"] = round(sum(x[5] for x in z), 6)
        out["gated_fraction_W"] = out["gated_slow_L1_W"] / out["native_slow_L1_W"]
        out["conflict_coordinates_W"] = sum(x[3] for x in z)
    if arm in audit_a.P_ARMS:
        p = [x for x in W if x[0] == "P"]
        out["pre_norm_delta_L1_total"] = round(sum(x[3] for x in p), 6)
        out["installed_weight_change_L1_total"] = round(sum(x[4] for x in p), 6)
        out["weight_max_end"] = p[-1][5]
        out["weight_min_end"] = p[-1][6]
        out["updates_per_store"] = len(p) // 4
    return out


def _replay(arm):
    r = test_tamper.load(arm)
    try:
        if arm in audit_a.Z_ARMS:
            return {"pass": True, **audit_replay.replay_z(r, test_tamper.load("Z2") if arm == "Z2_rand" else None)}
        if arm in audit_a.P_ARMS:
            return {"pass": True, **audit_replay.replay_p(r)}
        cal = json.loads((paths.ROOT / "results" / "calibration" / "R_CALIBRATION.json").read_text())
        out = audit_replay.replay_r_novelty(r)
        if arm in audit_a.SIGNED:
            out.update(audit_replay.replay_r_shared(r, cal["scales"]["shared"]))
        return {"pass": True, **out}
    except Exception as exc:  # noqa: BLE001
        return {"pass": False, "error": repr(exc)[:400]}


def main() -> None:
    cal_path = paths.ROOT / "results" / "calibration" / "R_CALIBRATION.json"
    cal = json.loads(cal_path.read_text())
    arms = list(runner.ARMS)
    receipts, audits, diags, res, hashes = {}, {}, {}, {}, {}
    for arm in arms:
        raw = (RES / arm / "190000.json.gz").read_bytes()
        r = json.loads(gzip.decompress(raw))
        receipts[arm] = hashlib.sha256(raw).hexdigest()
        audits[arm] = audit_a.audit_receipt(r, **test_tamper.kwargs_for(arm))
        diags[arm] = diag(arm, r)
        res[arm] = {**r["resources"], "receipt_bytes": len(raw)}
        hashes[arm] = r["source_sha256"]
    same_source = all(h == hashes[arms[0]] for h in hashes.values())
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        replay = dict(zip(arms, ex.map(_replay, arms)))
    # the calibration file predates the final Z/P edits: recompute it under the final code and require identity
    import calibrate_r
    with ProcessPoolExecutor(4) as ex:
        rep = list(ex.map(calibrate_r.one_world, calibrate_r.WORLDS))
    reproduced = {
        "scales_equal": (calibrate_r.scale([r for w in rep for r in w["shared"]]) == cal["scales"]["shared"] and
                         calibrate_r.scale([r for w in rep for r in w["private"]]) == cal["scales"]["private"]),
        "theta_equal": statistics.median([v for w in rep for v in w["novelty"]]) == cal["theta"],
        "per_world_values_equal": all(a[k] == b[k] for a, b in zip(rep, cal["per_world"])
                                      for k in ("novelty", "shared", "private", "cues", "fixture_digest"))}
    cal_src_ok = all(reproduced.values())
    current = runner.source_hashes()
    import lock
    lock.package_files()
    tamper = test_tamper.run()
    # a joint change of Z2's gated amount and claimed target is self-consistent in the log; the replay must reject it
    import copy
    z2 = copy.deepcopy(test_tamper.load("Z2"))
    zr = next(z for z in z2["mechanism_events"]["W"] if z[0] == "Z" and z[2] == 0 and 0 < z[5] < 0.9 * z[4])
    k = max(zr[7], key=lambda name: zr[7][name][2])
    bump = min(zr[7][k][1] - zr[7][k][2], zr[5]) * 0.5
    zr[5] += bump; zr[6] += bump; zr[7][k][2] += bump  # noqa: E702
    try:
        audit_a.audit_receipt(z2)
        log_ok = True
    except AssertionError:
        log_ok = False
    try:
        audit_replay.replay_z(z2)
        replay_rejects = False
    except audit_replay.ReplayMismatch:
        replay_rejects = True
    tamper["Z2_joint_gated_and_target_via_replay"] = {
        "log_audit": "accepted (self-consistent by construction)" if log_ok else "rejected",
        "replay": "rejected" if replay_rejects else "ACCEPTED"}
    unit = sh([PY, "tests/test_units.py"], paths.ROOT)
    pd = {a: [row[3] for row in test_tamper.load(a)["mechanism_events"]["W"] if row[0] == "PD"]
          for a in arms if a in audit_a.P_ARMS}
    verify_full = sh([PY, "verify_bundle.py"], paths.PKG)
    gate_all = sh([PY, "causal_branch_gate.py", "--receipts-dir", str(RES)], paths.PKG)
    per_arm_s = {a: res[a]["life_s"] + res[a]["construct_s"] for a in arms}
    core_h = sum(per_arm_s.values()) * 64 / 3600
    doc = {
        "schema": "MINIFLY-A3-CLAUDE-TECHNICAL-AUDIT-v1",
        "package": "MINIFLY_A_V3_CLAUDE — new implementation from the V3 scaffold; not Muse V2, not a resume or audit of it",
        "world": 190000, "science_trajectories_run": 0, "science_source_lock": "not written (technical stage only)",
        "roster_technical": arms,
        "not_implemented": {"P3": "technically unqualified/unresolved (Muse's reported failure unverified, no receipt); "
                                  "no comparison or replacement", "P3_shuffle": "unrun"},
        "family": "m=26 retained program-wide; P3's two comparisons unavailable",
        "technical_score_used_for_tuning": False, "probe_accuracy_computed": False,
        "r_calibration": {"worlds": cal["worlds"], "scales": cal["scales"], "theta": cal["theta"],
                          "sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
                          "calibration_source_matches_current": all(
                              runner.source_hashes().get(f) == h for f, h in cal["source_sha256"].items()),
                          "reproduced_exactly_under_final_code": reproduced},
        "receipt_sha256": receipts, "independent_audit": audits, "independent_replay": replay,
        "tamper_rejection": tamper,
        "all_tamper_rejected": all(v != "ACCEPTED" for t in tamper.values() for v in t.values()),
        "tamper_note": "regression battery, not exhaustive mechanism validation; see independent_replay",
        "v3_gate_receipts_dir": gate_all, "unit_tests": unit, "verify_bundle_full": verify_full,
        "receipts_share_one_source_hash_set": same_source, "source_sha256": hashes[arms[0]],
        "executed_closure_matches_current_tree": all(current.get(f) == hashes[arms[0]].get(f) for f in EXECUTED),
        "executed_closure": list(EXECUTED),
        "tooling_changed_after_technical_lives": sorted(f for f in set(current) | set(hashes[arms[0]])
                                                        if current.get(f) != hashes[arms[0]].get(f)),
        "mechanism_diagnostics_W_branch": diags,
        "p_concentration_store0_W": pd,
        "resources": res,
        "estimate_64_worlds": {"per_arm_hours": {a: round(s * 64 / 3600, 2) for a, s in per_arm_s.items()},
                               "core_hours_total": round(core_h, 1),
                               "wall_hours_at_4_workers": round(core_h / 4, 1),
                               "caveat": "technical lives ran one arm per worker; Package B's science lives ran ~1.8x "
                                         "slower than its technical measurement at the same concurrency"},
        "host": {"python": platform.python_version(), "machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpus": __import__("os").cpu_count()},
    }
    doc["pass"] = (all(a["pass"] for a in audits.values()) and all(v["pass"] for v in replay.values()) and
                   doc["all_tamper_rejected"] and same_source and
                   doc["executed_closure_matches_current_tree"] and cal_src_ok and
                   set(doc["tooling_changed_after_technical_lives"]) <= {"src/technical_audit.py"} and unit["returncode"] == 0 and
                   verify_full["returncode"] == 0 and gate_all["returncode"] == 0)
    (RES / "TECHNICAL_AUDIT.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": doc["pass"], "same_source": same_source, "cal_src_ok": cal_src_ok,
                      "unit": unit["returncode"], "verify": verify_full, "gate": gate_all["stdout_tail"],
                      "estimate": doc["estimate_64_worlds"]}, indent=1))


if __name__ == "__main__":
    main()
