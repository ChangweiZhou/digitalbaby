"""Independent joint actuation and convex-CE reduction of frozen prefix reads.

Input is produced by credit171r1.py using new, explicitly counted no-write
checkpoint queries. This reducer imports no native model and makes no queries.
All correct prefixes share one frozen checkpoint per world/stage. Consequently
an independent label-informed gate feasible at every required prefix really
constructs a correct own-output/EOS path in the relaxed oracle. It does not
construct a single shared recurrent policy and is not a learnability result.

Only the original and actual-cap outer boxes are solved here. Historical
nonnegative-unbounded and signed relaxations are already available in the
separate CEILING_RESULTS.json; no LP budget expansion is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from collections import Counter, defaultdict

START_CPU, START_WALL = time.process_time(), time.monotonic()
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.dont_write_bytecode = True

import numpy as np
import scipy
from scipy.special import logsumexp
import ceiling171r1 as ceiling

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "minifly_v171_dev_20261026"
TOL = 1e-8
CONVEX_GAP_TOL = 1e-7


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, obj):
    ceiling.atomic_json(HERE / name, obj)


def resource_info():
    return {"CPU_seconds": time.process_time()-START_CPU,
            "wall_seconds": time.monotonic()-START_WALL,
            "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}


def basic_budget():
    assert time.process_time()-START_CPU < 1800, "CPU_CAP"
    assert time.monotonic()-START_WALL < 3600, "WALL_CAP"
    assert resource_info()["peak_RSS_bytes"] < 1073741824, "RSS_CAP"


def stage_own_correct(published, world, stage, phase):
    wr = published["worlds"][str(world)]["panels"]
    if stage == "FINAL":
        return set(wr["final_OLD" if phase == "OLD" else "NEW256"]["NATIVE"]["correct_ids"])
    return set(wr["OLD16"]["NATIVE"]["correct_ids"])


def main():
    assert not (HERE / "FROZEN_REDUCTION_STARTED.json").exists(), "No automatic retries"
    source = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
              str(Path(ceiling.__file__).relative_to(ROOT)): sha(ceiling.__file__)}
    save("FROZEN_REDUCTION_SOURCE_LOCK.json", source)
    save("FROZEN_REDUCTION_STARTED.json", {
        "status": "STARTED", "source_locked_before_first_input_read": True,
        "LP_variants": ["original_box", "parameter_cap_envelope"],
        "convex_selection": "All FINAL NEW sites, plus first 16 FINAL OLD sites per world sorted by (id,site)",
        "convex_max_solver_evaluations_per_site": 127,
        "convex_max_additional_final_certificate_evaluations_per_site": 1,
        "convex_additional_evaluation_ceiling": 15616,
        "new_native_queries": 0, "new_training_records": 0})
    input_path = HERE / "local/FROZEN_PREFIX_READS.json"
    code_path = ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json"
    pub_path = PARENT / "CALIBRATION_RESULTS.json"
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (input_path, code_path, pub_path)}
    sites = json.loads(input_path.read_text())
    published = json.loads(pub_path.read_text())
    words = json.loads(code_path.read_text())["words_by_class"]
    C = np.array([[1. if w & (1 << j) else -1. for j in range(16)] for w in words])
    Cc = C - C.mean(1, keepdims=True)
    assert len(sites) == 920 and C.shape == (257, 16)
    radius = .5 * float(np.tanh(1.25))
    variants = {"original_box": (.5, 1.5), "parameter_cap_envelope": (1-radius, 1+radius)}
    records = defaultdict(list)
    details = []
    lp_count = 0
    for site in sites:
        world, stage, phase = site["world"], site["stage"], site["phase"]
        assert world in (202610269711, 202610269712)
        assert stage in ("OLD16", "FINAL") and phase in ("OLD", "NEW")
        v, y = np.array(site["v"]), site["y"]
        assert v.shape == (16,) and 0 <= y <= 256
        rivals = np.delete(np.arange(257), y)
        A = (Cc[y] - Cc[rivals]) * v[None, :] / 4
        logits = Cc @ v / 4
        row = {k:site[k] for k in ("world", "stage", "phase", "id", "site", "y")}
        row["native_argmax"] = int(logits.argmax())
        row["native_target_correct"] = row["native_argmax"] == y
        row["native_target_bits"] = float((logsumexp(logits)-logits[y])/np.log(2))
        row["variants"] = {}
        for name, (lo, hi) in variants.items():
            sol = ceiling.lp_margin(A, [(lo, hi)]*16)
            sol["binding_rivals"] = [int(rivals[j]) for j in sol.pop("binding_rival_rows")]
            row["variants"][name] = sol
            lp_count += 1
        details.append(row)
        records[(world, stage, phase, site["id"])].append(row)
        basic_budget()
    assert lp_count == 1840 and len(records) == 320
    summaries = {}
    for world in (202610269711, 202610269712):
        stages = {}
        for stage in ("OLD16", "FINAL"):
            phases = {}
            for phase, expected_n in (("OLD",64), ("NEW",16)):
                rr = {key:rows for key, rows in records.items() if key[:3] == (world,stage,phase)}
                assert len(rr) == expected_n
                for key, rows in rr.items():
                    assert [x["site"] for x in rows] == list(range(len(rows))), key
                    assert rows[-1]["y"] == 256
                phase_summary = {"records": expected_n, "target_sites": sum(map(len,rr.values())),
                                 "exposure_at_checkpoint": "Already taught" if phase == "OLD" or stage == "FINAL" else "Future-taught calibration NEW; not taught yet at OLD16",
                                 "native_all_correct_prefixes": sum(all(x["native_target_correct"] for x in rows) for rows in rr.values()),
                                 "variants": {}}
                published_correct = stage_own_correct(published, world, stage, phase)
                phase_summary["published_native_own_correct"] = sum(key[3] in published_correct for key in rr)
                # Correct argmax at every correct prefix iff deterministic own path is exact.
                assert phase_summary["native_all_correct_prefixes"] == phase_summary["published_native_own_correct"]
                for name in variants:
                    by_site = Counter(x["variants"][name]["classification"] for rows in rr.values() for x in rows)
                    by_record = Counter()
                    for rows in rr.values():
                        cls = [x["variants"][name]["classification"] for x in rows]
                        if "CERTIFIED_INFEASIBLE" in cls:
                            by_record["certified_impossible_correct_path"] += 1
                        elif all(x == "STRICTLY_FEASIBLE" for x in cls):
                            by_record["strict_correct_path_witness"] += 1
                        else:
                            by_record["boundary_unresolved"] += 1
                    phase_summary["variants"][name] = {
                        "sites": dict(by_site), "records": dict(by_record),
                        "independent_oracle_max_correct_records_lower": by_record["strict_correct_path_witness"],
                        "shared_policy_correct_records_upper": expected_n-by_record["certified_impossible_correct_path"],
                        "note": "Oracle gates may independently depend on target and prefix. Shared learned controller reachability is relaxed."}
                phases[phase] = phase_summary
            stages[stage] = phases
        summaries[str(world)] = stages

    # Fixed, outcome-independent convex subset; not selected using LP outcomes.
    subset = [x for x in details if x["stage"] == "FINAL" and x["phase"] == "NEW"]
    for world in (202610269711, 202610269712):
        subset += sorted([x for x in details if x["world"] == world and x["stage"] == "FINAL" and x["phase"] == "OLD"], key=lambda x:(x["id"],x["site"]))[:16]
    assert len(subset) == 122
    by_key = {(s["world"],s["stage"],s["id"],s["site"]):s for s in sites}
    actual_convex_calls = 0
    real_minimize = ceiling.minimize
    fit_solver_calls = 0
    def bounded_minimize(fun, *args, **kwargs):
        nonlocal actual_convex_calls, fit_solver_calls
        fit_solver_calls = 0
        def metered_fun(x):
            nonlocal actual_convex_calls, fit_solver_calls
            if fit_solver_calls >= 127:
                raise RuntimeError("PER_FIT_EVALUATION_CAP")
            fit_solver_calls += 1
            actual_convex_calls += 1
            return fun(x)
        opts = dict(kwargs.get("options", {}))
        opts.update(maxfun=96, maxiter=96, maxls=20)
        kwargs["options"] = opts
        return real_minimize(metered_fun, *args, **kwargs)
    ceiling.minimize = bounded_minimize
    convex_details = []
    for row in subset:
        site = by_key[(row["world"],row["stage"],row["id"],row["site"])]
        before = actual_convex_calls
        try:
            sol = ceiling.convex_gate_optimum(np.array(site["v"]),site["y"],C)
            actual_convex_calls += 1  # final objective and convex certificate evaluation
            assert sol["gradient_evaluations"]+1 == actual_convex_calls-before
            solved = sol["convex_first_order_gap_nats"] <= CONVEX_GAP_TOL
            convex_details.append({**{k:row[k] for k in ("world","stage","phase","id","site","y")},
                                   "before_bits": row["native_target_bits"], "result": sol,
                                   "objective_gap_qualified": bool(solved),
                                   "charged_objective_gradient_evaluations": actual_convex_calls-before})
        except RuntimeError as error:
            if str(error) != "PER_FIT_EVALUATION_CAP":
                raise
            convex_details.append({**{k:row[k] for k in ("world","stage","phase","id","site","y")},
                                   "status": "UNRESOLVED_PER_FIT_EVALUATION_CAP",
                                   "charged_objective_gradient_evaluations": actual_convex_calls-before})
        assert actual_convex_calls <= 15616
        basic_budget()
    ceiling.minimize = real_minimize
    convex_summary = {}
    for world in (202610269711,202610269712):
        ss = [x for x in convex_details if x["world"] == world]
        ok = [x for x in ss if "result" in x]
        convex_summary[str(world)] = {
            "selected_FINAL_NEW_sites": sum(x["phase"] == "NEW" for x in ss),
            "selected_FINAL_OLD_sites": sum(x["phase"] == "OLD" for x in ss),
            "completed_results": len(ok),
            "objective_gap_qualified": sum(x.get("objective_gap_qualified",False) for x in ss),
            "unresolved": sum(not x.get("objective_gap_qualified",False) for x in ss),
            "mean_native_bits_on_completed_subset": float(np.mean([x["before_bits"] for x in ok])),
            "mean_convex_box_bits_on_completed_subset": float(np.mean([x["result"]["CE_bits"] for x in ok])),
            "maximum_objective_gap_nats": max(x["result"]["convex_first_order_gap_nats"] for x in ok),
            "argmax_correct_after_convex_CE_fit_sites": sum(x["result"]["argmax_correct"] for x in ok),
            "note": "The CE optimum need not maximize top-1 margin. Partial OLD subset is not an all-80 loss estimate."}
    save("local/FROZEN_CEILING_DETAILS.json", details)
    save("local/FROZEN_CONVEX_DETAILS.json", convex_details)
    for p,h in source.items():
        assert sha(ROOT/p) == h, "SOURCE_CHANGED_DURING_REDUCTION"
    for p,h in input_hashes.items():
        assert sha(ROOT/p) == h, "INPUT_CHANGED_DURING_REDUCTION"
    save("FROZEN_CEILING_RESULTS.json", {
        "status": "COMPLETE", "version": "V171-R1", "worlds": summaries,
        "convex_summary": convex_summary,
        "counts": {"new_native_queries":0,"new_training_records":0,"frozen_prefix_records":len(records),
                   "saved_target_positions":len(sites),"LP_solves":lp_count,
                   "LP_rival_inequalities":lp_count*256,"convex_fits":len(convex_details),
                   "convex_objective_gradient_target_reuses":actual_convex_calls},
        "limits": ["New counted native reads were collected by credit171r1.py; this reducer adds zero native queries.",
                   "Oracle gates independently see each target. Reachability in this relaxation is not evidence of stored answer information or a shared learned policy.",
                   "Correct-prefix native states remain frozen and receive no teaching, so grouped certificates now apply to the exact own-output/EOS path of this relaxed oracle.",
                   "Actual-cap box remains an outer envelope of the parameter-constrained recurrent network.",
                   "OLD16 NEW records are future-taught calibration items, not yet taught at that checkpoint.",
                   "Convex CE fits use a declared subset (all FINAL NEW plus 16 FINAL OLD sites per world); they are independent per-site instruments and do not fit the controller.",
                   "Historical V171 verdict is unchanged. No new own-output performance improvement or E3 claim is made."],
        "convex_gap_tolerance_nats":CONVEX_GAP_TOL,"source_sha256":source,"input_sha256":input_hashes,
        "detail_sha256":{p:sha(HERE/p) for p in ("local/FROZEN_CEILING_DETAILS.json","local/FROZEN_CONVEX_DETAILS.json")},
        "numpy_version":np.__version__,"scipy_version":scipy.__version__,"resources":resource_info()})
    print(json.dumps({"status":"COMPLETE","LP_solves":lp_count,"convex_fits":len(convex_details),
                      "convex_objective_gradient_target_reuses":actual_convex_calls,"CPU_seconds":resource_info()["CPU_seconds"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        save("FROZEN_REDUCTION_FAILURE.json", {"error":str(error),"resources":resource_info(),
              "traceback_redacted":traceback.format_exc().replace(str(ROOT),"$REPO").replace(str(Path.home()),"$HOME")})
        raise
