"""Joint actuation audit of existing V171 saved native reads; no model imports.

For each pre-update, teacher-forced training read v, solve all 256 rival
inequalities together. This is a label-informed oracle relaxation, not a fitted
controller and not proof that v encodes the answer. Each target gets its own
gate; shared controller parameters and state reachability are relaxed.

IMPORTANT: native teaching changes memory between saved target sites. Combining
site certificates by record therefore describes that saved training-read
sequence, not the frozen no-write own-generation trajectory. It is not an
own-answer accuracy ceiling. Any parent report must retain this distinction.

Run with the project's numpy/scipy runtime:
  .venv_loop/bin/python ceiling171r1.py --output-dir <new output directory>
The parent V171 data, code and counters are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT / "minifly_v171_dev_20261026"
WORLDS = (202610269711, 202610269712)
TOL = 1e-8
CERT_TOL = 2e-7
VARIANTS = (
    "original_box", "parameter_cap_envelope", "nonnegative_unbounded_direction",
    "signed_effective_output",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def lp_margin(A, bounds, equality=None):
    """Maximise t with A g >= t; independently check HiGHS primal/dual proof.

    SciPy minimizes c.x, c=[0,...,0,-1]. Its inequality/upper-bound
    marginals are nonpositive, lower-bound marginals nonnegative. The dual
    value includes finite lower/upper bounds. Because t has no bounds, the
    t-stationarity identity also checks that the rival weights sum to one.
    """
    d = A.shape[1]
    Aub = np.column_stack((-A, np.ones(len(A))))
    c = np.r_[np.zeros(d), -1.0]
    all_bounds = list(bounds) + [(None, None)]
    Aeq = None if equality is None else np.array([np.r_[np.ones(d), 0.0]])
    beq = None if equality is None else np.array([float(equality)])
    result = linprog(c, A_ub=Aub, b_ub=np.zeros(len(A)), A_eq=Aeq,
                     b_eq=beq, bounds=all_bounds, method="highs",
                     options={"dual_feasibility_tolerance": 1e-9,
                              "primal_feasibility_tolerance": 1e-9})
    if not result.success:
        raise RuntimeError({"solver_status": result.status, "message": result.message})
    x = result.x
    gaps = A @ x[:d]
    ineq = result.ineqlin.marginals
    low, high = result.lower.marginals, result.upper.marginals
    eql = np.array([]) if equality is None else result.eqlin.marginals
    dualvec = Aub.T @ ineq + low + high
    dualval = 0.0  # b_ub is zero
    if Aeq is not None:
        dualvec += Aeq.T @ eql
        dualval += float(beq @ eql)
    bound_violation = 0.0
    for j, (lo, hi) in enumerate(all_bounds):
        if lo is not None:
            bound_violation = max(bound_violation, lo - x[j])
            dualval += lo * low[j]
        if hi is not None:
            bound_violation = max(bound_violation, x[j] - hi)
            dualval += hi * high[j]
    primal_violation = max(0.0, float(np.max(Aub @ x)), bound_violation)
    equality_violation = 0.0 if Aeq is None else float(np.max(abs(Aeq @ x - beq)))
    stationarity = float(np.max(abs(c - dualvec)))
    sign_violation = max(0.0, float(np.max(ineq)), float(np.max(-low)),
                         float(np.max(high)))
    dual_gap = abs(float(c @ x) - dualval)
    certificate_valid = bool(max(primal_violation, equality_violation, stationarity,
                                sign_violation, dual_gap) < CERT_TOL)
    if not certificate_valid:
        raise RuntimeError({"bad_LP_certificate": True, "primal": primal_violation,
                            "equality": equality_violation, "stationarity": stationarity,
                            "sign": sign_violation, "gap": dual_gap})
    primal_margin = float(gaps.min())
    # Repair the rival mixture to a demonstrably feasible dual distribution.
    # For any g, min_i A_i g <= sum_i lambda_i A_i g. Maximize that mixture
    # exactly over the box (or nonnegative simplex). This upper bound does
    # not depend on interpreting a small stationarity residual as exact zero.
    lam = np.maximum(0., -ineq)
    lam /= lam.sum()
    mix = lam @ A
    if equality is not None:
        assert all(lo == 0. and hi is None for lo, hi in bounds)
        dual_upper_margin = float(equality * np.max(mix))
    else:
        assert all(lo is not None and hi is not None for lo, hi in bounds)
        dual_upper_margin = float(sum(mix[j] * (hi if mix[j] >= 0. else lo)
                                      for j, (lo, hi) in enumerate(bounds)))
    repaired_gap = max(0., dual_upper_margin-primal_margin)
    assert repaired_gap < CERT_TOL, repaired_gap
    if primal_margin > TOL:
        classification = "STRICTLY_FEASIBLE"
    elif dual_upper_margin < -TOL:
        classification = "CERTIFIED_INFEASIBLE"
    else:
        classification = "TIE_OR_NUMERICAL_BOUNDARY"
    return {
        "classification": classification,
        "optimal_min_margin": float(x[-1]),
        "primal_min_margin": primal_margin,
        "dual_upper_margin": dual_upper_margin,
        "primal_inequality_or_bound_violation": primal_violation,
        "equality_violation": equality_violation,
        "dual_stationarity_residual": stationarity,
        "dual_sign_violation": sign_violation,
        "primal_dual_gap": dual_gap,
        "repaired_dual_gap": repaired_gap,
        "certificate_valid": certificate_valid,
        "solution": x[:d].tolist(),
        "binding_rival_rows": np.flatnonzero(gaps <= primal_margin + TOL).tolist(),
    }


def numerical_checks():
    # Each rival separately can be beaten, yet they cannot both be beaten.
    joint = lp_margin(np.array([[1., -2.], [-2., 1.]]), [(0.5, 1.5)] * 2)
    assert abs(joint["optimal_min_margin"] + .5) < TOL
    assert joint["classification"] == "CERTIFIED_INFEASIBLE"
    assert np.all(np.array([[1., -2.], [-2., 1.]]).sum(1) +
                  .5 * abs(np.array([[1., -2.], [-2., 1.]])).sum(1) > 0)
    good = lp_margin(np.eye(2), [(0.5, 1.5)] * 2)
    assert abs(good["optimal_min_margin"] - 1.5) < TOL
    tie = lp_margin(np.array([[1., -1.], [-1., 1.]]), [(0.5, 1.5)] * 2)
    assert tie["classification"] == "TIE_OR_NUMERICAL_BOUNDARY"
    return {"simultaneous_rivals_defeat_separate_rival_bounds": True,
            "known_feasible_margin": good["optimal_min_margin"],
            "tie_not_claimed_feasible_or_infeasible": True,
            "synthetic_LP_solves": 3}


def load_tape(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(z["metadata"].tobytes()), z["v"].copy()


def solve_site(v, y, C):
    """Pure four-oracle site API; caller owns provenance/counting, no I/O.

    Returns variants keyed by VARIANTS. Each LP has 16 actuation variables,
    one margin variable and all 256 target-vs-rival constraints. Bounds on g
    are independent per site; learned policy restrictions are not imposed.
    """
    v, C = np.asarray(v, dtype=float), np.asarray(C, dtype=float)
    assert v.shape == (16,) and C.shape == (257, 16)
    rivals = np.delete(np.arange(257), int(y))
    D = C[y] - C[rivals]
    D -= D.mean(1, keepdims=True)
    A = D * v[None, :] / 4
    radius = .5 * float(np.tanh(1.25))
    per_variant = {}
    for name in VARIANTS:
        if name == "original_box":
            sol = lp_margin(A, [(.5, 1.5)] * 16)
        elif name == "parameter_cap_envelope":
            sol = lp_margin(A, [(1-radius, 1+radius)] * 16)
        elif name == "nonnegative_unbounded_direction":
            sol = lp_margin(A, [(0., None)] * 16, equality=16.)
        else:
            bnd = [(0., 0.) if x == 0. else (-1., 1.) for x in v]
            sol = lp_margin(D / 4, bnd)
            sol["zero_native_dimensions"] = int(np.sum(v == 0.))
        sol["binding_rivals"] = [int(rivals[j]) for j in sol.pop("binding_rival_rows")]
        per_variant[name] = sol
    return per_variant


def convex_gate_optimum(v, y, C, bounds=(.5, 1.5)):
    """Exact-gradient *convex* saved-read CE instrument, no shared policy fit.

    A first-order convex lower bound provides an independent objective-gap
    certificate. Merely reporting optimizer success is insufficient. This
    instrument is optional and is NOT called by the historical main audit.
    """
    v, C = np.asarray(v, dtype=float), np.asarray(C, dtype=float)
    L = (C - C.mean(1, keepdims=True)) * v[None, :] / 4
    lo, hi = bounds
    def fg(g):
        logits = L @ g
        logz = logsumexp(logits)
        p = np.exp(logits-logz)
        e = p.copy()
        e[y] -= 1
        return float(logz-logits[y]), L.T @ e
    opt = minimize(fg, np.ones(16), method="L-BFGS-B", jac=True,
                   bounds=[(lo, hi)]*16,
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000,
                            "maxls": 50})
    loss, grad = fg(opt.x)
    s = np.where(grad >= 0., lo, hi)
    gap = max(0., float(grad @ (opt.x-s)))
    logits = L @ opt.x
    rivals = np.delete(np.arange(257), y)
    return {"CE_nats": loss, "CE_bits": loss / np.log(2),
            "convex_objective_lower_bound_nats": loss-gap,
            "convex_first_order_gap_nats": gap,
            "probability_target": float(np.exp(logits[y]-logsumexp(logits))),
            "min_target_logit_gap": float(np.min(logits[y]-logits[rivals])),
            "argmax": int(logits.argmax()), "argmax_correct": int(logits.argmax()) == y,
            "solution": opt.x.tolist(), "optimizer_success": bool(opt.success),
            "iterations": int(opt.nit), "gradient_evaluations": int(opt.njev)}


def main(output_dir):
    cpu0, wall0 = time.process_time(), time.monotonic()
    output_dir = Path(output_dir).resolve()
    if output_dir == PARENT or PARENT in output_dir.parents:
        raise ValueError("Refuse to write into frozen parent experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "CEILING_RESULTS.json"
    if result_file.exists():
        raise FileExistsError("Refuse automatic overwrite of completed analysis")
    code_path = ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json"
    words = json.loads(code_path.read_text())["words_by_class"]
    C = np.array([[1. if word & (1 << j) else -1. for j in range(16)]
                  for word in words])
    assert C.shape == (257, 16) and len(set(words)) == 257
    assert np.all(C.sum(1) == 0), "Actual codebook has constant-weight balanced codewords"
    inputs = {str(code_path.relative_to(ROOT)): sha(code_path)}
    test = numerical_checks()
    worlds = {}
    total_sites = total_records = 0
    max_cert = Counter()
    radius = .5 * float(np.tanh(1.25))
    details_hashes = {}
    for world in WORLDS:
        phases = {}
        for phase, nrecords in (("OLD", 64), ("NEW", 16)):
            tape_path = PARENT / "local" / f"cal_{world}_{phase}_pass16_NATIVE.npz"
            inputs[str(tape_path.relative_to(ROOT))] = sha(tape_path)
            metadata, reads = load_tape(tape_path)
            records = defaultdict(list)
            site_rows = []
            for md, v in zip(metadata, reads):
                assert md["phase"] == phase and md["pass_no"] == 16 and md["world"] == world
                y = md["y"]
                rivals = np.delete(np.arange(257), y)
                D = C[y] - C[rivals]
                D -= D.mean(1, keepdims=True)  # actual output centering; zero here
                A = D * v[None, :] / 4
                per_variant = solve_site(v, y, C)
                for name, sol in per_variant.items():
                    for k in ("primal_inequality_or_bound_violation", "equality_violation",
                              "dual_stationarity_residual", "dual_sign_violation", "primal_dual_gap"):
                        max_cert[k] = max(max_cert[k], sol[k])
                separate = A.sum(1) + .5 * abs(A).sum(1)
                old_single = bool(separate.min() < -TOL)
                assert not old_single or per_variant["original_box"]["classification"] == "CERTIFIED_INFEASIBLE"
                row = {"id": md["id"], "site": md["site"], "y": y,
                       "native_argmax_correct": md["argmax"] == y,
                       "native_zero_dimensions": int(np.sum(v == 0)),
                       "old_separate_rival_certificate": old_single,
                       "variants": per_variant}
                site_rows.append(row)
                records[md["id"]].append(row)
            assert len(records) == nrecords
            for rid, rows in records.items():
                assert [x["site"] for x in rows] == list(range(len(rows))), rid
                assert rows[-1]["y"] == 256
            summary = {"records": len(records), "sites": len(site_rows),
                       "native_correct_at_saved_target_sites": sum(x["native_argmax_correct"] for x in site_rows),
                       "native_all_saved_training_targets_correct_records": sum(all(x["native_argmax_correct"] for x in rs) for rs in records.values()),
                       "separate_rival_certified_infeasible_sites": sum(x["old_separate_rival_certificate"] for x in site_rows),
                       "variants": {}}
            for name in VARIANTS:
                counts = Counter(x["variants"][name]["classification"] for x in site_rows)
                by_record = Counter()
                for rs in records.values():
                    cls = [x["variants"][name]["classification"] for x in rs]
                    if "CERTIFIED_INFEASIBLE" in cls:
                        by_record["contains_infeasible_saved_site"] += 1
                    elif all(x == "STRICTLY_FEASIBLE" for x in cls):
                        by_record["all_saved_sites_strictly_feasible"] += 1
                    else:
                        by_record["boundary_unresolved"] += 1
                summary["variants"][name] = {"sites": dict(counts), "saved_training_records": dict(by_record),
                    "min_margin": min(x["variants"][name]["optimal_min_margin"] for x in site_rows),
                    "max_margin": max(x["variants"][name]["optimal_min_margin"] for x in site_rows)}
            details = output_dir / "local" / f"CEILING_{world}_{phase}.json"
            atomic_json(details, site_rows)
            details_hashes[str(details.relative_to(output_dir))] = sha(details)
            phases[phase] = summary
            total_sites += len(site_rows)
            total_records += len(records)
        worlds[str(world)] = phases
    output = {
        "version": "V171-R1", "status": "SAVED_READ_JOINT_ORACLE_AUDIT_COMPLETE",
        "parent_version": "V171", "parent_publication_commit": "09bad998608fc95f37dff50fc54c540ac8a23e9b",
        "worlds": worlds, "sites": total_sites, "distinct_taught_records": total_records,
        "variants": {
            "original_box": "Independent target-informed gates in [0.5,1.5]; joint 256-rival LP",
            "parameter_cap_envelope": {"bounds": [1-radius, 1+radius], "a_abs_bound": 1.25,
                "interpretation": "Closed outer envelope of actual capped gate network; h∈[-1,1], G 8 inputs of abs≤1/8, c abs≤1/4. Shared parameters/state reachability still relaxed."},
            "nonnegative_unbounded_direction": "g>=0 with sum(g)=16; tests all nonzero nonnegative gate directions via harmless positive scaling. Different oracle intervention; not original candidate.",
            "signed_effective_output": "Arbitrary signed gate, with effective vector w=g*v normalized to [-1,1]; zero v dimensions fixed zero. Different oracle intervention; not original candidate."},
        "numerical_checks": test, "certificate_tolerances": {"classification": TOL, "residual": CERT_TOL},
        "max_certificate_residuals": dict(max_cert),
        "counts": {"saved_unique_target_positions": total_sites,
                   "label_informed_oracle_target_reuses": total_sites * len(VARIANTS),
                   "data_LP_solves": total_sites * len(VARIANTS), "synthetic_LP_solves": 3,
                   "data_rival_inequalities": total_sites * len(VARIANTS) * 256,
                   "new_native_queries": 0, "native_imports": 0, "new_training_records": 0,
                   "held_assessment_labels_opened": 0, "parent_trajectories_replayed": 0},
        "limits": [
            "Every oracle sees the target label and independently chooses a gate per site; this is not learned behavior or evidence of stored answer knowledge.",
            "Simultaneous feasibility at one read relaxes shared controller parameters and temporal reachability.",
            "Reads are pre-update TEACHER-FORCED TRAINING states; native writes occur between targets. Record aggregation is not a frozen no-write own-generation ceiling.",
            "Original pass16 OLD precedes NEW training; it is not the final post-NEW OLD checkpoint.",
            "Negative certificates are restrictions at a saved read, not an architecture impossibility theorem.",
            "Signed/nonnegative-unbounded oracle interventions change the original gate restrictions and are not candidate/reference training outcomes.",
            "Existing V171 verdict and original counts remain unchanged."],
        "input_sha256": inputs, "detail_sha256": details_hashes,
        "source_sha256": sha(__file__),
        "resources": {"CPU_seconds": time.process_time()-cpu0,
                      "wall_seconds": time.monotonic()-wall0,
                      "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)},
    }
    atomic_json(result_file, output)
    print(json.dumps({"result": str(result_file), "sites": total_sites,
                      "records": total_records, "CPU_seconds": output["resources"]["CPU_seconds"]}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().output_dir)
