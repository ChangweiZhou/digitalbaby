"""V155 decision rules (design sections 5 and 12), applied to score files. No numbers are entered by hand."""
import numpy as np

V153_P81_HELD = 2.9628953791236685      # FROZEN on Thursday (V153 rehearsal; reproduced by the headroom check)


def boot_sd(a, b, n, reps=1000, seed=0):
    a, b, n = np.asarray(a), np.asarray(b), np.asarray(n, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        d.append(a[i].sum() / n[i].sum() - b[i].sum() / n[i].sum())
    return float(np.std(d))


def testbed_verdict(scores, th):
    """Design section 12.9: parity, ceiling (dense free reference), floor (permuted birth weights), qualification."""
    A = scores["arms"]
    n = np.asarray(scores["held_line_len"])
    f, pm, rn, rd = (A[k]["held_bpb"] for k in ("FROZEN", "PERM", "REF-N", "REF-D"))
    checks = {
        "parity_frozen": (abs(f - V153_P81_HELD) <= th["parity"]) if scores["text"] == "thursday" else None,
        "ceiling_dense_reference_headroom": f - rd >= th["g0"],
        "floor_permuted_keys_no_gain": pm >= f - th["floor_tol"],
        "references_not_cut_off": not (A["REF-N"]["cut_off_by_budget"] or A["REF-D"]["cut_off_by_budget"]),
    }
    if checks["parity_frozen"] is False or not checks["floor_permuted_keys_no_gain"]:
        verdict = "INVALID"
    elif checks["ceiling_dense_reference_headroom"] and checks["references_not_cut_off"]:
        verdict = "CONTINUE"
    else:
        verdict = "CANNOT_TEST_CLAIM_AS_SPECIFIED"
    return {"held": {"FROZEN": f, "PERM": pm, "REF-N": rn, "REF-D": rd},
            "differences": {"FROZEN-REF-D": f - rd, "FROZEN-REF-N": f - rn, "REF-N-REF-D (wiring, descriptive)": rn - rd,
                            "PERM-FROZEN": pm - f},
            "bootstrap_sd": {k: boot_sd(A["FROZEN"]["lines"], A[k]["lines"], n) for k in ("PERM", "REF-N", "REF-D")},
            "chosen_dropout": {k: A[k]["chosen_dropout"] for k in ("REF-N", "REF-D")},
            "checks": checks, "verdict": verdict}
