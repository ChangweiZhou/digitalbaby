"""V149 summary: tables from a RESULTS.json and, for the main run, the decision rule and the predictions as fixed in
the design (section 6 and 7) and LOCK.json.  python summarize.py results/<tag>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def score(r, margin):
    L, A, ok = r["ladder"], r["arms"], r["readout_checks_all_pass"]
    B, U = L["bigram"]["held_bpb"], L["unigram"]["held_bpb"]
    C = A["PERM-obs"]["R1"]["held_bpb"]
    out = {"readout_checks_all_pass": ok, "C_PERM-obs_R1": C, "B_bigram": B, "unigram": U}
    if not ok:
        out["decision"] = "CANNOT_TEST_AS_SPECIFIED"
        return out
    out["decision"] = "CONTEXT_FIRST" if C >= B - margin else "OUTPUT_FIRST"
    out["P1"] = ("AS PREDICTED" if B - margin <= C < U else
                 "WEAKENED (C beats bigram by more than the margin)" if C < B - margin else
                 "NOT AS PREDICTED (C not better than unigram)")
    last, px = A["PERM-last"]["R1"]["held_bpb"], A["PERM-x"]["R1"]["held_bpb"]
    out["P2"] = ("WEAKENED (the trace adds history: PERM-x beats PERM-last by more than 0.10)" if px < last - 0.10 else
                 "AS PREDICTED" if abs(last - B) <= 0.15 else
                 "NOT AS PREDICTED (PERM-last not within 0.15 of bigram)")
    out["P2_values"] = {"PERM-last_R1": last, "PERM-x_R1": px, "bigram": B}
    r2 = A["PERM-obs"]["R2"]["held_bpb"]
    out["P3"] = "AS PREDICTED" if r2 >= C + 0.30 else "WEAKENED (R2 within 0.30 of R1)"
    out["P3_values"] = {"PERM-obs_R2": r2, "PERM-obs_R1": C}
    return out


def main():
    d = Path(sys.argv[1])
    r = json.loads((d / "RESULTS.json").read_text())
    if not r.get("complete"):
        raise SystemExit("RESULTS incomplete")
    L, rc, A = r["ladder"], r["readout_checks"], r["arms"]
    f = lambda x: f"{x:.3f}"  # noqa: E731
    lines = [f"# V149 summary: {r['tag']} ({r['text']}; taught/calibration/held-out KB {r['sizes_kb']})", "",
             "Counting ladder, held-out bits/symbol: " + ", ".join(f"{k} {f(v['held_bpb'])}" for k, v in L.items()), "",
             "## Readout checks (R1 on evaluator-side codes)", "", "| Code | held bits/symbol | L2 | check |",
             "|---|---:|---:|---|"]
    names = {"onehot_last1": "last1_within_0.15_of_bigram", "onehot_last2": "last2_within_0.20_of_ppmc2",
             "const": "const_not_below_unigram_minus_0.02"}
    for k, v in rc.items():
        key = names.get(k)
        lines.append(f"| {k} | {f(v['held_bpb'])} | {v['lam']:g} | {'PASS' if r['readout_checks_pass'].get(key) else 'FAIL'} |")
    lines += ["", "## Arms (native KC states of an untaught memory)", "",
              "| Arm | R1 held | R1 acc | R1 L2 | R2 held | R2 acc | R2 mix |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k, v in A.items():
        lines.append(f"| {k} | {f(v['R1']['held_bpb'])} | {f(v['R1']['held_acc'])} | {v['R1']['lam']:g} | "
                     f"{f(v['R2']['held_bpb'])} | {f(v['R2']['held_acc'])} | {v['R2']['lam']:g} |")
    lines += ["", "Line-bootstrap SDs: " + ", ".join(f"{k} {v:.3f}" for k, v in r["bootstrap_sd"].items()),
              "", f"CPU: states {r['states_cpu_s']} s, total {r['total_cpu_s']} s; peak memory {r['peak_rss_mb']} MB."]
    out = {"tag": r["tag"], "text": r["text"]}
    lock_path = HERE / "LOCK.json"
    if r["tag"] == "main" and lock_path.exists():
        lock = json.loads(lock_path.read_text())
        out["scoring"] = score(r, lock["decision"]["margin_bits"])
        lines += ["", "## Decision and predictions (design sections 6-7; fixed in LOCK.json before the run)"]
        lines += [f"* {k}: {v}" for k, v in out["scoring"].items()]
    (d / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (d / "SUMMARY.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
