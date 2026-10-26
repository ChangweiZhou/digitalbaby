"""Apply PLAN.md section 3 to results/<tag>/LINEAR.json and REFERENCE.json; write SUMMARY.json and SUMMARY.md."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V153_P81, V153_PPM4 = 2.9628953791236685, 2.6062837325973773
BEAT, ROOM = V153_PPM4 - 0.05, V153_P81 - 0.10          # 2.556 and 2.863, fixed in PLAN.md


def boot_sd(a, b, n, reps=1000, seed=0):
    a, b, n = np.asarray(a), np.asarray(b), np.asarray(n, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(n), len(n))
        d.append(a[i].sum() / n[i].sum() - b[i].sum() / n[i].sum())
    return float(np.std(d))


def main(tag="main"):
    d = HERE / "results" / tag
    L = json.loads((d / "LINEAR.json").read_text())
    F = json.loads((d / "REFERENCE.json").read_text())
    n = np.load(d / "states.npz")["held_line_len"]          # held line lengths, for the bootstrap
    t = {"T1": L["T1"]["held_bpb"], "T2": L["T2"]["held_bpb"], "T3": F["T3"]["held_bpb"], "T4": F["T4"]["held_bpb"]}
    ppm4 = L["ladder"]["ppmc4"]
    parity = L["parity"]
    qual = {"T4_at_most_2.90": t["T4"] <= 2.90,
            "T3_not_cut_off": not F["T3"]["cut_off_by_budget"], "T4_not_cut_off": not F["T4"]["cut_off_by_budget"]}
    if parity["applies"] and not (parity["T2_within_0.005_of_v153"] and parity["ladder_within_0.001_of_v153"]):
        verdict = "PARITY_FAILED"
    elif not all(qual.values()):
        verdict = "REFERENCE_NOT_READY"
    elif t["T3"] <= BEAT:
        verdict = "LEARNED_CAN_BEAT_COUNTING"
    elif t["T3"] <= ROOM:
        verdict = "ROOM_BUT_NOT_COUNTING"
    else:
        verdict = "NO_ROOM_AT_40KB"
    sds = {"T3-T2": boot_sd(F["T3"]["lines"], L["T2"]["lines"], n),
           "T1-T2": boot_sd(L["T1"]["lines"], L["T2"]["lines"], n),
           "T4-PPM4": boot_sd(F["T4"]["lines"], L["ladder"]["ppmc4"]["lines"], n)}
    s = {"tag": tag, "held": t, "ladder": {k: v["held_bpb"] for k, v in L["ladder"].items()},
         "differences": {"T3-T2": t["T3"] - t["T2"], "T1-T2": t["T1"] - t["T2"], "T4-PPM4": t["T4"] - ppm4["held_bpb"]},
         "bootstrap_sd": sds, "thresholds": {"beat": BEAT, "room": ROOM}, "parity": parity,
         "code_parity_first20": L["parity_code_vs_v153_collector_first20"], "qualification": qual, "verdict": verdict,
         "chosen": {"T1_lam": L["T1"]["lam"], "T2_lam": L["T2"]["lam"], "T3": F["T3"]["chosen"], "T4": F["T4"]["chosen"]},
         "accuracy": {"T1": L["T1"]["held_acc"], "T2": L["T2"]["held_acc"], "T3": F["T3"]["held_acc"],
                      "T4": F["T4"]["held_acc"]},
         "train_class_bits": {"T1": L["T1"]["train_class_bits"], "T2": L["T2"]["train_class_bits"],
                              "T3": [r["train_class_bits"] for r in F["T3"]["grid"]
                                     if (r["H"], r["dropout"]) == (F["T3"]["chosen"]["H"], F["T3"]["chosen"]["dropout"])][0],
                              "T4": [r["train_class_bits"] for r in F["T4"]["grid"]
                                     if (r["H"], r["dropout"]) == (F["T4"]["chosen"]["H"], F["T4"]["chosen"]["dropout"])][0]},
         "grids": {"T1": L["T1"]["path"], "T2": L["T2"]["path"],
                   "T3": [{k: r[k] for k in ("H", "dropout", "best_epoch", "epochs_run", "calib_bpb", "held_bpb")}
                          for r in F["T3"]["grid"]],
                   "T4": [{k: r[k] for k in ("H", "dropout", "best_epoch", "epochs_run", "calib_bpb", "held_bpb")}
                          for r in F["T4"]["grid"]]},
         "cpu_s": {"collect": L["cpu_s_total"], "reference": F["cpu_s_total"]}}
    (d / "SUMMARY.json").write_text(json.dumps(s, indent=1))
    lines = [f"# Headroom check summary: {tag}", "",
             "Ladder (held bits/symbol): " + ", ".join(f"{k} {v:.3f}" for k, v in s["ladder"].items()), "",
             "| Reading | held | acc | train (class bits) |", "|---|---:|---:|---:|"]
    for k in ("T1", "T2", "T3", "T4"):
        lines.append(f"| {k} | {t[k]:.3f} | {s['accuracy'][k]:.3f} | {s['train_class_bits'][k]:.3f} |")
    lines += ["", "Differences (line-bootstrap SD): " + ", ".join(
        f"{k} {v:+.3f} ({sds[k]:.3f})" for k, v in s["differences"].items()),
        "", f"Qualification: {qual}", f"Parity: {parity}; code parity {s['code_parity_first20']}",
        f"Thresholds: beat counting <= {BEAT:.3f}; room <= {ROOM:.3f}", f"**Verdict: {verdict}**", ""]
    (d / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
