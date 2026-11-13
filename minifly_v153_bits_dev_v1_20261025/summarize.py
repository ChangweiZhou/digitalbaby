"""V153 summary and scoring: tables from a RESULTS.json and the predictions/decision rule exactly as in the design
(sections 5-6). python summarize.py results/<stage>/<tag>
"""
import json
import sys
from pathlib import Path


def score(r):
    out = {}
    if "A" in r:
        A = {c: r["A"][c]["held_bpb"] for c in r["A"]}
        scr = {"fade_0.807": A["P81"] - A["N81"], "fade_0.95": A["P95"] - A["N95"]}
        fade = {"scrambled": A["P95"] - A["P81"], "unscrambled": A["N95"] - A["N81"]}
        if any(abs(v) > 0.10 for v in scr.values()):
            p1 = "WEAKENED (scrambling effect above 0.10 at some fade)"
        elif all(v >= 0.20 for v in fade.values()):
            p1 = "AS PREDICTED"
        else:
            p1 = "NOT AS PREDICTED (scrambling small, but fade effect below 0.20)"
        out["P1"] = {"result": p1, "scrambling_effect": scr, "fade_effect": fade}
        out["best_cell"] = min(A, key=A.get)
    if "B" in r:
        B = r["B"]
        k = max(int(x.split("pass")[1]) for x in B["ONLINE"]["grid"][0] if x.startswith("held_bpb_pass"))
        best = next(g for g in B["ONLINE"]["grid"] if g["eta"] == B["ONLINE"]["chosen_eta"])
        r1 = B["R1"]["held_bpb"]
        g1, gk = best["held_bpb_pass1"] - r1, best[f"held_bpb_pass{k}"] - r1
        p2 = ("AS PREDICTED" if gk <= 0.15 and g1 <= 0.40 else
              "WEAKENED (gap after the last pass above 0.30)" if gk > 0.30 else
              "PARTLY (gap after the last pass between 0.15 and 0.30, or first-pass gap above 0.40)")
        out["P2"] = {"result": p2, "gap_pass1": g1, f"gap_pass{k}": gk}
        port = B["C16"]["held_bpb"] - r1
        out["P3"] = {"result": "AS PREDICTED" if port >= 1.0 else "WEAKENED (port cost below 1.0)", "port_cost": port}
        nat = B["NATIVE"]["held_bpb"] - B["C16"]["held_bpb"]
        out["P4"] = {"result": "AS PREDICTED" if nat >= 0.5 else "WEAKENED (native-structure cost below 0.5)",
                     "native_structure_cost": nat}
        costs = {"online": gk, "port": port, "native_structure": nat}
        order = sorted(costs, key=costs.get, reverse=True)
        target = "port" if costs[order[0]] - costs[order[1]] < 0.20 and "port" in order[:2] else order[0]
        out["decision"] = {"costs": costs, "next_round_targets": target}
    return out


def main():
    d = Path(sys.argv[1])
    r = json.loads((d / "RESULTS.json").read_text())
    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = r["ladder"]
    lines = [f"# V153 summary: {r['stage']} / {r['tag']} ({r['text']}, KB {r['sizes_kb']})", "",
             "Ladder (held bits/symbol): " + ", ".join(f"{k} {f(v['held_bpb'])}" for k, v in L.items())]
    if "testbed" in r:
        lines += ["", "Testbed: " + ", ".join(f"{k} {f(v['held_bpb'])}" for k, v in r["testbed"].items()),
                  f"Checks: {r['testbed_pass']} -> {r['testbed_verdict']}"]
    if "A" in r:
        lines += ["", "## Block A (R1 on binary KC codes)", "", "| Cell | held | acc | L2 |", "|---|---:|---:|---:|"]
        lines += [f"| {c} | {f(v['held_bpb'])} | {f(v['held_acc'])} | {v['lam']:g} |" for c, v in r["A"].items()]
        lines += ["", "Bootstrap SDs: " + ", ".join(f"{k} {v:.3f}" for k, v in r["A_bootstrap_sd"].items())]
    if "B" in r:
        B = r["B"]
        best = next(g for g in B["ONLINE"]["grid"] if g["eta"] == B["ONLINE"]["chosen_eta"])
        k = max(int(x.split("pass")[1]) for x in best if x.startswith("held_bpb_pass"))
        lines += ["", "## Block B (FE0 observed KC state)", "", "| Arm | held | acc |", "|---|---:|---:|",
                  f"| R1 | {f(B['R1']['held_bpb'])} | {f(B['R1']['held_acc'])} |",
                  f"| ONLINE eta {best['eta']:g} pass 1 | {f(best['held_bpb_pass1'])} | {f(best['held_acc_pass1'])} |",
                  f"| ONLINE eta {best['eta']:g} pass {k} | {f(best[f'held_bpb_pass{k}'])} | {f(best[f'held_acc_pass{k}'])} |",
                  f"| C16 | {f(B['C16']['held_bpb'])} | {f(B['C16']['held_acc'])} |",
                  f"| C16Q | {f(B['C16Q']['held_bpb'])} | {f(B['C16Q']['held_acc'])} |",
                  f"| NATIVE | {f(B['NATIVE']['held_bpb'])} | {f(B['NATIVE']['held_acc'])} |",
                  f"| R2 | {f(B['R2']['held_bpb'])} | {f(B['R2']['held_acc'])} |", "",
                  "Decomposition: " + ", ".join(f"{k2} {v:.3f}" for k2, v in B["decomposition"].items()),
                  "Bootstrap SDs: " + ", ".join(f"{k2} {v:.3f}" for k2, v in B["bootstrap_sd"].items())]
    sc = score(r)
    if sc:
        lines += ["", "## Predictions and decision (rules of design sections 5-6)"] + [f"* {k}: {v}" for k, v in sc.items()]
    lines += ["", f"CPU {r['cpu_s']} s; learner records {r['learner_records']}; peak memory {r['peak_rss_mb']} MB."]
    (d / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (d / "SUMMARY.json").write_text(json.dumps(sc, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
