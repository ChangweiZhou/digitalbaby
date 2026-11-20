#!/usr/bin/env python3
"""Tables for nano/replay PLAN_5.md (the whole system's own answer) from res/d_x_D_*.json, with a check that the runs give PLAN_4's arm D again.
     python3 report5.py > report5.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIVE = (73129, 93683, 116089, 406364, 545353)
def ld(l): return json.load(open(os.path.join(HERE, "res", f"d_{l}.json")))
def row(name, vals, f="{:6.1f}"): print(f"{name:54s} " + " ".join(f.format(v) for v in vals) + "   median " + f.format(float(np.median(vals))))
if __name__ == "__main__":
    R = [ld(f"x_D_{s}") for s in FIVE]; old = [ld(f"w_D_{s}") for s in FIVE]
    same = sum(all(a[k] == b[k] for a, b in zip(o["rows"], n["rows"]) for k in ("add", "sub_clean_wide", "dreams_kept", "replays_wrong")) and o["cells"] == n["cells"] for o, n in zip(old, R))
    print(f"Runs that give every reading of PLAN_4's arm D again: {same} of 5.   Seeds:", *FIVE)
    for ph, t in ((0, "At the end of the addition phase (400,000 real lines)"), (1, "At the end of the life (after 200,000 real lines of subtraction)")):
        print(f"\n================ {t}")
        for key, name in (("add", "never-shown addition, the 500 held pairs"), ("add_wide", "never-shown addition, all 2,000 never-shown pairs"), ("sub_clean_wide", "never-shown subtraction, clean (wide)"), ("sub_mixed", "never-shown subtraction, mixed (the 500; mostly shown lines)")):
            G = [o[f"whole_after_phase_{ph}"][key] for o in R]; print(f"\n{name}; pairs: " + " ".join(str(g["n"]) for g in G))
            row("  the whole system, %", [g["whole"] for g in G]); row("  the relay system alone, %", [g["relay_alone"] for g in G]); row("  the store alone, %", [g["store_alone"] for g in G])
            row("  answers wholly from the store, % of pairs", [100 * g["from_store"] / g["n"] for g in G]); row("  answers wholly from the relay system, % of pairs", [100 * g["from_relay"] / g["n"] for g in G]); row("  answers from both, % of pairs", [100 * g["mixed"] / g["n"] for g in G])
            row("  right, of the answers from the store, %", [100 * g["from_store_right"] / max(1, g["from_store"]) for g in G]); row("  right, of the answers from the relay system, %", [100 * g["from_relay_right"] / max(1, g["from_relay"]) for g in G])
            row("  store alone and relay alone differ: pairs", [g["disagree"] for g in G], "{:6.0f}"); row("    the store right", [g["disagree_store_right"] for g in G], "{:6.0f}"); row("    the relay system right", [g["disagree_relay_right"] for g in G], "{:6.0f}"); row("    neither right", [g["disagree_neither"] for g in G], "{:6.0f}")
    o = ld("x_D_1"); print("\nSeed 1 (development pass; nothing chosen on it), whole / relay alone / store alone: " + "; ".join(f"phase {ph} {k}: {o[f'whole_after_phase_{ph}'][k]['whole']:.1f} / {o[f'whole_after_phase_{ph}'][k]['relay_alone']:.1f} / {o[f'whole_after_phase_{ph}'][k]['store_alone']:.1f}" for ph in (0, 1) for k in ("add", "sub_clean_wide")))
