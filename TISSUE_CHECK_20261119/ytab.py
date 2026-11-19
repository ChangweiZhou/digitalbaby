"""Table of rule Y beside B and D on the recorded driver inputs (out/own_end_Y_<tissue>_<seed>.json): last quarter, run | clip8,
first window reaching ref_1 and ref_2, largest and median weight-row norm at the end."""
import json, os
for s in (93683, 73129, 116089):
    for k in ("L8", "L128", "A4"):
        f = f"/home/user/tissue_check/out/own_end_Y_{k}_{s}.json"
        if not os.path.exists(f): continue
        d = json.load(open(f)); lq = d["levels_last_quarter"]
        print(f"{s} {k}: ref_1 {100*lq['ref_1']:.2f} ref_2 {100*lq['ref_2']:.2f} end {100*lq['end mark']:.2f}")
        for nm, v in d["rules"].items():
            fw = v["first_window_reaching"]
            print(f"   {nm:34s} {100*v['last_quarter']:5.2f}  ref_1 {fw['ref_1']} ref_2 {fw['ref_2']}  |W| max {v['weight_row_norm']['max']:.3g} median {v['weight_row_norm']['median']:.3g}  last windows {' '.join(f'{100*x:.1f}' for x in v['windows'][-4:])}")
