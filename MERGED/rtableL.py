# The next-byte reader (reach2.py) for the long-axon tissues: share right on the last quarter, all positions.
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
R = ('a_near_output', 'a8_near_output_by_1_8', 'b_all_non_feedback', 'onehot_present', 'onehot_present_previous', 'ref_3')
print("| tissue | seed | near: drivers + enlarged neighbourhoods (cells) | near: drivers + cells within 1/8 (cells) | all 1,984 cells | one-hot present | one-hots present + previous | 3-byte table |")
print("|---|---|---|---|---|---|---|---|")
for T in ("T5L", "T6L64", "T6L128", "T6L256", "T7L", "T8L"):
    for s in (73129, 93683, 116089):
        f = os.path.join(HERE, 'reach', f'{T}_s{s}.json')
        if not os.path.exists(f): continue
        r = json.load(open(f)); v = {k: r['readers'][k]['shares']['all']['right'] for k in R}
        print(f"| {T} | {s} | {v['a_near_output']:.4f} ({r['sets']['a_near_output']['cells']}) | {v['a8_near_output_by_1_8']:.4f} ({r['sets']['a8_near_output_by_1_8']['cells']}) | "
              f"{v['b_all_non_feedback']:.4f} | {v['onehot_present']:.4f} | {v['onehot_present_previous']:.4f} | {v['ref_3']:.4f} |")
