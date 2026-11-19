# The table of the next-byte reader (reach.py) for T1 to T4: share right on the last quarter (positions 49,152 to 65,535), all positions.
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
R = ('a_near_output', 'b_all_non_feedback', 'onehot_present', 'onehot_present_previous', 'ref_3')
print("| T | seed | cells near the output lines (distances) | near the output lines | all 1,984 cells | one-hot of the present byte | one-hots of the present and previous byte | table of the last 3 bytes |")
print("|---|---|---|---|---|---|---|---|")
for T in ("T1", "T2", "T3", "T4"):
    for s in (73129, 93683, 116089):
        f = os.path.join(HERE, 'reach', f'{T}_s{s}.json')
        if not os.path.exists(f): continue
        r = json.load(open(f)); a = r['sets']['a_near_output']; dd = sorted(int(k) for k in a['distance_from_input_lines'])
        v = [r['readers'][k]['shares']['all']['right'] for k in R]
        print(f"| {T} | {s} | {a['cells']} (d {dd[0]}–{dd[-1]}) | " + " | ".join(f"{x:.4f}" for x in v) + " |")
