# Step 1 table from res/x*.json: last quarter share right of each rule against the exact tables, and the first window reaching them.
import os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
print("| inputs | seed | bytes | A eps 0 | A eps 1/16 | B | D eta 1/64 | D eta 1/16 | ref_1 | ref_2 | ref_3 | B: first window reaching ref_1 / ref_2 / ref_3 | D 1/64: first window reaching ref_1 / ref_2 / ref_3 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for nb in (64, 256):
    for i in ('1', '2', '3', '3c'):
        for s in (73129, 93683, 116089):
            f = os.path.join(HERE, 'res', f'x{i}_s{s}_{nb}k.json')
            if not os.path.exists(f): continue
            r = json.load(open(f)); R = r['rules']; q = r['last_quarter_refs']
            k = list(R)
            fw = lambda nm: " / ".join(f"{x:,}" if x else "never" for x in (R[nm]['first_window_reaching'][c] for c in ('ref_1', 'ref_2', 'ref_3')))
            print(f"| {i} | {s} | {r['bytes']:,} | " + " | ".join(f"{100 * R[n]['last_quarter']:.2f}" for n in k) + " | " + " | ".join(f"{100 * q[c]:.2f}" for c in ('ref_1', 'ref_2', 'ref_3')) + f" | {fw(k[2])} | {fw(k[3])} |")
