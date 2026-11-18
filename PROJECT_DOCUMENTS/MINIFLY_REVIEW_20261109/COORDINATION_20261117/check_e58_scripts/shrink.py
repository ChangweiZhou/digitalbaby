"""What `term-shrink` does to the lived network, computed without any training: every tensor outside the byte table multiplied r times by
float32(0.99975). (a) The six earlier panels, whose items never open the gate, so that the term plays no part: these are the figures the arm
itself would give at row r. (b) The reduction score of E48's term of rank 16 as saved at row 1,024 (trained on the unshrunk network), on the
shrunk network: not what the arm gives (there the term is trained while the network shrinks), but how far the shrinking alone moves the
function on the new family's items."""
import sys, json, time
import numpy as np
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, npnet as N
SEEDS = (73129, 93683, 116089); E6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73"); LIFE4 = E6[2:]
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
f = np.float32(0.99975)
res = {}
for seed in SEEDS:
    P = N.earlier_panels(seed)
    red = {f"{t[0]}83": N.reduction_panel(t, .10, seed) for t in N.F4[:2]}
    for rule in "SB":
        sd = ptread.load(f"{B}/w/life_{rule}_{seed}.pt"); term = ptread.load(f"{B}/w/e48_{rule}_{seed}_r16_row1024.pt")
        e40 = {r["updates"]: r for r in json.load(open(f"{B}/rec/e40_lived_{rule}_{seed}.json"))["rows"]}
        cur = {k: v.copy() for k, v in sd.items()}; u = 0; line = []
        for r in ROWS:
            while u < r:
                for k in cur:
                    if k != "book.weight": cur[k] = cur[k] * f          # float32 times float32, rounded to nearest, as the in-place product
                u += 1
            net = N.Net(cur)
            c = {n: net.score(P[n])[0] / len(P[n]) * 100 for n in E6}
            life4 = sum(c[n] for n in LIFE4) / 4; six = sum(c[n] for n in E6) / 6
            sdt = dict(cur); bk = sdt["book.weight"].copy(); bk[[101, 102, 103, 104]] = term["code_rows"]; sdt["book.weight"] = bk
            tn = N.Net(sdt, term); rs = sum(100.0 * tn.score(red[n], True)[0] / len(red[n]) for n in red) / 2
            pct = lambda d: 100.0 * d["exact"] / d["n"]
            fr = (sum(pct(e40[r]["scores"]["earlier_k12"][n]) for n in LIFE4) / 4) if r in e40 else None
            res[(rule, seed, r)] = (life4, six, rs, fr)
            line.append(f"{r}: life4 {life4:.1f} six {six:.1f} E48term {rs:.1f}" + (f" [free {fr:.1f}]" if fr is not None else ""))
        ratio = float(np.linalg.norm(cur['l2.weight'].astype(np.float64)) / np.linalg.norm(sd['l2.weight'].astype(np.float64)))
        print(rule, seed, f"(norm ratio at 1024: {ratio:.4f})", " | ".join(line), flush=True)
print("summary over the six networks: range of the mean of the four life-family panels (life4), of the six panels, and of E48's term's reduction score")
for r in ROWS:
    v = [res[(rule, s, r)] for rule in "SB" for s in SEEDS]
    fr = [x[3] for x in v if x[3] is not None]
    print(r, f"factor {0.99975**r:.3f}", f"life4 {min(x[0] for x in v):.1f}..{max(x[0] for x in v):.1f}", f"six {min(x[1] for x in v):.1f}..{max(x[1] for x in v):.1f}",
          f"E48 term on shrunk net {min(x[2] for x in v):.1f}..{max(x[2] for x in v):.1f}", (f"free(E40) life4 {min(fr):.1f}..{max(fr):.1f}" if fr else ""))
    v0 = [res[(rule, s, 0)] for rule in "SB" for s in SEEDS]
    print("    loss against row 0: life4", " ".join(f"{a[0]-b[0]:.1f}" for a, b in zip(v0, v)), "| E48 term", " ".join(f"{a[2]-b[2]:.1f}" for a, b in zip(v0, v)))
