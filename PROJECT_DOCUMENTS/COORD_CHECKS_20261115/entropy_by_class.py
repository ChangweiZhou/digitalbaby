"""Coordinator check for the E34 brief (15 Nov 2026). Read-only. Predictive entropy, cross-entropy and exp(-H/tau) weights by target class on
the saved rows of an E28 birth, first eight training batches of the seed.  python3 -I entropy_by_class.py <seed> <arm e.g. S-all>"""
import sys, json
from pathlib import Path
H = Path(__file__).resolve().parents[2] / "MOD97_LOCAL_RULE_20261106"
sys.path.insert(0, str(H))
import exp33_targets as X33
import e32_read as R32
T = X33.T; E28 = X33.E28; arch26 = X33.arch26
from torch.nn.functional import cross_entropy, log_softmax
arch26.install()
seed = int(sys.argv[1]); arm = sys.argv[2]
bs = X33.first_batches(seed, 8)
rp = H / "exp45" / "e28" / f"e28_dev_{arm}_{seed}.json"
rec, files = R32.row_files(rp)
print("row", "role: H mean | CE mean | n ; tau-weighted shares of sum(w*CE) for tau .5 1 2 inf")
for row in sorted(files):
    f, want = files[row]
    try: m, sha = R32.load_net(f, want, seed)
    except Exception as e: print(row, "skip", str(e)[:60]); continue
    acc = {n: [0., 0., 0, {t: 0. for t in (.5, 1, 2, 1e9)}, {t: 0. for t in (.5, 1, 2)}] for n in X33.FINE}
    with T.no_grad():
        for items in bs:
            x, y, amask, real, fine = X33.masks(items)
            lg = arch26.forward(m, x, True, True)[0]
            lp = log_softmax(lg, -1); Hh = -(lp.exp() * lp).sum(-1)
            ce = T.zeros(y.shape); ce[real] = cross_entropy(lg[real], y[real], reduction="none")
            for i, n in enumerate(X33.FINE):
                s = fine == i
                acc[n][0] += float(Hh[s].sum()); acc[n][1] += float(ce[s].sum()); acc[n][2] += int(s.sum())
                for t in acc[n][3]: acc[n][3][t] += float((T.exp(-Hh[s] / t) * ce[s]).sum())
                for t in acc[n][4]: acc[n][4][t] += float(T.exp(-Hh[s] / t).sum())
    tot = {t: sum(a[3][t] for a in acc.values()) for t in (.5, 1, 2, 1e9)}
    print(row)
    for n, a in acc.items():
        print(f"  {n:13s} H {a[0]/a[2]:.3f} CE {a[1]/a[2]:.3f} n {a[2]:5d} share " + " ".join(f"{100*a[3][t]/tot[t]:5.1f}" for t in tot) + " | mean w " + " ".join(f"{a[4][t]/a[2]:.4f}" for t in a[4]))
    aw = {t: (acc["answer_digit"][4][t] + acc["eos"][4][t]) / (acc["answer_digit"][2] + acc["eos"][2]) for t in (.5, 1, 2)}
    print("  per-position weight, distractor / answer(digits and EOS): " + " ".join(f"tau {t}: {acc['distractor'][4][t]/acc['distractor'][2]/aw[t]:.4f}" for t in aw))
