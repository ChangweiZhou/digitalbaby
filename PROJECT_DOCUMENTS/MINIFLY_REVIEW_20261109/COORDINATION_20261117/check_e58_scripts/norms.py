import sys; sys.path.insert(0, sys.argv[1])
import ptread, numpy as np
W = sys.argv[1] + "/w/"
SEEDS = (73129, 93683, 116089)
names = None
print("ratio of each tensor's norm in E40's free run to its norm at the start; pure decay would give 0.9841 at row 64 and 0.1290 at row 8192")
print("then: relative size of the change, |w_row - w_start| / |w_start|; and the cosine of w_row with w_start")
for rule in "SB":
    for s in SEEDS:
        a = ptread.load(W + f"life_{rule}_{s}.pt"); b = ptread.load(W + f"e40_{rule}_{s}_row64.pt"); c = ptread.load(W + f"e40_{rule}_{s}_row8192.pt")
        out = []
        tot = lambda d: np.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for k, v in d.items() if k != "book.weight"))
        for k in a:
            if k == "book.weight": continue
            n0 = np.linalg.norm(a[k].astype(np.float64))
            r64 = np.linalg.norm(b[k].astype(np.float64)) / n0; r8 = np.linalg.norm(c[k].astype(np.float64)) / n0
            d8 = np.linalg.norm((c[k] - a[k]).astype(np.float64)) / n0
            cos8 = float((c[k].astype(np.float64) * a[k]).sum() / (np.linalg.norm(c[k].astype(np.float64)) * n0))
            out.append(f"{k.replace('rnn.','').replace('_l0','')}: {r64:.3f}/{r8:.3f} d{d8:.2f} c{cos8:.2f}")
        print(rule, s, f"all non-table: {tot(b)/tot(a):.4f} at 64, {tot(c)/tot(a):.4f} at 8192;", "max|w| start", f"{max(np.abs(v).max() for k, v in a.items() if k != 'book.weight'):.3f}")
        print("    " + "; ".join(out))
print(0.99975 ** 64, 0.99975 ** 256, 0.99975 ** 512, 0.99975 ** 1024, 0.99975 ** 8192)
f = np.float32(0.99975); print("float32(0.99975) =", repr(float(f)), " product of 1024 in float32:", float(np.prod(np.full(1024, f, dtype=np.float64))))
x = np.float32(0.5)
for _ in range(1024): x = np.float32(x * f)
print("a float32 weight of 0.5 after 1,024 multiplications:", float(x) / 0.5)
