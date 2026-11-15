"""What structure have the saved GRUs learned for the answer, and does a life keep it?  (15 November 2026)

Read-only: forward passes on saved weights with the numpy reader; nothing trains.  For each file and task, on
every operand pair without distractor letters:
  own    the network's own exact answer (all three target bytes right)
  share  share of the variance of the state at '=' that lies between result values
  f2     share of the result code (the mean state per result value) in its two strongest Fourier frequencies
         over Z_m, with those frequencies
  circ   share of the variance of squared distances between result codes explained by the circular difference
  lin    the same for the plain difference (a ring code has circ ~ lin; an ordered code has lin >> circ)

Run from the repository root:
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/learned_structure.py add89,sub89,add97,sub97 <file> [<file> ...]
  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/learned_structure.py add89,add97 untrained
Files are paths under MOD97_LOCAL_RULE_20261106/exp45/ on the default branch, e.g. e26/e26_dev_S_73129.pt.
"""
import os, sys, contextlib, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, forward, DEV, LIFE, BOS, EOS, P, apply, SYM, exact_teacher_forced

def fourier(H):
    Hc = H - H.mean(0); m = len(H)
    p = (np.abs(np.fft.fft(Hc, axis=0)) ** 2).sum(1)[1:]
    share = (p[: (m - 1) // 2] + p[::-1][: (m - 1) // 2]) / p.sum()       # frequencies 1..(m-1)/2
    order = np.argsort(-share)
    return share, order

def dist_r2(H):
    m = len(H); D = ((H[:, None] - H[None]) ** 2).sum(-1)
    i, j = np.triu_indices(m, 1); d = D[i, j]; diff = j - i
    out = []
    for cls in (np.minimum(diff, m - diff), diff):
        res = sum(((d[cls == c] - d[cls == c].mean()) ** 2).sum() for c in np.unique(cls))
        out.append(1 - res / ((d - d.mean()) ** 2).sum())
    return out

TASKS = {"add89": DEV[0], "sub89": DEV[1], "max89": DEV[2], "add97": LIFE[0], "sub97": LIFE[1], "max97": LIFE[2]}

def run(W, task):
    op, m, code = task
    ab = np.array([(a, b) for a in range(m) for b in range(m)])
    c = np.array([apply(op, m, int(u), int(v)) for u, v in ab])
    x = np.array([[BOS] + list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")] + list(f"{r:02d}".encode()) for (a, b), r in zip(ab, c)], dtype=np.int64)
    tgt = np.array([list(f"{r:02d}".encode()) + [EOS] for r in c], dtype=np.int64)
    full = 'l2.weight' in W                               # the untrained control carries no read-out weights
    Z, R, Hs, logits = forward(W, x, want_logits=full)
    own = exact_teacher_forced(logits, tgt).mean() * 100 if full else float('nan')
    H = Hs[:, -3].astype(np.float64); Hc = H - H.mean(0)
    means = np.stack([H[c == v].mean(0) for v in range(m)]); n = np.bincount(c, minlength=m)
    share = (n[:, None] * (means - H.mean(0)) ** 2).sum() / (Hc ** 2).sum()
    fs, order = fourier(means); ci, li = dist_r2(means)
    return own, share, fs[order[:2]].sum(), sorted(int(k) + 1 for k in order[:2]), ci, li

if __name__ == "__main__":
    want = sys.argv[1].split(",")
    if sys.argv[2] == "untrained":
        with contextlib.redirect_stdout(io.StringIO()):          # untrained_gates prints its own table on import
            import untrained_gates as UG
        nets = [(f"untrained draw {d}", UG.init(np.random.default_rng(d))) for d in range(2)]
    else:
        nets = [(f, load_weights(P + f)[0]) for f in sys.argv[2:]]
    for name, W in nets:
        for t in want:
            own, share, f2, ks, ci, li = run(W, TASKS[t])
            print(f"{name:34s} {t}: own {own:5.1f} | share {share:.3f} | f2 {f2:.2f} {ks} | circ {ci:.2f} lin {li:.2f}", flush=True)
