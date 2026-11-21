"""The depth reading on the positions where the previous bytes tell nothing about the present one.

In E63's stream the distractor letters between an operator and the second number are drawn independently and uniformly
(`tissue_step1.render`), so at a position where the present byte is a distractor letter and the byte before it is one too,
no earlier byte predicts the present byte. On those positions only, for every distance d, the score S_L(d) is E63's
nearest-centroid classifier (step1_read.classify, balanced accuracy, its folds) of the present letter from the recorded outputs
of up to 128 cells at distance d (outputs below 0.01 set to 0, as in E63's readings). Its level is set by the same classifier
with the labels shuffled: 20 shuffles at every d; the level is the largest null score over all d and shuffles.

Usage: python3 letters_read.py <run name> ...    (prints one line a run; writes nothing)"""
import sys, json
from pathlib import Path
from random import Random
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "TISSUE_STEP1")]
import step1_read as R
import tissue_step1 as T
LET = set(T.DISTRACT)
NSH = 20

def letters(name):
    f = HERE / "TISSUE_EVENT" / "runs" / name
    rec = json.loads((f / "receipt.json").read_text())
    st, _ = T.stream(rec["seed"], rec["stream"]["length"])
    z = np.load(f / "A2.npz"); X, w0, dist = z["x"], int(z["positions"][0]), z["distance"]; wl = X.shape[0]
    pos_all = np.arange(w0, w0 + wl)
    m = np.array([int(st[t]) in LET and int(st[t - 1]) in LET for t in pos_all])
    pos = pos_all[m]; lab = st[pos].astype(np.int64)
    prev = R.classify_exact(R.onehot([int(st[t - 1]) for t in pos]), lab, pos, w0, wl, 0)[0]
    S, null = {}, []
    rng = np.random.default_rng(rec["seed"])
    for d in range(1, 12):
        cells = sorted(int(i) for i in np.nonzero(dist == d)[0])
        if not cells:
            S[d] = None; continue
        samp = Random(rec["seed"] + 900000).sample(cells, 128) if len(cells) > 128 else cells
        F = X[m][:, samp].astype(float); F[F < 0.01] = 0
        S[d] = R.classify(F, lab, pos, w0, wl, 0)[0]
        for _ in range(NSH):
            sh = lab.copy(); rng.shuffle(sh)
            null.append(R.classify(F, sh, pos, w0, wl, 0)[0])
    level = max(null)
    far = max([d for d, s in S.items() if s is not None and s > level], default=0)
    return dict(n=int(m.sum()), prev=prev, S=S, level=level, null_mean=float(np.mean(null)), far=far)

if __name__ == "__main__":
    for n in sys.argv[1:]:
        o = letters(n)
        print(f"| {n} | {o['n']} | {o['prev']:.3f} | {' '.join('–' if s is None else f'{s:.3f}' for s in o['S'].values())} | {o['null_mean']:.3f} / {o['level']:.3f} | {o['far']} |", flush=True)
