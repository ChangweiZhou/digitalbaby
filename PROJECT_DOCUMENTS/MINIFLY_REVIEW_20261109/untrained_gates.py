"""Gate values of an untrained network of this shape. PyTorch's default distributions drawn with numpy: the same distributions, NOT the
same draws as arch26.new_net(seed) (torch is absent here). Five draws; add-89 panel of seed 73129 at 12 letters."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_gru import forward, panel, split, DEV, BOS, roles_of
def init(rng):
    u = lambda shape, fan: rng.uniform(-fan ** -.5, fan ** -.5, shape).astype(np.float32)
    W = {'book.weight': rng.standard_normal((258, 8)).astype(np.float32), 'l1.weight': u((512, 8), 8), 'l1.bias': u(512, 8),
         'rnn.weight_ih_l0': u((768, 512), 256), 'rnn.weight_hh_l0': u((768, 256), 256), 'rnn.bias_ih_l0': u(768, 256), 'rnn.bias_hh_l0': u(768, 256)}
    W['book.weight'][BOS] = 0
    return W
items = panel(DEV[0], split(DEV[0], .5, 73129)[1], 73129, (12,), 300)[12]
x = np.array([[BOS] + p + t[:-1] for p, t in items], dtype=np.int64)
letter = np.array([[r == 'letter' for r in roles_of(row)] for row in x])
for d in range(5):
    Z = forward(init(np.random.default_rng(d)), x, want_logits=False)[0][:, 1:]
    zl = forward(init(np.random.default_rng(d)), x, want_logits=False)[0][letter]            # (letter positions, units)
    up = (zl > .5).mean(0)                                                                    # per unit: share of letter positions with z above one half
    q = np.quantile(Z, [.001, .01, .5, .99, .999])
    print(f"draw {d}: U = {100 * ((Z >= .1) & (Z <= .9)).mean():6.2f}%   z quantiles 0.1% {q[0]:.3f}  1% {q[1]:.3f}  50% {q[2]:.3f}  99% {q[3]:.3f}  99.9% {q[4]:.3f}"
          f"   units on one side of 0.5 at >= 90% of letter positions: {int(((up >= .9) | (up <= .1)).sum())} of 256")
