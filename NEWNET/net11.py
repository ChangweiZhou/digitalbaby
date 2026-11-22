"""net11: net10 with its context regions wired as a cortical sheet (22 November 2026). Everything is net10 joint/mesh as in
net10 mesh=K mesh_base=0 mesh_dag=1 (same seeds, same lines wiring, same holders); only the graph of sources differs:
  - L regions at random positions in the unit square (sheet_seed), ordered by the first coordinate; region d is the d-th in that
    order, so it reads only regions earlier in the order (no loops);
  - region d reads its K nearest earlier regions (Euclidean distance on the sheet; fewer when fewer exist);
  - with probability p one of those sources (a random one) is replaced by a random earlier region not already a source (long range;
    only when one exists);
  - region 0 has no source and reads the byte before (as net7 mesh_dag).
Usage: Tissue(..., K=2, p=0.0) instead of mesh=; mesh_base=0 and mesh_dag=1 are fixed."""
import numpy as np
from net10 import Tissue as Net10


class Tissue(Net10):
    def __init__(self, K=2, p=0.0, sheet_seed=None, **kw):
        kw.update(mesh=K, mesh_base=0, mesh_dag=1)
        super().__init__(**kw)
        seed = kw.get("seed", 1)
        r = np.random.default_rng((seed if sheet_seed is None else sheet_seed) + 4242)
        L = self.L
        pos = r.random((L, 2)); pos = pos[np.argsort(pos[:, 0])]
        src = []
        for d in range(L):
            if d == 0: src.append(np.zeros(0, int)); continue
            dist = np.hypot(*(pos[:d] - pos[d]).T)
            s = list(np.argsort(dist)[: min(K, d)])
            if r.random() < p:
                others = [j for j in range(d) if j not in s]
                if others: s[int(r.integers(len(s)))] = int(r.choice(others))
            src.append(np.array(sorted(s), int))
        self.pos, self.ctx.msrc = pos, src
