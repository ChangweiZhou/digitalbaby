"""V155 local key-learning rules (design sections 4, 12.5 and 12.10), online on cached receptor traces.

Only the PN-to-KC weights on existing synapses change. Notation: c_j birth column sum of KC j, S_j its birth synapses,
q = p[pn] the PN input, q̂^(j) = q restricted to S_j and normalised to sum 1 there (no update when that sum is 0).
  L-COMP  for each active j: B_Sj,j += eta (c_j q̂^(j) - B_Sj,j)                     (column sums stay c_j)
  L-PRED  at position t with next position t+1 in the same line and x_t non-empty:
          positive pair b = x_{t+1}; negative b = code of a uniformly drawn earlier presented position (non-empty code)
          s = mean over b x a of W; gamma = +1 if positive and s < m, -1 if negative and s > -m, else 0
          forward term   on active j of b:   dB_ij = eta c_j gamma (W ã)_j  q̂^(j)_b,i
          retrograde term on active j of x_t: dB_ij = eta c_j gamma (Wᵀ b̃)_j q̂^(j)_t,i
          then clip at 0, rescale every touched column to c_j (a column at 0 is reset to birth, counted);
          W[b, a] = clip(W[b, a] + eta gamma, -1, 1)
All of a position's updates are computed from the keys and W as they were before that position.
"""
import numpy as np
from scipy import sparse


class Keys:
    """Native-wiring key matrix held as CSC arrays; KC codes use the same ranking as codes.topk (and encode)."""

    def __init__(self, birth):
        C = birth["B"].tocsc()
        C.sort_indices()
        self.n_pn, self.n_kc = C.shape
        self.indptr, self.indices = C.indptr.copy(), C.indices.copy()
        self.data = C.data.astype(np.float64).copy()
        self.birth_data = self.data.copy()
        self.seg = np.diff(self.indptr)
        self.c = np.bincount(np.repeat(np.arange(self.n_kc), self.seg), weights=self.data, minlength=self.n_kc)
        self.BT = sparse.csr_matrix((self.n_kc, self.n_pn))
        self.BT.data, self.BT.indices, self.BT.indptr = self.data, self.indices, self.indptr     # shares self.data
        self.pn, self.side, self.k = birth["pn"], birth["side"], birth["k"]
        self.ix = [np.flatnonzero(self.side == s) for s in (0, 1)]
        self.resets = 0

    def q(self, p):
        return np.asarray(p, np.float64)[self.pn]

    def code(self, q):
        r = self.BT @ q
        out = []
        for s, ix in enumerate(self.ix):
            rs = r[ix]
            order = np.argsort(-rs, kind="stable")[:self.k[s]]
            out.append(ix[order[rs[order] > 0]])
        return np.sort(np.concatenate(out))

    def slots(self, act):
        """Synapse slots of the units in act, and the position in act that owns each slot."""
        lens = self.seg[act]
        owner = np.repeat(np.arange(len(act)), lens)
        start = np.repeat(self.indptr[act] - (np.cumsum(lens) - lens), lens)
        return np.arange(lens.sum()) + start, owner

    def qhat(self, q, slots, owner, n):
        qs = q[self.indices[slots]]
        sums = np.bincount(owner, weights=qs, minlength=n)
        ok = sums[owner] > 0
        return np.where(ok, qs / np.where(ok, sums[owner], 1.0), 0.0), ok

    def renorm(self, act):
        """Clip at 0 and rescale the columns in act to their birth sums; reset columns that reached 0."""
        slots, owner = self.slots(act)
        d = np.maximum(self.data[slots], 0.0)
        sums = np.bincount(owner, weights=d, minlength=len(act))
        zero = (sums <= 0) & (self.c[act] > 0)
        scale = np.where(sums > 0, self.c[act] / np.where(sums > 0, sums, 1.0), 1.0)
        d = d * scale[owner]
        reset = zero[owner]
        d[reset] = self.birth_data[slots[reset]]
        self.data[slots] = d
        self.resets += int(zero.sum())

    def csr(self):
        return sparse.csc_matrix((self.data.copy(), self.indices, self.indptr), shape=(self.n_pn, self.n_kc)).tocsr()


def comp_step(K, q, eta):
    act = K.code(q)
    if len(act) == 0:
        return act
    slots, owner = K.slots(act)
    qh, ok = K.qhat(q, slots, owner, len(act))
    target = K.c[act][owner] * qh
    K.data[slots[ok]] += eta * (target[ok] - K.data[slots[ok]])
    return act


class Pred:
    def __init__(self, K, eta, margin, seed):
        self.K, self.eta, self.m = K, eta, margin
        self.W = np.zeros((K.n_kc, K.n_kc), np.float32)
        self.rng = np.random.default_rng(seed)
        self.seen = []                      # presented positions with a non-empty code
        self.counts = {"positive_updates": 0, "negative_updates": 0, "replays": 0, "pairs": 0}

    def score(self, b, a):
        Wsub = self.W[np.ix_(b, a)].astype(np.float64)
        return Wsub.mean(), Wsub.mean(1), Wsub.mean(0)          # s, (W ã)_b, (Wᵀ b̃)_a

    def step(self, qa, qb, Q, pos):
        """Position pos (context qa) with next input qb; Q (n x 302) gives replayed inputs by position."""
        K = self.K
        a = K.code(qa)
        if len(a) == 0:
            return
        b = K.code(qb)
        pairs = []
        if len(b):
            s, fwd, back = self.score(b, a)
            if s < self.m:
                pairs.append((b, qb, +1.0, fwd, back))
            self.counts["pairs"] += 1
        if self.seen:
            n = self.seen[self.rng.integers(len(self.seen))]
            qn = Q[n]
            bn = K.code(qn)
            self.counts["replays"] += 1
            if len(bn):
                s, fwd, back = self.score(bn, a)
                if s > -self.m:
                    pairs.append((bn, qn, -1.0, fwd, back))
        touched = []
        for bb, qq, g, fwd, back in pairs:
            for units, qin, pred in ((bb, qq, fwd), (a, qa, back)):        # forward term, retrograde term
                slots, owner = K.slots(units)
                qh, _ = K.qhat(qin, slots, owner, len(units))
                K.data[slots] += self.eta * K.c[units][owner] * g * pred[owner] * qh
                touched.append(units)
            self.counts["positive_updates" if g > 0 else "negative_updates"] += 1
        if touched:
            K.renorm(np.unique(np.concatenate(touched)))
        for bb, _, g, _, _ in pairs:
            ix = np.ix_(bb, a)
            self.W[ix] = np.clip(self.W[ix] + self.eta * g, -1.0, 1.0)
        self.seen.append(pos)


def run(rule, birth, P, lines, eta, passes, settings, checkpoints=(), on_pass=None, start=None):
    """Online learning in record order over cached traces P (n x 88) with record lengths `lines`.
    Returns {pass: key matrix (CSR)} for the requested checkpoints, the final state, and counters.
    start: an earlier state to continue from (K, pred, passes_done)."""
    if start is None:
        K = Keys(birth)
        pred = Pred(K, eta, settings["margin"], settings["neg_seed"]) if rule == "L-PRED" else None
        done = 0
    else:
        K, pred, done = start
    Q = P[:, K.pn].astype(np.float64)
    edges = np.cumsum(np.concatenate([[0], lines]))
    snaps = {}
    for p in range(done + 1, done + passes + 1):
        for r in range(len(lines)):
            lo, hi = edges[r], edges[r + 1]
            for t in range(lo, hi):
                if rule == "L-COMP":
                    comp_step(K, Q[t], eta)
                elif t + 1 < hi:
                    pred.step(Q[t], Q[t + 1], Q, t)
        if p in checkpoints:
            snaps[p] = K.csr()
        if on_pass:
            on_pass(p, K)
    counters = {"column_resets": K.resets, **(pred.counts if pred else {})}
    return snaps, (K, pred, done + passes), counters
