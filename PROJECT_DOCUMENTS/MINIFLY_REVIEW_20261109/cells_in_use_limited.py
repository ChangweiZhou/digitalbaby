"""Cells in use and directions in use in the networks with limited or local connections (E42, E47, E49).  Read-only.

The question.  In the dense network every cell is in use on every item and the first family's items occupy 255 or 256 of the
257 directions of the cells' state (cells_in_use.py, directions_in_use.py; memo section 29).  Is the same true in the networks
whose connections were limited or made local, or do those networks leave cells, or directions of the cells' state, unused
by the first family?

Files (seed 73129, final weights, read with `git show`, one at a time; nothing is trained, torch is not used, no code of
another branch is run):
  E42  origin/claude/gallant-dijkstra-i9zgsr      G, F10, P8 under rules S and B        reader e42_cells_reader.py
  E47  origin/claude/e47-worker-stage-0-3d0ya6    arms one, place under rules S and B   reader e47_local_reader.py
  E49  origin/claude/peaceful-noether-vcvrmf      in16 under rule B                     reader e49_interface_reader.py

Method.  The cells' states after every position come from the reader's own forward.  The readers do not return the gates, so
this script computes the gates of every cell at every position from the reader's states of the position before and the
weights (dense matrices built from the saved masks or source lists; every cell computed at every position and selected
afterwards, which is not the readers' arrangement), and checks at every position of every item that (1 - z) n + z h gives the
reader's next state for the cells updated there, and that the others keep their state.  The largest difference is printed.

Items.
  family     the first family's taught items as directions_in_use.py builds them: the taught half (split 0.5, seed 73129)
             of add, sub, max, min under modulus 89; for each number of letters 0 to 12, 40 pairs of each operation
             (2,080 items, 13 lengths of 160 items)
  unrelated  e43_unrelated_reader.panel(73129, "held"): 300 held-out pairs at 12 fillers, bytes 128 to 225, none of which
             occurs in an item of the first family
An item's input is the start symbol, the prompt and the answer without its last symbol; position 0 is the start symbol.

Terms (for a cell at one position of one item, with update gate z, candidate n, previous state h, new state (1-z) n + z h).
  change factor   (1-z)(1-n^2) + z(1-z)|h - n|, as cells_in_use.py defines it: every change made at that position to the
                  cell's incoming connections is a product containing it.  A cell that is updated only every p-th position
                  has it computed at the positions at which the cell is updated, and nowhere else.
  state size      |new state|; every change to a connection leaving the cell is a product containing the cell's state.
  mean over a set of items: over all pairs (item, position) with position >= 1 (for the change factor: at which the cell
                  is updated).
  variation with the item   the standard deviation of the cell's state across the items of the set at one position
                  (positions counted from the start symbol; at a position, the items long enough to have it), averaged
                  over the positions 1 to the last.  "within lengths": the same computed inside each group of items of one
                  length and averaged over groups and positions.  "largest": the largest over positions in place of the mean.
  directions      the eigenvectors of the second-moment matrix of the vectors (state of every cell before a position,
                  1), over all pairs (item, position) of the set: these are the vectors that the items present to the
                  connections between cells, as directions_in_use.py takes them for its "recurrent" row.  The count at a
                  share s is the least number of leading eigenvectors that hold the share s of the trace; "above 1e-6" is
                  the number of eigenvalues above 1e-6 of the largest.  Both are relative to the size of the whole vector,
                  which grows with the number of cells; "root-mean-square size above 0.01 (0.05)" is the number of
                  eigenvalues above 1e-4 (2.5e-3), an absolute mark on the same scale as the marks for single cells.
                  "after": the same with the states after each position in place of the states before it.
  restricted to the cells a cell reads   the same counts for the principal submatrix of the cell's own sources, itself
                  and the constant (12 directions where a cell reads 10 cells).

Parts.
  1  per cell: change factor and state size, on the family and on the unrelated items; cells below 0.01 and 0.05
  2  per cell: variation with the item; cells below 0.01 and 0.05
  3  directions the family's items occupy, whole network and restricted to the cells each cell reads
  4  overlap: correlation over cells of the change factor on the two sets; share of the squared size of the unrelated
     items' vectors that lies inside the family's directions (99.99% level)
  5  E47: the cells that do not vary with the item (variation on the family below 0.01, below 0.05), by ring (0 the
     interface, then the far cells 1 to 3, 4 to 6, 7 or 8 grid steps from it, as e47_trial_verify.py); whether unrelated
     items change them (their variation on the unrelated items, and the difference between their mean state at a
     position on the unrelated items and on the family, averaged over positions 1 to 18); of their 10 sources, how many
     are cells that do vary, and how many cells that do vary read them; how many are "silent" (state size on the family
     below 0.05) and how many "closed" (change factor below 0.05 on both sets)
  6  E49 in16, the input side: of the 16 input values a cell reads (and the constant), the directions the family's bytes
     occupy, and the share of the unrelated bytes' codes inside them

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/cells_in_use_limited.py [names]
names restrict the run: G_S G_B F10_S F10_B P8_S P8_B one_S one_B place_S place_B in16_B (default: all eleven).
"""
import sys
sys.dont_write_bytecode = True                      # nothing is written beside this script
import json, os, subprocess, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_pt_tensors import tensors
from np_gru import split, apply, SYM, DEV, BOS
import e42_cells_reader as R42
import e47_local_reader as R47
import e49_interface_reader as R49
import e43_unrelated_reader as U43

SEED = 73129
P = "MOD97_LOCAL_RULE_20261106/exp45/"
REF42, REF47, REF49 = "origin/claude/gallant-dijkstra-i9zgsr", "origin/claude/e47-worker-stage-0-3d0ya6", "origin/claude/peaceful-noether-vcvrmf"
NETS = [("G", "S"), ("G", "B"), ("F10", "S"), ("F10", "B"), ("P8", "S"), ("P8", "B"),
        ("one", "S"), ("one", "B"), ("place", "S"), ("place", "B"), ("in16", "B")]
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
SHARES = (0.99, 0.999, 0.9999)
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
only = set(sys.argv[1:])
assert only <= {f"{a}_{b}" for a, b in NETS}, sys.argv[1:]
T0 = time.time()
sigmoid = R42.sigmoid


def show(path, ref):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True, cwd=HERE).stdout


def load(path, ref):
    return {name.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for name, (stype, shape, raw) in tensors(show(path, ref)).items()}


# ---------------------------------------------------------------- the items
def render(task, a, b, k, rng):                     # as directions_in_use.py renders an item
    op, m, code = task
    d = [rng.choice(LETTERS) for _ in range(k)]
    return (list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [256])


rng = Random(f"directions|{SEED}")                  # the same generator, in the same order, as directions_in_use.py
TAUGHT = {k: [] for k in range(13)}
for t in DEV:
    taught, _ = split(t, .5, SEED)
    for k in range(13):
        TAUGHT[k] += [render(t, a, b, k, rng) for a, b in rng.sample(taught, 40)]
FAMILY = [TAUGHT[k] for k in range(13)]             # 13 groups of 160 items, one length each
UNRELATED = [U43.panel(SEED, "held")]               # one group of 300 items
fam_bytes = sorted({v for g in FAMILY for p, a in g for v in [BOS] + p + a[:-1]})
unr_bytes = sorted({v for g in UNRELATED for p, a in g for v in [BOS] + p + a[:-1]})


def arrays(items):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64)
    return x, np.array([a for _, a in items], dtype=np.int64)


# ---------------------------------------------------------------- the networks
class Saved:
    """The structure an E42 weights file holds, in the form e42_cells_reader.forward takes."""

    def __init__(self, T, variant):
        self.p, self.f, self.mask = T["periods"].astype(int), T["phases"].astype(int), T["mask"].astype(bool)
        self.K, self.variant = int(self.p.max()), variant
        self.a, self.Wx = T.get("a"), T.get("Wx")

    def mask3(self):
        return np.tile(self.mask, (3, 1))


def step_weights(a, p):
    """(K, cells): the weight cell i gives to step t - k of its period; 1 at k = 0 for a cell of period 1."""
    K = int(p.max())
    A = np.zeros((K, len(p)), np.float32)
    A[0, p == 1] = 1.0
    if K > 1:
        inside = (np.arange(K)[:, None] < p[None, :]) & (p[None, :] > 1)
        A[inside] = a[:K][inside]
    return A


class Net256:
    """E42's and E49's networks: 256 cells, every cell receives the input (through mask_in in E49) and is read by the readout."""

    def __init__(self, cfg, rule):
        self.name, self.rule, self.C = cfg, rule, 256
        if cfg == "in16":
            T = load(f"{P}e49/e49_in16_{rule}_{SEED}.pt", REF49)
            self.F = R49.Interface(SEED, *R49.SETTINGS["in16"])
            cells = R42.Cells(SEED, N=10, pset="1")
            assert np.array_equal(T["mask_in"].astype(bool), self.F.mask_in) and np.array_equal(T["mask_out"].astype(bool), self.F.mask_out)
            assert np.array_equal(T["mask"].astype(bool), cells.mask) and (T["periods"] == 1).all()
            self.mask, self.mask_in = cells.mask, self.F.mask_in
            self.p, self.f, a = np.ones(256, int), np.zeros(256, int), None
            self.receipt = json.loads(show(f"{P}e49/e49_in16_{rule}_{SEED}.json", REF49))
        else:
            T = load(f"{P}e42/e42_{cfg}_{rule}_{SEED}.pt", REF42)
            self.receipt = json.loads(show(f"{P}e42/e42_{cfg}_{rule}_{SEED}.json", REF42))
            self.S = Saved(T, self.receipt["variant"])
            assert self.S.variant == "step" and self.S.Wx is None
            self.mask, self.mask_in = self.S.mask, np.ones((256, 512), bool)
            self.p, self.f, a = self.S.p, self.S.f, self.S.a
        self.W = T
        self.dense = bool(self.mask.all())
        self.sources = None if self.dense else [np.nonzero(self.mask[i])[0] for i in range(256)]
        assert self.dense or {len(s) for s in self.sources} == {11}
        self.A = step_weights(a, self.p)
        z1 = np.tanh(T["book.weight"] @ T["l1.weight"].T + T["l1.bias"])                       # (258, 512): the code of each byte
        self.z1 = z1
        self.gi = z1 @ (T["rnn.weight_ih_l0"] * np.tile(self.mask_in, (3, 1))).T + T["rnn.bias_ih_l0"]    # (258, 768)
        self.WhT = np.ascontiguousarray((T["rnn.weight_hh_l0"] * np.tile(self.mask, (3, 1))).T)
        self.bh = T["rnn.bias_hh_l0"]
        self.groups = {"all cells": np.ones(256, bool)}
        if len(set(self.p.tolist())) > 1:
            self.groups.update({f"period {q}": self.p == q for q in sorted(set(self.p.tolist()))})

    def reader(self, x):
        """The reader's own forward: every cell's state after every position, and the logits."""
        if self.name == "in16":
            return R49.forward(self.W, self.mask, self.F, x)
        return R42.forward(self.W, self.S, x, form="afactor")

    def graw(self, xb):
        return self.gi[xb].reshape(len(xb), 3, 256)

    def hraw(self, h):
        return (h @ self.WhT + self.bh).reshape(len(h), 3, 256)


class Net47:
    """E47's network: 1,024 cells on a 32 x 32 grid, 256 interface cells that receive the input and are read."""

    def __init__(self, arm, rule):
        self.name, self.rule, self.C = arm, rule, 1024
        T = load(f"{P}e47/e47_32_{arm}_{rule}_{SEED}.pt", REF47)
        self.receipt = json.loads(show(f"{P}e47/e47_32_{arm}_{rule}_{SEED}.json", REF47))
        S = R47.Grid(SEED, 32, arm)
        assert (T["sources"] == S.idx).all() and (T["periods"] == S.p).all() and (T["phases"] == S.f).all()
        assert (T["interface"] == S.ifx).all() and (T["far"] == S.farx).all()
        self.S, self.W, self.p, self.f = S, T, S.p, S.f
        self.A = step_weights(T.get("a"), S.p)
        self.sources = [S.idx[i] for i in range(S.C)]
        self.dense = False
        z1 = np.tanh(T["book.weight"] @ T["l1.weight"].T + T["l1.bias"])
        self.z1 = z1
        self.gi = z1 @ T["rnn.weight_ih_l0"].T + T["rnn.bias_ih_l0"]                           # (258, 768): gate g, interface cell j at g * 256 + j
        self.WdT = []
        for g in range(3):
            Wd = np.zeros((S.C, S.C), np.float32)
            np.add.at(Wd, (np.repeat(np.arange(S.C), 11), S.idx.ravel()), T["Wc"][g].ravel())
            self.WdT.append(np.ascontiguousarray(Wd.T))
        self.ring = -(-S.dist // 3)                                                             # 0 interface, then 1, 2, 3 (as e47_trial_verify.py)
        self.groups = {"all cells": np.ones(S.C, bool), "interface": self.ring == 0, "ring 1": self.ring == 1, "ring 2": self.ring == 2, "ring 3": self.ring == 3}

    def reader(self, x):
        _, logits, Hall = R47.forward(self.W, self.S, x, keep_all=True)
        return Hall, logits

    def graw(self, xb):
        G = np.empty((len(xb), 3, self.C), np.float32)
        G[:, :, self.S.ifx] = self.gi[xb].reshape(len(xb), 3, 256)
        G[:, :, self.S.farx] = self.W["bfar"][None]
        return G

    def hraw(self, h):
        return np.stack([h @ self.WdT[g] + self.W["bc"][g] for g in range(3)], 1)


def gates(net, x, Hs):
    """z and n of every cell at every position (B, L, C), which cells are updated where (L, C), and the largest difference
    between (1-z) n + z h (updated cells; h for the others) and the reader's next state."""
    B, L = x.shape
    C, A = net.C, net.A
    Z = np.empty((B, L, C), np.float32); N = np.empty_like(Z); upd = np.zeros((L, C), bool)
    Gs, Hh, worst = [], [], 0.0
    h = np.zeros((B, C), np.float32)
    for t in range(L):
        Gs.append(net.graw(x[:, t])); Hh.append(net.hraw(h))
        G, H = A[0] * Gs[t], A[0] * Hh[t]
        for k in range(1, min(len(A), t + 1)):
            G = G + A[k] * Gs[t - k]; H = H + A[k] * Hh[t - k]
        r = sigmoid(G[:, 0] + H[:, 0]); z = sigmoid(G[:, 1] + H[:, 1]); n = np.tanh(G[:, 2] + r * H[:, 2])
        u = (t >= net.f) & ((t - net.f) % net.p == 0)
        worst = max(worst, float(np.abs(np.where(u, (1 - z) * n + z * h, h) - Hs[:, t]).max()))
        Z[:, t], N[:, t], upd[t] = z, n, u
        h = Hs[:, t]
    return Z, N, upd, worst


# ---------------------------------------------------------------- the measures on one set of items
def measure(net, groups):
    C = net.C
    Lmax = max(len(g[0][0]) + len(g[0][1]) for g in groups)
    cf_sum, cf_cnt, st_sum, st_cnt = np.zeros(C), np.zeros(C), np.zeros(C), 0
    s1, s2, cnt = np.zeros((Lmax, C)), np.zeros((Lmax, C)), np.zeros(Lmax)
    within, n_within = np.zeros(C), 0
    Mb, Ma, nvec, Mi = np.zeros((C + 1, C + 1)), np.zeros((C + 1, C + 1)), 0, np.zeros((513, 513))
    right, n_items, worst = 0, 0, 0.0
    for items in groups:
        x, tgt = arrays(items)
        B, L = x.shape
        Hs, logits = net.reader(x)
        right += int((logits[:, -tgt.shape[1]:].argmax(-1) == tgt).all(1).sum()); n_items += B
        Z, N, upd, w = gates(net, x, Hs)
        worst = max(worst, w)
        hprev = np.concatenate([np.zeros((B, 1, C), np.float32), Hs[:, :-1]], 1)
        f = (1 - Z) * (1 - N * N) + Z * (1 - Z) * np.abs(hprev - N)
        m = upd[1:]
        cf_sum += (f[:, 1:] * m[None]).sum((0, 1), dtype=np.float64); cf_cnt += B * m.sum(0)
        st_sum += np.abs(Hs[:, 1:]).sum((0, 1), dtype=np.float64); st_cnt += B * (L - 1)
        H64 = Hs.astype(np.float64)
        s1[:L] += H64.sum(0); s2[:L] += (H64 ** 2).sum(0); cnt[:L] += B
        within += H64[:, 1:].std(0).sum(0); n_within += L - 1
        one = np.ones((B * L, 1))
        Xb = np.concatenate([hprev.reshape(B * L, C).astype(np.float64), one], 1); Mb += Xb.T @ Xb
        Xa = np.concatenate([H64.reshape(B * L, C), one], 1); Ma += Xa.T @ Xa
        Xi = np.concatenate([net.z1[x.ravel()].astype(np.float64), one], 1); Mi += Xi.T @ Xi
        nvec += B * L
        del Hs, Z, N, f, H64, Xb, Xa, Xi, hprev
    mean = s1 / cnt[:, None]
    sd = np.sqrt(np.clip(s2 / cnt[:, None] - mean ** 2, 0, None))
    assert (cf_cnt > 0).all()
    return dict(cf=cf_sum / cf_cnt, st=st_sum / st_cnt, sd=sd[1:].mean(0), sd_max=sd[1:].max(0), sd_within=within / n_within, mean=mean,
                Mb=Mb / nvec, Ma=Ma / nvec, Mi=Mi / nvec, nvec=nvec, right=right, n=n_items, worst=worst, updates=cf_cnt / n_items)


def counts(M):
    """Leading eigenvectors holding 99%, 99.9%, 99.99% of the trace; eigenvalues above 1e-6 of the largest; eigenvalues above
    1e-4 and above 2.5e-3 (a root-mean-square size of the vectors along the direction of 0.01 and of 0.05); eigenvalues, vectors."""
    w, V = np.linalg.eigh(M)
    w, V = w[::-1].clip(min=0), V[:, ::-1]
    c = np.cumsum(w) / w.sum()
    return [int(np.searchsorted(c, s)) + 1 for s in SHARES] + [int((w > 1e-6 * w[0]).sum()), int((w > 1e-4).sum()), int((w > 2.5e-3).sum())], w, V


def sub(M, cells):
    ids = np.concatenate([np.asarray(cells), [M.shape[0] - 1]])
    return M[np.ix_(ids, ids)]


def q3(v):
    return "/".join(f"{x:.3f}" for x in np.quantile(v, [.05, .5, .95]))


def below(v, sel=None):
    v = v if sel is None else v[sel]
    return f"{int((v < 0.01).sum())}, {int((v < 0.05).sum())}"


# ---------------------------------------------------------------- the run
print(f"items: family {sum(len(g) for g in FAMILY)} in {len(FAMILY)} lengths ({len(FAMILY[0][0][0]) + 3} to {len(FAMILY[-1][0][0]) + 3} positions), {len(fam_bytes)} different input symbols;"
      f" unrelated {len(UNRELATED[0])} items of {len(UNRELATED[0][0][0]) + 3} positions, {len(unr_bytes)} different input symbols;"
      f" symbols in both sets: {sorted(set(fam_bytes) & set(unr_bytes))} (257 is the start symbol)")
ROWS = []
for cfg, rule in NETS:
    if only and f"{cfg}_{rule}" not in only:
        continue
    t0 = time.time()
    net = Net47(cfg, rule) if cfg in ("one", "place") else Net256(cfg, rule)
    C = net.C
    fam, unr = measure(net, FAMILY), measure(net, UNRELATED)
    last = net.receipt["curve"][-1]
    print(f"\n===== {cfg} {rule}: {C} cells; receipt's last row (update {last['step']}): taught {last['taught']:.1f}, A {last['A']:.1f}, R {last['R']:.1f}")
    print(f"   the reader on these items: whole answers right, family {fam['right']} of {fam['n']} ({100 * fam['right'] / fam['n']:.1f}%), unrelated {unr['right']} of {unr['n']};"
          f" vectors (item, position): family {fam['nvec']:,}, unrelated {unr['nvec']:,}")
    print(f"   check of the gates computed here against the reader's next state, largest difference: family {fam['worst']:.1e}, unrelated {unr['worst']:.1e}")
    assert max(fam["worst"], unr["worst"]) < 1e-4
    if len(set(net.p.tolist())) > 1:
        print("   cells by period: " + ", ".join(f"{int(q)}: {int((net.p == q).sum())}" for q in sorted(set(net.p.tolist())))
              + "; updates of a cell per item at positions >= 1 (family; mean over the cells of a period): "
              + ", ".join(f"{int(q)}: {fam['updates'][net.p == q].mean():.2f}" for q in sorted(set(net.p.tolist()))))
    # ---- 1 and 2
    print("   1. change factor and state size per cell (5%/50%/95% of cells; cells below 0.01, below 0.05)")
    for gname, sel in net.groups.items():
        for sname, r in (("family", fam), ("unrelated", unr)):
            print(f"      {gname:10s} ({int(sel.sum()):4d}) {sname:9s} change factor {q3(r['cf'][sel])}; below: {below(r['cf'], sel)} | state size {q3(r['st'][sel])}; below: {below(r['st'], sel)}")
    print("   2. variation of a cell's state with the item (5%/50%/95% of cells; cells below 0.01, below 0.05)")
    for gname, sel in net.groups.items():
        for sname, r in (("family", fam), ("unrelated", unr)):
            print(f"      {gname:10s} ({int(sel.sum()):4d}) {sname:9s} {q3(r['sd'][sel])}; below: {below(r['sd'], sel)} | within lengths: below {below(r['sd_within'], sel)}"
                  f" | largest over positions: below {below(r['sd_max'], sel)}")
    # ---- 3
    kf, wf, Vf = counts(fam["Mb"])
    ka = counts(fam["Ma"])[0]
    ku = counts(unr["Mb"])[0]
    print(f"   3. directions the family's items occupy, of {C + 1}: 99% {kf[0]}, 99.9% {kf[1]}, 99.99% {kf[2]}, eigenvalues above 1e-6 of the largest {kf[3]};"
          f" root-mean-square size along the direction above 0.01: {kf[4]}, above 0.05: {kf[5]}; largest eigenvalue {wf[0]:.1f}, trace {wf.sum():.1f}")
    print(f"      the same six counts with the states after each position: {', '.join(map(str, ka))}; for the unrelated items' own vectors: {', '.join(map(str, ku))}")
    bygroup = {}
    for gname, sel in list(net.groups.items())[1:] + ([("rings 2 and 3", net.ring >= 2), ("far cells", net.ring >= 1)] if isinstance(net, Net47) else []):
        kg = counts(sub(fam["Mb"], np.nonzero(sel)[0]))[0]
        bygroup[gname] = (int(sel.sum()) + 1, kg)
        print(f"      restricted to {gname} and the constant, of {int(sel.sum()) + 1}: 99% {kg[0]}, 99.9% {kg[1]}, 99.99% {kg[2]}, above 1e-6 {kg[3]}; root-mean-square size above 0.01: {kg[4]}, above 0.05: {kg[5]}")
    percell = None
    if not net.dense:
        percell = np.array([counts(sub(fam["Mb"], net.sources[i]))[0] for i in range(C)])            # (C, 4)
        for gname, sel in net.groups.items():
            pc = percell[sel]
            print(f"      restricted to the cells a cell reads (of 12), {gname} ({int(sel.sum())}): least/median/most over cells: 99% {pc[:, 0].min()}/{int(np.median(pc[:, 0]))}/{pc[:, 0].max()},"
                  f" 99.9% {pc[:, 1].min()}/{int(np.median(pc[:, 1]))}/{pc[:, 1].max()}, 99.99% {pc[:, 2].min()}/{int(np.median(pc[:, 2]))}/{pc[:, 2].max()},"
                  f" above 1e-6 {pc[:, 3].min()}/{int(np.median(pc[:, 3]))}/{pc[:, 3].max()}; cells with all 12 at 99.99%: {int((pc[:, 2] == 12).sum())}, above 1e-6: {int((pc[:, 3] == 12).sum())};"
                  f" cells with 6 or fewer at 99.99%: {int((pc[:, 2] <= 6).sum())}, with 2 or fewer: {int((pc[:, 2] <= 2).sum())}")
    # ---- 4
    V = Vf[:, :kf[2]]
    inside = float(np.trace(V.T @ unr["Mb"] @ V) / np.trace(unr["Mb"]))
    V3 = Vf[:, :kf[1]]
    inside3 = float(np.trace(V3.T @ unr["Mb"] @ V3) / np.trace(unr["Mb"]))
    corr = {g: float(np.corrcoef(fam["cf"][sel], unr["cf"][sel])[0, 1]) for g, sel in net.groups.items()}
    corr_st = float(np.corrcoef(fam["st"], unr["st"])[0, 1])
    print(f"   4. correlation over cells of the change factor on the family and on the unrelated items: " + ", ".join(f"{g} {v:+.2f}" for g, v in corr.items())
          + f"; of the state size (all cells) {corr_st:+.2f}")
    print(f"      share of the squared size of the unrelated items' vectors inside the family's directions: 99.99% level ({kf[2]} directions) {inside:.4f},"
          f" 99.9% level ({kf[1]}) {inside3:.4f}; a random direction would give {kf[2] / (C + 1):.3f} and {kf[1] / (C + 1):.3f}")
    row = dict(name=f"{cfg} {rule}", C=C, right=100 * fam["right"] / fam["n"], cf_f=below(fam["cf"]), cf_u=below(unr["cf"]), st_f=below(fam["st"]), st_u=below(unr["st"]),
               sd_f=below(fam["sd"]), sd_u=below(unr["sd"]), k=kf, percell=percell, corr=corr["all cells"], inside=inside)
    # ---- 5
    if isinstance(net, Net47):
        S, ring = net.S, net.ring
        print("   5. the cells whose state does not vary with the item (family), by ring; unrelated items; connections")
        Lc = unr["mean"].shape[0]
        shift = np.abs(unr["mean"][1:Lc] - fam["mean"][1:Lc]).mean(0)                                # per cell: mean over positions 1..18 of |mean on unrelated - mean on family|
        src = S.idx[:, 1:]                                                                           # (C, 10): the ten cells a cell reads beside itself
        wsize = np.abs(net.W["Wc"]).mean((0, 2))                                                     # per cell: mean size of its 33 recurrent weights
        print("      mean size of a recurrent weight, by group: " + ", ".join(f"{g} {float(wsize[sel].mean()):.3f}" for g, sel in list(net.groups.items())[1:]))
        row["five"] = {}
        for thr in (0.01, 0.05):
            idle = fam["sd"] < thr
            used = ~idle
            n_src_used = used[src].sum(1)                                                            # of a cell's ten sources, how many vary with the item
            readers = np.zeros(S.C, int)
            np.add.at(readers, src[used].ravel(), 1)                                                 # how many cells that vary with the item read the cell
            byring = [int((idle & (ring == r)).sum()) for r in range(4)]
            if_reads = (S.dist[src] == 0).any(1)                                                     # the cell reads at least one interface cell
            read_by_if = np.zeros(S.C, bool)
            read_by_if[src[S.ifx].ravel()] = True                                                    # at least one interface cell reads the cell
            silent, closed = idle & (fam["st"] < 0.05), idle & (fam["cf"] < 0.05) & (unr["cf"] < 0.05)
            print(f"      below {thr}: {int(idle.sum())} cells; interface {byring[0]} of {int((ring == 0).sum())}, ring 1 {byring[1]} of {int((ring == 1).sum())},"
                  f" ring 2 {byring[2]} of {int((ring == 2).sum())}, ring 3 {byring[3]} of {int((ring == 3).sum())}")
            if idle.any():
                print(f"         on unrelated items, of these {int(idle.sum())} cells: variation with the item below {thr}: {int((unr['sd'][idle] < thr).sum())} (largest {unr['sd'][idle].max():.4f});"
                      f" mean state at a position differs from the family's by less than {thr} (mean over positions 1 to {Lc - 1}): {int((shift[idle] < thr).sum())} (largest {shift[idle].max():.4f});"
                      f" both: {int(((unr['sd'][idle] < thr) & (shift[idle] < thr)).sum())}")
                print(f"         of their 10 sources, cells that vary with the item: mean {n_src_used[idle].mean():.2f}, none in {int((n_src_used[idle] == 0).sum())} cells, 1 to 3 in {int(((n_src_used[idle] >= 1) & (n_src_used[idle] <= 3)).sum())},"
                      f" 4 or more in {int((n_src_used[idle] >= 4).sum())}; cells that vary with the item and read them: mean {readers[idle].mean():.2f}, none for {int((readers[idle] == 0).sum())} cells;"
                      f" neither a source nor a reader that varies: {int(((n_src_used[idle] == 0) & (readers[idle] == 0)).sum())} cells")
                print(f"         read by at least one interface cell: {int((idle & read_by_if).sum())}; reading at least one interface cell: {int((idle & if_reads).sum())};"
                      f" state size below 0.05 on the family (silent): {int(silent.sum())}; change factor below 0.05 on both sets (closed): {int(closed.sum())}; both: {int((silent & closed).sum())};"
                      f" neither (a state that is not near zero and gates that are not shut): {int((idle & ~silent & ~closed).sum())}")
                for r in (1, 2, 3):
                    g = idle & (ring == r)
                    if g.any():
                        print(f"            ring {r} ({int(g.sum())}): sources that vary: mean {n_src_used[g].mean():.2f}, none in {int((n_src_used[g] == 0).sum())}; readers that vary: mean {readers[g].mean():.2f}, none for {int((readers[g] == 0).sum())};"
                              f" neither: {int(((n_src_used[g] == 0) & (readers[g] == 0)).sum())}; change factor on the family, median {np.median(fam['cf'][g]):.3f}; state size, median {np.median(fam['st'][g]):.3f};"
                              f" silent {int((silent & g).sum())}, closed {int((closed & g).sum())}; mean size of a recurrent weight of these cells {float(wsize[g].mean()):.3f}")
            row["five"][thr] = dict(n=int(idle.sum()), byring=byring, unr_sd=int((unr["sd"][idle] < thr).sum()), unr_shift=int((shift[idle] < thr).sum()),
                                    unr_both=int(((unr["sd"][idle] < thr) & (shift[idle] < thr)).sum()),
                                    src_mean=float(n_src_used[idle].mean()) if idle.any() else float("nan"), src_none=int((n_src_used[idle] == 0).sum()),
                                    rd_mean=float(readers[idle].mean()) if idle.any() else float("nan"), rd_none=int((readers[idle] == 0).sum()),
                                    neither=int(((n_src_used[idle] == 0) & (readers[idle] == 0)).sum()),
                                    read_by_if=int((idle & read_by_if).sum()), silent=int(silent.sum()), closed=int(closed.sum()), open_=int((idle & ~silent & ~closed).sum()))
        row["bygroup"] = bygroup
    # ---- 6
    if cfg == "in16":
        print("   6. the input side: the 16 input values a cell reads and the constant (17 directions)")
        ki = counts(fam["Mi"])[0]
        per = np.array([counts(sub(fam["Mi"], net.F.in_idx[i]))[0] for i in range(256)])
        sh = []
        for i in range(256):
            k4, _, Vi = counts(sub(fam["Mi"], net.F.in_idx[i]))
            Mu = sub(unr["Mi"], net.F.in_idx[i]); Vk = Vi[:, :k4[2]]
            sh.append(float(np.trace(Vk.T @ Mu @ Vk) / np.trace(Mu)))
        k4, _, Vi = counts(fam["Mi"])
        whole = float(np.trace(Vi[:, :k4[2]].T @ unr["Mi"] @ Vi[:, :k4[2]]) / np.trace(unr["Mi"]))
        print(f"      the whole code (512 values and the constant, of 513): family's directions at 99% {ki[0]}, 99.9% {ki[1]}, 99.99% {ki[2]}, above 1e-6 {ki[3]};"
              f" share of the unrelated items' input vectors inside (99.99% level): {whole:.4f}")
        print(f"      per cell (of 17), least/median/most over the 256 cells: 99% {per[:, 0].min()}/{int(np.median(per[:, 0]))}/{per[:, 0].max()}, 99.9% {per[:, 1].min()}/{int(np.median(per[:, 1]))}/{per[:, 1].max()},"
              f" 99.99% {per[:, 2].min()}/{int(np.median(per[:, 2]))}/{per[:, 2].max()}, above 1e-6 {per[:, 3].min()}/{int(np.median(per[:, 3]))}/{per[:, 3].max()};"
              f" share of the unrelated items' input vectors inside the family's directions (99.99% level): least {min(sh):.4f}, median {np.median(sh):.4f}, most {max(sh):.4f}")
        row["six"] = dict(whole=ki, per=per, share=(min(sh), float(np.median(sh)), max(sh)), whole_share=whole)
    row["groups"] = {g: dict(n=int(sel.sum()), sd_f=below(fam["sd"], sel), sd_u=below(unr["sd"], sel), st_f=below(fam["st"], sel), cf_f=below(fam["cf"], sel), cf_u=below(unr["cf"], sel),
                             k=bygroup[g][1] if g in bygroup else kf, pc=None if percell is None else percell[sel]) for g, sel in net.groups.items()}
    row["seconds"] = time.time() - t0
    ROWS.append(row)
    print(f"   [{time.time() - t0:.0f} s; {time.time() - T0:.0f} s since the start]", flush=True)
    del net, fam, unr

# ---------------------------------------------------------------- the tables
print("\n===== tables (cells below 0.01, below 0.05)")
print("| Network | Cells | Family right, % | Change factor, family | Change factor, unrelated | State size, family | State size, unrelated | Variation with the item, family | Variation, unrelated |")
print("|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    print(f"| {r['name']} | {r['C']} | {r['right']:.1f} | {r['cf_f']} | {r['cf_u']} | {r['st_f']} | {r['st_u']} | {r['sd_f']} | {r['sd_u']} |")
print("\n| Network | Directions, of | 99% | 99.9% | 99.99% | Above 1e-6 | Not among the 99.99%; not above 1e-6 | Root-mean-square size above 0.01; above 0.05 | Of the 12 of a cell's sources, 99.99%: least / median / most | Cells with all 12 | Correlation of the change factor | Unrelated inside the family's directions (a random direction) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    pc = r["percell"]
    cellpart = "not applicable | not applicable" if pc is None else f"{pc[:, 2].min()} / {int(np.median(pc[:, 2]))} / {pc[:, 2].max()} | {int((pc[:, 2] == 12).sum())}"
    print(f"| {r['name']} | {r['C'] + 1} | {r['k'][0]} | {r['k'][1]} | {r['k'][2]} | {r['k'][3]} | {r['C'] + 1 - r['k'][2]}; {r['C'] + 1 - r['k'][3]} | {r['k'][4]}; {r['k'][5]} | {cellpart} | {r['corr']:+.2f} | {r['inside']:.4f} ({r['k'][2] / (r['C'] + 1):.3f}) |")
print("\n| Network | Group | Cells | Variation with the item, family | Variation, unrelated | State size, family | Change factor, family | Change factor, unrelated | Directions of the group and the constant: 99.99%, above 1e-6, of | Root-mean-square size above 0.01; above 0.05 | Of the 12 of a cell's sources, 99.99%: least / median / most |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    if len(r["groups"]) > 1:
        for g, d in r["groups"].items():
            pc = d["pc"]
            print(f"| {r['name']} | {g} | {d['n']} | {d['sd_f']} | {d['sd_u']} | {d['st_f']} | {d['cf_f']} | {d['cf_u']} | {d['k'][2]}, {d['k'][3]}, {d['n'] + 1} | {d['k'][4]}; {d['k'][5]} | {pc[:, 2].min()} / {int(np.median(pc[:, 2]))} / {pc[:, 2].max()} |")
print("\n| Network | Below | Cells that do not vary | Interface, ring 1, ring 2, ring 3 | Of these, on unrelated items: variation below the same mark; mean state within the mark of the family's | Sources that vary, of 10: mean; cells with none | Cells that vary and read them: mean; cells with none | Neither | Read by an interface cell | Silent; closed; neither |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    for thr, d in r.get("five", {}).items():
        print(f"| {r['name']} | {thr} | {d['n']} | {', '.join(str(v) for v in d['byring'])} | {d['unr_sd']}; {d['unr_shift']} | {d['src_mean']:.2f}; {d['src_none']} | {d['rd_mean']:.2f}; {d['rd_none']} | {d['neither']} | {d['read_by_if']} | {d['silent']}; {d['closed']}; {d['open_']} |")
print(f"done in {time.time() - T0:.0f} s")
