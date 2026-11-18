"""What an item of another family leaves idle in the networks with a sparse input (E50) and in E57's `in16` networks.

The document is WHAT_AN_ITEM_LEAVES_IDLE_MEASUREMENT_20261118.md (same folder); this script prints every figure of it, each line
with the number of the section of the document to which it belongs.

Read-only.  Nothing is trained, torch is not used, no code of another branch is run, and no file is written: every weight file
and receipt is read with `git show` into memory, one weight file at a time.  Standard library and numpy only.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/what_an_item_leaves_idle.py [names]

    names   any of  C-dense_S C-dense_B C46-one_S C46-one_B C16-in_S C16-in_B in16-row_S in16-none_B   (default: all eight)

Sections of the output:
    3     the files read, and the method checks (two forms of the forward; the local derivatives against finite differences)
    4     the check against the receipts: the counts of the last scored row of each receipt, from this script's forward
    5.1   cells in use
    5.2   connections between cells
    5.3   the input map
    5.4   the readout (`lm` and `l2`)
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, os, subprocess, time
from random import Random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from compare_pt_tensors import tensors                          # an unpickler that refuses every class but those of a tensor
from np_gru import split, panel, render, apply, DEV, LIFE, SYM, BOS, EOS
import e42_cells_reader as R42
import e49_interface_reader as R49
import e50_code_reader as R50

T0 = time.time()
SEED = 73129
U, D = 256, 512
P = "MOD97_LOCAL_RULE_20261106/exp45/"
REF50 = "origin/codex/minifly-consolidated-20261023"
REF57 = "85417d0626ca89a6c5187400003603a7a2448af2"
NETS = {   # name: (ref, weight file, receipt, code)
    "C-dense_S":   (REF50, "e50/e50_C-dense_S_73129.pt", "e50/e50_C-dense_S_73129.json", "fixed"),
    "C-dense_B":   (REF50, "e50/e50_C-dense_B_73129.pt", "e50/e50_C-dense_B_73129.json", "fixed"),
    "C46-one_S":   (REF50, "e50/e50_C46-one_S_73129.pt", "e50/e50_C46-one_S_73129.json", "fixed"),
    "C46-one_B":   (REF50, "e50/e50_C46-one_B_73129.pt", "e50/e50_C46-one_B_73129.json", "fixed"),
    "C16-in_S":    (REF50, "e50/e50_C16-in_S_73129.pt", "e50/e50_C16-in_S_73129.json", "fixed"),
    "C16-in_B":    (REF50, "e50/e50_C16-in_B_73129.pt", "e50/e50_C16-in_B_73129.json", "fixed"),
    "in16-row_S":  (REF57, "e57/e57_in16_row_S_u49152_73129.pt", "e57/e57_in16_row_S_u49152_73129.json", "learned"),
    "in16-none_B": (REF57, "e57/e57_in16_none_B_73129.pt", "e57/e57_in16_none_B_73129.json", "learned"),
}
names = sys.argv[1:] or list(NETS)
assert set(names) <= set(NETS), f"names are {list(NETS)}"
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
MARKS = (0.05, 0.01)                     # the marks of the memo's section 29.2 and of CELLS_IN_USE_IN_LIMITED_NETWORKS_20261117.md
SHARES = (1e-3, 1e-2)                    # a change "larger than a small share of the largest"
KINDS = ("a", "b", "c")
LETTERS = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
USED = R50.USED                          # the 46 symbols of the first family, start symbol last
MAP_C = {b: 128 + k for k, b in enumerate(USED)}                # kind (c): the k-th of the 46 symbols becomes byte 128 + k
LIFE_CODE = {t[2][0]: l[2][0] for t, l in zip(DEV, LIFE)}       # p -> r, q -> t, v -> x, w -> n


def show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{P}{path}"], capture_output=True, check=True, cwd=ROOT).stdout


def load(ref, path):
    blob = show(ref, path)
    T = {k.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(shape).copy() for k, (st, shape, raw) in tensors(blob).items()}
    return T, hashlib.sha256(blob).hexdigest(), len(blob)


def pr(sec, text):
    print(f"[{sec}] {text}", flush=True)


def q(v, digits=0):
    """least / median / most"""
    v = np.asarray(v, float)
    f = (lambda a: f"{a:.{digits}f}")
    return f"{f(v.min())} / {f(np.median(v))} / {f(v.max())}"


# ---------------------------------------------------------------- the forward pass, from the tensors of the weight file alone
class Net:
    def __init__(self, T, code):
        self.T, self.code_kind = T, code
        m3 = np.tile(T["mask"], (3, 1))
        self.mi3 = np.tile(T["mask_in"], (3, 1))
        self.Wi = T["rnn.weight_ih_l0"] * self.mi3
        self.Wh = T["rnn.weight_hh_l0"] * m3
        self.Wo = T["lm.weight"] * T["mask_out"]
        self.src = np.array([np.nonzero(T["mask"][i])[0] for i in range(U)])          # (256, 11): the cells a cell reads, itself included
        self.outside_zero = all(float(np.abs(T[k][~mk]).max(initial=0)) == 0 for k, mk in
                                (("rnn.weight_ih_l0", self.mi3), ("rnn.weight_hh_l0", m3), ("lm.weight", T["mask_out"])))
        if code == "fixed":
            self.table = T["code_table"]
        else:
            self.table = np.tanh(T["book.weight"] @ T["l1.weight"].T + T["l1.bias"]).astype(np.float32)   # (258, 512): every byte's code

    def run(self, x, keep=False):
        T, U_ = self.T, U
        z1 = self.table[x]                                                               # (B, L, 512)
        gi = z1 @ self.Wi.T + T["rnn.bias_ih_l0"]
        B, L = x.shape
        h = np.zeros((B, U_), np.float32)
        Hs = np.empty((B, L, U_), np.float32)
        K = {k: np.empty((B, L, U_), np.float32) for k in ("hprev", "r", "z", "n", "ghn")} if keep else None
        for t in range(L):
            gh = h @ self.Wh.T + T["rnn.bias_hh_l0"]
            r = R42.sigmoid(gi[:, t, :U_] + gh[:, :U_])
            z = R42.sigmoid(gi[:, t, U_:2 * U_] + gh[:, U_:2 * U_])
            n = np.tanh(gi[:, t, 2 * U_:] + r * gh[:, 2 * U_:])
            if keep:
                K["hprev"][:, t], K["r"][:, t], K["z"][:, t], K["n"][:, t], K["ghn"][:, t] = h, r, z, n, gh[:, 2 * U_:]
            h = ((1 - z) * n + z * h).astype(np.float32)
            Hs[:, t] = h
        z2 = np.tanh(z1 @ T["l2.weight"].T + T["l2.bias"] + Hs @ self.Wo.T + T["lm.bias"])
        logits = z2 @ T["l3.weight"].T + T["l3.bias"]
        return (logits, Hs, z1, gi, K) if keep else logits

    def exact(self, items, chunk=200):
        n = 0
        for s in range(0, len(items), chunk):
            it = items[s:s + chunk]
            x = np.array([[BOS] + p + a[:-1] for p, a in it], dtype=np.int64)
            tgt = np.array([a for _, a in it], dtype=np.int64)
            n += int((self.run(x)[:, -tgt.shape[1]:].argmax(-1) == tgt).all(1).sum())
        return n


# ---------------------------------------------------------------- the receipts' panels (as e50_stage2_verify.panels)
def receipt_panels():
    held = {t: split(t, .5, SEED) for t in DEV}
    nm = lambda t: f"{t[0]}{t[1]}"
    AP = {nm(t): panel(t, held[t][1], SEED, ks=(12,), n=300)[12] for t in DEV}
    TA = {nm(t): panel(t, held[t][0], SEED + 1, ks=(12,), n=300)[12] for t in DEV}
    RP = {}
    for t in DEV[:2]:
        red = [(a, b) for a, b in held[t][1] if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        RP[nm(t)] = [render(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]
    LT = ("add", 89, b"p")
    LP = panel(LT, split(LT, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    return AP, RP, TA, LP


# ---------------------------------------------------------------- the three kinds of items
def kinds_of_items():
    """X[kind][k]: (40, 10 + k) inputs (start symbol, prompt, answer without its end symbol); 10 taught pairs of each of the
    four operations at each number k of letters from 0 to 12: 520 items of each kind, matched item by item."""
    rng = Random(f"idle|{SEED}")
    X = {kd: {} for kd in KINDS}
    for k in range(13):
        rows = {kd: [] for kd in KINDS}
        for t in DEV:
            op, _, code = t
            for a, b in rng.sample(split(t, .5, SEED)[0], 10):
                d = [rng.choice(LETTERS) for _ in range(k)]
                body = list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
                xa = [BOS, code[0]] + body + list(f"{apply(op, 89, a, b):02d}".encode())
                xb = [BOS, LIFE_CODE[code[0]]] + body + list(f"{apply(op, 97, a, b):02d}".encode())
                rows["a"].append(xa); rows["b"].append(xb); rows["c"].append([MAP_C[v] for v in xa])
        for kd in KINDS:
            X[kd][k] = np.array(rows[kd], dtype=np.int64)
    return X


# ---------------------------------------------------------------- local derivatives
def factors(K):
    """The size of the derivative of a cell's new state with respect to one weight of its row g, per unit of that weight's
    input: r row (1 - z)(1 - n^2) |gh_n| r (1 - r); z row z (1 - z) |h - n|; n row (1 - z)(1 - n^2), times r for the
    recurrent map.  The memo's change factor is the n-row factor of the input map plus the z-row factor."""
    z, n, r, hp, ghn = K["z"], K["n"], K["r"], K["hprev"], K["ghn"]
    a = (1 - z) * (1 - n * n)
    fr = a * np.abs(ghn) * r * (1 - r)
    fz = z * (1 - z) * np.abs(hp - n)
    return dict(r=fr, z=fz, n_in=a, n_rec=a * r, cf=a + fz)


def fd_check(net, x):
    """The factors against finite differences (float64) of the new state at one position, the earlier state held."""
    T = {k: v.astype(np.float64) if v.dtype == np.float32 else v for k, v in net.T.items()}
    _, _, z1, _, K = net.run(x[:1], keep=True)
    t = 5
    f = factors(K)
    hp = K["hprev"][0, t].astype(np.float64); xin = z1[0, t].astype(np.float64)
    Wi = T["rnn.weight_ih_l0"] * net.mi3; Wh = T["rnn.weight_hh_l0"] * np.tile(T["mask"], (3, 1))

    def new_h(Wi_, Wh_):
        gi = Wi_ @ xin + T["rnn.bias_ih_l0"]; gh = Wh_ @ hp + T["rnn.bias_hh_l0"]
        r = 1 / (1 + np.exp(-(gi[:U] + gh[:U]))); z = 1 / (1 + np.exp(-(gi[U:2 * U] + gh[U:2 * U]))); n = np.tanh(gi[2 * U:] + r * gh[2 * U:])
        return (1 - z) * n + z * hp
    worst, rs, eps = 0.0, np.random.default_rng(7), 1e-6
    for _ in range(40):
        i = int(rs.integers(U)); g = int(rs.integers(3)); rec = bool(rs.integers(2))
        if rec:
            j = int(net.src[i][rs.integers(net.src.shape[1])]); M = Wh; inp = hp[j]
        else:
            vs = np.nonzero(net.T["mask_in"][i] & (np.abs(xin) > 0))[0]
            if not len(vs):
                continue
            j = int(vs[rs.integers(len(vs))]); M = Wi; inp = xin[j]
        Mp, Mm = M.copy(), M.copy(); Mp[g * U + i, j] += eps; Mm[g * U + i, j] -= eps
        num = abs((new_h(Mp, Wh) if not rec else new_h(Wi, Mp))[i] - (new_h(Mm, Wh) if not rec else new_h(Wi, Mm))[i]) / (2 * eps)
        fac = {0: f["r"], 1: f["z"], 2: f["n_rec"] if rec else f["n_in"]}[g][0, t, i] * abs(inp)
        worst = max(worst, abs(num - fac) / max(abs(num), 1e-4))
    return worst


# ---------------------------------------------------------------- directions in use
def directions(Xa):
    X = np.concatenate([Xa.astype(np.float64), np.ones((len(Xa), 1))], 1)
    w, V = np.linalg.eigh(X.T @ X / len(X))
    w, V = w[::-1].clip(min=0), V[:, ::-1]
    c = np.cumsum(w) / w.sum()
    k999, k9999 = int(np.searchsorted(c, 0.999)) + 1, int(np.searchsorted(c, 0.9999)) + 1
    rank = int((w > 1e-10 * w[0]).sum())
    return dict(k999=k999, k9999=k9999, rank=rank, V_rank=V[:, :rank], V_9999=V[:, :k9999], dim=X.shape[1])


def inside(Y, V):
    Y = np.concatenate([Y.astype(np.float64), np.ones((len(Y), 1))], 1)
    return float(np.linalg.norm(Y @ V) ** 2 / np.linalg.norm(Y) ** 2)


# ---------------------------------------------------------------- one network
def measure(name, X, panels):
    ref, wpath, rpath, code = NETS[name]
    t0 = time.time()
    T, sha, size = load(ref, wpath)
    rc = json.loads(show(ref, rpath)); row = rc["curve"][-1]
    net = Net(T, code)
    pr("3", f"{name}: {ref[:8]}:{P}{wpath}, {size:,} bytes, SHA-256 equal to the receipt's: {sha == rc['weights_sha256']}; receipt complete: {rc.get('complete')}; "
       f"kept weights: input map {int(net.mi3.sum()):,}, recurrent {int(np.tile(T['mask'], (3, 1)).sum()):,} ({net.src.shape[1]} sources a cell, itself included), lm {int(T['mask_out'].sum()):,}; "
       f"weights outside the three masks all zero: {net.outside_zero}")
    # 3: the second form of the forward (the reviewer's readers, by index lists and their own draws)
    AP, RP, TA, LP = panels
    it = AP["add89"][:6] + AP["sub89"][:6] + AP["max89"][:6] + AP["min89"][:6]
    x = np.array([[BOS] + p + a[:-1] for p, a in it], dtype=np.int64)
    W = {k: v for k, v in T.items() if v.dtype == np.float32 and v.ndim > 0 and k != "code_table"}
    cells = R42.Cells(SEED, N=10, pset="1").mask
    if code == "fixed":
        F = R50.OneOfEach(SEED) if name.startswith("C46") else R49.Interface(SEED, 16 if name.startswith("C16") else D, U)
        lb = R50.forward(W, cells, F, R50.Code(SEED, amplitude=float(T["code_amplitude"])), x)[1]
    else:
        F = R49.Interface(SEED, 16, U)
        lb = R49.forward(W, cells, F, x)[1]
    la = net.run(x)
    same_masks = bool(np.array_equal(T["mask"], cells) and np.array_equal(T["mask_in"], F.mask_in) and np.array_equal(T["mask_out"], F.mask_out))
    pr("3", f"{name}: the file's three masks equal the reader's draws: {same_masks}; this script's forward against the reader's on 24 items: largest difference of a logit "
       f"{float(np.abs(la - lb).max()):.1e} (largest logit {float(np.abs(la).max()):.1f}), same largest output at every position: {bool((la.argmax(-1) == lb.argmax(-1)).all())}")
    pr("3", f"{name}: local derivatives against finite differences at 40 random kept weights (one item of kind (a), position 5): largest relative difference {fd_check(net, X['a'][4]):.1e}")
    # 4: the receipt
    mine = dict(A={k: net.exact(v) for k, v in AP.items()}, R={k: net.exact(v) for k, v in RP.items()}, taught={k: net.exact(v) for k, v in TA.items()},
                length={str(k): net.exact(v) for k, v in LP.items()})
    theirs = dict(A=row["A_counts"], R=row["R_counts"], taught=row["taught_counts"], length=row["length_counts"])
    ok = all(mine[k] == theirs[k] for k in mine)
    pr("4", f"{name}: receipt row {row['step']:,}: A {list(mine['A'].values())} / {list(theirs['A'].values())}; R {list(mine['R'].values())} / {list(theirs['R'].values())}; "
       f"taught {list(mine['taught'].values())} / {list(theirs['taught'].values())}; length 12, 40 {list(mine['length'].values())} / {list(theirs['length'].values())}; all equal: {ok}")

    # ------------------------------------------------------------ the three kinds
    kept_in = net.mi3
    use = {kd: dict(pos=[], pos01=[], some=[], every=[], some01=[], p1=[], p1_01=[], st_pos=[], cf_sum=np.zeros(U), h_sum=np.zeros(U), n=0,
                    rec_cnt={s: [] for s in SHARES}, rec_ov={s: [] for s in SHARES}, rec_union={s: np.zeros((3, U, net.src.shape[1]), bool) for s in SHARES},
                    in_cnt={s: [] for s in SHARES}, in_ov={s: [] for s in SHARES}, in_union={s: np.zeros((3 * U, D), bool) for s in SHARES},
                    recv_p1=[], recv_any=[], drive_p1=[], lm=[], l2=[], right=0, spread=0.0) for kd in KINDS}
    for k in range(13):
        R = {}
        for kd in KINDS:
            x = X[kd][k]
            logits, Hs, z1, gi, K = net.run(x, keep=True)
            f = factors(K)
            R[kd] = (x, Hs, z1, gi, K, f)
            u = use[kd]
            cf = f["cf"]
            u["pos"].append((cf >= MARKS[0]).sum(-1).ravel()); u["pos01"].append((cf >= MARKS[1]).sum(-1).ravel())
            u["st_pos"].append((np.abs(Hs) >= MARKS[0]).sum(-1).ravel())
            u["some"].append((cf >= MARKS[0]).any(1)); u["every"].append((cf >= MARKS[0]).all(1).sum(-1)); u["some01"].append((cf >= MARKS[1]).any(1))
            u["p1"].append(cf[:, 1] >= MARKS[0]); u["p1_01"].append(cf[:, 1] >= MARKS[1])
            u["cf_sum"] += cf.sum((0, 1)); u["h_sum"] += np.abs(Hs).sum((0, 1)); u["n"] += cf.shape[0] * cf.shape[1]
            # input received: a cell receives input at a position if one of the values it reads is not zero there
            nz = (np.abs(z1) > 0).astype(np.float32)
            recv = (nz @ T["mask_in"].T.astype(np.float32)) > 0                         # (B, L, 256)
            u["recv_p1"].append(recv[:, 1].sum(-1)); u["recv_any"].append(recv.any(1).sum(-1))
            u["drive_p1"].append(np.abs(gi[:, 1] - T["rnn.bias_ih_l0"]).reshape(len(x), 3, U).sum(1).mean(-1))
            u["spread"] = max(u["spread"], float(np.abs(Hs - Hs[:1]).max()))
            u["lm"].append(Hs[:, -3:].reshape(-1, U)); u["l2"].append(z1[:, -3:].reshape(-1, D))
            tgt = x[:, -2:]
            if kd == "a":
                u["right"] += int((logits[:, -3:-1].argmax(-1) == tgt).all(1).sum())
        # connections between cells and input weights, item by item, matched across the kinds
        Srec, Sin = {}, {}
        for kd in KINDS:
            x, Hs, z1, gi, K, f = R[kd]
            hsrc = np.abs(K["hprev"][:, :, net.src])                                     # (B, L, 256, 11)
            Srec[kd] = np.stack([np.einsum("btu,btuk->buk", f[g], hsrc) for g in ("r", "z", "n_rec")], 1)   # (B, 3, 256, 11)
            Sin[kd] = (np.concatenate([f["r"], f["z"], f["n_in"]], -1), np.abs(z1))       # (B, L, 768), (B, L, 512)
        for b in range(X["a"][k].shape[0]):
            ins = {kd: (Sin[kd][0][b].T @ Sin[kd][1][b]) * kept_in for kd in KINDS}
            ra, ia = Srec["a"][b], ins["a"]
            for s in SHARES:
                tr, ti = s * ra.max(), s * ia.max()        # the scale: the largest size in the matched item of kind (a)
                ca, ci = ra > tr, ia > ti
                for kd in KINDS:
                    u = use[kd]
                    cr, cn = Srec[kd][b] > tr, ins[kd] > ti
                    u["rec_cnt"][s].append(int(cr.sum())); u["in_cnt"][s].append(int(cn.sum()))
                    u["rec_ov"][s].append(int((cr & ca).sum())); u["in_ov"][s].append(int((cn & ci).sum()))
                    u["rec_union"][s] |= cr; u["in_union"][s] |= cn
    for kd in KINDS:
        u = use[kd]
        for key in ("pos", "pos01", "st_pos", "every", "recv_p1", "recv_any", "drive_p1"):
            u[key] = np.concatenate(u[key])
        for key in ("some", "some01", "p1", "p1_01"):
            u[key] = np.concatenate(u[key])
        u["lm"], u["l2"] = np.concatenate(u["lm"]), np.concatenate(u["l2"])
    N = len(use["a"]["every"])
    a = use["a"]
    pr("5.1", f"{name}: items of kind (a) with the whole answer right (both digits, teacher forcing): {a['right']} of {N}")
    for kd in KINDS:
        u = use[kd]
        some = u["some"].sum(-1); some01 = u["some01"].sum(-1)
        pr("5.1", f"{name} kind ({kd}): cells in use at a position (change factor >= 0.05), over all (item, position) pairs: least / median / most {q(u['pos'])}; "
           f"at the mark 0.01: {q(u['pos01'])}; cells whose state is 0.05 or more in size: {q(u['st_pos'])}")
        pr("5.1", f"{name} kind ({kd}): over an item, cells in use at some position {q(some)} (at 0.01: {q(some01)}), at every position {q(u['every'])}; "
           f"at the task byte's position (position 1) {q(u['p1'].sum(-1))} (at 0.01: {q(u['p1_01'].sum(-1))})")
        mcf, mh = u["cf_sum"] / u["n"], u["h_sum"] / u["n"]
        pr("5.1", f"{name} kind ({kd}): cells whose mean change factor over all (item, position) pairs is below 0.05, below 0.01: {int((mcf < .05).sum())}, {int((mcf < .01).sum())}; "
           f"whose mean state size is below 0.05, below 0.01: {int((mh < .05).sum())}, {int((mh < .01).sum())}; median over cells of the mean change factor {np.median(mcf):.3f}, of the mean state size {np.median(mh):.3f}")
        use[kd]["mcf"] = mcf
        pr("5.1", f"{name} kind ({kd}): largest difference between the states of two items of the same length at the same position: {u['spread']:.2e}")
    for kd in ("b", "c"):
        u = use[kd]
        setl = int(((a["mcf"] >= .05) & (u["mcf"] < .05)).sum()); setl01 = int(((a["mcf"] >= .01) & (u["mcf"] < .01)).sum())
        item = (a["some"] & ~u["some"]).sum(-1); item01 = (a["some01"] & ~u["some01"]).sum(-1)
        p1 = (a["p1"] & ~u["p1"]).sum(-1); p1_01 = (a["p1_01"] & ~u["p1_01"]).sum(-1)
        rev = (u["p1"] & ~a["p1"]).sum(-1)
        pr("5.1", f"{name} (a) and not ({kd}): cells in use for kind (a) and not for kind ({kd}) by the mean change factor: {setl} (at 0.01: {setl01}); "
           f"item by item, in use at some position of the item of kind (a) and at no position of the matched item: {q(item)} (at 0.01: {q(item01)}); "
           f"at position 1: {q(p1)} (at 0.01: {q(p1_01)}); the converse at position 1 (in use for ({kd}) and not for (a)): {q(rev)}")
    # 5.2 connections between cells
    nrec = int(np.tile(T["mask"], (3, 1)).sum())
    for s in SHARES:
        for kd in KINDS:
            u = use[kd]
            cnt = np.array(u["rec_cnt"][s]); ov = np.array(u["rec_ov"][s])
            line = f"{name} share {s:g}, kind ({kd}): kept recurrent weights ({nrec:,}) changed by an item: {q(cnt)}; by some item of the kind: {int(u['rec_union'][s].sum()):,}"
            if kd != "a":
                frac = ov / np.maximum(cnt, 1)
                uni = u["rec_union"][s]; una = a["rec_union"][s]
                line += (f"; of those an item changes, also changed by the matched item of kind (a): share {q(frac, 4)}; of those some item of the kind changes, "
                         f"also changed by some item of kind (a): {int((uni & una).sum()):,} of {int(uni.sum()):,}; changed by the kind and by no item of kind (a): {int((uni & ~una).sum()):,}")
                sel = uni & ~una
            else:
                sel = ~u["rec_union"][s]
            line += (f"; {'those last' if kd != 'a' else 'changed by no item of the kind'}, by gate row r, z, n: {', '.join(str(int(sel[g].sum())) for g in range(3))}, "
                     f"in {int(sel.any((0, 2)).sum())} receiving cells, from {len(set(net.src[sel.any(0)].tolist()))} source cells")
            pr("5.2", line)
    # 5.3 the input map
    nin = int(kept_in.sum())
    for kd in KINDS:
        u = use[kd]
        pr("5.3", f"{name} kind ({kd}): cells that receive any input at the task byte's position: {q(u['recv_p1'])}; at some position of the item: {q(u['recv_any'])}; "
           f"mean over cells of the input's contribution at position 1 (sum over the three gates of |input-map row times code|, bias excluded): mean over items {u['drive_p1'].mean():.3f}")
    for s in SHARES:
        for kd in KINDS:
            u = use[kd]
            cnt = np.array(u["in_cnt"][s]); ov = np.array(u["in_ov"][s])
            line = f"{name} share {s:g}, kind ({kd}): kept input weights ({nin:,}) changed by an item: {q(cnt)}; by some item of the kind: {int(u['in_union'][s].sum()):,}"
            if kd != "a":
                frac = ov[cnt > 0] / cnt[cnt > 0]
                uni = u["in_union"][s]; una = a["in_union"][s]
                line += (f"; items that change none: {int((cnt == 0).sum())}; of those an item changes, also changed by the matched item of kind (a): share "
                         f"{q(frac, 4) if len(frac) else 'no item changes any'}; of those some item of the kind changes, also changed by some item of kind (a): "
                         f"{int((uni & una).sum()):,} of {int(uni.sum()):,}; changed by the kind and by no item of kind (a): {int((uni & ~una).sum()):,}")
            pr("5.3", line)
    if code == "fixed":
        owner = T["code_owner"]; free = owner < 0
        tb = T["code_table"]
        pr("5.3", f"{name}: values of the code owned by no byte: {int(free.sum())} of {D}; rows of the code table not zero: {int((np.abs(tb).sum(1) > 0).sum())} "
           f"(the 46 symbols: {sorted(int(b) for b in np.nonzero(np.abs(tb).sum(1) > 0)[0]) == sorted(USED)}); cells that read at least one value owned by no byte: "
           f"{int(T['mask_in'][:, free].any(1).sum())}; kept input weights on such values: {3 * int(T['mask_in'][:, free].sum())}; columns of l2 on such values with any weight not zero: "
           f"{int((np.abs(T['l2.weight'][:, free]).sum(0) > 0).sum())}")
    else:
        tb = net.table.astype(np.float64)
        fam = [b for b in USED]
        Xf = np.concatenate([tb[fam], np.ones((len(fam), 1))], 1)
        _, sv, Vt = np.linalg.svd(Xf, full_matrices=False)
        r = int((sv > 1e-6 * sv[0]).sum())
        for grp, bs in (("the task bytes of kind (b) (r, t, x, n)", list(LIFE_CODE.values())), ("the bytes 128 to 173 of kind (c)", list(range(128, 174)))):
            Y = np.concatenate([tb[bs], np.ones((len(bs), 1))], 1)
            sh = np.linalg.norm(Y @ Vt[:r].T, axis=1) ** 2 / np.linalg.norm(Y, axis=1) ** 2
            sh9 = np.linalg.norm(Y @ Vt[:9].T, axis=1) ** 2 / np.linalg.norm(Y, axis=1) ** 2
            pr("5.3", f"{name}: the code of {grp}, constant appended: share of its squared length inside the span of the 46 symbols' codes ({r} directions): least {sh.min():.5f}, mean {sh.mean():.5f}; "
               f"inside their nine leading directions: least {sh9.min():.5f}, mean {sh9.mean():.5f}; values of the code that are exactly zero for some of these bytes: {int((tb[bs] == 0).sum())}")
    # 5.4 the readout
    for mat, key, dim in (("lm", "lm", U), ("l2", "l2", D)):
        dA = directions(a[key])
        line = (f"{name} {mat} (inputs at the answer positions, constant appended, {dA['dim']}): kind (a) uses {dA['k999']} directions at 99.9% of the trace, {dA['k9999']} at 99.99%, "
                f"{dA['rank']} above 1e-10 of the largest eigenvalue")
        for kd in ("b", "c"):
            u = use[kd]
            dK = directions(u[key])
            msq = float((u[key].astype(np.float64) ** 2).sum(1).mean() / max((a[key].astype(np.float64) ** 2).sum(1).mean(), 1e-30))
            line += (f"; kind ({kd}): own directions {dK['k9999']} at 99.99%, {dK['rank']} above 1e-10; share of its squared length inside the span of kind (a)'s {dA['rank']} directions "
                     f"{inside(u[key], dA['V_rank']):.5f}, inside the {dA['k9999']} at 99.99% {inside(u[key], dA['V_9999']):.5f} (a random direction: {dA['rank'] / dA['dim']:.3f}, {dA['k9999'] / dA['dim']:.3f}); "
                     f"mean squared length without the constant, against kind (a): {msq:.4f}")
        pr("5.4", line)
    pr("t", f"{name}: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    pr("3", f"the worker's branch origin/claude/e50-fixed-input-code-017dSE is at {subprocess.run(['git', 'rev-parse', 'origin/claude/e50-fixed-input-code-017dSE'], capture_output=True, text=True, cwd=ROOT).stdout.strip()[:8]}; "
       f"E57's files are read at {REF57[:8]}; E50's on {REF50}")
    X = kinds_of_items()
    lens = sorted({X['a'][k].shape[1] for k in X['a']})
    pr("3", f"items: {sum(len(v) for v in X['a'].values())} of each kind, {len(lens)} lengths of {lens[0]} to {lens[-1]} positions; "
       f"bytes of kind (a) all among the 46 symbols: {set(np.concatenate([v.ravel() for v in X['a'].values()]).tolist()) <= set(USED)}; "
       f"bytes of kind (b) not among them: {sorted(set(np.concatenate([v.ravel() for v in X['b'].values()]).tolist()) - set(USED))}; "
       f"kind (b) differs from kind (a) at position 1 in every item: {all(bool((X['a'][k][:, 1] != X['b'][k][:, 1]).all()) for k in X['a'])}, and at an answer position in "
       f"{sum(int((X['a'][k][:, 2:] != X['b'][k][:, 2:]).any(1).sum()) for k in X['a'])} items; bytes of kind (c): {min(int(v.min()) for v in X['c'].values())} to {max(int(v.max()) for v in X['c'].values())}")
    panels = receipt_panels()
    for nm in names:
        measure(nm, X, panels)
    pr("t", f"total {time.time() - T0:.0f} s")
