"""Check of the brief TASK_E61_THE_ERROR_RELAYED_OVER_A_LIMITED_NUMBER_OF_STEPS_20261118.md (revision 1).

Prints every figure that CHECK_OF_TASK_E61_BRIEF_20261118.md (beside this file) quotes. Run from the repository's root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/check_e61_brief.py

The Python standard library and numpy only. Files of the default branch are read into memory with `git show`; no code of the
repository is executed (the item layout, the stream and the draw of connections are written again here from the text of the
sources, and the rewritten stream is tested against the batch digest of the record's receipt). No file is written.

Parts: 1 the numerical test of the update with the error relayed over K steps; 2 the source; 3 the layout of an item and the
stream; 4 the connections of the network and the steps an error needs to reach every cell; 5 the record's figures; 6 the cost.
"""
import hashlib
import json
import subprocess
import sys
from random import Random

import numpy as np

D = "origin/codex/minifly-consolidated-20261023"
M = "MOD97_LOCAL_RULE_20261106"


def show(path):
    r = subprocess.run(["git", "show", f"{D}:{path}"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"git show failed for {path}: {r.stderr.decode()[:200]}")
    return r.stdout.decode()


# =====================================================================================================================
# Part 1. The numerical test (double precision, backward passes written by hand)
# =====================================================================================================================
def part1():
    print("=" * 100)
    print("PART 1. The update with the error relayed over K steps, against the direct definition and the full gradient")
    rng = np.random.default_rng(20261118)
    U, DIN, H, V = 12, 10, 9, 7                     # cells, input values, readout width, output symbols
    B, L = 5, 14
    ks = np.array([0, 1, 3, 4, 2])                  # letters of each item: answer positions 7+k, 8+k, 9+k differ between items
    A = np.stack([7 + ks, 8 + ks, 9 + ks], 1)       # (B, 3)
    assert A.max() <= L - 1
    amax = int(A.max())                             # 13: the farthest answer position; "the whole item" is K >= amax

    Mh = np.eye(U, dtype=bool)                      # mask of the recurrent map: each cell reads 3 others and itself
    for i in range(U):
        Mh[i, rng.choice([j for j in range(U) if j != i], 3, replace=False)] = True
    Mh3 = np.concatenate([Mh] * 3, 0).astype(float)                 # (3U, U)
    Mi = rng.random((U, DIN)) < 0.5
    Mi3 = np.concatenate([Mi] * 3, 0).astype(float)                 # (3U, DIN)
    P = dict(Wi=rng.normal(0, .5, (3 * U, DIN)) * Mi3, bi=rng.normal(0, .1, 3 * U),
             Wh=rng.normal(0, .6, (3 * U, U)) * Mh3, bh=rng.normal(0, .1, 3 * U),
             W2=rng.normal(0, .4, (H, DIN)), b2=rng.normal(0, .1, H), Wm=rng.normal(0, .4, (H, U)), bm=rng.normal(0, .1, H),
             W3=rng.normal(0, .5, (V, H)), b3=rng.normal(0, .1, V))
    table = rng.normal(0, 1, (6, DIN))
    x = rng.integers(0, 6, (B, L)); z1 = table[x]                   # (B, L, DIN), fixed code
    tgt = rng.integers(0, V, (B, 3))
    NMASK = 3 * B
    sig = lambda v: 1 / (1 + np.exp(-v))

    def cross(W, maskf, rule):                      # what the error crosses on its way back through a linear map
        if rule == "B":
            return W * maskf
        return np.sign(W * maskf) * (np.abs(W) * maskf).sum() / maskf.sum()     # sign(W) times one scale for the map

    def forward(P):
        gi = z1 @ (P["Wi"] * Mi3).T + P["bi"]
        h = np.zeros((B, U)); Hp, Hs = [], []
        for t in range(L):
            Hp.append(h)
            gh = h @ (P["Wh"] * Mh3).T + P["bh"]
            r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U]); c = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
            h = (1 - z) * c + z * h; Hs.append(h)
        Hp, Hs = np.stack(Hp, 1), np.stack(Hs, 1)
        z2 = np.tanh(z1 @ P["W2"].T + P["b2"] + Hs @ P["Wm"].T + P["bm"])
        lg = z2 @ P["W3"].T + P["b3"]
        return gi, Hp, Hs, z2, lg

    def loss_of(P, which=(0, 1, 2)):
        lg = forward(P)[4]; tot = 0.0
        for j in which:
            for b in range(B):
                v = lg[b, A[b, j]]; m = v.max(); tot += -(v[tgt[b, j]] - m - np.log(np.exp(v - m).sum()))
        return tot / NMASK                           # each position's loss is divided by the number of answer entries

    def readout_back(P, z2, lg, Hs, rule, which=(0, 1, 2)):
        dlg = np.zeros_like(lg)
        for j in which:
            for b in range(B):
                v = lg[b, A[b, j]]; p = np.exp(v - v.max()); p /= p.sum(); p[tgt[b, j]] -= 1
                dlg[b, A[b, j]] = p / NMASK
        g = {}
        g["W3"] = np.einsum("blv,blh->vh", dlg, z2); g["b3"] = dlg.sum((0, 1))
        one = np.ones_like(P["W3"])
        dz2 = dlg @ cross(P["W3"], one, rule)
        dp = dz2 * (1 - z2 ** 2)
        g["W2"] = np.einsum("blh,bld->hd", dp, z1); g["b2"] = dp.sum((0, 1))
        g["Wm"] = np.einsum("blh,blu->hu", dp, Hs); g["bm"] = dp.sum((0, 1))
        dH = dp @ cross(P["Wm"], np.ones_like(P["Wm"]), rule)
        return g, dH

    def step_back(P, dh, gi_t, hp, rule):
        """The backward of one step for any rows: dh (..., U), gi_t (..., 3U), hp (..., U). Returns dhp, dgi, dgh."""
        gh = hp @ (P["Wh"] * Mh3).T + P["bh"]
        r = sig(gi_t[..., :U] + gh[..., :U]); z = sig(gi_t[..., U:2 * U] + gh[..., U:2 * U]); c = np.tanh(gi_t[..., 2 * U:] + r * gh[..., 2 * U:])
        dc = dh * (1 - z); dz = dh * (hp - c); dhp = dh * z           # the gate and the term z * h(t-1): differentiated exactly
        dn = dc * (1 - c ** 2); dr = dn * gh[..., 2 * U:]; dghn = dn * r
        dar = dr * r * (1 - r); daz = dz * z * (1 - z)
        dgh = np.concatenate([dar, daz, dghn], -1); dgi = np.concatenate([dar, daz, dn], -1)
        dhp = dhp + dgh @ cross(P["Wh"], Mh3, rule)                     # the crossing between cells, under the rule
        return dhp, dgi, dgh

    def local_grads(dgi, dgh, hp, z1rows):
        gi2, z2r = dgi.reshape(-1, 3 * U), z1rows.reshape(-1, DIN); gh2, hp2 = dgh.reshape(-1, 3 * U), hp.reshape(-1, U)
        return dict(Wi=(gi2.T @ z2r) * Mi3, bi=gi2.sum(0), Wh=(gh2.T @ hp2) * Mh3, bh=gh2.sum(0))

    def add(a, b):
        return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}

    # (i) the brief's procedure: errors at all states from the readout, then K rounds on all positions at once with the
    # result shifted one position back, then the weight gradients from the summed error
    def brief(P, K, rule, shift="brief", rows_only=False):
        gi, Hp, Hs, z2, lg = forward(P)
        g, d = readout_back(P, z2, lg, Hs, rule)
        total = d.copy(); nrows = []
        for _ in range(K):
            if rows_only:                                 # the second economy: only the rows at which d is not zero
                idx = np.nonzero(np.abs(d).sum(-1) != 0); nrows.append(len(idx[0]))
                gg = np.zeros_like(d)
                gg[idx] = step_back(P, d[idx], gi[idx], Hp[idx], rule)[0]
            else:
                gg = step_back(P, d, gi, Hp, rule)[0]    # gg[:, t]: the error at h(t-1), the state entering position t
            d = np.zeros_like(gg)
            if shift == "brief":
                d[:, :-1] = gg[:, 1:]
            else:                                         # a wrong shift, to show that the checks would see it
                d[:, 1:] = gg[:, :-1]
            total = total + d
        _, dgi, dgh = step_back(P, total, gi, Hp, rule)
        return add(g, local_grads(dgi, dgh, Hp, z1)), nrows

    # (ii) the direct definition: for each answer position separately, the state entering position a - K is a constant
    def direct(P, K, rule):
        gi, Hp, Hs, z2, lg = forward(P)
        out = {}
        for j in range(3):
            gj, dH = readout_back(P, z2, lg, Hs, rule, which=(j,))
            out = add(out, gj)
            for b in range(B):
                a = A[b, j]; dh = dH[b, a]
                for t in range(a, max(a - K, 0) - 1, -1):
                    dhp, dgi, dgh = step_back(P, dh, gi[b, t], Hp[b, t], rule)
                    out = add(out, local_grads(dgi[None], dgh[None], Hp[b, t][None], z1[b, t][None]))
                    dh = dhp                                  # not used after t = a - K: h(a-K-1) is a constant there
        return out

    # (iii) the full gradient through the item; (iv) the gradient with the state a constant at every position
    def bptt(P, rule, cut_every=False):
        gi, Hp, Hs, z2, lg = forward(P)
        g, dH = readout_back(P, z2, lg, Hs, rule)
        carry = np.zeros((B, U))
        for t in range(L - 1, -1, -1):
            dh = dH[:, t] + carry
            dhp, dgi, dgh = step_back(P, dh, gi[:, t], Hp[:, t], rule)
            g = add(g, local_grads(dgi, dgh, Hp[:, t], z1[:, t]))
            carry = np.zeros_like(dhp) if cut_every else dhp
        return g

    def rel(g1, g2):
        return max(float(np.abs(g1[k] - g2[k]).max() / max(np.abs(g2[k]).max(), 1e-300)) for k in g2)

    # the hand-written full gradient against finite differences of the loss (exact rule)
    full_B = bptt(P, "B")
    fd = []
    for trial in range(4):
        v = {k: rng.normal(0, 1, P[k].shape) * (Mi3 if k == "Wi" else Mh3 if k == "Wh" else 1) for k in P}
        eps = 1e-6
        Pp = {k: P[k] + eps * v[k] for k in P}; Pm = {k: P[k] - eps * v[k] for k in P}
        num = (loss_of(Pp) - loss_of(Pm)) / (2 * eps); ana = sum(float((full_B[k] * v[k]).sum()) for k in P)
        fd.append(abs(num - ana) / abs(ana))
    print(f"1.0 the hand-written full gradient (exact rule) against central differences of the loss along 4 random directions: "
          f"largest relative difference {max(fd):.1e}")
    tot_loss = loss_of(P); parts = [loss_of(P, (j,)) for j in range(3)]
    print(f"1.0 the three losses of the answer positions add up to the batch's loss: {abs(sum(parts) - tot_loss):.1e}")
    print(f"    test network: {U} cells, {L} positions, {B} items, answer positions {A.tolist()}; the farthest answer position is {amax}")

    worst = {}
    for rule in ("B", "S"):
        full = bptt(P, rule); cut = bptt(P, rule, cut_every=True)
        print(f"-- rule {rule} ({'exact gradient' if rule == 'B' else 'sweep: sign(W) times one scale at every crossing, gates exact'})")
        for K in (0, 1, 2, 5, amax - 1, amax, L + 3):
            gb, _ = brief(P, K, rule); gd = direct(P, K, rule)
            e_bd = rel(gb, gd); e_full = rel(gb, full); e_cut = rel(gb, cut)
            worst[(rule, K)] = e_bd
            print(f"   K={K:2d}: brief's procedure against the direct definition {e_bd:.1e}; against the full gradient {e_full:.1e}; "
                  f"against the state a constant at every position {e_cut:.1e}")
        g_wrong, _ = brief(P, 2, rule, shift="wrong"); g2 = direct(P, 2, rule); g3 = direct(P, 3, rule)
        g_rows, nrows = brief(P, 5, rule, rows_only=True); g5, _ = brief(P, 5, rule)
        print(f"   K=2 with the shift d[:, 1:] = g[:, :-1] in place of the brief's: difference from the direct definition {rel(g_wrong, g2):.2e}")
        print(f"   K=2 against the direct definition with K=3 (an error of one step): {rel(brief(P, 2, rule)[0], g3):.2e}")
        print(f"   the second economy (only the rows at which d is not zero), K=5: difference from the reference form {rel(g_rows, g5):.1e}; "
              f"rows relayed in the five rounds {nrows} of {B * L} (three for each of {B} items = {3 * B})")
        g_rows16, nrows16 = brief(P, L + 3, rule, rows_only=True)
        print(f"   the same with K={L + 3}: rows relayed in each round {nrows16}")
    gS = bptt(P, "S"); gB = bptt(P, "B")
    print(f"1.9 the two rules differ (largest relative difference of the full gradients) {rel(gS, gB):.2f}")
    ok = all(v < 1e-12 for v in worst.values())
    print(f"1.10 THE NUMERICAL TEST {'PASSED' if ok else 'FAILED'}: the brief's procedure equals the direct definition for every K and rule "
          f"(largest {max(worst.values()):.1e}); K >= {amax} equals the full gradient; K = 0 equals the gradient with the state a constant")


# =====================================================================================================================
# Part 2. Statements about the source
# =====================================================================================================================
def part2():
    print("=" * 100)
    print("PART 2. Statements about the source (text read with git show; True means the text is present)")
    f = {n: show(f"{M}/{n}") for n in ("exp50_code.py", "e50_sparse_code.py", "e42_cells.py", "e49_interface.py", "exp42_cells.py",
                                       "exp49_interface.py", "exp3_state.py", "exp11_cell.py", "exp45.py", "exp26_candidate.py",
                                       "exp20_reference.py", "exp52_resume.py", "arch26.py")}
    T = [
        ("exp50_code.py", "def build(setting, seed):"), ("exp50_code.py", "def fwd(n, x, local, silence=False):"),
        ("exp50_code.py", "def install(n):"), ("exp50_code.py", "def step(n, opt, items, local):"),
        ("exp50_code.py", "def map_grads(n, x, y, mask, local):"), ("exp50_code.py", "    return g, untouched"),
        ("exp50_code.py", 'g = {k: P[k].grad.detach().clone() for k in ("rnn.weight_ih_l0", "lm.weight", "rnn.weight_hh_l0")}'),
        ("exp50_code.py", 'SOURCES = ("exp50_code.py", "e50_sparse_code.py", "exp49_interface.py", "e49_interface.py", "exp42_cells.py", "e42_cells.py") + C26.SOURCES'),
        ("exp50_code.py", "        if first is None:"), ("exp50_code.py", '            rec["first_batch_loss_start"] = first_batch_loss(n, first)'),
        ("exp50_code.py", '            rec["cosine_first_batch_start"] = cosines(n, first)'), ("exp50_code.py", "cosine_first_batch=cosines(n, first))"),
        ("exp50_code.py", '    rec["cosine_first_batch_end"] = cosines(n, first)'), ("exp50_code.py", "            if u % 1024 == 0 or u == dose:"),
        ("exp50_code.py", 'os.environ.get("E50_BESIDE", "not recorded")'), ("exp50_code.py", "multiply_adds=ma * X.BATCH * u"),
        ("exp50_code.py", "    mid = dose > FIRST_DOSE"), ("exp50_code.py", '    opt = R42.make_opt(m, layer)'),
        ("exp50_code.py", 'SETTINGS = {"C-dense": (512, 256), "C16-in": (16, 256), "C46-one": (46, 256)}'),
        ("e50_sparse_code.py", "    scale = layer.sign_scale(m) if (local and not layer.full) else None"),
        ("e50_sparse_code.py", "    out = lin(Hs, m.lm, local) if iface.k_out == U else"),
        ("e42_cells.py", "def rec(layer, m, h, W, bias, local, scale, kstep=None):"),
        ("e42_cells.py", "        gw = (g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])) * maskf"),
        ("e49_interface.py", "def masked(x, W, b, maskf, local):"), ("exp3_state.py", "def lin(x, layer, local):"),
        ("exp3_state.py", "        S = weight.sign() * weight.abs().mean()"),
        ("exp11_cell.py", "            yield render(t, a, b, rng.randint(0, kmax), rng)"),
        ("exp11_cell.py", '    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]'),
        ("exp11_cell.py", 'KMAX = {"M": 12, "N": 3}'), ("exp45.py", "        mask[i, len(p):len(p) + len(t)] = True"),
        ("exp45.py", "    L = max(len(r) for r in rows) + 1"),
        ("exp42_cells.py", "    A = {E.NAME(*t): E.panel(t, th[t][1], seed, ks=(12,), n=300)[12] for t in E.DEV}"),
        ("exp42_cells.py", "        Rs[E.NAME(*t)] = [E.render(t, a, b, 0, Random(0)) for a, b in s]"),
        ("exp42_cells.py", "    opt = C26.make_opt(m)"), ("exp26_candidate.py", "WD = 0.25"),
        ("exp20_reference.py", "LR, WD, BETAS = .001, .3, (.9, .98)"),
        ("exp52_resume.py", "    assert dose % P.ROW == 0 and ckpt_every % P.ROW == 0"),
        ("exp52_resume.py", 'assert cp.exists() and not wp.exists(), f"nothing to resume: {cp}"'),
        ("exp52_resume.py", "        assert R_.opt_groups(m, opt) == g_exp, R_.opt_groups(m, opt)"),
        ("arch26.py", "        hp = h if through_time else h.detach()"),
    ]
    for fn, s in T:
        print(f"  {s in f[fn]!s:5} {fn}: {s.strip()[:110]}")
    print(f"  {'needs_input_grad' not in f['e42_cells.py']!s:5} e42_cells.py: MaskedSign.backward does not consult ctx.needs_input_grad (it forms gw and gb on every call)")
    print(f"  {'import exp26_candidate as C26' in f['exp50_code.py'] and 'R_, X16, E, X = C26.R, C26.X16, C26.E, C26.X' in f['exp50_code.py']!s:5} "
          f"exp50_code.py: E and X are C26.E and C26.X (names the brief's code uses without importing them)")
    return f


# =====================================================================================================================
# Part 3. The layout of an item and the stream (written again from exp11_cell.py and exp45.py)
# =====================================================================================================================
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EOS, BOS, BATCH = 256, 257, 64


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt, list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]); rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(tasks_pairs, seed, kmax):
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tasks_pairs for pr in pairs]
    while True:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, kmax)
            yield render(t, a, b, k, rng) + (k,)


def batch_arrays(items):
    rows = [p + t for p, t, _ in items]
    L = max(len(r) for r in rows) + 1
    data = np.full((len(rows), L), BOS, dtype=np.int64)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = r
    x, y = data[:, :-1], data[:, 1:]
    mask = np.zeros(y.shape, dtype=bool)
    for i, (p, t, _) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


def part3(rec_S):
    print("=" * 100)
    print("PART 3. The layout of an item, the answer positions and the stream of E.batches(tp, 73129, 12)")
    seed = 73129
    tp = [(t, split(t, .5, seed)[0]) for t in DEV]
    st = stream(tp, seed, 12)
    h = hashlib.sha256(); nb = 0
    letters = [0] * 13; Ls = {}; ans_ok = True; maxpos = 0
    for bi in range(256):
        items = [next(st) for _ in range(BATCH)]
        x, y, mask = batch_arrays(items)
        for t_ in (x, y, mask):
            h.update(str(tuple(t_.shape)).encode()); h.update(np.ascontiguousarray(t_).tobytes())
        nb += 1
        Ls[x.shape[1]] = Ls.get(x.shape[1], 0) + 1
        for i, (p, t, k) in enumerate(items):
            letters[k] += 1
            ans_ok &= [int(v) for v in np.nonzero(mask[i])[0]] == [7 + k, 8 + k, 9 + k] and 1 + len(p) + len(t) - 1 == 10 + k
            maxpos = max(maxpos, 9 + k)
            if bi == 0 and i == 0:
                print(f"3.1 first item (k={k}): input positions 0..{9 + k} = {[BOS] + (p + t)[:-1]}; targets at {[int(v) for v in np.nonzero(mask[i])[0]]}")
    rowd = next(r for r in rec_S["curve"] if r["step"] == 256)["data_digest"]
    print(f"3.2 the stream written again here gives the record's batch digest at update 256: {h.hexdigest() == rowd['sha256'] and nb == rowd['batches']} "
          f"({h.hexdigest()[:16]} against {rowd['sha256'][:16]})")
    n = sum(letters)
    print(f"3.3 answer positions are 7+k, 8+k, 9+k and an item has 10+k input positions, in all {n} items of the first 256 batches: {bool(ans_ok)}")
    print(f"3.4 items by number of letters (k = 0..12): {letters}; expected under a uniform draw {n / 13:.1f} each; "
          f"chi-square {sum((c - n / 13) ** 2 / (n / 13) for c in letters):.1f} on 12 degrees of freedom")
    print(f"3.5 positions of a batch (input length): {dict(sorted(Ls.items()))}; the largest answer position {maxpos}, "
          f"so an error from the last answer position needs {maxpos} steps to reach position 0")
    print("3.6 distances from the first answer position (7+k): second operand 1 and 2; operation byte 3+k; first operand 4+k (second digit) "
          "and 5+k (first digit); task byte 6+k; start symbol 7+k")
    print("    share of training items reached by the error of the first answer position, uniform k on 0..12 (exact) and in the stream:")
    for K in (0, 1, 2, 4, 5, 8, 16, 17, 21):
        row = []
        for name, dd in (("second digit of first operand", 4), ("both digits", 5), ("operation byte", 3), ("task byte", 6)):
            exact = sum(1 for k in range(13) if K >= dd + k)
            emp = sum(letters[k] for k in range(13) if K >= dd + k) / n
            row.append(f"{name} {exact}/13 ({emp:.3f})")
        print(f"    K={K:2d}: " + "; ".join(row))
    print("3.7 the panels: A has 12 letters (exp42_cells.panels, ks=(12,)): first operand 16 and 17 steps from the first answer position; "
          "R has no letters (render(t, a, b, 0, ...)): first operand 4 and 5 steps; T (taught) has 12 letters")
    print(f"    so K >= 5 reaches both digits of the first operand in every item of R; K >= 17 in every item of A (K = 16 reaches the second digit only)")


# =====================================================================================================================
# Part 4. The connections (written again from e42_cells.draws) and the steps an error needs to reach every cell
# =====================================================================================================================
def part4():
    print("=" * 100)
    print("PART 4. The connections of the network and the steps an error needs to reach every cell")
    U, SIDE, RADIUS, seed = 256, 16, 3, 73129

    def nb(i):
        r, c = divmod(i, SIDE)
        return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})
    rc = Random(f"e42|connections|{seed}")
    A = [sorted(rc.sample(sorted(nb(i)), 10)) for i in range(U)]
    dg = hashlib.sha256(json.dumps(A).encode()).hexdigest()[:16]
    print(f"4.1 the draw written again here has the digest of e42_cells.py's self-check for N = 10, seed 73129: {dg == 'fde78220c57da3d4'} ({dg}); "
          f"neighbourhood size {len(nb(0))}")

    def steps_to_all(reads):
        out = []
        for i in range(U):
            reach = {i}; s = 0
            while len(reach) < U:
                reach = reach | {j for q in reach for j in reads[q]}; s += 1
            out.append(s)
        return min(out), max(out)
    reads = [set(a) | {i} for i, a in enumerate(A)]
    readby = [set() for _ in range(U)]
    for i, a in enumerate(A):
        for j in a:
            readby[j].add(i)
    readby = [s | {i} for i, s in enumerate(readby)]
    print(f"4.2 this network (10 of 48 neighbours): an error at one cell reaches every cell after (fewest, most over the starting cell) "
          f"{steps_to_all(reads)} steps; a cell's state reaches every cell forward after {steps_to_all(readby)} steps")
    full = [set(nb(i)) | {i} for i in range(U)]
    print(f"4.3 a cell that read all 48 neighbours (fan-in 48, near fifty, from the neighbourhood): {steps_to_all(full)} steps")
    rr = Random("check_e61|wide50")
    wide = [set(rr.sample([j for j in range(U) if j != i], 50)) | {i} for i in range(U)]
    print(f"4.4 a cell that read 50 cells drawn from all cells (one draw): {steps_to_all(wide)} steps; 50**2 = 2500 > 256")


# =====================================================================================================================
# Part 5. The record's figures
# =====================================================================================================================
def reached(c, key, th=80):
    st = [r["step"] for r in c]; v = [r[key] for r in c]
    three = next((st[i] for i in range(len(v) - 2) if all(v[i + j] >= th for j in range(3))), None)
    one = next((st[i] for i in range(len(v)) if v[i] >= th), None)
    stay = next((st[i] for i in range(len(v)) if all(w >= th for w in v[i:])), None)
    return three, one, stay


def level(c, key, u):
    st = [r["step"] for r in c]; i = st.index(u)
    return sum(c[j][key] for j in range(i - 2, i + 1)) / 3


def part5():
    print("=" * 100)
    print("PART 5. The record's figures (receipts of the default branch)")
    recs = {r: json.loads(show(f"{M}/exp45/e50/e50_C46-one_{r}_73129.json")) for r in "BS"}
    for r in "BS":
        d = recs[r]; c = d["curve"]
        assert d["setting"] == "C46-one" and d["rule"] == r and d["dose"] == 49152 and d["complete"]
        R3, R1, Rs = reached(c, "R"); A3, A1, As = reached(c, "A")
        print(f"5.1 {r}: rows {len(c)}; updates to 80 on R {R3} [single row {R1}] (stays at or above from {Rs}); on A {A3} [{A1}] (stays from {As})")
        for u in (24576, 49152):
            print(f"    levels at {u}: A {level(c, 'A', u):.4f}, R {level(c, 'R', u):.4f}, T {level(c, 'taught', u):.4f}")
        row = next(x for x in c if x["step"] == R3)
        print(f"    CPU-hours of the run {(d['cpu_updates'] + d['cpu_scoring']) / 3600:.3f} (updates {d['cpu_updates'] / 3600:.3f}, scoring {d['cpu_scoring'] / 3600:.3f}); "
              f"CPU-seconds a update {d['cpu_updates'] / 49152:.4f}, a scored row {d['cpu_scoring'] / len(c):.2f}; "
              f"CPU-seconds of the updates to the row of 80 on R {row['cpu_updates']:.0f}; 1.25 times the updates to 80 on R {1.25 * R3:,.0f}")
        print(f"    versions {d['versions']['torch']}, Python {d['versions']['python']}; processes beside {d['processes_beside']}; "
              f"multiply_adds at the last row {c[-1]['multiply_adds']:,} = {d['multiply_adds_per_position']:,} x 64 x 49,152")
    print(f"5.1 first-batch loss at the start under the two rules (receipt of B): {recs['B']['first_batch_loss_start']}; "
          f"spacing of single-precision numbers at that loss {float(np.spacing(np.float32(5.591))):.2e}")
    rng = np.random.default_rng(7); lg = rng.normal(0, 1, (192, 257)).astype(np.float32); tg = rng.integers(0, 257, 192)

    def ce(v):
        m = v.max(1, keepdims=True); return float(np.mean(-(v[np.arange(192), tg] - m[:, 0] - np.log(np.exp(v - m).sum(1)))))
    noise = (1 + rng.choice([-1, 1], lg.shape) * np.float32(2 ** -23) * rng.integers(0, 4, lg.shape)).astype(np.float32)
    print(f"5.1 a single-precision loss of 192 answer entries changes by {abs(ce(lg) - ce((lg * noise).astype(np.float32))):.1e} when every logit "
          f"moves by 0 to 3 units of its last place (the size of a change of summation order in a matrix product)")
    print(f"5.2 ratio of the sweep's updates to 80 on R to the exact gradient's: {reached(recs['S']['curve'], 'R')[0] / reached(recs['B']['curve'], 'R')[0]:.4f}")
    for r in "BS":
        d = json.loads(show(f"{M}/exp45/e50/e50_C-dense_{r}_73129.json")); c = d["curve"]
        print(f"5.3 the dense-input reference of E50 (C-dense, {r}, {d['dose']:,} updates): updates to 80 on R {reached(c, 'R')[0]}, on A {reached(c, 'A')[0]}")
    for r in "BS":
        d = json.loads(show(f"{M}/exp45/e42/e42_G_{r}_73129.json")); c = d["curve"]
        print(f"5.4 the dense network of E42 (configuration G, every cell reads every cell, learned code; {r}, {d['dose']:,} updates): "
              f"updates to 80 on R {reached(c, 'R')[0]}, on A {reached(c, 'A')[0]}; levels at {d['dose']:,}: R {level(c, 'R', d['dose']):.2f}, A {level(c, 'A', d['dose']):.2f}")
    for r in "BS":
        c = recs[r]["curve"]
        print(f"5.5 the record ({r}): levels of the length test at 12 and 40 letters at 49,152: {level(c, 'len12', 49152):.2f}, {level(c, 'len40', 49152):.2f}")
    return recs


# =====================================================================================================================
# Part 6. The cost
# =====================================================================================================================
def part6(recs):
    print("=" * 100)
    print("PART 6. The cost: multiply-adds of an update and a forecast for the twelve runs (an estimate; torch is not run here)")
    B, L, U, Dn, H, V = 64, 22, 256, 512, 512, 257
    BL = B * L
    inp = BL * Dn * 3 * U; recur = BL * U * 3 * U; l2 = BL * Dn * H; lm = BL * U * H; l3 = BL * H * V
    fwd = inp + recur + l2 + lm + l3
    bwd_B = inp + 2 * recur + l2 + 2 * lm + 2 * l3                 # weight gradients everywhere; input gradients where an input needs them
    bwd_S = bwd_B + inp + l2                                         # the sweep's custom backward also forms the input's gradient of the input map and l2
    upd = {"B": fwd + bwd_B, "S": fwd + bwd_S}
    per_round = {"B": recur, "S": 2 * recur}                         # one round of (d): the error crossing (and, under the sweep, MaskedSign's gw)
    extra = {"B": 0, "S": recur}                                     # (a) and (b) beside the record; under B (e) needs no input gradient
    print(f"6.1 multiply-adds of a record's update (B={B}, L={L}): B {upd['B'] / 1e9:.2f} G, S {upd['S'] / 1e9:.2f} G; one round of relay over all "
          f"positions: B {per_round['B'] / 1e9:.3f} G ({per_round['B'] / upd['B']:.3f} of an update), S {per_round['S'] / 1e9:.3f} G ({per_round['S'] / upd['S']:.3f})")
    Ks = (0, 1, 2, 4, 8, 16); tot = 0; tot_noK1 = 0; tot_rows = 0
    for r in "BS":
        d = recs[r]; pu = d["cpu_updates"] / 49152; pr = d["cpu_scoring"] / len(d["curve"])
        for K in Ks:
            f_ref = 1 + (extra[r] + K * per_round[r]) / upd[r]
            f_rows = 1 + (extra[r] + K * per_round[r] * 3 / L) / upd[r]
            h = (49152 * pu * f_ref + 193 * pr) / 3600; hr = (49152 * pu * f_rows + 193 * pr) / 3600
            tot += h; tot_rows += hr
            if K != 1:
                tot_noK1 += h
            print(f"    {r} K={K:2d}: factor {f_ref:.2f} on the record's CPU per update, {h:.2f} CPU-hours (with the second economy {hr:.2f})")
    print(f"6.2 forecast for the twelve runs, reference form: {tot:.1f} CPU-hours; without the two runs of K = 1: {tot_noK1:.1f}; "
          f"with the second economy: {tot_rows:.1f}; the brief's limit 36")


def main():
    part1()
    part2()
    recs = part5()
    part3(recs["S"])
    part4()
    part6(recs)


if __name__ == "__main__":
    main()
