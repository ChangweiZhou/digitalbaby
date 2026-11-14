"""Checks for the audit of revision 3 of the E32 brief. Post hoc, unregistered; numpy on saved weights; nothing is trained.

D. The birth's batch stream rebuilt in pure Python and checked against the receipt's data digest at 4,096 batches: so the batch of
   update 4,097 is available without training.
E. A numpy backward pass for the gate rows (rows 256-511 of the two GRU matrices and their biases), checked against finite differences
   on the exact path; the sweep's path replaces W by sign(W) * mean|W| where the error crosses l3, lm and weight_hh_l0.
F. The nine strengths: task-gradient norm (sweep) and hinge-gradient norm on the gate rows at E28's row-4,096 weights and the batch of
   update 4,097, and lam for 0.25 / 0.05 / 0.01. A third, independent implementation for the workers' two.
G. The floor's keepers at update 4,096 followed through the saved rows: their mean pre-activation at letter steps.
H. A first-order estimate of what the hinge adds, per update, to those keepers' pre-activation at each strength, beside what the floor
   takes away. Model: AdamW at its stationary step, lr * (mean g / rms g) per weight over the next 128 batches at frozen weights, plus
   decay 0.25. An estimate, not a run.
I. The marks: keeps-and-writes at 12 / 20 / 40 letters, the floor's last-three-row means, and what "moves" asks for at each distance.
"""
import hashlib, json, sys
from pathlib import Path
from random import Random
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import np_gru as G
from np_gru import load_weights, panel, split, render, DEV, BOS, EOS, git_bytes
from e32_r2_checks import panels, run, keeping, FINALS, SEEDS, ROWS, REF, P
C = 4.59512; U = 256; KMAX = 12; BATCH = 64; LR = 0.001; WD = 0.25
f64 = np.float64

def row_file(s, row):
    return P + (f'e28/e28_dev_S-all_{s}.pt' if row == 24576 else f'e28/e28_dev_S-all_{s}_row{row}.pt')

# ---------------------------------------------------------------- D. the stream (exp11_cell.stream / batches, exp45.batch_tensor, exp26_candidate.Digest)
def stream(seed):
    tp = [(t, split(t, .5, seed)[0]) for t in DEV]
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tp for pr in pairs]
    while True:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            yield render(t, a, b, rng.randint(0, KMAX), rng)

def batch_tensor(items):
    rows = [p + t for p, t in items]; L = max(map(len, rows)) + 1
    data = np.full((len(rows), L), BOS, np.int64)
    for i, r in enumerate(rows): data[i, 1:len(r) + 1] = r
    mask = np.zeros((len(rows), L - 1), bool)
    for i, (p, t) in enumerate(items): mask[i, len(p):len(p) + len(t)] = True
    return np.ascontiguousarray(data[:, :-1]), np.ascontiguousarray(data[:, 1:]), mask

def batches_after(seed, skip, n, check=True):
    st = stream(seed); h = hashlib.sha256()
    for _ in range(skip):
        for t in batch_tensor([next(st) for _ in range(BATCH)]):
            h.update(str(tuple(t.shape)).encode()); h.update(t.tobytes())
    ok = None
    if check:
        rec = json.loads(git_bytes(P + f'e28/e28_dev_S-all_{seed}.json', REF))
        want = {r['step']: r['data_digest'] for r in rec['curve']}[skip]
        ok = (want['batches'] == skip and want['sha256'] == h.hexdigest())
    return [batch_tensor([next(st) for _ in range(BATCH)]) for _ in range(n)], ok

# ---------------------------------------------------------------- E. forward with everything kept, and the backward for the gate rows
def sig(v): return 1.0 / (1.0 + np.exp(-v))

def fwd(W, x):
    g = lambda k: W[k].astype(f64)
    z1 = np.tanh(g('book.weight')[x] @ g('l1.weight').T + g('l1.bias'))
    gi = z1 @ g('rnn.weight_ih_l0').T + g('rnn.bias_ih_l0')
    Whh, bhh = g('rnn.weight_hh_l0'), g('rnn.bias_hh_l0')
    N, L = x.shape; h = np.zeros((N, U), f64)
    K = {k: np.empty((N, L, U), f64) for k in ('hp', 'z', 'r', 'n', 'ghn', 'az', 'h')}
    for t in range(L):
        gh = h @ Whh.T + bhh
        az = gi[:, t, U:2 * U] + gh[:, U:2 * U]
        r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(az); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        K['hp'][:, t] = h; h = (1 - z) * n + z * h
        for k, v in (('z', z), ('r', r), ('n', n), ('ghn', gh[:, 2 * U:]), ('az', az), ('h', h)): K[k][:, t] = v
    z2 = np.tanh(z1 @ g('l2.weight').T + g('l2.bias') + K['h'] @ g('lm.weight').T + g('lm.bias'))
    K.update(z1=z1, z2=z2, logits=z2 @ g('l3.weight').T + g('l3.bias'))
    return K

def task_loss(W, x, y):
    lg = fwd(W, x)['logits']; real = y != BOS
    lg = lg - lg.max(-1, keepdims=True); lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    return float(-np.take_along_axis(lp, np.where(real, y, 0)[..., None], -1)[..., 0][real].mean())

def task_grad(W, x, y, path, K=None):
    """Gradient of the every-byte cross entropy (mean over y != BOS) on the gate rows. path 'exact' or 'sign' (the sweep)."""
    K = K or fwd(W, x); real = y != BOS; N, L = x.shape
    lg = K['logits'] - K['logits'].max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    d = p; np.put_along_axis(d, np.where(real, y, 0)[..., None], np.take_along_axis(d, np.where(real, y, 0)[..., None], -1) - 1, -1)
    d = d * real[..., None] / real.sum()
    S = (lambda M: M) if path == 'exact' else (lambda M: np.sign(M) * np.abs(M).mean())
    dH = ((d @ S(W['l3.weight'].astype(f64))) * (1 - K['z2'] ** 2)) @ S(W['lm.weight'].astype(f64))
    Shh = S(W['rnn.weight_hh_l0'].astype(f64))
    Gih, Ghh, gb = np.zeros((U, K['z1'].shape[-1])), np.zeros((U, U)), np.zeros(U); nxt = 0.0
    for t in reversed(range(L)):
        dh = dH[:, t] + nxt
        z, n, r, hp, ghn = (K[k][:, t] for k in ('z', 'n', 'r', 'hp', 'ghn'))
        dan = dh * (1 - z) * (1 - n ** 2); daz = dh * (hp - n) * z * (1 - z)
        gh = np.concatenate([dan * ghn * r * (1 - r), daz, dan * r], 1)
        Gih += daz.T @ K['z1'][:, t]; Ghh += daz.T @ hp; gb += daz.sum(0)
        nxt = dh * z + gh @ Shh
    return Gih, Ghh, gb

def hinge(K, x, y):
    """The brief's term at lam = 1 and its gradient on the gate rows."""
    msk = (x != BOS) & (y != BOS); a = K['az']; n = msk.sum() * U
    q = -2 * np.maximum(C - np.abs(a), 0) * np.sign(a) * msk[..., None] / n
    term = float((np.maximum(C - np.abs(a), 0) ** 2 * msk[..., None]).sum() / n)
    inside = float((np.abs(a) < C)[msk].mean())
    return (np.einsum('nlu,nli->ui', q, K['z1']), np.einsum('nlu,nli->ui', q, K['hp']), q.sum((0, 1))), term, inside

def norm(g):          # rows 256-511 of weight_ih_l0 and weight_hh_l0, and of both biases (each bias gets the same gradient)
    return float(np.sqrt((g[0] ** 2).sum() + (g[1] ** 2).sum() + 2 * (g[2] ** 2).sum()))

if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'DEFGHI'
    cache = {}
    def first_batches(s, n):
        if (s, n) not in cache: cache[(s, n)] = batches_after(s, 4096, n)
        return cache[(s, n)]
    if 'D' in part:
        print('D. the stream rebuilt without the harness: digest at 4,096 batches equals the receipt\'s; shape of the batch of update 4,097')
        for s in SEEDS:
            bs, ok = first_batches(s, 128); x, y, m = bs[0]
            both = (x != BOS) & (y != BOS)
            print(f"   {s}: digest equal {ok} | x {x.shape} | masked by (x != BOS) & (y != BOS): {int(both.sum())}, by x != BOS alone: {int((x != BOS).sum())}, real targets: {int((y != BOS).sum())}")
    if 'E' in part:
        s = SEEDS[0]; W, _ = load_weights(row_file(s, 4096), ref=REF); x, y, _ = first_batches(s, 128)[0][0]
        Gih, Ghh, gb = task_grad(W, x, y, 'exact'); rng = np.random.default_rng(0); worst = 0.0; eps = 1e-5
        for name, Gm in (('rnn.weight_ih_l0', Gih), ('rnn.weight_hh_l0', Ghh), ('rnn.bias_ih_l0', gb), ('rnn.bias_hh_l0', gb)):
            for _ in range(5):
                j = int(rng.integers(U)); idx = (U + j, int(rng.integers(W[name].shape[1]))) if W[name].ndim == 2 else (U + j,)
                Wp = {k: v.astype(f64) for k, v in W.items()}; Wm = {k: v.copy() for k, v in Wp.items()}
                Wp[name][idx] += eps; Wm[name][idx] -= eps
                num = (task_loss(Wp, x, y) - task_loss(Wm, x, y)) / (2 * eps); ana = Gm[(j,) + idx[1:]]
                worst = max(worst, abs(num - ana) / max(abs(num), abs(ana), 1e-12))
        print(f"E. backward on the exact path against central differences, 20 entries of the gate rows: worst relative difference {worst:.2e}")
    if 'F' in part:
        print('F. row-4,096 weights, batch of update 4,097 | loss | task-gradient norm on the gate rows (sweep) | hinge-gradient norm at lam = 1 | hinge value at lam = 1 | gate values inside the hinge | lam at 0.25 / 0.05 / 0.01')
        for s in SEEDS:
            W, _ = load_weights(row_file(s, 4096), ref=REF); x, y, _ = first_batches(s, 128)[0][0]; K = fwd(W, x)
            nt = norm(task_grad(W, x, y, 'sign', K)); gp, term, inside = hinge(K, x, y); ng = norm(gp)
            print(f"   {s} | {task_loss(W, x, y):.4f} | {nt:.6f} | {ng:.6f} | {term:.4f} | {100 * inside:.1f}% | " + ' / '.join(f'{r * nt / ng:.5f}' for r in (.25, .05, .01))
                  + f" | at 0.25 the term adds {0.25 * nt / ng * term:.3f} to the loss")
    if 'G' in part or 'H' in part:
        print('G. the floor\'s keepers at update 4,096 (12-letter panel), followed through the rows: mean pre-activation at letter steps, the gate it gives, and how many still keep')
        keepers = {}
        for s in SEEDS:
            pk = panels(s); x = np.array([[BOS] + p + t[:-1] for p, t in pk[12]], dtype=np.int64); line = []
            for row in ROWS:
                W, _ = load_weights(row_file(s, row), ref=REF); K = fwd(W, x)
                letter = np.array([[r == 'letter' for r in G.roles_of(xr)] for xr in x])
                keep = (np.where(letter[..., None], K['z'], 1.0).prod(1) >= .5).sum(0) >= len(x) / 2
                if row == 4096: keepers[s] = keep
                a = K['az'][letter][:, keepers[s]].mean(); al = np.abs(K['az'][letter]).mean()
                gate_rows = np.sqrt((W['rnn.weight_ih_l0'][U:2 * U].astype(f64) ** 2).sum() + (W['rnn.weight_hh_l0'][U:2 * U].astype(f64) ** 2).sum())
                line.append(f"{row}: a {a:5.2f} (z {sig(a):.3f}), {int((keep & keepers[s]).sum())} of {int(keepers[s].sum())} keep, gate-row norm {gate_rows:5.2f}, all units mean |a| at letters {al:4.2f}")
            print(f"   {s}\n      " + '\n      '.join(line))
    if 'H' in part:
        print('H. first-order drift of those keepers\' letter pre-activation per 1,000 updates at the row-4,096 weights (128 batches from update 4,097, frozen weights)')
        print('   seed | observed in the floor, rows 4,096 to 8,192 | model: decay alone | model: floor (decay + task) | hinge adds at 0.25 / 0.05 / 0.01 | updates for the hinge alone to carry a gate from its present value to the limit, median over all units and letter steps, at 0.25 / 0.05 / 0.01')
        for s in SEEDS:
            bs, _ = first_batches(s, 128); W, _ = load_weights(row_file(s, 4096), ref=REF); W8, _ = load_weights(row_file(s, 8192), ref=REF)
            Gt, Gp, zb, hb, cnt, a0, a8, allA = [], [], 0, 0, 0, [], [], []
            for x, y, _ in bs:
                K = fwd(W, x); Gt.append(task_grad(W, x, y, 'sign', K)); Gp.append(hinge(K, x, y)[0])
                letter = (x >= 65) & (x <= 90)
                zb = zb + K['z1'][letter].sum(0); hb = hb + K['hp'][letter].sum(0); cnt += int(letter.sum())
                a0.append(K['az'][letter][:, keepers[s]].mean()); a8.append(fwd(W8, x)['az'][letter][:, keepers[s]].mean()); allA.append(K['az'][letter])
            zb, hb = zb / cnt, hb / cnt; lam = [r * norm(Gt[0]) / norm(Gp[0]) for r in (.25, .05, .01)]
            wih, whh = W['rnn.weight_ih_l0'][U:2 * U].astype(f64), W['rnn.weight_hh_l0'][U:2 * U].astype(f64)
            bias = (W['rnn.bias_ih_l0'][U:2 * U] + W['rnn.bias_hh_l0'][U:2 * U]).astype(f64)
            def drift(l):                                   # per update, every unit: change of the mean letter pre-activation
                out = 0.0
                for i, xin in ((0, zb), (1, hb)):
                    g = np.stack([Gt[b][i] + l * Gp[b][i] for b in range(len(bs))]); u = g.mean(0) / np.sqrt((g ** 2).mean(0) + 1e-30)
                    out = out + (-LR * u) @ xin
                g = np.stack([Gt[b][2] + l * Gp[b][2] for b in range(len(bs))]); u = g.mean(0) / np.sqrt((g ** 2).mean(0) + 1e-30)
                return out + 2 * (-LR * u)
            decay = -LR * WD * (wih @ zb + whh @ hb + bias); d0 = drift(0.0); k = keepers[s]
            obs = (np.mean(a8) - np.mean(a0)) / 4096
            A = np.concatenate(allA); gap = np.maximum(C - np.abs(A), 0)                                   # (letter steps, units)
            adds = [drift(l) - d0 for l in lam]
            speed = [np.abs(ad) for ad in adds]                                                             # per unit, per update, at the present hinge force
            t_lim = [float(np.median((gap / np.maximum(sp, 1e-12))[gap > 0])) for sp in speed]
            print(f"   {s} | {1000 * obs:+.2f} | {1000 * decay[k].mean():+.2f} | {1000 * (decay + d0)[k].mean():+.2f} | " + ' / '.join(f'{1000 * ad[k].mean():+.1f}' for ad in adds)
                  + ' | ' + ' / '.join(f'{t:.0f}' for t in t_lim) + f" | keepers' a at 4,096 on these batches {np.mean(a0):.2f}, at 8,192 {np.mean(a8):.2f}")
            nts = np.array([norm(g) for g in Gt]); ngs = np.array([norm(g) for g in Gp])
            print(f"      task-gradient norm on the gate rows over the 128 batches: batch 4,097 {nts[0]:.4f}, mean {nts.mean():.4f}, min {nts.min():.4f}, max {nts.max():.4f}; "
                  f"hinge-gradient norm: batch 4,097 {ngs[0]:.4f}, min {ngs.min():.4f}, max {ngs.max():.4f}; lam at 0.25 from single batches spans {0.25 * (nts / ngs).min():.5f} to {0.25 * (nts / ngs).max():.5f}")
    if 'I' in part:
        print('I. keeps-and-writes (keepers) at 12 / 20 / 40 letters, by the brief\'s definitions')
        for s in SEEDS:
            pk = panels(s); acc = {}
            for row in ROWS:
                W, _ = load_weights(row_file(s, row), ref=REF); cells = []
                for k in (12, 20, 40):
                    _, Z, roles, a = run(W, pk[k]); b, _, w = keeping(Z, roles); acc[(row, k)] = a; cells.append(f"{int((b & w).sum())} ({int(b.sum())})")
                if row in (4096, 16384, 20480, 24576): print(f"   floor {s} row {row:5d}: " + '  '.join(cells))
            m = {k: np.mean([acc[(r, k)] for r in (16384, 20480, 24576)]) for k in (12, 20, 40)}
            print(f"   floor {s} mean of the last three rows, accuracy at 12 / 20 / 40: {m[12]:.1f} / {m[20]:.1f} / {m[40]:.1f}  ->  'moves' asks for {m[12] + 30:.1f} / {m[20] + 30:.1f} / {m[40] + 30:.1f}")
        for arm, pat in FINALS:
            for s in SEEDS:
                W, _ = load_weights(P + pat.format(s=s), ref=REF); pk = panels(s); cells = []
                for k in (12, 20, 40):
                    _, Z, roles, a = run(W, pk[k]); b, _, w = keeping(Z, roles); cells.append(f"{int((b & w).sum())} ({int(b.sum())}) acc {a:5.1f}")
                print(f"   final {arm:6s} {s}: " + '  '.join(cells))
