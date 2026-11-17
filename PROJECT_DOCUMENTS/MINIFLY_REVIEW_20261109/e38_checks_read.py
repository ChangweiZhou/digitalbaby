#!/usr/bin/env python3
"""E38, the two checks that failed under the sweep: what the record holds and what the saved weights say (17 November 2026).

Read-only.  numpy and the standard library; no torch, no harness code is imported or run, and nothing is written.  Every file is
read with `git show <REF>:<path>` (through np_gru.git_bytes), weight files one at a time and only in memory.

Run from anywhere:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e38_checks_read.py [A] [B] [C] [D]

A  The receipts of checks 2 and 3 under both rules and of the worker's diagnostic: the largest difference per tensor, the tensors
   beyond the tolerance, the update at which the difference leaves the rounding level, the hash of the runner in each receipt.
B  The scores of all 20 committed weight files (ten lives at rows 6,144 and 12,288) recomputed with the reviewer's numpy reader
   and set beside the receipts as whole counts: add and sub under 89 on all pairs without letters, split by whether the pair
   needs the reduction; the held-out panels of the first family at 12 letters; add and sub under 97 held out at 12 letters.
C  How far each life's weights are from the birth weights, per tensor: what a pull toward the birth should show and what a pull
   that did not act, or acted with the wrong sign or strength, would show.
D  On the first two batches of the life, from the birth file: the error gradient under both rules in my own numpy backward pass
   (checked against finite differences); what a difference of rounding size in the weights does to it; what the change of sign
   of one weight that is next to zero does to it; the arithmetic of the pull's two forms and of the optimizer's own decay; and
   how many weights lie where a difference of rounding size changes their sign.  Two updates' arithmetic is carried out in
   memory for this and nothing is kept: no network is trained.

Names: rule S is the sweep (the error crosses each linear map through sign(W) * mean|W|, exp3_state.SignLinear); rule B is exact
back-propagation.  "Relative difference" of two tensors is the brief's and the runner's: the largest absolute difference over
the largest absolute entry (exp38_erase.max_rel).
"""
import sys
sys.dont_write_bytecode = True
import hashlib, json, os, time
from random import Random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from np_gru import git_bytes, load_weights, forward as fwd32, exact_teacher_forced, sigmoid, split, render, panel, DEV, LIFE, BOS, EOS

REF = '7cabb13adc38753483d34dd5472fc3c84c4de675'          # origin/codex/minifly-consolidated-20261023 on 17 November 2026
M = 'MOD97_LOCAL_RULE_20261106/'; E38 = M + 'exp45/e38/'; E26 = M + 'exp45/e26/'
SEED = 73129; ARMS = ('zero', 'none', 'anchor', 'anchor10', 'rehearse'); RULES = ('S', 'B'); ROWS = (6144, 12288)
LR, B1, B2, EPS, BATCH, KMAX = 0.001, 0.9, 0.98, 1e-8, 64, 12
PULLED = ('l1.weight', 'l1.bias', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0', 'lm.weight', 'lm.bias',
          'l2.weight', 'l2.bias', 'l3.weight', 'l3.bias')            # group 0 of make_opt: everything but the byte table
BACK = ('l3.weight', 'lm.weight', 'l2.weight', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'l1.weight')   # maps the error crosses
want = {a for a in sys.argv[1:] if len(a) == 1} or set('ABCD')
js = lambda path: json.loads(git_bytes(path, REF))
short = lambda n: n.replace('rnn.', '').replace('_l0', '').replace('weight', 'w').replace('bias', 'b')
t00 = time.process_time()


# ================================================================ A. the receipts
if 'A' in want:
    print('A. receipts of checks 2 and 3, and of the diagnostic (ref %s)' % REF[:8])
    for which, tol in (('2', 1e-5), ('3', 1e-6)):
        for rule in RULES:
            r = js(E38 + f'e38_check{which}_{rule}_{SEED}.json'); res = r['result']
            keys = [k for k in res if isinstance(res[k], dict) and 'l1.weight' in res[k]]
            print(f"   check {which} rule {rule}: pass {res['pass']}, tolerance {res['tolerance']:g}; commit {r['provenance']['git']['commit'][:8]}, "
                  f"exp38_erase.py sha {r['provenance']['sources_sha256']['exp38_erase.py'][:12]}, torch {r['provenance']['host']['torch']}")
            for k in keys:
                v = res[k]; worst = max(v, key=v.get); over = sorted((n for n in v if v[n] > tol), key=lambda n: -v[n])
                print(f"      {k}: largest {v[worst]:.3e} ({worst}); beyond tolerance: " + (', '.join(f'{short(n)} {v[n]:.1e}' for n in over) or 'none')
                      + f"; largest among the others {max([v[n] for n in v if n not in over] or [0]):.1e}")
    for dt in ('float32', 'float64'):
        r = js(E38 + f'e38_check2_diag_S_{SEED}_{dt}.json')
        print(f"   diagnostic, rule S, {dt}: commit {r['provenance']['git']['commit'][:8]}, cpu {r['cpu_seconds']:.0f} s")
        for key, v in r['max_rel'].items():
            mx = {int(g): max(t.values()) for g, t in v.items()}
            line = ' '.join(f'{g}:{x:.1e}' for g, x in mx.items())
            tol = 1e-5 if key.startswith('torch') else 1e-6
            jump = [g for g in mx if mx[g] > tol]
            first = ''
            if jump:
                g = str(jump[0]); t = v[g]; big = sorted((n for n in t if t[n] > tol), key=lambda n: -t[n])
                first = f"  | first stop beyond {tol:g}: update {g}; tensors beyond it: " + ', '.join(f'{short(n)} {t[n]:.1e}' for n in big) \
                        + f"; the other {len(t) - len(big)} at most {max(t[n] for n in t if n not in big):.1e}"
            print(f"      {key}: largest over tensors by update  {line}{first}")
    shas = {}
    for arm in ARMS:
        for rule in RULES:
            st = js(E38 + f'e38_{arm}_{rule}_{SEED}.json')['stages'][0]['settings']
            shas[(arm, rule)] = (st['provenance']['sources_sha256']['exp38_erase.py'][:12], st['provenance']['git']['commit'][:8], st['pull_form'], st['decay_target'], st['decay_strength'])
    print('   the ten lives: exp38_erase.py sha, commit, pull form, decay target, strength')
    for k, v in shas.items():
        print(f'      {k[0]:9s} {k[1]}: {v}')


# ================================================================ scoring with the reviewer's reader
def exact_counts(W, items):
    """Whole answers right.  Teacher-forced exactness of every answer byte equals greedy decoding of the whole answer (exp11_cell.respond
    stops at the end symbol and exp16_integrated.score compares whole lists).  The forward is np_gru.forward's arithmetic with the maps
    that depend on the input byte alone computed once per byte (as wrap_split in e36_e38_brief_checks.py); part B checks it against
    np_gru.forward itself on one file."""
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    ok = np.zeros(len(items), bool); by_len = {}
    for i, (p, t) in enumerate(items):
        by_len.setdefault((len(p), len(t)), []).append(i)
    for (lp, lt), ids in by_len.items():
        x = np.array([[BOS] + items[i][0] + items[i][1][:-1] for i in ids], dtype=np.int64)
        tgt = np.array([items[i][1] for i in ids], dtype=np.int64)
        h = np.zeros((len(ids), U), np.float32); good = np.ones(len(ids), bool); L = x.shape[1]
        for t in range(L):
            gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
            r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
            h = (1 - z) * n + z * h
            if t >= L - lt:
                good &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - lt)]
        ok[ids] = good
    return ok


def exact_counts_reader(W, items, chunk=1024):
    """The same count through np_gru.forward and np_gru.exact_teacher_forced, unchanged (slower)."""
    ok = np.zeros(len(items), bool)
    for j in range(0, len(items), chunk):
        part = items[j:j + chunk]
        x = np.array([[BOS] + p + t[:-1] for p, t in part], dtype=np.int64); tgt = np.array([t for _, t in part], dtype=np.int64)
        ok[j:j + chunk] = exact_teacher_forced(fwd32(W, x)[3], tgt)
    return ok


def all_pairs(task):
    op, m, _ = task
    items = [render(task, a, b, 0, None) for a in range(m) for b in range(m)]
    wrap = np.array([(a + b >= m) if op == 'add' else (a - b < 0) for a in range(m) for b in range(m)])
    return items, wrap


if 'B' in want or 'C' in want:
    held = {f'{t[0]}{t[1]}': panel(t, split(t, .5, SEED)[1], SEED, (12,), 300)[12] for t in DEV}                 # exp16_integrated.panels
    new = {f'{t[0]}{t[1]}': panel(t, split(t, .25, SEED)[1], SEED, (0, 8, 12, 20, 40), 400)[12] for t in LIFE[:2]}
    pairs = {f'{t[0]}{t[1]}': all_pairs(t) for t in DEV[:2]}
    birth = {rule: load_weights(E26 + f'e26_dev_{rule}_{SEED}.pt', REF) for rule in RULES}

if 'B' in want:
    print('B. scores recomputed from the committed weight files, beside the receipts (whole counts)')
    print('   columns: wrap pairs add89, sub89 (of 3,916 each); no-wrap add89, sub89 (of 4,005); held-out at 12 letters add89 sub89 max89 min89 (of 300); add97 sub97 (of 400)')
    n_files = n_equal = n_sha = 0; worst = 0; table = {}
    for rule in RULES:
        W0, sha0 = birth[rule]
        rec0 = js(E38 + f'e38_zero_{rule}_{SEED}.json')
        r0 = rec0['rows'][0]; st = rec0['stages'][0]['settings']
        ok0 = {n: exact_counts(W0, pairs[n][0]) for n in ('add89', 'sub89')}
        mine0 = [int(ok0[n][pairs[n][1]].sum()) for n in ('add89', 'sub89')]
        same_reader = all((exact_counts_reader(W0, pairs[n][0]) == ok0[n]).all() for n in ('add89', 'sub89')) and \
            all((exact_counts_reader(W0, it) == exact_counts(W0, it)).all() for it in list(held.values()) + list(new.values()))
        print(f"   birth {rule}: file sha256 equals the receipt's birth_sha256: {sha0 == st['birth_sha256']}; wrap pairs at row 0, mine {mine0}, receipt "
              f"{[r0['birth_allpairs_k0'][n]['wrap']['exact'] for n in ('add89', 'sub89')]}; my forward and np_gru.forward agree item by item on all eight panels: {same_reader}")
        for arm in ARMS:
            rec = js(E38 + f'e38_{arm}_{rule}_{SEED}.json'); rows = {r['updates']: r for r in rec['rows']}
            for row in ROWS:
                W, sha = load_weights(E38 + f'e38_{arm}_{rule}_{SEED}_row{row}.pt', REF); r = rows[row]
                ok = {n: exact_counts(W, pairs[n][0]) for n in ('add89', 'sub89')}
                mine = [int(ok[n][pairs[n][1]].sum()) for n in ('add89', 'sub89')] + [int(ok[n][~pairs[n][1]].sum()) for n in ('add89', 'sub89')]
                theirs = [r['birth_allpairs_k0'][n]['wrap']['exact'] for n in ('add89', 'sub89')] + [r['birth_allpairs_k0'][n]['nowrap']['exact'] for n in ('add89', 'sub89')]
                for n in ('add89', 'sub89', 'max89', 'min89'):
                    mine.append(int(exact_counts(W, held[n]).sum())); theirs.append(r['birth_heldout_k12'][n]['exact'])
                for n in ('add97', 'sub97'):
                    mine.append(int(exact_counts(W, new[n]).sum())); theirs.append(r['new_heldout_k12'][n]['exact'])
                same = mine == theirs; n_files += 1; n_equal += same; n_sha += sha == r['weights_sha256']
                worst = max(worst, max(abs(a - b) for a, b in zip(mine, theirs)))
                table[(arm, rule, row)] = mine
                print(f"   {arm:9s} {rule} row {row:6d}: sha256 of the file equals the receipt's: {sha == r['weights_sha256']}; mine {mine} "
                      + ('= receipt' if same else f'!= receipt {theirs}'))
                del W
    print(f"   files read {n_files}; file hash equal to the receipt's in {n_sha}; all ten counts equal to the receipt's in {n_equal}; largest difference in any count {worst}")
    print("   the report's table of marks at row 12,288, recomputed: held = mean of add89 and sub89 held out; wrap = mean of the two wrap shares; new = mean of add97 and sub97")
    for arm in ARMS:
        cells = []
        for rule in RULES:
            c = table[(arm, rule, 12288)]
            cells.append(f"{rule}: {100 * (c[4] + c[5]) / 600:.2f} / {100 * (c[0] + c[1]) / 7832:.2f} / {100 * (c[8] + c[9]) / 800:.2f}")
        print(f"      {arm:9s} " + '   '.join(cells))


# ================================================================ C. distance from the birth
if 'C' in want:
    print('C. distance of each life from its birth, per tensor: largest |w - w_birth| at row 6,144 and at row 12,288')
    print('   a pull of strength k bounds it near 1/k once the pull has acted for several times 1/(rate * k) updates (0.4 for anchor10, 4 for anchor);')
    print('   without a pull the bound is rate * updates (6.1 and 12.3)')
    names = PULLED + ('book.weight',)
    for rule in RULES:
        W0 = birth[rule][0]
        print(f'   rule {rule}:  ' + ' '.join(f'{short(n):>11s}' for n in names) + '   | all pulled tensors: largest, root mean square, share of weights with another sign than at birth')
        for arm in ARMS:
            for row in ROWS:
                W = load_weights(E38 + f'e38_{arm}_{rule}_{SEED}_row{row}.pt', REF)[0]
                d = {n: W[n].astype(np.float64) - W0[n] for n in names}
                allp = np.concatenate([d[n].ravel() for n in PULLED])
                flips = np.concatenate([(np.sign(W[n]) != np.sign(W0[n])).ravel() for n in PULLED]).mean()
                print(f'      {arm:9s} {row:6d} ' + ' '.join(f'{np.abs(d[n]).max():11.4f}' for n in names)
                      + f'   | {np.abs(allp).max():.4f}  {np.sqrt((allp ** 2).mean()):.5f}  {100 * flips:.2f}%')
                del W


# ================================================================ D. the gradient under both rules, in numpy
def entry_stream(seed):
    """exp16_integrated.entry_stream(family_entries('F1', seed), seed + 101): the stream of the first block of a life (the audit's part C)."""
    ent = [(t, a, b) for t in LIFE for a, b in split(t, .25, seed)[0]]
    rng = Random(seed + 101 + 77)
    while True:
        order = list(range(len(ent))); rng.shuffle(order)
        for i in order:
            t, a, b = ent[i]
            yield render(t, a, b, rng.randint(0, KMAX), rng)


def batch_tensor(items):
    """exp45.batch_tensor: x, y, answer mask; rows padded with the start symbol."""
    rows = [p + t for p, t in items]; L = max(map(len, rows)) + 1
    data = np.full((len(rows), L), BOS, np.int64)
    for i, r in enumerate(rows): data[i, 1:len(r) + 1] = r
    mask = np.zeros((len(rows), L - 1), bool)
    for i, (p, t) in enumerate(items): mask[i, len(p):len(p) + len(t)] = True
    return np.ascontiguousarray(data[:, :-1]), np.ascontiguousarray(data[:, 1:]), mask


def sig(v):
    with np.errstate(over='ignore'):
        return 1.0 / (1.0 + np.exp(-v))


def loss_and_grad(W, batch, rule, dtype=np.float64, grad=True):
    """arch26.forward(m, x, local, True) and the backward pass of exp26_candidate.step's loss, written out.
    rule 'B': every map sends the error back through its weights.  rule 'S': through sign(W) * mean|W| (exp3_state.SignLinear)."""
    x, y, mask = batch
    w = {k: v.astype(dtype) for k, v in W.items()}
    fb = (lambda A: A) if rule == 'B' else (lambda A: np.sign(A) * np.abs(A).mean(dtype=dtype))
    N, L = x.shape; U = w['rnn.weight_hh_l0'].shape[1]
    u = w['book.weight'][x]
    z1 = np.tanh(u @ w['l1.weight'].T + w['l1.bias'])
    gi = z1 @ w['rnn.weight_ih_l0'].T + w['rnn.bias_ih_l0']
    Whh, bhh = w['rnn.weight_hh_l0'], w['rnn.bias_hh_l0']
    h = np.zeros((N, U), dtype); K = {k: np.empty((N, L, U), dtype) for k in ('hp', 'z', 'r', 'n', 'ghn', 'h')}
    for t in range(L):
        gh = h @ Whh.T + bhh
        r = sig(gi[:, t, :U] + gh[:, :U]); z = sig(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        K['hp'][:, t] = h; h = (1 - z) * n + z * h
        for k, v in (('z', z), ('r', r), ('n', n), ('ghn', gh[:, 2 * U:]), ('h', h)): K[k][:, t] = v
    z2 = np.tanh(z1 @ w['l2.weight'].T + w['l2.bias'] + K['h'] @ w['lm.weight'].T + w['lm.bias'])
    lg = z2 @ w['l3.weight'].T + w['l3.bias']
    lg = lg - lg.max(-1, keepdims=True); p = np.exp(lg); p /= p.sum(-1, keepdims=True)
    nm = int(mask.sum()); yy = np.where(mask, y, 0)[..., None]                       # padded positions carry the start symbol and no loss
    loss = float(-np.log(np.take_along_axis(p, yy, -1)[..., 0][mask]).sum() / nm)
    if not grad:
        return loss
    d = p.copy(); np.put_along_axis(d, yy, np.take_along_axis(d, yy, -1) - 1, -1)
    d = (d * mask[..., None] / nm).astype(dtype)                                    # dLoss/dlogits
    two = lambda a: a.reshape(-1, a.shape[-1])
    G = {}
    G['l3.weight'] = two(d).T @ two(z2); G['l3.bias'] = two(d).sum(0)
    da2 = (d @ fb(w['l3.weight'])) * (1 - z2 ** 2)
    G['l2.weight'] = two(da2).T @ two(z1); G['l2.bias'] = two(da2).sum(0)
    G['lm.weight'] = two(da2).T @ two(K['h']); G['lm.bias'] = two(da2).sum(0)
    dz1 = da2 @ fb(w['l2.weight'])
    dH = da2 @ fb(w['lm.weight'])
    Shh = fb(Whh); dgi = np.empty((N, L, 3 * U), dtype); G['rnn.weight_hh_l0'] = np.zeros_like(Whh); G['rnn.bias_hh_l0'] = np.zeros_like(bhh); nxt = 0.0
    for t in reversed(range(L)):
        dh = dH[:, t] + nxt
        z, n, r, hp, ghn = (K[k][:, t] for k in ('z', 'n', 'r', 'hp', 'ghn'))
        dan = dh * (1 - z) * (1 - n ** 2); daz = dh * (hp - n) * z * (1 - z); dar = dan * ghn * r * (1 - r)
        dgi[:, t] = np.concatenate([dar, daz, dan], 1)
        dgh = np.concatenate([dar, daz, dan * r], 1)
        G['rnn.weight_hh_l0'] += dgh.T @ hp; G['rnn.bias_hh_l0'] += dgh.sum(0)
        nxt = dh * z + dgh @ Shh
    G['rnn.weight_ih_l0'] = two(dgi).T @ two(z1); G['rnn.bias_ih_l0'] = two(dgi).sum(0)
    dz1 = dz1 + dgi @ fb(w['rnn.weight_ih_l0'])
    da1 = dz1 * (1 - z1 ** 2)
    G['l1.weight'] = two(da1).T @ two(u); G['l1.bias'] = two(da1).sum(0)
    du = da1 @ fb(w['l1.weight'])
    gb = np.zeros_like(w['book.weight']); np.add.at(gb, x.ravel(), two(du)); gb[BOS] = 0          # padding_idx = 257: that row has no gradient
    G['book.weight'] = gb
    return loss, G, dict(da1=da1)


def rel(a, b):
    """exp38_erase.max_rel for one tensor."""
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max() / max(np.abs(a).max(), 1e-30))


def adam_step(Wt, G, state, t, kappa, form, W0=None, dtype=np.float32):
    """One update on group 0 and on the byte table, in torch.optim.AdamW's order (decay or pull, then the Adam step), in `dtype`.
    form 'torch': p *= 1 - rate * kappa (the optimizer's own decay).  form 1: p -= rate * kappa * (p - p0).  form 2: p = p0 + (1 - rate * kappa) * (p - p0).
    form None: neither.  The byte table has no decay and no pull.  state: {name: (m, v)}; it is replaced, not changed in place."""
    out, ns = {}, {}
    f = dtype
    for n, p in Wt.items():
        p = p.astype(f); g = G[n].astype(f)
        if n != 'book.weight' and form is not None:
            p0 = np.zeros_like(p) if W0 is None else W0[n].astype(f)
            if form == 'torch': p = p * f(1 - LR * kappa)
            elif form == 1: p = p - f(LR * kappa) * (p - p0)
            else: p = p0 + f(1 - LR * kappa) * (p - p0)
        m, v = state.get(n, (np.zeros_like(p), np.zeros_like(p)))
        m = m + (g - m) * f(1 - B1)                                              # exp_avg.lerp_(grad, 1 - beta1)
        v = v * f(B2) + f(1 - B2) * g * g
        denom = np.sqrt(v) / f(np.sqrt(1 - B2 ** t)) + f(EPS)
        out[n] = (p - f(LR / (1 - B1 ** t)) * (m / denom)).astype(f); ns[n] = (m, v)
    return out, ns


if 'D' in want:
    print('D. the first two batches of the life, from the birth file of rule S (and of rule B where said)')
    # ---- D1. the stream
    st = entry_stream(SEED); h = hashlib.sha256(); batches = []
    for i in range(256):
        b = batch_tensor([next(st) for _ in range(BATCH)])
        if i < 2: batches.append(b)
        for t in b: h.update(str(tuple(t.shape)).encode()); h.update(t.tobytes())
    dry = js(E38 + f'dry/e38_zero_S_{SEED}_dry256.json')['stages'][1]['data_digest']
    print(f"   D1. my rebuild of the first 256 batches has the digest of the worker's dry life (e38_zero_S_{SEED}_dry256.json): {h.hexdigest() == dry['sha256']} ({dry['batches']} batches)")
    b1, b2 = batches
    print(f"       batch 1: {b1[0].shape[0]} items, {b1[0].shape[1]} positions, {int(b1[2].sum())} answer bytes under the loss")
    WS = load_weights(E26 + f'e26_dev_S_{SEED}.pt', REF)[0]; WB = load_weights(E26 + f'e26_dev_B_{SEED}.pt', REF)[0]

    # ---- D2. the backward pass against finite differences (exact rule), float64
    loss, GB, _ = loss_and_grad(WS, b1, 'B'); _, GS, _ = loss_and_grad(WS, b1, 'S')
    rng = np.random.default_rng(0); worst = 0.0; worst_top = 0.0; n_fd = 0
    for n in WS:
        flat = np.abs(GB[n]).ravel(); top = np.argsort(-flat)[:2].tolist(); cand = top + rng.integers(0, flat.size, 2).tolist()
        for i in cand:
            idx = np.unravel_index(i, WS[n].shape)
            if n == 'book.weight' and idx[0] == BOS: continue
            hstep = 1e-5
            Wp = {k: v.astype(np.float64) for k, v in WS.items()}; Wp[n][idx] += hstep; lp = loss_and_grad(Wp, b1, 'B', grad=False)
            Wp[n][idx] -= 2 * hstep; lm = loss_and_grad(Wp, b1, 'B', grad=False)
            fd = (lp - lm) / (2 * hstep); n_fd += 1
            worst = max(worst, abs(fd - GB[n][idx]) / flat.max())
            if i in top: worst_top = max(worst_top, abs(fd - GB[n][idx]) / abs(GB[n][idx]))
    cosine = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    print(f"   D2. loss on batch 1 at the sweep birth {loss:.6f}; exact gradient against central differences on {n_fd} entries (the two largest and two random ones of each tensor): largest error {worst:.1e} of the tensor's largest gradient entry; on the largest entries, {worst_top:.1e} of the entry itself")
    print('       the sweep gradient against the exact one at the same weights (cosine per tensor): ' + ', '.join(f'{short(n)} {cosine(GS[n], GB[n]):.3f}' for n in WS))

    # ---- D3. a difference of rounding size that changes no sign
    Wq = {n: (v.astype(np.float64) * (1 + 1e-7 * rng.standard_normal(v.shape))) for n, v in WS.items()}
    assert all((np.sign(Wq[n]) == np.sign(WS[n])).all() for n in WS)
    for rule, G in (('B', GB), ('S', GS)):
        Gq = loss_and_grad(Wq, b1, rule)[1]
        print(f"   D3. every weight changed by a relative 1e-7 (no sign changes), rule {rule}: relative difference of the gradient, largest over tensors "
              f"{max(rel(G[n], Gq[n]) for n in WS):.1e}")

    # ---- D4. one weight next to zero takes the other sign
    print('   D4. one weight next to zero given the other sign (the weight set changes by twice its size); relative difference of the gradient per tensor')
    for name in ('l2.weight', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'lm.weight', 'l3.weight', 'l1.weight'):
        i = np.unravel_index(np.abs(WS[name]).argmin(), WS[name].shape); val = float(WS[name][i])
        Wf = {n: v.astype(np.float64) for n, v in WS.items()}; Wf[name][i] = -val
        print(f"       {name}{list(map(int, i))} = {val:+.3e}  (change {2 * abs(val):.1e}, {2 * abs(val) / np.abs(WS[name]).max():.1e} of the tensor's largest entry)")
        for rule, G in (('B', GB), ('S', GS)):
            Gf = loss_and_grad(Wf, b1, rule)[1]; r = {n: rel(G[n], Gf[n]) for n in WS}
            big = sorted((n for n in r if r[n] > 1e-5), key=lambda n: -r[n])
            line = ', '.join(f'{short(n)} {r[n]:.1e}' for n in big) or 'none'
            extra = ''
            if rule == 'S' and name in ('l2.weight', 'rnn.weight_ih_l0'):
                j = int(i[1]); row = np.linalg.norm(Gf['l1.weight'][j] - G['l1.weight'][j]) / np.linalg.norm(G['l1.weight'][j])
                extra = f"; row {j} of l1.weight changes by {100 * row:.1f}% of its own length, l1.bias[{j}] by {100 * abs(Gf['l1.bias'][j] - G['l1.bias'][j]) / abs(G['l1.bias'][j]):.1f}%"
            print(f"          rule {rule}: tensors beyond 1e-5: {line}; largest among the others {max([r[n] for n in r if n not in big] or [0]):.1e}{extra}")

    # ---- D5. the arithmetic of the forms in float32, and how near zero the weights lie
    print('   D5. the forms in float32 (my arithmetic in numpy, in the order of torch.optim.AdamW): relative difference per tensor, largest over the 12 pulled tensors')
    G1 = {r: loss_and_grad(W, b1, r, np.float32)[1] for r, W in (('S', WS), ('B', WB))}
    for rule, W in (('S', WS), ('B', WB)):
        for kappa in (0.25, 2.5):
            a1, sa = adam_step(W, G1[rule], {}, 1, kappa, 'torch'); p1, sp = adam_step(W, G1[rule], {}, 1, kappa, 1)
            d1 = max(rel(a1[n], p1[n]) for n in PULLED)
            ga, gp = loss_and_grad(a1, b2, rule, np.float32)[1], loss_and_grad(p1, b2, rule, np.float32)[1]
            a2, _ = adam_step(a1, ga, sa, 2, kappa, 'torch'); p2, _ = adam_step(p1, gp, sp, 2, kappa, 1)
            d2 = {n: rel(a2[n], p2[n]) for n in PULLED}
            disc = sum(int((np.sign(a1[n]) != np.sign(p1[n])).sum()) for n in BACK)
            # check 3's pair: birth = the birth file, the two forms of the pull at update 2 (at update 1 the pull is zero in both)
            n1, sn = adam_step(W, G1[rule], {}, 1, kappa, 1, W0=W)
            gn = loss_and_grad(n1, b2, rule, np.float32)[1]
            f1, _ = adam_step(n1, gn, sn, 2, kappa, 1, W0=W); f2, _ = adam_step(n1, gn, sn, 2, kappa, 2, W0=W)
            d3 = max(rel(f1[n], f2[n]) for n in PULLED)
            wn = max(d2, key=d2.get)
            print(f"       rule {rule}, strength {kappa}: the optimizer's decay against the pull to zero: after update 1 {d1:.1e}, after update 2 {d2[wn]:.1e} ({short(wn)}); "
                  f"weights of the six maps with unequal signs in the two after update 1: {disc}; the two forms of the pull to the birth after update 2: {d3:.1e}")
            if rule == 'S' and kappa == 0.25:
                keep = dict(a1=a1, p1=p1, sa=sa, sp=sp, a2=a2, p2=p2)
    a1, p1, a2, p2 = keep['a1'], keep['p1'], keep['a2'], keep['p2']
    print('       rule S, strength 0.25, the six maps the error crosses (1,118,720 weights): how near zero they lie, and how far apart the two runs are')
    cross = sum(int((np.sign(a1[n]) != np.sign(WS[n])).sum()) for n in BACK)
    absw = np.concatenate([np.abs(a1[n]).ravel() for n in BACK]); tot = absw.size
    print(f"          weights that changed sign in update 1: {cross}; weights within 1e-3 / 1e-4 / 1e-5 / 1e-6 / 1e-7 / 1e-8 of zero after it: "
          + ' / '.join(str(int((absw < x).sum())) for x in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)) + f' of {tot}')
    dens = (absw < 1e-4).sum() / 2e-4
    for lab, x, y in (('after update 1', a1, p1), ('after update 2', a2, p2)):
        dd = np.concatenate([np.abs(x[n].astype(np.float64) - y[n]).ravel() for n in BACK]); small = np.concatenate([np.abs(x[n]).ravel() for n in BACK]) < 1e-3
        disc = sum(int((np.sign(x[n]) != np.sign(y[n])).sum()) for n in BACK)
        print(f"          {lab}: |difference between the two runs| per weight: mean {dd.mean():.2e}, median {np.median(dd):.2e}, largest {dd.max():.2e}; "
              f"among weights within 1e-3 of zero: mean {dd[small].mean():.2e}, largest {dd[small].max():.2e}; "
              f"weights with unequal signs {disc}; expected number from density at zero ({dens:.2e} per unit) times that mean: {dens * dd[small].mean():.1e}")

    # ---- D6. one weight with unequal signs in the two runs at update 2: what the weights look like after that update
    print('   D6. update 2 from the weights after update 1 (strength 0.25, the optimizer\'s own decay), with one weight next to zero given the other sign in a copy')
    for name in ('l2.weight', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0'):
        i = np.unravel_index(np.abs(a1[name]).argmin(), a1[name].shape); val = float(a1[name][i])
        c1 = {n: v.copy() for n, v in a1.items()}; c1[name][i] = -val
        print(f"       {name}{list(map(int, i))} = {val:+.3e} against {-val:+.3e}: before the update the two sets differ by {rel(a1[name], c1[name]):.1e} in that tensor and by 0 elsewhere")
        for rule in ('B', 'S'):
            g1 = loss_and_grad(WS, b1, rule, np.float32)[1]; w1, s1 = adam_step(WS, g1, {}, 1, 0.25, 'torch')    # the optimizer's averages after update 1 under this rule
            ga, gc = loss_and_grad(a1, b2, rule, np.float32)[1], loss_and_grad(c1, b2, rule, np.float32)[1]
            ua, _ = adam_step(a1, ga, s1, 2, 0.25, 'torch'); uc, _ = adam_step(c1, gc, s1, 2, 0.25, 'torch')
            r = {n: rel(ua[n], uc[n]) for n in WS}; big = sorted((n for n in r if r[n] > 1e-5), key=lambda n: -r[n])
            print(f"          rule {rule}: after the update, tensors beyond 1e-5: " + (', '.join(f'{short(n)} {r[n]:.1e}' for n in big) or 'none')
                  + f"; largest among the others {max([r[n] for n in r if n not in big] or [0]):.1e}")

print(f'cpu {time.process_time() - t00:.0f} s')
