"""Experiment 1: development-versus-life transfer on modular arithmetic.

Development: train the shared network on mod-97 addition (Codex's taught split) with backprop,
AdamW lr .001, decoupled decay 0.3 (the best regularizer from the 6 Nov sweep). A matched
"shuffled" birth gets the same budget with answer bytes permuted across taught prompts.

Life: a new task (mod-97 subtraction `a-b=c`, or mod-89 addition) learned with a LOCAL rule only.

  A1  aligned birth, embedding + first layer frozen, second layer + readout learned by sign-symmetric feedback
  A2  aligned birth, all layers learned by sign-symmetric feedback (warm start)
  B   shuffled birth, as A1
  C   random birth, all layers learned by sign-symmetric feedback
  D   random birth, backprop (reference ceiling), AdamW decay 0.3
  E   aligned birth, trunk frozen, second layer + readout learned by a three-factor rule:
      scalar reward per record (fraction of the three answer symbols sampled correctly),
      running-mean baseline, eligibility = node perturbation on the hidden layer and
      (sampled - expected) x presynaptic on the readout. AdamW decay 1.0 like the sign arms.
  E0  as E with no weight decay (sanity arm for the reward rule)

Local life arms use AdamW lr .001, decoupled decay 1.0 (the "signw10" setting).
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from itertools import islice
from pathlib import Path
from random import Random

import torch as T
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod97_local as M

T.set_num_threads(1)
EOS, BOS = M.EOS, M.BOS
SEEDS = M.SEEDS
BATCH, BLOCK = 64, 512
OUT = Path(__file__).resolve().parent / "exp1"
DEV_UPDATES = 12288
LIFE_UPDATES = 12288
LIFE_ARMS = ("A1", "A2", "B", "C", "D", "E", "E0")
TASKS = ("sub97", "add89")


# ---------------------------------------------------------------- life tasks
def render(task, a, b):
    if task == "sub97":
        return list(f"{a:02d}-{b:02d}=".encode()) + list(f"{(a - b) % 97:02d}".encode()) + [EOS]
    if task == "add89":
        return list(f"{a:02d}+{b:02d}=".encode()) + list(f"{(a + b) % 89:02d}".encode()) + [EOS]
    raise ValueError(task)


def modulus(task):
    return 97 if task == "sub97" else 89


def life_split(task, frac, seed):
    """Taught = frac of unordered pairs; both orders stay together. Rest halved into monitor / sealed."""
    m = modulus(task)
    groups = [(a, b) for a in range(m) for b in range(a, m)]
    rng = Random(seed * 1000 + round(frac * 100) + (0 if task == "sub97" else 7))
    rng.shuffle(groups)
    k = round(frac * len(groups))
    taught, rest = groups[:k], groups[k:]
    monitor, sealed = rest[0::2], rest[1::2]
    expand = lambda gs: [p for a, b in gs for p in (((a, b), (b, a)) if a != b else ((a, b),))]
    return expand(sorted(taught)), expand(sorted(monitor)), expand(sorted(sealed))


def stream(rows, seed):
    order = Random(seed + 1)
    chunk = []
    while True:
        for row in order.sample(rows, len(rows)):
            chunk.append(row)
            if len(chunk) == BATCH:
                yield chunk
                chunk = []


def score(m, task, pairs):
    rows = [render(task, *p) for p in pairs]
    prompts = [r[:6] for r in rows]
    gold = [r[6:] for r in rows]
    outs = M.respond(m, prompts)
    exact = {p: o == g for p, o, g in zip(pairs, outs, gold)}
    mod = modulus(task)
    if task == "sub97":
        wrap = lambda a, b: a < b
        carry = lambda a, b: a % 10 < b % 10          # decimal borrow
    else:
        wrap = lambda a, b: a + b >= mod
        carry = lambda a, b: a % 10 + b % 10 >= 10
    return dict(n=len(pairs), exact=sum(exact.values()),
                first=sum(o[:1] == g[:1] for o, g in zip(outs, gold)),
                eos=sum(EOS in o for o in outs),
                both_orders=sum(exact[a, b] and exact[b, a] for a, b in pairs if a < b),
                both_order_n=sum(a < b for a, b in pairs),
                wrap=[sum(exact[a, b] for a, b in pairs if wrap(a, b)), sum(wrap(a, b) for a, b in pairs)],
                carry=[sum(exact[a, b] for a, b in pairs if carry(a, b)), sum(carry(a, b) for a, b in pairs)])


def shortcuts(task, taught, sealed):
    """Ceilings on the sealed panel for simple non-rule strategies."""
    mod = modulus(task)
    ans = lambda a, b: (a - b) % mod if task == "sub97" else (a + b) % mod
    taught_set = set(taught)
    canon = {(min(a, b), max(a, b)) for a, b in taught}
    from collections import Counter
    freq = Counter(ans(a, b) for a, b in taught).most_common(1)[0][0]
    out = dict(n=len(sealed), exact_lookup=sum(p in taught_set for p in sealed),
               canonical_lookup=sum((min(a, b), max(a, b)) in canon for a, b in sealed),
               frequency=sum(ans(a, b) == freq for a, b in sealed))
    # nearest taught pair by L1 distance in operand space
    tl = taught
    near = 0
    for a, b in sealed:
        best = min(tl, key=lambda q: abs(q[0] - a) + abs(q[1] - b))
        near += ans(*best) == ans(a, b)
    out["nearest_taught"] = near
    if task == "sub97":
        out["addition_answer"] = sum((a + b) % mod == ans(a, b) for a, b in sealed)
        out["no_wrap_abs_diff"] = sum(abs(a - b) == ans(a, b) for a, b in sealed)
    else:
        out["no_wrap"] = sum(a + b == ans(a, b) for a, b in sealed)
        out["mod97_answer"] = sum((a + b) % 97 == ans(a, b) for a, b in sealed)
    return out


# ---------------------------------------------------------------- development
def dev_rows(seed, shuffled):
    train, _, _ = M.panels(seed)
    rows = [M.arithmetic(p) for p in train]
    if shuffled:
        rng = Random(seed + 4242)
        answers = [r[6:8] for r in rows]
        rng.shuffle(answers)
        rows = [r[:6] + a + [EOS] for r, a in zip(rows, answers)]
    return rows


def develop(seed, shuffled):
    OUT.mkdir(exist_ok=True)
    tag = "shuffled" if shuffled else "aligned"
    path = OUT / f"dev_{tag}_{seed}.pt"
    if path.exists():
        return
    T.manual_seed(seed)
    m = M.Net()
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    rows = dev_rows(seed, shuffled)
    F = dict(drop=0., rng=T.Generator().manual_seed(seed + 7))
    train, monitor, sealed = M.panels(seed)
    t0 = time.process_time()
    curve = []
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = [M.update_bp(m, F, opt, r) for r in islice(stream(rows, seed), BLOCK)]
        if stop % 2048 == 0 or stop == DEV_UPDATES:
            curve.append(dict(step=stop, loss=sum(losses) / len(losses), taught=M.score(m, train),
                              sealed=M.score(m, sealed), cpu=time.process_time() - t0))
            print("dev", tag, seed, stop, curve[-1]["taught"]["exact"], curve[-1]["sealed"]["exact"], flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"dev_{tag}_{seed}.json").write_text(json.dumps(dict(seed=seed, birth=tag, curve=curve), indent=1))


# ---------------------------------------------------------------- life rules
def sign_update(m, F, opt, rows, plastic):
    g, loss = M.local_grads(m, F, rows, freeze_emb="book.weight" not in plastic)
    opt.zero_grad()
    for name, p in m.named_parameters():
        p.grad = g[name] if name in plastic and name in g else None
    opt.step()
    return loss


def bp_update(m, F, opt, rows):
    return M.update_bp(m, F, opt, rows)


class ThreeFactor:
    """Scalar-reward rule on l2 (node perturbation) and l3 (REINFORCE on sampled output)."""
    def __init__(self, sigma=.1, baseline_rate=.02, seed=0):
        self.sigma, self.rate = sigma, baseline_rate
        self.baseline = 0.
        self.rng = T.Generator().manual_seed(seed + 99)

    @T.no_grad()
    def update(self, m, opt, rows):
        data = M.rows_tensor(rows)
        x, y = data[:, :-1], data[:, 1:]
        u = m.window(x)
        z1 = T.tanh(m.l1(u))
        a2 = m.l2(z1)
        xi = T.randn(a2.shape, generator=self.rng) * self.sigma
        z2 = T.tanh(a2 + xi)
        logits = m.l3(z2)
        ans = slice(6, 9)                                   # answer byte, answer byte, EOS
        p = logits[:, ans].softmax(-1)                      # B x 3 x 257
        s = T.multinomial(p.flatten(0, 1), 1, generator=self.rng).reshape(p.shape[:2])
        R = (s == y[:, ans]).float().mean(1)                # scalar reward per record in [0,1]
        adv = R - self.baseline
        self.baseline += self.rate * (R.mean().item() - self.baseline)
        onehot = T.zeros_like(p).scatter_(2, s[..., None], 1.)
        e3 = ((onehot - p) * adv[:, None, None]).flatten(0, 1)          # (B*3) x 257
        g3w = -(e3.T @ z2[:, ans].flatten(0, 1)) / len(rows)
        g3b = -e3.sum(0) / len(rows)
        e2 = (xi[:, ans] * adv[:, None, None] / self.sigma ** 2).flatten(0, 1)   # (B*3) x 128
        g2w = -(e2.T @ z1[:, ans].flatten(0, 1)) / len(rows)
        g2b = -e2.sum(0) / len(rows)
        opt.zero_grad()
        for name, prm in m.named_parameters():
            prm.grad = {"l2.weight": g2w, "l2.bias": g2b, "l3.weight": g3w, "l3.bias": g3b}.get(name)
        opt.step()
        return float(R.mean())


def make_life(arm, seed, task):
    m = M.Net()
    T.manual_seed(seed + 11)
    birth = {"A1": "aligned", "A2": "aligned", "A1bp": "aligned", "B": "shuffled", "B2": "shuffled", "E": "aligned", "E0": "aligned"}.get(arm)
    if birth:
        m.load_state_dict(T.load(OUT / f"dev_{birth}_{seed}.pt"))
    else:
        T.manual_seed(seed)
        m = M.Net()
    all_p = [n for n, _ in m.named_parameters()]
    upper = [n for n in all_p if n.startswith("l2.") or n.startswith("l3.")]
    plastic = {"A1": upper, "A2": all_p, "A1bp": upper, "B": upper, "B2": all_p, "C": all_p, "C1": upper,
               "D": all_p, "E": upper, "E0": upper}[arm]
    params = [p for n, p in m.named_parameters() if n in plastic]
    for n, p in m.named_parameters():        # frozen parameters must not receive autograd grads in backprop arms
        p.requires_grad_(n in plastic)
    if arm in ("D", "A1bp"):
        opt = T.optim.AdamW(params, lr=.001, weight_decay=.3, betas=(.9, .98))
    elif arm == "E0":
        opt = T.optim.AdamW(params, lr=.001, weight_decay=0., betas=(.9, .98))
    else:
        opt = T.optim.AdamW(params, lr=.001, weight_decay=1., betas=(.9, .98))
    F = dict(mode="sign", drop=0., rng=T.Generator().manual_seed(seed + 7))
    return m, F, opt, plastic


def life(arm, seed, task, frac):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"life_{task}_f{round(frac * 100)}_{arm}_{seed}.json"
    taught, monitor, sealed = life_split(task, frac, seed)
    m, F, opt, plastic = make_life(arm, seed, task)
    rows = [render(task, *p) for p in taught]
    tf = ThreeFactor(seed=seed) if arm in ("E", "E0") else None
    before = score(m, task, sealed)
    t0 = time.process_time()
    curve = []
    batches = stream(rows, seed)
    for stop in range(BLOCK, LIFE_UPDATES + 1, BLOCK):
        if arm in ("D", "A1bp"):
            losses = [bp_update(m, F, opt, r) for r in islice(batches, BLOCK)]
        elif tf:
            losses = [tf.update(m, opt, r) for r in islice(batches, BLOCK)]
        else:
            losses = [sign_update(m, F, opt, r, plastic) for r in islice(batches, BLOCK)]
        row = dict(step=stop, signal=sum(losses) / len(losses), taught=score(m, task, taught),
                   monitor=score(m, task, monitor), sealed=score(m, task, sealed), cpu=time.process_time() - t0)
        curve.append(row)
        print(task, frac, arm, seed, stop, f"taught {row['taught']['exact']}/{row['taught']['n']}",
              f"sealed {row['sealed']['exact']}/{row['sealed']['n']}", f"cpu {row['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, task=task, frac=frac, taught_n=len(taught),
                                        plastic=plastic, sealed_before=before,
                                        shortcuts=shortcuts(task, taught, sealed), curve=curve), indent=1))


# ---------------------------------------------------------------- qualification
def qualify():
    out = {}
    for task in TASKS:
        for frac in (.1, .25, .5):
            t, mo, se = life_split(task, frac, SEEDS[0])
            m = modulus(task)
            assert len(t) + len(mo) + len(se) == m * m and not (set(t) & set(se)) and not (set(t) & set(mo))
            assert all((b, a) in se for a, b in se)
            assert all(int(bytes(render(task, a, b)[6:8])) == ((a - b) % 97 if task == "sub97" else (a + b) % 89) for a, b in t + se)
            out[f"{task}_f{frac}"] = dict(sizes=[len(t), len(mo), len(se)], shortcuts=shortcuts(task, t, se))
    # dev rows: shuffled keeps answer marginals, differs from aligned on most rows
    al, sh = dev_rows(SEEDS[0], False), dev_rows(SEEDS[0], True)
    from collections import Counter
    assert Counter(tuple(r[6:8]) for r in al) == Counter(tuple(r[6:8]) for r in sh)
    out["dev_shuffled_rows_changed"] = sum(a != b for a, b in zip(al, sh)) / len(al)
    # three-factor smoke: finite, plastic only l2/l3, trunk unchanged
    T.manual_seed(1); m = M.Net()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = T.optim.AdamW([p for n, p in m.named_parameters() if n.startswith(("l2.", "l3."))], lr=.001, weight_decay=1.)
    tf = ThreeFactor(seed=1)
    t, _, _ = life_split("sub97", .5, SEEDS[0])
    rows = [render("sub97", *p) for p in t]
    for r in islice(stream(rows, 1), 32):
        R = tf.update(m, opt, r)
    assert all(T.equal(before[n], p) for n, p in m.named_parameters() if n.startswith(("book", "l1.")))
    assert all(not T.equal(before[n], p) for n, p in m.named_parameters() if n.startswith(("l2.", "l3.")))
    out["three_factor_smoke_reward"] = R
    # sign rule with frozen trunk leaves trunk unchanged
    T.manual_seed(1); m = M.Net(); F = dict(mode="sign", drop=0., rng=T.Generator().manual_seed(8))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    upper = [n for n, _ in m.named_parameters() if n.startswith(("l2.", "l3."))]
    opt = T.optim.AdamW([p for n, p in m.named_parameters() if n in upper], lr=.001, weight_decay=1.)
    for r in islice(stream(rows, 1), 8):
        sign_update(m, F, opt, r, upper)
    assert all(T.equal(before[n], p) for n, p in m.named_parameters() if n not in upper)
    out["sign_frozen_trunk_ok"] = True
    OUT.mkdir(exist_ok=True)
    (OUT / "QUALIFY.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "qualify":
        qualify()
    elif cmd == "develop":
        develop(int(sys.argv[2]), sys.argv[3] == "shuffled")
    elif cmd == "life":
        life(sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
