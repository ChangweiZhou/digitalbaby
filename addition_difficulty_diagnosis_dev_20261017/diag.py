"""Why does a byte-in/byte-out recurrent learner memorise addition but not generalise it?

Development diagnostic only (not formal science, no adoption). Same task split as
addition_core_lineage_comparison_dev_v1_20261015 (digits 0-9, HELD = 10 commutative orbits never shown in
either order), same reference family (one GRU reads BEGIN, prompt bytes, REPLY, then emits answer bytes + EOS).

Factors varied, one at a time from the task as registered:
  input   digit  "3+8="            (the task)          | unary "|||+||||||||="   (ordinal structure given)
  output  dec    "11" EOS          (the task)          | sym   chr(65+sum) EOS     (no carry / mod-10 formatting)
  regime  online batch-1 RMSprop, 16,384 presentations (the lineage dose)
          | full-batch AdamW, long training, weight decay 0 or wd (simplicity pressure)
  aux     none (the task) | count: 20 extra records "3=" -> "|||" and "|||=" -> "3" for every digit (counting
          experience, still raw bytes; the HELD addition facts stay unseen) | count_shuf: the same records with the
          digit<->tally pairing permuted (same bytes and record count, no correct grounding)
The unary / sym formats are diagnostics that localise the difficulty; they are not proposed task encodings.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

EOS, BEGIN, REPLY = 256, 257, 258


def split(world):
    # Copied from addition_core_lineage_comparison_dev_v1_20261015/validate_design.py (identical draws).
    rng = random.Random(world)
    diagonal = [(d, d) for d in range(10)]
    original = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    for _ in range(1, 10001):
        off = original.copy()
        rng.shuffle(off)
        held, new, old = off[:10], off[10:20], diagonal + off[20:]
        answers = {a + b for a, b in old}
        degrees = [sum(d in pair for pair in old) for d in range(10)]
        if answers == set(range(19)) and min(degrees) >= 3:
            return dict(old=old, new=new, held=held)
    raise ValueError("split failed")


def ordered(orbits):
    return sorted({p for a, b in orbits for p in ((a, b), (b, a))})


def render(pair, inp, out):
    a, b = pair
    if inp == "digit":
        prompt = f"{a}+{b}=".encode()
    else:
        prompt = ("|" * a + "+" + "|" * b + "=").encode()
    s = a + b
    answer = str(s).encode() if out == "dec" else bytes([65 + s])
    return list(prompt), list(answer)


class Net(torch.nn.Module):
    # KEEP-sized: byte embedding 24, GRU width 64, softmax over 256 bytes + EOS.
    def __init__(self, seed, width=64):
        super().__init__()
        torch.manual_seed(seed)
        self.E = torch.nn.Embedding(259, 24)
        self.rnn = torch.nn.GRU(24, width, batch_first=True)
        self.V = torch.nn.Linear(width, 257)

    def forward(self, tokens, h=None):
        out, h = self.rnn(self.E(tokens), h)
        return self.V(out), h, out


def batch(items):
    # items: list of (prompt, answer). Sequence = BEGIN prompt REPLY answer; targets = answer EOS from REPLY on.
    seqs, tgts = [], []
    for p, a in items:
        seq = [BEGIN] + p + [REPLY] + a
        tgt = [-100] * (len(p) + 1) + a + [EOS]
        seqs.append(seq)
        tgts.append(tgt)
    n = max(map(len, seqs))
    x = torch.tensor([s + [0] * (n - len(s)) for s in seqs])
    y = torch.tensor([t + [-100] * (n - len(t)) for t in tgts])
    return x, y


@torch.no_grad()
def generate(net, items, cap=20):
    out = []
    for p, _ in items:
        logits, h, _ = net(torch.tensor([[BEGIN] + p + [REPLY]]))
        tok, got = int(logits[0, -1].argmax()), []
        while tok != EOS and len(got) < cap:
            got.append(tok)
            logits, h, _ = net(torch.tensor([[tok]]), h)
            tok = int(logits[0, -1].argmax())
        out.append(got)
    return out


def accuracy(net, items):
    gen = generate(net, items)
    return float(np.mean([g == a for g, (_, a) in zip(gen, items)])), gen


@torch.no_grad()
def probe(net, inp):
    """How ordered is the state after reading the first operand and '+'?  Spearman rho of the first principal
    component with operand value (|rho| = 1: a number line; ~0: unordered symbols)."""
    states = []
    for a in range(10):
        p = list(f"{a}+".encode()) if inp == "digit" else list(("|" * a + "+").encode())
        _, h, _ = net(torch.tensor([[BEGIN] + p]))
        states.append(h[0, 0].numpy())
    s = np.array(states) - np.mean(states, 0)
    pc = np.linalg.svd(s, full_matrices=False)[2][0] @ s.T
    r = np.argsort(np.argsort(pc))
    return float(abs(np.corrcoef(r, np.arange(10))[0, 1]))


def counting(world, shuffled):
    tally = list(range(10))
    if shuffled:
        rng = random.Random(world + 7)
        while any(t == d for d, t in enumerate(tally)):
            rng.shuffle(tally)
    recs = []
    for d, t in enumerate(tally):
        recs.append((list(f"{d}=".encode()), list(b"|" * t)))
        recs.append((list(b"|" * t + b"="), list(f"{d}".encode())))
    return recs


def run(world, seed, inp, out, regime, steps, wd, log_every, aux="none", width=64):
    sp = split(world)
    train_pairs = ordered(sp["old"]) + ordered(sp["new"])
    held_pairs = ordered(sp["held"])
    train = [render(p, inp, out) for p in train_pairs]
    held = [render(p, inp, out) for p in held_pairs]
    fit = train + (counting(world, aux == "count_shuf") if aux != "none" else [])
    net = Net(seed, width)
    curve = []
    t0 = time.process_time()
    if regime == "online":
        # Lineage dose: one record per step, RMSprop-style lr 1e-3, global-norm clip 1, shuffled epochs.
        opt = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.99, eps=1e-8)
        rng = random.Random(seed)
        order = []
        for step in range(1, steps + 1):
            if not order:
                order = list(range(len(fit)))
                rng.shuffle(order)
            x, y = batch([fit[order.pop()]])
            loss = F.cross_entropy(net(x)[0].reshape(-1, 257), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == steps:
                curve.append((step, accuracy(net, train)[0], accuracy(net, held)[0], float(loss.detach())))
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
        x, y = batch(fit)
        for step in range(1, steps + 1):
            loss = F.cross_entropy(net(x)[0].reshape(-1, 257), y.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % log_every == 0 or step == steps:
                curve.append((step, accuracy(net, train)[0], accuracy(net, held)[0], float(loss.detach())))
    held_acc, gen = accuracy(net, held)
    return dict(world=world, seed=seed, inp=inp, out=out, regime=regime, steps=steps, wd=wd, aux=aux, width=width,
                params=sum(x.numel() for x in net.parameters()),
                train_final=curve[-1][1], held_final=held_acc, held_best=max(c[2] for c in curve),
                probe_order=probe(net, inp), cpu_s=time.process_time() - t0, curve=curve,
                held=[dict(pair=p, got=bytes(g).decode("latin1"), want=bytes(a).decode("latin1"))
                      for p, g, (_, a) in zip(held_pairs, gen, held)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inp", choices=("digit", "unary"), default="digit")
    ap.add_argument("--out", choices=("dec", "sym"), default="dec")
    ap.add_argument("--regime", choices=("online", "batch"), default="batch")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--aux", choices=("none", "count", "count_shuf"), default="none")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    r = run(a.world, a.seed, a.inp, a.out, a.regime, a.steps, a.wd, a.log_every, a.aux, a.width)
    with open(a.save, "w") as f:
        json.dump(r, f)
    print(json.dumps({k: r[k] for k in ("inp", "out", "regime", "wd", "aux", "train_final", "held_final", "held_best",
                                         "probe_order", "cpu_s")}))
