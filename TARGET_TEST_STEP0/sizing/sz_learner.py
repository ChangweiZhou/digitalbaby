"""Sizing sweep: the one learner, and the two ways of feeding it (brief OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3).

The network is E20-B's G1 shape, built directly (the arithmetic harness is not imported): byte table `book` (258 x 8, BOS = 257 a padding
row), l1 (8 -> 512), one GRU layer of 256 units, the two-layer readout z2 = tanh(l2 z1 + lm h), logits = l3 z2 over 257 outputs. Modules
are created in arch20.NetRNN's order, so a given seed gives the same initial weights. 1,124,113 parameters.

Training: backprop, AdamW at E20-B's birth settings (lr 0.001, weight decay 0.3, betas (0.9, 0.98), the `book` table in a group without
decay, no gradient clipping), every next byte the target, no answer mask, the loss the mean over the targets of a batch or chunk, one update
per batch or chunk, model seed = world seed + 1000. One thread.

    Arm A (EpisodeLearner): one item per episode, the state starts from zero at every episode ([BOS] then the episode's bytes; targets are
      the episode's bytes); batches of 64 episodes in stream order.
    Arm B (StreamLearner): the stream read once, in order, state carried from byte to byte; input at target position k is BOS for k = 0 and
      the stream byte k - 1 otherwise; credit cut every 256 target positions (the state is detached, not reset).

Probes (both arms; the order of events of the brief's section "A probe inside the stream (arm B), exactly"):
  * the boundary before a probe is set by the last loss TARGET used: training stops at the last batch or chunk whose final target comes
    before the probe's first answer byte, and batches or chunks are cut there; the cut positions are part of the schedule and are the same
    whether probes are scored or not;
  * respond(prompts) answers each prompt from the boundary state, independently (the twin fork: one saved state before the first byte of
    either member), writes the whole answer (two digits and the end byte) greedily, feeding back its own bytes; it changes nothing: no
    weight, optimizer state, carried state, cursor or counter.
"""
import time

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

BOS, NOUT, UNITS, H = 257, 257, 256, 512     # BOS = 257 is E20-B's padding row; output 256 (E20-B's EOS) is never a target here
LR, WD, BETAS = 0.001, 0.3, (0.9, 0.98)
CHUNK, BATCH = 256, 64
ANSWER_LEN = 3                      # two digits and the end byte


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(8, H)
        self.rnn = nn.GRU(H, UNITS, num_layers=1, batch_first=True)
        self.lm = nn.Linear(UNITS, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, NOUT)

    def forward(self, x, h):
        z1 = T.tanh(self.l1(self.book(x)))
        hs, hT = self.rnn(z1, h)
        return self.l3(T.tanh(self.l2(z1) + self.lm(hs))), hT


def new_net(model_seed):
    T.manual_seed(model_seed)
    return Net()


def make_opt(m):
    groups = [{"params": [p for n, p in m.named_parameters() if not n.startswith("book.")]},
              {"params": [p for n, p in m.named_parameters() if n.startswith("book.")], "weight_decay": 0.0}]
    return T.optim.AdamW(groups, lr=LR, weight_decay=WD, betas=BETAS)


def opt_groups(m):
    return dict(decay=[n for n, _ in m.named_parameters() if not n.startswith("book.")],
                no_decay=[n for n, _ in m.named_parameters() if n.startswith("book.")], lr=LR, weight_decay=WD, betas=list(BETAS))


@T.no_grad()
def generate(m, h, inputs, n=ANSWER_LEN):
    """inputs: list of byte lists, all of one length, read from state h (1, 1, UNITS) each; then n bytes written greedily, each fed back.
    Returns (list of bytes objects, logits of the first written byte)."""
    B = len(inputs)
    x = T.tensor(inputs, dtype=T.long)
    hb = h.expand(1, B, UNITS).contiguous()
    lg, hb = m(x, hb)
    out = [[] for _ in range(B)]; first = lg[:, -1].clone()
    y = lg[:, -1].argmax(1)
    for k in range(n):
        for i in range(B):
            out[i].append(int(y[i]))
        if k == n - 1:
            break
        lg, hb = m(y[:, None], hb)
        y = lg[:, -1].argmax(1)
    return [bytes(o) if all(b < 256 for b in o) else bytes(b if b < 256 else 0 for b in o) + b"?" for o in out], first


class _Base:
    def __init__(self, model_seed):
        self.model_seed = model_seed
        self.m = new_net(model_seed); self.opt = make_opt(self.m)
        self.updates = 0; self.targets = 0; self.inputs_read = 0; self.cuts_used = []
        self.cpu_train = 0.0; self.cpu_respond = 0.0; self.responses = 0
        self.loss_log = []            # (update, n targets, mean loss)

    def _step(self, loss):
        self.opt.zero_grad(); loss.backward(); self.opt.step(); self.updates += 1

    def respond(self, prompts, start=None):
        """Each prompt answered from the boundary state, independently; grouped by length (rows of one batch never interact). Returns bytes."""
        t0 = time.process_time()
        h = self.boundary_state()
        pend = self.pending_input()
        groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        out = [None] * len(prompts)
        for L, idx in groups.items():
            for j in range(0, len(idx), 256):
                ids = idx[j:j + 256]
                res, _ = generate(self.m, h, [[pend] + list(prompts[i]) for i in ids])
                for i, r in zip(ids, res):
                    out[i] = r
        self.cpu_respond += time.process_time() - t0; self.responses += len(prompts)
        return out

    def respond_logits(self, prompt):
        """One prompt, alone: (answer bytes, logits of its first answer byte). Used by the fixtures."""
        res, lg = generate(self.m, self.boundary_state(), [[self.pending_input()] + list(prompt)])
        return res[0], lg[0]

    def snapshot(self):
        """Everything a probe must not change, as plain tensors and numbers (compared bitwise by the fixtures)."""
        st = {f"w/{n}": p.detach().clone() for n, p in self.m.named_parameters()}
        for gi, g in enumerate(self.opt.param_groups):
            for pi, p in enumerate(g["params"]):
                s = self.opt.state.get(p, {})
                for k, v in s.items():
                    st[f"opt/{gi}/{pi}/{k}"] = v.detach().clone() if T.is_tensor(v) else T.tensor(float(v))
        st["state"] = self.boundary_state().detach().clone()
        st["counters"] = T.tensor([self.cursor, self.updates, self.targets, self.inputs_read, len(self.cuts_used)], dtype=T.float64)
        return st


class StreamLearner(_Base):
    """Arm B. Positions are target indices into the stream bytes s: target k is s[k], its input s[k - 1] (BOS for k = 0)."""

    def __init__(self, model_seed, data, cuts, outcome_mask=None):
        super().__init__(model_seed)
        self.s = data; self.n = len(data)
        self.grid = sorted(set(range(CHUNK, self.n, CHUNK)) | {c for c in cuts if 0 < c < self.n} | {self.n})
        self.cuts = sorted({c for c in cuts if 0 < c < self.n and c % CHUNK})
        self.gi = 0; self.cursor = 0
        self.h = T.zeros(1, 1, UNITS)
        self.xs = T.tensor([BOS] + list(data[:-1]), dtype=T.long)
        self.ys = T.tensor(list(data), dtype=T.long)
        self.outcome_mask = outcome_mask      # bool tensor over target positions (the two answer digits of every episode), for the loss curves
        self.tenths = T.zeros(10, 2, 2, dtype=T.float64)   # [tenth][outcome?][sum, count]
        self.observe = None

    def boundary_state(self):
        return self.h

    def pending_input(self):
        return BOS if self.cursor == 0 else self.s[self.cursor - 1]

    def advance_to(self, b):
        """Train chunk by chunk until the cursor is at b; b must be on the schedule (a nominal cut, a probe cut or the end)."""
        assert b >= self.cursor
        t0 = time.process_time()
        while self.cursor < b:
            e = self.grid[self.gi]; self.gi += 1
            assert e <= b, f"boundary {b} not on the schedule (next {e})"
            a = self.cursor
            x = self.xs[a:e][None]; y = self.ys[a:e]
            lg, hT = self.m(x, self.h)
            per = cross_entropy(lg[0], y, reduction="none")
            loss = per.mean()
            if self.observe is not None:       # fixtures only: sees this chunk's predictions, made before its update
                self.observe(a, e, lg[0].detach())
            self._step(loss)
            self.h = hT.detach()
            self.targets += e - a; self.inputs_read += e - a; self.cursor = e
            if e in self.cuts:
                self.cuts_used.append(e)
            self.loss_log.append((self.updates, e - a, float(loss.detach())))
            if self.outcome_mask is not None:
                pd = per.detach().double(); om = self.outcome_mask[a:e]
                tenth = (T.arange(a, e) * 10 // self.n)
                for t in tenth.unique().tolist():
                    sel = tenth == t
                    for o in (0, 1):
                        ss = sel & (om if o else ~om)
                        self.tenths[t, o, 0] += pd[ss].sum(); self.tenths[t, o, 1] += ss.sum()
        self.cpu_train += time.process_time() - t0


class EpisodeLearner(_Base):
    """Arm A. Positions are episode indices: advance_to(i) trains every episode before i, in batches of 64 cut at the schedule."""

    def __init__(self, model_seed, episodes, cuts):
        """episodes: list of episode byte strings, in stream order."""
        super().__init__(model_seed)
        self.eps = episodes; self.n = len(episodes)
        self.grid = sorted(set(range(BATCH, self.n, BATCH)) | {c for c in cuts if 0 < c < self.n} | {self.n})
        self.cuts = sorted({c for c in cuts if 0 < c < self.n and c % BATCH})
        self.gi = 0; self.cursor = 0
        self.h0 = T.zeros(1, 1, UNITS)
        self.tenths = T.zeros(10, 2, 2, dtype=T.float64)

    def boundary_state(self):
        return self.h0

    def pending_input(self):
        return BOS

    def advance_to(self, b):
        assert b >= self.cursor
        t0 = time.process_time()
        while self.cursor < b:
            e = self.grid[self.gi]; self.gi += 1
            assert e <= b, f"boundary {b} not on the schedule (next {e})"
            a = self.cursor; items = self.eps[a:e]; L = max(len(it) for it in items)
            x = T.full((len(items), L), BOS, dtype=T.long); y = T.full((len(items), L), -100, dtype=T.long)
            om = T.zeros((len(items), L), dtype=T.bool)
            for r, it in enumerate(items):
                x[r, 0] = BOS; x[r, 1:len(it)] = T.tensor(list(it[:-1])); y[r, :len(it)] = T.tensor(list(it))
                q = it.index(b"="); om[r, q + 1:q + 3] = True
            lg, _ = self.m(x, self.h0.expand(1, len(items), UNITS).contiguous())
            per = cross_entropy(lg.reshape(-1, NOUT), y.reshape(-1), reduction="none", ignore_index=-100).reshape(y.shape)
            valid = y != -100
            loss = per[valid].mean()
            self._step(loss)
            nt = int(valid.sum()); self.targets += nt; self.inputs_read += nt; self.cursor = e
            if e in self.cuts:
                self.cuts_used.append(e)
            self.loss_log.append((self.updates, nt, float(loss.detach())))
            pd = per.detach().double()
            for r in range(len(items)):
                t = (a + r) * 10 // self.n
                for o in (0, 1):
                    ss = valid[r] & (om[r] if o else ~om[r])
                    self.tenths[t, o, 0] += pd[r][ss].sum(); self.tenths[t, o, 1] += ss.sum()
        self.cpu_train += time.process_time() - t0


def n_params(m):
    return sum(p.numel() for p in m.parameters())
