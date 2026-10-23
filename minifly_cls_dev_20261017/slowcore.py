"""Slow core: a small learned recurrent layer, byte in / byte out, trained by error that reaches its own weights.

h_t = (1 - alpha) * h_{t-1} + alpha * tanh(W_in[:, token_t] + W_rec h_{t-1} + b),  logits_t = W_out h_t + c
* alpha: fixed per-unit leak rates, log-spaced over `alpha_range` (several time scales, no gates).
* Sequence: BEGIN, prompt bytes, REPLY, `ticks` silent THINK steps, then the answer bytes are emitted one at a time
  (each emitted byte is fed back as the next input); EOS ends the answer. The THINK steps carry no information.
* Learning rules (both give a gradient estimate; the same AdamW step and global-norm clip are applied):
    bptt   exact backpropagation through time (development reference only).
    eprop  e-prop (Bellec et al. 2020): each synapse keeps a local eligibility trace of its own recent effect on its
           postsynaptic unit, eps_t = (1 - alpha) eps_{t-1} + alpha tanh'(a_t) pre_t; a learning signal per unit,
           L_t = (softmax(logits_t) - target_t) projected back through fixed feedback weights (`symmetric`: W_out^T;
           `random`: a fixed random matrix), multiplies it. No error is propagated backward through time.
* Weight decay (decoupled, AdamW) is the "downscaling" pressure; during sleep it is applied at every replay step.
"""
import numpy as np

EOS, BEGIN, REPLY, THINK = 256, 257, 258, 259
VOCAB_IN, VOCAB_OUT = 260, 257


def sequences(records, ticks):
    """Token and target arrays for a batch of (prompt, answer) records; target -1 = no target at that step."""
    toks, tgts = [], []
    for prompt, answer in records:
        prompt, answer = list(bytes(prompt)), list(bytes(answer))
        seq = [BEGIN] + prompt + [REPLY] + [THINK] * ticks + answer
        tgt = [-1] * (len(prompt) + 1 + ticks) + answer + [EOS]
        toks.append(seq)
        tgts.append(tgt)
    n = max(map(len, toks))
    tok = np.zeros((len(toks), n), dtype=np.int64)
    tgt = np.full((len(toks), n), -1, dtype=np.int64)
    for i, (s, t) in enumerate(zip(toks, tgts)):
        tok[i, :len(s)] = s
        tgt[i, :len(t)] = t
    return tok, tgt


class SlowCore:
    def __init__(self, seed, width=64, ticks=0, rule="bptt", feedback="symmetric", lr=1e-3, wd=0.0,
                 alpha_range=(0.1, 0.9), clip=1.0):
        if rule not in ("bptt", "eprop") or feedback not in ("symmetric", "random"):
            raise ValueError("rule in {bptt, eprop}; feedback in {symmetric, random}")
        rng = np.random.default_rng(seed)
        self.width, self.ticks, self.rule, self.feedback = width, ticks, rule, feedback
        self.lr, self.wd, self.clip = lr, wd, clip
        q, r = np.linalg.qr(rng.normal(size=(width, width)))
        self.p = dict(W_in=rng.normal(0, 1.0, (width, VOCAB_IN)),
                      W_rec=q * np.sign(np.diag(r))[None, :],
                      b=np.zeros(width),
                      W_out=np.zeros((VOCAB_OUT, width)),
                      c=np.zeros(VOCAB_OUT))
        self.alpha = np.exp(np.linspace(np.log(alpha_range[0]), np.log(alpha_range[1]), width))
        self.B_random = rng.normal(0, 1 / np.sqrt(width), (VOCAB_OUT, width))
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.steps = 0

    # ------------------------------------------------------------------ forward / gradients
    def _forward(self, tok):
        p, a_ = self.p, self.alpha
        B, L = tok.shape
        h = np.zeros((B, self.width))
        H, A = np.zeros((B, L, self.width)), np.zeros((B, L, self.width))
        for t in range(L):
            a = p["W_in"][:, tok[:, t]].T + h @ p["W_rec"].T + p["b"]
            h = (1 - a_) * h + a_ * np.tanh(a)
            H[:, t], A[:, t] = h, a
        return H, A

    def _out_grad(self, H, tgt):
        """Softmax cross-entropy: mean loss over target steps, dL/dlogits per step (zero where no target)."""
        logits = H @ self.p["W_out"].T + self.p["c"]
        logits -= logits.max(-1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(-1, keepdims=True)
        mask = tgt >= 0
        n = max(1, int(mask.sum()))
        idx = np.where(mask, tgt, 0)
        picked = np.take_along_axis(prob, idx[..., None], -1)[..., 0]
        loss = float(-np.log(np.maximum(picked[mask], 1e-300)).sum() / n)
        g = prob.copy()
        np.put_along_axis(g, idx[..., None], np.take_along_axis(g, idx[..., None], -1) - 1, -1)
        g *= mask[..., None] / n
        return loss, g

    def gradients(self, tok, tgt):
        p, a_ = self.p, self.alpha
        H, A = self._forward(tok)
        loss, g = self._out_grad(H, tgt)
        B, L = tok.shape
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["W_out"] = np.einsum("btk,btj->kj", g, H)
        grads["c"] = g.sum((0, 1))
        dphi = 1 - np.tanh(A) ** 2
        H_prev = np.concatenate([np.zeros((B, 1, self.width)), H[:, :-1]], 1)
        if self.rule == "bptt":
            dh_next = np.zeros((B, self.width))
            for t in reversed(range(L)):
                dh = g[:, t] @ p["W_out"] + dh_next
                da = dh * a_ * dphi[:, t]
                np.add.at(grads["W_in"].T, tok[:, t], da)
                grads["W_rec"] += da.T @ H_prev[:, t]
                grads["b"] += da.sum(0)
                dh_next = dh * (1 - a_) + da @ p["W_rec"]
        else:
            Bfb = p["W_out"] if self.feedback == "symmetric" else self.B_random
            used, local = np.unique(tok, return_inverse=True)      # input eligibility only for tokens that occur
            local = local.reshape(tok.shape)
            eps_rec = np.zeros((B, self.width, self.width))
            eps_in = np.zeros((B, self.width, len(used)))
            eps_b = np.zeros((B, self.width))
            g_in = np.zeros((self.width, len(used)))
            rows = np.arange(B)
            for t in range(L):
                k = a_ * dphi[:, t]                                    # (B, H): alpha * tanh'(a_t)
                eps_rec = (1 - a_)[None, :, None] * eps_rec + k[:, :, None] * H_prev[:, t][:, None, :]
                eps_in *= (1 - a_)[None, :, None]
                eps_in[rows, :, local[:, t]] += k
                eps_b = (1 - a_) * eps_b + k
                if np.any(tgt[:, t] >= 0):
                    Ls = g[:, t] @ Bfb                                  # (B, H) learning signal
                    grads["W_rec"] += np.einsum("bj,bji->ji", Ls, eps_rec)
                    g_in += np.einsum("bj,bjv->jv", Ls, eps_in)
                    grads["b"] += (Ls * eps_b).sum(0)
            grads["W_in"][:, used] += g_in
        return loss, grads

    def step(self, records):
        """One AdamW update on a batch of records. Returns the batch loss."""
        tok, tgt = sequences(records, self.ticks)
        loss, grads = self.gradients(tok, tgt)
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, self.clip / max(norm, 1e-30))
        self.steps += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        for k, w in self.p.items():
            gk = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.steps)
            vh = self.v[k] / (1 - b2 ** self.steps)
            w -= self.lr * (mh / (np.sqrt(vh) + e) + self.wd * w)
        return loss

    # ------------------------------------------------------------------ recall
    def _run(self, h, tokens):
        p, a_ = self.p, self.alpha
        for t in tokens:
            h = (1 - a_) * h + a_ * np.tanh(p["W_in"][:, t] + p["W_rec"] @ h + p["b"])
        return h

    def generate(self, prompt, cap=4):
        """Own greedy output. Returns (bytes, mean top-2 probability margin)."""
        h = self._run(np.zeros(self.width), [BEGIN, *bytes(prompt), REPLY] + [THINK] * self.ticks)
        out, margins = bytearray(), []
        for _ in range(cap + 1):
            z = self.p["W_out"] @ h + self.p["c"]
            pr = np.exp(z - z.max())
            pr /= pr.sum()
            top = np.sort(pr)[-2:]
            margins.append(float(top[1] - top[0]))
            tok = int(z.argmax())
            if tok == EOS or len(out) == cap:
                break
            out.append(tok)
            h = self._run(h, [tok])
        return bytes(out), float(np.mean(margins))

    def accuracy(self, records):
        return float(np.mean([self.generate(p)[0] == bytes(a) for p, a in records]))

    def probe_order(self):
        """|Spearman rho| of the first principal component of the state after `a+` with a (as in the diagnosis)."""
        S = np.array([self._run(np.zeros(self.width), [BEGIN, ord(str(a)), ord("+")]) for a in range(10)])
        S -= S.mean(0)
        pc = np.linalg.svd(S, full_matrices=False)[2][0] @ S.T
        r = np.argsort(np.argsort(pc))
        return float(abs(np.corrcoef(r, np.arange(10))[0, 1]))
