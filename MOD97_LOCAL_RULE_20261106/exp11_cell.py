"""E11: a diagonal memory cell with a time-local trace inside the E8 learner. Predictions: E45_PREDICTIONS.md, E11.

The learner is the E8 window net (hidden 512) with a 10-byte window, plus a diagonal memory cell:
    p_t = tanh(Wp z1_t)                       64-d input projection from the first hidden layer
    c_t = tanh(Wc p_t + bc), r_t = sigmoid(Wr p_t + br)
    h_t = r_t * h_{t-1} + (1 - r_t) * c_t     512 units, each with its own gate; no unit reads another's state
    z2_t = tanh(W2 z1_t + Wm h_t), logits = W3 z2_t
Because the recurrence is diagonal and the gates do not read h, the per-synapse eligibility trace
    e_t = r_t * e_{t-1} + (direct partial of h_t)        (one scalar per synapse, carried forward in time)
makes  dLoss/dWc = sum_t L_t * e_t  EXACT, with L_t = dLoss_t/dh_t the learning signal from the readout at time t.
That is the time-local credit-through-time rule tested here (Exp 3's self-path trace through a full GRU failed).

Task: single-step operations with k uppercase distractor letters between the operator and the second operand,
    <code> a a <op> D1..Dk b b = c c EOS      k in 0..12 (0..3 for the no-memory birth); the window shows the first
operand at the answer only for k <= 3, so k >= 4 needs the cell.  Sealed panels at k = 0, 8, 12 and 20 (never seen).

Births (development by backprop through time on four operations mod 89):
    M  memory birth, k in 0..12         N  no-memory birth, k in 0..3 (letters familiar, holding never required)
Life (four operations mod 97 in sequence; replay of 1024 stored episodes per stage, prompt and answer as presented,
which is what E8's exact-key store returned at replay when its keys fit; with distractor letters every presentation is a new
key and the 65,536-key store overflows within a stage, so the episodes are kept directly):
    T  sign rule everywhere + exact diagonal trace for the cell's synapses       (the demonstration)
    S  sign rule everywhere, cell synapses by the within-step rule (state detached)   (control: no credit through time)
    D  backprop through time, joint ground truth of everything seen so far           (ceiling)
"""
import os
os.environ["EXP45_WINDOW"] = "10"
os.environ.setdefault("EXP45_HIDDEN", "512")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy, pad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp45 as X
from exp3_state import lin

T.set_num_threads(1)
assert X.W == 10 and X.H == 512
EOS, BOS, OUT, SEEDS = X.EOS, X.BOS, X.OUT, X.SEEDS
W, H, D, C = X.W, X.H, 64, 512
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]
NAME = lambda op, m, c: f"{op}{m}"
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
KMAX = {"M": 12, "N": 3}
KS_PANEL = (0, 8, 12, 20)
DEV_UPDATES, STAGE_UPDATES = X.DEV_UPDATES, int(os.environ.get("E11_STAGE_UPDATES", "6144"))   # override only for dry runs
DRYTAG = "" if STAGE_UPDATES == 6144 else f"_dry{STAGE_UPDATES}"
# E11-D diagnosis knobs (defaults reproduce E11 exactly): E11_WD local decay, E11_REPLAY episode|fresh, E11_SCHEDULE blocked|interleaved, E11_TRACE forward|scan
WD = float(os.environ.get("E11_WD", "1.0"))
REPLAY = os.environ.get("E11_REPLAY", "episode")
SCHEDULE = os.environ.get("E11_SCHEDULE", "blocked")
TRACE = os.environ.get("E11_TRACE", "forward")
DIAGTAG = ("" if WD == 1.0 else f"_wd{WD:g}") + ("" if REPLAY == "episode" else "_fr") + ("" if SCHEDULE == "blocked" else "_int")


class CellNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.book = nn.Embedding(258, 8, padding_idx=BOS)
        self.l1 = nn.Linear(8 * W, H)
        self.lp = nn.Linear(H, D)
        self.lc, self.lr = nn.Linear(D, C), nn.Linear(D, C)
        self.lm = nn.Linear(C, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, 257)

    def window(self, x):
        return self.book(pad(x, (W - 1, 0), value=BOS).unfold(1, W, 1)).flatten(2)


def new_net(seed):
    T.manual_seed(seed)
    return CellNet()


def forward(m, x, local, through_time, keep=False, silence=False):
    u = m.window(x)
    z1 = T.tanh(lin(u, m.l1, local))
    p = T.tanh(lin(z1, m.lp, local))
    c = T.tanh(lin(p, m.lc, local))
    r = T.sigmoid(lin(p, m.lr, local))
    B, L, _ = c.shape
    h = T.zeros(B, C)
    hs, prev = [], []
    for t in range(L):
        hp = h if through_time else h.detach()
        prev.append(hp)
        h = r[:, t] * hp + (1 - r[:, t]) * c[:, t]
        if keep:
            h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    if silence:
        Hs = T.zeros_like(Hs)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), dict(p=p, c=c, r=r, hs=hs, prev=prev)


@T.no_grad()
def trace_grads(m, it):
    """Exact gradient of the loss for the cell's synapses (Wc, bc, Wr, br) from the per-synapse traces and the
    per-step learning signals L_t = hs[t].grad (readout path only, since the state was detached in the forward)."""
    p, c, r, hs, prev = it["p"], it["c"], it["r"], it["hs"], it["prev"]
    B = p.shape[0]
    ec = T.zeros(B, C, D); er = T.zeros(B, C, D); ebc = T.zeros(B, C); ebr = T.zeros(B, C)
    gWc = T.zeros(C, D); gWr = T.zeros(C, D); gbc = T.zeros(C); gbr = T.zeros(C)
    for t in range(len(hs)):
        rt, ct, pt, hp = r[:, t], c[:, t], p[:, t], prev[t]
        gc = (1 - rt) * (1 - ct * ct)
        gr = (hp - ct) * rt * (1 - rt)
        ec = rt[:, :, None] * ec + gc[:, :, None] * pt[:, None, :]
        er = rt[:, :, None] * er + gr[:, :, None] * pt[:, None, :]
        ebc = rt * ebc + gc
        ebr = rt * ebr + gr
        Lt = hs[t].grad
        if Lt is not None:
            gWc += T.einsum("bc,bcd->cd", Lt, ec); gWr += T.einsum("bc,bcd->cd", Lt, er)
            gbc += (Lt * ebc).sum(0); gbr += (Lt * ebr).sum(0)
    m.lc.weight.grad, m.lc.bias.grad, m.lr.weight.grad, m.lr.bias.grad = gWc, gbc, gWr, gbr


@T.no_grad()
def trace_grads_scan(m, it):
    """The same gradient as trace_grads, computed by the algebraically identical backward scan of the learning signal:
    Lambda_s = L_s + r_{s+1} * Lambda_{s+1}; dLoss/dWc = sum_s (Lambda_s * gc_s)^T p_s. Not local in time (a compute trick);
    equality with the forward trace is checked in qualify()."""
    p, c, r, hs, prev = it["p"], it["c"], it["r"], it["hs"], it["prev"]
    Ls = T.stack([h.grad if h.grad is not None else T.zeros_like(h) for h in hs], 1)   # (B, L, C)
    Lam = T.zeros_like(Ls)
    acc = T.zeros_like(Ls[:, 0])
    for t in range(Ls.shape[1] - 1, -1, -1):
        acc = Ls[:, t] + (r[:, t + 1] * acc if t + 1 < Ls.shape[1] else 0)
        Lam[:, t] = acc
    hp = T.stack(prev, 1)
    gc = (1 - r) * (1 - c * c)
    gr = (hp - c) * r * (1 - r)
    m.lc.weight.grad = T.einsum("btc,btd->cd", Lam * gc, p); m.lc.bias.grad = (Lam * gc).sum((0, 1))
    m.lr.weight.grad = T.einsum("btc,btd->cd", Lam * gr, p); m.lr.bias.grad = (Lam * gr).sum((0, 1))


def update(m, opt, items, arm, store=None):
    x, y, mask = X.batch_tensor(items)
    logits, it = forward(m, x, local=(arm != "D"), through_time=(arm == "D"), keep=(arm == "T"))
    loss = cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    if arm == "T":
        (trace_grads_scan if TRACE == "scan" else trace_grads)(m, it)
    opt.step()
    if store is not None:
        store.write(x, y, mask)
    return float(loss.detach())


@T.no_grad()
def respond(m, prompts, store=None, cap=4, silence=False):
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            p = T.tensor([[BOS, *prompts[i]] for i in ids])
            gen = [[] for _ in ids]
            done = [False] * len(ids)
            for _ in range(cap):
                lg = forward(m, p, False, True, silence=silence)[0][:, -1]
                if store is not None:
                    lg = lg + store.logits(p)[:, -1]
                yy = lg.argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q]))
                        if int(yy[q]) == EOS:
                            done[q] = True
                p = T.cat([p, yy[:, None]], 1)
                if all(done):
                    break
            for q, i in enumerate(ids):
                out[i] = gen[q]
    return out


def score(m, items, store=None, silence=False):
    outs = respond(m, [p for p, _ in items], store, silence=silence)
    return dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)),
                final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))


# ---------------------------------------------------------------- task
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
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(tasks_pairs, seed, kmax):
    """tasks_pairs: list of (task, pairs). Each presentation: a uniformly drawn task-pair, k uniform in 0..kmax, fresh letters."""
    rng = Random(seed + 77)
    flat = [(t, pr) for t, pairs in tasks_pairs for pr in pairs]
    while True:
        order = list(range(len(flat)))
        rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            yield render(t, a, b, rng.randint(0, kmax), rng)


def batches(tasks_pairs, seed, kmax):
    st = stream(tasks_pairs, seed, kmax)
    while True:
        yield [next(st) for _ in range(X.BATCH)]


def panel(task, pairs, seed, ks=KS_PANEL, n=400):
    rng = Random(seed + 3)
    sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


# ---------------------------------------------------------------- development and life
def develop(birth, seed):
    path = OUT / f"e11_dev_{birth}_{seed}.pt"
    if path.exists():
        return
    m = new_net(seed)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    tp = [(t, split(t, .5, seed)[0]) for t in DEV]
    panels = {NAME(*t): panel(t, split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in DEV}
    bt = batches(tp, seed, KMAX[birth])
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, DEV_UPDATES + 1, X.BLOCK):
        losses = [update(m, opt, next(bt), "D") for _ in range(X.BLOCK)]
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: {k: score(m, it) for k, it in pk.items()} for n, pk in panels.items()})
            curve.append(row)
            print("e11 dev", birth, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
    row["silenced_k12"] = {n: score(m, pk[12], silence=True) for n, pk in panels.items()}
    T.save(m.state_dict(), path)
    (OUT / f"e11_dev_{birth}_{seed}.json").write_text(json.dumps(dict(birth=birth, seed=seed, kmax=KMAX[birth], curve=curve), indent=1))


def life(birth, arm, seed):
    path = OUT / f"e11_life_{birth}{arm}{DIAGTAG}{DRYTAG}_{seed}.json"
    m = new_net(seed)
    m.load_state_dict(T.load(OUT / f"e11_dev_{birth}_{seed}.pt"))
    local = arm != "D"
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(WD if local else .3), betas=(.9, .98))
    store = X.Store(seed) if local else None
    rng = Random(seed + 9)
    panels = {NAME(*t): panel(t, split(t, .25, seed)[1], seed) for t in LIFE}
    panels.update({NAME(*t): panel(t, split(t, .5, seed)[1], seed, ks=(12,), n=300) for t in DEV})
    buffer, seen = [], []
    t0 = time.process_time(); stages = []

    def evaluate():
        res = {n: {k: score(m, it, store) for k, it in pk.items()} for n, pk in panels.items()}
        res["silenced_k12"] = {n: score(m, pk[12], store, silence=True) for n, pk in panels.items() if n.endswith("97")}
        return res

    stages.append(dict(stage=0, task=None, scores=evaluate()))
    if SCHEDULE == "interleaved":
        joint_tp = [(t, split(t, .25, seed)[0]) for t in LIFE]
        schedule = [("int%d" % i, joint_tp) for i in range(1, 5)]      # four blocks of the one joint stream, scored after each
    else:
        schedule = [(NAME(*t), [(t, split(t, .25, seed)[0])]) for t in LIFE]
    for si, (name, tp) in enumerate(schedule, 1):
        t, pairs = tp[0]
        cur = batches(tp, seed + si, 12)
        if arm == "D":
            seen = seen + [(t, pairs)]; joint = batches(seen, seed + si, 12)
        losses, presented = [], []
        for u in range(STAGE_UPDATES):
            if arm == "D":
                batch = next(joint)
            elif buffer:
                rp = rng.sample(buffer, X.BATCH // 2)
                if REPLAY == "fresh":   # same stored pairs and answers, fresh distractor letters and k
                    rp = [render(tk, a, b, rng.randint(0, 12), rng) for tk, a, b in rp]
                batch = next(cur)[:X.BATCH // 2] + rp   # half replay
                presented += batch[:X.BATCH // 2]
            else:
                batch = next(cur)
                presented += batch
            losses.append(update(m, opt, batch, arm, store))
        stages.append(dict(stage=si, task=name, loss=sum(losses[-512:]) / 512, cpu=time.process_time() - t0, scores=evaluate()))
        sc = stages[-1]["scores"]
        print("e11", birth + arm, seed, "after", name, {n: [sc[n][k]["exact"] for k in KS_PANEL] for n in sc if n.endswith("97")},
              "dev89@12", [sc[n][12]["exact"] for n in sc if n.endswith("89")], "silenced", [v["exact"] for v in sc["silenced_k12"].values()], f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(birth=birth, arm=arm, seed=seed, stages=stages), indent=1))
        if local and SCHEDULE == "blocked":
            ep = rng.sample(presented, min(1024, len(presented)))
            buffer += [(t, int(bytes(p[1:3])), int(bytes(p[-3:-1]))) for p, _ in ep] if REPLAY == "fresh" else ep   # 1024 episodes (prompt, answer as presented) of this stage: what E8's exact-key store returned when its keys fit


# ---------------------------------------------------------------- qualification
def qualify():
    out = {}
    rng = Random(1)
    for t in DEV + LIFE:
        tr, se = split(t, .25, SEEDS[0]); assert not (set(tr) & set(se))
        for a, b in tr[:30]:
            for k in (0, 5, 12, 20):
                p, tt = render(t, a, b, k, rng)
                assert len(p) == 7 + k and int(bytes(tt[:2])) == apply(t[0], t[1], a, b) and all(ch in DISTRACT for ch in p[4:4 + k])
    # window visibility: at the position predicting the first answer byte the window holds the last W prompt bytes
    out["first_operand_visible_at_k"] = [k for k in range(21) if 7 + k <= W]
    # trace == backprop through time, for the cell's synapses, on a random batch (no sign feedback, so the comparison is exact)
    m = new_net(SEEDS[0])
    items = [render(t, a, b, rng.randint(0, 12), rng) for t in DEV for a, b in split(t, .5, SEEDS[0])[0][:16]]
    x, y, mask = X.batch_tensor(items)
    logits, it = forward(m, x, False, through_time=True)
    loss = cross_entropy(logits[mask], y[mask]); m.zero_grad(); loss.backward()
    ref = {n: p.grad.clone() for n, p in m.named_parameters()}
    logits, it = forward(m, x, False, through_time=False, keep=True)
    loss2 = cross_entropy(logits[mask], y[mask]); m.zero_grad(); loss2.backward()
    within = {n: m.get_parameter(n).grad.clone() for n in ("lc.weight", "lr.weight")}
    trace_grads(m, it)
    out["trace_vs_bptt_max_rel_err"] = {n: float((m.get_parameter(n).grad - ref[n]).abs().max() / ref[n].abs().max()) for n in ("lc.weight", "lc.bias", "lr.weight", "lr.bias")}
    fwd = {n: m.get_parameter(n).grad.clone() for n in ("lc.weight", "lc.bias", "lr.weight", "lr.bias")}
    trace_grads_scan(m, it)
    out["scan_vs_forward_trace_max_rel_err"] = {n: float((m.get_parameter(n).grad - fwd[n]).abs().max() / fwd[n].abs().max()) for n in fwd}
    for n in fwd: m.get_parameter(n).grad = fwd[n]
    out["within_step_vs_bptt_max_rel_err"] = {n: float((within[n] - ref[n]).abs().max() / ref[n].abs().max()) for n in within}
    out["loss_same_forward"] = [float(loss.detach()), float(loss2.detach())]
    out["params"] = sum(p.numel() for p in m.parameters())
    # smoke: ms per update per arm
    opt = T.optim.AdamW(m.parameters(), lr=.001)
    bt = batches([(t, split(t, .5, SEEDS[0])[0]) for t in DEV], 5, 12)
    sm = {}
    for arm in ("D", "S", "T"):
        t0 = time.process_time()
        for _ in range(20):
            update(m, opt, next(bt), arm)
        sm[arm] = (time.process_time() - t0) / 20 * 1000
    out["ms_per_update"] = sm
    t0 = time.process_time(); score(m, panel(LIFE[0], split(LIFE[0], .25, SEEDS[0])[1], SEEDS[0])[12]); out["s_per_panel400"] = time.process_time() - t0
    (OUT / "QUALIFY_E11.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop(sys.argv[2], int(sys.argv[3]))
    else:
        life(sys.argv[1][0], sys.argv[1][1], int(sys.argv[2]))
