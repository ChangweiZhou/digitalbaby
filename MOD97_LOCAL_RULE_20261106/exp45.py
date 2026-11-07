"""E4 (retention) and E5 (composition). Design and predictions: MOD97_LOCAL_RULE_20261106/E45_PREDICTIONS.md.

Learner: harness window network (W=10, H=256), stateless. Development by backprop (AdamW lr .001, decay .3).
Life by the sign-symmetric rule (every linear map uses the sign of its weight backward; exp3_state.SignLinear),
answer-only loss, AdamW lr .001, decay per arm.
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from collections import Counter
from pathlib import Path
from random import Random

import torch as T
from torch.nn.functional import cross_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod97_local as M
from exp3_state import lin

T.set_num_threads(1)
EOS, BOS = M.EOS, M.BOS
SEEDS = M.SEEDS
W, H = 10, 256
BATCH, BLOCK = 64, 512
OUT = Path(__file__).resolve().parent / "exp45"
DEV_UPDATES = 24576
STAGE_UPDATES = 8192

TASKS = {"add83": ("add", 83, b"p"), "sub83": ("sub", 83, b"p"), "add89": ("add", 89, b"q"), "sub89": ("sub", 89, b"q"),
         "add97": ("add", 97, b"r"), "sub97": ("sub", 97, b"t"), "add73": ("add", 73, b"s"), "sub73": ("sub", 73, b"u")}
E4_DEV = ("add83", "sub83", "add89", "sub89")
E4_LIFE = ("add97", "sub97", "add73", "sub73")
E4_ARMS = ("A", "A0", "B", "C", "Cs", "D")
E5_DEV = ("add97", "sub97")
E5_ARMS = ("Z", "DIR", "SCR", "SCRsh", "DIRbp", "SCRbp", "SCRr", "SCRrbp")
TYPES = {"pp": ("+", "+"), "mm": ("-", "-"), "pm": ("+", "-"), "mp": ("-", "+")}
TAUGHT_TYPES, HELD_TYPES = ("pp", "mm"), ("pm", "mp")


# ---------------------------------------------------------------- model forward with optional local (sign) backward
def forward(m, x, local):
    u = m.window(x)
    z1 = T.tanh(lin(u, m.l1, local))
    z2 = T.tanh(lin(z1, m.l2, local))
    return lin(z2, m.l3, local)


def new_net(seed):
    T.manual_seed(seed)
    return M.Net(W=W, H=H)


# ---------------------------------------------------------------- single-step tasks
def answer(name, a, b):
    op, m, _ = TASKS[name]
    return (a + b) % m if op == "add" else (a - b) % m


def render(name, a, b):
    op, m, code = TASKS[name]
    sym = b"+" if op == "add" else b"-"
    prompt = list(code + f"{a:02d}".encode() + sym + f"{b:02d}".encode() + b"=")
    return prompt, list(f"{answer(name, a, b):02d}".encode()) + [EOS]


def split(name, frac, seed):
    op, m, _ = TASKS[name]
    groups = [(a, b) for a in range(m) for b in range(a, m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 2 + (op == "sub"))
    rng.shuffle(groups)
    k = round(frac * len(groups))
    taught, rest = groups[:k], groups[k:]
    monitor, sealed = rest[0::2], rest[1::2]
    expand = lambda gs: [p for a, b in gs for p in (((a, b), (b, a)) if a != b else ((a, b),))]
    return expand(sorted(taught)), expand(sorted(monitor)), expand(sorted(sealed))


def sample_rows(name, pairs, seed, n):
    rng = Random(seed + 77 + len(pairs))
    sub = pairs if len(pairs) <= n else rng.sample(pairs, n)
    return [render(name, *p) for p in sub]


# ---------------------------------------------------------------- two-step tasks (E5)
def compose_answer(t, a, b, c, mod=97):
    o1, o2 = TYPES[t]
    i = (a + b) % mod if o1 == "+" else (a - b) % mod
    f = (i + c) % mod if o2 == "+" else (i - c) % mod
    return i, f


def render_compose(t, a, b, c, scratch):
    o1, o2 = TYPES[t]
    prompt = list(b"c" + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=")
    i, f = compose_answer(t, a, b, c)
    target = (list(f"{i:02d}".encode()) + [ord(";")] if scratch else []) + list(f"{f:02d}".encode()) + [EOS]
    return prompt, target


def compose_triples(t, seed, n, exclude=frozenset()):
    rng = Random(seed * 31 + sum(map(ord, t)))
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


# ---------------------------------------------------------------- fast store (E4 arms B, C, Cs)
class Store:
    """Episodic fast store: exact key on the 10-byte window at each answer position, value = counts of the next
    byte seen there. Reads add +gain to the logit of the majority stored byte; unseen keys add nothing.
    This is the perfectly pattern-separated limit of a sparse expansion code (the project's HASH8), kept exact
    so that what it contributes is auditable. Capacity-bounded, first-in first-out."""
    def __init__(self, seed, gain=6., capacity=65536):
        self.gain, self.capacity = gain, capacity
        self.mem = {}
        self.order = []

    def _keys(self, x):
        win = T.nn.functional.pad(x, (W - 1, 0), value=BOS).unfold(1, W, 1)
        return win.tolist()

    @T.no_grad()
    def logits(self, x):
        B, L = x.shape
        out = T.zeros(B, L, 257)
        keys = self._keys(x)
        for b in range(B):
            for l in range(L):
                cnt = self.mem.get(tuple(keys[b][l]))
                if cnt:
                    out[b, l, max(cnt, key=cnt.get)] = self.gain
        return out

    @T.no_grad()
    def write(self, x, y, mask):
        keys = self._keys(x)
        for b, l in mask.nonzero().tolist():
            k = tuple(keys[b][l])
            if k not in self.mem:
                self.mem[k] = {}
                self.order.append(k)
                if len(self.order) > self.capacity:
                    self.mem.pop(self.order.pop(0), None)
            v = int(y[b, l])
            self.mem[k][v] = self.mem[k].get(v, 0) + 1


# ---------------------------------------------------------------- batching, update, generation, scoring
def batch_tensor(items):
    """items: list of (prompt, target). Returns x, y, answer mask."""
    rows = [p + t for p, t in items]
    L = max(len(r) for r in rows) + 1
    data = T.full((len(rows), L), BOS, dtype=T.long)
    for i, r in enumerate(rows):
        data[i, 1:len(r) + 1] = T.tensor(r)
    x, y = data[:, :-1], data[:, 1:]
    mask = T.zeros_like(y, dtype=T.bool)
    for i, (p, t) in enumerate(items):
        mask[i, len(p):len(p) + len(t)] = True
    return x, y, mask


def update(m, opt, items, local, store=None):
    x, y, mask = batch_tensor(items)
    logits = forward(m, x, local)
    loss = cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if store is not None:
        store.write(x, y, mask)
    return float(loss.detach())


@T.no_grad()
def respond(m, prompts, store=None, cap=7):
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
                lg = m(p)[:, -1]
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


def score(m, items, store=None):
    outs = respond(m, [p for p, _ in items], store)
    exact = sum(o == t for o, (_, t) in zip(outs, items))
    final = sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items))        # final two digits + EOS
    inter = sum(o[:3] == t[:3] for o, (_, t) in zip(outs, items)) if len(items[0][1]) > 3 else None
    return dict(n=len(items), exact=exact, final=final, intermediate=inter)


def stream(items, seed):
    order = Random(seed + 1)
    while True:
        for it in order.sample(items, len(items)):
            yield it


def batches(items, seed):
    st = stream(items, seed)
    while True:
        yield [next(st) for _ in range(BATCH)]


# ================================================================ E4
def e4_develop(seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"e4_dev_{seed}.pt"
    if path.exists():
        return
    m = new_net(seed)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    items = [render(n, *p) for n in E4_DEV for p in split(n, .5, seed)[0]]
    bt = batches(items, seed)
    t0 = time.process_time()
    curve = []
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = [update(m, opt, next(bt), False) for _ in range(BLOCK)]
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: score(m, sample_rows(n, split(n, .5, seed)[2], seed, 600)) for n in E4_DEV})
            curve.append(row)
            print("e4 dev", seed, stop, {n: v["exact"] for n, v in row["sealed"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e4_dev_{seed}.json").write_text(json.dumps(dict(seed=seed, curve=curve), indent=1))


def e4_life(arm, seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"e4_life_{arm}_{seed}.json"
    m = new_net(seed)
    m.load_state_dict(T.load(OUT / f"e4_dev_{seed}.pt"))
    local = arm != "D"
    decay = {"A": 1., "A0": .1, "B": 1., "C": 1., "Cs": 1., "D": .3}[arm]
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=decay, betas=(.9, .98))
    store = Store(seed) if arm in ("B", "C", "Cs") else None
    splits = {n: split(n, .25, seed) for n in E4_LIFE}
    panels = {n: dict(sealed=sample_rows(n, splits[n][2], seed, 600), taught=sample_rows(n, splits[n][0], seed, 300)) for n in E4_LIFE}
    panels.update({n: dict(sealed=sample_rows(n, split(n, .5, seed)[2], seed, 300)) for n in E4_DEV})
    shortcuts = {n: dict(nearest=0) for n in E4_LIFE}
    rng = Random(seed + 9)
    buffer = []                      # earlier tasks' taught prompts (for C, Cs)
    seen_items = []                  # ground truth of all tasks so far (for D)
    t0 = time.process_time()
    stages = []

    def evaluate(stage):
        res = {}
        for n, pn in panels.items():
            res[n] = {k: score(m, rows, store) for k, rows in pn.items()}
        return res

    stages.append(dict(stage=0, task=None, scores=evaluate(0)))
    for si, name in enumerate(E4_LIFE, 1):
        items = [render(name, *p) for p in splits[name][0]]
        cur = batches(items, seed + si)
        if arm == "D":
            seen_items = seen_items + items
            joint = batches(seen_items, seed + si)
        losses = []
        for u in range(1, STAGE_UPDATES + 1):
            if arm == "D":
                batch = next(joint)
            elif arm in ("C", "Cs") and buffer:
                half = next(cur)[:BATCH // 2]
                rp = rng.sample(buffer, BATCH // 2)
                if arm == "C":
                    outs = respond(m, rp, store)
                    ans = [o if (len(o) == 3 and o[-1] == EOS) else [ord("0"), ord("0"), EOS] for o in outs]
                else:
                    ans = [list(f"{rng.randrange(97):02d}".encode()) + [EOS] for _ in rp]
                batch = half + list(zip(rp, ans))
            else:
                batch = next(cur)
            losses.append(update(m, opt, batch, local, store))
        if arm in ("C", "Cs"):
            buffer += [p for p, _ in rng.sample(items, min(1024, len(items)))]
        stages.append(dict(stage=si, task=name, loss=sum(losses[-512:]) / 512, cpu=time.process_time() - t0, scores=evaluate(si)))
        print("e4", arm, seed, "after", name, {n: stages[-1]["scores"][n]["sealed"]["exact"] for n in E4_LIFE}, f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, decay=decay, stages=stages), indent=1))


# ================================================================ E5
def e5_develop(seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"e5_dev_{seed}.pt"
    if path.exists():
        return
    m = new_net(seed)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    items = [render(n, *p) for n in E5_DEV for p in split(n, .5, seed)[0]]
    bt = batches(items, seed)
    t0 = time.process_time()
    curve = []
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = [update(m, opt, next(bt), False) for _ in range(BLOCK)]
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: score(m, sample_rows(n, split(n, .5, seed)[2], seed, 600)) for n in E5_DEV})
            curve.append(row)
            print("e5 dev", seed, stop, {n: v["exact"] for n, v in row["sealed"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e5_dev_{seed}.json").write_text(json.dumps(dict(seed=seed, curve=curve), indent=1))


def e5_panels(seed, scratch):
    taught = {t: compose_triples(t, seed, 8000) for t in TAUGHT_TYPES}
    held = {t: compose_triples(t, seed + 1, 600, exclude=frozenset(taught[t])) for t in TAUGHT_TYPES}
    held.update({t: compose_triples(t, seed + 1, 600) for t in HELD_TYPES})
    teach_items = [render_compose(t, *tr, scratch) for t in TAUGHT_TYPES for tr in taught[t]]
    panel_items = {t: [render_compose(t, *tr, scratch) for tr in trs] for t, trs in held.items()}
    return teach_items, panel_items, taught


def compose_shortcuts(seed, panel_items, taught):
    out = {}
    for t, items in panel_items.items():
        finals = [tt[-3:-1] for _, tt in items]
        freq = Counter(tuple(tt[-3:-1]) for tr in taught.get(t, []) for _, tt in [render_compose(t, *tr, False)]).most_common(1)
        n = len(items)
        o1, o2 = TYPES[t]
        # answer from the second operation only (b o2 c), and from the first only (a o1 b)
        trs = compose_triples(t, seed + 1, 600, exclude=frozenset(taught.get(t, []))) if t in TAUGHT_TYPES else compose_triples(t, seed + 1, 600)
        last_only = sum(((b + c) % 97 if o2 == "+" else (b - c) % 97) == compose_answer(t, a, b, c)[1] for a, b, c in trs)
        first_only = sum(compose_answer(t, a, b, c)[0] == compose_answer(t, a, b, c)[1] for a, b, c in trs)
        out[t] = dict(n=n, frequency=(sum(tuple(f) == freq[0][0] for f in finals) if freq else 0), last_op_only=last_only, first_op_only=first_only)
    return out


def e5_life(arm, seed):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"e5_life_{arm}_{seed}.json"
    m = new_net(seed)
    m.load_state_dict(T.load(OUT / f"e5_dev_{seed}.pt"))
    scratch = arm.startswith("SCR")
    local = arm not in ("DIRbp", "SCRbp", "SCRrbp")
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(1. if local else .3), betas=(.9, .98))
    teach, panels, taught = e5_panels(seed, scratch)
    single = {n: sample_rows(n, split(n, .5, seed)[2], seed, 600) for n in E5_DEV}
    replay = arm in ("SCRr", "SCRrbp")
    store, buffer, rrng = None, [], Random(seed + 13)
    if replay:
        # the episodic store holds what the learner experienced in development: the taught single-step items
        store = Store(seed)
        dev_items = [render(n, *pr) for n in E5_DEV for pr in split(n, .5, seed)[0]]
        for j in range(0, len(dev_items), 256):
            x, y, mask = batch_tensor(dev_items[j:j + 256])
            store.write(x, y, mask)
        buffer = [pr for pr, _ in rrng.sample(dev_items, 2048)]
    if arm == "SCRsh":
        rng = Random(seed + 5)
        finals = [tt[-3:-1] for _, tt in teach]
        rng.shuffle(finals)
        teach = [(p, tt[:-3] + f + [EOS]) for (p, tt), f in zip(teach, finals)]
    sc = compose_shortcuts(seed, panels, taught)
    t0 = time.process_time()
    curve = [dict(step=0, compose={t: score(m, it) for t, it in panels.items()}, single={n: score(m, it) for n, it in single.items()})]
    print("e5", arm, seed, 0, {t: v["final"] for t, v in curve[-1]["compose"].items()}, flush=True)
    if arm != "Z":
        bt = batches(teach, seed + 3)
        for stop in range(BLOCK, STAGE_UPDATES + 1, BLOCK):
            losses = []
            for _ in range(BLOCK):
                batch = next(bt)
                if replay:
                    rp = rrng.sample(buffer, BATCH // 2)
                    outs = respond(m, rp, store)          # self-generated answers: shared path + store
                    ans = [o if (len(o) == 3 and o[-1] == EOS) else [ord("0"), ord("0"), EOS] for o in outs]
                    batch = batch[:BATCH // 2] + list(zip(rp, ans))
                losses.append(update(m, opt, batch, local))
            if stop % 2048 == 0:
                curve.append(dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                                  compose={t: score(m, it) for t, it in panels.items()}, single={n: score(m, it) for n, it in single.items()}))
                print("e5", arm, seed, stop, {t: v["final"] for t, v in curve[-1]["compose"].items()}, "single", {n: v["exact"] for n, v in curve[-1]["single"].items()}, f"cpu {curve[-1]['cpu']:.0f}s", flush=True)
    path.write_text(json.dumps(dict(arm=arm, seed=seed, scratch=scratch, replay=replay, shortcuts=sc, curve=curve), indent=1))


# ================================================================ qualification
def qualify():
    out = {}
    # renderings and oracle
    for n in TASKS:
        t, mo, se = split(n, .25, SEEDS[0])
        assert not (set(t) & set(se)) and all((b, a) in se for a, b in se)
        for a, b in t[:100]:
            p, tt = render(n, a, b)
            assert len(p) == 7 and int(bytes(tt[:2])) == answer(n, a, b) and tt[-1] == EOS
    for t in TYPES:
        for tr in compose_triples(t, 1, 50):
            p, tt = render_compose(t, *tr, True)
            i, f = compose_answer(t, *tr)
            assert len(p) == 10 and int(bytes(tt[:2])) == i and tt[2] == ord(";") and int(bytes(tt[3:5])) == f and tt[-1] == EOS
            p2, tt2 = render_compose(t, *tr, False)
            assert int(bytes(tt2[:2])) == f and len(tt2) == 3
    out["renderings_ok"] = True
    # held compositions disjoint from taught triples; codes distinct
    teach, panels, taught = e5_panels(SEEDS[0], False)
    for t in TAUGHT_TYPES:
        assert not (set(compose_triples(t, SEEDS[0] + 1, 600, exclude=frozenset(taught[t]))) & set(taught[t]))
    assert len({v[2] for v in TASKS.values()} | {b"c"}) == 7
    out["panels_ok"] = dict(teach=len(teach), panels={t: len(v) for t, v in panels.items()})
    # window visibility for compositions: the final digits' window must contain the intermediate and c
    p, tt = render_compose("pm", 24, 81, 7, True)
    x = T.tensor([[BOS, *p, *tt[:3]]])            # inputs up to ';'
    win = T.nn.functional.pad(x, (W - 1, 0), value=BOS).unfold(1, W, 1)[0, -1].tolist()
    out["scratch_window_at_final"] = bytes(b for b in win if b < 256).decode()
    assert "08;" in out["scratch_window_at_final"] and "07" in out["scratch_window_at_final"]
    # local forward/backward: sign path equals manual rule on this net
    m = new_net(1)
    items = [render("add97", a, b) for a, b in split("add97", .25, 1)[0][:64]]
    x, y, mask = batch_tensor(items)
    logits = forward(m, x, True); loss = cross_entropy(logits[mask], y[mask]); loss.backward()
    g_sign = {n: p.grad.clone() for n, p in m.named_parameters()}
    m.zero_grad()
    logits = forward(m, x, False); loss = cross_entropy(logits[mask], y[mask]); loss.backward()
    cos = lambda a, b: float(T.nn.functional.cosine_similarity(a.flatten(), b.flatten(), 0))
    out["sign_vs_true_cosine"] = {n: round(cos(g_sign[n], p.grad), 3) for n, p in m.named_parameters()}
    assert out["sign_vs_true_cosine"]["l3.weight"] > .999
    # store: writes a batch, then prefers the written answers; untouched prompts near uniform
    st = Store(1)
    st.write(x, y, mask)
    lg = st.logits(x)[mask]
    out["store_written_top1_match"] = float((lg.argmax(1) == y[mask]).float().mean())
    assert out["store_written_top1_match"] == 1.0
    x2, y2, mask2 = batch_tensor([render("add97", a, b) for a, b in split("add97", .25, 1)[2][:64]])
    out["store_unseen_logit_max"] = float(st.logits(x2)[mask2].abs().max())
    assert out["store_unseen_logit_max"] == 0.0
    # smoke: E4 life mechanics for each arm, 4 updates per stage, no crash (uses a throwaway birth)
    T.save(new_net(1).state_dict(), OUT / "e4_dev_1.pt") if not (OUT / "e4_dev_1.pt").exists() else None
    global STAGE_UPDATES
    keep = STAGE_UPDATES; STAGE_UPDATES = 4
    for arm in E4_ARMS:
        e4_life(arm, 1)
        (OUT / f"e4_life_{arm}_1.json").unlink()
    STAGE_UPDATES = keep
    (OUT / "e4_dev_1.pt").unlink()
    out["e4_smoke_ok"] = True
    (OUT / "QUALIFY.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "qualify":
        OUT.mkdir(exist_ok=True); qualify()
    elif cmd == "e4dev":
        e4_develop(int(sys.argv[2]))
    elif cmd == "e4":
        e4_life(sys.argv[2], int(sys.argv[3]))
    elif cmd == "e5dev":
        e5_develop(int(sys.argv[2]))
    elif cmd == "e5":
        e5_life(sys.argv[2], int(sys.argv[3]))
