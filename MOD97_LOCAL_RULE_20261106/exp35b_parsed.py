"""E35-B stage 0 (brief OPUS_TASK_E35B_INNATE_CIRCUITS_INVENTORY_YI_20261115.md, revision 2): the upper bound with parsing supplied.

Inventory 乙: a code is K = 8 unit phasors P (..., 8, 2) and one scalar s (...). Four fixed primitives, none of which mentions 89:
mul (angles add), div (angles subtract), big / small (a comparator on the scalars routing a whole code). Shaped as E10's RingNet:
operands as digit pairs, the operator as its byte, no letters, backpropagation, batch 64, 32,768 updates, held-out exact answers every 4,096.

    python3 exp35b_parsed.py run <form> <opt> <frac> <arm> <seed>     # form: angle|complex|table; opt: e10|stream; frac: 0.10|0.25; arm: module|scrambled
    python3 exp35b_parsed.py all [procs]                              # every stage-0 run not yet on file, at most `procs` processes
    python3 exp35b_parsed.py table                                    # writes exp45/e35b/stage0/E35B_STAGE0_TABLE.{json,md}

The primitives, the scrambled maps and module_path are imported by exp35b_stream.py, so that both stages run the same fixed operations.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path
from random import Random

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp11_cell as E          # noqa: E402  (split, DEV, SYM, apply; not edited)

T.set_num_threads(1)
K = 8
OUT0 = HERE / "exp45" / "e35b" / "stage0"
SEEDS = (73129, 93683, 116089)
FORMS = ("angle", "complex", "table")
OPTS = ("e10", "stream")
FRACS = (0.10, 0.25)
UPDATES, BATCH, EVERY = 32768, 64, 4096
OPNAMES = [t[0] for t in E.DEV]     # add sub max min

Code = namedtuple("Code", "P s")


# ---------------------------------------------------------------- the fixed primitives (the brief's lines)
def mul(a, b):
    return T.stack([a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]], -1)     # angles add


def div(a, b):
    return T.stack([a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1]], -1)     # angles subtract


def big(a, b, sa, sb):
    g = T.sigmoid(sa - sb)[..., None, None]
    return g * a + (1 - g) * b


def small(a, b, sa, sb):
    g = T.sigmoid(sa - sb)[..., None, None]
    return (1 - g) * a + g * b


OP = [lambda A, Bq: mul(A.P, Bq.P), lambda A, Bq: div(A.P, Bq.P), lambda A, Bq: big(A.P, Bq.P, A.s, Bq.s), lambda A, Bq: small(A.P, Bq.P, A.s, Bq.s)]


# ---------------------------------------------------------------- the scrambled module (arm X / stage-0 scrambled arm)
SCR_SEED, SCALE_SEED, SCALE_N = 35, 3535, 10000


def random_unit_codes(n, gen):
    ang = T.rand(n, K, generator=gen) * (2 * math.pi)
    return T.stack([ang.cos(), ang.sin()], -1)


def _scrambled():
    W = T.randn(4, 16, 16, 16, generator=T.Generator().manual_seed(SCR_SEED))
    sha_raw = hashlib.sha256(W.numpy().tobytes()).hexdigest()
    g = T.Generator().manual_seed(SCALE_SEED)
    a, b = random_unit_codes(SCALE_N, g).reshape(-1, 16), random_unit_codes(SCALE_N, g).reshape(-1, 16)
    consts = []
    for j in range(4):
        out = T.einsum("bi,bj,ijk->bk", a, b, W[j])
        consts.append(math.sqrt(8) / float(out.norm(dim=-1).mean()))
    Ws = T.stack([W[j] * consts[j] for j in range(4)])
    after = []
    for j in range(4):
        after.append(float(T.einsum("bi,bj,ijk->bk", a, b, Ws[j]).norm(dim=-1).mean()))
    info = dict(generator_seed=SCR_SEED, scale_codes_seed=SCALE_SEED, scale_pairs=SCALE_N, constants=consts, mean_norm_after=after,
                sha256_raw=sha_raw, sha256_scaled_each=[hashlib.sha256(Ws[j].contiguous().numpy().tobytes()).hexdigest() for j in range(4)])
    return Ws, info


W_SCR, SCR_INFO = _scrambled()


def _scr(j):
    def f(A, Bq):
        a, b = A.P.reshape(-1, 16), Bq.P.reshape(-1, 16)
        return T.einsum("bi,bj,ijk->bk", a, b, W_SCR[j]).reshape(A.P.shape)
    return f


OP_SCR = [_scr(j) for j in range(4)]


def module_path(c, g, o, ops=OP):
    """The registers and the fixed operations, as the brief writes them. c: Code with P (B, L, 8, 2), s (B, L); g: (B, L, 2) write gates;
    o: (B, L, 4) operation weights. Every item starts from the neutral code. Returns r: (B, L, 8, 2)."""
    B, L = g.shape[:2]
    A = Bq = Code(P=T.stack([T.ones(B, K), T.zeros(B, K)], -1), s=T.zeros(B))
    r = []
    for t in range(L):
        ga, gb = g[:, t, 0:1], g[:, t, 1:2]
        A = Code(P=(1 - ga[..., None]) * A.P + ga[..., None] * c.P[:, t], s=(1 - ga[:, 0]) * A.s + ga[:, 0] * c.s[:, t])
        Bq = Code(P=(1 - gb[..., None]) * Bq.P + gb[..., None] * c.P[:, t], s=(1 - gb[:, 0]) * Bq.s + gb[:, 0] * c.s[:, t])
        r.append(sum(o[:, t, j:j + 1, None] * ops[j](A, Bq) for j in range(4)))
    return T.stack(r, 1)


# ---------------------------------------------------------------- stage 0 network (E10's RingNet shape)
class ParsedNet(nn.Module):
    def __init__(self, form, scrambled=False, H=128):
        super().__init__()
        self.form, self.scrambled = form, scrambled
        if form == "table":
            self.tens_ang, self.units_ang = nn.Embedding(10, K), nn.Embedding(10, K)
            self.tens_s, self.units_s = nn.Embedding(10, 1), nn.Embedding(10, 1)
        else:
            self.digit = nn.Embedding(10, 16)
            self.enc1 = nn.Linear(32, H)
            self.wa = nn.Linear(H, K if form == "angle" else 2 * K)
            self.ws = nn.Linear(H, 1)
        self.opsel = nn.Embedding(256, 4)
        self.dec1, self.dec2 = nn.Linear(16, H), nn.Linear(H, 20)

    def code(self, tens, units):
        if self.form == "table":
            ang = self.tens_ang(tens) + self.units_ang(units)
            return Code(P=T.stack([ang.cos(), ang.sin()], -1), s=(self.tens_s(tens) + self.units_s(units)).squeeze(-1))
        h = T.tanh(self.enc1(T.cat([self.digit(tens), self.digit(units)], -1)))
        if self.form == "angle":
            ang = self.wa(h)
            P = T.stack([ang.cos(), ang.sin()], -1)
        else:
            P = self.wa(h).reshape(*h.shape[:-1], K, 2)
        return Code(P=P, s=self.ws(h).squeeze(-1))

    def forward(self, a, b, opbyte):
        A, Bq = self.code(*a), self.code(*b)
        w = self.opsel(opbyte).softmax(-1)
        ops = OP_SCR if self.scrambled else OP
        r = sum(w[:, j:j + 1, None] * ops[j](A, Bq) for j in range(4))
        return self.dec2(T.tanh(self.dec1(r.reshape(-1, 16))))


def loss_digits(logits, y):                 # E10's
    return cross_entropy(logits[:, :10], y // 10) + cross_entropy(logits[:, 10:], y % 10)


def digits(v):
    v = T.tensor(v)
    return v // 10, v % 10


def tensors(items):
    a, b, o, y = zip(*items)
    return digits(list(a)), digits(list(b)), T.tensor([ord(E.SYM[x]) for x in o]), T.tensor(y)


def make_opt(m, opt):
    if opt == "e10":
        return T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
    return T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.25, betas=(.9, .98))


@T.no_grad()
def score(m, items):
    a, b, o, y = tensors(items)
    lg = m(a, b, o)
    p = lg[:, :10].argmax(1) * 10 + lg[:, 10:].argmax(1)
    return dict(n=len(items), exact=int((p == y).sum()))


def data(frac, seed):
    taught, held = [], {}
    for t in E.DEV:
        op, mod, _ = t
        tr, he = E.split(t, frac, seed)
        taught += [(a, b, op, E.apply(op, mod, a, b)) for a, b in tr]
        held[op] = [(a, b, op, E.apply(op, mod, a, b)) for a, b in he]
    return taught, held


def run_path(form, opt, frac, arm, seed):
    return OUT0 / f"e35b_s0_{form}_{opt}_f{round(frac * 100)}_{arm}_{seed}.json"


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:       # noqa: BLE001
        return None


def run(form, opt, frac, arm, seed):
    path = run_path(form, opt, frac, arm, seed)
    assert not path.exists(), f"exists, refused: {path}"
    assert arm in ("module", "scrambled") and form in FORMS and opt in OPTS
    assert not (arm == "scrambled" and form != "angle"), "the scrambled arm is run for the angle form only"
    T.manual_seed(seed)
    m = ParsedNet(form, scrambled=(arm == "scrambled"))
    o = make_opt(m, opt)
    taught, held = data(frac, seed)
    order = Random(seed + 1)
    t0 = time.process_time(); curve = []; losses = []
    finite = True
    for u in range(1, UPDATES + 1):
        a, b, ob, y = tensors(order.sample(taught, BATCH))
        loss = loss_digits(m(a, b, ob), y)
        o.zero_grad(); loss.backward(); o.step()
        lv = float(loss.detach()); losses.append(lv); finite &= math.isfinite(lv)
        if u % EVERY == 0:
            per = {op: score(m, it) for op, it in held.items()}
            pct = {op: 100 * v["exact"] / v["n"] for op, v in per.items()}
            tsc = score(m, taught)
            curve.append(dict(step=u, loss=sum(losses) / len(losses), held=per, held_pct=pct, held_mean=sum(pct.values()) / 4,
                              taught_exact_pct=100 * tsc["exact"] / tsc["n"], cpu=time.process_time() - t0))
            losses = []
            print("e35b s0", form, opt, frac, arm, seed, u, {k: round(v, 1) for k, v in pct.items()}, f"cpu {curve[-1]['cpu']:.0f}s", flush=True)
    rec = dict(experiment="E35B", stage=0, form=form, optimizer=opt, optimizer_settings=o.defaults | {"betas": list(o.defaults["betas"])},
               fraction=frac, arm=arm, seed=seed, updates=UPDATES, batch=BATCH, params=sum(p.numel() for p in m.parameters()),
               taught_pairs=len(taught), held_pairs={k: len(v) for k, v in held.items()}, finite_losses=finite, curve=curve,
               scrambled=SCR_INFO if arm == "scrambled" else None, cpu_total=time.process_time() - t0,
               host=dict(platform=platform.platform(), python=sys.version.split()[0], torch=T.__version__, threads=T.get_num_threads()),
               git_head=git_head(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT0.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))


def all_runs():
    out = []
    for frac in FRACS:
        for seed in SEEDS:
            for opt in OPTS:
                for form in FORMS:
                    out.append((form, opt, frac, "module", seed))
                out.append(("angle", opt, frac, "scrambled", seed))
    return out


def run_all(procs):
    todo = [r for r in all_runs() if not run_path(*r).exists()]
    running = []
    while todo or running:
        while todo and len(running) < procs:
            r = todo.pop(0)
            log = OUT0 / "logs" / (run_path(*r).stem + ".log"); log.parent.mkdir(parents=True, exist_ok=True)
            running.append(subprocess.Popen([sys.executable, __file__, "run", *map(str, r)], stdout=open(log, "w"), stderr=subprocess.STDOUT))
        time.sleep(2)
        for p in list(running):
            if p.poll() is not None:
                running.remove(p)
                if p.returncode:
                    print("FAILED", p.args, flush=True)


def table():
    rows = []
    for r in all_runs():
        p = run_path(*r)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        fin = d["curve"][-1]
        rows.append(dict(form=d["form"], optimizer=d["optimizer"], fraction=d["fraction"], arm=d["arm"], seed=d["seed"], finite=d["finite_losses"],
                         final_mean=fin["held_mean"], final_per_op=fin["held_pct"], final_taught=fin["taught_exact_pct"],
                         curve_mean=[c["held_mean"] for c in d["curve"]], cpu=d["cpu_total"]))
    # gate: at f = 0.25 some (form, optimizer) reaches >= 90 mean, final row, in >= 2 of 3 seeds (module arm)
    passing = {}
    for form in FORMS:
        for opt in OPTS:
            v = [x["final_mean"] for x in rows if x["form"] == form and x["optimizer"] == opt and x["fraction"] == 0.25 and x["arm"] == "module"]
            passing[f"{form}/{opt}"] = dict(seeds=len(v), at_least_90=sum(y >= 90 for y in v), passes=sum(y >= 90 for y in v) >= 2)
    forms_pass = {f: any(passing[f"{f}/{o}"]["passes"] for o in OPTS) for f in FORMS}
    if forms_pass["angle"]:
        verdict = "PASS: the stream stage uses the angle form"
    elif forms_pass["complex"]:
        verdict = "PASS: the stream stage uses the complex form"
    elif forms_pass["table"]:
        verdict = "STOP: only the digit tables passed; reported, the coordinator decides"
    else:
        verdict = "STOP: nothing passed; the stream stage does not run"
    cpu = sum(x["cpu"] for x in rows)
    res = dict(runs=len(rows), expected_runs=len(all_runs()), gate=passing, forms_pass=forms_pass, verdict=verdict, cpu_seconds_total=cpu, rows=rows,
               scrambled=SCR_INFO)
    (OUT0 / "E35B_STAGE0_TABLE.json").write_text(json.dumps(res, indent=1))
    L = [f"# E35-B stage 0 table ({len(rows)} of {len(all_runs())} runs; total CPU {cpu:.0f} s)", "",
         "Held-out exact two-digit answers, % of every held-out pair, final row (32,768 updates). Mean over four operations, then each.", "",
         "| fraction | form | optimizer | arm | seed | mean | add | sub | max | min | taught (mean of 4 ops pooled) | mean curve every 4,096 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(rows, key=lambda x: (x["fraction"], x["arm"], x["form"], x["optimizer"], x["seed"])):
        po = x["final_per_op"]
        L.append(f"| {x['fraction']:.2f} | {x['form']} | {x['optimizer']} | {x['arm']} | {x['seed']} | {x['final_mean']:.1f} | "
                 + " | ".join(f"{po[o]:.1f}" for o in OPNAMES) + f" | {x['final_taught']:.1f} | " + " ".join(f"{v:.0f}" for v in x["curve_mean"]) + " |")
    L += ["", "## Gate (f = 0.25, module arm, final-row mean >= 90 in >= 2 of 3 seeds)", "", "| form/optimizer | seeds | seeds >= 90 | passes |", "|---|---|---|---|"]
    L += [f"| {k} | {v['seeds']} | {v['at_least_90']} | {v['passes']} |" for k, v in passing.items()]
    L += ["", f"**Verdict:** {verdict}", ""]
    (OUT0 / "E35B_STAGE0_TABLE.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
    elif sys.argv[1] == "all":
        run_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif sys.argv[1] == "table":
        table()
