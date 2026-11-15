"""E35-A stage 0 (brief OPUS_TASK_E35A_INNATE_CIRCUITS_INVENTORY_JIA_20261115.md, revision 2): the upper bound with parsing supplied.

E10's RingNet (exp10_ring.py, not flat), copied here with MOD = 89 because exp10_ring fixes MOD = 97 at import (nothing existing is
edited). Operands are given as digit pairs and the operator as its byte (exp11_cell.SYM: + - > <); no letters. The four operations are
exp11_cell.DEV's (add, sub mod 89; max, min). Backpropagation (local=False). AdamW(lr .003, weight decay .01) as exp10_ring.develop,
batch 64, 32,768 updates, items drawn by Random(seed + 1).sample(taught items, 64) as exp10_ring.develop draws them.

Arms:  ring   the four fixed operations (exp10_ring.op_add/op_sub/op_max/op_min with MOD = 89, as the brief writes them)
       table  the same network with each operation replaced by a fixed random table drawn from T.Generator().manual_seed(35)

Fractions 0.10 and 0.25 by exp11_cell.split(task, f, seed) (each task its own split); held-out accuracy on every held-out pair of every
task, exact two-digit answer, every 4,096 updates.

    python3 exp35a_parsed.py run <ring|table> <10|25> <seed> [--out DIR]
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from random import Random

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp11_cell as E          # noqa: E402  (split, apply, DEV, SYM; unchanged)
from exp3_state import lin      # noqa: E402

T.set_num_threads(1)
MOD = 89
IDX = T.arange(MOD)
BATCH, BLOCK, UPDATES, EVERY = 64, 512, int(os.environ.get("E35A_PARSED_UPDATES", "32768")), 4096
SEEDS = (73129, 93683, 116089)
OPS = ["add", "sub", "max", "min"]
OPB = {o: ord(E.SYM[o]) for o in OPS}
TASK = {t[0]: t for t in E.DEV}
assert all(TASK[o][1] == MOD for o in OPS)


def op_add(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[:, None] - IDX[None, :]) % MOD])
def op_sub(a, b): return T.einsum("bi,bki->bk", a, b[:, (IDX[None, :] - IDX[:, None]) % MOD])
def op_max(a, b): F = a.cumsum(1) * b.cumsum(1); return T.cat([F[:, :1], F[:, 1:] - F[:, :-1]], 1)
def op_min(a, b): S = a.flip(1).cumsum(1).flip(1) * b.flip(1).cumsum(1).flip(1); return T.cat([S[:, :-1] - S[:, 1:], S[:, -1:]], 1)


def table_op(tab):
    flat = tab.reshape(-1)
    def f(a, b):
        outer = (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)
        return T.zeros(len(a), 89, dtype=a.dtype).index_add(1, flat, outer)
    return f


def random_tables():
    return T.randint(0, 89, (4, 89, 89), generator=T.Generator().manual_seed(35))


INNATE = [op_add, op_sub, op_max, op_min]
TABLES = random_tables()
SCRAMBLED = [table_op(TABLES[j]) for j in range(4)]


class RingNet(nn.Module):
    """exp10_ring.RingNet (flat=False) with MOD = 89 and a choice of the four operations."""

    def __init__(self, ops, H=128):
        super().__init__()
        self.ops = ops
        self.digit = nn.Embedding(10, 16)
        self.enc1 = nn.Linear(32, H); self.enc2 = nn.Linear(H, MOD)
        self.opsel = nn.Embedding(256, 4)
        self.dec = nn.Linear(MOD, 20)

    def bump(self, tens, units, local):
        x = T.cat([self.digit(tens), self.digit(units)], 1)
        return lin(T.tanh(lin(x, self.enc1, local)), self.enc2, local).softmax(1)

    def combine(self, a, b, opbyte, local):
        w = self.opsel(opbyte).softmax(1)
        outs = T.stack([f(a, b) for f in self.ops], 1)
        return (w[:, :, None] * outs).sum(1)

    def forward(self, a, b, o1, local=False):
        x = self.combine(self.bump(*a, local), self.bump(*b, local), o1, local)
        return lin(x, self.dec, local)


def digits(v):
    v = T.tensor(v); return v // 10, v % 10


def batch_single(items):
    a, b, o, y = zip(*items)
    return digits(list(a)), digits(list(b)), T.tensor([OPB[x] for x in o]), T.tensor(y)


def loss_digits(logits, y):
    return cross_entropy(logits[:, :10], y // 10) + cross_entropy(logits[:, 10:], y % 10)


def update(m, opt, items):
    a, b, o, y = batch_single(items)
    loss = loss_digits(m(a, b, o, local=False), y)
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


@T.no_grad()
def score_single(m, items):
    n = ex = 0
    for j in range(0, len(items), 4096):
        a, b, o, y = batch_single(items[j:j + 4096])
        li = m(a, b, o)
        p = li[:, :10].argmax(1) * 10 + li[:, 10:].argmax(1)
        n += len(y); ex += int((p == y).sum())
    return dict(n=n, exact=ex, pct=100 * ex / n)


def items_of(pairs, o):
    return [(a, b, o, E.apply(o, MOD, a, b)) for a, b in pairs]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(arm, fpct, seed, out, cmd):
    frac = {10: .10, 25: .25}[fpct]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"e35a_parsed_{arm}_f{fpct}_{seed}.json"
    assert not path.exists(), f"output exists, refused: {path}"
    T.manual_seed(seed); m = RingNet(INNATE if arm == "ring" else SCRAMBLED)
    opt = T.optim.AdamW(m.parameters(), lr=.003, weight_decay=.01)
    taught, held, counts = [], {}, {}
    for o in OPS:
        t, h = E.split(TASK[o], frac, seed)
        taught += items_of(t, o); held[o] = items_of(h, o); counts[o] = dict(taught=len(t), held=len(h))
        assert not set(t) & set(h)
    order = Random(seed + 1); t0 = time.process_time(); curve = []
    for stop in range(BLOCK, UPDATES + 1, BLOCK):
        losses = [update(m, opt, order.sample(taught, BATCH)) for _ in range(BLOCK)]
        if stop % EVERY == 0 or stop == UPDATES:
            sc = {o: score_single(m, it) for o, it in held.items()}
            row = dict(step=stop, loss=sum(losses) / len(losses), finite=all(v == v and abs(v) != float("inf") for v in losses),
                       cpu=time.process_time() - t0, held=sc, held_mean_pct=sum(v["pct"] for v in sc.values()) / 4,
                       taught_pct=score_single(m, taught)["pct"])
            curve.append(row)
            print("e35a parsed", arm, fpct, seed, stop, f"held {row['held_mean_pct']:.1f}", {o: round(v["pct"], 1) for o, v in sc.items()},
                  f"taught {row['taught_pct']:.1f} cpu {row['cpu']:.0f}s", flush=True)
    rec = dict(experiment="E35A", stage=0, kind="parsed", arm=arm, fraction=frac, seed=seed, updates=UPDATES, batch=BATCH, mod=MOD,
               optimizer=dict(name="AdamW", lr=.003, weight_decay=.01), params=sum(p.numel() for p in m.parameters()), counts=counts,
               tables_sha256=hashlib.sha256(TABLES.numpy().tobytes()).hexdigest(), curve=curve, final_held_mean_pct=curve[-1]["held_mean_pct"],
               cpu_total=time.process_time() - t0, command=cmd,
               host=dict(machine=platform.machine(), platform=platform.platform(), python=sys.version.split()[0], torch=T.__version__,
                         threads=T.get_num_threads()),
               sources_sha256={s: sha_file(HERE / s) for s in ("exp35a_parsed.py", "exp11_cell.py", "exp3_state.py")})
    path.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    out = a[a.index("--out") + 1] if "--out" in a else str(HERE / "exp45" / "e35a" / "stage0")
    assert a[0] == "run" and a[1] in ("ring", "table")
    run(a[1], int(a[2]), int(a[3]), out, cmd)
