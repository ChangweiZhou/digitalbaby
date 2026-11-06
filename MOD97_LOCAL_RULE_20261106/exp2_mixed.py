"""Experiment 2: mixed-family development, then transfer to a held-out modulus and to decimal addition.

Every record carries a one-byte task code before the operands so that moduli are distinguishable:
  p a+b=  mod 83      q a+b=  mod 89      r a+b=  mod 97      (and the same codes with '-')
  s a+b= / s a-b=     mod 73   <- held out of every development stream
  d a+b=              plain decimal sum, three-digit zero-padded answer (carries, new output length)

Development births (backprop, AdamW lr .001, decoupled decay 0.3, 24,576 updates, batch 64):
  mixed     six tasks: add/sub x {83, 89, 97}, half of each task's unordered pairs
  single    add97 only, same budget (the Experiment 1 birth, now with the task byte)
  shuffled  the mixed stream with answers permuted within each task (sham)

Life arms (12,288 updates): sign-symmetric feedback, AdamW lr .001, decay 1.0, all layers plastic
  M  mixed birth    S  single birth    H  shuffled birth    C  random birth    D  random birth, backprop decay 0.3
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from random import Random

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mod97_local as M
import exp1_transfer as X

T.set_num_threads(1)
EOS, BOS = M.EOS, M.BOS
SEEDS = M.SEEDS
BATCH, BLOCK = 64, 512
OUT = Path(__file__).resolve().parent / "exp2"
DEV_UPDATES = 24576
LIFE_UPDATES = 12288
CODE = {83: b"p", 89: b"q", 97: b"r", 73: b"s", None: b"d"}
DEV_TASKS = [(op, m) for m in (83, 89, 97) for op in ("add", "sub")]
LIFE_TASKS = {"add73": ("add", 73), "sub73": ("sub", 73), "dec": ("add", None)}
LIFE_ARMS = ("M", "S", "H", "C", "D")


WINDOW = 10
HID = int(os.environ.get("EXP2_HIDDEN", "128"))   # hidden width; the width test sets 256 and 512
TAG = "" if HID == 128 else f"_h{HID}"   # the 8-byte window of earlier rounds cannot see the task code or the first operand's units digit by the third answer digit


def task_key(task):
    """Deterministic small integer per task (Python's str hash is randomized per process)."""
    op, m = task
    return (m or 0) * 2 + (op == "sub")


def operand_range(task):
    op, m = task
    return m if m else 100


def answer(task, a, b):
    op, m = task
    if m is None:
        return a + b
    return (a + b) % m if op == "add" else (a - b) % m


def render(task, a, b):
    op, m = task
    sym = b"+" if op == "add" else b"-"
    prompt = CODE[m] + f"{a:02d}".encode() + sym + f"{b:02d}".encode() + b"="
    y = answer(task, a, b)
    digits = f"{y:03d}" if m is None else f"{y:02d}"
    return list(prompt) + list(digits.encode()) + [EOS]


PROMPT_LEN = 7


def split(task, frac, seed):
    n = operand_range(task)
    groups = [(a, b) for a in range(n) for b in range(a, n)]
    rng = Random(seed * 1000 + round(frac * 100) + task_key(task))
    rng.shuffle(groups)
    k = round(frac * len(groups))
    taught, rest = groups[:k], groups[k:]
    monitor, sealed = rest[0::2], rest[1::2]
    expand = lambda gs: [p for a, b in gs for p in (((a, b), (b, a)) if a != b else ((a, b),))]
    return expand(sorted(taught)), expand(sorted(monitor)), expand(sorted(sealed))


def score(m, task, pairs):
    rows = [render(task, *p) for p in pairs]
    prompts = [r[:PROMPT_LEN] for r in rows]
    gold = [r[PROMPT_LEN:] for r in rows]
    outs = M.respond(m, prompts)
    exact = {p: o == g for p, o, g in zip(pairs, outs, gold)}
    op, mod = task
    if mod is None:
        wrap = lambda a, b: a + b >= 100            # three-digit answers
        carry = lambda a, b: a % 10 + b % 10 >= 10
    elif op == "add":
        wrap = lambda a, b: a + b >= mod
        carry = lambda a, b: a % 10 + b % 10 >= 10
    else:
        wrap = lambda a, b: a < b
        carry = lambda a, b: a % 10 < b % 10
    return dict(n=len(pairs), exact=sum(exact.values()),
                first=sum(o[:1] == g[:1] for o, g in zip(outs, gold)),
                eos=sum(EOS in o for o in outs),
                both_orders=sum(exact[a, b] and exact[b, a] for a, b in pairs if a < b),
                both_order_n=sum(a < b for a, b in pairs),
                wrap=[sum(exact[a, b] for a, b in pairs if wrap(a, b)), sum(wrap(a, b) for a, b in pairs)],
                carry=[sum(exact[a, b] for a, b in pairs if carry(a, b)), sum(carry(a, b) for a, b in pairs)])


def shortcuts(task, taught, sealed):
    op, mod = task
    ans = lambda a, b: answer(task, a, b)
    ts = set(taught)
    canon = {(min(a, b), max(a, b)) for a, b in taught}
    freq = Counter(ans(a, b) for a, b in taught).most_common(1)[0][0]
    out = dict(n=len(sealed), exact_lookup=sum(p in ts for p in sealed),
               canonical_lookup=sum((min(a, b), max(a, b)) in canon for a, b in sealed),
               frequency=sum(ans(a, b) == freq for a, b in sealed))
    near = 0
    for a, b in sealed:
        best = min(taught, key=lambda q: abs(q[0] - a) + abs(q[1] - b))
        near += ans(*best) == ans(a, b)
    out["nearest_taught"] = near
    if mod is None:
        out["no_carry_digitwise"] = sum(((a // 10 + b // 10) % 10) * 10 + (a % 10 + b % 10) % 10 == ans(a, b) for a, b in sealed)
    elif op == "add":
        out["no_wrap"] = sum(a + b == ans(a, b) for a, b in sealed)
        for dm in (83, 89, 97):
            out[f"mod{dm}_answer"] = sum((a + b) % dm == ans(a, b) for a, b in sealed)
    else:
        out["abs_diff"] = sum(abs(a - b) == ans(a, b) for a, b in sealed)
        for dm in (83, 89, 97):
            out[f"mod{dm}_answer"] = sum((a - b) % dm == ans(a, b) for a, b in sealed)
    return out


# ---------------------------------------------------------------- development
def dev_rows(seed, birth):
    tasks = DEV_TASKS if birth in ("mixed", "shuffled") else [("add", 97)]
    rows = []
    for task in tasks:
        taught, _, _ = split(task, .5, seed)
        trows = [render(task, *p) for p in taught]
        if birth == "shuffled":
            rng = Random(seed + 4242 + task_key(task))
            answers = [r[PROMPT_LEN:-1] for r in trows]
            rng.shuffle(answers)
            trows = [r[:PROMPT_LEN] + a + [EOS] for r, a in zip(trows, answers)]
        rows += trows
    return rows


def develop(seed, birth):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"dev_{birth}{TAG}_{seed}.pt"
    if path.exists():
        return
    T.manual_seed(seed)
    m = M.Net(W=WINDOW, H=HID)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    rows = dev_rows(seed, birth)
    F = dict(drop=0., rng=T.Generator().manual_seed(seed + 7))
    t0 = time.process_time()
    curve = []
    batches = X.stream(rows, seed)
    for stop in range(BLOCK, DEV_UPDATES + 1, BLOCK):
        losses = [M.update_bp(m, F, opt, r) for r in islice(batches, BLOCK)]
        if stop % 4096 == 0 or stop == DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0, sealed={})
            for task in DEV_TASKS:
                _, _, sealed = split(task, .5, seed)
                s = score(m, task, sealed)
                row["sealed"][f"{task[0]}{task[1]}"] = [s["exact"], s["n"]]
            curve.append(row)
            print("dev", birth, seed, stop, row["sealed"], flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"dev_{birth}{TAG}_{seed}.json").write_text(json.dumps(dict(seed=seed, birth=birth, rows=len(rows), curve=curve), indent=1))


# ---------------------------------------------------------------- life
def make_life(arm, seed):
    birth = {"M": "mixed", "S": "single", "H": "shuffled"}.get(arm)
    T.manual_seed(seed)
    m = M.Net(W=WINDOW, H=HID)
    if birth:
        m.load_state_dict(T.load(OUT / f"dev_{birth}{TAG}_{seed}.pt"))
    for p in m.parameters():
        p.requires_grad_(True)
    if arm == "D":
        opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    else:
        opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=1., betas=(.9, .98))
    F = dict(mode="sign", drop=0., rng=T.Generator().manual_seed(seed + 7))
    return m, F, opt


def life(arm, seed, name, frac):
    OUT.mkdir(exist_ok=True)
    task = LIFE_TASKS[name]
    path = OUT / f"life_{name}_f{round(frac * 100)}_{arm}{TAG}_{seed}.json"
    taught, monitor, sealed = split(task, frac, seed)
    m, F, opt = make_life(arm, seed)
    plastic = [n for n, _ in m.named_parameters()]
    rows = [render(task, *p) for p in taught]
    before = score(m, task, sealed)
    t0 = time.process_time()
    curve = []
    batches = X.stream(rows, seed)
    for stop in range(BLOCK, LIFE_UPDATES + 1, BLOCK):
        if arm == "D":
            losses = [M.update_bp(m, F, opt, r) for r in islice(batches, BLOCK)]
        else:
            losses = [X.sign_update(m, F, opt, r, plastic) for r in islice(batches, BLOCK)]
        row = dict(step=stop, signal=sum(losses) / len(losses), taught=score(m, task, taught),
                   monitor=score(m, task, monitor), sealed=score(m, task, sealed), cpu=time.process_time() - t0)
        curve.append(row)
        print(name, frac, arm, seed, stop, f"taught {row['taught']['exact']}/{row['taught']['n']}",
              f"sealed {row['sealed']['exact']}/{row['sealed']['n']}", f"cpu {row['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, task=name, frac=frac, taught_n=len(taught),
                                        sealed_before=before, shortcuts=shortcuts(task, taught, sealed), curve=curve), indent=1))


# ---------------------------------------------------------------- qualification
def qualify():
    out = {}
    for name, task in list(LIFE_TASKS.items()) + [(f"{t[0]}{t[1]}", t) for t in DEV_TASKS]:
        n = operand_range(task)
        for frac in ((.1, .25) if name in LIFE_TASKS else (.5,)):
            t, mo, se = split(task, frac, SEEDS[0])
            assert len(t) + len(mo) + len(se) == n * n and not (set(t) & set(se)) and not (set(t) & set(mo))
            assert all((b, a) in se for a, b in se)
            for a, b in t[:500] + se[:500]:
                r = render(task, a, b)
                assert len(r[:PROMPT_LEN]) == PROMPT_LEN and int(bytes(r[PROMPT_LEN:-1])) == answer(task, a, b) and r[-1] == EOS
            if name in LIFE_TASKS:
                out[f"{name}_f{frac}"] = dict(sizes=[len(t), len(mo), len(se)], shortcuts=shortcuts(task, t, se))
    # task codes are distinct and prompts of different tasks never collide
    prompts = {}
    for task in DEV_TASKS + list(LIFE_TASKS.values()):
        for a, b in ((1, 2), (50, 60), (72, 3)):
            if max(a, b) < operand_range(task):
                key = tuple(render(task, a, b)[:PROMPT_LEN])
                assert key not in prompts or prompts[key] == task, ("prompt collision", task, prompts[key])
                prompts[key] = task
    out["prompt_collisions"] = 0
    # development streams: mixed has six tasks, shuffled keeps per-task answer marginals and changes most rows
    mixed, sham = dev_rows(SEEDS[0], "mixed"), dev_rows(SEEDS[0], "shuffled")
    assert len(mixed) == len(sham)
    assert Counter(tuple(r[PROMPT_LEN:]) for r in mixed) == Counter(tuple(r[PROMPT_LEN:]) for r in sham)
    out["dev_rows"] = dict(mixed=len(mixed), single=len(dev_rows(SEEDS[0], "single")),
                           shuffled_rows_changed=sum(a != b for a, b in zip(mixed, sham)) / len(mixed))
    # held modulus 73 and decimal never appear in any development stream
    codes = {r[0] for r in mixed} | {r[0] for r in dev_rows(SEEDS[0], "single")}
    assert CODE[73][0] not in codes and CODE[None][0] not in codes
    out["held_codes_absent_from_development"] = True
    # smoke: 16 updates each life arm from a random birth, finite
    for arm in ("C", "D"):
        m, F, opt = make_life(arm, 1)
        plastic = [n for n, _ in m.named_parameters()]
        t, _, _ = split(LIFE_TASKS["dec"], .25, 1)
        rows = [render(LIFE_TASKS["dec"], *p) for p in t]
        for r in islice(X.stream(rows, 1), 16):
            l = M.update_bp(m, F, opt, r) if arm == "D" else X.sign_update(m, F, opt, r, plastic)
        assert l == l
        out[f"smoke_{arm}"] = l
    OUT.mkdir(exist_ok=True)
    (OUT / "QUALIFY.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "qualify":
        qualify()
    elif cmd == "develop":
        develop(int(sys.argv[2]), sys.argv[3])
    elif cmd == "life":
        life(sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
