"""E8: the persistent-life demonstration (step 5). Predictions: E45_PREDICTIONS.md, E8 section.

One learner (18-byte window, hidden 512), developed by backprop on four operations mod 89 with left noise.
Life, by the sign-symmetric local rule: eight new single-step operations learned in sequence
(add, sub, max, min mod 97; then the same four mod 73), each on its own code byte, then a ninth stage that
teaches two-step compositions of the mod-97 operations (twelve of sixteen operator pairs, plain scratchpad).
After every stage every operation's sealed panel is scored; after the last stage the held operator pairs too.

Arms:  C  store + self-generated replay (the demonstration)   A  no replay (control)
       D  backprop on joint ground truth of everything seen so far (ceiling)
"""
import os
os.environ["EXP45_WINDOW"] = "18"
os.environ.setdefault("EXP45_HIDDEN", "512")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import json
from itertools import accumulate
import sys
import time
from pathlib import Path
from random import Random
from fractions import Fraction

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp45 as X

T.set_num_threads(1)
assert X.W == 18 and X.H == 512
EOS, BOS = X.EOS, X.BOS
OUT = X.OUT
SEEDS = X.SEEDS
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n"),
        ("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
NAME = lambda op, m, c: f"{op}{m}"
BATCH = int(os.environ.get("E8_BATCH", "64"))
if BATCH <= 0 or 393216 % BATCH or 524288 % BATCH:
    raise ValueError("Batch must exactly divide both exposure budgets")
X.BATCH = BATCH
STAGE_UPDATES, COMPOSE_UPDATES = 393216 // BATCH, 524288 // BATCH
BUFFER_CAP = int(os.environ.get("E8_BUFFER_CAP", "1024"))
REPLAY_SAMPLING = os.environ.get("E8_REPLAY_SAMPLING", "uniform")
SAVE_STAGES = os.environ.get("E8_SAVE_STAGES", "0") == "1"
RUN_PREFIX = os.environ.get("E8_RUN_PREFIX", "")
COMPOSE_CHECKPOINTS = int(os.environ.get("E8_COMPOSE_CHECKPOINTS", "0"))
if COMPOSE_CHECKPOINTS < 0:
    raise ValueError("E8_COMPOSE_CHECKPOINTS must be nonnegative")
if BUFFER_CAP < 1 or REPLAY_SAMPLING not in ("uniform", "recency", "balanced", "share"):
    raise ValueError("Invalid replay configuration")
REPLAY_SHARE = Fraction(os.environ.get("E8_REPLAY_SHARE", "1/2"))
SCHEDULE = os.environ.get("E8_SCHEDULE", "blocked")
if not 0 <= REPLAY_SHARE < 1 or (REPLAY_SHARE * X.BATCH).denominator != 1:
    raise ValueError("Replay share must be a whole number of items in the batch, below one.")
if SCHEDULE not in ("blocked", "interleaved", "late_family"):
    raise ValueError("Schedule must be blocked, interleaved or late_family.")
if REPLAY_SAMPLING == "balanced" and SCHEDULE == "interleaved":
    raise ValueError("Balanced replay is supported for per-stage blocked/late_family buffers only")
REPLAY_COUNT = int(REPLAY_SHARE * X.BATCH)
TAG = "" if (REPLAY_SHARE == Fraction(1, 2) and SCHEDULE == "blocked") else (
    f"_rp{REPLAY_SHARE.numerator}of{REPLAY_SHARE.denominator}_{SCHEDULE}")
TAG += (f"_b{BATCH}" if BATCH != 64 else "") + (f"_cap{BUFFER_CAP}" if BUFFER_CAP != 1024 else "") + ("_" + REPLAY_SAMPLING if REPLAY_SAMPLING != "uniform" else "")
PAIRS = [(a, b) for a in SYM.values() for b in SYM.values()]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]
OP_OF = {v: k for k, v in SYM.items()}
CODE97 = {"+": b"r", "-": b"t", ">": b"x", "<": b"n"}


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b):
    op, m, code = task
    return list(code + f"{a:02d}".encode() + SYM[op].encode() + f"{b:02d}".encode() + b"="), list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def render_pair(o1, o2, a, b, c):
    prompt = list(b"c" + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=")
    i = apply(OP_OF[o1], 97, a, b); f = apply(OP_OF[o2], 97, i, c)
    return prompt, list(f"{i:02d}".encode()) + [ord(";")] + list(f"{f:02d}".encode()) + [EOS]


def triples(pair, seed, n, exclude=frozenset()):
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1]))
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


def panel(task, seed, n=600):
    return [render(task, a, b) for a, b in Random(seed + 2).sample(split(task, .25, seed)[1], n)]


def develop(seed):
    path = OUT / f"e8_dev_{seed}.pt"
    if path.exists():
        return
    m = X.new_net(seed)
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=.3, betas=(.9, .98))
    base = [render(t, a, b) for t in DEV for a, b in split(t, .5, seed)[0]]
    rng, order = Random(seed + 21), Random(seed + 1)
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, X.DEV_UPDATES + 1, X.BLOCK):
        losses = [X.update(m, opt, X.with_left_noise(order.sample(base, X.BATCH), rng), False) for _ in range(X.BLOCK)]
        if stop % 4096 == 0 or stop == X.DEV_UPDATES:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={NAME(*t): X.score(m, [render(t, a, b) for a, b in Random(seed + 2).sample(split(t, .5, seed)[1], 600)]) for t in DEV})
            curve.append(row); print("e8 dev", seed, stop, {k: v["exact"] for k, v in row["sealed"].items()}, flush=True)
    T.save(m.state_dict(), path)
    (OUT / f"e8_dev_{seed}.json").write_text(json.dumps(dict(seed=seed, curve=curve), indent=1))


def life(arm, seed):
    path = OUT / f"{RUN_PREFIX}e8_life_{arm}{TAG}_{seed}.json"
    m = X.new_net(seed)
    m.load_state_dict(T.load(OUT / f"e8_dev_{seed}.pt"))
    local = arm != "D"
    life_lr = float(os.environ.get("E8_LR", ".001"))
    life_wd = float(os.environ.get("E8_WD", "1." if local else ".3"))
    if not (life_lr > 0 and life_wd >= 0 and life_lr * life_wd < 1):
        raise ValueError("Invalid life optimizer settings")
    opt = T.optim.AdamW(m.parameters(), lr=life_lr, weight_decay=life_wd, betas=(.9, .98))
    store = X.Store(seed) if arm == "C" else None
    diagnostic = None
    if os.environ.get("E8_DIAGNOSTICS", "0") == "1":
        from e14_diagnostics import Logger
        diagnostic = Logger(path.with_suffix(".diagnostics.jsonl"))
    rng = Random(seed + 9)
    compose_checkpoints = []
    panels = {NAME(*t): panel(t, seed) for t in LIFE}
    panels.update({NAME(*t): panel(t, seed, 300) for t in DEV})
    taught_pairs = {p: triples(p, seed, 2000) for p in TAUGHT_PAIRS}
    comp_panels = {f"{o1}{o2}": [render_pair(o1, o2, *tr) for tr in triples((o1, o2), seed + 1, 600)] for o1, o2 in HELD_PAIRS}
    within = []
    for p in (("+", "+"), ("-", "-"), (">", ">"), ("<", "<")):
        within += [render_pair(*p, *tr) for tr in triples(p, seed + 1, 150, exclude=frozenset(taught_pairs[p]))]
    comp_panels["within"] = within
    buffer, seen_items, buffer_ages = [], [], []
    stage_buffers, admissions = [], []
    interleaved_prompts = {t[2][0]: {} for t in LIFE}
    t0 = time.process_time(); stages = []

    def evaluate(final=False):
        res = {k: X.score(m, it, store) for k, it in panels.items()}
        if final or SCHEDULE == "late_family":
            res["compose"] = {k: X.score(m, it) for k, it in comp_panels.items()}
        return res

    def checkpoint(stage):
        if SAVE_STAGES:
            T.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), torch_rng=T.get_rng_state(),
                        replay_rng=rng.getstate(), stage=stage, seed=seed, arm=arm,
                        buffer=buffer, buffer_ages=buffer_ages),
                   OUT/f"{RUN_PREFIX}e8_state_{arm}{TAG}_{seed}_stage{stage}.pt")
    checkpoint(0)
    stages.append(dict(stage=0, task=None, scores=evaluate()))
    schedule = [(NAME(*t), [render(t, a, b) for a, b in split(t, .25, seed)[0]], STAGE_UPDATES) for t in LIFE]
    if SCHEDULE == "interleaved":
        combined = [item for _, items, _ in schedule for item in items]
        schedule = [("interleaved", combined, len(LIFE) * STAGE_UPDATES)]
    schedule.append(("compose", [render_pair(*p, *tr) for p in TAUGHT_PAIRS for tr in taught_pairs[p]], COMPOSE_UPDATES))
    if SCHEDULE == "late_family":
        schedule = schedule[:4] + [schedule[-1]] + schedule[4:-1]
    for si, (name, items, dose) in enumerate(schedule, 1):
        cur = X.batches(items, seed + si)
        if SCHEDULE == "interleaved" and name == "interleaved":
            # Eight samples per task per full batch, interleaved so a 32-item
            # prefix also contains four per task. This matches task exposure.
            def mixed_batches():
                order = Random(seed + si)
                pools = [[render(t, a, b) for a, b in split(t, .25, seed)[0]] for t in LIFE]
                while True:
                    draws = [order.sample(pool, X.BATCH // len(LIFE)) for pool in pools]
                    yield [row for position in zip(*draws) for row in position]
            cur = mixed_batches()
        if arm == "D":
            seen_items = seen_items + items; joint = X.batches(seen_items, seed + si)
        if REPLAY_SAMPLING == "recency" and buffer:
            weights = list(accumulate(0.5 ** (si-age) for age in buffer_ages))
        losses = []
        presented = set()
        draw_counts = [0] * len(stage_buffers)
        malformed = {"operation": 0, "composition": 0}
        for u in range(dose):
            current_replay_count = 0
            if arm == "D":
                batch = next(joint)
            elif arm == "C" and buffer and REPLAY_COUNT:
                if REPLAY_SAMPLING == "share" and name.endswith("73"):
                    comp = [p for p in buffer if p[0] == 99]
                    ops = [p for p in buffer if p[0] != 99]
                    rp = rng.sample(comp, REPLAY_COUNT // 2) + rng.sample(ops, REPLAY_COUNT // 2)
                elif REPLAY_SAMPLING in ("uniform", "share"):
                    rp = rng.sample(buffer, min(REPLAY_COUNT, len(buffer)))
                elif REPLAY_SAMPLING == "balanced":
                    rp = []
                    for _ in range(min(REPLAY_COUNT, len(buffer))):
                        bi = rng.randrange(len(stage_buffers))
                        rp.append(rng.choice(stage_buffers[bi]))
                else:
                    # Sequential weighted sampling without replacement; same eligible buffer.
                    chosen, used = [], set()
                    while len(chosen) < min(REPLAY_COUNT, len(buffer)):
                        idx = rng.choices(range(len(buffer)), cum_weights=weights, k=1)[0]
                        if idx not in used:
                            chosen.append(buffer[idx]); used.add(idx)
                    rp = chosen
                current_replay_count = len(rp)
                outs = X.respond(m, rp, store)
                if SCHEDULE == "late_family":
                    ans = []
                    for prompt, output in zip(rp, outs):
                        composition = prompt[0] == ord("c")
                        valid = (len(output) == 6 and output[2] == ord(";") and output[-1] == EOS
                                 and all(48 <= output[i] <= 57 for i in (0, 1, 3, 4))) if composition else (len(output) == 3 and output[-1] == EOS)
                        if not valid:
                            malformed["composition" if composition else "operation"] += 1
                        ans.append(output if valid else ([48, 48, 59, 48, 48, EOS] if composition else [48, 48, EOS]))
                    for prompt in rp:
                        # Instrumentation only: identical prompts may be in several stage buffers.
                        memberships = [i for i, sb in enumerate(stage_buffers) if tuple(prompt) in stage_members[i]]
                        assert memberships
                        draw_counts[memberships[0]] += 1
                else:
                    ans = [o if (len(o) == 3 and o[-1] == EOS) else [ord("0"), ord("0"), EOS] for o in outs]
                batch = next(cur)[:X.BATCH - len(rp)] + list(zip(rp, ans))
            else:
                batch = next(cur)
            if SCHEDULE == "late_family" and arm == "C":
                # Current examples only; replayed outputs never become admission provenance.
                for prompt, _ in batch[:X.BATCH - current_replay_count]:
                    presented.add(tuple(prompt))
            target_store = store if (name != "compose" or SCHEDULE == "late_family") else None
            if diagnostic is None:
                losses.append(X.update(m, opt, batch, local, target_store))
            else:
                losses.append(diagnostic.update(m, opt, batch, local, target_store, name, u + 1))
            if name == "compose" and COMPOSE_CHECKPOINTS and (u + 1) % COMPOSE_CHECKPOINTS == 0:
                with T.no_grad():
                    compose_checkpoints.append(dict(update=u + 1, scores={k: X.score(m, it) for k, it in comp_panels.items()}))
            if SCHEDULE == "interleaved" and arm == "C" and name == "interleaved":
                # Replay only prompts already delivered to the learner. Keep a
                # deterministic bounded FIFO of 1,024 distinct prompts per task.
                for prompt, _ in batch[:X.BATCH - current_replay_count]:
                    saved = interleaved_prompts[prompt[0]]
                    key = tuple(prompt)
                    if key not in saved:
                        if len(saved) >= BUFFER_CAP:
                            del saved[next(iter(saved))]
                        saved[key] = prompt
                buffer = [p for saved in interleaved_prompts.values() for p in saved.values()]
        if arm == "C" and ((name != "compose" and SCHEDULE == "blocked") or SCHEDULE == "late_family"):
            eligible = items if SCHEDULE != "late_family" else [(p, t) for p, t in items if tuple(p) in presented]
            additions = [p for p, _ in rng.sample(eligible, min(BUFFER_CAP, len(eligible)))]
            buffer += additions
            buffer_ages += [si] * len(additions)
            stage_buffers.append(additions)
            stage_members = [set(map(tuple, sb)) for sb in stage_buffers]
            if SCHEDULE == "late_family":
                import hashlib
                admissions.append(dict(stage=si, task=name, prompts=additions, n=len(additions),
                    presented_unique=len(presented), all_presented=all(tuple(p) in presented for p in additions),
                    sha256=hashlib.sha256(json.dumps(additions,separators=(",", ":")).encode()).hexdigest(),
                    replay_draws_by_prior_stage=draw_counts, malformed=malformed))
                path.with_suffix(".buffer.json").write_text(json.dumps(admissions, indent=1))
        checkpoint(si)
        final = si == len(schedule)
        stages.append(dict(stage=si, task=name, loss=sum(losses[-512:]) / 512, cpu=time.process_time() - t0, scores=evaluate(final)))
        sc = stages[-1]["scores"]
        print("e8", arm, seed, "after", name, {k: sc[k]["exact"] for k in panels if k.endswith(("97", "73"))}, ({k: v["final"] for k, v in sc["compose"].items()} if final else ""), f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        receipt = dict(arm=arm, seed=seed, stages=stages)
        if COMPOSE_CHECKPOINTS:
            receipt["compose_checkpoints"] = compose_checkpoints
        path.write_text(json.dumps(receipt, indent=1))


def qualify():
    out = {}
    codes = {t[2] for t in DEV + LIFE} | {b"c"}
    assert len(codes) == 13
    for t in DEV + LIFE:
        tr, se = split(t, .25, SEEDS[0]); assert not (set(tr) & set(se))
        for a, b in tr[:50]:
            p, tt = render(t, a, b); assert len(p) == 7 and int(bytes(tt[:2])) == apply(t[0], t[1], a, b)
    for pair in PAIRS:
        for trp in triples(pair, 1, 10):
            p, tt = render_pair(*pair, *trp); i = apply(OP_OF[pair[0]], 97, trp[0], trp[1]); assert int(bytes(tt[:2])) == i and int(bytes(tt[3:5])) == apply(OP_OF[pair[1]], 97, i, trp[2])
    out["codes"] = sorted(c.decode() for c in codes); out["stages"] = [NAME(*t) for t in LIFE] + ["compose"]
    (OUT / "QUALIFY_E8.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop(int(sys.argv[2]))
    else:
        life(sys.argv[1], int(sys.argv[2]))
