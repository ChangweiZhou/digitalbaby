"""E16: the integrated toy learner, the Phase A exit gate (predictions: E45_PREDICTIONS.md, E16).

One E11 cell learner (10-byte window, hidden 512, diagonal memory cell with the exact trace) from the E11 memory birth,
at the adopted settings (decay 0.25 on local arms, fresh-render replay at half of every batch, scan trace), on ONE
stream that carries every family the toy stage has, in six blocks of STAGE_UPDATES updates:
    blocks 1-2  F1: add/sub/max/min mod 97 (codes r t x n), interleaved, k = 0..12 distractor letters (holding)
    blocks 3-4  F2: 12 taught ordered operator pairs on the mod-97 operations, E8 scratchpad form
                    c a a o1 b b o2 c c =  ->  i i ; f f EOS      (no distractors; the window is 10, so the code byte
                    and the first operand leave the window before the final answer, the operators do not)
    blocks 5-6  F3: add/sub/max/min mod 73 (codes s u y z), interleaved, k = 0..12  (a later family: does F1 survive it)
Every block after the first mixes half new items with half replay of stored (family, operands) entries re-rendered fresh.
Scored after every block: F1 and F3 sealed panels at k = 0, 8, 12, 20; the four held cross pairs (+-, -+, ><, <>) and
held triples of taught pairs ("within"); the E11 development operations (mod 89) at k = 12; the silenced-cell audit.
Arms: T  sign rule everywhere + exact trace, decay 0.25, replay (the learner)
      D  backprop through time, each batch drawn from the families seen so far with equal weight, decay 0.3 (the twin)
"""
import os
os.environ.setdefault("E11_WD", "0.25")
os.environ.setdefault("E11_REPLAY", "fresh")
os.environ.setdefault("E11_TRACE", "scan")
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp11_cell as E
import exp45 as X

assert E.WD == 0.25 and E.REPLAY == "fresh" and E.TRACE == "scan" and E.GATES == "single"
OUT, EOS, BATCH, STAGE_UPDATES, DRYTAG = E.OUT, E.EOS, X.BATCH, E.STAGE_UPDATES, E.DRYTAG
# E16-D replay controls on arm T (default reproduces E16): uniform = one buffer of 1,024 entries per block, replay drawn uniformly;
# balanced = the same buffer kept per family, each replay draw from a family chosen uniformly among those in the buffer;
# all_balanced = every presented entry kept per family (distinct), draws balanced per family;
# oracle = the whole batch is the twin's joint stream (fresh items of every family seen so far, equal weight), learned by the local rule.
REPLAY16 = os.environ.get("E16_REPLAY", "uniform")
assert REPLAY16 in ("uniform", "balanced", "all_balanced", "oracle")
RTAG = "" if REPLAY16 == "uniform" else f"_{REPLAY16}"
F1 = E.LIFE                                                                        # mod 97: r t x n
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]   # mod 73 (E8's second four)
SYM_OP = {v: k for k, v in E.SYM.items()}
PAIRS = [(a, b) for a in E.SYM.values() for b in E.SYM.values()]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]
KMAX, KS = 12, E.KS_PANEL
CAP = 6   # longest answer: i i ; f f EOS


# ---------------------------------------------------------------- items: an entry is replayable; render makes the bytes
def render_pair(o1, o2, a, b, c):
    i = E.apply(SYM_OP[o1], 97, a, b); f = E.apply(SYM_OP[o2], 97, i, c)
    prompt = list(b"c" + f"{a:02d}".encode() + o1.encode() + f"{b:02d}".encode() + o2.encode() + f"{c:02d}".encode() + b"=")
    return prompt, list(f"{i:02d}".encode()) + [ord(";")] + list(f"{f:02d}".encode()) + [EOS]


def render(entry, rng):
    if entry[0] == "op":
        _, task, a, b = entry
        return E.render(task, a, b, rng.randint(0, KMAX), rng)
    _, o1, o2, a, b, c = entry
    return render_pair(o1, o2, a, b, c)


def triples(pair, seed, n, exclude=frozenset()):
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1]))
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


def family_entries(fam, seed):
    if fam == "F2":
        return [("comp", o1, o2, a, b, c) for o1, o2 in TAUGHT_PAIRS for a, b, c in triples((o1, o2), seed, 2000)]
    return [("op", t, a, b) for t in (F1 if fam == "F1" else F3) for a, b in E.split(t, .25, seed)[0]]


def entry_stream(entries, seed):
    rng = Random(seed + 77)
    while True:
        order = list(range(len(entries)))
        rng.shuffle(order)
        for i in order:
            yield render(entries[i], rng)


def joint_stream(streams, seed):
    """Arm D: each item from a family chosen uniformly among those seen so far (ground truth, equal weight per family)."""
    rng = Random(seed + 78)
    while True:
        yield next(streams[rng.randrange(len(streams))])


# ---------------------------------------------------------------- panels
def score(m, items, store=None, silence=False):
    outs = E.respond(m, [p for p, _ in items], store, cap=CAP, silence=silence)
    return dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)),
                final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))


def panels(seed):
    P = {E.NAME(*t): E.panel(t, E.split(t, .25, seed)[1], seed) for t in F1 + F3}          # held pairs, k = 0, 8, 12, 20
    P.update({E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(12,), n=300) for t in E.DEV})
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    comp = {f"{o1}{o2}": [render_pair(o1, o2, *tr) for tr in triples((o1, o2), seed + 1, 600)] for o1, o2 in HELD_PAIRS}
    rng = Random(seed + 5)
    comp["within"] = [render_pair(*p, *tr) for p in TAUGHT_PAIRS for tr in triples(p, seed + 2, 50, exclude=taught[p])]
    return P, comp


def shortcut_ceilings(comp):
    """Final answer from a visible-only rule: ignore the first operator (f = o2(a, c), f = o2(b, c)) or the second (f = o1(a, b))."""
    out = {}
    for name, items in comp.items():
        if name == "within":
            continue
        o1, o2 = name
        hits = {"ignore_o1_use_a": 0, "ignore_o1_use_b": 0, "ignore_o2": 0, "answer_is_c": 0}
        for p, t in items:
            a, b, c = int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])); f = int(bytes(t[3:5]))
            hits["ignore_o1_use_a"] += E.apply(SYM_OP[o2], 97, a, c) == f
            hits["ignore_o1_use_b"] += E.apply(SYM_OP[o2], 97, b, c) == f
            hits["ignore_o2"] += E.apply(SYM_OP[o1], 97, a, b) == f
            hits["answer_is_c"] += c == f
        out[name] = {k: v / len(items) for k, v in hits.items()}
    return out


# ---------------------------------------------------------------- life
SCHEDULE = [("F1", 1), ("F1", 2), ("F2", 1), ("F2", 2), ("F3", 1), ("F3", 2)]


def life(arm, seed):
    path = OUT / f"e16_life_{arm}{RTAG if arm != 'D' else ''}{DRYTAG}_{seed}.json"
    m = E.new_net(seed)
    m.load_state_dict(T.load(OUT / f"e11_dev_M_{seed}.pt"))
    local = arm != "D"
    opt = T.optim.AdamW(m.parameters(), lr=.001, weight_decay=(E.WD if local else .3), betas=(.9, .98))
    store = X.Store(seed) if local else None
    rng = Random(seed + 9)
    P, comp = panels(seed)
    entries = {f: family_entries(f, seed) for f in ("F1", "F2", "F3")}
    buffer, seen_streams = [], []
    fbuf, fset = {}, {}      # balanced modes: per-family entry lists (and sets, for all_balanced's distinct entries)
    t0 = time.process_time(); stages = []

    def evaluate():
        res = {n: {k: score(m, it, store) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, store) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], store, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, store, silence=True) for n, it in comp.items() if n != "within"}
        return res

    stages.append(dict(stage=0, task=None, scores=evaluate(), shortcut_ceilings=shortcut_ceilings(comp),
                       entries={f: len(v) for f, v in entries.items()}, settings=dict(wd=E.WD, replay=E.REPLAY, trace=E.TRACE, stage_updates=STAGE_UPDATES, replay16=REPLAY16)))
    for si, (fam, block) in enumerate(SCHEDULE, 1):
        cur = entry_stream(entries[fam], seed + si)
        if arm == "D" or REPLAY16 == "oracle":
            if block == 1:
                seen_streams.append(entry_stream(entries[fam], seed + 100 + si))
            joint = joint_stream(seen_streams, seed + si)
        losses, presented = [], []
        for u in range(STAGE_UPDATES):
            if arm == "D" or REPLAY16 == "oracle":
                batch = [next(joint) for _ in range(BATCH)]
            elif fbuf:
                fams = sorted(fbuf)
                rp = [render(rng.choice(fbuf[fams[rng.randrange(len(fams))]]), rng) for _ in range(BATCH // 2)]
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif buffer:
                rp = [render(e, rng) for e in rng.sample(buffer, BATCH // 2)]   # fresh-render replay: stored entries, new letters and k
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            else:
                batch = [next(cur) for _ in range(BATCH)]; presented += batch
            losses.append(E.update(m, opt, batch, arm, store))
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)), cpu=time.process_time() - t0, scores=evaluate()))
        sc = stages[-1]["scores"]
        print("e16", arm, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in KS] for n in sc if n.endswith("97")},
              "F3@12", [sc[n][12]["exact"] for n in sc if n.endswith("73")],
              "comp.final", {n: v["final"] for n, v in sc["compose"].items()},
              "dev89@12", [sc[n][12]["exact"] for n in sc if n.endswith("89")],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, schedule=SCHEDULE, stages=stages), indent=1))
        if local and REPLAY16 != "oracle":
            # the stage's presented items as replayable entries (family + operands), 1,024 of them, as E11-D/E15
            ents = []
            for p, _ in presented:
                if p[0] == ord("c"):
                    ents.append(("comp", chr(p[3]), chr(p[6]), int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9]))))
                else:
                    task = next(t for t in F1 + F3 if t[2][0] == p[0])
                    ents.append(("op", task, int(bytes(p[1:3])), int(bytes(p[-3:-1]))))
            if REPLAY16 == "uniform":
                buffer += rng.sample(ents, min(1024, len(ents)))
            elif REPLAY16 == "balanced":
                fbuf.setdefault(fam, []).extend(rng.sample(ents, min(1024, len(ents))))
            else:   # all_balanced: every distinct presented entry of the family
                fs = fset.setdefault(fam, set()); fl = fbuf.setdefault(fam, [])
                for e in ents:
                    if e not in fs:
                        fs.add(e); fl.append(e)
            stages[-1]["buffer"] = {f: len(v) for f, v in fbuf.items()} if fbuf else len(buffer)


def qualify():
    """Mechanics only; the trace itself is qualified in QUALIFY_E11.json."""
    out = {}
    rng = Random(1)
    seed = X.SEEDS[0]
    ents = {f: family_entries(f, seed) for f in ("F1", "F2", "F3")}
    out["entries"] = {f: len(v) for f, v in ents.items()}
    # composition rendering and the round trip from bytes back to an entry
    for e in rng.sample(ents["F2"], 200):
        p, t = render(e, rng)
        assert len(p) == 10 and len(t) == 6 and t[2] == ord(";") and t[-1] == EOS
        back = ("comp", chr(p[3]), chr(p[6]), int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])))
        assert back == e, (back, e)
        i = E.apply(SYM_OP[e[1]], 97, e[3], e[4]); assert int(bytes(t[:2])) == i and int(bytes(t[3:5])) == E.apply(SYM_OP[e[2]], 97, i, e[5])
    for e in rng.sample(ents["F1"] + ents["F3"], 200):
        p, t = render(e, rng)
        task = next(tk for tk in F1 + F3 if tk[2][0] == p[0])
        assert ("op", task, int(bytes(p[1:3])), int(bytes(p[-3:-1]))) == e
    # held pairs and held triples are disjoint from the taught sets
    P, comp = panels(seed)
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    for name, items in comp.items():
        for p, _ in items:
            pair = (chr(p[3]), chr(p[6])); tr = (int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])))
            assert (pair in HELD_PAIRS) if name != "within" else (pair in TAUGHT_PAIRS and tr not in taught[pair])
    out["panel_sizes"] = {n: len(v) for n, v in comp.items()}
    out["shortcut_ceilings"] = shortcut_ceilings(comp)
    # window visibility on the composition prompt: positions whose prediction no longer sees the code byte / first operand
    out["composition_prompt_len"] = 10; out["window"] = E.W
    out["code_byte_visible_until_output_index"] = E.W - 10     # 0: visible only for the first output byte
    # a batch of mixed items passes through update on every arm
    m = E.new_net(seed); m.load_state_dict(T.load(OUT / f"e11_dev_M_{seed}.pt"))
    opt = T.optim.AdamW(m.parameters(), lr=.001)
    mixed = [render(e, rng) for e in rng.sample(ents["F1"], 20) + rng.sample(ents["F2"], 24) + rng.sample(ents["F3"], 20)]
    sm = {}
    for arm in ("D", "T"):
        t0 = time.process_time()
        for _ in range(10):
            E.update(m, opt, mixed, arm, None)
        sm[arm] = (time.process_time() - t0) / 10 * 1000
    out["ms_per_update_mixed"] = sm
    t0 = time.process_time(); score(m, comp["+-"]); out["s_per_compose_panel600"] = time.process_time() - t0
    out["birth_compose_final_before_life"] = {n: score(m, it)["final"] for n, it in comp.items()}
    (OUT / "QUALIFY_E16.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    else:
        life(sys.argv[1], int(sys.argv[2]))
