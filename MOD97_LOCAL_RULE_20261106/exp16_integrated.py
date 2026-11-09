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
# oracle = the whole batch is the twin's joint stream (fresh items of every family seen so far, equal weight), learned by the local rule;
# share = per-family buffers of 1,024 per block, the replay half split equally among the EARLIER families (the current family is never
#         replayed), so each earlier family's exposure rises (F1 50% of the batch during F2; F1 and F2 25% each during F3).
# Arm M (added 8 Nov 12:30 EST, before E16-D's results): backprop through time at decay 0.25 on exactly arm T's uniform-replay batches,
# so the rule is the only difference from T.
# reservoir (E16-R, 8 Nov 14:45 EST): per-family reservoir of at most E16_CAP entries (RCAP; CAP below is the response length cap) filled by Algorithm R over the family's
#         presentations (one counter and one fixed-size list per family: bounded memory, O(1) work per event); draws balanced per family.
# total (E16-R2, 8 Nov 16:25 EST): ONE reservoir of E16_CAP entries over all presentations of the whole life, no family label anywhere
#         (Algorithm R with a single counter); the replay half is drawn uniformly from it. Bounded by a total budget, label-free.
REPLAY16 = os.environ.get("E16_REPLAY", "uniform")
# raw (E16-R3, 8 Nov 17:30 EST): as total, but the reservoir holds the presented BYTES (prompt and answer exactly as they appeared) and
#         replays them verbatim: no parsing into task and operands, no re-rendering, no recomputed answer. The parser is out of the replay path.
assert REPLAY16 in ("uniform", "balanced", "all_balanced", "oracle", "share", "reservoir", "total", "raw")
RCAP = int(os.environ.get("E16_CAP", "4096"))
RTAG = "" if REPLAY16 == "uniform" else (f"_res{RCAP}" if REPLAY16 == "reservoir" else f"_tot{RCAP}" if REPLAY16 == "total" else f"_raw{RCAP}" if REPLAY16 == "raw" else f"_{REPLAY16}")
# E16-S / E16-L (8 Nov 19:10 EST), the environment-side scaffolds, one change each:
#   E16_STREAM=blocks|drift   drift: no stages; each new item's family is drawn from weights that move smoothly through the life
#                             (raised-cosine bumps centred where the blocks were, half-width a third of the life), scored at the same six checkpoints
#   E16_ADMIT=block|online    online: the raw reservoir admits each presented item as it arrives (Algorithm R per item) instead of at checkpoint ends
#   E16_LOSS=answer|all       all: next-byte loss on every byte of every item (prompt and answer), the answer mask removed; scoring unchanged
STREAM16 = os.environ.get("E16_STREAM", "blocks"); ADMIT16 = os.environ.get("E16_ADMIT", "block"); LOSS16 = os.environ.get("E16_LOSS", "answer")
#   E16_STORE=on|off          off: the E8 exact-key store is absent from the local arms (no writes, no logit bias at scoring); the architecture audit (E16-A)
#   E16_ARCH=window|cell1|bank  (E17, 8 Nov 22:30 EST) window: unchanged; cell1 / bank: the window removed, see arch17.py; births e17_dev_{arch}_{seed}
ARCH16 = os.environ.get("E16_ARCH", "window"); assert ARCH16 in ("window", "cell1", "bank", "cell1d2", "bank2")
STORE16 = os.environ.get("E16_STORE", "on" if ARCH16 == "window" else "off"); assert STORE16 in ("on", "off")
if ARCH16 != "window":
    import arch17
    arch17.install(ARCH16)
PFX = "e16" if ARCH16 == "window" else "e17"
#   E16_FREEZE=none|bank1  (E17 Phase 1b): bank1 freezes the first bank's write path (l1, lp, lc, lw) after birth; only the second bank, the
#                           projections from the first bank's state and the readout learn in life. Separates the truncated first-bank trace from the rest.
FREEZE16 = os.environ.get("E16_FREEZE", "none"); assert FREEZE16 in ("none", "bank1")
assert FREEZE16 == "none" or ARCH16 == "bank2", "E16_FREEZE=bank1 is defined for bank2 only"
#   E16_FEEDBACK=sign|exact  (E17 Phase 1c): exact replaces the sign-symmetric feedback through every layer by the true transpose (SignLinear -> linear)
#                           while the state synapses still learn from the eligibility trace (through_time stays off for arm T): separates the error
#                           signal delivered to the state from the trace that credits its synapses. Arm T only; D and M already use exact feedback.
FEEDBACK16 = os.environ.get("E16_FEEDBACK", "sign"); assert FEEDBACK16 in ("sign", "exact")
if FEEDBACK16 == "exact":
    _fwd16 = E.forward
    E.forward = lambda m, x, local=False, through_time=False, keep=False, silence=False: _fwd16(m, x, False, through_time, keep, silence)
#   E16_TRACEUP=none|proj    (E17 Phase 1d): proj extends the scan trace one layer upstream, to the projection feeding each bank's write (lp2; lp when
#                           the first bank learns), replacing its within-step credit by the exact credit through time given Lambda (arch17.TRACEUP).
TRACEUP16 = os.environ.get("E16_TRACEUP", "none"); assert TRACEUP16 in ("none", "proj")
assert TRACEUP16 == "none" or ARCH16 != "window", "E16_TRACEUP is defined for the E17 architectures only"
if ARCH16 != "window":
    arch17.TRACEUP = TRACEUP16 == "proj"; arch17.FB_SIGN = FEEDBACK16 == "sign"
#   E16_BOOKDECAY=on|off     (9 Nov audit, Fable 5.2 memo section 3): under the local arms the byte embeddings of every prompt symbol that enters only
#                           through the state (operators, letters, task codes) receive no gradient, and AdamW's decay erased them (factor 0.215 per
#                           block), which confounds E17 Phases 1-1c. "off" puts `book` in a no-decay group in lives AND births (birth file tagged
#                           `_nbd`), and the frozen arm freezes `book` with the first bank. Default "on" reproduces every receipt made before.
BOOKDECAY16 = os.environ.get("E16_BOOKDECAY", "on"); assert BOOKDECAY16 in ("on", "off")
assert BOOKDECAY16 == "on" or ARCH16 != "window", "E16_BOOKDECAY=off is defined for the E17 architectures only"
BTAG = "_nbd" if BOOKDECAY16 == "off" else ""
ATAG = ("" if ARCH16 == "window" else f"_{ARCH16}") + BTAG + ("_frz1" if FREEZE16 == "bank1" else "") + ("_xfb" if FEEDBACK16 == "exact" else "") + ("_tup" if TRACEUP16 == "proj" else "")


def adamw16(m, lr, wd, frozen=()):
    """AdamW over the learnable parameters; with E16_BOOKDECAY=off the embedding table is in a group of its own with no decay."""
    learn = [(n_, p_) for n_, p_ in m.named_parameters() if not n_.startswith(frozen)]
    for n_, p_ in m.named_parameters():
        p_.requires_grad_(not n_.startswith(frozen))
    if BOOKDECAY16 == "off":
        groups = [{"params": [p_ for n_, p_ in learn if not n_.startswith("book.")]},
                  {"params": [p_ for n_, p_ in learn if n_.startswith("book.")], "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [p_ for n_, p_ in learn]
    return T.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(.9, .98))


@T.no_grad()
def zero_grad_report(m, x):
    """After an update, before zero_grad: which parameters got no gradient at all, and how many embedding rows present in the batch got none
    (9 Nov audit: the check that would have caught the decayed embeddings in E17 Phase 1)."""
    none = [n_ for n_, p_ in m.named_parameters() if p_.requires_grad and (p_.grad is None or not bool(p_.grad.any()))]
    rows = sorted(set(int(v) for v in x.unique().tolist()))
    g = m.book.weight.grad
    dead = [r for r in rows if g is None or not bool(g[r].any())] if m.book.weight.requires_grad else rows
    return dict(params_without_gradient=none, book_rows_in_batch=len(rows), book_rows_in_batch_without_gradient=len(dead),
                dead_rows=[chr(r) if 32 <= r < 127 else r for r in dead])
assert STREAM16 in ("blocks", "drift") and ADMIT16 in ("block", "online") and LOSS16 in ("answer", "all")
assert ADMIT16 == "block" or REPLAY16 == "raw", "online admission is implemented for the raw reservoir only"
RTAG += ("_drift" if STREAM16 == "drift" else "") + ("_online" if ADMIT16 == "online" else "") + ("_lossall" if LOSS16 == "all" else "") + ("_nostore" if STORE16 == "off" and ARCH16 == "window" else "")   # off is the default for the windowless architectures
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
    if entry[0] == "raw":          # verbatim replay of presented bytes
        return list(entry[1]), list(entry[2])
    if entry[0] == "op":
        _, task, a, b = entry
        return E.render(task, a, b, rng.randint(0, KMAX), rng)
    _, o1, o2, a, b, c = entry
    return render_pair(o1, o2, a, b, c)


def triples(pair, seed, n, exclude=frozenset(), split="taught"):
    """Taught triples keep the original seed (training streams unchanged). Held and within panels are keyed by split name and pair
    (audit of 8 Nov 2026: `seed + 1` for the held +- pair collided with the taught -> pair's seed, so those 600 operand triples were shared;
    the pairing and prompts were still held out). Pass exclude = every taught triple so the panels are operand-disjoint from all teaching."""
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split == "taught" else Random(f"{split}|{seed}|{pair[0]}{pair[1]}")
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


def update16(m, opt, items, rule, store):
    """E.update, with the answer mask replaced by every-byte prediction when E16_LOSS=all (default path calls E.update unchanged)."""
    if LOSS16 == "answer":
        return E.update(m, opt, items, rule, store)
    x, y, mask = X.batch_tensor(items)
    mask = y != E.BOS            # every real byte of prompt and answer; padding (BOS) excluded
    logits, it = E.forward(m, x, local=(rule != "D"), through_time=(rule == "D"), keep=(rule == "T"))
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad(); loss.backward()
    if rule == "T":
        (E.trace_grads_scan if E.TRACE == "scan" else E.trace_grads)(m, it)
    opt.step()
    if store is not None:
        store.write(x, y, mask)
    return float(loss.detach())


def drift_weights(u, total):
    """Family weights at update u of total: raised-cosine bumps centred at 1/6, 1/2, 5/6 of the life, half-width 1/2
    (F1 alone at the start, F1 and F2 equal at a third, F2 alone at the middle, F2 and F3 equal at two thirds, F3 alone at the end)."""
    import math
    f = u / total
    return {fam: max(0.0, math.cos(math.pi * (f - c) / 0.5)) if abs(f - c) < 0.5 else 0.0 for fam, c in (("F1", 1 / 6), ("F2", 1 / 2), ("F3", 5 / 6))}


# ---------------------------------------------------------------- panels
def score(m, items, store=None, silence=False):
    outs = E.respond(m, [p for p, _ in items], store, cap=CAP, silence=silence)
    return dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)),
                final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))


def panels(seed):
    ks = E.KS_PANEL + ((40,) if ARCH16 != "window" else ())                                   # E17 adds k = 40, beyond any window
    P = {E.NAME(*t): E.panel(t, E.split(t, .25, seed)[1], seed, ks=ks) for t in F1 + F3}   # held pairs, k = 0, 8, 12, 20 (, 40)
    P.update({E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(12,), n=300) for t in E.DEV})
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    all_taught = frozenset().union(*taught.values())
    comp = {f"{o1}{o2}": [render_pair(o1, o2, *tr) for tr in triples((o1, o2), seed, 600, exclude=all_taught, split="held")] for o1, o2 in HELD_PAIRS}
    comp["within"] = [render_pair(*p, *tr) for p in TAUGHT_PAIRS for tr in triples(p, seed, 50, exclude=all_taught, split="within")]
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
    DTAG = ("_drift" if STREAM16 == "drift" else "") + ("_lossall" if LOSS16 == "all" else "")   # arm D carries the stream/loss tags, not the replay tag
    path = OUT / f"{PFX}_life_{arm}{ATAG}{RTAG if arm != 'D' else DTAG}{DRYTAG}_{seed}.json"
    m = E.new_net(seed)
    birth_file = OUT / (f"e11_dev_M_{seed}.pt" if ARCH16 == "window" else f"e17_dev_{ARCH16}{BTAG}{E.DEVTAG}_{seed}.pt")
    import hashlib
    birth_sha = hashlib.sha256(birth_file.read_bytes()).hexdigest()
    m.load_state_dict(T.load(birth_file))
    assert STREAM16 == "blocks" or arm == "D" or REPLAY16 in ("raw", "total", "oracle"), "the drifting stream is implemented for the raw/total reservoirs and the twin only"
    local = arm not in ("D", "M")
    admits = local or arm == "M"       # arm M keeps T's whole data pipeline, including online admission (9 Nov: before this line M admitted at block ends only)
    FROZEN = ((("book.",) if BOOKDECAY16 == "off" else ()) + ("l1.", "lp.", "lc.", "lw.")) if FREEZE16 == "bank1" else ()
    opt = adamw16(m, .001, (E.WD if arm != "D" else .3), FROZEN)
    store = X.Store(seed) if (local and STORE16 == "on") else None
    rule = "D" if arm == "M" else arm      # arm M: T's data pipeline, D's update
    rng = Random(seed + 9)
    P, comp = panels(seed)
    entries = {f: family_entries(f, seed) for f in ("F1", "F2", "F3")}
    buffer, seen_streams = [], []
    fbuf, fset, fseen = {}, {}, {}      # balanced modes: per-family entry lists (sets for all_balanced's distinct entries; counters for reservoir)
    t0 = time.process_time(); stages = []

    def evaluate():
        res = {n: {k: score(m, it, store) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, store) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], store, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, store, silence=True) for n, it in comp.items() if n != "within"}
        return res

    stages.append(dict(stage=0, task=None, scores=evaluate(), shortcut_ceilings=shortcut_ceilings(comp),
                       entries={f: len(v) for f, v in entries.items()}, settings=dict(wd=opt.defaults["weight_decay"], replay=E.REPLAY, trace=E.TRACE, stage_updates=STAGE_UPDATES, replay16=REPLAY16, cap=(RCAP if REPLAY16 in ("reservoir", "total", "raw") else None), stream=STREAM16, admit=ADMIT16, loss=LOSS16, arch=ARCH16, freeze=FREEZE16, frozen=list(FROZEN), bookdecay=BOOKDECAY16, feedback=FEEDBACK16, traceup=TRACEUP16, birth=birth_file.name, birth_sha256=birth_sha, panels="held/within keyed by split, operand-disjoint from all taught triples (8 Nov audit)",
                                     arm=arm, rule=rule, store=store is not None)))   # E16-V: settings.wd is now the optimizer's actual decay (arm D runs at .3)))
    fstreams = {f: entry_stream(entries[f], seed + 200 + i) for i, f in enumerate(("F1", "F2", "F3"))} if STREAM16 == "drift" else None
    drng = Random(seed + 300); TOTAL = STAGE_UPDATES * len(SCHEDULE); nseen = {}
    def drift_item(u):
        w = drift_weights(u, TOTAL); fams = [f for f in ("F1", "F2", "F3") if w[f] > 0] or ["F3"]
        r = drng.random() * sum(w[f] for f in fams); acc = 0.0
        for f in fams:
            acc += w[f]
            if r <= acc:
                return next(fstreams[f])
        return next(fstreams[fams[-1]])
    for si, (fam, block) in enumerate(SCHEDULE, 1):
        cur = entry_stream(entries[fam], seed + si)
        if arm == "D" or REPLAY16 == "oracle":
            if STREAM16 == "drift":
                for f in ("F1", "F2", "F3"):     # the twin's joint stream admits a family once its weight is positive at this checkpoint's start
                    if drift_weights((si - 1) * STAGE_UPDATES, TOTAL)[f] > 0 and f not in nseen:
                        nseen[f] = True; seen_streams.append(entry_stream(entries[f], seed + 100 + len(nseen)))
            elif block == 1:
                seen_streams.append(entry_stream(entries[fam], seed + 100 + si))
            joint = joint_stream(seen_streams, seed + si)
        losses, presented = [], []
        for u in range(STAGE_UPDATES):
            npres_before = len(presented)
            if arm == "D" or REPLAY16 == "oracle":
                batch = [next(joint) for _ in range(BATCH)]
            elif fbuf and REPLAY16 == "share" and sorted(fbuf) != [fam]:
                fams = [f for f in sorted(fbuf) if f != fam]
                rp = [render(rng.choice(fbuf[fams[i % len(fams)]]), rng) for i in range(BATCH // 2)]
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif REPLAY16 in ("total", "raw") and fbuf:
                rp = [render(rng.choice(fbuf["all"]), rng) for _ in range(BATCH // 2)]
                new = [(drift_item(si * STAGE_UPDATES - STAGE_UPDATES + u) if STREAM16 == "drift" else next(cur)) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif fbuf and REPLAY16 != "share":
                fams = sorted(fbuf)
                rp = [render(rng.choice(fbuf[fams[rng.randrange(len(fams))]]), rng) for _ in range(BATCH // 2)]
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif buffer:
                rp = [render(e, rng) for e in rng.sample(buffer, BATCH // 2)]   # fresh-render replay: stored entries, new letters and k
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            else:
                batch = [(drift_item(si * STAGE_UPDATES - STAGE_UPDATES + u) if STREAM16 == "drift" else next(cur)) for _ in range(BATCH)]; presented += batch
            nnew = len(presented) - npres_before if admits else 0
            losses.append(update16(m, opt, batch, rule, store))
            if si == 1 and u == 0:    # 9 Nov audit: record what received no gradient on the first update (grads persist until the next zero_grad)
                stages[0]["settings"]["zero_grad_first_update"] = zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e16 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"], flush=True)
            if ADMIT16 == "online" and admits:     # raw reservoir, Algorithm R per presented item, as it arrives (new items are the first nnew of the batch)
                fl = fbuf.setdefault("all", []); n = fseen.get("all", 0)
                for p_, t_ in batch[:nnew]:
                    n += 1
                    if len(fl) < RCAP:
                        fl.append(("raw", tuple(p_), tuple(t_)))
                    else:
                        j = rng.randrange(n)
                        if j < RCAP:
                            fl[j] = ("raw", tuple(p_), tuple(t_))
                fseen["all"] = n
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)), cpu=time.process_time() - t0, scores=evaluate()))
        sc = stages[-1]["scores"]
        print("e16", arm, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in KS] for n in sc if n.endswith("97")},
              "F3@12", [sc[n][12]["exact"] for n in sc if n.endswith("73")],
              "comp.final", {n: v["final"] for n, v in sc["compose"].items()},
              "dev89@12", [sc[n][12]["exact"] for n in sc if n.endswith("89")],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(arm=arm, seed=seed, schedule=SCHEDULE, stages=stages), indent=1))
        if ADMIT16 == "online" and admits:
            stages[-1]["buffer"] = {f: len(v) for f, v in fbuf.items()}; stages[-1]["reservoir_distinct"] = {f: len(set(v)) for f, v in fbuf.items()}
            stages[-1]["reservoir_bytes"] = sum(len(e[1]) + len(e[2]) for e in fbuf["all"]); stages[-1]["admitted"] = fseen.get("all", 0)
            path.write_text(json.dumps(dict(arm=arm, seed=seed, schedule=SCHEDULE, stages=stages), indent=1))
        elif (local or arm == "M") and REPLAY16 != "oracle":
            # the stage's presented items as replayable entries (family + operands), 1,024 of them, as E11-D/E15
            ents = []
            for p, t in presented:
                if REPLAY16 == "raw":
                    ents.append(("raw", tuple(p), tuple(t))); continue
                if p[0] == ord("c"):
                    ents.append(("comp", chr(p[3]), chr(p[6]), int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9]))))
                else:
                    task = next(t for t in F1 + F3 if t[2][0] == p[0])
                    ents.append(("op", task, int(bytes(p[1:3])), int(bytes(p[-3:-1]))))
            if REPLAY16 == "uniform":
                buffer += rng.sample(ents, min(1024, len(ents)))
            elif REPLAY16 in ("balanced", "share"):
                fbuf.setdefault(fam, []).extend(rng.sample(ents, min(1024, len(ents))))
            elif REPLAY16 in ("reservoir", "total", "raw"):   # Algorithm R over presentations: per family (reservoir) or one for the whole life (total, raw)
                key = fam if REPLAY16 == "reservoir" else "all"
                fl = fbuf.setdefault(key, []); n = fseen.get(key, 0)
                for e in ents:
                    n += 1
                    if len(fl) < RCAP:
                        fl.append(e)
                    else:
                        j = rng.randrange(n)
                        if j < RCAP:
                            fl[j] = e
                fseen[key] = n
                stages[-1]["reservoir_distinct"] = {f: len(set(v)) for f, v in fbuf.items()}
                if REPLAY16 == "total":   # how the label-free reservoir happens to be composed, by family of entry (for reading only)
                    stages[-1]["reservoir_by_family"] = {f: sum(1 for e in set(fbuf["all"]) if (e[0] == "comp") == (f == "F2") and (f == "F2" or e[1][1] == (97 if f == "F1" else 73))) for f in ("F1", "F2", "F3")}
                elif REPLAY16 == "raw":   # by first byte of the stored prompt (for reading only; the learner never sees this grouping)
                    f1c, f3c = {t[2][0] for t in F1}, {t[2][0] for t in F3}
                    stages[-1]["reservoir_by_family"] = {"F1": sum(1 for e in set(fbuf["all"]) if e[1][0] in f1c), "F2": sum(1 for e in set(fbuf["all"]) if e[1][0] == ord("c")), "F3": sum(1 for e in set(fbuf["all"]) if e[1][0] in f3c)}
                    stages[-1]["reservoir_bytes"] = sum(len(e[1]) + len(e[2]) for e in fbuf["all"])
            else:   # all_balanced: every distinct presented entry of the family
                fs = fset.setdefault(fam, set()); fl = fbuf.setdefault(fam, [])
                for e in ents:
                    if e not in fs:
                        fs.add(e); fl.append(e)
            stages[-1]["buffer"] = {f: len(v) for f, v in fbuf.items()} if fbuf else len(buffer)
            path.write_text(json.dumps(dict(arm=arm, seed=seed, schedule=SCHEDULE, stages=stages), indent=1))   # E16-V: keep the last block's buffer count too
    T.save(m.state_dict(), path.with_suffix(".pt"))   # final weights, so panels can be re-scored later (8 Nov audit)


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
    all_taught = frozenset().union(*taught.values())
    for name, items in comp.items():
        for p, _ in items:
            pair = (chr(p[3]), chr(p[6])); tr = (int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])))
            assert (pair in HELD_PAIRS) if name != "within" else (pair in TAUGHT_PAIRS)
            assert tr not in all_taught, "panel triple appears in some taught pair's training triples"   # operand-level holdout
    out["panel_sizes"] = {n: len(v) for n, v in comp.items()}
    out["shortcut_ceilings"] = shortcut_ceilings(comp)
    # window visibility on the composition prompt: positions whose prediction no longer sees the code byte / first operand
    out["composition_prompt_len"] = 10; out["window"] = E.W
    out["code_byte_visible_until_output_index"] = E.W - 10     # 0: visible only for the first output byte
    # a batch of mixed items passes through update on every arm
    m = E.new_net(seed); m.load_state_dict(T.load(OUT / f"e11_dev_M_{seed}.pt"))
    out["birth_compose_final_before_life"] = {n: score(m, it)["final"] for n, it in comp.items()}   # scored BEFORE the smoke updates (audit, 8 Nov)
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
    out["compose_final_after_smoke_updates"] = {n: score(m, it)["final"] for n, it in comp.items()}
    (OUT / "QUALIFY_E16.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def develop17(seed):
    """Birth for a windowless architecture: exp11's development (backprop through time on the four mod-89 operations with 0-12
    distractor letters, 24,576 updates, sealed panels at k = 0/4/8/12/20) on the E16_ARCH net; writes e17_dev_{arch}_{seed}."""
    assert ARCH16 != "window"
    path = OUT / f"e17_dev_{ARCH16}{BTAG}{E.DEVTAG}_{seed}.pt"
    if path.exists():
        return
    m = E.new_net(seed)
    opt = adamw16(m, .001, .3)      # E16_BOOKDECAY=off: the embedding table without decay, so rows of symbols unseen at birth keep their init
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = [E.update(m, opt, next(bt), "D") for _ in range(X.BLOCK)]
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()})
            curve.append(row)
            print("e17 dev", ARCH16, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    T.save(m.state_dict(), path)
    (OUT / f"e17_dev_{ARCH16}{BTAG}{E.DEVTAG}_{seed}.json").write_text(json.dumps(dict(arch=ARCH16, seed=seed, kmax=E.KMAX["M"], bookdecay=BOOKDECAY16, curve=curve), indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop17(int(sys.argv[2]))
    else:
        life(sys.argv[1], int(sys.argv[2]))
