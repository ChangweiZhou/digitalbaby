"""Checks behind the audit of the E40 brief at 3aa9dbf3 (15 November 2026).

Read-only: saved tensors and a pure-Python rebuild of the split, the panels and the stream's seeds.  Nothing trains and no
harness code is run.

A. The family: its code bytes and modulus beside every earlier family's; the generator seeds of its splits and of its
   stream beside every seed an earlier split or stream used.
B. The taught tenth: pairs taught and held out, how often each value appears in each operand position, the share of
   held-out pairs that need the reduction at 83, and showings per taught pair at row 8,192.
C. The embedding rows of the four new code bytes in the birth and life-end files of a seed: equal across all of them or not
   (if equal across networks trained differently from one start, they are the starting values a fresh network also has).
D. Row 0 of the nine lived runs on the new family, before any update: the brief's scores 1 to 3, and what the answers on
   the pairs that need the reduction equal.
E. What the brief's reading does on the row-0 figures.
F. A fresh network that has memorised a tenth of the pairs (E35-A's arm G at f = 0.10, mod 89, seed 73129, end of a full
   birth, on the worker's branch): its held-out pairs without letters split by the reduction, as a scale for the fresh
   runs of E40.

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e40_brief_checks.py [A B C D E F]
"""
import os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, split, panel, sigmoid, apply, SYM, DEV, LIFE, BOS, EOS, P

REF = '3aa9dbf3'; SEEDS = (73129, 93683, 116089)
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
LIVES = {"S": "e26/e26_life_S_{s}.pt", "B": "e26/e26_life_B_{s}.pt", "K": "e31/e31_life_K_{s}.pt"}
want = set(sys.argv[1:]) or set("ABCDEF")
split_seed = lambda t, f, s: s * 1000 + round(f * 100) + t[1] * 7 + t[2][0]

def run_items(W, items):
    """Whole answers right (all three target bytes, teacher forced), for items of one length."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]; pred = np.zeros((len(x), 3), np.int64)
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            pred[:, t - (L - 3)] = (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1)
            ok &= pred[:, t - (L - 3)] == tgt[:, t - (L - 3)]
    return ok, pred

def plain(task, pairs):
    op, m, code = task
    return [(list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]) for a, b in pairs]

if "A" in want:
    print("A. the family beside the earlier ones")
    fams = {"birth": DEV, "F1": LIFE, "F3": F3}
    used = {t[2] for f in fams.values() for t in f} | {b"c"}
    other = set(b"0123456789+-<>=;") | set(range(65, 91)) | {BOS, EOS}
    print("   codes of F4:", [t[2] for t in F4], "| any in an earlier family:", any(t[2] in used for t in F4), "| any a digit, symbol or upper-case letter:", any(t[2][0] in other for t in F4))
    print("   modulus 83 among earlier moduli", sorted({t[1] for f in fams.values() for t in f}), ":", 83 in {t[1] for f in fams.values() for t in f})
    for s in SEEDS:
        mine = {split_seed(t, .10, s) for t in F4}
        theirs = {split_seed(t, f, s) for fam, f in (("birth", .5), ("F1", .25), ("F3", .25)) for t in fams[fam]} | {split_seed(t, f, s) for t in DEV for f in (.10, .25)}
        stream_mine = s + 140 + 77
        stream_theirs = {s + 77, s + 36 + 77, s + 100 + 77} | {s + 100 + si + 77 for si in range(1, 7)} | {s + si + 78 for si in range(1, 7)} | {s + 3, s + 4}
        allseeds = {x + d for x in SEEDS for d in (77, 113, 177, 3, 4)} | {x + 100 + si + 77 for x in SEEDS for si in range(1, 7)} | {x + si + 78 for x in SEEDS for si in range(1, 7)}
        print(f"   seed {s}: split generator seeds of F4 {sorted(mine)}; shared with an earlier split: {bool(mine & theirs)} | stream generator seed {stream_mine}; shared with any earlier stream or panel generator of any seed: {stream_mine in (stream_theirs | allseeds)}")
    print("   round(.10 * 100) =", round(.10 * 100), "| round(.10 * 6889) =", round(.10 * 6889))

if "B" in want:
    print("B. the taught tenth")
    for s in SEEDS:
        for t in F4:
            taught, held = split(t, .10, s)
            cnt = [min(sum(p[k] == v for p in taught) for v in range(83)) for k in (0, 1)]
            wrap = [(a + b >= 83) if t[0] == "add" else (a - b < 0) for a, b in held]
            extra = f"; held-out pairs needing the reduction {100 * np.mean(wrap):.1f}%" if t[0] in ("add", "sub") else ""
            print(f"   seed {s} {t[0]}83: taught {len(taught)}, held out {len(held)}; fewest appearances of a value, first / second position: {cnt[0]} / {cnt[1]}{extra}")
    print(f"   row 8,192: {8192 * 64} items, {8192 * 64 // 4} per task, {8192 * 64 / 4 / 689:.0f} showings of each taught pair")

if "C" in want:
    print("C. embedding rows of the new code bytes (e f g h), and of one seen byte (p) for comparison")
    for s in SEEDS:
        Ws = {n: load_weights(P + f.format(s=s), REF)[0]['book.weight'] for n, f in {**LIVES, "birth S": "e26/e26_dev_S_{s}.pt", "birth B": "e26/e26_dev_B_{s}.pt"}.items()}
        base = Ws["birth S"]
        new_equal = all(np.array_equal(w[[101, 102, 103, 104]], base[[101, 102, 103, 104]]) for w in Ws.values())
        seen_equal = all(np.array_equal(w[112], base[112]) for w in Ws.values())
        unseen = [b for b in range(256) if all(np.array_equal(w[b], base[b]) for w in Ws.values())]
        print(f"   seed {s}: rows e f g h bit-equal in birth S, birth B, life S, life B, life K: {new_equal} | row p equal in all: {seen_equal} | bytes whose row is equal in all five files: {len(unseen)} of 256")

if "D" in want or "E" in want:
    print("D. row 0 of the lived runs on the new family (whole answers, %)")
    rows = {}
    for arm, f in LIVES.items():
        for s in SEEDS:
            W = load_weights(P + f.format(s=s), REF)[0]; r = {}
            for t in F4:
                taught, held = split(t, .10, s)
                r[t[0] + "_held12"] = 100 * run_items(W, panel(t, held, s, (12,), 300)[12])[0].mean()
                r[t[0] + "_taught12"] = 100 * run_items(W, panel(t, taught, s + 1, (12,), 300)[12])[0].mean()
                if t[0] in ("add", "sub"):
                    ok, pred = run_items(W, plain(t, held)); a = np.array(held)[:, 0]; b = np.array(held)[:, 1]
                    raw = a + b if t[0] == "add" else a - b; wrap = (raw >= 83) if t[0] == "add" else (raw < 0)
                    val = np.where((pred[:, :2] >= 48).all(1) & (pred[:, :2] <= 57).all(1), (pred[:, 0] - 48) * 10 + pred[:, 1] - 48, -1)
                    r[t[0] + "_nowrap"] = 100 * ok[~wrap].mean(); r[t[0] + "_wrap"] = 100 * ok[wrap].mean()
                    for m in (97, 73, 89):
                        r[t[0] + f"_wrap_as{m}"] = 100 * np.mean(val[wrap] == raw[wrap] % m)
            rows[(arm, s)] = r
            print(f"   lived {arm} {s}: held-out at 12 letters add {r['add_held12']:.1f} sub {r['sub_held12']:.1f} (mean {(r['add_held12'] + r['sub_held12']) / 2:.1f}) max {r['max_held12']:.1f} min {r['min_held12']:.1f} | taught add/sub mean {(r['add_taught12'] + r['sub_taught12']) / 2:.1f}"
                  f" | held-out pairs, no letters: no reduction add {r['add_nowrap']:.1f} sub {r['sub_nowrap']:.1f}; reduction add {r['add_wrap']:.1f} sub {r['sub_wrap']:.1f}"
                  f" | reduction pairs answered as mod 97 / 73 / 89: add {r['add_wrap_as97']:.0f} / {r['add_wrap_as73']:.0f} / {r['add_wrap_as89']:.0f}, sub {r['sub_wrap_as97']:.0f} / {r['sub_wrap_as73']:.0f} / {r['sub_wrap_as89']:.0f}", flush=True)
    if "E" in want:
        print("E. the brief's reading on row 0 (a fresh network scores 0 before any update)")
        for arm in LIVES:
            m = [(rows[(arm, s)]['add_held12'] + rows[(arm, s)]['sub_held12']) / 2 for s in SEEDS]
            w = [(rows[(arm, s)]['add_wrap'] + rows[(arm, s)]['sub_wrap']) / 2 for s in SEEDS]
            print(f"   lived {arm}: held-out mean at row 0 = " + ", ".join(f"{v:.1f}" for v in m) + f" -> seeds already 15 or more above a fresh network at row 0: {sum(v >= 15 for v in m)} of 3 | reduction score at row 0 = " + ", ".join(f"{v:.1f}" for v in w))

if "F" in want:
    print("F. E35-A arm G at f = 0.10 (fresh, backpropagation, mod 89, seed 73129), held-out and taught pairs without letters, whole answers %")
    s = 73129
    for tag, suffix in (("row 16,384", "_row16384.pt"), ("row 20,480", "_row20480.pt"), ("end", ".pt")):
        try:
            W = load_weights(P + f"e35a/e35a_dev_G_f10_{s}{suffix}", "origin/claude/nice-rubin-y5de8k")[0]
        except AssertionError:
            print("   ", tag, "not on the branch"); continue
        out = []
        for t in DEV[:2]:
            taught, held = split(t, .10, s)
            ok, _ = run_items(W, plain(t, held)); a = np.array(held)[:, 0]; b = np.array(held)[:, 1]
            wrap = (a + b >= 89) if t[0] == "add" else (a - b < 0)
            okt, _ = run_items(W, plain(t, taught))
            out.append(f"{t[0]}: held-out no reduction {100 * ok[~wrap].mean():.1f}, reduction {100 * ok[wrap].mean():.1f}; taught {100 * okt.mean():.1f}")
        print(f"   {tag}: " + " | ".join(out), flush=True)
