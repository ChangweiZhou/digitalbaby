"""Checks behind the audit of the E43 brief and its specification file at 75613729 (16 November 2026).

Read-only.  Nothing is trained, torch is not used, and no code of the default branch is run: files are read with `git show`,
saved weights with the numpy reader (np_gru.py), and the unrelated task with e43_unrelated_reader.py, written from the brief.

  A  the bytes every family uses, rebuilt from the harness's source, beside the bytes of the unrelated task
  B  the byte table: rows 128 to 225 in the births and the lives; which rows any birth or life has ever moved
  C  what decay alone does to the first family: every parameter but the byte table multiplied by (1 - 0.001 * 0.25)^t
  D  the two numbers of the decay arms
  E  the reader's digests beside the committed output's, and what the other readings of the brief's words give
  F  E38's receipts: which rows and hashes exist for check 1, and what row 0 holds for check 2
  G  cost

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_brief_checks.py [A B C D E F G]
"""
import ast, json, os, re, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, sigmoid, apply, SYM, BOS, EOS, P, DEV
import e43_unrelated_reader as U

REF = "75613729"
M = "MOD97_LOCAL_RULE_20261106/"
K = "PROJECT_DOCUMENTS/COORD_CHECKS_20261116/"
SEEDS = (73129, 93683, 116089)
TASK = set(U.SYMBOLS) | set(U.FILLERS) | {U.CODE, U.QUERY}
want = set(sys.argv[1:]) or set("ABCDEFG")
show = lambda path, ref=REF: subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def run_items(W, items):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; Un = Whh.shape[1]
    h = np.zeros((len(x), Un), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :Un] + gh[:, :Un]); z = sigmoid(gi[:, Un:2 * Un] + gh[:, Un:2 * Un]); n = np.tanh(gi[:, 2 * Un:] + r * gh[:, 2 * Un:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            ok &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - 3)]
    return ok


def plain(task, pairs):
    op, m, code = task
    return [(list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + list(f"{b:02d}".encode()) + [ord("=")], list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]) for a, b in pairs]


if "A" in want:
    print("A. bytes used by the families, from the harness's source at", REF)
    e11 = show(M + "exp11_cell.py"); e16 = show(M + "exp16_integrated.py"); e40 = show(M + "e40_new_family.py")
    used = set(b"0123456789")                                     # f"{a:02d}" in exp11_cell.render and exp16_integrated.render_pair
    assert 'f"{a:02d}".encode()' in e11 and 'f"{a:02d}".encode()' in e16
    sym = ast.literal_eval(re.search(r"^SYM = (\{.*\})", e11, re.M).group(1)); used |= {ord(v) for v in sym.values()}
    dis = ast.literal_eval(re.search(r"^DISTRACT = list\((b\".*\")\)", e11, re.M).group(1)); used |= set(dis)
    assert '[ord("=")]' in e11 and 'b"="' in e16 and 'ord(";")' in e16 and 'lead + b"c"' in e16
    used |= {ord("="), ord(";"), ord("c")}
    codes = {}
    for name, src in (("exp11_cell.py", e11), ("exp16_integrated.py", e16), ("e40_new_family.py", e40)):
        for op, mod, code in re.findall(r'\("(add|sub|max|min)", (\d+), b"(.)"\)', src):
            codes.setdefault(code, set()).add((name, int(mod)))
    used |= {ord(c) for c in codes}
    others = set()
    for f in subprocess.run(["git", "ls-tree", "--name-only", REF, M], capture_output=True, text=True).stdout.split():
        if f.endswith(".py"):
            for op, mod, code in re.findall(r'\("(add|sub|max|min)", (\d+), b"(.)"\)', show(f)):
                others.add(code)
    print(f"   operator symbols {sorted(sym.values())}; letters {bytes(sorted(dis)).decode()}; codes by file {dict(sorted((c, sorted(m for _, m in v)) for c, v in codes.items()))}")
    print(f"   codes in task tuples of every .py file of the harness folder: {''.join(sorted(others))}")
    used |= {ord(c) for c in others}
    print(f"   {len(used)} bytes in all, the largest {max(used)}; the unrelated task's {len(TASK)} bytes run from {min(TASK)} to {max(TASK)}; shared: {sorted(used & TASK)}")
    spec_used = set(b"0123456789+-><=;ABCDEFGHIJKLMNOPQRSTUVWXYZpqvwrtxnsuyzcefgh")
    print(f"   the specification file's own list equals this one: {spec_used == used}; in mine and not in its: {bytes(sorted(used - spec_used))}; in its and not in mine: {bytes(sorted(spec_used - used))}")
    print(f"   start symbol {BOS} and end symbol {EOS}: shared by construction (the end symbol is the third answer byte of both)")

if "B" in want:
    print("B. the byte table (258 rows of 8) in the saved files")
    rows = sorted(TASK)
    for s in SEEDS:
        bS = load_weights(P + f"e26/e26_dev_S_{s}.pt")[0]["book.weight"]; bB = load_weights(P + f"e26/e26_dev_B_{s}.pt")[0]["book.weight"]
        lives = {a: load_weights(P + f)[0]["book.weight"] for a, f in (("S", f"e26/e26_life_S_{s}.pt"), ("B", f"e26/e26_life_B_{s}.pt"), ("K", f"e31/e31_life_K_{s}.pt"))}
        moved_birth = [i for i in range(258) if not np.array_equal(bS[i], bB[i])]
        moved_life = sorted({i for a, start in (("S", bS), ("B", bB), ("K", bS)) for i in range(258) if not np.array_equal(lives[a][i], start[i])})
        txt = lambda idx: "".join(chr(i) if 32 < i < 127 else f"<{i}>" for i in idx)
        print(f"   seed {s}: rows 128..225, sweep birth against backpropagation birth: largest difference {np.abs(bS[rows] - bB[rows]).max():.1e} (bit-equal {np.array_equal(bS[rows], bB[rows])});"
              f" each birth against its lives: {max(np.abs(lives[a][rows] - st[rows]).max() for a, st in (('S', bS), ('B', bB))):.1e}")
        print(f"      rows that differ between the two births (rows a birth has moved): {txt(moved_birth)}")
        print(f"      rows a life has moved beyond its birth: {txt([i for i in moved_life if i not in moved_birth])}")
        never = [i for i in range(258) if i not in moved_birth and i not in moved_life]
        print(f"      rows no birth or life has moved: {len(never)}, among them all of 128..225: {set(rows) <= set(never)}; row sizes of 128..225 from {np.linalg.norm(bS[rows], axis=1).min():.2f} to {np.linalg.norm(bS[rows], axis=1).max():.2f};"
              f" largest entry {np.abs(bS[rows]).max():.2f}, where one step of a 32-bit float is {np.spacing(np.float32(np.abs(bS[rows]).max())):.1e}")

if "C" in want:
    print("C. decay alone: every parameter but the byte table multiplied by (1 - 0.00025)^t, nothing else changed; the first family, 1,500 pairs of each")
    allp = [(a, b) for a in range(89) for b in range(89)]
    probe = {t[0]: Random(f"kept|{t[0]}").sample(allp, 1500) for t in DEV}
    wrap = {"add": lambda a, b: a + b >= 89, "sub": lambda a, b: a - b < 0}
    print("   file                     updates  factor | reduction pairs (add, sub; share of start) | without reduction | max  | min")
    for f in ("e26/e26_dev_S_{s}.pt", "e26/e26_dev_B_{s}.pt"):
        for s in SEEDS:
            W0, _ = load_weights(P + f.format(s=s)); start = None
            for t in (0, 64, 256, 1024, 2048, 4096):
                c = (1 - 0.001 * 0.25) ** t
                W = {k: (v if k == "book.weight" else (v * np.float32(c))) for k, v in W0.items()}
                out = {}
                for task in DEV:
                    ok = run_items(W, plain(task, probe[task[0]]))
                    if task[0] in wrap:
                        w = np.array([wrap[task[0]](a, b) for a, b in probe[task[0]]]); out[task[0]] = (100 * ok[w].mean(), 100 * ok[~w].mean())
                    else:
                        out[task[0]] = 100 * ok.mean()
                red = (out["add"][0] + out["sub"][0]) / 2; start = red if t == 0 else start
                print(f"   {f.format(s=s):24s} {t:6d}  {c:6.3f} | {out['add'][0]:5.1f}, {out['sub'][0]:5.1f}; {100 * red / start:5.1f}%                | {out['add'][1]:5.1f}, {out['sub'][1]:5.1f}       | {out['max']:5.1f} | {out['min']:5.1f}", flush=True)

if "D" in want:
    print("D. the decay arms")
    print(f"   (1 - 0.001 * 0.25)^4096 = {(1 - 0.00025) ** 4096:.4f}; ^1024 = {(1 - 0.00025) ** 1024:.4f}; ^256 = {(1 - 0.00025) ** 256:.4f}; ^64 = {(1 - 0.00025) ** 64:.4f}")
    print("   AdamW: p <- p * (1 - lr * decay), then p <- p - lr * m_hat / (sqrt(v_hat) + 1e-8).  With a gradient of exactly zero the second step is zero;")
    print("   with any gradient whose size is well above 1e-8 the step is about lr whatever that size, so a weight the new task barely uses still moves.")
    print("   Exactly zero gradient under the unrelated task: the byte table's rows of bytes it does not show.  Everything else takes part in every item.")

if "E" in want:
    print("E. the reader beside the committed output")
    out = show(K + "OUTPUT_E43_SPEC_MAC.txt")
    for s in SEEDS:
        m = re.search(rf"seed {s}: .*digest of the first 64 updates (\w+); held panel digest (\w+)", out)
        mine = (U.stream_digest(s, 64)[:16], U.panel_digest(s, "held")[:16])
        print(f"   seed {s}: stream {mine[0]} (output {m.group(1)}, equal {mine[0] == m.group(1)}); held panel {mine[1]} (output {m.group(2)}, equal {mine[1] == m.group(2)})")
    s = SEEDS[0]
    print("   other readings of the brief's words, seed 73129 (stream digest of 64 updates; held panel digest):")
    print(f"      each pass shuffles the order the last pass left:   {U.stream_digest(s, 64, pass_order='running')[:16]}   (the first pass, 32 updates: {U.stream_digest(s, 32, pass_order='running')[:16]} against {U.stream_digest(s, 32)[:16]})")
    print(f"      pairs listed with the second symbol running slowest: {U.stream_digest(s, 64, pairs_order='s2 outer')[:16]}   {U.panel_digest(s, 'held', pairs_order='s2 outer')[:16]}")
    print(f"      the panel's 300 drawn from the pairs sorted:          (stream unchanged)   {U.panel_digest(s, 'held', population='sorted')[:16]}")
    lens = [len(p) for b in U.batches(s, 64) for p, _ in b]
    print(f"   an item's prompt has {min(lens)} to {max(lens)} bytes; with the start symbol and two answer bytes the input is at most {max(lens) + 3} long (the output's batch tensor: 64 x 19)")
    print(f"   showings of a taught pair in 4,096 updates: {4096 * 64 / 2048:.0f}; taught and held-out pairs in which a given symbol stands first: 64 each, about 32 taught")

if "F" in want:
    print("F. E38's receipts for seed 73129 (merged on the default branch)")
    for rule in ("S", "B"):
        for arm in ("zero", "none"):
            d = json.loads(show(M + f"exp45/e38/e38_{arm}_{rule}_73129.json"))
            r = {x["updates"]: x for x in d["rows"]}
            print(f"   {arm:4s} {rule}: rows {sorted(r)[:10]}...; tensors_sha256 at 64, 256, 1,024, 4,096: {[r[u]['tensors_sha256'][:8] for u in (64, 256, 1024, 4096)]}; schedule {d['schedule']}")
    d = json.loads(show(M + "exp45/e38/e38_zero_S_73129.json")); r0 = d["rows"][0]
    print(f"   row 0 holds: {[k for k in r0 if k not in ('weights', 'weights_sha256', 'tensors_sha256', 'cpu', 'updates')]}")
    print(f"   birth_allpairs_k0 tasks: {list(r0['birth_allpairs_k0'])}; fields of one: {list(r0['birth_allpairs_k0']['add89'])}")
    print(f"   birth_heldout_k12 tasks: {list(r0['birth_heldout_k12'])}; new_heldout_k12: {dict((k, v['n']) for k, v in r0['new_heldout_k12'].items())}")
    brief = show("OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md")
    print(f"   the brief quotes f3819049, 95f5de74, e2d170be for the sweep's `zero`: all three in the receipt: {all(any(x['tensors_sha256'].startswith(h) for x in d['rows']) for h in ('f3819049', '95f5de74', 'e2d170be'))}")

if "G" in want:
    print("G. cost")
    upd = 24 * 4096
    print(f"   24 runs x 4,096 updates = {upd:,} updates; at 0.11 to 0.13 CPU-seconds an update (E40's check 6, E26's births): {upd * 0.11 / 3600:.1f} to {upd * 0.13 / 3600:.1f} CPU-hours")
    print(f"   scoring: 24 runs x 5 rows at 14 to 17 CPU-seconds a row (E40's check 6; E43's score 1 is all 4 x 7,921 pairs, about E36's): {24 * 5 * 14 / 3600:.1f} to {24 * 5 * 17 / 3600:.1f} CPU-hours")
    print(f"   the memo's '24 runs, about 2 CPU-hours' was written for two arms, which is 12 runs: {12 * 4096 * 0.12 / 3600:.1f} CPU-hours of updates")
