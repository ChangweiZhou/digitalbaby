"""Checks behind the audit of the E48 brief (revision 1) and its specification file on the default branch at 7da5c453
(16 November 2026).

Read-only.  Nothing is trained, torch is not used, and no code of the default branch is run: files are read with `git show`,
saved weights with the numpy reader (np_gru.py), and the gated forward with e48_gated_reader.py, written from the brief.

  A  the numbers that may change at each rank; the size of a term file; what a replay store of the same size would hold
  B  the reader on the lived networks of E40: row 0 of every score beside E40's receipts; gate off and gate on at zero
     against the plain reader, value for value; what a term and changed code rows can and cannot reach
  C  the two rules on the two factors: the first and second update of each, on stand-in errors
  D  E40's reduction score by row in the fifteen receipts; how far one row moves; the statistic proposed in its place
  E  cost, from E40's receipts
  F  the bytes of every earlier family beside the four code bytes; how E40's scorer reaches the forward

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e48_brief_checks.py [A B C D E F] [quick]
(`quick` in part B reads one network instead of six.)
"""
import ast, json, os, re, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import load_weights, forward as plain_forward, split, panel, render, batch, apply, SYM, DEV, LIFE, BOS, EOS, P
import e48_gated_reader as G

REF = "7da5c453"
M = "MOD97_LOCAL_RULE_20261106/"
SEEDS = (73129, 93683, 116089)
RANKS = (4, 16, 64)
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
EARLIER6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
E40ROWS = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
want = {a for a in sys.argv[1:] if a in set("ABCDEF")} or set("ABCDEF")
quick = "quick" in sys.argv[1:]
show = lambda path, ref=REF: subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
receipt = lambda rule, seed: json.loads(show(M + f"exp45/e40/e40_lived_{rule}_{seed}.json"))
name = lambda t: f"{t[0]}{t[1]}"


# ---------------------------------------------------------------- E40's panels, rebuilt from the text of e40_new_family.Scorer
def e40_panels(seed, ks_life):
    """held and taught at 12 letters (300 pairs), every held-out add and sub pair without letters split by the reduction,
    and the six earlier tasks at 12 letters as exp16_integrated.panels builds them (ks_life: the lengths it renders, in order)."""
    held, taught, plain = {}, {}, {}
    for t in F4:
        tp, hp = split(t, .10, seed)
        held[name(t)] = panel(t, hp, seed, (12,), 300)[12]
        taught[name(t)] = panel(t, tp, seed + 1, (12,), 300)[12]
        if t[0] in ("add", "sub"):
            red, nored = [], []
            for a, b in hp:
                need = (a + b >= 83) if t[0] == "add" else (a - b < 0)
                (red if need else nored).append(render(t, a, b, 0, Random(0)))
            plain[name(t)] = dict(reduction=red, no_reduction=nored)
    earlier = {name(t): panel(t, split(t, .25, seed)[1], seed, ks_life, 400) for t in LIFE + F3}
    earlier.update({name(t): panel(t, split(t, .5, seed)[1], seed, (12,), 300) for t in DEV})
    return held, taught, plain, earlier


def exact(W, term, items):
    outs = []
    for j in range(0, len(items), 2048):
        outs += G.generate(W, term, [p for p, _ in items[j:j + 2048]])
    return sum(o == t for o, (_, t) in zip(outs, items)), outs


if "A" in want:
    print("A. numbers that may change, and what the same memory would hold")
    total = 1124113
    # an item as exp11_cell.render writes it: code, two digits, symbol, k letters, two digits, '=' ; answer two digits and the end symbol
    prompt = lambda k: 1 + 2 + 1 + k + 2 + 1
    assert prompt(12) == 19 and prompt(0) == 7
    mean_item = sum(prompt(k) + 3 for k in range(13)) / 13
    taught_pairs = round(.10 * 83 * 83); taught_items = 4 * taught_pairs
    print(f"   an item: prompt 7 + k bytes (k letters, 0 to 12 in training), answer 3 bytes; the prompt at 12 letters is {prompt(12)} bytes (the brief's figure),"
          f" a whole item at 12 letters {prompt(12) + 3}, the mean whole item of the stream {mean_item:.0f}; an entry of the stream as the harness stores it (task, a, b) is 3 bytes")
    print(f"   the family's taught set: {taught_pairs} pairs an operation, {taught_items} entries in all = {3 * taught_items:,} bytes as entries, {int(mean_item) * taught_items:,} bytes as whole items of mean length")
    print("   rank | numbers | share of the network | bytes at 4 a number | items at 19 bytes | at 16 bytes | entries at 3 bytes | times the family's whole taught set (as entries)")
    for r in RANKS:
        n = G.count(r); assert n == 2304 * r + 32
        b = 4 * n
        print(f"   {r:4d} | {n:7,} | {100 * n / total:5.2f}% | {b:7,} | {b // 19:6,} | {b // 16:6,} | {b // 3:7,} | {b / (3 * taught_items):5.1f}")
    print("   a term file (four matrices and the four code rows, float32) is the bytes column plus a header of about 2 KB;"
          f" over the 18 runs one scored row is {sum(4 * G.count(r) for r in RANKS) * 6 / 1e6:.1f} MB, eight rows {8 * sum(4 * G.count(r) for r in RANKS) * 6 / 1e6:.0f} MB")

if "B" in want:
    print("B. the reader on the lived networks (nothing trained)")
    env = receipt("S", 73129)["provenance"]["environment_set_by_runner"]
    ks_life = (0, 8, 12, 20) + ((40,) if env.get("E16_ARCH", "window") != "window" else ())
    print(f"   E40's runner sets {env}; exp16_integrated.panels then renders the life's panels at lengths {ks_life}")
    rng = np.random.default_rng(20261116)
    never = True
    for rule in ("S", "B"):
        for seed in (SEEDS if not quick else SEEDS[:1]):
            W, _ = load_weights(P + f"e26/e26_life_{rule}_{seed}.pt", REF)
            held, taught, plain, earlier = e40_panels(seed, ks_life)
            zero = G.new_term(16, rng)                                    # up at zero: the start of every run
            row0 = receipt(rule, seed)["rows"][0]["scores"]
            mine, theirs, gen_codes = [], [], 0
            for n, it in held.items():
                mine.append(exact(W, zero, it)[0]); theirs.append(row0["held_k12"][n]["exact"])
            for n, it in taught.items():
                mine.append(exact(W, zero, it)[0]); theirs.append(row0["taught_k12"][n]["exact"])
            for n, v in plain.items():
                for w, it in v.items():
                    mine.append(exact(W, zero, it)[0]); theirs.append(row0["held_plain_k0"][n][w]["exact"])
            base_out = {}
            for n in EARLIER6:
                c, outs = exact(W, zero, earlier[n][12]); base_out[n] = outs
                mine.append(c); theirs.append(row0["earlier_k12"][n]["exact"])
                gen_codes += sum(b in G.CODES for o in outs for b in o)
            never &= gen_codes == 0
            red = [100 * exact(W, zero, plain[n]["reduction"])[0] / len(plain[n]["reduction"]) for n in ("add83", "sub83")]
            print(f"   lived {rule} {seed}: the 18 counts of row 0 (4 held, 4 taught, 4 without letters, 6 earlier) from the reader with the term at zero equal the receipt's: {mine == theirs}"
                  f" | reduction score {np.mean(red):.1f} (receipt {row0['summary']['reduction_score']:.1f}) | bytes e f g h among the bytes generated for the earlier tasks: {gen_codes}", flush=True)
            if (rule, seed) != ("S", 73129):
                continue
            # value for value, on every panel E40 scores, under teacher forcing
            eq_off = eq_on0 = True; n_items = 0
            groups = [it for it in held.values()] + [it for it in taught.values()] + [it for v in plain.values() for it in v.values()]
            for it in groups:
                for j in range(0, len(it), 1024):
                    x, _ = batch(it[j:j + 1024]); ref = plain_forward(W, x)[3]; n_items += len(x)
                    eq_off &= np.array_equal(G.forward(W, zero, x, gate=False)[1], ref)
                    eq_on0 &= np.array_equal(G.forward(W, zero, x, gate=True)[1], ref)
            for n in EARLIER6:
                x, _ = batch(earlier[n][12]); ref = plain_forward(W, x)[3]; n_items += len(x)
                eq_off &= np.array_equal(G.forward(W, zero, x, gate=False)[1], ref)
                eq_on0 &= np.array_equal(G.forward(W, zero, x, gate=True)[1], ref)
            print(f"      on the {n_items:,} items of E40's panels: gate off equals the plain reader value for value: {eq_off}; gate on with `up` at zero equals it: {eq_on0}")
            # the quick generator against the word-for-word one, with a term that holds something
            tt = G.new_term(16, rng, up_scale=0.05)
            trio = [[p for p, _ in earlier["add97"][12]], [p for p, _ in held["add83"]], [p for p, _ in plain["sub83"]["reduction"][:300]]]
            print(f"      the quick generator gives the word-for-word generator's bytes on {sum(map(len, trio))} items of three panels: {all(G.generate(W, tt, pr) == G.generate_slow(W, tt, pr) for pr in trio)}")
            try:
                G.generate(W, tt, trio[0] + trio[1]); refused = False
            except AssertionError:
                refused = True
            print(f"      a call that holds items of an earlier task and of the new family at one length is refused by the reader: {refused}")
            # a term that holds something, and code rows that have moved: what they reach
            for r in RANKS:
                term = G.new_term(r, rng, up_scale=0.05)
                V = G.with_code_rows(W, W["book.weight"][list(G.CODES)] + rng.normal(0, 1.0, (4, 8)).astype(np.float32))
                same_logits = same_answers = True; forced = 0.0
                for n in EARLIER6:
                    x, _ = batch(earlier[n][12]); ref = plain_forward(W, x)[3]
                    assert not G.gate_of(x)
                    same_logits &= np.array_equal(G.forward(V, term, x)[1], ref)
                    forced = max(forced, float(np.abs(G.forward(V, term, x, gate=True)[1] - ref).max()))
                    same_answers &= exact(V, term, earlier[n][12])[1] == base_out[n]
                moved = 0.0; red_now = []
                for n in ("add83", "sub83"):
                    x, _ = batch(plain[n]["reduction"][:512]); assert G.gate_of(x)
                    moved = max(moved, float(np.abs(G.forward(V, term, x)[1] - plain_forward(W, x)[3]).max()))
                    red_now.append(100 * exact(V, term, plain[n]["reduction"])[0] / len(plain[n]["reduction"]))
                print(f"      rank {r:2d}, a term with `up` drawn in +-0.05 and the four code rows moved by a standard normal: the earlier tasks' logits with the gate read from the code byte"
                      f" equal the lived network's value for value: {same_logits}; their generated answers item for item: {same_answers}; forced on, their logits move by up to {forced:.1f};"
                      f" the new family's logits move by up to {moved:.1f} (reduction score {np.mean(red_now):.1f})", flush=True)
            # the other lengths of the life's panels, and the composition panels' code
            other = sum(b in G.CODES for t in LIFE + F3 for k in ks_life for o in exact(W, zero, earlier[name(t)][k])[1] for b in o)
            print(f"      bytes e f g h among the bytes generated on the life's panels at every length {ks_life}: {other}")
    print(f"   no earlier task's generated byte is e, f, g or h in any network read: {never}")

if "C" in want:
    print("C. the two rules on the two factors (the algebra of exp3_state.SignLinear and of AdamW, on stand-in errors)")
    rng = np.random.default_rng(48)
    lr, wd, b1, b2, eps = 0.001, 0.25, 0.9, 0.98, 1e-8

    def adamw(w, g, m, v, t):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w = w * (1 - lr * wd) - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        return w, m, v

    S = lambda w: np.sign(w) * np.abs(w).mean()                           # what the sweep sends errors back through
    for r in RANKS:
        x = np.tanh(rng.normal(0, 1, (1024, 512)))                        # stand-in for the 512 inputs of the input map
        down = rng.uniform(-512 ** -0.5, 512 ** -0.5, (r, 512)); up = np.zeros((768, r))
        g1 = rng.normal(0, 1e-3, (1024, 768))                             # stand-in for the error at the term's output
        gu = g1.T @ (x @ down.T)                                          # the same under both rules
        back_B, back_S = g1 @ up, g1 @ S(up)
        gd_B, gd_S = back_B.T @ x, back_S.T @ x
        up1, mu, vu = adamw(up, gu, 0, 0, 1)
        down1, md, vd = adamw(down, gd_S, 0, 0, 1)
        g2 = rng.normal(0, 1e-3, (1024, 768))
        e_B, e_S = g2 @ up1, g2 @ S(up1)
        cos = float((e_B * e_S).sum() / np.linalg.norm(e_B) / np.linalg.norm(e_S))
        down2, _, _ = adamw(down1, e_S.T @ x, md, vd, 2)
        print(f"   rank {r:2d}: update 1: gradient of `up` up to {np.abs(gu).max():.1e}; error sent on to `down`: backpropagation {np.abs(back_B).max():.0e}, sweep {np.abs(back_S).max():.0e};"
              f" gradient of `down` {np.abs(gd_B).max():.0e} / {np.abs(gd_S).max():.0e}; after it every entry of `up` is {np.abs(up1).min() / lr:.4f} to {np.abs(up1).max() / lr:.4f} times the rate,"
              f" `down` is its start times {float((down1 / down).mean()):.5f}")
        print(f"            update 2: sign(up) x mean|up| differs from `up` by at most {np.abs(S(up1) - up1).max() / lr:.4f} of the rate; cosine of the two rules' errors into `down` {cos:.6f};"
              f" `down` then moves by {np.abs(down2 - down1 * (1 - lr * wd)).mean() / lr:.2f} of the rate an entry (its mean size {np.abs(down).mean():.3f})")
    print(f"   decay alone over 256 / 1,024 / 2,048 / 8,192 updates multiplies a weight by {(1 - lr * wd) ** 256:.3f} / {(1 - lr * wd) ** 1024:.3f} / {(1 - lr * wd) ** 2048:.3f} / {(1 - lr * wd) ** 8192:.3f}")

if "D" in want:
    print("D. E40's reduction score (held-out add and sub pairs that need the reduction, no letters) by row, lived runs")
    tab = {}
    for rule in ("S", "B", "K"):
        for seed in SEEDS:
            d = receipt(rule, seed)
            assert [r["updates"] for r in d["rows"]] == list(E40ROWS)
            tab[(rule, seed)] = {r["updates"]: r["scores"]["summary"]["reduction_score"] for r in d["rows"]}
            n = [d["rows"][0]["scores"]["held_plain_k0"][t]["reduction"]["n"] for t in ("add83", "sub83")]
            v = tab[(rule, seed)]
            print(f"   lived {rule} {seed} ({n[0]} + {n[1]} pairs): " + ", ".join(f"{v[u]:.1f}" for u in E40ROWS)
                  + " | row to row from 64: " + ", ".join(f"{v[b] - v[a]:+.1f}" for a, b in zip(E40ROWS[1:], E40ROWS[2:])))
    moves = {k: [abs(v[b] - v[a]) for a, b in zip(E40ROWS[1:], E40ROWS[2:])] for k, v in tab.items()}
    for rule in ("S", "B", "K"):
        allm = [m for (ru, _), ms in moves.items() if ru == rule for m in ms]
        print(f"   lived {rule}: of {len(allm)} moves between neighbouring rows from 64 on, {sum(m > 10 for m in allm)} exceed 10 points; the largest {max(allm):.1f}; the median {np.median(allm):.1f}")
    p = 0.8; n = 2 * 3100
    print(f"   sampling error of one score at {100 * p:.0f}% on about {n:,} pairs: {100 * (p * (1 - p) / n) ** 0.5:.1f} points; a move of 10 points is not sampling")
    mean3 = lambda v, rows: float(np.mean([v[u] for u in rows]))
    print("   the reading of the brief applied to E40 against itself, one row along (a run that is E40's own curve read a row early or late)")
    print("   run | row 256 against row 128, against row 512 | mean of rows 128, 256, 512 against mean of 64, 128, 256, against mean of 256, 512, 1,024 | best of 128, 256, 512 against best of 64, 128, 256 and of 256, 512, 1,024")
    worst = dict(single=0.0, mean=0.0, best=0.0)
    for (rule, seed), v in tab.items():
        s = (v[128] - v[256], v[512] - v[256])
        m = (mean3(v, (64, 128, 256)) - mean3(v, (128, 256, 512)), mean3(v, (256, 512, 1024)) - mean3(v, (128, 256, 512)))
        b0 = max(v[u] for u in (128, 256, 512)); b = (max(v[u] for u in (64, 128, 256)) - b0, max(v[u] for u in (256, 512, 1024)) - b0)
        if rule != "K":
            for key, val in (("single", s), ("mean", m), ("best", b)):
                worst[key] = max(worst[key], max(abs(x) for x in val))
        print(f"   lived {rule} {seed} | {s[0]:+.1f}, {s[1]:+.1f} | {mean3(v, (128, 256, 512)):.2f}: {m[0]:+.1f}, {m[1]:+.1f} | {b0:.1f}: {b[0]:+.1f}, {b[1]:+.1f}")
    print(f"   largest difference over the six runs of the sweep and of backpropagation: single row {worst['single']:.1f}; mean of three rows {worst['mean']:.1f}; best of three rows {worst['best']:.1f}")
    print("   the reading itself (within 10 points or above = no price; two of three seeds), E40's shifted curve against E40:")
    for rule in ("S", "B"):
        for label, f in (("row 128 read as row 256", lambda v: v[128] - v[256]), ("row 512 read as row 256", lambda v: v[512] - v[256]),
                         ("three-row mean, a row early", lambda v: mean3(v, (64, 128, 256)) - mean3(v, (128, 256, 512))),
                         ("three-row mean, a row late", lambda v: mean3(v, (256, 512, 1024)) - mean3(v, (128, 256, 512)))):
            dv = [f(tab[(rule, s)]) for s in SEEDS]
            print(f"      {rule}, {label}: differences " + ", ".join(f"{x:+.1f}" for x in dv) + f"; seeds within 10 points or above: {sum(x >= -10 for x in dv)} of 3; the reverse comparison: {sum(-x >= -10 for x in dv)} of 3")
    print("   prediction P48.7's row: mean of rows 512 and 1,024, against the mean of 256 and 512 and of 1,024 and 2,048; and row 1,024 alone against rows 512 and 2,048")
    for rule in ("S", "B"):
        for seed in SEEDS:
            v = tab[(rule, seed)]; c = mean3(v, (512, 1024))
            print(f"      lived {rule} {seed}: {c:.2f}: {mean3(v, (256, 512)) - c:+.1f}, {mean3(v, (1024, 2048)) - c:+.1f} | {v[1024]:.1f}: {v[512] - v[1024]:+.1f}, {v[2048] - v[1024]:+.1f}")
    print("   the taught panel (add and sub, 12 letters) at rows 64, 128, 256, 512, 1,024")
    for rule in ("S", "B"):
        for seed in SEEDS:
            d = receipt(rule, seed)
            print(f"      lived {rule} {seed}: " + ", ".join(f"{r['scores']['summary']['taught_add_sub_mean']:.1f}" for r in d["rows"][1:6]))

if "E" in want:
    print("E. cost, from E40's receipts (the same stream, the same scores, one thread)")
    per_u, per_row = [], []
    for rule in ("S", "B"):
        for seed in SEEDS:
            d = receipt(rule, seed)
            u = d["cpu_updates"] / d["updates"]; s = d["cpu_scoring"] / len(d["rows"])
            per_u.append(u); per_row.append(s)
            print(f"   lived {rule} {seed}: {u:.3f} CPU-seconds an update, {s:.1f} a scored row (row 0: {d['rows'][0]['cpu']:.1f}); host {d['provenance']['host']['platform'][:28]}, torch {d['provenance']['host']['torch']}")
    run = 1024 * max(per_u) + 8 * max(per_row)
    print(f"   one run of E48 at the dearest of these: 1,024 updates and 8 scored rows = {run:.0f} CPU-seconds; 18 runs {18 * run / 3600:.2f} CPU-hours; with the earlier tasks answered a second time for the item-for-item check, under {18 * (run + 8 * 5) / 3600:.2f}")

if "F" in want:
    print("F. the bytes of every earlier family, from the harness's source at", REF)
    e11 = show(M + "exp11_cell.py"); e16 = show(M + "exp16_integrated.py"); e40 = show(M + "e40_new_family.py"); a26 = show(M + "arch26.py")
    used = set(b"0123456789")
    assert 'f"{a:02d}".encode()' in e11 and 'f"{a:02d}".encode()' in e16
    sym = ast.literal_eval(re.search(r"^SYM = (\{.*\})", e11, re.M).group(1)); used |= {ord(v) for v in sym.values()}
    dis = ast.literal_eval(re.search(r"^DISTRACT = list\((b\".*\")\)", e11, re.M).group(1)); used |= set(dis)
    assert '[ord("=")]' in e11 and 'b"="' in e16 and 'ord(";")' in e16 and 'lead + b"c"' in e16
    used |= {ord("="), ord(";"), ord("c")}
    codes = {}
    for f in subprocess.run(["git", "ls-tree", "--name-only", REF, M], capture_output=True, text=True).stdout.split():
        if f.endswith(".py"):
            for op, mod, code in re.findall(r'\("(add|sub|max|min)", (\d+), b"(.)"\)', show(f)):
                codes.setdefault(code, set()).add(int(mod))
    earlier_codes = {c: m for c, m in codes.items() if ord(c) not in G.CODES}
    used |= {ord(c) for c in earlier_codes}
    print(f"   task codes in every .py file of the harness folder: {dict(sorted((c, sorted(m)) for c, m in codes.items()))}")
    print(f"   bytes of the earlier families and of composition items: {bytes(sorted(used)).decode()} ({len(used)} bytes, values {min(used)} to {max(used)}); any of 101 to 104 among them: {bool(used & set(G.CODES))};"
          f" the codes e f g h belong to modulus {sorted(set().union(*(codes[c] for c in 'efgh')))} only")
    assert not used & set(G.CODES)
    print("   how E40's scorer reaches the forward:")
    for label, src, pat in (("exp16_integrated.score", e16, r"outs = E\.respond\(m, \[p for p, _ in items\], store, cap=CAP, silence=silence\)"),
                            ("exp11_cell.respond", e11, r"p = T\.tensor\(\[\[BOS, \*prompts\[i\]\] for i in ids\]\)"),
                            ("exp11_cell.respond", e11, r"lg = forward\(m, p, False, True, silence=silence\)\[0\]\[:, -1\]"),
                            ("exp11_cell.respond", e11, r"p = T\.cat\(\[p, yy\[:, None\]\], 1\)"),
                            ("arch26.install", a26, r"E\.forward = forward"),
                            ("e40_new_family.Scorer", e40, r"plain = \{n: \{w: s\(m, it, None\) for w, it in v\.items\(\)\} for n, v in self\.plain\.items\(\)\}"),
                            ("e40_new_family.Scorer", e40, r"earlier = \{n: s\(m, self\.P\[n\]\[12\], None\) for n in EARLIER6\}"),
                            ("e40_new_family.Learner", e40, r"return C\.step\(self\.m, self\.opt, items, self\.rule == \"S\"\)")):
        hit = re.search(pat, src)
        print(f"      {label}: `{hit.group(0) if hit else 'NOT FOUND'}`")
        assert hit
    c26 = show(M + "exp26_candidate.py")
    hit = re.search(r"logits = arch26\.forward\(m, x, local, True\)\[0\]", c26); assert hit
    print(f"      exp26_candidate.step: `{hit.group(0)}`  (E40's update calls arch26.forward by name; the spec's forward takes (m, term, x, local, gate))")
    spec = show("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py")
    print(f"      the specification file: `def forward(m, term, x, local, gate, silence=False)` present: {'def forward(m, term, x, local, gate, silence=False):' in spec};"
          f" a line that reads the gate from the item's bytes: {bool(re.search(r'x\[:, 1\]|isin|CODES\)', spec.split('def freeze')[0].split('def forward')[1]))};"
          f" a line that replaces E.forward: {'E.forward' in spec}")
