"""Checks behind the audit of the E44 brief and its specification files at 436d9677 (16 November 2026).

Read-only.  Nothing is trained, torch is not used, and no code of the default branch is run: files are read with `git show`,
the world comes from e44_world_reader.py (written from the brief's text), saved weights from the numpy reader (np_gru.py).

  A  the reader beside the committed output for worlds 1 to 3; digests for worlds 1 to 8 (the calibration's streams, the
     schedule, the tasks); what the other readings of the brief's words give
  B  moduli and bytes used so far, rebuilt from the harness's source, beside the world's moduli and code bytes
  C  taught pairs by modulus; showings of each taught pair in a visit; the items behind the gate's level; the mirror share
  D  the schedule: families met in 64, 256 and 1,024 visits, first visits and returns, absences in visits and in updates,
     returns beyond the reach of decay
  E  the learner: E41's half-width births of seed 73129 on unseen families before any update (the brief's row-0 check)
  F  decay alone on the half-width births, and the decay over the spans the calibration reports
  G  cost, from E41's measured CPU-seconds per update
  H  the gate's limits beside what is on file: E40's lived and fresh networks, E41 at a tenth of the pairs
  I  the scorer's new quantity on an event file made without a network (a fixture the worker's scorer can be run on)

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_brief_checks.py [A B C D E F G H I]
"""
import hashlib, json, os, re, statistics, subprocess, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e44_world_reader as LW
from np_gru import load_weights, sigmoid

REF = "436d9677"
M = "MOD97_LOCAL_RULE_20261106/"
K = "PROJECT_DOCUMENTS/COORD_CHECKS_20261116/"
BOS, EOS = 257, 256
want = set(sys.argv[1:]) or set("ABCDEFGHI")
show = lambda path, ref=REF: subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def run_items(W, items):
    """Whole answers right (two digits and the end symbol), items of one length, at whatever width the weights have."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            ok &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - 3)]
    return ok


def pct(ok):
    return 100 * float(np.mean(ok)) if len(ok) else float("nan")


def schedule_sha(w, visits):
    return hashlib.sha256(bytes(w.schedule(visits))).hexdigest()


def tasks_sha(w):
    h = hashlib.sha256()
    for i, (f, op, p, code) in enumerate(w.task):
        h.update(bytes(code) + bytes([p, LW.SYMBOL[op], len(w.taught[i]) // 256, len(w.taught[i]) % 256]))
        for a, b in w.taught[i] + w.asked[i]:
            h.update(bytes([a, b]))
    return h.hexdigest()


if "A" in want:
    print("A. the reader beside the committed output, and digests for the calibration's worlds")
    mine = [ln for s in (1, 2, 3) for ln in LW.describe(s)]
    theirs = show(K + "OUTPUT_LW_WORLD_MAC.txt").rstrip("\n").split("\n")
    print(f"   worlds 1 to 3: {len(mine)} lines from the reader, {len(theirs)} committed; identical: {mine == theirs}")
    print("   world | 8 visits in turn at 64 updates | at 128 | at 256 | first 4 visits of a life | schedule of 1,024 visits | tasks (codes, moduli, splits)")
    for s in range(1, 9):
        d = [LW.World(s, visit_updates=u).digest(8, in_turn=True)[:16] for u in (64, 128, 256)]
        w = LW.World(s)
        print(f"   {s}     | {d[0]} | {d[1]} | {d[2]} | {w.digest(4)[:16]} | {schedule_sha(w, 1024)[:16]} | {tasks_sha(w)[:16]}")
    print("   the other reading of round(0.10 p^2) (a half rounded upwards):")
    for s in (1, 2, 3):
        a, b = LW.World(s, visit_updates=64), LW.World(s, visit_updates=64, half_up=True)
        changed = [i for i in range(224) if len(a.taught[i]) != len(b.taught[i])]
        first8 = sorted({a.task[i][2] for i in changed if a.task[i][0] < 8})
        print(f"      world {s}: {len(changed)} tasks get one more taught pair (moduli {sorted({a.task[i][2] for i in changed})}); among the first eight families: {first8};"
              f" in-turn digest {b.digest(8, in_turn=True)[:16]} ({'equal' if a.digest(8, in_turn=True) == b.digest(8, in_turn=True) else 'different'})")
    print("   a second lap by the runner's `branch` command from the state after 8 visits (families 0 to 7 again), and the probe list of 100 reduction pairs a task:")
    print("   world | second lap at 64 updates | at 128 | at 256 | probe list, families 0 to 7 (1,600 items)")
    for s in range(1, 9):
        d = []
        for u in (64, 128, 256):
            w = LW.World(s, visit_updates=u); d.append(w.stream_digest(w.branch(8, range(8)))[:16])
        w = LW.World(s); pl = w.probe_list(range(8)); h = hashlib.sha256()
        for i, a, b, prompt, target in pl:
            h.update(bytes([i // 256, i % 256, a, b]))
        print(f"   {s}     | {d[0]} | {d[1]} | {d[2]} | {h.hexdigest()[:16]} ({len(pl)} items)")
    # what the committed digests do and do not exercise
    for s in (1, 2, 3):
        w = LW.World(s)
        print(f"      world {s}: families in the first 4 visits of a life {sorted(set(w.schedule(4)))}; so that digest covers {4 * len(set(w.schedule(4)))} of 224 tasks; the in-turn digest covers 32")
    # the fall-back when the drawn absence class is empty: the E39 brief's words settle it; another order gives another life
    def schedule_cyclic(w, visits):
        r = Random(f"lw|{w.seed}|schedule"); out, last, met = [0], {0: 0}, 1
        for v in range(1, visits):
            if r.random() < w.p_new and met < 52:
                f = met; met += 1
            else:
                c = r.randrange(3); cl = ([], [], [])
                for h, lv in last.items():
                    cl[0 if v - lv <= 4 else (1 if v - lv <= 32 else 2)].append(h)
                ch = next(cl[(c + k) % 3] for k in range(3) if cl[(c + k) % 3])
                f = sorted(ch)[r.randrange(len(ch))]
            out.append(f); last[f] = v
        return out
    for s in (1, 2, 3):
        w = LW.World(s); a, b = w.schedule(1024), schedule_cyclic(w, 1024)
        print(f"      world {s}: with the next class in turn as the fall-back, {sum(x != y for x, y in zip(a, b))} of 1,024 visits go to another family (first at visit {next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)})")

if "B" in want:
    print("B. moduli and bytes used so far, from the harness's source at", REF)
    files = [f for f in subprocess.run(["git", "ls-tree", "-r", "--name-only", REF, M], capture_output=True, text=True).stdout.split() if f.endswith(".py")]
    tuples = {}
    for f in files:
        for op, mod, code in re.findall(r'\("(add|sub|max|min)", (\d+), b"(.)"\)', show(f)):
            tuples.setdefault((int(mod), code), set()).add(f.split("/")[-1])
    mods = sorted({m for m, _ in tuples}); codes = sorted({c for _, c in tuples})
    print(f"   {len(files)} .py files; moduli in task tuples: {mods}; codes: {''.join(codes)}")
    e11 = show(M + "exp11_cell.py"); e16 = show(M + "exp16_integrated.py")
    used = set(b"0123456789") | {ord(v) for v in "+-><"} | set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {ord("="), ord(";"), ord("c")} | {ord(c) for c in codes}
    assert 'SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}' in e11 and 'DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")' in e11 and 'ord(";")' in e16
    sw = show(M + "smallworld/sw_world.py")
    sw_sym = re.search(r'^SYMBOLS = b"(.*?)"', sw, re.M).group(1); sw_mod = re.search(r"^MODULI = list\(range\((\d+), (\d+)\)\)", sw, re.M).groups()
    small = set(sw_sym.encode()) | set(b"abcdefghijklmnopqrstuvwxyz")
    e43 = show(M + "e43_task.py")
    print(f"   family bytes: {len(used)}, the largest {max(used)}; the small world adds lower-case codes and the symbols {sw_sym} (largest byte {max(small)}), moduli {sw_mod[0]} to {int(sw_mod[1]) - 1}")
    rng = re.findall(r"range\((\d+), (\d+)\)", e43)
    print(f"   E43's task (e43_task.py): ranges {rng[:4]} and single bytes {re.findall(r'^(CODE|QUERY)[ ,A-Z]* = ([0-9, ]+)', e43, re.M)}")
    world_mods = [m for m in range(41, 101) if m not in (73, 83, 89, 97)]
    print(f"   the world's moduli: {len(world_mods)} from {min(world_mods)} to {max(world_mods)}; any in a task tuple of the harness: {sorted(set(world_mods) & set(mods))};"
          f" moduli of the harness inside 41 to 100: {[m for m in mods if 41 <= m <= 100]}")
    for s in (1, 2, 3):
        w = LW.World(s); cb = [b for t in w.task for b in t[3]]
        print(f"   world {s}: {len({t[3] for t in w.task})} distinct codes of {len(w.task)}; code bytes from {min(cb)} to {max(cb)}; in a family's or the small world's bytes: {sorted(set(cb) & (used | small))};"
              f" inside E43's 128 to 225: {sum(128 <= b <= 225 for b in cb)} of {len(cb)}; codes with both bytes equal: {sum(t[3][0] == t[3][1] for t in w.task)}")
    Ws, _ = load_weights(M + "exp45/e41/e41_S_h256_f50_73129.pt", REF); Wb, _ = load_weights(M + "exp45/e41/e41_B_h256_f50_73129.pt", REF)
    same = [i for i in range(258) if np.array_equal(Ws["book.weight"][i], Wb["book.weight"][i])]
    moved = [i for i in range(258) if i not in same]
    print(f"   byte table of the two half-width births (one seed, two rules): rows equal in both {len(same)}, of them at or above 128: {sum(i >= 128 for i in same)} of 130;"
          f" rows that differ: {bytes(b for b in moved if b < 256).decode('latin-1')!r}")
    print(f"   so rows 128 to 255 were moved by neither birth: a code's two bytes are read as the births' untouched initial rows (norms {np.linalg.norm(Ws['book.weight'][128:256], axis=1).min():.2f} to {np.linalg.norm(Ws['book.weight'][128:256], axis=1).max():.2f})")

if "C" in want:
    print("C. taught pairs by modulus, and what a visit gives")
    world_mods = [m for m in range(41, 101) if m not in (73, 83, 89, 97)]
    print("   modulus | taught pairs per operation | showings of each taught pair in a visit of 64 / 128 / 256 updates (12 taught items a task an update)")
    for p in (41, 45, 50, 60, 70, 80, 90, 100):
        n = round(0.10 * p * p)
        print(f"   {p:7d} | {n:5d} | {12 * 64 / n:5.2f} / {12 * 128 / n:5.2f} / {12 * 256 / n:5.2f}")
    ns = [round(0.10 * p * p) for p in world_mods]
    print(f"   over the 56 moduli: taught pairs {min(ns)} to {max(ns)} (median {statistics.median(ns):.0f}); below E40's 689: {sum(n < 689 for n in ns)}; below L1a's largest task (684): {sum(n < 684 for n in ns)};"
          f" below 264 (L1a's middle band's largest): {sum(n <= 264 for n in ns)}")
    print(f"   E40 for comparison: 689 taught pairs, 16 taught items a task an update: {16 * 64 / 689:.2f} showings in 64 updates, {16 * 8192 / 689:.0f} in 8,192")
    for s in range(1, 9):
        w = LW.World(s)
        print(f"   world {s}: moduli of the eight families taken in turn {w.moduli[:8]}; taught pairs {[len(w.taught[4 * f]) for f in range(8)]}")
    w = LW.World(1, visit_updates=256)
    cnt = {}
    for u, (taught, asked) in enumerate(w.visit(1, 1)):
        for i, a, b, prompt, target in asked:
            if w.task[i][1] in ("add", "sub"):
                key = (w.task[i][1], u >= 192, w.wrap(i, a, b), w.mirror(i, a, b))
                cnt[key] = cnt.get(key, 0) + 1
    for op in ("add", "sub"):
        last = sum(v for (o, late, wr, mi), v in cnt.items() if o == op and late and wr)
        whole = sum(v for (o, late, wr, mi), v in cnt.items() if o == op and wr)
        mir = sum(v for (o, late, wr, mi), v in cnt.items() if o == op and wr and mi)
        print(f"   world 1, family 1 (modulus {w.moduli[1]}), a visit of 256 updates: asked reduction items of {op} in the last 64 updates {last}, in the whole visit {whole}, of them with a taught mirror {mir} ({mir / whole:.3f})")
    mi = []
    for s in (1, 2, 3):
        w = LW.World(s)
        for f in range(56):
            i = 4 * f; red = [(a, b) for a, b in w.asked[i] if w.wrap(i, a, b)]
            mi.append(sum(w.mirror(i, a, b) for a, b in red) / len(red))
    print(f"   share of a family's asked add reduction pairs whose mirrored pair is taught, worlds 1 to 3: {min(mi):.3f} to {max(mi):.3f} (mean {statistics.mean(mi):.3f})")

if "D" in want:
    print("D. the schedule")
    def facts(s, visits):
        rows = LW.World(s).absences(visits)
        fam = sorted({r[1] for r in rows}); new = [r[0] for r in rows if r[2]]
        ab = [r[3] for r in rows if r[3] is not None]
        return rows, fam, new, ab
    print("   world | families met in 64 / 256 / 1,024 visits | visit of the 2nd, 3rd, 4th family | visits on family 0 alone before the 2nd | returns of 1,024 | returns by absence 1-4 / 5-8 / 9-16 / 17-32 / 33+")
    agg = {64: [], 256: [], 1024: []}; second = []; allab = []
    for s in range(1, 9):
        r1024, fam, new, ab = facts(s, 1024)
        met = [len(facts(s, v)[1]) for v in (64, 256, 1024)]
        b = [sum(lo <= d <= hi for d in ab) for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 9))]
        print(f"   {s}     | {met[0]:2d} / {met[1]:2d} / {met[2]:2d} | {new[1:4]} | {new[1]} | {len(ab)} | {b}")
    for s in range(1, 201):
        for v in (64, 256, 1024):
            agg[v].append(len(facts(s, v)[1]))
        r, fam, new, ab = facts(s, 1024); second.append(new[1]); allab += ab
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
    for v in (64, 256, 1024):
        print(f"   200 worlds, {v} visits: families met min {min(agg[v])}, median {statistics.median(agg[v]):.0f}, max {max(agg[v])}; lives with one family only {sum(x == 1 for x in agg[v])}, with two or fewer {sum(x <= 2 for x in agg[v])}")
    print(f"   200 worlds: the second family arrives at visit median {statistics.median(second):.0f} (tenth {q(second, .1)}, ninetieth {q(second, .9)}, latest {max(second)}); all 52 met within 1,024 visits in {sum(x == 52 for x in agg[1024])} of 200")
    n = len(allab)
    print(f"   200 worlds, returns of a 1,024-visit life: {n / 200:.0f} a life; absence of 1 visit {sum(d == 1 for d in allab) / n:.3f}; 1 to 4 {sum(d <= 4 for d in allab) / n:.3f}; 5 to 32 {sum(5 <= d <= 32 for d in allab) / n:.3f}; 33 or more {sum(d >= 33 for d in allab) / n:.3f}")
    print("   returns after more than 2,048 updates of absence (decay alone leaves nothing of a family that receives no gradient, memo 24.6), and after more than 1,024:")
    for vu in (64, 128, 256, 512):
        print(f"      visits of {vu:3d} updates: more than {2048 // vu} visits away {sum(d * vu > 2048 for d in allab) / n:.3f}; more than {1024 // vu} visits away {sum(d * vu > 1024 for d in allab) / n:.3f};"
              f" decay factor over a short (4), medium (32), long (64) absence {0.99975 ** (4 * vu):.2f}, {0.99975 ** (32 * vu):.2f}, {0.99975 ** (64 * vu):.3f}")
    per = []
    for s in range(1, 9):
        rows = LW.World(s).absences(1024); c = {}
        for r in rows:
            c[r[1]] = c.get(r[1], 0) + 1
        per.append(sorted(c.values()))
    flat = sorted(x for p in per for x in p)
    print(f"   worlds 1 to 8, visits to a family in a 1,024-visit life: median {statistics.median(flat):.0f}, least {flat[0]}, most {flat[-1]}; families with one visit only {sum(x == 1 for x in flat)} of {len(flat)}")

if "E" in want:
    print("E. the half-width births of seed 73129 on unseen families before any update (the brief's row-0 check, its sample)")
    theirs = show(K + "OUTPUT_LW_ROW0_MAC.txt").rstrip("\n").split("\n")
    out = []
    for rule, wseed in (("S", 1), ("B", 1), ("B", 2)):
        Wt, sha = load_weights(M + f"exp45/e41/e41_{rule}_h256_f50_73129.pt", REF)
        if (rule, wseed) in (("S", 1), ("B", 1)):
            print(f"   e41_{rule}_h256_f50_73129.pt: {sum(v.size for v in Wt.values()):,} parameters, GRU of {Wt['rnn.weight_hh_l0'].shape[1]} units, file sha256 {sha[:12]}")
        w = LW.World(wseed); rng = Random(0)
        for f in (0, 1, 2):
            row = []
            for j, op in enumerate(LW.OPERATIONS):
                i = 4 * f + j
                asked = rng.sample(w.asked[i], min(600, len(w.asked[i])))
                mk = lambda pairs: [(w.item(i, a, b, [])) for a, b in pairs]
                if op in ("add", "sub"):
                    red = [pr for pr in asked if w.wrap(i, *pr)]; non = [pr for pr in asked if not w.wrap(i, *pr)]
                    row.append(f"{op}: reduction {pct(run_items(Wt, mk(red))):.1f}, none {pct(run_items(Wt, mk(non))):.1f}")
                else:
                    row.append(f"{op}: {pct(run_items(Wt, mk(asked))):.1f}")
            out.append(f"born {rule} 73129 (half width), world {wseed}, family {f} (mod {w.moduli[f]}), before any update: " + "; ".join(row))
    for a, b in zip(out, theirs):
        print("   " + a + ("   [as committed]" if a == b else "   [COMMITTED: " + b.split("before any update: ")[1] + "]"))
    print(f"   lines identical to the committed output: {sum(a == b for a, b in zip(out, theirs))} of {len(theirs)}")

if "F" in want:
    print("F. decay alone on the half-width births: every parameter but the byte table multiplied by 0.99975^t, the birth family (mod 89), 1,500 pairs of each")
    DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
    allp = [(a, b) for a in range(89) for b in range(89)]
    probe = {t[0]: Random(f"kept|{t[0]}").sample(allp, 1500) for t in DEV}
    def plain(task, pairs):
        op, m, code = task
        return [(list(code) + list(f"{a:02d}".encode()) + [LW.SYMBOL[op]] + list(f"{b:02d}".encode()) + [ord("=")], list(f"{LW.answer(op, m, a, b):02d}".encode()) + [EOS]) for a, b in pairs]
    for rule in "SB":
        W0, _ = load_weights(M + f"exp45/e41/e41_{rule}_h256_f50_73129.pt", REF)
        line = []
        for t in (0, 256, 512, 1024, 2048, 4096):
            Wt = {k: (v if k == "book.weight" else v * np.float32(0.99975 ** t)) for k, v in W0.items()}
            red = []
            for task in DEV[:2]:
                ok = run_items(Wt, plain(task, probe[task[0]]))
                wr = np.array([LW.needs_wrap(task[0], 89, a, b) for a, b in probe[task[0]]])
                red.append(pct(ok[wr]))
            mx = pct(run_items(Wt, plain(DEV[2], probe["max"])))
            line.append((t, sum(red) / 2, mx))
        base = line[0][1]
        print(f"   {rule}: " + " | ".join(f"{t}: reduction {r:5.1f} ({100 * r / base:5.1f}% of start), max {m:5.1f}" for t, r, m in line))
    print("   the spans the calibration reports on (eight families in turn; the first family is 7 visits old at the end):")
    for vu in (64, 128, 256):
        print(f"      visits of {vu} updates: one visit later {vu} updates, factor {0.99975 ** vu:.3f}; seven visits later {7 * vu} updates, factor {0.99975 ** (7 * vu):.3f}")

if "G" in want:
    print("G. cost")
    rep = show(M + "E41_REPORT_73129.md")
    line = next(l for l in rep.split("\n") if "Idle CPU-s per update" in l)
    b = [float(x) for x in re.search(r"B ([\d.]+) / ([\d.]+) / ([\d.]+)", line).groups()]; s_ = [float(x) for x in re.search(r"S ([\d.]+) / ([\d.]+) / ([\d.]+)", line).groups()]
    print(f"   E41's check 6 (idle host, 64 items learned an update), half / present / double width: backpropagation {b}, sweep {s_} CPU-seconds an update")
    cal = 8 * 2 * 8 * (64 + 128 + 256)
    print(f"   the calibration: 8 worlds x 2 arms x 8 visits x (64 + 128 + 256) = {cal:,} updates; at the half width's rates {cal / 2 * (b[0] + s_[0]) / 3600:.2f} CPU-hours of updates"
          f" (an update here learns from 48 items and answers 16, three bytes each)")
    for vu in (64, 128, 256, 512):
        n = 1024 * vu
        print(f"   a life of 1,024 visits of {vu:3d} updates: {n:,} updates; half width {n * b[0] / 3600:.1f} (backpropagation) and {n * s_[0] / 3600:.1f} (sweep) CPU-hours; present width {n * b[1] / 3600:.1f} and {n * s_[1] / 3600:.1f}")

if "H" in want:
    print("H. the gate's limits beside what is on file")
    t = json.loads(show(M + "exp45/e40/E40_TABLE.json"))
    red = lambda run, row: (t["curves"][run][str(row)]["add83_red"] + t["curves"][run][str(row)]["sub83_red"]) / 2
    for kind in ("lived_S", "lived_B", "lived_K", "fresh_S", "fresh_B"):
        runs = [f"{kind}_{s}" for s in (73129, 93683, 116089)]
        print(f"   E40 {kind}: reduction pairs of the new family at row 0 {[round(red(r, 0), 1) for r in runs]}, 64 {[round(red(r, 64), 1) for r in runs]}, 256 {[round(red(r, 256), 1) for r in runs]}, 8,192 {[round(red(r, 8192), 1) for r in runs]}")
    for rule in "SB":
        vals = []
        for H in (256, 512, 1024):
            d = json.loads(show(M + f"exp45/e41/e41_{rule}_h{H}_f10_73129.json")); c = d["curve"]
            vals.append((H, c[-1]["step"], round(max(r["R"] for r in c), 1), round(c[-1]["R"], 1)))
        print(f"   E41 at a tenth of the pairs, seed 73129, {rule}: (width, dose, highest R, last R) {vals}")
    n = 2 * 125
    for p in (0.10, 0.25, 0.50, 0.90):
        print(f"   a family's first-visit level from about {n} asked reduction items (add and sub): at a true level of {p:.2f} the standard error is {(p * (1 - p) / n) ** 0.5:.3f}")

if "I" in want:
    print("I. the scorer's new quantity on an event file made without a network")
    def events(w, visits, in_turn, right):
        ev = dict(update=[], visit=[], task=[], correct=[], wrap=[], mirror=[])
        u = 0
        for v, f, taught, asked in w.stream(visits, in_turn):
            for i, a, b, prompt, target in asked:
                ev["update"].append(u); ev["visit"].append(v); ev["task"].append(i); ev["correct"].append(int(right(u, i, a, b)))
                ev["wrap"].append(int(w.wrap(i, a, b))); ev["mirror"].append(w.mirror(i, a, b))
            u += 1
        return {k: np.array(x) for k, x in ev.items()}
    def first_visit_levels(ev, w, sched, window=64):
        """family -> (level on asked reduction items of add, of sub, their mean, items counted): the family's first visit,
        its last `window` updates (the whole visit when it is shorter)."""
        out = {}
        for v, f in enumerate(sched):
            if f in out:
                continue
            lo = (v + 1) * w.visit_updates - min(window, w.visit_updates); hi = (v + 1) * w.visit_updates
            sel = (ev["update"] >= lo) & (ev["update"] < hi) & (ev["wrap"] == 1)
            lv = [], []
            for j in (0, 1):
                m = sel & (ev["task"] == 4 * f + j)
                lv[j].extend([float(ev["correct"][m].mean()) if m.any() else float("nan"), int(m.sum())])
            out[f] = (lv[0][0], lv[1][0], (lv[0][0] + lv[1][0]) / 2, lv[0][1] + lv[1][1])
        return out
    rule = lambda u, i, a, b: (7 * a + 3 * b + u + i) % 4 == 0
    for vu in (64, 256):
        w = LW.World(1, visit_updates=vu); sched = w.schedule(8, in_turn=True)
        ev = events(w, 8, True, rule)
        lv = first_visit_levels(ev, w, sched)
        print(f"   world 1, eight families in turn, visits of {vu} updates, an item right when (7a + 3b + update + task id) % 4 == 0: {len(ev['update'])} asked items")
        print("      family: level on add, on sub, the family's figure, items counted -> " + "; ".join(f"{f}: {a:.4f}, {b:.4f}, {m:.4f}, {n}" for f, (a, b, m, n) in lv.items()))
        fig = [lv[f][2] for f in range(1, 8)]
        print(f"      median of the family's figure over the second to the eighth family: {statistics.median(fig):.4f}")
        h = hashlib.sha256(); [h.update(bytes([int(c)])) for c in ev["correct"]]
        print(f"      sha256 of the `correct` column: {h.hexdigest()[:16]}; of `wrap`: {hashlib.sha256(bytes(int(x) for x in ev['wrap'])).hexdigest()[:16]}; of `mirror`: {hashlib.sha256(bytes(int(x) for x in ev['mirror'])).hexdigest()[:16]}")
