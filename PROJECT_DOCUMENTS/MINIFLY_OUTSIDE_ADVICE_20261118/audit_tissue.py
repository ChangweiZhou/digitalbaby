#!/usr/bin/env python3
"""Outside audit of the tissue of E63 at the level of its programs, 19 November 2026.

numpy, the standard library and gcc. The programs audited are read with `git show` at fixed commits and copied into a temporary
folder; the reviewer's tissue.c is built there and run through the reviewer's own set-up (tis.py). Nothing is written in the
repository. No learner is trained: each tissue lives one life of 16,384 bytes of the project's own stream on seed 73129, as in
the reviewer's runs, and is then shown a probe of 7,200 bytes.

    python3 audit_tissue.py            all parts (about ten minutes on four cores)
    python3 audit_tissue.py quick      parts 1, 2 and 4 only (about two minutes)
    python3 audit_tissue.py first      all parts, and in part 3 also the two first forms of the definition (about 20 minutes)

Part 1  the wiring at birth (E63's `birth`): who receives the input lines, and how far the output side lies
Part 2  the stream: the best possible share of next bytes right with and without the arithmetic, the project's yardsticks, and
        how many items a life holds
Part 3  four tissues run with the reviewer's program: activity by distance from the input lines; how much of the cells' output
        the last bytes explain when their effects are simply added; and, from a balanced probe of all 100 pairs of digits, how
        much of it belongs to the two digits jointly (a cell answering to two bytes together)
Part 4  the lines of the programs on which the audit's statements about feedback, emission and addition rest
"""
import ctypes
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=str(Path(__file__).resolve().parent)).stdout.strip()
C = "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/"
RT = C + "reviewer_tissue_20261119/"
SOURCES = {                                             # local name: (commit, path)
    "tissue.c": ("21246a44e", RT + "tissue.c"),
    "tis.py": ("21246a44e", RT + "tis.py"),
    "outq.py": ("21246a44e", RT + "outq.py"),
    "deps/TISSUE_STEP1/tissue_step1.py": ("21246a44e", RT + "deps/TISSUE_STEP1/tissue_step1.py"),
    "deps/TISSUE_EVENT/readings.py": ("21246a44e", RT + "deps/TISSUE_EVENT/readings.py"),
    "yesno/deps/score_emissions.py": ("21246a44e", RT + "yesno/deps/score_emissions.py"),
    "yesno/deps/life_stream.py": ("21246a44e", RT + "yesno/deps/life_stream.py"),
    "add_e57.py": ("43f6487c3", "ADDITION/addsys.py"),
    "add_e55.py": ("b33cc7128", C + "output_reach4_20261119/addcheck.py"),
    "add_e62.py": ("033d9b52e", "TISSUE_CHECK_20261119/addsys.py"),
}
SEEDS = (73129, 93683, 116089)
AS_RUN = dict(rule="timing", prune_only="1", norm="1", lines_learn="1", we="3/4", wi="1/4", gain="3", rho="1", gmax="8",
              mk_long="1", a_make="1/4096", make_period="1024")
FIRST_FORMS = [("the first form of the definition (formula 7)", dict(rule="first")),
               ("the coincidence form of the definition", dict(rule="coincidence"))]
FORMS = [("the wiring of the definition, strengths never changing", dict(rule="fixed")),
         ("N0 as run (THE_TISSUE_AS_RUN, sections 2 and 3)", dict(AS_RUN)),
         ("N256 as run (time constants up to 256 steps)", dict(AS_RUN, tau="u2-256")),
         ("L8-256 as run (long axons as well)", dict(AS_RUN, tau="u2-256", long="8-1/2"))]


def rule():
    print("=" * 100)


def fetch(tmp):
    for name, (ref, path) in SOURCES.items():
        r = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, cwd=ROOT)
        assert r.returncode == 0, (name, r.stderr.decode()[:200])
        p = Path(tmp) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.stdout)


# ---------------------------------------------------------------- part 1
def part1(ts):
    rule()
    print("PART 1. The wiring at birth (E63's birth(seed); 2,048 cells, 256 input lines on x = 0, 256 output lines on x = 1)")
    N = ts.N
    for seed in SEEDS:
        B = ts.birth(seed)
        b, _ = ts.stream(seed, 65536)
        used = set(b.tolist())
        enc = [i for i in range(N) if B["kind"][i] == ts.ENC]
        n_lines = [sum(1 for m in B["members"][i] if m >= N) for i in enc]
        n_used = [sum(1 for m in B["members"][i] if m >= N and (m - N) in used) for i in enc]
        nb = [sum(1 for m in B["members"][i] if m < N) for i in range(N) if i not in B["fbset"]]
        recv, dead = Counter(), 0
        for i in range(N):
            for s in B["src"][i]:
                if s >= N:
                    if (s - N) in used:
                        recv[s - N] += 1
                    else:
                        dead += 1
        cells_used = sum(1 for i in range(N) if any(s >= N and (s - N) in used for s in B["src"][i]))
        none = sorted(v for v in used if recv[v] == 0)
        dist = np.full(N, -1)
        fr = [i for i in range(N) if any(s >= N and (s - N) in used for s in B["src"][i])]
        dist[fr] = 1
        readers = defaultdict(list)
        for i in range(N):
            for s in B["src"][i]:
                if s < N:
                    readers[s].append(i)
        d = 1
        while fr:
            nxt = []
            for j in fr:
                for i in readers[j]:
                    if dist[i] < 0:
                        dist[i] = d + 1; nxt.append(i)
            fr, d = nxt, d + 1
        dd = dist[np.array(B["driver"])]
        print(f"1.1 seed {seed}: byte values in use {len(used)} of 256; other cells in a cell's neighbourhood, median {int(np.median(nb))}; "
              f"cells with an input line in reach {len(enc)}, each with a median of {int(np.median(n_lines))} lines of which {int(np.median(n_used))} are ever used")
        print(f"    connections from input lines at birth: {sum(recv.values())} from lines in use, {dead} from lines that are never used "
              f"({100 * dead / (dead + sum(recv.values())):.0f}%); cells that receive a line in use: {cells_used} of {N}")
        print(f"    receiving cells of a byte value in use: fewest {min(recv[v] for v in used)}, median {int(np.median([recv[v] for v in used]))}, "
              f"most {max(recv[v] for v in used)}; values with none: {len(none)} ({''.join(chr(v) if v != 10 else '<end>' for v in none) or 'none'})")
        print(f"    the 256 cells that drive the output lines lie {int(dd.min())} to {int(dd.max())} connections from the input lines (median {int(np.median(dd))})")
    b, _ = ts.stream(SEEDS[0], 262144)
    f = Counter(b.tolist()); n = len(b)
    print(f"1.2 how often a byte value is present (seed {SEEDS[0]}): the most frequent, '=' and the end mark, {100 * f[61] / n:.2f}% of bytes each; a digit "
          f"{100 * sum(f[48 + i] for i in range(10)) / 10 / n:.2f}% on average; a letter {100 * sum(f[65 + i] for i in range(26)) / 26 / n:.2f}%. "
          f"The definition's target of a cell's mean output is 1/8 = 12.50%")


# ---------------------------------------------------------------- part 2
def part2(ts, SE):
    rule()
    print("PART 2. The stream of step 1 (the first family's items one after another) and what a score on it can show")
    for seed in SEEDS:
        n = 262144
        b, its = ts.stream(seed, n)
        bl = b.tolist()
        form, arith, kind = [None] * n, [None] * n, [None] * n
        for (op, a, b2, k, s) in its:
            L = 10 + k
            seq = bl[s:s + L]
            for p in range(min(L, n - s)):
                t = s + p
                if p == 0: c, kd = ("task",), "the task byte"
                elif p == 1: c, kd = ("a1", op), "first number"
                elif p == 2: c, kd = ("a2", op, seq[1]), "first number"
                elif p == 3: c, kd = ("op", op), "the operation byte"
                elif p <= 3 + k: c, kd = ("gap", p - 4), "letters"
                elif p == 4 + k: c, kd = ("gap", k), "second number"
                elif p == 5 + k: c, kd = ("b2", seq[4 + k]), "second number"
                elif p == 6 + k: c, kd = ("eq",), "the equals sign"
                elif p == 7 + k: c, kd = ("ans1", op), "the answer"
                elif p == 8 + k: c, kd = ("ans2", op, seq[7 + k]), "the answer"
                else: c, kd = ("end",), "the end mark"
                form[t] = c; kind[t] = kd
                arith[t] = ("known", seq[p]) if kd == "the answer" else c

        def best(ctx):                                  # the most frequent byte of each context, over the whole stream
            tab = defaultdict(Counter)
            for t in range(n):
                tab[ctx[t]][bl[t]] += 1
            top = {c: max(cn, key=cn.get) for c, cn in tab.items()}
            return np.array([top[ctx[t]] == bl[t] for t in range(n)])
        rf, ra = best(form), best(arith)
        refs = SE.references(b)
        q0 = 3 * n // 4
        taught = sum(len(ts.split(t, .5, seed)[0]) for t in ts.DEV)
        whole = {nb: sum(1 for it in its if it[4] + 10 + it[3] <= nb) for nb in (16384, 65536, 262144)}
        print(f"2.1 seed {seed}: share of all next bytes right for a guesser that knows the form of an item and no arithmetic {100 * rf.mean():.2f}%; "
              f"that also does the arithmetic {100 * ra.mean():.2f}%")
        print(f"    the project's yardsticks (tables of the last 1 / 2 / 3 bytes, filled as the stream goes; last quarter of {n} bytes): "
              + " / ".join(f"{100 * (refs[k] == b)[q0:].mean():.2f}" for k in ("ref_1", "ref_2", "ref_3")) + "%")
        print(f"    whole items in a life of 16,384 / 65,536 / 262,144 bytes: {whole[16384]} / {whole[65536]} / {whole[262144]}; taught pairs of the four "
              f"tasks {taught}; so these lives show {whole[16384] / taught:.3f} / {whole[65536] / taught:.3f} / {whole[262144] / taught:.3f} of one pass over them")
        if seed == SEEDS[0]:
            kk = np.array(kind)
            print("    by kind of byte, seed 73129 (share of the stream; right by form alone; right with the arithmetic):")
            for kd in ("the task byte", "first number", "the operation byte", "letters", "second number", "the equals sign", "the answer", "the end mark"):
                m = kk == kd
                print(f"       {kd:20s} {100 * m.mean():5.1f}%   {100 * rf[m].mean():6.2f}   {100 * ra[m].mean():6.2f}")
    print("2.2 for comparison, from the record (receipt e42_G_B_73129.json, part 6 of audit_e61_relay.py): the dense network under back-propagation reached 80 on")
    print("    never-taught pairs after 6,400 updates of 64 items, which is 409,600 items, or 25.9 passes over the 15,840 taught pairs")


# ---------------------------------------------------------------- part 3
def distances(ts, A, B, present):
    N = ts.N
    lines = set(ts.LINE0 + int(v) for v in present)
    deg, src, isfb = A["deg"], A["c_src"], B["kind"] == ts.FBK
    dist = np.zeros(N, np.int16)
    fr = np.array([(not isfb[i]) and any(int(s) in lines for s in src[i, :deg[i]]) for i in range(N)])
    dist[fr] = 1; d = 1
    while fr.any():
        nxt = np.array([(dist[i] == 0) and (not isfb[i]) and bool(fr[src[i, :deg[i]][src[i, :deg[i]] < N]].any()) for i in range(N)])
        d += 1; dist[nxt] = d; fr = nxt
    return dist


def onehot(v, vals):
    pos = {x: i for i, x in enumerate(vals)}
    M = np.zeros((len(v), len(vals)))
    M[np.arange(len(v)), [pos[int(x)] for x in v]] = 1.0
    return M


def added(rec, b, lo, lags):
    """Share of the cells' output variance, pooled over cells, that the bytes at the given lags explain when their effects are
    added: fitted on the first half of the window, measured on the second."""
    Tn = rec.shape[0]; h = Tn // 2
    vals = sorted(set(b.tolist()))
    X = rec[:, rec.std(0) > 1e-9]
    D = np.concatenate([onehot(b[lo - L: lo - L + Tn], vals) for L in lags] + [np.ones((Tn, 1))], 1)
    W, *_ = np.linalg.lstsq(D[:h], X[:h], rcond=None)
    return 1 - ((X[h:] - D[h:] @ W) ** 2).sum() / ((X[h:] - X[:h].mean(0)) ** 2).sum()


def probe(reps, seed):
    """Every pair of digits (d1, d2) once in each repetition, in random order, each as the three bytes '+', d1, d2."""
    rng = np.random.default_rng(seed)
    out, a, b = [], [], []
    for _ in range(reps):
        for q in rng.permutation(100):
            d1, d2 = divmod(int(q), 10)
            out += [43, 48 + d1, 48 + d2]; a.append(d1); b.append(d2)
    return np.array(out, np.uint8), np.array(a), np.array(b)


def two_way(X, a, b, reps):
    """Sums of squares of a balanced layout of 10 x 10 pairs with `reps` readings each, for every cell (column of X)."""
    g = X.mean(0)
    ma = np.stack([X[a == l].mean(0) for l in range(10)]); mb = np.stack([X[b == l].mean(0) for l in range(10)])
    mab = np.stack([X[(a == l) & (b == m)].mean(0) for l in range(10) for m in range(10)])
    ss_a = 10 * reps * ((ma - g) ** 2).sum(0); ss_b = 10 * reps * ((mb - g) ** 2).sum(0)
    inter = mab - np.repeat(ma, 10, 0) - np.tile(mb, (10, 1)) + g
    ss_ab = reps * (inter ** 2).sum(0)
    ss_w = ((X - mab[a * 10 + b]) ** 2).sum(0)
    return ss_a, ss_b, ss_ab, ss_w, ((X - g) ** 2).sum(0)


def one_form(args):
    tmp, name, kw = args
    sys.path.insert(0, tmp)
    import tis
    ts = tis.ts
    kw = dict(kw, seed="73129", nbytes="16384")
    T = tis.make(kw)
    t, A, B, L, b, N, NB = T["t"], T["A"], T["B"], T["L"], T["b"], T["N"], T["NB"]
    REPS = 24
    pb, pa, pc = probe(REPS, 20261119)
    allb = np.ascontiguousarray(np.concatenate([b, pb]), np.uint8)
    run = lambda b0, b1, r0, r1, rec: L.tis_run(ctypes.byref(t), ctypes.c_void_p(allb.ctypes.data), ctypes.c_int64(b0), ctypes.c_int64(b1),
                                                ctypes.c_int64(r0), ctypes.c_int64(r1), ctypes.c_void_p(rec.ctypes.data))
    W = 4096
    rec = np.zeros((W, N), np.float32); run(0, NB, NB - W, NB, rec); rec = rec.astype(np.float64)
    lines = [f"3. {name}"]
    lines.append(f"   at byte 16,384: connections a cell {A['deg'].sum() / N:.2f}; made {int(t.made)}, withdrawn {int(t.given)}; cells whose threshold is above zero "
                 f"{int((A['th'] > 0).sum())} of {N}; gain, median {np.median(A['gain']):.2f}, cells at the bound {int((A['gain'] >= t.gmax).sum())}")
    dist = distances(ts, A, B, sorted(set(b[NB - W:].tolist())))
    mx, mean = rec.max(0), rec.mean(0)
    row = []
    for d in (1, 2, 3, 4, 8, int(dist.max())):
        m = dist == d
        if m.any():
            row.append(f"{d}: {int(m.sum())} cells, mean {mean[m].mean():.5f}, largest {np.median(mx[m]):.4f}, never above 0.01 {int((mx[m] < 0.01).sum())}")
    lines.append("   output in the last 4,096 bytes, by connections from the input lines (largest: of the median cell): " + "; ".join(row))
    drv = np.array(B["driver"])
    lines.append(f"   the cells that drive the output lines: largest output of any of them {mx[drv].max():.2e}; the definition emits a byte when one exceeds 1/2")
    lo = NB - W
    lines.append(f"   share of the output variance explained on bytes not used for the fit, effects added: the present byte {added(rec, b, lo, [0]):.3f}; "
                 f"the present and the previous byte {added(rec, b, lo, [0, 1]):.3f}; the last eight bytes {added(rec, b, lo, list(range(8))):.3f}")
    rp = np.zeros((len(pb), N), np.float32); run(NB, NB + len(pb), NB, NB + len(pb), rp)
    X = rp[2::3].astype(np.float64)                     # the reading at the second digit of each pair
    near = np.zeros(N, bool)
    for i in drv:
        near[i] = True
        for m in B["members"][int(i)]:
            if m < N:
                near[m] = True
    near &= B["kind"] != ts.FBK
    dfw = 100 * (REPS - 1)
    for label, sel in (("all cells", np.ones(N, bool)), ("the cells near the output lines", near)):
        use = sel & (X.std(0) > 1e-12)
        sa, sb, sab, sw, st = two_way(X[:, use], pa, pc, REPS)
        msw = sw / dfw
        tot = st.sum()
        own = np.sort((sab - 81 * msw) / np.maximum(st, 1e-300))        # each cell's own joint share
        lines.append(f"   probe, {label} ({len(st)} that vary): of the variance of the output at the second digit, the present digit {max(0.0, (sb - 9 * msw).sum() / tot):.3f}; "
                     f"the digit before it {max(0.0, (sa - 9 * msw).sum() / tot):.3f}; the two jointly, beyond their sum {max(0.0, (sab - 81 * msw).sum() / tot):.4f}; "
                     f"the rest (earlier bytes and the tissue's own state) {sw.sum() / tot:.3f}. The joint share cell by cell: median {max(0.0, own[len(own) // 2]):.4f}, "
                     f"99th cell in 100 {own[int(0.99 * (len(own) - 1))]:.3f}, largest {own[-1]:.3f}")
        if label == "all cells":
            raw = (sab - 81 * msw) / np.maximum(st, 1e-300); big = raw > 0.25; top = X[:, use].max(0)
            lines.append(f"      cells with a joint share above 0.25: {int(big.sum())}" + (f"; the largest output of such a cell in the probe: median {np.median(top[big]):.1e}, "
                         f"largest {top[big].max():.1e}" if big.any() else "") + f"; joint share among the {int((top >= 0.01).sum())} cells whose output reaches 0.01: "
                         f"median {max(0.0, float(np.median(raw[top >= 0.01]))):.4f}, largest {raw[top >= 0.01].max():.3f}")
    return "\n".join(lines)


def part3(tmp, forms):
    rule()
    print("PART 3. Tissues run with the reviewer's program (seed 73129; a life of 16,384 bytes, then a probe: every pair of digits")
    print("        24 times as the bytes '+', d1, d2, the output read at d2)")
    sys.path.insert(0, tmp)
    import tis
    tis.make(dict(rule="fixed", nbytes="64"))           # builds tissue.so once, before the processes divide
    with get_context("fork").Pool(4) as pool:
        for text in pool.imap(one_form, [(tmp, n, k) for n, k in forms]):
            print(text); sys.stdout.flush()


# ---------------------------------------------------------------- part 4
def part4(tmp):
    rule()
    print("PART 4. The lines of the programs on which the audit's statements rest (file as named in this script's SOURCES; line; text)")
    wanted = [("deps/TISSUE_STEP1/tissue_step1.py", ["z[YES_LINE] = 0.0; z[NO_LINE] = 0.0", "return b if xo[b] > EMIT else -1", "EMIT = 1 / 2", "REC_FLOOR = 0.01"]),
              ("tissue.c", ["if (t->kind[i] == FBK) { x[i] = 0.0; continue; }", "if (t->prune_only != 0.0) { t->c_cu[b + d]",
                            "double th = t->th[i] + ETA_T * (xi - a); t->th[i] = th < 0.0 ? 0.0 : th;"]),
              ("outq.py", ["the output end reads and", "rec = np.zeros((NB, N), np.float32)"]),
              ("add_e57.py", ["m = int(np.floor(w1 * hist[d1] + w2 * hist[d2]))", "act.append(col(('m', d1, d2, G, m) + tuple(hist[:G])))"]),
              ("add_e55.py", ["m1 = val[hist[-2]] + val[hist[-4]]; m2 = val[hist[-3]] + val[hist[-5]]", "f1 = m1 * NA + AI[p]"]),
              ("add_e62.py", ["tot = s.A + s.B; cnt = (tot >= np.arange(19)).astype(float)"])]
    for name, pats in wanted:
        text = (Path(tmp) / name).read_text().splitlines()
        ref, path = SOURCES[name]
        print(f"4. {name} = {path} at {ref}")
        for p in pats:
            hits = [(i + 1, l.strip()) for i, l in enumerate(text) if p in l]
            assert hits, (name, p)
            i, l = hits[0]
            print(f"      line {i}: {l[:230]}")


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    forms = FORMS + (FIRST_FORMS if len(sys.argv) > 1 and sys.argv[1] == "first" else [])
    with tempfile.TemporaryDirectory(prefix="audit_tissue_") as tmp:
        fetch(tmp)
        sys.path.insert(0, os.path.join(tmp, "deps", "TISSUE_STEP1")); sys.path.insert(0, os.path.join(tmp, "yesno", "deps"))
        import tissue_step1 as ts
        import score_emissions as SE
        part1(ts)
        part2(ts, SE)
        if not quick:
            part3(tmp, forms)
        part4(tmp)
        rule()


if __name__ == "__main__":
    main()
