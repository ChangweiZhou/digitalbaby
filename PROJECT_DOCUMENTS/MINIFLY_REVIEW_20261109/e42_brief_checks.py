"""Checks behind the audit of the E42 brief and its specification file at 95f2d80f (16 November 2026).

Read-only. Nothing is trained, torch is not used, and no code of the default branch is run: files are read with `git show`,
and everything computed here comes from e42_cells_reader.py (a reader written from the brief's text) and np_gru.py.

  A  the draws from the text: digests and first values for three seeds; which configurations share their connections
  B  the reader with every cell connected and all periods 1 against the existing GRU reader, on an E26 birth
  C  `mean`, `step` and `source` at the start, under the brief's rules and under the two readings the brief fixed
  D  time: no cell depends on a later byte; a cell changes only at its own update positions; zero before its phase
  E  counts of weights that take part in the function, for the weight-items of each configuration
  F  the sign rule's one scale in `source`, beside the size of the weights it stands for, by the reader's period
  G  the replacement rule on synthetic strengths: the limit, the neighbourhood, the candidates, the hardest cases
  H  cost: the brief's ratios, the committed output's ratios, and what each gives for the 48 runs
  I  what the specification file's path line inserts when the file is copied to the harness folder
  J  where E41's scoring code is; the hashes and figures the brief quotes, against the files they come from

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e42_brief_checks.py [A B ... J]
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from random import Random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import np_gru as G
import e42_cells_reader as C

REF = "95f2d80f"
D = "origin/codex/minifly-consolidated-20261023"
SEEDS = (73129, 93683, 116089)
U = C.U
want = set(sys.argv[1:]) or set("ABCDEFGHIJ")


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def standin(seed=0):
    """Weights with the distributions torch gives a new network (not its values, which need torch's generator)."""
    rng = np.random.default_rng(seed)
    u = lambda shape, b: rng.uniform(-b, b, shape).astype(np.float32)
    W = {"book.weight": rng.standard_normal((258, 8)).astype(np.float32)}
    W["book.weight"][G.BOS] = 0
    W["l1.weight"], W["l1.bias"] = u((512, 8), 8 ** -.5), u((512,), 8 ** -.5)
    W["rnn.weight_ih_l0"], W["rnn.weight_hh_l0"] = u((768, 512), 1 / 16), u((768, 256), 1 / 16)
    W["rnn.bias_ih_l0"], W["rnn.bias_hh_l0"] = u((768,), 1 / 16), u((768,), 1 / 16)
    W["lm.weight"], W["lm.bias"] = u((512, 256), 1 / 16), u((512,), 1 / 16)
    W["l2.weight"], W["l2.bias"] = u((512, 512), 512 ** -.5), u((512,), 512 ** -.5)
    W["l3.weight"], W["l3.bias"] = u((257, 512), 512 ** -.5), u((257,), 512 ** -.5)
    return W


def items(seed, k, n=64):
    rng = Random(seed)
    out = []
    for _ in range(n):
        t = G.DEV[rng.randrange(4)]
        out.append(G.render(t, rng.randrange(89), rng.randrange(89), k, rng))
    return G.batch(out)[0]


if "A" in want:
    print("A. the draws, from the brief's text (sha256 of json.dumps of the list, first 16 digits)")
    print(f"   neighbourhood: sizes {sorted({len(C.neighbourhood(i)) for i in range(U)})}; cell 0's first eight {C.neighbourhood(0)[:8]}; "
          f"symmetric {all(i in C.neighbourhood(j) for i in range(U) for j in C.neighbourhood(i))}")
    for s in SEEDS:
        print(f"   seed {s}")
        for ps in ("4", "8", "16"):
            p = C.draw_periods(s, ps)
            f = C.draw_phases(s, p)
            print(f"      periods {ps:>2}: {C.digest(p)[:16]}  first twelve {p[:12]} | phases {C.digest(f)[:16]}  first twelve {f[:12]}")
        for N, wide in ((5, False), (10, False), (20, False), (10, True)):
            A = C.draw_connections(s, N, wide)
            print(f"      connections N={N:2d}{' wide' if wide else '     '}: {C.digest(A)[:16]}  cell 0 reads {A[0][:10]}{' ...' if N > 10 else ''}")
    s = SEEDS[0]
    same = C.draw_connections(s, 10) == C.Cells(s, 10, "8").A == C.Cells(s, 10, "16", "source").A
    sub = sum(set(a) <= set(b) for a, b in zip(C.draw_connections(s, 5), C.draw_connections(s, 10)))
    print(f"   F10, A10, P4, P8, P16 and the three variants share one set of connections per seed: {same}; "
          f"cells whose N=5 connections are among their N=10 connections: {sub} of 256")
    print(f"   a cell of period 1 draws phase 0: {set(C.draw_phases(s, [1] * U)) == {0}}")

if "B" in want:
    print("B. every cell connected, all periods 1: the cell reader against the GRU reader, E26 birth of seed 73129 under the sweep")
    W, sha = G.load_weights(G.P + "e26/e26_dev_S_73129.pt", D)
    S = C.Cells(73129, N=255, pset="1")
    for k in (0, 12, 40):
        x = items(5, k)
        Hs, lg = C.forward(W, S, x)
        _, _, Hs0, lg0 = G.forward(W, x)
        print(f"   {k:2d} letters: largest difference in states {np.abs(Hs - Hs0).max():.2e}, in logits {np.abs(lg - lg0).max():.2e}; "
              f"same answers at the three target positions: {bool((lg[:, -3:].argmax(-1) == lg0[:, -3:].argmax(-1)).all())}")
    t = G.DEV[0]
    pk = G.panel(t, G.split(t, .5, 73129)[1], 73129, (12, 40), 300)
    for k in (12, 40):
        x, tgt = G.batch(pk[k])
        ok = G.exact_teacher_forced(C.forward(W, S, x)[1], tgt)
        print(f"   length test at {k} letters through the cell reader: {100 * ok.mean():.2f} (E30 has 98.33 and 96.0 for this file)")

if "C" in want:
    print("C. the variants at the start (stand-in initial weights; 64 items with 12 letters; periods 1, 2, 4, 8, N = 10)")
    W0, x = standin(), items(7, 12)
    ref = None
    for v, kw, label in (("mean", {}, "mean, the brief"), ("step", {}, "step"), ("source", {}, "source, the brief"),
                         ("source", dict(form="afactor"), "source, the audit's form"), ("last", {}, "last"),
                         ("mean", dict(mean_by="existing"), "mean dividing by the steps that exist"),
                         ("source", dict(bias="once"), "source with the bias added once")):
        S = C.Cells(73129, 10, "8", v)
        W1 = S.start(W0, form=kw.get("form", "brief"))
        Hs, lg = C.forward(W1, S, x, **kw)
        if ref is None:
            ref, refH = lg, Hs
        dpos = np.abs(Hs - refH).max((0, 2))
        late = [t for t in range(x.shape[1]) if dpos[t] > 1e-5]
        print(f"   {label:38s} largest difference from `mean`: logits {np.abs(lg - ref).max():.2e}, states {dpos.max():.2e}"
              + (f"; positions where states differ by more than 1e-5: {late[0]} to {late[-1]}" if late else ""))
    for ps in ("4", "16"):
        d = []
        for v in ("step", "source"):
            S = C.Cells(73129, 10, ps, v); W1 = S.start(W0)
            Sm = C.Cells(73129, 10, ps, "mean"); Wm = Sm.start(W0)
            d.append(np.abs(C.forward(W1, S, x)[1] - C.forward(Wm, Sm, x)[1]).max())
        print(f"   periods set {ps}: step and source against mean, largest logit difference {d[0]:.2e}, {d[1]:.2e}")

if "D" in want:
    print("D. time, in the reader, for each variant (stand-in weights; periods 1, 2, 4, 8, 16; 64 items with 12 letters)")
    W0, x = standin(), items(9, 12)
    for v in C.VARIANTS:
        S = C.Cells(73129, 10, "16", v); W1 = S.start(W0)
        Hs, _ = C.forward(W1, S, x)
        x2 = x.copy(); t0 = 9; x2[:, t0] = (x2[:, t0] + 1) % 256
        Hs2, _ = C.forward(W1, S, x2)
        upd = np.array([(t >= S.f) & ((t - S.f) % S.p == 0) for t in range(x.shape[1])])
        changed = (Hs[:, 1:] != Hs[:, :-1]).any(0)
        zero = all((Hs[:, :S.f[i], i] == 0).all() for i in range(U))
        print(f"   {v:6s}: no state before position {t0} changes {bool(np.array_equal(Hs[:, :t0], Hs2[:, :t0]))}; some state at or after it changes "
              f"{bool((Hs[:, t0:] != Hs2[:, t0:]).any())}; changes only at update positions {bool((~changed | upd[1:]).all())}; zero before the phase {zero}")

if "E" in want:
    print("E. weights that take part in the function (the present network has 1,124,113; its recurrent matrix 196,608)")
    base = 1124113 - 196608
    for name, kw in (("N5", dict(N=5)), ("F10 A10 W10", dict(N=10)), ("N20", dict(N=20)), ("P4", dict(N=10, pset="4")), ("P8", dict(N=10, pset="8")),
                     ("P16", dict(N=10, pset="16")), ("P8-last, P8-mean", dict(N=10, pset="8", variant="mean")), ("P8-source", dict(N=10, pset="8", variant="source"))):
        S = C.Cells(73129, **kw); rec, a, wx = S.live_counts()
        print(f"   {name:17s} recurrent inside the mask {rec:6d}; weights a {a:4d}; Wh[1..] inside mask and period {wx:6d}; network {base + rec + a + wx:,}")
    S = C.Cells(73129, 10, "8", "source"); S.start(standin())
    print(f"   the tensors as the file makes them: a has {S.a.size} entries ({int((S.a != 0).sum())} non-zero, of which period-1 cells hold {int((S.p == 1).sum())}); "
          f"Wh[1..] has {S.Wx.size:,} entries ({int((S.Wx != 0).sum()):,} non-zero)")

if "F" in want:
    print("F. the sign rule's one scale in `source` beside the weights it stands for (stand-in start, N = 10, periods 1, 2, 4, 8)")
    W0 = standin()
    for form in ("brief", "afactor"):
        S = C.Cells(73129, 10, "8", "source"); W1 = S.start(W0, form)
        m3 = S.mask3(); Wh = np.abs(W1["rnn.weight_hh_l0"]) * m3
        live = np.tile(S.inside()[1:], (1, 3))[:, :, None] * m3[None]
        scale = (Wh.sum() + (np.abs(S.Wx) * live).sum()) / (m3.sum() + live.sum())
        row = []
        for per in (1, 2, 4, 8):
            rows = np.tile(S.p == per, 3)
            w = Wh[rows].sum() / m3[rows].sum()                        # mean size of a step-0 weight of such a cell
            eff = w * (1.0 if form == "brief" else 1.0 / per)           # the weight one step carries in the forward
            back = scale * (1.0 if form == "brief" else 1.0 / per)      # what the sign rule sends back for it
            row.append(f"period {per}: forward {eff:.4f}, back {back:.4f}, ratio {back / eff:.2f}")
        print(f"   {'the brief: Wh[k] = masked Wh / p' if form == 'brief' else 'the audit: a[k] times undivided Wh[k]'}: one scale {scale:.4f} | " + " | ".join(row))
    S = C.Cells(73129, 10, "8", "step"); W1 = S.start(W0)
    m3 = S.mask3(); print(f"   `step` for comparison: one scale {(np.abs(W1['rnn.weight_hh_l0']) * m3).sum() / m3.sum():.4f}, then times a = 1/p exactly, so the ratio is 1 for every period at the start")

if "G" in want:
    print("G. the replacement rule from the text, on synthetic strengths")
    for N, wide, label in ((10, False, "N = 10"), (20, False, "N = 20"), (5, False, "N = 5"), (10, True, "W10")):
        S = C.Cells(73129, N, wide=wide)
        mask = S.mask.copy(); made = np.zeros((U, U), int); rr = Random(f"e42|replace|73129"); rng = np.random.default_rng(1)
        tot, worst_cand = 0, 10 ** 9
        for ev, update in enumerate(range(512, 512 + 256 * 12, 256)):
            s = rng.uniform(0, 1, (U, U)) ** (4 if ev % 2 else 1)
            if ev == 3:                                                 # the hardest case: all but one connection of every cell at zero
                s[:] = 0
                for i in range(U):
                    s[i, np.nonzero(mask[i] & (np.arange(U) != i))[0][0]] = 1
            before = mask.copy()
            out = C.replace_event(mask, made, s, update, 0.5, rr, wide)
            tot += len(out)
            for i in range(U):
                gone = [j for (c, j, n) in out if c == i]
                if gone:
                    worst_cand = min(worst_cand, len([q for q in C.neighbourhood(i, wide) if not before[i, q]]) - (len(gone) - 1))
            assert all((mask[i] & (np.arange(U) != i)).sum() == N for i in range(U)) and mask.diagonal().all()
            assert all(n in C.neighbourhood(i, wide) and n != i for (i, j, n) in out)
            assert all(update - made[i, n] == 0 for (i, j, n) in out) and all(not before[i, n] for (i, j, n) in out)
            if ev == 3:
                most = max(sum(1 for (c, j, n) in out if c == i) for i in range(U))
        print(f"   {label:6s}: 12 events, {tot} replacements; the limit, the neighbourhood and 'not connected before' hold at every event; "
              f"most given up by one cell in the hardest case {most} of {N}; fewest candidates left for a cell's last draw {worst_cand}")

if "H" in want:
    print("H. cost")
    out = show("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/OUTPUT_E42_SPEC_MAC.txt")
    t = {ln.split(":")[0].split()[-1]: float(ln.split(":")[1].split()[0]) for ln in out.splitlines() if ln.startswith("7.")}
    print(f"   committed output, CPU-seconds per forward and backward under the sweep: {t}")
    print(f"   ratios to G: A10 {t['A10'] / t['G']:.2f}, P8 {t['P8'] / t['G']:.2f}, P8-source {t['P8-source'] / t['G']:.2f}  (the brief's text: 2.6 for P8, 5.4 for P8-source)")
    perk = (t["P8"] - t["A10"]) / 8
    r_out = dict(G=1, lim=t["A10"] / t["G"], P4=(t["A10"] + 4 * perk) / t["G"], P8=t["P8"] / t["G"], P16=(t["A10"] + 16 * perk) / t["G"], src=t["P8-source"] / t["G"])
    lim = t["A10"] / t["G"]; perk_b = (2.6 - lim) / 8
    r_brief = dict(G=1, lim=lim, P4=lim + 4 * perk_b, P8=2.6, P16=lim + 16 * perk_b, src=5.4)
    g = {"S": 0.1197, "B": 0.1114}                       # E41 check 6, present width, CPU-seconds per update on the cloud host
    row = 5.1 * (1 + (300 * 22 + 300 * 50) / (600 * 22 + 600 * 22 + 2000 * 10))    # E41's scored row plus the length test, by bytes read
    stage1 = [("G", "G"), ("F10", "lim"), ("A10", "lim"), ("P8", "P8")]
    stage2 = [("N5", "lim"), ("N20", "lim"), ("W10", "lim"), ("P4", "P4"), ("P16", "P16"), ("P8-last", "P8"), ("P8-mean", "P8"), ("P8-source", "src")]
    print(f"   cloud host, E41's check 6 at the present width: {g} CPU-seconds per update, 5.1 per scored row; with the length test about {row:.1f} per row for G")
    for label, r in (("the committed output's ratios", r_out), ("the brief's ratios (2.6, 5.4)", r_brief)):
        def hours(kind, rule):
            return (24576 * g[rule] + 96 * row) * r[kind] / 3600
        per = {n: hours(k, "S") for n, k in stage1 + stage2}
        s1 = 3 * sum(hours(k, "S") + hours(k, "B") for _, k in stage1)
        s2 = 3 * sum(hours(k, "S") for _, k in stage2)
        print(f"   {label}: CPU-hours for one run under the sweep: " + ", ".join(f"{n} {v:.1f}" for n, v in per.items()))
        print(f"      stage 1 {s1:.0f}, stage 2 {s2:.0f}, all 48 runs {s1 + s2:.0f} CPU-hours; "
              f"without P8-mean {s1 + s2 - 3 * per['P8-mean']:.0f}; without P8-mean and P8-last {s1 + s2 - 6 * per['P8-mean']:.0f}; "
              f"without those and N5, N20 {s1 + s2 - 6 * per['P8-mean'] - 6 * per['N5']:.0f}")
    print("   (a ratio of forward-and-backward times overstates the ratio per update, which also holds the optimizer's step; check 9 measures it)")
    B = 64
    parts = dict(input_map=B * 768 * 512, recurrent=B * 768 * 256, l2=B * 512 * 512, lm=B * 512 * 256, l3=B * 257 * 512)
    base = sum(parts.values())
    print(f"   multiply-adds per position in the forward, batch 64: {parts}; sum {base / 1e6:.1f} million")
    print(f"      added by the sums over a period of 8 (two sums of 8 terms over 64 x 768 values): {2 * 8 * B * 768 / 1e6:.2f} million, {100 * 2 * 8 * B * 768 / base:.1f}% of the sum")
    print(f"      added by `source` at K = 8 (seven more crossings of the recurrent map): {7 * parts['recurrent'] / 1e6:.1f} million, {100 * 7 * parts['recurrent'] / base:.0f}% of the sum")
    r_fast = dict(G=1, lim=t["A10"] / t["G"], P4=1.05 * t["A10"] / t["G"], P8=1.05 * t["A10"] / t["G"], P16=1.1 * t["A10"] / t["G"], src=(t["A10"] / t["G"]) * (1 + 7 * parts["recurrent"] / base))
    hours = lambda kind, rule: (24576 * g[rule] + 96 * row) * r_fast[kind] / 3600
    s1 = 3 * sum(hours(k, "S") + hours(k, "B") for _, k in stage1); s2 = 3 * sum(hours(k, "S") for _, k in stage2)
    print(f"   if the loop cost what its arithmetic costs (periods 5 to 10% above A10, `source` {r_fast['src'] / r_fast['lim']:.2f} times A10): stage 1 {s1:.0f}, stage 2 {s2:.0f}, "
          f"all 48 runs {s1 + s2:.0f} CPU-hours; without P8-mean {s1 + s2 - 3 * hours('P8', 'S'):.0f}")

if "I" in want:
    print("I. the path the specification file inserts, by where the file stands")
    for p in ("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e42_cells_spec.py", "MOD97_LOCAL_RULE_20261106/e42_cells.py"):
        print(f"   <root>/{p}  ->  {Path('/root_of_repo/' + p).parents[2] / 'MOD97_LOCAL_RULE_20261106'}")
    line = [ln for ln in show("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e42_cells_spec.py").splitlines() if "sys.path.insert" in ln]
    print(f"   the line: {line[0].strip()}")

if "J" in want:
    print("J. files and figures the brief relies on")
    ls = lambda ref, path: subprocess.run(["git", "ls-tree", "--name-only", ref, "--", path], capture_output=True, text=True).stdout.strip()
    print(f"   exp41_scale.py at {REF} on the default branch: {bool(ls(REF, 'MOD97_LOCAL_RULE_20261106/exp41_scale.py'))}; "
          f"on claude/wonderful-edison-8hx6i6: {bool(ls('origin/claude/wonderful-edison-8hx6i6', 'MOD97_LOCAL_RULE_20261106/exp41_scale.py'))}")
    e41 = show("OPUS_TASK_E41_COMPUTATION_TO_THRESHOLD_BY_WIDTH_AND_FRACTION_20261116.md")
    e42 = show("OPUS_TASK_E42_CELLS_WITH_LIMITED_LOCAL_CONNECTIONS_AND_PERIODS_20261116.md")
    for h in ("79618ab2", "adf933d7", "bbd71ab7", "65b1c7e0", "1df203fb"):
        print(f"   {h}: in the E42 brief {h in e42}, in the E41 brief {h in e41}")
    e30 = json.loads(show("MOD97_LOCAL_RULE_20261106/exp45/e30/E30_UNSEEN.json", D))
    got = {(r["arm"], r["seed"]): r["acc40"] for r in e30["rows"] if r["experiment"] == "E26"}
    print(f"   E30's accuracy at 40 letters for the E26 births: sweep {[got[('S', s)] for s in SEEDS]}, backpropagation {[got[('B', s)] for s in SEEDS]}")
    for d in ("PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e42_design_checks.py",):
        r = subprocess.run([sys.executable, "-I", d], capture_output=True, text=True).stdout
        import re
        got = {ln.split()[0]: re.search(r"\]\s+(\d\.\d\d)\s+\[", ln).group(1) for ln in r.splitlines() if re.match(r"\s+(2|4|8|16)\s+\[", ln)}
        print("   the reviewer's count of carrying cases, share of training cases by period (e42_design_checks.py, part D):", got)
