"""The scorer for what a tissue gives out, on the stream of one life (PREP/life_stream.py). Python 3 and numpy only.

    from score_emissions import score
    res = score(given, L)                 # given: one entry per position, the byte given out for that position before the byte
                                          # arrived (0 to 255), or -1 for none; L = life_stream.life(seed)
    python3 score_emissions.py            # the test: three made-up emitters on seed 73129, and agreement with comparator_read.py
    python3 score_emissions.py --comparator DIR   # also score the comparator's records in DIR (comparator_<seed>_<N>.npz)

For each window and each kind of position it returns:
  n            the number of positions of the kind in the window;
  right        the share of them at which the given byte equals the byte of the stream (right at the first attempt: one byte is
               given per position, so the first attempt is the only one);
  given        the share of them at which a byte is given (given >= 0);
  ref_mode, ref_1, ref_2, ref_3   the share right of four reference guessers that the scorer computes from the stream alone:
               the most frequent byte, and tables of the last one, two and three bytes. Each is fitted on all earlier positions of
               the life (positions 0 to t - 1 for the guess at t) and gives at t the byte seen most often so far after the same
               context (for ref_mode: the most frequent byte so far); ties go to the smaller byte value; a context never seen before
               gives no byte, which counts as not right;
  ceiling      1 for RECALL and the table answers (the stream fixes the right byte there), absent for the other kinds.

Kinds (the labels of life_stream.py, the same as PREP/comparator_read.py on claude/e55-comparator-017dSEpW): the first family's answer
digits R1 and R2 in phase FIRST (where nothing is due); SIGN, EQ, END (phase KNOWABLE); RECALL by gap 1, 2, 4, 8 (DELAY); ANSWER1 and
ANSWER2 (the table phases), and the same on the first asking of a key. A kind with no position in a window is left out of that window.

Windows. windows="life" (the default): consecutive windows of 16,384 positions from position 0 through the whole life, the last one
shorter. windows="phase": each phase (QUIET left out) cut into windows of 16,384 from its start, plus its last 16,384 positions (or the
whole phase if shorter), as comparator_read.py does; the test shows that this mode gives comparator_read.py's shares exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import life_stream as LS          # noqa: E402

W = 16384
CEILING_KINDS = ("RECALL", "ANSWER")


def kinds(L):
    ph, cls, due, gap = L["phase"], L["cls"], L["due"], L["gap"]
    k = {"R1 (first family, phase FIRST)": (ph == 0) & (cls == 9), "R2 (first family, phase FIRST)": (ph == 0) & (cls == 10),
         "SIGN": due == 1, "EQ": due == 2, "END": due == 3}
    for g in (1, 2, 4, 8):
        k[f"RECALL gap {g}"] = (due == 4) & (gap == g)
    k["ANSWER1"] = due == 5
    k["ANSWER2"] = due == 6
    k["ANSWER1, key first asked"] = (due == 5) & L["first"]
    k["ANSWER2, key first asked"] = (due == 6) & L["first"]
    return k


def references(b):
    """The four reference guessers' bytes at every position (-1 for none), each fitted on positions 0 to t - 1 only."""
    n = len(b)
    out = {name: np.full(n, -1, np.int16) for name in ("ref_mode", "ref_1", "ref_2", "ref_3")}
    bl = b.tolist()
    # most frequent byte so far
    cnt = [0] * 256; best, bc = -1, 0
    rm = out["ref_mode"]
    for t, x in enumerate(bl):
        rm[t] = best
        cnt[x] += 1
        if cnt[x] > bc or (cnt[x] == bc and x < best):
            best, bc = x, cnt[x]
    # tables of the last k bytes: context -> [counts dict, best byte, best count]
    for k in (1, 2, 3):
        tab = {}; r = out[f"ref_{k}"]
        for t in range(n):
            if t < k:
                continue
            ctx = tuple(bl[t - k:t]) if k > 1 else bl[t - 1]
            e = tab.get(ctx)
            if e is not None:
                r[t] = e[1]
            else:
                e = [{}, -1, 0]; tab[ctx] = e
            x = bl[t]; c = e[0].get(x, 0) + 1; e[0][x] = c
            if c > e[2] or (c == e[2] and x < e[1]):
                e[1], e[2] = x, c
    return out


def windows_of(L, mode):
    n = len(L["bytes"])
    if mode == "life":
        return [("life", a, min(a + W, n), False) for a in range(0, n, W)]
    ws = []
    for name, s, e in L["spans"]:
        if name == "QUIET":
            continue
        ws += [(f"{name} {s}-{e}", a, min(a + W, e), False) for a in range(s, e, W)]
        ws.append((f"{name} {s}-{e}", max(s, e - W), e, True))
    return ws


def score(given, L, mode="life", refs=None):
    given = np.asarray(given, np.int16)
    b = L["bytes"].astype(np.int16)
    assert len(given) == len(b), "one entry per position of the life"
    refs = references(L["bytes"]) if refs is None else refs
    right = given == b
    rr = {k: v == b for k, v in refs.items()}
    K = kinds(L)
    rows = []
    for label, a, e, last in windows_of(L, mode):
        row = dict(window=label, start=a, end=e, last_window=last, kinds={})
        for kn, m in K.items():
            mm = m[a:e]; c = int(mm.sum())
            if not c:
                continue
            v = dict(n=c, right=float(right[a:e][mm].mean()), given=float((given[a:e][mm] >= 0).mean()))
            for rn, rv in rr.items():
                v[rn] = float(rv[a:e][mm].mean())
            if kn.startswith(CEILING_KINDS):
                v["ceiling"] = 1.0
            row["kinds"][kn] = v
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ the test
def test(seed=73129, comparator_dir=None):
    L = LS.life(seed); b = L["bytes"].astype(np.int16); n = len(b)
    refs = references(L["bytes"])
    K = kinds(L)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    print(f"seed {seed}: {n:,} positions; kinds: " + ", ".join(f"{k} {int(m.sum()):,}" for k, m in K.items()))

    # emitter 1: always the end mark
    print("\nEmitter 1: always the end mark (byte 10)")
    g1 = np.full(n, 10, np.int16)
    s1 = score(g1, L, refs=refs)
    check("given share is 1 in every window and kind", all(v["given"] == 1.0 for r in s1 for v in r["kinds"].values()))
    check("END is right everywhere (share 1); every other kind 0 (their bytes are never 10)",
          all((v["right"] == 1.0) == (k == "END") and (v["right"] in (0.0, 1.0)) for r in s1 for k, v in r["kinds"].items()))

    # emitter 2: a copy of the stream delayed by one
    print("\nEmitter 2: the stream delayed by one (the byte at t - 1 given for t; none at t = 0)")
    g2 = np.concatenate([[-1], b[:-1]]).astype(np.int16)
    s2 = score(g2, L, refs=refs)
    exp = {k: float((b[1:][m[1:]] == b[:-1][m[1:]]).mean()) for k, m in K.items()}
    agg = {}
    for r in s2:
        for k, v in r["kinds"].items():
            a = agg.setdefault(k, [0, 0.0]); a[0] += v["n"]; a[1] += v["right"] * v["n"]
    check("pooled over windows, each kind's share right equals the share of its positions whose byte repeats the byte before",
          all(abs(agg[k][1] / agg[k][0] - exp[k]) < 1e-12 for k in agg))
    print("       its shares, pooled: " + ", ".join(f"{k} {exp[k]:.4f}" for k in exp))
    check("RECALL is never right (the byte before the due X is always '='); ANSWER2 is right only where the answer's two digits are equal",
          all(exp[k] == 0.0 for k in exp if k.startswith("RECALL") or k.startswith("ANSWER1")))

    # emitter 3: the 3-byte table itself, written again independently of references()
    print("\nEmitter 3: the table of the last three bytes (an independent implementation with collections.Counter)")
    from collections import Counter, defaultdict
    tab = defaultdict(Counter); g3 = np.full(n, -1, np.int16); bl = L["bytes"].tolist()
    for t in range(3, n):
        c = tab[tuple(bl[t - 3:t])]
        if c:
            m = max(c.values()); g3[t] = min(x for x, v in c.items() if v == m)
        c[bl[t]] += 1
    check("its guesses equal the scorer's ref_3 at every position", np.array_equal(g3, refs["ref_3"]))
    s3 = score(g3, L, refs=refs)
    check("its share right equals ref_3 in every window and kind", all(v["right"] == v["ref_3"] for r in s3 for v in r["kinds"].values()))
    check("its given share is the share of positions whose 3-byte context was seen before",
          all(v["given"] >= v["right"] for r in s3 for v in r["kinds"].values()))

    # a table of the scores of emitter 3 and the references, pooled by phase's last window
    print("\nReference guessers in the last window of each phase (mode 'phase'), seed", seed)
    sp = score(g3, L, mode="phase", refs=refs)
    print("| phase | kind | n | ref_mode | ref_1 | ref_2 | ref_3 | ceiling |\n|---|---|---|---|---|---|---|---|")
    for r in sp:
        if r["last_window"]:
            for k, v in r["kinds"].items():
                print(f"| {r['window'].split()[0]} | {k} | {v['n']} | {v['ref_mode']:.4f} | {v['ref_1']:.4f} | {v['ref_2']:.4f} | {v['ref_3']:.4f} | {v.get('ceiling', '—')} |")

    # agreement with comparator_read.py
    if comparator_dir:
        print("\nAgreement with PREP/comparator_read.py (COMPARATOR_READ.json on claude/e55-comparator-017dSEpW, a77a3a0c)")
        cr = json.loads((Path(comparator_dir) / "COMPARATOR_READ.json").read_text())
        for p in sorted(Path(comparator_dir).glob("comparator_*_7*.npz")):
            sd = int(p.stem.split("_")[1]); d = np.load(p)
            Ls = L if sd == seed else LS.life(sd)
            assert np.array_equal(d["shown"], Ls["bytes"])
            res = score(d["given"].astype(np.int16), Ls, mode="phase")
            ref = cr[str(sd)]["phases"]; cmp = 0; bad = 0
            byph = {}
            for r in res:
                byph.setdefault(r["window"], []).append(r)
            for ph, rs in byph.items():
                theirs = ref[ph]["windows"] + [ref[ph]["last_window"]]
                assert len(theirs) == len(rs)
                for mine, th in zip(rs, theirs):
                    assert (mine["start"], mine["end"]) == (th["start"], th["end"])
                    for k, v in mine["kinds"].items():
                        cmp += 1; bad += (th[k]["n"] != v["n"]) or (th[k]["share"] != v["right"])
                    bad += len([k for k in th if k not in ("start", "end") and k not in mine["kinds"]])
            check(f"seed {sd}: {cmp} (window, kind) shares, all equal to comparator_read.py's (exactly)", bad == 0 and cmp > 0)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    return ok


if __name__ == "__main__":
    cd = sys.argv[sys.argv.index("--comparator") + 1] if "--comparator" in sys.argv else None
    sys.exit(0 if test(comparator_dir=cd) else 1)
