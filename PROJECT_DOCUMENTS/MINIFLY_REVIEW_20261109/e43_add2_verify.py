"""Verification of E43's second addendum (worker's branch claude/peaceful-hopper-8ofq4i at 58ed6b25; runs 2aecccd2).

Sixteen runs of 4,096 updates: the `unrelated` arm with items of the first family mixed into the stream (`unrelated-rehearse`
without decay and `unrelated-rehearse-decay` with the standing 0.25; six births each), and the `unrelated` arm started from
E42's final networks F10 and P8 of seed 73129 under each rule.

Read-only.  Nothing is trained, torch is not used, and no code of the worker's or the default branch is run: files are read
with `git show`, one at a time.  Forwards: for the present network and for F10 (period 1; weights outside the mask are zero)
a plain GRU written here over per-byte tables; for P8 the cell reader e42_cells_reader.py.  Items: np_gru.py (the first
family) and e43_unrelated_reader.py (the unrelated task, written from the brief's words).

  A  the receipts of the twelve rehearsed runs: the first family and the unrelated task at each scored row, the share of the
     first family's reduction score at 4,096, the items of each kind (recounted from the brief's generator), the first row at
     which the unrelated task's taught panel is at 80% beside the same arm without rehearsal, how far the weights moved
  B  the sixteen committed weight files: sha256 against the receipt; for the twelve rehearsed networks the reader's counts
     on score 1 (the first family, all pairs, no letters), score 2 (held out, 12 letters) and score 3 (the unrelated task's
     panels) beside the receipt's row 4,096
  C  E42's networks: the receipts row by row beside the six present networks of E43's `unrelated` arm; the reader on E42's
     final files (row 0) and on the two F10 files at 4,096; what the two P8 files at 4,096 lack, and what the reader gets
     from them with E42's final coefficients in place of the missing ones
  D  the predictions P43.15 to P43.18

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e43_add2_verify.py [A B C D]
"""
import hashlib, json, os, subprocess, sys, time
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, panel, render, DEV, BOS, EOS, sigmoid
import e43_unrelated_reader as U43
import e42_cells_reader as R42

REF = "58ed6b25"                       # claude/peaceful-hopper-8ofq4i
E42REF = "44f91edc"                    # claude/gallant-dijkstra-i9zgsr (the files of seed 73129 are those of 534df8f0; checked by sha256)
E43 = "MOD97_LOCAL_RULE_20261106/exp45/e43/"
SEEDS = (73129, 93683, 116089)
ROWS = (0, 64, 256, 1024, 2048, 4096)
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
want = set(sys.argv[1:]) or set("ABCD")


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref=REF):
    blob = show(path, ref)
    return {n.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(sh).copy() for n, (st, sh, raw) in tensors(blob).items()}, hashlib.sha256(blob).hexdigest()


def rec(tag, sub="addendum2/"):
    return json.loads(show(f"{E43}{sub}{tag}.json"))


def s1(row):
    """(reduction score, pairs without reduction, max, min), percent, from a receipt's row."""
    s = row["score1_first_family_all_pairs_k0"]
    other = 100 * (s["add89"]["other"]["exact"] + s["sub89"]["other"]["exact"]) / (s["add89"]["other"]["n"] + s["sub89"]["other"]["n"])
    return s["reduction_score"], other, s["max89"]["all"]["pct"], s["min89"]["all"]["pct"]


def new(row, which="taught"):
    return 100 * row["score3_new_task"][which]["exact"] / row["score3_new_task"][which]["n"]


def first80(rows, which="taught"):
    return next((r["updates"] for r in rows if new(r, which) >= 80), None)


ARMS = (("unrelated-rehearse", "unrelated"), ("unrelated-rehearse-decay", "unrelated-decay"))

# ---------------------------------------------------------------- A: the rehearsed runs, from the receipts
if want & set("AD"):
    R = {}
    for arm, plain in ARMS:
        for rule in "SB":
            for seed in SEEDS:
                R[(arm, rule, seed)] = rec(f"e43_{arm}_{rule}_{seed}")
                R[(plain, rule, seed)] = rec(f"e43_{plain}_{rule}_{seed}", sub="")

if "A" in want:
    print("A. the rehearsed runs, from the receipts")
    counts = {}
    for seed in SEEDS:
        rng = Random(f"e43|{seed}|joint"); n0 = 0; counts[seed] = {0: (0, 0)}
        for i in range(1, 64 * 4096 + 1):
            n0 += rng.randrange(2) == 0
            if i % 64 == 0 and i // 64 in ROWS:
                counts[seed][i // 64] = (n0, i - n0)
    for arm, plain in ARMS:
        print(f"   {arm} (decay {R[(arm, 'S', SEEDS[0])]['settings']['decay']}; comparison arm without rehearsal: {plain})")
        print("   rule seed    | row: first family reduction / without reduction / max / min . unrelated taught / held out")
        for rule in "SB":
            for seed in SEEDS:
                d = R[(arm, rule, seed)]; rows = d["rows"]
                assert [r["updates"] for r in rows] == list(ROWS)
                cells = [f"{u:,}: {s1(r)[0]:.1f}/{s1(r)[1]:.1f}/{s1(r)[2]:.1f}/{s1(r)[3]:.1f} . {new(r):.1f}/{new(r, 'held'):.1f}" for u, r in zip(ROWS, rows)]
                print(f"   {rule}    {seed:6d} | " + " | ".join(cells))
        print("   rule seed    | share of the starting reduction score at 4,096; lowest share at any row (row) | held out at 12 letters, add + sub of 600, rows 0 and 4,096 |"
              " taught panel at 80%: first row with rehearsal (unrelated items seen) / without | items at 4,096: unrelated, first family (recount equal) | decay read back | birth's row 0 equal to the comparison arm's")
        for rule in "SB":
            for seed in SEEDS:
                d = R[(arm, rule, seed)]; p = R[(plain, rule, seed)]; rows = d["rows"]
                start = s1(rows[0])[0]; share = [100 * s1(r)[0] / start for r in rows]
                low = min(range(1, 6), key=lambda i: share[i])
                h = lambda r: r["score2_first_family_heldout_k12"]["add89"]["exact"] + r["score2_first_family_heldout_k12"]["sub89"]["exact"]
                f_r, f_p = first80(rows), first80(p["rows"])
                seen = next(r["items_so_far"]["unrelated"] for r in rows if r["updates"] == f_r) if f_r else None
                it = rows[-1]["items_so_far"]; recount = all((r["items_so_far"]["unrelated"], r["items_so_far"]["first_family"]) == counts[seed][r["updates"]] for r in rows)
                wd = [g["weight_decay"] for g in d["settings"]["optimizer_groups_read_back"]]
                same0 = s1(rows[0]) == s1(p["rows"][0]) and rows[0]["tensors_sha256"] == p["rows"][0]["tensors_sha256"]
                print(f"   {rule}    {seed:6d} | {share[-1]:6.1f}%; {share[low]:5.1f}% ({ROWS[low]:,}) | {h(rows[0])}, {h(rows[-1])} | {f_r:,} ({seen:,}) / {f_p:,} ({f_p * 64:,}) |"
                      f" {it['unrelated']:,}, {it['first_family']:,} ({recount}; first family {100 * it['first_family'] / (it['unrelated'] + it['first_family']):.2f}% of items) | {wd} | {same0}")
        print("   the unrelated task's taught panel at rows 256 and 1,024, with rehearsal / without (the rows between which the 80% is passed)")
        for rule in "SB":
            print(f"   {rule}    " + "; ".join(f"{seed}: 256: {new(R[(arm, rule, seed)]['rows'][2]):.1f} / {new(R[(plain, rule, seed)]['rows'][2]):.1f}, 1,024: {new(R[(arm, rule, seed)]['rows'][3]):.1f} / {new(R[(plain, rule, seed)]['rows'][3]):.1f}" for seed in SEEDS))
        print("   how far the weights moved by 4,096 (norm of the change over the birth's norm; cosine with the birth), with rehearsal / without")
        for rule in "SB":
            for g in ("input map", "recurrent matrix", "lm", "l2", "l3"):
                a = [R[(arm, rule, seed)]["rows"][-1]["score4_movement"]["groups"][g] for seed in SEEDS]; b = [R[(plain, rule, seed)]["rows"][-1]["score4_movement"]["groups"][g] for seed in SEEDS]
                print(f"   {rule}    {g:16s} change {np.mean([x['rel_diff'] for x in a]):.2f} / {np.mean([x['rel_diff'] for x in b]):.2f}; cosine {np.mean([x['cosine'] for x in a]):.2f} / {np.mean([x['cosine'] for x in b]):.2f}")
    cpu = sum(R[k]["complete"]["cpu_total"] for k in R if "rehearse" in k[0]) / 3600
    print(f"   CPU of the twelve rehearsed runs: {cpu:.2f} hours")


# ---------------------------------------------------------------- forwards and panels
def fast(W, items, chunk=8000):
    """Whole answers right under teacher forcing, for a network of period 1 (the present network, or F10 with zeros outside its mask)."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"])
    gi_tok = z1 @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]; l2_tok = z1 @ W["l2.weight"].T + W["l2.bias"]
    Whh, bhh = W["rnn.weight_hh_l0"], W["rnn.bias_hh_l0"]; n = Whh.shape[1]; L = x.shape[1]; na = tgt.shape[1]
    out = []
    for s in range(0, len(x), chunk):
        xs, ts = x[s:s + chunk], tgt[s:s + chunk]
        h = np.zeros((len(xs), n), np.float32); ok = np.ones(len(xs), bool)
        for t in range(L):
            gi = gi_tok[xs[:, t]]; gh = h @ Whh.T + bhh
            r = sigmoid(gi[:, :n] + gh[:, :n]); z = sigmoid(gi[:, n:2 * n] + gh[:, n:2 * n]); c = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:]); h = (1 - z) * c + z * h
            if t >= L - na:
                ok &= (np.tanh(l2_tok[xs[:, t]] + h @ W["lm.weight"].T + W["lm.bias"]) @ W["l3.weight"].T + W["l3.bias"]).argmax(1) == ts[:, t - (L - na)]
        out.append(ok)
    return int(np.concatenate(out).sum())


class Saved:
    """The structure of an E42 weights file, in the form e42_cells_reader.forward takes."""

    def __init__(self, T, a=None):
        self.p, self.f, self.mask = T["periods"].astype(int), T["phases"].astype(int), T["mask"].astype(bool)
        self.K, self.variant, self.Wx = int(self.p.max()), "step", None
        self.a = T.get("a") if a is None else a

    def mask3(self):
        return np.tile(self.mask, (3, 1))


def cells(W, S, items, chunk=4000):
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64); na = tgt.shape[1]
    ok = [(R42.forward(W, S, x[s:s + chunk], form="afactor")[1][:, -na:].argmax(-1) == tgt[s:s + chunk]).all(1) for s in range(0, len(x), chunk)]
    return int(np.concatenate(ok).sum())


ALL = [(a, b) for a in range(89) for b in range(89)]
RED = {"add": [p for p in ALL if p[0] + p[1] >= 89], "sub": [p for p in ALL if p[0] - p[1] < 0]}
NOR = {"add": [p for p in ALL if p[0] + p[1] < 89], "sub": [p for p in ALL if p[0] - p[1] >= 0]}
plain = lambda t, pairs: [render(t, a, b, 0, None) for a, b in pairs]
S1 = {"add red": plain(DEV[0], RED["add"]), "add other": plain(DEV[0], NOR["add"]), "sub red": plain(DEV[1], RED["sub"]), "sub other": plain(DEV[1], NOR["sub"]),
      "max": plain(DEV[2], ALL), "min": plain(DEV[3], ALL)}


def s1_receipt(row):
    s = row["score1_first_family_all_pairs_k0"]
    return [s["add89"]["reduction"]["exact"], s["add89"]["other"]["exact"], s["sub89"]["reduction"]["exact"], s["sub89"]["other"]["exact"], s["max89"]["all"]["exact"], s["min89"]["all"]["exact"]]


def panels(seed):
    held = {f"{t[0]}{t[1]}": panel(t, split(t, .5, seed)[1], seed, ks=(12,), n=300)[12] for t in DEV}
    return held, {w: U43.panel(seed, w) for w in ("held", "taught")}


# ---------------------------------------------------------------- B: the weight files
if "B" in want:
    t0 = time.time()
    print("B. the committed weight files at 4,096 (reader / receipt's row 4,096)")
    print("   score 1: add with reduction, add without, sub with, sub without, max, min (of 3,916, 4,005, 3,916, 4,005, 7,921, 7,921); score 2: add, sub, max, min held out at 12 letters (of 300); score 3: unrelated held out, taught (of 300)")
    for arm, _ in ARMS:
        for rule in "SB":
            for seed in SEEDS:
                tag = f"e43_{arm}_{rule}_{seed}"; row = rec(tag)["rows"][-1]
                W, sha = load(f"{E43}addendum2/{tag}_row4096.pt")
                held, unrel = panels(seed)
                m1 = [fast(W, it) for it in S1.values()]; m2 = [fast(W, it) for it in held.values()]; m3 = [fast(W, unrel[w]) for w in ("held", "taught")]
                t2 = [row["score2_first_family_heldout_k12"][n]["exact"] for n in held]; t3 = [row["score3_new_task"][w]["exact"] for w in ("held", "taught")]
                print(f"   {arm:24s} {rule} {seed:6d}  file's sha256 equals the receipt's {sha == row['weights_sha256']} | score 1 {m1} / {s1_receipt(row)} equal {m1 == s1_receipt(row)}"
                      f" | score 2 {m2} / {t2} equal {m2 == t2} | score 3 {m3} / {t3} equal {m3 == t3}   [{time.time() - t0:.0f} s]", flush=True)
                del W


# ---------------------------------------------------------------- C: E42's networks
if want & set("CD"):
    E42RUNS = [(c, r) for c in ("F10", "P8") for r in "SB"]
    X = {k: rec(f"e43_unrelated-e42{k[0]}_{k[1]}_73129") for k in E42RUNS}
    DENSE = {(r, s): rec(f"e43_unrelated_{r}_{s}", sub="") for r in "SB" for s in SEEDS}

if "C" in want:
    t0 = time.time()
    print("C. E42's networks under the unrelated task (seed 73129, no decay, no replacement of connections), beside the six present networks")
    print("   first family at each row: reduction / without reduction / max / min, percent of all pairs; then the unrelated task's taught panel")
    for k, d in X.items():
        print(f"   E42 {k[0]:3s} {k[1]}        | " + " | ".join(f"{u:,}: {s1(r)[0]:.1f}/{s1(r)[1]:.1f}/{s1(r)[2]:.1f}/{s1(r)[3]:.1f} . {new(r):.1f}" for u, r in zip(ROWS, d["rows"])))
    for k, d in DENSE.items():
        print(f"   present {k[0]} {k[1]:6d} | " + " | ".join(f"{u:,}: {s1(r)[0]:.1f}/{s1(r)[1]:.1f}/{s1(r)[2]:.1f}/{s1(r)[3]:.1f} . {new(r):.1f}" for u, r in zip(ROWS, d["rows"])))
    print("   answers still right at rows 64 and 256, counts of all 31,684 first-family items (add, sub, max, min together)")
    tot = lambda r: sum(s1_receipt(r))
    for k, d in X.items():
        print(f"   E42 {k[0]:3s} {k[1]}        | row 0: {tot(d['rows'][0]):,}; 64: {tot(d['rows'][1]):,}; 256: {tot(d['rows'][2]):,}; 1,024: {tot(d['rows'][3]):,}; 4,096: {tot(d['rows'][5]):,}")
    for k, d in DENSE.items():
        print(f"   present {k[0]} {k[1]:6d} | row 0: {tot(d['rows'][0]):,}; 64: {tot(d['rows'][1]):,}; 256: {tot(d['rows'][2]):,}; 1,024: {tot(d['rows'][3]):,}; 4,096: {tot(d['rows'][5]):,}")
    print("   the start: the receipt's check of row 0 against E42's final row; optimizer groups; the reader on E42's final file (row 0) for score 1")
    held, unrel = panels(73129)
    for k, d in X.items():
        st = d["settings"]; chk = st["row0_against_e42_final_row"]["check"]
        T, sha = load(f"MOD97_LOCAL_RULE_20261106/exp45/e42/e42_{k[0]}_{k[1]}_73129.pt", E42REF)
        S = Saved(T); W = {n: v for n, v in T.items() if v.dtype == np.float32 and n not in ("a",)}
        # F10 has period 1 and zeros outside its mask, so the plain forward is its forward; P8 needs the cell reader
        zero0 = float(np.abs(W["rnn.weight_hh_l0"][~S.mask3()]).max())
        m1 = [fast(W, it) for it in S1.values()] if k[0] == "F10" else [cells(W, S, it) for it in S1.values()]
        if k[0] == "F10":
            sample = S1["add red"][:200]
            assert fast(W, sample) == cells(W, S, sample) and zero0 == 0.0 and int(S.p.max()) == 1, "the plain forward is not F10's"
        wd = [g["weight_decay"] for g in st["optimizer"]["groups"]]
        print(f"   E42 {k[0]:3s} {k[1]}  E42's final file's sha256 equals the start recorded {sha == st['start']['weights_sha256']}; row 0 equals E42's final row on A, R, taught, length {all(chk.values())}; decay {wd};"
              f" replacement events {st['replacement_events']} | score 1 at row 0 {m1} / {s1_receipt(d['rows'][0])} equal {m1 == s1_receipt(d['rows'][0])}   [{time.time() - t0:.0f} s]", flush=True)
        # the file at 4,096
        row = d["rows"][-1]
        W4, sha4 = load(f"{E43}addendum2/e43_unrelated-e42{k[0]}_{k[1]}_73129_row4096.pt")
        names = sorted(W4)
        zero_outside = float(np.abs(W4["rnn.weight_hh_l0"][~S.mask3()]).max())
        t3 = [row["score3_new_task"][w]["exact"] for w in ("held", "taught")]
        if k[0] == "F10":
            m1 = [fast(W4, it) for it in S1.values()]; m3 = [fast(W4, unrel[w]) for w in ("held", "taught")]
            print(f"            file at 4,096: sha256 equals the receipt's {sha4 == row['weights_sha256']}; largest recurrent weight outside E42's final mask {zero_outside:.1e};"
                  f" score 1 {m1} / {s1_receipt(row)} equal {m1 == s1_receipt(row)} | score 3 {m3} / {t3} equal {m3 == t3}")
        else:
            a_end = T["a"]
            m3 = [cells(W4, Saved(T, a=a_end), unrel[w]) for w in ("held", "taught")]
            a0 = (np.arange(S.K)[:, None] < S.p[None, :]) / S.p[None, :]
            m3b = [cells(W4, Saved(T, a=a0.astype(np.float32)), unrel[w]) for w in ("held", "taught")]
            print(f"            file at 4,096: sha256 equals the receipt's {sha4 == row['weights_sha256']}; tensors in the file: {len(names)}, the coefficients `a` among them: {'a' in W4};"
                  f" largest recurrent weight outside E42's final mask {zero_outside:.1e}")
            print(f"            `a` in E42's final file: shape {a_end.shape}, cells of period above 1: {int((S.p > 1).sum())}; the receipt's relative change of `a` by 4,096: {row['score4_added_parameters']['a']['rel_diff']:.3f}")
            print(f"            the reader on the file at 4,096 with E42's final `a` in place of the missing one: score 3 (unrelated held out, taught, of 300) {m3} / receipt {t3};"
                  f" with the starting `a` (1 / period): {m3b}   [{time.time() - t0:.0f} s]", flush=True)
        del T, W4

# ---------------------------------------------------------------- D: the predictions
if "D" in want:
    print("D. the predictions, scored from the receipts")

    def reading(arm, rule):
        out = []
        for seed in SEEDS:
            rows = R[(arm, rule, seed)]["rows"]; share = 100 * s1(rows[-1])[0] / s1(rows[0])[0]; tp = new(rows[-1])
            out.append("kept at no cost" if share >= 80 and tp >= 80 else "kept at a cost" if share >= 80 else "not kept" if share <= 20 else "between")
        best = max(set(out), key=out.count)
        return (best if out.count(best) >= 2 else "no reading in two of three seeds"), out

    for arm, _ in ARMS:
        for rule in "SB":
            rd, per = reading(arm, rule)
            print(f"   {arm}, {'the sweep' if rule == 'S' else 'backpropagation'}: {rd} {per}")
    p15 = reading("unrelated-rehearse", "S")[0] == "kept at no cost"; p16 = reading("unrelated-rehearse-decay", "S")[0] == "kept at no cost"
    def twice(arm, plain_, rule):
        return [first80(R[(arm, rule, s)]["rows"]) <= 2 * first80(R[(plain_, rule, s)]["rows"]) for s in SEEDS]
    t17 = {(arm, rule): twice(arm, pl, rule) for arm, pl in ARMS for rule in "SB"}
    p17 = all(sum(v) >= 2 for v in t17.values())
    lost = {k: 100 * s1(d["rows"][-1])[0] / s1(d["rows"][0])[0] <= 20 and new(d["rows"][-1]) >= 80 for k, d in X.items()}
    yn = lambda b: "right" if b else "wrong"
    print(f"   P43.15 unrelated-rehearse, the sweep: kept at no cost (60%): {yn(p15)}")
    print(f"   P43.16 unrelated-rehearse-decay, the sweep: kept at no cost (65%): {yn(p16)}")
    print(f"   P43.17 with rehearsal the taught panel reaches 80% no more than twice as many updates later, two of three seeds, under each rule (60%): {yn(p17)}; seeds that meet it: "
          + "; ".join(f"{a} {r}: {sum(v)} of 3" for (a, r), v in t17.items()))
    print(f"   P43.18 E42's F10 and P8 lost in all four runs (85%): {yn(all(lost.values()))} {lost}")
