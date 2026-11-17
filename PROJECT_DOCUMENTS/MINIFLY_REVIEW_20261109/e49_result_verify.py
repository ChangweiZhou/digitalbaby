"""Verification of E49's eight runs (a limited interface, seed 73129; worker's branch claude/peaceful-noether-vcvrmf at a36de661).

Read-only.  Nothing is trained, torch is not used, and no code of the worker's or the default branch is run: files are read
with `git show`, one at a time.  The forward is e49_interface_reader.py (written from the brief's text: every sum runs over
index lists drawn by the reader, not over masked matrices); the panels come from np_gru.py.

  A  the final weights of the eight runs through the reader: the saved masks against the reader's draws, every weight outside
     a mask zero, the start file against E42's start times the mask times the brief's scale, the counts of right answers on
     the panels of A, R, the taught panel and the length test beside the receipt's last row, and every recorded answer of
     the length-test panel (300 at 12 letters, 300 at 40) as the largest output after the bytes recorded before it
  B  the ten receipts recomputed (the eight runs and E42's F10 under each rule): updates and multiply-adds to the 50, 80 and
     90 thresholds on A and R in the three-row and the single-row version, the three levels, the checks kept in every row,
     the training loss, the cosines between the sweep's and the exact gradient
  C  the arms in words, with the multiply-adds of one item at one position counted from the index lists
  D  the runs that do not reach 80 on R: the change of R at the end of the dose and a linear extrapolation
  E  the descriptions against F10 of seed 73129 in the brief's fixed forms, what a seed alone changes (E42's F10 on three
     seeds under each rule, from E42's branch at 44f91edc), and the eight predictions
  F  the stage-0 check receipts
  G  what the answers are (working rule 3): on the length-test panel at 12 letters, from the worker's answers files, how many
     answers have the form of two digits and the end symbol, how many different answers are given, and how often each digit
     is right; the same for a network that has not learned would be 300 different right answers reduced to a few values
  H  an observation outside the brief's forms, on the start and final files: the size of each map at the end over its size at
     the start, the code of a byte (the 512 values that `l1` makes of the byte table's row) for the bytes the task uses, and
     how much the cells' states at the `=` sign differ between items; E42's F10 of the same seed beside them

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e49_result_verify.py [A B C D E F G H]
"""
import json, os, subprocess, sys, time
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, panel, render, DEV, BOS, EOS
import e42_cells_reader as R42
import e49_interface_reader as R49

REF = "a36de661"                       # claude/peaceful-noether-vcvrmf, the report's commit
E42REF = "44f91edc"                    # claude/gallant-dijkstra-i9zgsr, E42 complete on three seeds
P = "MOD97_LOCAL_RULE_20261106/exp45/"
SEED, U, D = 73129, 256, 512
SETTINGS = ("I64", "I16", "in16", "out8")
RUNS = [(s, r) for r in "BS" for s in SETTINGS]
RULE = {"S": "the sweep", "B": "backpropagation"}
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
want = set(sys.argv[1:]) or set("ABCDEFGH")


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref=REF):
    return {name.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy() for name, (stype, shape, raw) in tensors(show(path, ref)).items()}


def receipt(setting, rule):
    if setting == "F10":
        return json.loads(show(f"{P}e42/e42_F10_{rule}_{SEED}.json", E42REF))
    return json.loads(show(f"{P}e49/e49_{setting}_{rule}_{SEED}.json"))


# ---------------------------------------------------------------- A: the weights through the reader
class Fast:
    """The reader's sums (e49_interface_reader.forward: every sum over the listed sources only), evaluated once per byte for
    the input map, which depends on the byte alone, and for the readout only at the positions that are scored.  Part A
    checks it against e49_interface_reader.forward itself on a sample of every run."""

    def __init__(self, W, mask_cells, F):
        cells = np.arange(U)
        z1 = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"])                 # (258, 512): the code of each byte
        Wi = W["rnn.weight_ih_l0"]
        self.gi = []
        for g in range(3):
            Wg = Wi[g * U + cells[:, None], F.in_idx]                                      # (256, K_IN)
            part = [np.einsum("vuk,uk->vu", z1[v:v + 32][:, F.in_idx], Wg) for v in range(0, len(z1), 32)]
            self.gi.append(np.concatenate(part) + W["rnn.bias_ih_l0"][g * U:(g + 1) * U])  # (258, 256)
        self.src = np.array([np.nonzero(mask_cells[i])[0] for i in range(U)])              # (256, N + 1)
        self.Wh = [W["rnn.weight_hh_l0"][g * U + cells[:, None], self.src] for g in range(3)]
        self.bh = [W["rnn.bias_hh_l0"][g * U:(g + 1) * U] for g in range(3)]
        self.Wo = W["lm.weight"][np.arange(D)[:, None], F.out_idx]                         # (512, K_OUT)
        self.out_idx, self.k_out = F.out_idx, F.k_out
        self.l2 = z1 @ W["l2.weight"].T + W["l2.bias"] + W["lm.bias"]                      # (258, 512)
        self.l3w, self.l3b = W["l3.weight"], W["l3.bias"]

    def states(self, x, at):
        """The cells' states after position `at`, (B, 256)."""
        h = np.zeros((x.shape[0], U), np.float32)
        for t in range(at + 1):
            hin = h[:, self.src]
            gh = [np.einsum("buk,uk->bu", hin, self.Wh[g]) + self.bh[g] for g in range(3)]
            r = R42.sigmoid(self.gi[0][x[:, t]] + gh[0]); z = R42.sigmoid(self.gi[1][x[:, t]] + gh[1])
            h = ((1 - z) * np.tanh(self.gi[2][x[:, t]] + r * gh[2]) + z * h).astype(np.float32)
        return h

    def logits(self, x, n_last):
        """Logits at the last n_last positions, (B, n_last, 257)."""
        B, L = x.shape
        h = np.zeros((B, U), np.float32); res = []
        step = max(8, int(2.0e7 / (D * self.k_out)))
        for t in range(L):
            hin = h[:, self.src]                                                           # (B, 256, N + 1)
            gh = [np.einsum("buk,uk->bu", hin, self.Wh[g]) + self.bh[g] for g in range(3)]
            r = R42.sigmoid(self.gi[0][x[:, t]] + gh[0]); z = R42.sigmoid(self.gi[1][x[:, t]] + gh[1])
            n = np.tanh(self.gi[2][x[:, t]] + r * gh[2])
            h = ((1 - z) * n + z * h).astype(np.float32)
            if t >= L - n_last:
                out = np.concatenate([np.einsum("bjk,jk->bj", h[s:s + step][:, self.out_idx], self.Wo) for s in range(0, B, step)])
                res.append(np.tanh(self.l2[x[:, t]] + out) @ self.l3w.T + self.l3b)
        return np.stack(res, 1)


def teacher(M, items):
    """Whole answers right under teacher forcing, which equals whole answers right under greedy generation."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    return int((M.logits(x, tgt.shape[1]).argmax(-1) == tgt).all(1).sum())


def recorded_is_greedy(M, items, recorded):
    """How many recorded answers are what greedy generation gives: each recorded byte is the largest output after the prompt
    and the recorded bytes before it, and the answer ends at the end symbol or at four bytes (exp11_cell.respond's cap)."""
    same = np.zeros(len(items), bool)
    by_len = {}
    for i, rec in enumerate(recorded):
        by_len.setdefault(len(rec), []).append(i)
    for n, idx in by_len.items():
        x = np.array([[BOS] + items[i][0] + recorded[i][:-1] for i in idx], dtype=np.int64); tgt = np.array([recorded[i] for i in idx], dtype=np.int64)
        same[np.array(idx)] = (M.logits(x, n).argmax(-1) == tgt).all(1)
    stops = all(EOS not in rec[:-1] and (rec[-1] == EOS or len(rec) == 4) for rec in recorded)
    return int(same.sum()), stops, sum(rec == a for rec, (_, a) in zip(recorded, items))


def against_reader(M, W, mask_cells, F, items, n=24):
    """The largest difference between Fast's logits and e49_interface_reader.forward's at the three answer positions, on n items."""
    x = np.array([[BOS] + p + a[:-1] for p, a in items[:n]], dtype=np.int64)
    ref = R49.forward(W, mask_cells, F, x)[1][:, -3:]
    mine = M.logits(x, 3)
    return float(np.abs(ref - mine).max()), bool((ref.argmax(-1) == mine.argmax(-1)).all())


if "A" in want:
    t0 = time.time()
    held = {t: split(t, .5, SEED) for t in DEV}
    name = lambda t: f"{t[0]}{t[1]}"
    AP = {name(t): panel(t, held[t][1], SEED, ks=(12,), n=300)[12] for t in DEV}
    TA = {name(t): panel(t, held[t][0], SEED + 1, ks=(12,), n=300)[12] for t in DEV}
    RP = {}
    for t in DEV[:2]:
        red = [(a, b) for a, b in held[t][1] if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        RP[name(t)] = [render(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]
    LEN_TASK = ("add", 89, b"p")
    LP = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    cells = R42.Cells(SEED, N=10, pset="1")                       # E42's F10: the reader's own draw of the ten connections
    print("A. the final weights through the reader (reader / receipt's last row)")
    F10_start = {r: load(f"{P}e49_qual/check2_73129_runs/e49_F10_{r}_dry2048_{SEED}_start.pt") for r in "SB"}
    same_start = all(np.array_equal(F10_start["S"][k], F10_start["B"][k]) for k in F10_start["S"])
    print(f"   E42's start of F10 (the start file of check 2's dry run) is the same file under the two rules: {same_start}")
    for setting, rule in RUNS:
        rc = receipt(setting, rule); last = rc["curve"][-1]
        k_in, k_out = R49.SETTINGS[setting]
        F = R49.Interface(SEED, k_in, k_out)
        T = load(f"{P}e49/e49_{setting}_{rule}_{SEED}.pt")
        m_in, m_out, m_rec = T["mask_in"].astype(bool), T["mask_out"].astype(bool), T["mask"].astype(bool)
        masks_ok = np.array_equal(m_in, F.mask_in) and np.array_equal(m_out, F.mask_out) and np.array_equal(m_rec, cells.mask)
        dig = (R49.digest16(F.in_src) if k_in < D else None, R49.digest16(F.out_src) if k_out < U else None)
        dig_ok = all(a is None or a == b for a, b in zip(dig, (rc["draw_digests"].get("input"), rc["draw_digests"].get("readout"))))
        z = lambda w, m: float(np.abs(w[~m]).max()) if (~m).any() else 0.0
        outside = (z(T["rnn.weight_ih_l0"], np.tile(m_in, (3, 1))), z(T["lm.weight"], m_out), z(T["rnn.weight_hh_l0"], np.tile(m_rec, (3, 1))))
        # the start file: E42's start, the kept weights times the brief's scale, the others zero, everything else untouched
        S0 = load(f"{P}e49/e49_{setting}_{rule}_{SEED}_start.pt"); base = F10_start[rule]
        exp_ih = (base["rnn.weight_ih_l0"] * np.tile(F.mask_in, (3, 1)) * np.float32((D / k_in) ** 0.5)).astype(np.float32) if k_in < D else base["rnn.weight_ih_l0"]
        exp_lm = (base["lm.weight"] * F.mask_out * np.float32((U / k_out) ** 0.5)).astype(np.float32) if k_out < U else base["lm.weight"]
        rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
        start = dict(ih=rel(S0["rnn.weight_ih_l0"], exp_ih), lm=rel(S0["lm.weight"], exp_lm),
                     rest=all(np.array_equal(S0[k], base[k]) for k in base if k not in ("rnn.weight_ih_l0", "lm.weight", "mask_in", "mask_out", "k_in", "k_out")),
                     differing=[k for k in base if k not in ("rnn.weight_ih_l0", "lm.weight", "mask_in", "mask_out", "k_in", "k_out") and not np.array_equal(S0[k], base[k])])
        del S0
        # scores
        W = {k: v for k, v in T.items() if v.dtype == np.float32}
        M = Fast(W, cells.mask, F)
        dmax, same_arg = against_reader(M, W, cells.mask, F, AP["add89"][:12] + AP["sub89"][:12])
        mine = dict(A={n: teacher(M, it) for n, it in AP.items()}, R={n: teacher(M, it) for n, it in RP.items()},
                    taught={n: teacher(M, it) for n, it in TA.items()})
        theirs = dict(A=last["A_counts"], R=last["R_counts"], taught=last["taught_counts"])
        eq = {k: mine[k] == theirs[k] for k in mine}
        ans = json.loads(show(f"{P}e49/e49_{setting}_{rule}_{SEED}_answers.json"))
        ln = {}
        for k in (12, 40):
            same, stops, right = recorded_is_greedy(M, LP[k], ans[str(k)])
            ln[k] = (same, stops, right, last["length_counts"][str(k)])
        print(f"   {setting:5s} {rule}  masks equal the reader's draws {masks_ok}; digests of the draws equal the receipt's {dig_ok}; largest weight outside a mask (input, lm, recurrent) {outside};")
        print(f"            start file: input map and lm against E42's start x mask x scale, largest relative difference {start['ih']:.1e}, {start['lm']:.1e}; every other tensor (weights, biases, E42's mask, periods and phases) equal {start['rest']}" + (f" (differing: {start['differing']})" if start['differing'] else ""))
        print(f"            the sums as evaluated here against e49_interface_reader.forward on 24 items: largest difference of a logit {dmax:.1e}, same largest output at every answer position {same_arg}")
        print(f"            A add, sub, max, min {list(mine['A'].values())} / {list(theirs['A'].values())} equal {eq['A']} | R add, sub {list(mine['R'].values())} / {list(theirs['R'].values())} equal {eq['R']}"
              f" | taught {list(mine['taught'].values())} / {list(theirs['taught'].values())} equal {eq['taught']}")
        print(f"            length test: recorded answers that the reader's forward gives, of 300: {ln[12][0]} at 12 letters, {ln[40][0]} at 40; recorded answers end properly {ln[12][1] and ln[40][1]};"
              f" right by the recorded answers {ln[12][2]}, {ln[40][2]} / receipt {ln[12][3]}, {ln[40][3]}   [{time.time() - t0:.0f} s]", flush=True)
        del T, W, M


# ---------------------------------------------------------------- B: the receipts recomputed
def reach(curve, key, thr, rows=3):
    v = [(r["step"], r[key]) for r in curve if r["step"] > 0]
    for i in range(len(v) - rows + 1):
        if all(x[1] >= thr for x in v[i:i + rows]):
            return v[i][0]
    return None


def level(curve, key, back=0):
    """Mean over three scored rows: the last three (back=0) or the three ending `back` updates before the last."""
    end = curve[-1]["step"] - back
    rows = [r[key] for r in curve if end - 512 <= r["step"] <= end]
    assert len(rows) == 3
    return float(np.mean(rows))


TAB = {}
if want & set("BDE"):
    for setting, rule in [("F10", "B"), ("F10", "S")] + RUNS:
        rc = receipt(setting, rule); c = rc["curve"]
        ma = 930560 if setting == "F10" else rc["multiply_adds_per_position"]
        t = dict(ma=ma, dose=c[-1]["step"], complete=rc["complete"], curve=c, rc=rc)
        for key in "AR":
            for thr in (50, 80, 90):
                t[f"{key}{thr}"] = reach(c, key, thr); t[f"{key}{thr}s"] = reach(c, key, thr, rows=1)
        for key in ("A", "R", "len40", "len12", "taught"):
            t["lv_" + key] = level(c, key)
        t["rise_R"] = level(c, "R") - level(c, "R", back=2048); t["rise_A"] = level(c, "A") - level(c, "A", back=2048)
        TAB[(setting, rule)] = t

if "B" in want:
    f = lambda v: f"{v:,}" if v else "none"
    print("B. the receipts recomputed. Updates to the thresholds: three scored rows in a row [a single row]")
    print("   setting rule | A50            A80            A90            | R50            R80            R90            | final A, R, taught | complete, rows")
    for (s, r), t in TAB.items():
        cell = lambda k: f"{f(t[k]):>6} [{f(t[k + 's']):>6}]"
        last = t["curve"][-1]
        print(f"   {s:5s} {r}     | {cell('A50')} {cell('A80')} {cell('A90')} | {cell('R50')} {cell('R80')} {cell('R90')} | {last['A']:5.2f}, {last['R']:5.2f}, {last['taught']:5.2f} | {t['complete']}, {len(t['curve'])}")
    print("   multiply-adds to the 80 threshold (multiply-adds a position x 64 items x updates), three-row; and the ratio to F10's under the same rule")
    for (s, r), t in TAB.items():
        ref = TAB[("F10", r)]
        cell = lambda k: (f"{t['ma'] * 64 * t[k]:.3g} (x{t['ma'] * t[k] / (ref['ma'] * ref[k]):.2f})" if t[k] else "not reached")
        print(f"   {s:5s} {r}  {t['ma']:,} a position ({t['ma'] / 930560:.3f} of F10's) | A80 {cell('A80')} | R80 {cell('R80')}")
    print("   levels (mean of the last three scored rows) and the change of R and of A against the three rows ending 2,048 updates earlier")
    for (s, r), t in TAB.items():
        print(f"   {s:5s} {r}  A {t['lv_A']:5.2f}  R {t['lv_R']:5.2f}  length at 40 {t['lv_len40']:5.2f}  at 12 {t['lv_len12']:5.2f}  taught {t['lv_taught']:5.2f} | R {t['rise_R']:+.2f}, A {t['rise_A']:+.2f}")
    print("   kept in every scored row of the eight runs: masks (weights outside zero, the optimizer's averages there zero), E42's limit, finite losses; the batch digest at row 4,096")
    for (s, r), t in TAB.items():
        if s == "F10":
            continue
        c = t["curve"]
        masks = all(all(v["outside_zero"] and v["averages_zero"] for v in row["masks"].values()) for row in c)
        limit = all(all(row["limit"].values()) for row in c)
        finite = all(row.get("finite_losses", True) for row in c)
        d4096 = [row["data_digest"]["sha256"][:16] for row in c if row["step"] == 4096][0]
        loss = {row["step"]: row.get("loss") for row in c}
        ls = ", ".join(f"{u:,}: {loss[u]:.3f}" for u in (256, 1024, 2048, 4096, 8192, 16384, 24576) if loss.get(u) is not None)
        print(f"   {s:5s} {r}  masks {masks}; limit {limit}; finite {finite}; digest at 4,096 {d4096} (E26's bbd71ab7ca27b72e: {d4096 == 'bbd71ab7ca27b72e'}); rows {len(c)}, optimizer steps {t['rc']['optimizer_steps']}")
        print(f"            training loss since the previous row at update {ls}")
    print("   cosine between the sweep's and the exact gradient of each masked map on the first batch, at the start and at the end of the run; mean magnitude of the kept weights at the start and the end")
    for (s, r), t in TAB.items():
        if s == "F10":
            continue
        rc = t["rc"]; a, b = rc["cosine_first_batch_start"], rc["cosine_first_batch_end"]
        m0, m1 = t["curve"][0]["kept_mean_magnitude"], t["curve"][-1]["kept_mean_magnitude"]
        nm = {"rnn.weight_ih_l0": "input", "lm.weight": "lm", "rnn.weight_hh_l0": "recurrent"}
        print(f"   {s:5s} {r}  " + "; ".join(f"{nm[k]} {a[k]:.2f} -> {b[k]:.2f} (magnitude {m0[k]:.3f} -> {m1[k]:.3f})" for k in nm))
    print("   A and R every 2,048 updates")
    for (s, r), t in TAB.items():
        by = {row["step"]: row for row in t["curve"]}
        print(f"   {s:5s} {r}  " + " | ".join(f"{by[u]['A']:4.1f}, {by[u]['R']:4.1f}" for u in range(2048, 24577, 2048)))


# ---------------------------------------------------------------- C: the arms, counted
if "C" in want:
    print("C. the arms. Every arm keeps E42's F10 cell layer (256 cells, each reading 10 cells of its neighbourhood and itself, period 1), the byte table,")
    print("   l1 (the 512 values of a byte's code), l2 and l3 dense. Multiply-adds of one item at one position, counted from the index lists (products only):")
    words = {"F10": "each cell reads all 512 values of the byte's code and each of the 512 readout units reads all 256 cells (E42's F10, the reference)",
             "I64": "each cell reads 64 of the 512 values and each readout unit reads 32 of the 256 cells",
             "I16": "each cell reads 16 of the 512 values and each readout unit reads 8 of the 256 cells",
             "in16": "each cell reads 16 of the 512 values and each readout unit reads all 256 cells",
             "out8": "each cell reads all 512 values and each readout unit reads 8 of the 256 cells"}
    for s in ("F10",) + SETTINGS:
        k_in, k_out = R49.SETTINGS[s]
        F = R49.Interface(SEED, k_in, k_out)
        n_in = 3 * sum(len(v) for v in F.in_src); n_out = sum(len(v) for v in F.out_src); n_cells = 3 * int(R42.Cells(SEED, N=10, pset="1").mask.sum())
        fixed = 8 * D + D * D + 257 * D
        total = fixed + n_in + n_out + n_cells
        rd_in = np.bincount(np.concatenate([np.array(v) for v in F.in_src]), minlength=D); rd_out = np.bincount(np.concatenate([np.array(v) for v in F.out_src]), minlength=U)
        rec = None if s == "F10" else json.loads(show(f"{P}e49/e49_{s}_B_{SEED}.json"))["multiply_adds_per_position"]
        print(f"   {s:5s} {words[s]}:")
        print(f"         input map {n_in:,} + readout's first map {n_out:,} + cells {n_cells:,} + byte code and readout layers {fixed:,} = {total:,}"
              f" ({total / 930560:.3f} of F10's{'' if rec is None else f'; receipt {rec:,}, equal {rec == total}'});"
              f" a cell with its share of the interface {(n_in + n_out + n_cells) / U:.0f};"
              f" an input value is read by {rd_in.min()} to {rd_in.max()} cells, a cell by {rd_out.min()} to {rd_out.max()} readout units")


# ---------------------------------------------------------------- D: rising at the end
if "D" in want:
    print("D. the runs that do not reach 80 on R within the dose: R and A at the end and a straight line through the last rows")
    print("   (slope by least squares, points per 1,000 updates, with its standard error, over the last 2,048, 4,096 and 8,192 updates; the update at which the line")
    print("    over the last 8,192 reaches 80, counted from the mean of the last three rows, with the slope one standard error either way; the scatter of single rows)")

    def slope(c, key, span):
        rows = [(r["step"], r[key]) for r in c if r["step"] >= c[-1]["step"] - span]
        x = np.array([a for a, _ in rows], float); y = np.array([b for _, b in rows], float)
        p_, cov = np.polyfit(x, y, 1, cov=True)
        return float(p_[0] * 1000), float(np.sqrt(cov[0, 0]) * 1000), float((y - np.polyval(p_, x)).std(ddof=2))

    for (s, r), t in TAB.items():
        if s == "F10" or t["R80"]:
            continue
        c = t["curve"]; out = []
        for key in "RA":
            lv = t["lv_" + key]
            sl = {span: slope(c, key, span) for span in (2048, 4096, 8192)}
            a, se, sd = sl[8192]; mid = c[-1]["step"] - 256
            if a - se > 0.2:
                when = f"{mid + (80 - lv) / a * 1000:,.0f} ({mid + (80 - lv) / (a + se) * 1000:,.0f} to {mid + (80 - lv) / (a - se) * 1000:,.0f})"
            else:
                when = "no date: the line does not rise"
            out.append(f"{key}: last three rows {lv:5.2f}, against 2,048 earlier {t['rise_' + key]:+.2f}; slope " + "; ".join(f"{sl[k][0]:+.2f} +- {sl[k][1]:.2f} over {k:,}" for k in sl)
                       + f"; single rows scatter by {sd:.1f} about the line; 80 at {when}")
        print(f"   {s:5s} {r}  " + "\n            ".join(out))
        print(f"            R at the last nine rows: {[round(x['R'], 1) for x in c[-9:]]}")
    print("   the shape of the curves that do reach 80 on R: updates from 50 to 80 and the rate, and the rate after 80")
    for (s, r) in (("F10", "B"), ("F10", "S"), ("in16", "B")):
        t = TAB[(s, r)]; c = t["curve"]
        after = slope(c, "R", c[-1]["step"] - t["R80"])
        print(f"   {s:5s} {r}  R50 at {t['R50']:,}, R80 at {t['R80']:,}: {30 / (t['R80'] - t['R50']) * 1000:.1f} points per 1,000 updates between them; after 80, {after[0]:+.2f} +- {after[1]:.2f} per 1,000 (level at the end {t['lv_R']:.1f})")
    for (s, r) in (("I64", "B"), ("out8", "B")):
        t = TAB[(s, r)]; lv = t["lv_R"]; span = (t["dose"] - 256) - t["R50"]
        print(f"   {s:5s} {r}  R50 at {t['R50']:,}; from there to the middle of the last three rows ({t['dose'] - 256:,}) R rose from about 50 to {lv:.1f}: {(lv - 50) / span * 1000:.1f} points per 1,000 updates")
    print("   the reading in multiply-adds that a continuation could give: the latest update at which 80 on R would still be *stays* against F10 (1.5 times F10's multiply-adds)")
    for (s, r) in (("I64", "B"), ("I16", "B"), ("out8", "B"), ("I64", "S"), ("I16", "S"), ("in16", "S"), ("out8", "S")):
        t = TAB[(s, r)]; ref = TAB[("F10", r)]
        print(f"   {s:5s} {r}  {1.5 * ref['R80'] * ref['ma'] / t['ma']:,.0f} updates" + (" (already passed)" if 1.5 * ref['R80'] * ref['ma'] / t['ma'] < t['dose'] - 512 else ""))


# ---------------------------------------------------------------- E: descriptions, seeds, predictions
def describe(x_ref, x_new, d_new):
    """E41's four cases read as fewer / more / about the same, candidate against reference; d_new is the last recordable update."""
    if x_ref and x_new:
        q = x_new / x_ref
        return ("fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same") + f" (x{q:.2f})"
    if x_ref and not x_new:
        return f"more (not reached; {d_new:,} recordable updates are x{d_new / x_ref:.2f})" if d_new / x_ref >= 1.5 else "not decided"
    return "neither reaches"


if "E" in want:
    print("E. against E42's F10 of seed 73129 under the same rule, in the brief's forms")
    for (s, r), t in TAB.items():
        if s == "F10":
            continue
        ref = TAB[("F10", r)]; D_ = t["dose"] - 512
        lv = []
        for key, nm in (("A", "A"), ("R", "R"), ("len40", "length at 40")):
            d = t["lv_" + key] - ref["lv_" + key]
            lv.append(f"{nm} {'lower' if d <= -15 else 'higher' if d >= 15 else 'about the same'} ({d:+.1f})")
        ratio = t["ma"] / ref["ma"]
        def in_ma(k):
            if t[k]:
                q = t[k] * ratio / ref[k]
                return ("falls" if q < 2 / 3 else "rises" if q > 1.5 else "stays") + f" (x{q:.2f})"
            return "rises" if D_ * ratio / ref[k] >= 1.5 else f"not decided (x{D_ * ratio / ref[k]:.2f} at the last recordable update)"
        print(f"   {s:5s} {r}  updates to 80: A {describe(ref['A80'], t['A80'], D_)}, R {describe(ref['R80'], t['R80'], D_)} | multiply-adds: A {in_ma('A80')}, R {in_ma('R80')} | levels: {'; '.join(lv)}")
    print("   what a seed alone changes: E42's F10 on three seeds (E42's branch at 44f91edc)")
    spread = {}
    for r in "SB":
        rows = []
        for seed in (73129, 93683, 116089):
            c = json.loads(show(f"{P}e42/e42_F10_{r}_{seed}.json", E42REF))["curve"]
            rows.append(dict(seed=seed, A80=reach(c, "A", 80), R80=reach(c, "R", 80), A=level(c, "A"), R=level(c, "R"), len40=level(c, "len40")))
        for q in rows:
            print(f"      F10 {r} {q['seed']}: A80 {q['A80']:,}, R80 {q['R80']:,}; levels A {q['A']:.1f}, R {q['R']:.1f}, length at 40 {q['len40']:.1f}")
        spread[r] = {k: max(q[k] for q in rows) - min(q[k] for q in rows) for k in ("A", "R", "len40")}
        spread[r]["R80"] = max(q["R80"] for q in rows) / min(q["R80"] for q in rows); spread[r]["A80"] = max(q["A80"] for q in rows) / min(q["A80"] for q in rows)
        print(f"      F10 {r}: largest ratio between seeds, A80 {spread[r]['A80']:.2f}, R80 {spread[r]['R80']:.2f}; largest difference of levels, A {spread[r]['A']:.1f}, R {spread[r]['R']:.1f}, length at 40 {spread[r]['len40']:.1f}")
    print("   differences of each run from F10 of seed 73129 that are not beyond the spread between F10's seeds:")
    none = True
    for (s, r), t in TAB.items():
        if s == "F10":
            continue
        ref = TAB[("F10", r)]
        for key in ("A", "R", "len40"):
            if abs(t["lv_" + key] - ref["lv_" + key]) <= spread[r][key]:
                print(f"      {s} {r} level {key}: {t['lv_' + key] - ref['lv_' + key]:+.1f} against a spread of {spread[r][key]:.1f}"); none = False
        for key in ("A80", "R80"):
            if t[key] and t[key] / ref[key] <= spread[r][key]:
                print(f"      {s} {r} {key}: x{t[key] / ref[key]:.2f} against a ratio between seeds of {spread[r][key]:.2f}"); none = False
    if none:
        print("      none")
    print("   the predictions (the coordinator's), scored from the receipts")
    T = TAB; yes = lambda b: "right" if b else "wrong"
    st2 = {r: (not T[("I16", r)]["R80"]) for r in "SB"}
    p8 = None
    if st2["B"]:
        a, b = T[("in16", "B")], T[("out8", "B")]
        if a["R80"] and b["R80"]:
            p8 = a["R80"] > b["R80"]
        elif a["R80"] or b["R80"]:
            p8 = bool(b["R80"])
        else:
            p8 = a["lv_R"] < b["lv_R"]
    rows = [("P49.1  checks 1 and 2 pass (90%)", None),
            ("P49.2  I64, backpropagation, reaches 80 on R within the dose (75%)", bool(T[("I64", "B")]["R80"])),
            ("P49.3  I64, the sweep, reaches 80 on R (55%)", bool(T[("I64", "S")]["R80"])),
            ("P49.4  I16, backpropagation, reaches 80 on R (50%)", bool(T[("I16", "B")]["R80"])),
            ("P49.5  I16, the sweep, reaches 80 on R (30%)", bool(T[("I16", "S")]["R80"])),
            ("P49.6  I64 under backpropagation about the same as F10 in updates to 80 on R (50%)", describe(T[("F10", "B")]["R80"], T[("I64", "B")]["R80"], T[("I64", "B")]["dose"] - 512).startswith("about")),
            ("P49.7  stage 2 runs under at least one rule (65%)", st2["S"] or st2["B"]),
            ("P49.8  in16 the worse of the two under backpropagation (60%)", p8)]
    for text, v in rows:
        print(f"      {text}: {'see part F' if v is None else yes(v)}")


# ---------------------------------------------------------------- F: the check receipts
if "F" in want:
    print("F. the stage-0 check receipts (e49_qual), as recorded by the worker; the reader's own confirmation is part A")
    q = P + "e49_qual/"
    c1 = json.loads(show(q + "check1_73129.json"))
    print(f"   check 1: pass {c1['pass']}; the runner's copy of the specification differs in the import line only {c1['copy_differs_in_import_line_only']};"
          f" {c1['lines_compared']} lines equal to the Mac output {c1['lines_equal']}; lines of self-check 3 present: {sum(l.startswith('3.') for l in c1['stdout'])} of 4")
    c2 = json.loads(show(q + "check2_73129.json"))
    for r, v in c2["runs"].items():
        print(f"   check 2, {RULE[r]}: tensors equal to the reference's {v['tensors_equal']} ({v['tensors_sha256'][:8]}), scores equal at every row {v['scores_equal_every_row']} ({v['rows']} rows), same data {v['data_digest_equal']};"
              f" reference {v['reference'][:100]}")
    c4 = json.loads(show(q + "check4_73129.json")); c5 = json.loads(show(q + "check5_73129.json"))
    print(f"   check 4: pass {c4['pass']} (digest at 4,096 {c4['digest_4096'][:16]}); check 5: pass {c5['pass']}, largest difference of a count on E26's six births {c5['max_abs_diff']}")
    for name in ("check6_73129_I64_I16.json", "check6_73129_in16_out8.json"):
        c6 = json.loads(show(q + name))
        print(f"   check 6 ({name}): pass {c6['pass']}; " + "; ".join(f"{k}: masks at every row {v['masks_every_row']}, forecast {v['forecast_cpu_h']:.2f} CPU-h" for k, v in c6["runs"].items()))
    cpu = sum((TAB[k]["rc"]["cpu_updates"] + TAB[k]["rc"]["cpu_scoring"]) for k in TAB if k[0] != "F10") / 3600 if TAB else None
    if cpu:
        print(f"   CPU of the eight runs, updates and scoring: {cpu:.2f} hours")


# ---------------------------------------------------------------- G: what the answers are
if "G" in want:
    from collections import Counter
    LEN_TASK = ("add", 89, b"p")
    LPG = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    print("G. the answers on the length-test panel (held-out pairs of add modulo 89; 300 items at 12 letters and 300 at 40), from the answers files")
    print("   setting rule letters | two digits and the end symbol | different answers given (right answers are " + ", ".join(f"{len({tuple(a) for _, a in LPG[k]})} different at {k}" for k in (12, 40)) + ")"
          " | the five most frequent answers, share | first digit right, second digit right, whole answer right")
    for setting, rule in RUNS:
        ans = json.loads(show(f"{P}e49/e49_{setting}_{rule}_{SEED}_answers.json"))
        for k in (12, 40):
            given = [tuple(a) for a in ans[str(k)]]; target = [tuple(a) for _, a in LPG[k]]
            form = sum(len(g) == 3 and all(48 <= v <= 57 for v in g[:2]) and g[2] == EOS for g in given)
            cnt = Counter(given); top = cnt.most_common(5)
            d1 = sum(len(g) > 0 and g[0] == t[0] for g, t in zip(given, target)); d2 = sum(len(g) > 1 and g[1] == t[1] for g, t in zip(given, target))
            whole = sum(g == t for g, t in zip(given, target))
            txt = lambda g: "".join(chr(v) if v < 128 else "#" for v in g[:-1]) if g and g[-1] == EOS else "".join(chr(v) if v < 128 else "#" for v in g) + "..."
            print(f"   {setting:5s} {rule}    {k:2d}      | {form:3d} of 300 | {len(cnt):3d} | {', '.join(txt(g) for g, _ in top)}: {sum(n for _, n in top) / 3:.0f}% | {d1 / 3:5.1f}%, {d2 / 3:5.1f}%, {whole / 3:5.1f}%")


# ---------------------------------------------------------------- H: the maps and the byte's code at the start and the end
if "H" in want:
    USED = sorted(set(b"0123456789+-<>=pqvwABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    DIG = list(range(48, 58)); iu = np.triu_indices(10, 1)
    LEN_TASK = ("add", 89, b"p")
    LPH = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12,), n=300)[12]
    xH = np.array([[BOS] + p for p, _ in LPH], dtype=np.int64)                      # the prompts end at the `=` sign
    cellsH = R42.Cells(SEED, N=10, pset="1")

    def code(W):
        z = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"])
        d = np.linalg.norm(z[DIG][:, None] - z[DIG][None], axis=-1)[iu]
        return float(np.abs(z[USED]).mean()), float(d.mean()), float(d.min())

    def spread(W, F):
        h = Fast(W, cellsH.mask, F).states(xH, xH.shape[1] - 1)
        return float(h.std(0).mean()), float(np.abs(h).mean())

    nrm = lambda W, k: float(np.linalg.norm(W[k]))
    MAPS = (("l1.weight", "l1"), ("rnn.weight_ih_l0", "input map"), ("rnn.weight_hh_l0", "recurrent"), ("lm.weight", "lm"), ("l2.weight", "l2"), ("l3.weight", "l3"))
    print("H. the maps and the byte's code, start and end (seed 73129). Size of a map: the norm of its kept weights at the end over the norm at the start.")
    print("   Code: mean magnitude of the 512 values over the 46 bytes the task uses; distance between the codes of two digits, mean and least of the 45 pairs.")
    print("   States: over the 300 items of the length-test panel at 12 letters, the standard deviation between items of a cell's state at the `=` sign, mean over the 256 cells.")
    base = load(f"{P}e49_qual/check2_73129_runs/e49_F10_S_dry2048_{SEED}_start.pt")
    c0 = code(base); F_full = R49.Interface(SEED, D, U)
    print(f"   E42's start (every setting's start has this code): code {c0[0]:.2f}, digits {c0[1]:.2f} / {c0[2]:.2f}; states differ by {spread({k: v for k, v in base.items() if v.dtype == np.float32}, F_full)[0]:.3f}")
    rows = [("F10", r, None) for r in "BS"] + [(s_, r, R49.SETTINGS[s_]) for s_, r in RUNS]
    for setting, rule, ks in rows:
        if ks is None:
            T = load(f"{P}e42/e42_F10_{rule}_{SEED}.pt", E42REF); S0 = base; F = F_full
        else:
            T = load(f"{P}e49/e49_{setting}_{rule}_{SEED}.pt"); S0 = load(f"{P}e49/e49_{setting}_{rule}_{SEED}_start.pt"); F = R49.Interface(SEED, *ks)
        W = {k: v for k, v in T.items() if v.dtype == np.float32}
        c1 = code(W); sp = spread(W, F)
        sizes = "; ".join(f"{nm} {nrm(W, k) / nrm(S0, k):.2f}" for k, nm in MAPS)
        drive = (nrm(W, "rnn.weight_ih_l0") / nrm(S0, "rnn.weight_ih_l0")) * (c1[0] / c0[0])
        print(f"   {setting:5s} {rule}  sizes: {sizes} | code {c1[0]:.2f}, digits {c1[1]:.2f} / {c1[2]:.2f} | input map's size x code's magnitude, against the start: {drive:.2f} | states differ by {sp[0]:.3f} (mean magnitude {sp[1]:.2f})", flush=True)
        del T, W

    # what the values a cell reads say about the byte (the audit's recommendation g: its section 4 repeated on each run's files)
    USED46 = [BOS] + list(b"pqvw") + list(b"0123456789") + list(b"+-><") + list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [ord("=")]
    NONLET = [i for i, b in enumerate(USED46) if not 65 <= b <= 90]; DIGS = list(range(5, 15))

    def among(d, rows):
        i2 = np.triu_indices(len(rows), 1)
        return d[..., rows, :][..., :, rows][..., i2[0], i2[1]]

    def on_cells(W, F):
        Z = np.tanh(W["book.weight"][USED46] @ W["l1.weight"].T + W["l1.bias"]).astype(np.float64)        # (46, 512)
        X = Z[:, F.in_idx].transpose(1, 0, 2)                                                              # (256, 46, K_IN)
        d = np.sqrt(((X[:, :, None] - X[:, None]) ** 2).sum(-1))                                           # (256, 46, 46)
        nn, dg, al = among(d, NONLET), among(d, DIGS), among(d, list(range(len(USED46))))
        return (nn.min(1) / nn.mean(1)), (dg.min(1) / dg.mean(1)), nn.mean(1), bool((al > 1e-6).all())

    print("   what the values a cell reads say about the byte, on each run's own lists: the closest two of the 20 bytes the task must tell apart, over the mean distance")
    print("   between two of them (least over the 256 cells, median); the same for the ten digits; the mean distance itself (median over cells); no two of the 46 bytes coincide on any cell")
    for setting, rule, ks in [("start", "", None)] + rows:
        if setting == "start":
            for s_ in ("I64", "I16"):
                a, b, m, dis = on_cells(base, R49.Interface(SEED, *R49.SETTINGS[s_]))
                print(f"   start, on {s_}'s lists: the 20 bytes {a.min():.3f}, {np.median(a):.3f}; digits {b.min():.3f}, {np.median(b):.3f}; mean distance {np.median(m):.2f}; distinct {dis}")
            continue
        if ks is None:
            T = load(f"{P}e42/e42_F10_{rule}_{SEED}.pt", E42REF); F = F_full
        else:
            T = load(f"{P}e49/e49_{setting}_{rule}_{SEED}.pt"); F = R49.Interface(SEED, *ks)
        a, b, m, dis = on_cells(T, F)
        print(f"   {setting:5s} {rule}  the 20 bytes {a.min():.3f}, {np.median(a):.3f}; digits {b.min():.3f}, {np.median(b):.3f}; mean distance {np.median(m):.2f}; distinct {dis}")
        del T
