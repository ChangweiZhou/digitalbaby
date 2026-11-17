"""Check of the brief of E57: the figures behind CHECK_OF_TASK_E57_BRIEF_20261117.md.  Read-only; nothing is trained.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I -B <this file> <part> [arguments]      (from the repository root)

    receipts            updates to 80, levels and costs of E42's and E49's runs of seed 73129; E42's three seeds; reference hashes
    figures <file>      the brief's cosines against <file>, the output of sweep_scale_variants.py (run beforehand, whole)
    weights             the tensors of E49's weight file of in16 under the sweep; its masks against the draws; rows and columns
                        without a kept entry; the brief's function `magnitudes` (transcribed to numpy) against the measurement's scale()
    where [networks]    the row scale applied at one map at a time (default: in16_S in16_B F10_S G_S)

Files of other branches are read with `git show`; no line of another branch is executed.  The parts `figures` and `where` use
sweep_scale_variants.py of the reviewer's folder on the current branch (`where` imports it; nothing is written beside it).
Standard library and numpy only.
"""
import sys
sys.dont_write_bytecode = True
import json, os, subprocess
from random import Random
import numpy as np

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
RV = os.path.join(REPO, "PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109")
M = "MOD97_LOCAL_RULE_20261106"
W, W9 = "origin/claude/gallant-dijkstra-i9zgsr", "origin/claude/peaceful-noether-vcvrmf"


def show(ref, path, text=True):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True, cwd=REPO, text=text).stdout


def reach(curve, key, thr=80.0):
    """The update of the first of three consecutive scored rows at or above thr."""
    v = [(r["step"], r[key]) for r in curve]
    for i in range(len(v) - 2):
        if all(x[1] >= thr for x in v[i:i + 3]):
            return v[i][0]
    return None


def level(curve, key):
    return sum(r[key] for r in curve[-3:]) / 3


# ---------------------------------------------------------------- receipts
def part_receipts(args):
    for ref, path in ((W, "e42/e42_G_S_73129"), (W, "e42/e42_G_B_73129"), (W, "e42/e42_F10_S_73129"), (W, "e42/e42_F10_B_73129"),
                      (W9, "e49/e49_in16_B_73129"), (W9, "e49/e49_in16_S_73129")):
        d = json.loads(show(ref, f"{M}/exp45/{path}.json")); c = d["curve"]
        cpu = (d["cpu_updates"] + d["cpu_scoring"] + d.get("cpu_replacement", 0.0)) / 3600
        print(f"{path}: rows {len(c)}; updates to 80: A {reach(c, 'A')} R {reach(c, 'R')}; taught to 50: {reach(c, 'taught', 50)};"
              f" levels A {level(c, 'A'):.2f} R {level(c, 'R'):.2f} taught {level(c, 'taught'):.2f}; last row A {c[-1]['A']:.2f} R {c[-1]['R']:.2f}"
              f" taught {c[-1]['taught']:.2f}; CPU-hours {cpu:.3f}; torch {d['provenance']['host']['torch']}; tensors_sha256 {d['tensors_sha256']}")
        if "e49" in path:
            print(f"   start_tensors_sha256 {d['start_tensors_sha256']}; multiply-adds {d['counts']['multiply_adds']}; runner {d['provenance']['runner_sha256']}")
            print("   loss at every 2,048 updates:", " ".join(f"{r['step']}:{r['loss']:.3f}" for r in c if r["step"] % 2048 == 0 and r["step"]))
            print("   taught at every 2,048 updates:", " ".join(f"{r['step']}:{r['taught']:.1f}" for r in c if r["step"] % 2048 == 0))
    print("E42, the three seeds (updates to 80 on A and on R):")
    for cfg in ("G", "F10"):
        for rule in "SB":
            out = []
            for seed in (73129, 93683, 116089):
                c = json.loads(show(W, f"{M}/exp45/e42/e42_{cfg}_{rule}_{seed}.json"))["curve"]
                out.append(f"{seed}: {reach(c, 'A')}, {reach(c, 'R')}")
            print(f"   {cfg} {rule}  " + "; ".join(out))
    for ref, path in ((W, "e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129"), (W, "e42_qual/check1_73129_runs/e42_G_S_dry2048_73129"),
                      (W9, "e49_qual/check2_73129_runs/e49_F10_S_dry2048_73129"), (W9, "e49_qual/check6_73129_runs/e49_in16_S_dry256_73129"),
                      (W9, "e49_qual/check6_73129_runs/e49_in16_B_dry256_73129")):
        d = json.loads(show(ref, f"{M}/exp45/{path}.json"))
        print(f"{path}: complete {d['complete']}; tensors_sha256 {d['tensors_sha256']}; start {d.get('start_tensors_sha256', 'not recorded')};"
              f" CPU-seconds an update {d['cpu_updates'] / d['dose']:.4f}")
    src = show("origin/codex/minifly-consolidated-20261023", f"{M}/exp42_cells.py")
    print("DRY2048 in exp42_cells.py holds 79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f:",
          '"S": "79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f"' in src)


# ---------------------------------------------------------------- figures
BRIEF = {  # network of the measurement: tensor: (matrix, row, rowcol) as the brief's table and its check 2 give them
    "in16_S": {"byte table": (0.367, 0.697, 0.638), "l1": (0.298, 0.624, 0.721), "input map": (0.603, 0.676, 0.714),
               "recurrent map": (0.651, 0.762, 0.710), "lm": (0.606, 0.694, 0.606), "l2": (0.643, 0.710, 0.643)},
    "F10_S": {"l1": (0.537, 0.631, 0.693), "input map": (0.647, 0.696, 0.710), "recurrent map": (0.685, 0.730, 0.738)},
    "G_S": {"input map": (0.788, 0.791, 0.800), "recurrent map": (0.819, 0.821, 0.830)},
    "in16_start": {"input map": (0.745, 0.749, 0.745), "recurrent map": (0.734, 0.740, 0.734)},
}


def part_figures(args):
    lines = open(args[0]).read().splitlines()

    def rows(net):
        out = {}
        for where, head in (("all", f"{net}, cosine with the exact gradient, variant at every map:"),
                            ("gru", f"{net}, cosine with the exact gradient, variant at the two GRU maps only")):
            j = next(k for k, ln in enumerate(lines) if ln.startswith(head)) + 1
            while not lines[j].startswith("| variant"):
                j += 1
            cols = [c.strip() for c in lines[j].strip("|").split("|")][1:]
            j += 2
            while lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip("|").split("|")]
                out[(cells[0], where)] = {c: float(v.split()[0]) for c, v in zip(cols, cells[1:]) if v != "–"}
                j += 1
        return out
    bad = 0
    for net, d in BRIEF.items():
        r = rows(net)
        for t, want in d.items():
            got = (r[("V0", "all")][t], r[("V2", "all")][t], r[("V4", "gru")][t])
            ok = [abs(a - b) < 5e-4 for a, b in zip(got, want)]
            bad += ok.count(False)
            print(f"{net:11s} {t:14s} brief {want[0]:.3f} {want[1]:.3f} {want[2]:.3f} | script V0 {got[0]:.3f}, V2 at every map {got[1]:.3f}, V4 at the GRU maps {got[2]:.3f} | {ok}")
    print("figures of the brief that differ from the script's output:", bad)
    ts = ("byte table", "l1", "input map", "recurrent map", "lm", "l2")
    print("\nmatrix (V0) | row (V2 at every map) | row at the two GRU maps (V2) | rowcol (V4 at the two GRU maps)")
    for net in ("in16_S", "in16_B", "F10_S", "G_S", "in16_start"):
        r = rows(net)
        for t in ts:
            print(f"{net:11s} {t:14s} " + " | ".join(f"{r[k][t]:.3f}" for k in (("V0", "all"), ("V2", "all"), ("V2", "gru"), ("V4", "gru"))))
    doc = set(ln.rstrip() for ln in open(os.path.join(RV, "SWEEP_SCALE_VARIANTS_ALIGNMENT_20261117.md")).read().splitlines())
    i0 = next(k for k, ln in enumerate(lines) if ln.startswith("===== tables"))
    i1 = next(k for k, ln in enumerate(lines) if ln.startswith("===== summary"))
    tab, keep = [], True
    for ln in lines[i0:i1]:
        if "cosine for the biases" in ln:
            keep = False                                    # the bias tables are not repeated in the document
        elif ln.startswith("**"):
            keep = True
        if keep and ln.startswith("| V"):
            tab.append(ln.rstrip())
    print(f"\nrows of the weight tables printed by this run: {len(tab)}; not found word for word in the document: {sum(ln not in doc for ln in tab)}")


# ---------------------------------------------------------------- weights
def load(ref, path):
    sys.path.insert(0, RV)
    from compare_pt_tensors import tensors
    dt = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
    raw = show(ref, path, text=False)
    return {n.lstrip("/"): np.frombuffer(b, dtype=dt[st]).reshape(sh).copy() for n, (st, sh, b) in tensors(raw).items()}


def brief_magnitudes(Wt, maskf, scale):
    """The brief's function `magnitudes`, line for line, in numpy.  A name it does not know falls through to the last line."""
    A = np.abs(Wt) * maskf
    mean = A.sum() / maskf.sum()
    if scale == "matrix":
        return mean
    row = A.sum(1, keepdims=True) / maskf.sum(1, keepdims=True)
    if scale == "row":
        return row
    col = A.sum(0, keepdims=True) / np.maximum(maskf.sum(0, keepdims=True), 1)
    return row * col / mean


def measurement_scale(Wt, Mk, variant):
    """sweep_scale_variants.scale for V0, V2, V4 and V5 (lines 285 to 305 of that script)."""
    A = np.abs(Wt) * Mk
    if variant == "V5":
        return A
    mean = A.sum() / Mk.sum()
    row = A.sum(1) / np.maximum(Mk.sum(1), 1)
    col = A.sum(0) / np.maximum(Mk.sum(0), 1)
    one = np.ones(Wt.shape, dtype=Wt.dtype)
    return {"V0": one * mean, "V2": one * row[:, None], "V4": row[:, None] * col[None, :] / mean}[variant]


def part_weights(args):
    T = load(W9, f"{M}/exp45/e49/e49_in16_S_73129.pt")
    print("tensors of e49_in16_S_73129.pt: " + ", ".join(f"{k} {v.shape}" for k, v in T.items()))
    U, D, SIDE, RADIUS, seed = 256, 512, 16, 3, 73129
    ri = Random(f"e49|input|{seed}|16")                                    # e49_interface.draws, lines 26 and 27
    mask_in = np.zeros((U, D), dtype=bool)
    for i in range(U):
        mask_in[i, sorted(ri.sample(range(D), 16))] = True

    def neighbourhood(i):                                                  # e42_cells.neighbourhood, lines 31 and 32
        r, c = divmod(i, SIDE)
        return sorted({((r + dr) % SIDE) * SIDE + (c + dc) % SIDE for dr in range(-RADIUS, RADIUS + 1) for dc in range(-RADIUS, RADIUS + 1)} - {i})
    rc = Random(f"e42|connections|{seed}")                                 # e42_cells.draws, lines 44 and 45, N = 10
    mask = np.eye(U, dtype=bool)
    for i in range(U):
        mask[i, sorted(rc.sample(sorted(neighbourhood(i)), 10))] = True
    print("mask_in of the file equals the draw:", np.array_equal(mask_in, T["mask_in"].astype(bool)), "| recurrent mask equals the draw:",
          np.array_equal(mask, T["mask"].astype(bool)), "| mask_out all ones:", bool(T["mask_out"].all()))
    Mk = {"ih": np.tile(mask_in, (3, 1)), "hh": np.tile(mask, (3, 1))}
    for k, m in Mk.items():
        print(f"{k}: kept {int(m.sum())}; rows without a kept entry {int((m.sum(1) == 0).sum())}; columns without {int((m.sum(0) == 0).sum())};"
              f" kept in a row {sorted(set(m.sum(1).tolist()))}; kept in a column {int(m.sum(0).min())} to {int(m.sum(0).max())}")
    names = {"l1": "l1.weight", "ih": "rnn.weight_ih_l0", "hh": "rnn.weight_hh_l0", "l2": "l2.weight", "lm": "lm.weight", "l3": "l3.weight"}
    for k, n in names.items():
        Wt = T[n].astype(np.float64); m = Mk.get(k, np.ones(Wt.shape, dtype=bool)).astype(np.float64)
        d = [float(np.abs(np.broadcast_to(brief_magnitudes(Wt, m, s), Wt.shape) - measurement_scale(Wt, m, v)).max())
             for s, v in (("matrix", "V0"), ("row", "V2"), ("rowcol", "V4"), ("exact", "V5"))]
        rowm = (np.abs(Wt) * m).sum(1) / m.sum(1)
        print(f"{k:3s} largest difference, the brief's code against the measurement: matrix {d[0]:.1e}, row {d[1]:.1e}, rowcol {d[2]:.1e}; given the name"
              f" 'exact' against V5 {d[3]:.2f}; row means {rowm.min():.4f} to {rowm.max():.4f}, matrix mean {(np.abs(Wt) * m).sum() / m.sum():.4f}")
    A3 = np.abs(T["l3.weight"].astype(np.float64)); rm = A3.mean(1); tg = list(range(48, 58)) + [256]
    rest = [i for i in range(257) if i not in tg]
    print(f"l3, mean magnitude of a row: the 11 rows of the classes that occur as targets {rm[tg].min():.4f} to {rm[tg].max():.4f} (mean {rm[tg].mean():.4f});"
          f" the other 246 rows, mean {rm[rest].mean():.4f}; the matrix {A3.mean():.4f}")


# ---------------------------------------------------------------- where
PLACES = [("matrix at every map", ()), ("row at l3 only", ("l3",)), ("row at lm only", ("lm",)), ("row at l2 only", ("l2",)), ("row at l1 only", ("l1",)),
          ("row at the input map only", ("ih",)), ("row at the recurrent map only", ("hh",)), ("row at the two maps of the recurrent layer", ("ih", "hh")),
          ("row at l1, l2, lm and l3", ("l1", "l2", "lm", "l3")), ("row at every map but l3", ("l1", "ih", "hh", "l2", "lm")),
          ("row at every map", ("l1", "ih", "hh", "l2", "lm", "l3")), ("rowcol at the two maps of the recurrent layer", None)]


def part_where(args):
    sys.path.insert(0, RV)
    import sweep_scale_variants as V
    ts = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight")

    def fb(net, maps):
        F = {}
        for k in net.maps:
            v = ("V4" if k in V.GRU_MAPS else "V0") if maps is None else ("V2" if k in maps else "V0")
            F[k] = np.sign(net.Wm(k)) * V.scale(net.P[V.WNAME[k]], net.M[k], v, k in V.GRU_MAPS)
        return F
    for name in args or ["in16_S", "in16_B", "F10_S", "G_S"]:
        net = V.Net(name)
        Fx = V.exact_feedback(net); Fs = [(lab, fb(net, maps)) for lab, maps in PLACES]
        acc = {lab: {t: [] for t in ts} for lab, _ in PLACES}
        for x, y, mask in V.stream_batches(1 + V.N_FURTHER)[1:]:
            c = V.forward(net, x, y, mask)
            ge = V.backward(net, c, Fx)
            for lab, F in Fs:
                g = V.backward(net, c, F)
                for t in ts:
                    acc[lab][t].append(V.cos(g[t], ge[t]))
        print(f"\n{name}: mean cosine with the exact gradient over batches 2 to 17 of the stream")
        print("| scale | " + " | ".join(V.SHORT[t] for t in ts) + " |")
        print("|---|" + "---|" * len(ts))
        for lab, _ in PLACES:
            print(f"| {lab} | " + " | ".join(f"{np.mean(acc[lab][t]):.3f}" for t in ts) + " |")


if __name__ == "__main__":
    {"receipts": part_receipts, "figures": part_figures, "weights": part_weights, "where": part_where}[sys.argv[1]](sys.argv[2:])
