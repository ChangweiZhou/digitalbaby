"""Verification of E47's trial (stage 0: four runs at 1,024 cells, seed 73129; worker's branch claude/e47-worker-stage-0-3d0ya6 at 578a877b).

Read-only.  Nothing is trained, torch is not used, and no code of the worker's or of the default branch is run: files are read with
`git show`, one at a time; the forward is e47_local_reader.py (written from the brief's text for the audit of 16 November); the
panels come from np_gru.py.  The E42 figures of the same seed are read from E42's receipts at 7fbdbdc2, the commit verified in
E42_SEED_73129_VERIFICATION_20261116.md.

  A  the files on the worker's branch; the eight check receipts; the specification file's copy; what each trial receipt records
     about the optimizer, the data, the runner and the start
  B  the structure saved in each weight file against the reader's draws (sources, periods, phases, interface, far cells)
  C  the reader against the worker's recorded answers: the length-test panel (300 items at 12 letters, 300 at 40), plain, with
     the far cells held at zero and with the far cells held at their mean, byte by byte
  D  the reader on the final weights on the panels behind the thresholds: A (300 held-out pairs of each operation at 12 letters)
     and R (1,000 held-out reduction pairs of add and of sub, no letters), plain and under both diagnostics, beside the
     receipt's last row
  E  the receipts recomputed: thresholds by the three-consecutive-rows rule, final scores, the mean at 40 letters over the last
     three rows, multiply-adds a position counted here from the wiring that ran, computation to threshold beside E42's G, F10
     and P8 of the same seed, and the brief's fixed descriptions
  F  measurements of my own on the saved weights: the fall under each diagnostic by the number of letters (0, 4, 8, 12, and
     40); the far cells held at their mean ring by ring (by the number of connections from the interface); the far cells given
     the states of another item; how much a far cell's state varies with the item; the share of the interface cells'
     connections that come from far cells
  G  the nine predictions scored from the receipts
  H  the measured cost of the trial and the roster's forecast

  I  on the starting weights: how much a cell's state varies with the item, by ring
  V  the arrangement of the reader's forward used for decoding (one position at a time) against the reader's forward itself

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e47_trial_verify.py [A B C D E F G H I V] [one_S one_B place_S place_B]
(the run names restrict part F to those runs)
"""
import copy, hashlib, json, os, subprocess, sys, time
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import tensors
from np_gru import split, panel, render, DEV, BOS, EOS
import e47_local_reader as R

REF = "578a877b"                         # the worker's branch at the trial's report
E42REF = "7fbdbdc2"                      # E42's first seed, as verified on 16 November
DREF = "origin/codex/minifly-consolidated-20261023"
M = "MOD97_LOCAL_RULE_20261106/"
P = M + "exp45/"
SEED, SIDE = 73129, 32
RUNS = [("one", "S"), ("one", "B"), ("place", "S"), ("place", "B")]
DTYPE = {"FloatStorage": "<f4", "LongStorage": "<i8", "BoolStorage": "?"}
want = {a for a in sys.argv[1:] if len(a) == 1} or set("ABCDEFGHIV")
FARMS = {tuple(a.split("_")) for a in sys.argv[1:] if "_" in a}      # part F on some runs only, e.g. one_S place_B
T0 = time.time()


def show(path, ref=REF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def load(path, ref=REF):
    out = {}
    for name, (stype, shape, raw) in tensors(show(path, ref)).items():
        out[name.lstrip("/")] = np.frombuffer(raw, dtype=DTYPE[stype]).reshape(shape).copy()
    return out


def receipt(arm, rule):
    return json.loads(show(f"{P}e47/e47_{SIDE}_{arm}_{rule}_{SEED}.json"))


def weights(arm, rule, start=False):
    return load(f"{P}e47/e47_{SIDE}_{arm}_{rule}_{SEED}{'_start' if start else ''}.pt")


def grid(arm):
    return R.Grid(SEED, SIDE, arm)


# ---------------------------------------------------------------- the reader's forward, arranged for decoding
class Net:
    """e47_local_reader.forward arranged for decoding: one position at a time (so that an answer byte adds one position and the
    prompt is not computed again), the parts that depend on the byte alone taken from tables over the 258 symbols, and, for a
    cell of period above 1, the step weights applied to the sources' states before the weights (the same sum, a third of the
    products).  Part V compares its logits and its answers with the reader's own forward.

    The two diagnostics act on the cells `held` (default: every far cell): `far_off` holds them at zero after every position,
    `hold` at given values; the other cells run freely."""

    def __init__(self, W, S):
        self.S = S
        z1 = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"])                      # (258, 512)
        self.gi = z1 @ W["rnn.weight_ih_l0"].T + W["rnn.bias_ih_l0"]                           # (258, 768): gate g, interface cell j at g * 256 + j
        self.l2 = z1 @ W["l2.weight"].T + W["l2.bias"]                                         # (258, 512)
        self.lmT, self.lmb, self.l3T, self.l3b = W["lm.weight"].T.copy(), W["lm.bias"], W["l3.weight"].T.copy(), W["l3.bias"]
        self.WcT = np.ascontiguousarray(W["Wc"].transpose(1, 2, 0))                            # (cells, 11, 3)
        self.bcT = np.ascontiguousarray(W["bc"].T)                                             # (cells, 3)
        self.bfarT = np.ascontiguousarray(W["bfar"].T)                                         # (far cells, 3)
        self.a = W.get("a")
        self.held = S.farx
        self.plans = {}

    def plan(self, t):
        if t not in self.plans:
            S, out = self.S, []
            for q in sorted(set(S.p.tolist())):
                cells = np.nonzero((S.p == q) & (t >= S.f) & ((t - S.f) % q == 0))[0]
                if len(cells):
                    at_if = S.is_if[cells]
                    out.append((q, cells, S.idx[cells], np.nonzero(at_if)[0], S.if_pos[cells[at_if]], np.nonzero(~at_if)[0], S.far_pos[cells[~at_if]]))
            self.plans[t] = out
        return self.plans[t]

    def start(self, B):
        self.h = np.zeros((B, self.S.C), np.float32); self.hist = [self.h]; self.t = 0          # hist[s] = the states after position s - 1

    def step(self, xb, far_off=False, hold=None):
        t, h = self.t, self.h
        new = h.copy()
        gi = self.gi[xb].reshape(len(xb), 3, 256)
        for q, cells, idx, m_if, ifpos, m_far, farpos in self.plan(t):
            if q == 1:
                src, csum = np.take(self.hist[t], idx, axis=1), None                            # (B, n, 11)
            else:
                P = min(q, t + 1)                                                               # the steps of the period that exist
                c = self.a[:P, cells]
                src = c[0][None, :, None] * np.take(self.hist[t], idx, axis=1)
                for k in range(1, P):
                    src += c[k][None, :, None] * np.take(self.hist[t - k], idx, axis=1)
                csum = c.sum(0)
            H = np.einsum("bns,nsg->bng", src, self.WcT[cells]) + (self.bcT[cells] if csum is None else self.bcT[cells] * csum[:, None])
            G = np.zeros((len(xb), len(cells), 3), np.float32)
            if len(m_if):
                assert q == 1                                                                   # interface cells have period 1 in every arm
                G[:, m_if] = gi[:, :, ifpos].transpose(0, 2, 1)
            if len(m_far):
                G[:, m_far] = self.bfarT[farpos] if csum is None else self.bfarT[farpos] * csum[m_far][:, None]
            r = R.sigmoid(G[..., 0] + H[..., 0]); z = R.sigmoid(G[..., 1] + H[..., 1]); n = np.tanh(G[..., 2] + r * H[..., 2])
            new[:, cells] = (1 - z) * n + z * h[:, cells]
        if far_off:
            new[:, self.held] = 0.0
        if hold is not None:
            new[:, self.held] = hold
        self.h = new; self.hist.append(new); self.t += 1

    def logits(self, xb):
        z2 = np.tanh(self.l2[xb] + self.h[:, self.S.ifx] @ self.lmT + self.lmb)
        return z2 @ self.l3T + self.l3b


def decode(net, prompts, cap=4, far_off=False, hold=None, held=None, record=False, chunk=256):
    """Greedy decoding as the harness does it: at most `cap` bytes, an answer ends at the end symbol.  Returns the answers and,
    with record=True, every cell's state after each position of the prompt and of the first three answer bytes
    (positions, items, cells); the fourth byte is never an input of a scored run.
    hold: (positions, held cells), or (positions, items, held cells) for values that differ by item; held: the cells the
    diagnostics act on (default: every far cell)."""
    net.held = net.S.farx if held is None else held
    out, parts, groups = [None] * len(prompts), [], {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    assert not record or len(groups) == 1
    for idx in groups.values():
        for j in range(0, len(idx), chunk):
            ids = idx[j:j + chunk]
            x = np.array([[BOS] + prompts[i] for i in ids], dtype=np.int64)
            net.start(len(ids)); gen, done, st = [[] for _ in ids], [False] * len(ids), []

            def feed(xb):
                hd = None if hold is None else (hold[net.t][ids] if hold.ndim == 3 else hold[net.t])
                net.step(xb, far_off, hd)
                if record:
                    st.append(net.h)

            for t in range(x.shape[1]):
                feed(x[:, t])
            last = x[:, -1]
            for s in range(cap):
                yy = net.logits(last).argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q]))
                        done[q] = int(yy[q]) == EOS
                if s == cap - 1 or (all(done) and not record):
                    break
                feed(yy); last = yy
            for q, i in enumerate(ids):
                out[i] = gen[q]
            if record:
                parts.append(np.stack(st))
    net.held = net.S.farx
    return out, (np.concatenate(parts, axis=1) if record else None)


def mean32(a, axis):
    return a.mean(axis, dtype=np.float64).astype(np.float32)


def right(outs, items):
    return sum(o == t for o, (_, t) in zip(outs, items))


def three_modes(net, items):
    """Answers under: plain, far cells at zero, far cells at their mean over the items of this set at each position (the means
    are those of the plain run on the prompt followed by the network's own answer, as the worker's far_means takes them).
    Also every cell's states in the plain run."""
    prompts = [p for p, _ in items]
    plain, st = decode(net, prompts, record=True)
    zero = decode(net, prompts, far_off=True)[0]
    mean = decode(net, prompts, hold=mean32(st[:, :, net.S.farx], 1))[0]
    return plain, zero, mean, st


def reader_decode(W, S, prompts, cap=4, far_off=False, far_hold=None):
    """The same decoding through e47_local_reader.forward itself, the whole sequence computed again for every byte (part V)."""
    x = np.array([[BOS] + p for p in prompts], dtype=np.int64)
    gen, done = [[] for _ in prompts], [False] * len(prompts)
    for _ in range(cap):
        yy = R.forward(W, S, x, far_off=far_off, far_hold=far_hold)[1][:, -1].argmax(1)
        for q in range(len(prompts)):
            if not done[q]:
                gen[q].append(int(yy[q]))
                done[q] = int(yy[q]) == EOS
        x = np.concatenate([x, yy[:, None]], 1)
        if all(done):
            break
    return gen


# ---------------------------------------------------------------- the panels
LEN_TASK = ("add", 89, b"p")
LP = panel(LEN_TASK, split(LEN_TASK, .5, SEED)[1], SEED, ks=(12, 40), n=300)
AP = {f"{t[0]}{t[1]}": panel(t, split(t, .5, SEED)[1], SEED, ks=(12,), n=300)[12] for t in DEV}
RP = {}
for t in DEV[:2]:
    held = split(t, .5, SEED)[1]
    red = [(a, b) for a, b in held if (a + b >= 89 if t[0] == "add" else a - b < 0)]
    RP[f"{t[0]}{t[1]}"] = [render(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]


def three(curve, key, thr, sub=None):
    v = [(r["step"], (r[sub][key] if sub else r[key])) for r in curve if r["step"] > 0]
    for i in range(len(v) - 2):
        if all(x[1] >= thr for x in v[i:i + 3]):
            return v[i][0]
    return None


def describe(x_ref, x_new, d_ref, d_new):
    """E41's four cases, read as fewer / more / about the same (candidate against reference)."""
    if x_ref and x_new:
        q = x_new / x_ref
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if x_ref and not x_new:
        return "more" if d_new / x_ref >= 1.5 else "not decided"
    if not x_ref and x_new:
        return "fewer" if x_new / d_ref <= 2 / 3 else "not decided"
    return "neither reaches"


# ---------------------------------------------------------------- multiply-adds for one item at one position (forward), counted here
HEAD = 8 * 512 + 512 * 512 + 257 * 512                 # l1, l2, l3: 397,824
IFACE = 3 * 256 * 512 + 512 * 256                      # the input map to 256 cells and lm from 256 cells: 524,288


def madds_e47(S):
    """Head and interface as above; a cell of period 1 costs 33 (its 11 sources, three gates); a cell of period p > 1 costs
    33 / p for its products and 11 for adding up its sources' values with the step weights (the audit's count, change 8)."""
    cells = sum(33.0 if q == 1 else 33.0 / q + 11.0 for q in S.p.tolist())
    return HEAD + IFACE + cells


MADDS_G = HEAD + IFACE + 3 * 256 * 256                 # the present network: 1,118,720
MADDS_F10 = HEAD + IFACE + 3 * 256 * 11                # E42's F10: 930,560


def madds_p8():
    """E42's P8 (N = 10, periods 1, 2, 4, 8 on 128, 64, 32, 32 cells, a learned weight for each step of a period, every cell
    wired to the whole input and to the readout), least count on the same convention.  Recurrent side as in madds_e47.  Input
    side of a cell of period p > 1: its 512 input values are added up with the step weights at every position (512) and the
    three gates' products are made once a period (1,536 / p); a cell of period 1 costs 1,536.  This count is mine; neither
    the E42 nor the E47 brief states one."""
    cells = {1: 128, 2: 64, 4: 32, 8: 32}
    inp = sum(n * (1536.0 if q == 1 else 512.0 + 1536.0 / q) for q, n in cells.items())
    rec = sum(n * (33.0 if q == 1 else 33.0 / q + 11.0) for q, n in cells.items())
    return HEAD + 512 * 256 + inp + rec


E42 = {}


def e42(cfg, rule):
    if (cfg, rule) not in E42:
        rc = json.loads(show(f"{P}e42/e42_{cfg}_{rule}_{SEED}.json", E42REF)); c = rc["curve"]
        E42[(cfg, rule)] = dict(a80=three(c, "A", 80), r50=three(c, "R", 50), r80=three(c, "R", 80), r90=three(c, "R", 90),
                                finalA=c[-1]["A"], finalR=c[-1]["R"], len12=c[-1]["len12"],
                                len40=float(np.mean([r["len40"] for r in c[-3:]])), dose=c[-1]["step"],
                                madds={"G": MADDS_G, "F10": MADDS_F10, "P8": madds_p8()}[cfg], cpu=rc.get("cpu_per_update"))
    return E42[(cfg, rule)]


RC = {}


def rcpt(arm, rule):
    if (arm, rule) not in RC:
        RC[(arm, rule)] = receipt(arm, rule)
    return RC[(arm, rule)]


def tab(arm, rule):
    rc = rcpt(arm, rule); c = rc["curve"]; last = c[-1]
    return dict(a80=three(c, "A", 80), r50=three(c, "R", 50), r80=three(c, "R", 80), r90=three(c, "R", 90), a90=three(c, "A", 90),
                finalA=last["A"], finalR=last["R"], taught=last["taught"], len12=last["len12"], len40_last=last["len40"],
                len40=float(np.mean([r["len40"] for r in c[-3:]])), dose=last["step"], madds=madds_e47(grid(arm)),
                zero=last["far_off"], mean=last["far_mean"], rc=rc)


def form(t):
    """The brief's forms for the use of the far cells, the first that applies."""
    fz = (t["finalA"] - t["zero"]["A"], t["finalR"] - t["zero"]["R"]); fm = (t["finalA"] - t["mean"]["A"], t["finalR"] - t["mean"]["R"])
    if max(fm) >= 15:
        f = "used"
    elif max(fz) < 5:
        f = "not used"
    elif max(fz) >= 15 and max(fm) < 5:
        f = "a constant is used"
    else:
        f = "partly"
    return f, fz, fm


# ================================================================ A
if "A" in want:
    print("A. files, checks, and what the receipts record")
    ls = subprocess.run(["git", "ls-tree", "-r", "-l", REF, "--", P + "e47", P + "e47_qual"], capture_output=True, text=True, check=True).stdout.splitlines()
    files = {l.split("\t")[1]: int(l.split()[3]) for l in ls}
    for arm, rule in RUNS:
        base = f"{P}e47/e47_{SIDE}_{arm}_{rule}_{SEED}"
        have = {s: files.get(base + s) for s in (".json", ".pt", "_start.pt", "_answers.json")}
        print(f"   {arm:5s} {rule}: receipt, final weights, start weights, answers on the branch (bytes): {list(have.values())}")
        assert all(have.values())
    for n in range(1, 9):
        d = json.loads(show(f"{P}e47_qual/check{n}.json"))
        src = d["provenance"]["sources_sha256"]["exp47_local.py"][:8]
        print(f"   check {n}: pass = {d['pass']}; runner {src}, commit {d['provenance']['git']['commit'][:8]}")
        assert d["pass"] is True
    blob = lambda ref, path: subprocess.run(["git", "rev-parse", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout.strip()
    b_spec, b_copy = blob(DREF, "PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e47_local_spec.py"), blob(REF, M + "e47_local.py")
    print(f"   the specification file on the default branch and the worker's e47_local.py are the same blob: {b_spec == b_copy} ({b_copy[:8]})")
    assert b_spec == b_copy
    starts = {}
    for arm, rule in RUNS:
        rc = rcpt(arm, rule)
        g = rc["optimizer"]["groups"]
        dec = [x for x in g if x["weight_decay"] == 0.25][0]["params"]; nodec = [x for x in g if x["weight_decay"] == 0.0][0]["params"]
        row4096 = [r for r in rc["curve"] if r["step"] == 4096][0]["data_digest"]["sha256"][:16]
        starts.setdefault(arm, set()).add(rc["start_tensors_sha256"])
        print(f"   {arm:5s} {rule}: complete {rc['complete']}; rows {len(rc['curve'])}; runner {rc['provenance']['sources_sha256']['exp47_local.py'][:8]};"
              f" data digest at row 4,096 {row4096}; decayed: Wc, bc, bfar{', a' if 'a' in dec else ''} in the group {all(k in dec for k in ('Wc', 'bc', 'bfar'))};"
              f" module's recurrent tensors in a group {any(k.startswith('rnn.weight_hh') or k.startswith('rnn.bias_hh') for k in dec + nodec)};"
              f" not decayed {nodec}; module's recurrent tensors without gradient {rc['module_recurrent_untouched']};"
              f" resumes {len(rc.get('resumes', []))}; weights {rc['weight_count']:,}")
        assert rc["complete"] and row4096 == "bbd71ab7ca27b72e" and len(rc["curve"]) == 97
    print(f"   one start for both rules of an arm: {all(len(v) == 1 for v in starts.values())} ({ {k: sorted(v)[0][:8] for k, v in starts.items()} })")

# ================================================================ B
if "B" in want:
    print("B. the saved structure against the reader's draws (final and starting files)")
    dg = lambda v: hashlib.sha256(json.dumps(v).encode()).hexdigest()[:16]
    for arm, rule in RUNS:
        S = grid(arm)
        for start in (False, True):
            T = weights(arm, rule, start)
            ok = dict(sources=bool((T["sources"] == S.idx).all()), periods=bool((T["periods"] == S.p).all()), phases=bool((T["phases"] == S.f).all()),
                      interface=bool((T["interface"] == S.ifx).all()), far=bool((T["far"] == S.farx).all()))
            extra = ""
            if "a" in T:
                inside = np.arange(T["a"].shape[0])[:, None] < S.p[None, :]
                extra = f"; a outside a cell's period: largest {float(np.abs(T['a'][~inside]).max()):.1e}; a of period-1 cells (not used): {float(T['a'][0, S.p == 1].min()):.3f} to {float(T['a'][0, S.p == 1].max()):.3f}"
            print(f"   {arm:5s} {rule} {'start' if start else 'final'}: {ok}; Wc {T['Wc'].shape}, bfar {T['bfar'].shape}{extra}")
            assert all(ok.values())
            del T
        if rule == "S":
            print(f"      digests of the draws ({arm}): connections {dg(S.A)}, periods {dg([int(v) for v in S.p])}, phases {dg([int(v) for v in S.f])};"
                  f" cells by period { {int(q): int((S.p == q).sum()) for q in sorted(set(S.p.tolist()))} }")
    for arm in ("one", "place"):
        a, b = weights(arm, "S", True), weights(arm, "B", True)
        print(f"   {arm}: the starting files of the two rules hold the same tensors: {all((a[k] == b[k]).all() for k in a)}")
        del a, b

# ================================================================ V
if "V" in want:
    NV = 16
    print(f"V. the arrangement used for decoding against the reader's own forward (final weights; {NV} items of the 12-letter panel and {NV} of R)")
    rng = np.random.default_rng(7)
    for arm, rule in RUNS:
        S = grid(arm); W = weights(arm, rule); net = Net(W, S)
        items = LP[12][:NV]
        x = np.array([[BOS] + pr + t[:-1] for pr, t in items], dtype=np.int64)
        hold = rng.uniform(-1, 1, (x.shape[1], len(S.farx))).astype(np.float32)
        worst = {}
        for name, kw_r, kw_n in (("plain", {}, {}), ("far cells at zero", dict(far_off=True), dict(far_off=True)), ("far cells at given values", dict(far_hold=hold), None)):
            lg_r = R.forward(W, S, x, **kw_r)[1]
            net.start(len(x)); d = 0.0
            for t in range(x.shape[1]):
                net.step(x[:, t], hold=hold[t]) if kw_n is None else net.step(x[:, t], **kw_n)
                d = max(d, float(np.abs(net.logits(x[:, t]) - lg_r[:, t]).max()))
            worst[name] = d
        same = {}
        for label, its in (("12 letters", LP[12][:NV]), ("no letters", RP["add89"][:NV])):
            prompts = [pr for pr, _ in its]
            pl, st = decode(net, prompts, record=True)
            mean = mean32(st[:, :, S.farx], 1)
            same[label] = [sum(u == v for u, v in zip(pl, reader_decode(W, S, prompts))),
                           sum(u == v for u, v in zip(decode(net, prompts, far_off=True)[0], reader_decode(W, S, prompts, far_off=True))),
                           sum(u == v for u, v in zip(decode(net, prompts, hold=mean)[0], reader_decode(W, S, prompts, far_hold=mean)))]
        print(f"   {arm:5s} {rule}: largest difference of a logit: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items())
              + f"; answers identical of {NV} (plain, zero, mean): " + ", ".join(f"{k} {v}" for k, v in same.items()) + f"   [{time.time() - T0:.0f} s]", flush=True)
        del W, net

# ================================================================ C and D
MODES = ("plain", "off", "mean")
if "C" in want or "D" in want:
    for arm, rule in RUNS:
        S = grid(arm); W = weights(arm, rule); net = Net(W, S); rc = rcpt(arm, rule); last = rc["curve"][-1]
        if "C" in want:
            theirs = json.loads(show(f"{P}e47/e47_{SIDE}_{arm}_{rule}_{SEED}_answers.json"))
            line = []
            for k in (12, 40):
                assert theirs["prompts"][str(k)] == [p for p, _ in LP[k]] and theirs["targets"][str(k)] == [t for _, t in LP[k]]
                mine = dict(zip(MODES, three_modes(net, LP[k])[:3]))
                for mode in MODES:
                    same = sum(m == t for m, t in zip(mine[mode], theirs["answers"][mode][str(k)]))
                    rec = last["length_counts"][str(k)] if mode == "plain" else last["far_" + mode]["length_counts"][str(k)]
                    line.append(f"{k} letters, {mode}: {same}/300 identical, right {right(mine[mode], LP[k])} (receipt {rec})")
            print(f"C. {arm:5s} {rule}  " + " | ".join(line) + f"   [{time.time() - T0:.0f} s]", flush=True)
        if "D" in want:
            outA, outR = {}, {}
            for n, it in AP.items():
                outA[n] = [right(o, it) for o in three_modes(net, it)[:3]]
            for n, it in RP.items():
                outR[n] = [right(o, it) for o in three_modes(net, it)[:3]]
            recA = {n: [last["A_counts"][n], last["far_off"]["A_counts"][n], last["far_mean"]["A_counts"][n]] for n in AP}
            recR = {n: [last["R_counts"][n], last["far_off"]["R_counts"][n], last["far_mean"]["R_counts"][n]] for n in RP}
            print(f"D. {arm:5s} {rule}  A (plain, zero, mean; reader / receipt): " + ", ".join(f"{n} {outA[n]} / {recA[n]}" for n in AP)
                  + f" equal {outA == recA}")
            print(f"            R: " + ", ".join(f"{n} {outR[n]} / {recR[n]}" for n in RP) + f" equal {outR == recR}   [{time.time() - T0:.0f} s]", flush=True)
        del W, net

# ================================================================ E
if "E" in want:
    print("E. the receipts recomputed (three scored rows in a row at or above the mark; updates)")
    print("   run        | A80    R50    R80    R90   | final A, R, taught | 12 letters | 40 letters: last row, mean of last three | multiply-adds a position (counted here; receipt)")
    for arm, rule in RUNS:
        t = tab(arm, rule)
        print(f"   {arm:5s} {rule}    | {str(t['a80']):>6} {str(t['r50']):>6} {str(t['r80']):>6} {str(t['r90']):>6} | {t['finalA']:.2f}, {t['finalR']:.2f}, {t['taught']:.2f}"
              f" | {t['len12']:.2f} | {t['len40_last']:.2f}, {t['len40']:.2f} | {t['madds']:,.0f}; {t['rc']['multiply_adds_with_periods']:,.0f}")
    for cfg in ("G", "F10", "P8"):
        for rule in ("S", "B"):
            t = e42(cfg, rule)
            print(f"   E42 {cfg:3s} {rule}  | {str(t['a80']):>6} {str(t['r50']):>6} {str(t['r80']):>6} {str(t['r90']):>6} | {t['finalA']:.2f}, {t['finalR']:.2f}"
                  f"          | {t['len12']:.2f} | mean of last three {t['len40']:.2f} | {t['madds']:,.0f}")
    print("   computation to the mark = multiply-adds a position x updates (the items of an update and their lengths are the same data in every run);")
    print("   ratios: to E42's full network G under the same rule, and to G under back-propagation (the programme's reference)")
    GB = e42("G", "B")
    for key in ("a80", "r80"):
        print(f"   --- {key.upper()}")
        rows = [(f"{arm} {rule}", tab(arm, rule), rule) for arm, rule in RUNS] + [(f"E42 {cfg} {rule}", e42(cfg, rule), rule) for cfg in ("F10", "P8", "G") for rule in ("S", "B")]
        for name, t, rule in rows:
            g = e42("G", rule)
            comp = t[key] * t["madds"]
            print(f"   {name:10s} updates {t[key]:6d}  x {t['madds']:>11,.0f} = {comp:.3e}; against G of the rule: updates x{t[key] / g[key]:.2f}, computation x{comp / (g[key] * g['madds']):.2f};"
                  f" against G under back-propagation: computation x{comp / (GB[key] * GB['madds']):.2f}")
    alt = HEAD + IFACE + 128 * 33 + 64 * (16.5 + 11) + 32 * (8.25 + 11) + 32 * (4.125 + 11)      # P8 with every cell's input products made at every position
    print(f"   P8 counted with the input products of every cell made at every position, as E42's program makes them: {alt:,.0f} multiply-adds a position;"
          f" computation against G under back-propagation: " + ", ".join(f"{key.upper()} {rule} x{e42('P8', rule)[key] * alt / (GB[key] * GB['madds']):.2f}" for key in ("a80", "r80") for rule in ("S", "B")))
    print("   the brief's descriptions (candidate against reference: updates to 80 on R, on A, the same in multiply-adds; 40 letters, mean of last three rows)")
    def line(new, ref, label):
        out = []
        D = new["dose"] - 512
        for key in ("r80", "a80"):
            out.append(f"{key.upper()} updates {describe(ref[key], new[key], D, D)} ({ref[key]}, {new[key]})")
        for key in ("r80", "a80"):
            out.append(f"{key.upper()} multiply-adds {describe(ref[key] * ref['madds'], new[key] * new['madds'], D * ref['madds'], D * new['madds'])} (x{new[key] * new['madds'] / (ref[key] * ref['madds']):.2f})")
        d = new["len40"] - ref["len40"]
        out.append(f"40 letters {'lower' if d <= -15 else 'higher' if d >= 15 else 'within 15'} ({d:+.1f})")
        print(f"   {label:24s} " + " | ".join(out))
    for arm in ("one", "place"):
        for rule in ("S", "B"):
            for cfg in (("F10", "P8", "G") if arm == "place" else ("F10", "G")):
                line(tab(arm, rule), e42(cfg, rule), f"{arm} {rule} against {cfg} {rule}")
    for rule in ("S", "B"):
        line(tab("place", rule), tab("one", rule), f"place {rule} against one {rule}")
    for arm in ("one", "place"):
        line(tab(arm, "S"), tab(arm, "B"), f"{arm} S against {arm} B")
    print("   use of the far cells at the last row (fall of A, of R: far cells at zero; at their mean) and the brief's form")
    for arm, rule in RUNS:
        t = tab(arm, rule); f, fz, fm = form(t)
        print(f"   {arm:5s} {rule}: at zero {fz[0]:.1f}, {fz[1]:.1f}; at the mean {fm[0]:.1f}, {fm[1]:.1f}; 40 letters plain {t['len40_last']:.1f}, zero {t['zero']['len40']:.1f}, mean {t['mean']['len40']:.1f}: {f}")
    print("   the falls over the last eight rows (A at zero, A at the mean, R at zero, R at the mean): mean and range")
    for arm, rule in RUNS:
        c = rcpt(arm, rule)["curve"][-8:]
        s = lambda v: f"{np.mean(v):.1f} ({min(v):.1f} to {max(v):.1f})"
        print(f"   {arm:5s} {rule}: " + "; ".join(s([r[k] - r[d][k] for r in c]) for k in ("A", "R") for d in ("far_off", "far_mean")))
    print("   magnitudes (mean size of a recurrent weight: interface cells, far cells, the sweep's scale) at the start, 8,192, 16,384 and the end")
    for arm, rule in RUNS:
        c = {r["step"]: r["magnitudes"] for r in rcpt(arm, rule)["curve"]}
        print(f"   {arm:5s} {rule}: " + " | ".join(f"{u:,}: {c[u]['interface']:.3f}, {c[u]['far']:.3f}, {c[u]['scale']:.3f}" for u in (0, 8192, 16384, 24576)))

# ================================================================ F
if "F" in want:
    print("F. measurements on the saved weights (the reader; answers right in percent)")
    rng_perm = np.random.default_rng(20261117)
    add, sub = DEV[0], DEV[1]
    KS = (0, 4, 8, 12)
    LET = {}
    for t in (add, sub):
        pn = panel(t, split(t, .5, SEED)[1], SEED, ks=KS, n=300)
        for k in KS:
            LET.setdefault(k, {})[t[0]] = pn[k]
    first = True
    for arm, rule in RUNS:
        if FARMS and (arm, rule) not in FARMS:
            continue
        S = grid(arm); W = weights(arm, rule); net = Net(W, S)
        ring = -(-S.dist // 3)                                   # connections from the interface: 0 for interface cells, then 1, 2, 3
        ring_far = ring[S.farx]
        if first:
            first = False
            src = S.idx[S.ifx][:, 1:]
            back = S.idx[S.farx][:, 1:]
            print(f"   structure: {100 * float((S.dist[src] > 0).mean()):.1f}% of the interface cells' connections come from far cells ({int(((S.dist[src] > 0).sum(1) > 0).sum())} of 256 interface cells have at least one);"
                  f" far cells by connections from the interface: { {int(r): int((ring_far == r).sum()) for r in sorted(set(ring_far.tolist()))} };"
                  f" {100 * float((S.dist[back] == 0).mean()):.1f}% of the far cells' connections come from interface cells"
                  f" (ring 1: {100 * float((S.dist[S.idx[S.farx[ring_far == 1]][:, 1:]] == 0).mean()):.1f}%)")
        print(f"   --- {arm} {rule}")
        # 1. by the number of letters
        for k in KS:
            res = np.zeros(3)
            for op in ("add", "sub"):
                o = three_modes(net, LET[k][op])
                res += np.array([right(x, LET[k][op]) for x in o[:3]])
            res = 100 * res / 600
            print(f"   {k:2d} letters (add and sub, 600 held-out items): plain {res[0]:.1f}, far cells at zero {res[1]:.1f} (fall {res[0] - res[1]:.1f}), at their mean {res[2]:.1f} (fall {res[0] - res[2]:.1f})", flush=True)
        # 2. rings, another item's far states, on A (add and sub at 12 letters), R (no letters) and the 40-letter panel
        for label, sets in (("A (12 letters)", [AP["add89"], AP["sub89"]]), ("R (no letters)", [RP["add89"], RP["sub89"]]), ("40 letters", [LP[40]])):
            n = sum(len(x) for x in sets)
            acc, sd_far, sd_if = {}, [], []
            def add_to(name, outs, items):
                acc[name] = acc.get(name, 0) + right(outs, items)
            for items in sets:
                prompts = [p for p, _ in items]
                plain, st = decode(net, prompts, record=True)     # (positions, items, cells)
                far = st[:, :, S.farx]; mean = mean32(far, 1)
                add_to("plain", plain, items)
                add_to("zero", decode(net, prompts, far_off=True)[0], items)
                add_to("mean", decode(net, prompts, hold=mean)[0], items)
                for name, sel in (("ring 1 at its mean", ring_far == 1), ("rings 2 and 3 at their mean", ring_far >= 2), ("ring 3 at its mean", ring_far == 3)):
                    add_to(name, decode(net, prompts, hold=mean[:, sel], held=S.farx[sel])[0], items)
                perm = rng_perm.permutation(len(items))
                while (perm == np.arange(len(items))).any():
                    perm = rng_perm.permutation(len(items))
                add_to("another item's states", decode(net, prompts, hold=far[:, perm])[0], items)
                add_to("another item's states in rings 2 and 3", decode(net, prompts, hold=far[:, perm][:, :, ring_far >= 2], held=S.farx[ring_far >= 2])[0], items)
                eq = len(prompts[0])                              # the position of '=' (the start symbol is position 0)
                sd_far.append(far[eq].std(0)); sd_if.append(st[eq][:, S.ifx].std(0))
                del st, far
            sdf, sdi = np.mean(sd_far, 0), np.mean(sd_if, 0)
            byring = ", ".join(f"ring {r}: {float(sdf[ring_far == r].mean()):.3f}" for r in (1, 2, 3))
            print(f"   {label:15s}: " + ", ".join(f"{k} {100 * v / n:.1f}" for k, v in acc.items()))
            print(f"   {'':15s}  standard deviation over items of a cell's state at '=': interface cells {float(sdi.mean()):.3f}; far cells {float(sdf.mean()):.3f} ({byring})   [{time.time() - T0:.0f} s]", flush=True)
        # 3. the weights by which interface cells read far cells, beside those by which they read interface cells
        Wc = np.abs(W["Wc"]).sum(0)                               # (cells, 11): the three gates' magnitudes added
        src = S.idx[S.ifx][:, 1:]; w = Wc[S.ifx][:, 1:]
        fromfar = S.dist[src] > 0
        wf = Wc[S.farx][:, 1:]
        print(f"   mean size of a connection's weights (three gates added): interface cell reading an interface cell {float(w[~fromfar].mean()):.3f}, reading a far cell {float(w[fromfar].mean()):.3f};"
              f" far cell reading any cell {float(wf.mean()):.3f} (ring 1: {float(wf[ring_far == 1].mean()):.3f}, ring 2: {float(wf[ring_far == 2].mean()):.3f}, ring 3: {float(wf[ring_far == 3].mean()):.3f})", flush=True)
        W0 = weights(arm, rule, start=True)
        m1, m0 = np.abs(W["Wc"]).mean((0, 2)), np.abs(W0["Wc"]).mean((0, 2))          # per cell: mean size of its 33 recurrent weights
        grp = [("interface", S.dist == 0)] + [(f"ring {r}" + (f" (period {int(S.p[ring == r].min())} to {int(S.p[ring == r].max())})" if arm == "place" else ""), ring == r) for r in (1, 2, 3)]
        if arm == "place":
            grp += [(f"period {q}", S.p == q) for q in (2, 4, 8, 16)]
        print("   mean size of a cell's recurrent weights, end / start: " + "; ".join(f"{n} {float(m1[g].mean()):.3f} / {float(m0[g].mean()):.3f}" for n, g in grp), flush=True)
        if "a" in W:
            a = W["a"]; inside = np.arange(a.shape[0])[:, None] < S.p[None, :]
            for q in (2, 4, 8, 16):
                c = a[:q, S.p == q]                                                    # (q, cells of this period): step k = 0 is the newest
                print(f"      step weights of the cells of period {q} at the end (start {1 / q:.4f} each): mean over cells by step, newest first: {[round(float(v), 3) for v in c.mean(1)]}")
        del W, W0, net

# ================================================================ I
if "I" in want:
    print("I. at the start, before any update: how much a cell's state varies with the item, by ring (standard deviation over the items of a panel")
    print("   of the state at '='; then the mean size of a state).  The two rules of an arm share one start.")
    for arm in ("one", "place"):
        S = grid(arm); W = weights(arm, "S", start=True); net = Net(W, S)
        ring = -(-S.dist // 3)
        for label, items in (("A, addition (12 letters)", AP["add89"]), ("R, addition (no letters; 300 items)", RP["add89"][:300]), ("40 letters", LP[40])):
            prompts = [p for p, _ in items]
            st = decode(net, prompts, record=True)[1]
            eq = len(prompts[0]); sd = st[eq].std(0)
            print(f"   {arm:5s} {label:36s}: interface {float(sd[ring == 0].mean()):.3f}, ring 1 {float(sd[ring == 1].mean()):.4f}, ring 2 {float(sd[ring == 2].mean()):.5f}, ring 3 {float(sd[ring == 3].mean()):.6f}"
                  f" | mean size of a state: " + ", ".join(f"{float(np.abs(st[eq][:, ring == r]).mean()):.3f}" for r in (0, 1, 2, 3)), flush=True)
        Wc = W["Wc"]
        print(f"   {arm:5s} starting recurrent weights: mean size {float(np.abs(Wc).mean()):.3f}, root mean square {float(np.sqrt((Wc ** 2).mean())):.3f}, largest {float(np.abs(Wc).max()):.3f} (drawn uniformly within 1 / sqrt(11) = {11 ** -0.5:.3f})")
        del W, net

# ================================================================ G
if "G" in want:
    print("G. the nine predictions, scored from the receipts")
    c1 = json.loads(show(f"{P}e47_qual/check1.json")); c2 = json.loads(show(f"{P}e47_qual/check2.json"))
    t = {(a, r): tab(a, r) for a, r in RUNS}
    forms = {k: form(v)[0] for k, v in t.items()}
    print(f"   P47.1 (85%) checks 1 and 2 pass under both rules: {c1['pass'] and c2['pass'] and c2['first_batch']['S']['pass'] and c2['first_batch']['B']['pass']}")
    print(f"   P47.2 (70%) back-propagation, place: reaches 80 on R: {t[('place', 'B')]['r80'] is not None} (from {t[('place', 'B')]['r80']})")
    print(f"   P47.3 (45%) sweep, place: reaches 80 on R: {t[('place', 'S')]['r80'] is not None} (from {t[('place', 'S')]['r80']}; final R {t[('place', 'S')]['finalR']:.2f})")
    print(f"   P47.4 (45%) back-propagation, one: far cells used: {forms[('one', 'B')] == 'used'} ({forms[('one', 'B')]})")
    print(f"   P47.5 (30%) sweep, one: far cells used: {forms[('one', 'S')] == 'used'} ({forms[('one', 'S')]})")
    print(f"   P47.10 (30%) a constant is used in at least one run: {any(f == 'a constant is used' for f in forms.values())} ({forms})")
    d = describe(t[('one', 'B')]['r80'], t[('place', 'B')]['r80'], 24064, 24064)
    print(f"   P47.11 (50%) back-propagation: place about the same as one in updates to 80 on R: {d == 'about the same'} ({t[('place', 'B')]['r80']} against {t[('one', 'B')]['r80']})")
    cs = {r: rcpt('place', r)['cosine_first_batch_end'] for r in ('S', 'B')}
    gap = {r: cs[r]['interface'] - cs[r]['far'] for r in cs}
    print(f"   P47.6 (60%) place: end cosine lower for far than for interface cells by 0.1 or more: sweep's run {gap['S']:.3f} ({gap['S'] >= 0.1}), back-propagation's run {gap['B']:.3f} ({gap['B'] >= 0.1})")
    d = describe(e42('F10', 'B')['r80'], t[('one', 'B')]['r80'], 24064, 24064)
    print(f"   P47.7 (80%) back-propagation, one: updates to 80 on R about the same as or more than E42's F10: {d in ('about the same', 'more')} ({d}: {t[('one', 'B')]['r80']} against {e42('F10', 'B')['r80']})")
    print("   cosines between the sweep's and the exact gradient of Wc on the first batch (interface, far): start and end")
    for arm, rule in RUNS:
        a, b = rcpt(arm, rule)["cosine_first_batch_start"], rcpt(arm, rule)["cosine_first_batch_end"]
        print(f"   {arm:5s} {rule}: start {a['interface']:.3f}, {a['far']:.3f}; end {b['interface']:.3f}, {b['far']:.3f}; share of far cells' recurrent weights with a gradient at the end: B {b['far_nonzero_share_B']:.3f}, S {b['far_nonzero_share_S']:.3f}")

# ================================================================ H
if "H" in want:
    print("H. cost")
    tot = 0.0
    for arm, rule in RUNS:
        rc = rcpt(arm, rule); h = (rc["cpu_updates"] + rc["cpu_scoring"]) / 3600; tot += h
        print(f"   {arm:5s} {rule}: {rc['cpu_updates'] / 3600:.2f} CPU-hours of updates ({rc['cpu_per_update']:.3f} s an update), {rc['cpu_scoring'] / 3600:.2f} of scoring ({rc['cpu_scoring'] / 97:.1f} s a row); {h:.2f} in all; wall {rc['wall'] / 3600:.2f} h")
    c7 = json.loads(show(f"{P}e47_qual/check7.json"))
    print(f"   the four runs: {tot:.1f} CPU-hours (check 7 forecast {c7['forecast_trial_total_cpu_h']:.1f}, measured serially on an idle host)")
    print(f"   check 7's forecast for the roster as planned: { {k: round(v, 1) for k, v in c7['forecast_roster_cpu_h'].items()} }; total {c7['forecast_roster_total_cpu_h']:.1f} CPU-hours")
    ratio = tot / c7["forecast_trial_total_cpu_h"]
    print(f"   measured / forecast on the trial: {ratio:.2f}; the roster at that ratio: {ratio * c7['forecast_roster_total_cpu_h']:.1f} CPU-hours")
    for k, v in c7["cost"].items():
        print(f"      {k}: {v['cpu_per_update']:.3f} s an update, {v['cpu_per_scored_row']:.1f} s a scored row, forecast {v['forecast_cpu_h']:.1f} CPU-hours")
print(f"done in {time.time() - T0:.0f} s")
