"""E39 (brief OPUS_TASK_E39_SMALL_WORLD_PROTOCOL_L0_L1_20261115.md, revision 2): the small world's runner.

One life = one arm in one world for V visits of the world's schedule (sw_world.World, the audited generator, imported unchanged).
Every update takes the visit's next 64 stream items (48 taught, then 16 asked):

  1. the 16 asked items are answered first, by greedy generation under no_grad (sw_net.answer); nothing is learned from them;
  2. the arm's batch of 48 is formed: `naive`/`sweep` learn from the 48 taught items; `replay`/`sweep+replay` (share rho) draw
     round(rho*48) items from the buffer with replacement by randrange(len(buffer)) from Random("world|S|replay"), once the buffer holds
     that many, and learn from the first 48 - round(rho*48) new taught items plus those; then the 48 new taught items are offered to the
     buffer in batch order from the same generator (stored while it is not full; after that j = randrange(n) for the n-th offered item,
     which replaces slot j if j < capacity). Capacity (4 * P) // 15 items, P the parameter count (embedding table included);
  3. one update, sw_net.step (exp26_candidate.step's text): local=False for backprop, True for the sweep.

Every learned item is parsed back from its own bytes (code -> task, operands) and asserted to be a taught pair with its true answer.
At the first update of every visit the weights' and the optimizer state's tensors hashes are asserted unchanged across the answering.

Outputs, in --out (default exp45/e39/), named sw_<arm>_w<world>_v<visits>[_<setting tag>]; every path is refused if it exists:
  <stem>.json            the receipt: settings, provenance, source hashes, schedule, stream and learned-batch digests, the ledger,
                         CPU per visit, losses, final hashes, the states written
  <stem>.npz             the event file: one entry per asked item in stream order (update, visit, task, correct, blind, wrap, mirror,
                         seen, replayed), compressed
  <stem>_state_v<n>.pt   complete state after n = 1, 2, 4, 8, ... visits and at the end: weights, optimizer state, buffer, replay
                         generator state, per-task counts, the events so far, the per-visit records

    python3 sw_run.py life <arm> <world> <visits> [--rho R] [--H 64] [--U 64] [--visit-updates 256] [--p-new 0.2] [--lr 0.001]
                           [--wd 0.25] [--tag TAG] [--out DIR] [--stop-after N]
    python3 sw_run.py life --resume <state> [--out DIR] [--stop-after N]          # continues a life exactly
    python3 sw_run.py branch <state> --groups g1,g2,... [--out DIR]                # the named groups in place of the schedule; k-th visit's
                                                                                  #   items from Random("world|S|branch|<n>|<k>"), k from 0
    python3 sw_run.py ask <state> --items FILE.json --to OUT.json                  # [[task, a, b(, letters)], ...], asked pairs only
    python3 sw_run.py solo <naive|sweep> <world> <task> [--H 64] [--updates 16384]   # revision 4, L1a: one task alone
    python3 sw_run.py life <arm> <world> <visits> ... --birth-groups 0,1 [--birth-cap 32768] [--birth-need 3]   # revision 4, L1b
    python3 sw_run.py life --birth-from <naive ..._state_v0.pt>                     # revision 4: sweep-after-naive

E44 (brief OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md, revision 2): the long-life world in the full-scale format.
    python3 sw_run.py life <naive|sweep|sweep-after-naive> <world> <visits> --world lw --learner <E41 birth .pt> --H 256 --U 128
                           [--in-turn] [--visit-updates 256] [--wd 0.25] [--taught-share 0.10] [--lowest 41] [--save-states 8]
  --world lw      the world is lw_world.World (byte-identical to COORD_CHECKS_20261116/lw_world_spec.py); --world sw (default) is E39's
  --learner F     the life starts from E41's saved birth F (weights only), loaded into sw_net.SWNet(H, U): arch20's parameter names in
                  arch20's order, as exp41_scale.NetW, run by arch26.forward (exp41_scale itself cannot be imported beside this runner:
                  its environment gate asserts on an exp11_cell already imported). Its tensors hash and file hash are asserted equal to
                  F's receipt (F with .json), and the arm to the birth's rule (naive and sweep-after-naive: B; sweep: S). One fresh
                  sw_net.make_opt(m, lr, wd) for the life.
  --in-turn       calibration only: visit v is family v
  every file      the event file gains g1, g2, g3 (uint16): the bytes generated for the asked item, 65535 once generation has ended
  the list        (lw) at the end of every visit, answered without learning and written to <stem>_list.npz: for add and for sub of every
                  family taken so far, the first 100 asked pairs in the split's order that need the reduction, no letters; and the birth
                  family's R items (E41's two samples, 1,000 each). Before visit 0 the birth family's items once. "Decay alone": at the end
                  of each family's first visit its weights, every parameter but the byte table multiplied by (1 - lr * wd)^u, answer its
                  items for every later visit of the life (u the updates from the end of its visit to the end of that visit); the born
                  weights so answer the birth family's items for every visit. Nothing is learned from the list.
  --save-states   which visit counts get a state (default: 1, 2, 4, 8, ... and the end, E39's rule); e.g. 8 or 4,8
"""
import argparse
import hashlib
import io
import json
import os
import platform
import socket
import subprocess
import sys
import time
from array import array
from pathlib import Path
from random import Random

import numpy as np
import torch as T

HERE = Path(__file__).resolve().parent
MDIR = HERE.parent
sys.path.insert(0, str(HERE))
import sw_world as W                  # noqa: E402  (the audited generator, byte-identical to COORD_CHECKS_20261115/sw_world_spec.py)
import lw_world as LW                 # noqa: E402  (E44: byte-identical to COORD_CHECKS_20261116/lw_world_spec.py)
import sw_net as N                    # noqa: E402
import sw_score as S                  # noqa: E402  (revision 4: the criterion of L1a, from the event file)
import arch26                         # noqa: E402  (already imported by sw_net; its exp11_cell gives E41's split and render)

T.set_num_threads(1)
OUT = MDIR / "exp45" / "e39"
ARMS = {"naive": (False, False), "sweep": (True, False), "replay": (False, True), "sweep+replay": (True, True),
        "sweep-after-naive": (True, False)}   # arm -> (local, replay); sweep-after-naive: the sweep's life from naive's state at the end of its birth
BIRTH_VISIT = 65535                   # E39 revision 4: the `visit` value of a birth's asked items in the event file
DEFAULTS = dict(H=64, U=64, visit_updates=W.VISIT_UPDATES, p_new=W.P_NEW, lr=0.001, wd=0.25)
LW_DEFAULTS = dict(H=256, U=128, visit_updates=LW.VISIT_UPDATES, p_new=LW.P_NEW, lr=0.001, wd=0.25, taught_share=LW.TAUGHT_FRACTION, lowest=LW.LOWEST)
TAGS = dict(H="h", U="u", visit_updates="vu", p_new="pn", lr="lr", wd="wd", taught_share="ts", lowest="lo")
ITEM_MAX = 15                         # bytes of the longest item: 12 of prompt, 2 digits and the end symbol
ITEM_MAX_LW = 23                      # E44: code 2, a 2, symbol, 12 letters, b 2, '=', 2 digits and the end symbol
GEN_END = 65535                       # E44: g1..g3 once generation has ended (after the end symbol)
BIRTH_FAMILY = -1                     # E44: the list's family number for the birth family (E26's, mod 89)
BIRTH_LIST_TASKS = (65534, 65535)     # E44: the list's task numbers for add89 and sub89
BEFORE_LIFE = 65535                   # E44: the list's at_visit for the answers before visit 0
SOURCES = {"smallworld/" + s: HERE / s for s in ("sw_world.py", "lw_world.py", "sw_net.py", "sw_run.py", "sw_score.py")}
SOURCES.update({s: MDIR / s for s in ("arch26.py", "arch20.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py")})
EVENT_COLUMNS = (("update", "I", np.uint32), ("visit", "H", np.uint16), ("task", "H", np.uint16), ("correct", "B", np.uint8),
                 ("blind", "B", np.uint8), ("wrap", "B", np.uint8), ("mirror", "B", np.uint8), ("seen", "I", np.uint32),
                 ("replayed", "I", np.uint32), ("g1", "H", np.uint16), ("g2", "H", np.uint16), ("g3", "H", np.uint16))
LIST_COLUMNS = (("at_visit", "H", np.uint16), ("source", "B", np.uint8), ("family", "h", np.int16), ("task", "H", np.uint16),
                ("a", "B", np.uint8), ("b", "B", np.uint8), ("g1", "H", np.uint16), ("g2", "H", np.uint16), ("g3", "H", np.uint16),
                ("correct", "B", np.uint8), ("decay_updates", "I", np.uint32), ("weights_from_visit", "H", np.uint16))
STATE_FORMAT = "E39 small-world life state, format 2"      # format 2 (E44): g1..g3, the world's kind, the learner, the list
assert all(array(c).itemsize == np.dtype(d).itemsize for _, c, d in EVENT_COLUMNS + LIST_COLUMNS)
assert (LW.BATCH, LW.TAUGHT_PER_BATCH, LW.EOS, LW.BOS) == (W.BATCH, W.TAUGHT_PER_BATCH, W.EOS, W.BOS)


# ---------------------------------------------------------------- hashes and provenance
def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tensors_sha(sd):
    """exp20_reference.tensors_sha: names, dtypes, shapes, values, in sorted order."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().contiguous().cpu()
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def opt_sha(opt):
    """The optimizer state: every tensor of every parameter's state (step, moments) in parameter order, and the groups' settings."""
    sd = opt.state_dict(); h = hashlib.sha256()
    for pid in sorted(sd["state"]):
        for k in sorted(sd["state"][pid]):
            v = T.as_tensor(sd["state"][pid][k]).detach().contiguous().cpu()
            h.update(f"{pid}|{k}|{v.dtype}|{tuple(v.shape)}".encode()); h.update(v.numpy().tobytes())
    h.update(json.dumps([{k: g[k] for k in sorted(g) if k != "params"} | {"params": g["params"]} for g in sd["param_groups"]],
                        sort_keys=True, default=str).encode())
    return h.hexdigest()


def arrays_sha(arrs):
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}".encode()); h.update(a.tobytes())
    return h.hexdigest()


def source_hashes():
    return {k: sha_file(p) for k, p in SOURCES.items()}


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        rel = [str(p.relative_to(MDIR)) for p in SOURCES.values()]
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=MDIR, capture_output=True, text=True).stdout.strip()
        return dict(commit=head, sources_modified_vs_commit=dirty.splitlines())
    except Exception as e:      # noqa: BLE001
        return dict(error=str(e))


def host():
    return dict(hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(), cores=os.cpu_count(),
                python=sys.version.split()[0], torch=T.__version__, numpy=np.__version__, torch_threads=T.get_num_threads())


def refuse_existing(*paths):
    for p in paths:
        assert not Path(p).exists(), f"refusing an existing path: {p}"


def write_json(path, obj):
    path = Path(path); tmp = Path(str(path) + ".tmp")
    refuse_existing(tmp)
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def is_pow2(n):
    return n > 0 and n & (n - 1) == 0


# ---------------------------------------------------------------- the world's task facts the runner needs
def symmetric(op):
    kind, c = op
    return kind != "lin" or c[0] == c[1]


class Facts:
    """The task facts the runner needs, for either world: E39's small world (sw_world) or E44's long-life world (lw_world, whose
    operations are named by strings and whose `mirror` and `needs_wrap` are its own)."""

    def __init__(self, world):
        self.lw = isinstance(world, LW.World)
        self.M = LW if self.lw else W
        self.world = world
        self.tasks = world.tasks
        self.taught = [set(t["taught"]) for t in world.tasks]
        self.asked = [set(t["asked"]) for t in world.tasks]
        self.sym = [(t["op"] != "sub") if self.lw else symmetric(t["op"]) for t in world.tasks]
        self.by_code = {bytes(t["code"]): t["id"] for t in world.tasks}

    def parse(self, prompt):
        """task, a, b from an item's own bytes: code(2) a(2) symbol d.. b(2) '='."""
        tid = self.by_code[bytes(prompt[:2])]
        t = self.tasks[tid]
        assert prompt[4] == t["symbol"] and prompt[-1] == ord("=")
        return tid, int(bytes(prompt[2:4])), int(bytes(prompt[-3:-1]))

    def mirror(self, tid, a, b):
        if self.lw:
            return self.world.mirror(self.tasks[tid], a, b)
        return int(self.sym[tid] and a != b and (b, a) in self.taught[tid])

    def wrap(self, tid, a, b):
        t = self.tasks[tid]
        return int(self.M.needs_wrap(t["op"], t["p"], a, b))

    def target(self, tid, a, b):
        t = self.tasks[tid]
        return list(f"{self.M.apply(t['op'], t['p'], a, b):02d}".encode()) + [W.EOS]


def gen_columns(g):
    """E44: the generated bytes of one asked item, in order; GEN_END for every position after the end symbol."""
    out, ended = [], False
    for x in g:
        out.append(GEN_END if ended else int(x))
        ended = ended or x == W.EOS
    assert len(out) == N.ANSWER_BYTES
    return out


# ---------------------------------------------------------------- E44: the learner and the list
def load_learner(path, H, U, arm):
    """E41's saved birth (weights only) into sw_net.SWNet(H, U); its hashes asserted equal to its receipt (path with .json)."""
    p = Path(path).resolve(); rp = p.with_suffix(".json")
    rec = json.loads(rp.read_text())
    want = {"naive": "B", "sweep": "S", "sweep-after-naive": "B"}[arm]
    assert rec["experiment"] == "E41" and rec["complete"] and rec["H"] == H and rec["U"] == U, "not a complete E41 birth at this width"
    assert rec["rule"] == want, f"arm {arm} lives from E41's rule {want}, not {rec['rule']}"
    m = N.SWNet(H, U); m.load_state_dict(T.load(p, weights_only=True))
    ts, fs = tensors_sha(m.state_dict()), sha_file(p)
    assert ts == rec["tensors_sha256"] and fs == rec["weights_sha256"], "the birth file's hashes differ from its receipt's"
    last = rec["curve"][-1]
    info = dict(file=str(p.relative_to(MDIR)), file_sha256=fs, tensors_sha256=ts, receipt=str(rp.relative_to(MDIR)), receipt_sha256=sha_file(rp),
                made_by="E41 (exp41_scale.py run)", rule=rec["rule"], local_in_birth=rec["local"], H=H, U=U, parameters=N.n_params(m),
                fraction=rec["fraction"], seed=rec["seed"], updates=rec["dose"], items_per_update=rec["batch"], taught_items=rec["dose"] * rec["batch"],
                final_row=dict(step=last["step"], A=last["A"], R=last["R"], R_counts=last["R_counts"]),
                reduction_sample_sha256=rec["reduction_sample_sha256"],
                loaded_into="sw_net.SWNet(H, U): arch20.NetRNN('gru')'s parameter names in its order, as exp41_scale.NetW; run by arch26.forward")
    return m, info


def e41_r_items(fraction, seed, want_sha):
    """The birth family's accuracy R items: E41's two samples (exp41_scale.reduction_sample, 1,000 held-out reduction pairs each of add89
    and sub89, rendered by exp11_cell.render with no letters), rebuilt here and asserted equal to the birth receipt's sample digests."""
    E = arch26.E
    out = []
    for j, t in enumerate(E.DEV[:2]):
        held = E.split(t, fraction, seed)[1]
        pairs = [(a, b) for a, b in held if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        smp = Random(f"e41|{seed}|{round(fraction * 100)}|{t[0]}").sample(sorted(pairs), 1000)
        assert hashlib.sha256(json.dumps([list(q) for q in smp]).encode()).hexdigest() == want_sha[E.NAME(*t)], "E41's R sample differs"
        for a, b in smp:
            prompt, y = E.render(t, a, b, 0, Random(0))
            out.append((BIRTH_LIST_TASKS[j], a, b, prompt, y))
    return out


def probe_items(world, family, per=100):
    """E44's list for one family: for add and then sub, the first `per` asked pairs in the split's order that need the reduction, no
    letters. Entries (task id, a, b, prompt, answer)."""
    out = []
    for t in world.groups[family][:2]:
        assert t["op"] in ("add", "sub")
        red = [(a, b) for a, b in t["asked"] if LW.needs_wrap(t["op"], t["p"], a, b)][:per]
        for a, b in red:
            prompt = list(t["code"] + f"{a:02d}".encode() + bytes([t["symbol"]]) + f"{b:02d}".encode() + b"=")
            out.append((t["id"], a, b, prompt, list(f"{LW.apply(t['op'], t['p'], a, b):02d}".encode()) + [LW.EOS]))
    return out


def list_digest(items):
    """The audit's digest of the list: SHA-256 over [id // 256, id % 256, a, b] of every entry in order."""
    h = hashlib.sha256()
    for tid, a, b, _, _ in items:
        h.update(bytes([tid // 256, tid % 256, a, b]))
    return h.hexdigest()


def answer_all(m, items, chunk=256):
    gens = []
    for i in range(0, len(items), chunk):
        gens += N.answer(m, [it[3] for it in items[i:i + chunk]])
    return gens


def decayed(m, factor):
    """A copy of m with every parameter but the byte table multiplied by `factor` (decay alone; no run)."""
    c = N.SWNet(m.l1.out_features, m.units)
    sd = {k: (v.clone() if k.startswith("book.") else v * T.tensor(factor, dtype=v.dtype)) for k, v in m.state_dict().items()}
    c.load_state_dict(sd)
    return c


def batches_with(world, rng, group):
    """sw_world.World.visit_batches with the generator given rather than Random("world|S|visit|v") (for --branch). The draws are the
    file's, in its order; sw_checks.py asserts that with the file's generator this gives the file's batches."""
    ts = world.groups[group]
    for _ in range(world.visit_updates):
        taught, asked = [], []
        for i in range(W.BATCH):
            t = ts[rng.randrange(len(ts))]
            pool = t["taught"] if i < W.TAUGHT_PER_BATCH else t["asked"]
            a, b = pool[rng.randrange(len(pool))]
            (taught if i < W.TAUGHT_PER_BATCH else asked).append((t["id"], a, b, *world.render(t, a, b, rng)))
        yield taught, asked


def solo_batches(world, tid, updates):
    """E39 revision 4, job L1a: one task alone. One generator for all the run's updates, Random("world|S|solo|<task id>"); each update
    draws 64 items in order, the first 48 taught: pair pool[randrange(len(pool))], then (world.render) randint(0, 4) for k and k letters
    by choice. No draw is made for the task."""
    t = world.tasks[tid]
    r = Random(f"world|{world.seed}|solo|{tid}")
    for _ in range(updates):
        taught, asked = [], []
        for i in range(W.BATCH):
            pool = t["taught"] if i < W.TAUGHT_PER_BATCH else t["asked"]
            a, b = pool[r.randrange(len(pool))]
            (taught if i < W.TAUGHT_PER_BATCH else asked).append((t["id"], a, b, *world.render(t, a, b, r)))
        yield taught, asked


def birth_tasks(world, groups):
    return sorted(t["id"] for g in groups for t in world.groups[g])


def birth_batches(world, groups):
    """E39 revision 4, job L1b: the birth's stream, without end. One generator for the whole birth, Random("world|S|birth"); each update
    draws 64 items in order, the first 48 taught: task tasks[randrange(len(tasks))] with tasks the birth groups' tasks in id order, then
    pair, k and letters as in a visit."""
    ts = [world.tasks[i] for i in birth_tasks(world, groups)]
    r = Random(f"world|{world.seed}|birth")
    while True:
        taught, asked = [], []
        for i in range(W.BATCH):
            t = ts[r.randrange(len(ts))]
            pool = t["taught"] if i < W.TAUGHT_PER_BATCH else t["asked"]
            a, b = pool[r.randrange(len(pool))]
            (taught if i < W.TAUGHT_PER_BATCH else asked).append((t["id"], a, b, *world.render(t, a, b, r)))
        yield taught, asked


def stream_digest(batches, updates):
    """Check 10: the world digest's rule over each update's taught then asked items, for the first `updates` updates."""
    h = hashlib.sha256()
    for _, (taught, asked) in zip(range(updates), batches):
        for it in taught + asked:
            h.update(stream_bytes(*it))
    return h.hexdigest()


def stream_bytes(tid, a, b, p, y):
    """One item as sw_world.digest hashes it."""
    return bytes([tid // 256, tid % 256, a, b, len(p)]) + bytes(p) + bytes(v % 256 for v in y)


def setting_tag(cfg, extra=None):
    parts = []
    if cfg["rho"] is not None:
        parts.append(f"rho{cfg['rho']:g}")
    lw = cfg.get("world_kind") == "lw"
    for k, short in TAGS.items():
        if k not in cfg:
            continue
        if cfg[k] != (LW_DEFAULTS if lw else DEFAULTS)[k]:
            parts.append(f"{short}{cfg[k]:g}")
    if cfg.get("in_turn"):
        parts.append("turn")
    if cfg.get("birth"):
        parts.append("birth" + "".join(str(g) for g in cfg["birth"]["groups"]))
    if extra:
        parts.append(extra)
    return "_".join(parts)


def life_stem(cfg):
    tag = setting_tag(cfg, cfg.get("tag"))
    pre = "lw" if cfg.get("world_kind") == "lw" else "sw"
    return f"{pre}_{cfg['arm']}_w{cfg['world']}_v{cfg['visits']}" + (f"_{tag}" if tag else "")


# ---------------------------------------------------------------- one life
class Life:
    def __init__(self, arm, world, visits, rho=None, H=64, U=64, visit_updates=W.VISIT_UPDATES, p_new=W.P_NEW, lr=0.001, wd=0.25, tag=None,
                 birth_groups=None, birth_cap=32768, birth_need=None, world_kind="sw", learner=None, in_turn=False, taught_share=None,
                 lowest=None, save_states=None):
        # sweep-after-naive: in E39 from a naive birth inside the world (Life.from_birth); in E44 from E41's backpropagation birth (--learner)
        assert arm in ARMS and (arm != "sweep-after-naive" or learner), arm
        assert world_kind in ("sw", "lw") and (world_kind == "lw") == (learner is not None), "--learner goes with --world lw"
        self.local, self.uses_replay = ARMS[arm]
        if self.uses_replay:
            assert rho is not None and 0 <= rho < 1 and float(rho * W.TAUGHT_PER_BATCH).is_integer(), "replay arms need rho with rho*48 whole"
        else:
            assert rho is None, "rho is for the replay arms"
        self.cfg = dict(arm=arm, world=world, visits=visits, rho=rho, H=H, U=U, visit_updates=visit_updates, p_new=p_new, lr=lr, wd=wd, tag=tag)
        if world_kind == "lw":
            assert not birth_groups, "E44's learner is born by E41, not inside the world"
            self.cfg.update(world_kind="lw", in_turn=bool(in_turn), taught_share=LW.TAUGHT_FRACTION if taught_share is None else taught_share,
                            lowest=LW.LOWEST if lowest is None else lowest)
        else:
            assert not in_turn and taught_share is None and lowest is None, "--in-turn, --taught-share and --lowest are the lw world's"
        if save_states is not None:
            self.cfg["save_states"] = sorted(set(save_states))
        if birth_groups:
            nb = 2 * len(birth_groups)
            self.cfg["birth"] = dict(groups=sorted(birth_groups), cap=birth_cap, need=birth_need if birth_need is not None else nb - 1, level=0.8, every=64)
        self.birth, self.birth_from = None, None
        self._build()
        if learner is not None:
            self.m, self.cfg["learner"] = load_learner(learner, H, U, arm)
        else:
            self.m = N.new_net(world, H, U)
        self.opt = N.make_opt(self.m, lr, wd)
        self.lv = {k: array(c) for k, c, _ in LIST_COLUMNS}
        self.families_taken, self.list_summary, self.list_cpu = [], [], 0.0
        self.buf, self.buf_bytes, self.offered = [], 0, 0
        self.rrng = Random(f"world|{world}|replay") if self.uses_replay else None
        nt = len(self.world.tasks)
        self.seen, self.replayed, self.last_answer = [0] * nt, [0] * nt, [None] * nt
        self.ev = {k: array(c) for k, c, _ in EVENT_COLUMNS}
        self.done_visits = self.done_updates = 0
        self.by_visit = []
        self.counters = dict(learned_items_verified=0, asking_checks=0, replayed_items=0, max_buffer_items=0, max_buffer_bytes=0,
                             first_replay_update=None)
        self.cpu_before, self.resumes, self.states = 0.0, [], []
        self.initial = dict(weights_tensors_sha256=tensors_sha(self.m.state_dict()))
        self.cpu0 = time.process_time()

    def _build(self):
        c = self.cfg
        if c.get("world_kind") == "lw":
            self.world = LW.World(c["world"], c["visit_updates"], c["p_new"], c["taught_share"], c["lowest"])
        else:
            self.world = W.World(c["world"], c["visit_updates"], c["p_new"])
        self.facts = Facts(self.world)
        self.k_rep = int(round(c["rho"] * W.TAUGHT_PER_BATCH)) if self.uses_replay else 0

    def is_lw(self):
        return self.cfg.get("world_kind") == "lw"

    def schedule(self, visits):
        return self.world.schedule(visits, in_turn=True) if self.cfg.get("in_turn") else self.world.schedule(visits)

    def wants_state(self, n):
        ss = self.cfg.get("save_states")
        return (is_pow2(n) or n == self.cfg["visits"]) if ss is None else n in ss

    # -------------------------------------------------- E44: the list
    def r_items(self):
        if not hasattr(self, "_r_items"):
            L = self.cfg["learner"]
            self._r_items = e41_r_items(L["fraction"], L["seed"], L["reduction_sample_sha256"])
        return self._r_items

    def _list_rows(self, m, items, at_visit, source, family, decay_updates, from_visit):
        gens = answer_all(m, items)
        right = 0
        for (tid, a, b, _, y), g in zip(items, gens):
            c = int(g == y); right += c
            for k, v in (("at_visit", at_visit), ("source", source), ("family", family), ("task", tid), ("a", a), ("b", b), ("correct", c),
                         ("decay_updates", decay_updates), ("weights_from_visit", from_visit)):
                self.lv[k].append(v)
            for k, v in zip(("g1", "g2", "g3"), gen_columns(g)):
                self.lv[k].append(v)
        self.list_summary.append(dict(at_visit=at_visit, source=source, family=family, n=len(items), right=right,
                                      share=right / len(items) if items else None, decay_updates=decay_updates, weights_from_visit=from_visit))

    def answer_list_before(self):
        """Before visit 0: the birth family's items, answered by the born network; and decay alone of the born weights for every visit."""
        c0 = time.process_time()
        before = (tensors_sha(self.m.state_dict()), opt_sha(self.opt))
        self._list_rows(self.m, self.r_items(), BEFORE_LIFE, 0, BIRTH_FAMILY, 0, BEFORE_LIFE)
        f = 1.0 - self.cfg["lr"] * self.cfg["wd"]; vu = self.cfg["visit_updates"]
        for k in range(self.cfg["visits"]):
            u = (k + 1) * vu
            self._list_rows(decayed(self.m, f ** u), self.r_items(), k, 1, BIRTH_FAMILY, u, BEFORE_LIFE)
        assert (tensors_sha(self.m.state_dict()), opt_sha(self.opt)) == before, "answering the list changed the weights or the optimizer"
        self.list_cpu += time.process_time() - c0

    def answer_list(self, s, decay=True):
        """At the end of visit s["visit"]: every family taken so far, the family just visited included, and the birth family, answered by
        the network as it stands. If this was the family's first visit and decay is on: decay alone from these weights for every later
        visit of the life (all visits are visit_updates long)."""
        c0 = time.process_time()
        v, g = s["visit"], s["group"]
        if g not in self.families_taken:
            self.families_taken.append(g)
            first = True
        else:
            first = False
        before = (tensors_sha(self.m.state_dict()), opt_sha(self.opt))
        for f in self.families_taken:
            self._list_rows(self.m, probe_items(self.world, f), v, 0, f, 0, v)
        self._list_rows(self.m, self.r_items(), v, 0, BIRTH_FAMILY, 0, v)
        if decay and first:
            fac = 1.0 - self.cfg["lr"] * self.cfg["wd"]; vu = self.cfg["visit_updates"]
            items = probe_items(self.world, g)
            for k in range(v + 1, self.cfg["visits"]):
                u = (k - v) * vu
                self._list_rows(decayed(self.m, fac ** u), items, k, 1, g, u, v)
        assert (tensors_sha(self.m.state_dict()), opt_sha(self.opt)) == before, "answering the list changed the weights or the optimizer"
        self.list_cpu += time.process_time() - c0

    def list_arrays(self):
        return {k: np.frombuffer(self.lv[k], dtype=d).copy() if len(self.lv[k]) else np.zeros(0, d) for k, _, d in LIST_COLUMNS}

    # -------------------------------------------------- state
    def capacity(self):
        P = N.n_params(self.m)
        im = ITEM_MAX_LW if self.is_lw() else ITEM_MAX
        return P, (4 * P) // im, 4 * P

    def events_arrays(self):
        return {k: np.frombuffer(self.ev[k], dtype=d).copy() if len(self.ev[k]) else np.zeros(0, d) for k, _, d in EVENT_COLUMNS}

    def state(self):
        bio = io.BytesIO(); np.savez_compressed(bio, **self.events_arrays())
        return dict(format=STATE_FORMAT, cfg=self.cfg, done_visits=self.done_visits, done_updates=self.done_updates,
                    model=self.m.state_dict(), opt=self.opt.state_dict(), buffer=self.buf, buffer_bytes=self.buf_bytes, offered=self.offered,
                    replay_rng=self.rrng.getstate() if self.rrng else None, seen=self.seen, replayed=self.replayed,
                    last_answer=self.last_answer, events_npz=bio.getvalue(), by_visit=self.by_visit, counters=self.counters,
                    cpu=self.cpu_before + time.process_time() - self.cpu0, resumes=self.resumes, states=self.states, initial=self.initial,
                    sources_sha256=source_hashes(), weights_tensors_sha256=tensors_sha(self.m.state_dict()), optimizer_sha256=opt_sha(self.opt),
                    birth=self.birth, birth_from=self.birth_from, list_npz=self._list_npz(), families_taken=self.families_taken,
                    list_summary=self.list_summary, list_cpu=self.list_cpu)

    def _list_npz(self):
        bio = io.BytesIO(); np.savez_compressed(bio, **self.list_arrays()); return bio.getvalue()

    @classmethod
    def load(cls, path):
        s = T.load(path, weights_only=False)
        assert s["format"] == STATE_FORMAT
        now = source_hashes()
        changed = [k for k in now if k != "smallworld/sw_score.py" and s["sources_sha256"].get(k) != now[k]]
        assert not changed, f"the sources differ from the state's (exact continuation not possible): {changed}"
        self = cls.__new__(cls)
        self.cfg = s["cfg"]
        self.local, self.uses_replay = ARMS[self.cfg["arm"]]
        self._build()
        self.m = N.SWNet(self.cfg["H"], self.cfg["U"]); self.m.load_state_dict(s["model"])
        self.opt = N.make_opt(self.m, self.cfg["lr"], self.cfg["wd"]); self.opt.load_state_dict(s["opt"])
        assert tensors_sha(self.m.state_dict()) == s["weights_tensors_sha256"] and opt_sha(self.opt) == s["optimizer_sha256"]
        self.buf, self.buf_bytes, self.offered = s["buffer"], s["buffer_bytes"], s["offered"]
        self.rrng = None
        if s["replay_rng"] is not None:
            self.rrng = Random(); self.rrng.setstate(s["replay_rng"])
        self.seen, self.replayed, self.last_answer = s["seen"], s["replayed"], s["last_answer"]
        f = np.load(io.BytesIO(s["events_npz"]))
        self.ev = {k: array(c, f[k].astype(d).tobytes()) for k, c, d in EVENT_COLUMNS}
        self.done_visits, self.done_updates = s["done_visits"], s["done_updates"]
        self.by_visit, self.counters, self.cpu_before = s["by_visit"], s["counters"], s["cpu"]
        self.resumes, self.states, self.initial = s["resumes"], s["states"], s["initial"]
        self.birth, self.birth_from = s.get("birth"), s.get("birth_from")
        g = np.load(io.BytesIO(s["list_npz"]))
        self.lv = {k: array(c, g[k].astype(d).tobytes()) for k, c, d in LIST_COLUMNS}
        self.families_taken, self.list_summary, self.list_cpu = s["families_taken"], s["list_summary"], s["list_cpu"]
        self.loaded_from = dict(path=str(path), file_sha256=sha_file(path), done_visits=self.done_visits)
        self.cpu0 = time.process_time()
        return self

    @classmethod
    def from_birth(cls, path):
        """sweep-after-naive: the naive arm's complete state at the end of its birth (weights, optimizer state, counts, the birth's events),
        continued under the sweep."""
        self = cls.load(path)
        assert self.cfg["arm"] == "naive" and self.done_visits == 0 and self.birth is not None, "needs a naive life's state at the end of its birth"
        self.birth_from = dict(self.loaded_from, arm="naive", weights_tensors_sha256=tensors_sha(self.m.state_dict()), optimizer_sha256=opt_sha(self.opt))
        self.cfg = dict(self.cfg, arm="sweep-after-naive")
        self.local, self.uses_replay = ARMS["sweep-after-naive"]
        self.states, self.resumes = [], []
        return self

    def run_birth(self):
        """Before visit 0: joint training on the birth groups' tasks, one generator for the whole birth (birth_batches). Checked after every
        64 updates; complete at the first check at which at least `need` of its tasks are at the criterion (share right over each task's
        asked items of the last 64 updates at least 0.8); at most `cap` updates. Asked items have visit = BIRTH_VISIT."""
        c = self.cfg["birth"]; tasks = birth_tasks(self.world, c["groups"])
        gen = birth_batches(self.world, c["groups"])
        hs, hl, h256 = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        c0, u0 = time.process_time(), self.done_updates
        right, n = {t: 0 for t in tasks}, {t: 0 for t in tasks}
        checks, losses, complete = [], [], False
        while self.done_updates - u0 < c["cap"]:
            taught, asked = next(gen)
            i = self.done_updates - u0
            if i < 256:
                for it in taught + asked:
                    h256.update(stream_bytes(*it))
            k0 = len(self.ev["correct"])
            lo, _ = self.update(BIRTH_VISIT, taught, asked, i % 1024 == 0, hs, hl)
            losses.append(lo)
            for j in range(k0, len(self.ev["correct"])):
                t = self.ev["task"][j]; right[t] += self.ev["correct"][j]; n[t] += 1
            if (i + 1) % c["every"] == 0:
                at = [t for t in tasks if n[t] and 5 * right[t] >= 4 * n[t]]      # share >= 0.8, exactly
                checks.append(dict(updates=i + 1, right=dict(right), n=dict(n), at_criterion=at))
                right, n = {t: 0 for t in tasks}, {t: 0 for t in tasks}
                if len(at) >= c["need"]:
                    complete = True
                    break
        B = self.done_updates - u0
        self.birth = dict(groups=c["groups"], tasks=tasks, cap=c["cap"], need=c["need"], updates=B, complete=complete, checks=checks,
                          end_shares={t: checks[-1]["right"][t] / checks[-1]["n"][t] if checks and checks[-1]["n"][t] else None for t in tasks},
                          stream_sha256=hs.hexdigest(), stream_first256_sha256=h256.hexdigest(), learned_sha256=hl.hexdigest(),
                          taught_items=W.TAUGHT_PER_BATCH * B, asked_items=(W.BATCH - W.TAUGHT_PER_BATCH) * B,
                          loss_per64=[float(np.mean(losses[k:k + 64])) for k in range(0, B, 64)], cpu=time.process_time() - c0,
                          generator=f"world|{self.cfg['world']}|birth")
        return self.birth

    def save_state(self, path):
        refuse_existing(path, str(path) + ".tmp")
        T.save(self.state(), str(path) + ".tmp"); os.replace(str(path) + ".tmp", path)
        self.states.append(dict(visits=self.done_visits, path=Path(path).name, file_sha256=sha_file(path),
                                weights_tensors_sha256=tensors_sha(self.m.state_dict()), optimizer_sha256=opt_sha(self.opt)))

    # -------------------------------------------------- one update, one visit
    def update(self, v, taught, asked, check_ask, hs, hl):
        m, F = self.m, self.facts
        if check_ask:
            before = (tensors_sha(m.state_dict()), opt_sha(self.opt))
        gen = N.answer(m, [p for _, _, _, p, _ in asked])                           # asked items: answered before the update
        if check_ask:
            assert (tensors_sha(m.state_dict()), opt_sha(self.opt)) == before, "answering changed the weights or the optimizer state"
            self.counters["asking_checks"] += 1
        u, ev = self.done_updates, self.ev
        for (tid, a, b, p, y), g in zip(asked, gen):
            ev["update"].append(u); ev["visit"].append(v); ev["task"].append(tid); ev["correct"].append(int(g == y))
            ev["blind"].append(int(self.last_answer[tid] == y)); ev["wrap"].append(F.wrap(tid, a, b)); ev["mirror"].append(F.mirror(tid, a, b))
            ev["seen"].append(self.seen[tid]); ev["replayed"].append(self.replayed[tid])
            for k, x in zip(("g1", "g2", "g3"), gen_columns(g)):
                ev[k].append(x)
        new = [(p, y) for _, _, _, p, y in taught]
        rep = []
        if self.uses_replay:
            if len(self.buf) >= self.k_rep:
                if self.counters["first_replay_update"] is None:
                    self.counters["first_replay_update"] = u
                rep = [self.buf[self.rrng.randrange(len(self.buf))] for _ in range(self.k_rep)]
                learn = new[:W.TAUGHT_PER_BATCH - self.k_rep] + rep
            else:
                learn = new
            _, cap_items, cap_bytes = self.capacity()
            for it in new:                                                            # offered in batch order, from the same generator
                self.offered += 1
                if len(self.buf) < cap_items:
                    self.buf.append(it); self.buf_bytes += len(it[0]) + len(it[1])
                else:
                    j = self.rrng.randrange(self.offered)
                    if j < cap_items:
                        self.buf_bytes += len(it[0]) + len(it[1]) - len(self.buf[j][0]) - len(self.buf[j][1])
                        self.buf[j] = it
            assert len(self.buf) <= cap_items and self.buf_bytes <= cap_bytes
            self.counters["max_buffer_items"] = max(self.counters["max_buffer_items"], len(self.buf))
            self.counters["max_buffer_bytes"] = max(self.counters["max_buffer_bytes"], self.buf_bytes)
        else:
            learn = new
        assert len(learn) == W.TAUGHT_PER_BATCH
        for p, y in learn:                                                           # nothing learned has an asked pair
            tid, a, b = F.parse(p)
            assert (a, b) in F.taught[tid] and (a, b) not in F.asked[tid] and y == F.target(tid, a, b)
        self.counters["learned_items_verified"] += len(learn)
        loss = N.step(m, self.opt, learn, self.local)
        for tid, a, b, p, y in taught:
            self.seen[tid] += 1; self.last_answer[tid] = y
        for p, y in rep:
            self.replayed[F.parse(p)[0]] += 1
        self.counters["replayed_items"] += len(rep)
        for it in taught + asked:
            hs.update(stream_bytes(*it))
        hl.update(b"|")
        for p, y in learn:
            hl.update(bytes([len(p), len(y)]) + bytes(p) + bytes(x % 256 for x in y))
        self.done_updates += 1
        return loss, len(rep)

    def visit(self, s, batches, max_updates=None, life_hash=None):
        c0, w0 = time.process_time(), time.time()
        hs, hl = hashlib.sha256(), hashlib.sha256()
        losses, reps = [], []
        for i, (taught, asked) in enumerate(batches):
            if max_updates is not None and i >= max_updates:
                break
            if life_hash is not None:
                for it in taught + asked:
                    life_hash.update(stream_bytes(*it))
            lo, r = self.update(s["visit"], taught, asked, i == 0, hs, hl)
            losses.append(lo); reps.append(r)
        self.done_visits += 1
        rec = dict(s, updates=len(losses), stream_sha256=hs.hexdigest(), learned_sha256=hl.hexdigest(), loss_mean=float(np.mean(losses)),
                   loss_first=losses[0], loss_last=losses[-1], replayed_items=int(sum(reps)), cpu=time.process_time() - c0, wall=time.time() - w0)
        self.by_visit.append(rec)
        return rec

    # -------------------------------------------------- outputs
    def ledger(self):
        P, cap_items, cap_bytes = self.capacity()
        sd = self.opt.state_dict()["state"]
        moments = sum(int(T.as_tensor(st[k]).numel()) for st in sd.values() for k in ("exp_avg", "exp_avg_sq") if k in st)
        steps = sum(1 for st in sd.values() if "step" in st)
        return dict(stored_numbers=dict(weights=P, optimizer_moments=moments, optimizer_step_counters=steps, anchor=0),
                    buffer=dict(used=self.uses_replay, capacity_items=cap_items if self.uses_replay else 0,
                                capacity_bytes=cap_bytes if self.uses_replay else 0, items_at_end=len(self.buf), bytes_at_end=self.buf_bytes,
                                max_items=self.counters["max_buffer_items"], max_bytes=self.counters["max_buffer_bytes"], offered=self.offered),
                    updates=self.done_updates, stream_items_per_update=W.BATCH, taught_per_update=W.TAUGHT_PER_BATCH,
                    asked_per_update=W.BATCH - W.TAUGHT_PER_BATCH, learned_per_update=W.TAUGHT_PER_BATCH, replayed_per_update=self.k_rep,
                    new_learned_per_update_once_replaying=W.TAUGHT_PER_BATCH - self.k_rep,
                    birth=None if not self.birth else dict(updates=self.birth["updates"], taught_items=self.birth["taught_items"],
                                                           asked_items=self.birth["asked_items"], cpu=self.birth["cpu"],
                                                           inherited_from_naive=self.birth_from is not None),
                    life_updates=self.done_updates - (self.birth["updates"] if self.birth else 0),
                    learner=None if "learner" not in self.cfg else dict(made_by=self.cfg["learner"]["made_by"], file=self.cfg["learner"]["file"],
                                                                        file_sha256=self.cfg["learner"]["file_sha256"], rule=self.cfg["learner"]["rule"],
                                                                        updates=self.cfg["learner"]["updates"], items_per_update=self.cfg["learner"]["items_per_update"],
                                                                        taught_items=self.cfg["learner"]["taught_items"],
                                                                        note="E41's birth on E26's first family (mod 89, half its pairs), before this life"))

    def receipt(self, kind, cmd, events_path, ev_arrays, extra):
        m = self.m
        if self.is_lw():
            wd_ = dict(kind="lw", file="smallworld/lw_world.py", seed=self.cfg["world"], visit_updates=self.cfg["visit_updates"], p_new=self.cfg["p_new"],
                       taught_share=self.cfg["taught_share"], lowest=self.cfg["lowest"], in_turn=self.cfg["in_turn"], probe_families=LW.PROBE_FAMILIES,
                       tasks=len(self.world.tasks), families=len(self.world.groups))
        else:
            wd_ = dict(seed=self.cfg["world"], visit_updates=self.cfg["visit_updates"], p_new=self.cfg["p_new"],
                       probe_groups=W.PROBE_GROUPS, tasks=len(self.world.tasks), groups=len(self.world.groups))
        return dict(experiment="E44" if self.is_lw() else "E39", job="F1 long-life world" if self.is_lw() else "L0/L1 small world", kind=kind,
                    settings=self.cfg, local=self.local, replay=self.uses_replay,
                    replay_items_per_update=self.k_rep,
                    model=dict(cls=f"{type(m).__module__}.{type(m).__name__}", H=self.cfg["H"], U=self.cfg["U"], parameters=N.n_params(m),
                               shapes={k: list(v.shape) for k, v in m.state_dict().items()}, initial=self.initial),
                    optimizer=dict(name="AdamW", lr=self.cfg["lr"], weight_decay=self.cfg["wd"], betas=[.9, .98], groups=N.opt_groups(m, self.opt)),
                    update="sw_net.step (exp26_candidate.step's text): exp45.batch_tensor, arch26.forward(m, x, local, True), answer-masked "
                           "cross entropy, zero_grad, backward, one AdamW step",
                    answering="sw_net.answer: greedy, 3 bytes, no_grad, through arch26.forward; right if all three are the answer",
                    world=wd_, learner=self.cfg.get("learner"),
                    provenance=dict(command=cmd, host=host(), git=git_state(), sources_sha256=source_hashes(),
                                    env={k: v for k, v in os.environ.items() if k.startswith(("OMP_", "MKL_", "OPENBLAS_", "E16_", "E11_", "EXP45_", "PYTHONHASHSEED"))}),
                    ledger=self.ledger(), counters=self.counters,
                    events=dict(file=Path(events_path).name, file_sha256=sha_file(events_path), arrays_sha256=arrays_sha(ev_arrays),
                                entries=int(len(ev_arrays["update"])), columns=[k for k, _, _ in EVENT_COLUMNS]),
                    final=dict(visits=self.done_visits, updates=self.done_updates, weights_tensors_sha256=tensors_sha(m.state_dict()),
                               optimizer_sha256=opt_sha(self.opt)),
                    cpu=dict(total=self.cpu_before + time.process_time() - self.cpu0, per_visit=[r["cpu"] for r in self.by_visit],
                             list_total=self.list_cpu, list_per_visit=[r.get("list_cpu") for r in self.by_visit],
                             mean_per_visit=float(np.mean([r["cpu"] for r in self.by_visit])) if self.by_visit else None,
                             birth=self.birth["cpu"] if self.birth else None,
                             peak_rss_kib=__import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss,
                             note=None if not self.birth_from else "total includes the naive arm's birth CPU, inherited with its state"),
                    birth=self.birth, birth_from=self.birth_from,
                    learned_sha256=hashlib.sha256("".join(r["learned_sha256"] for r in self.by_visit).encode()).hexdigest(),
                    by_visit=self.by_visit, resumes=self.resumes, states=self.states, complete=True, **extra)


# ---------------------------------------------------------------- commands
def list_record(life, lpath):
    """Writes the list's answers (<stem>_list.npz) and returns the receipt's entries for it."""
    la = life.list_arrays()
    np.savez_compressed(lpath, **la)
    fams = list(life.families_taken)
    return dict(list=dict(file=Path(lpath).name, file_sha256=sha_file(lpath), arrays_sha256=arrays_sha(la), entries=int(len(la["task"])),
                          columns=[k for k, _, _ in LIST_COLUMNS], families_taken=fams,
                          items_sha256=list_digest([it for f in fams for it in probe_items(life.world, f)]),
                          r_items=len(life.r_items()), decay_factor_per_update=1.0 - life.cfg["lr"] * life.cfg["wd"],
                          note="source 0: the network at the end of at_visit (65535: before visit 0); source 1: decay alone, the weights at the end "
                               "of weights_from_visit (65535: the born weights) with every parameter but the byte table multiplied by "
                               "decay_factor_per_update ** decay_updates; family -1 is the birth family (tasks 65534 add89, 65535 sub89)"),
                list_summary=life.list_summary)


def run_life(life, out, cmd, stop_after=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stem = life_stem(life.cfg); V = life.cfg["visits"]
    rpath, epath, lpath = out / f"{stem}.json", out / f"{stem}.npz", out / f"{stem}_list.npz"
    refuse_existing(rpath, epath, lpath)
    life.cpu0 = time.process_time()
    sched = life.schedule(V)
    lh = hashlib.sha256()                                       # the stream as sw_world.digest hashes it, over the whole life
    f0 = time.process_time()
    for s in sched[:life.done_visits]:                          # a resumed life: the earlier visits' stream, regenerated (no network)
        hv = hashlib.sha256()
        for taught, asked in life.world.visit_batches(s["visit"], s["group"]):
            for it in taught + asked:
                b = stream_bytes(*it); lh.update(b); hv.update(b)
        assert hv.hexdigest() == life.by_visit[s["visit"]]["stream_sha256"], f"visit {s['visit']}: regenerated stream differs from the state's"
    if life.done_visits:
        life.resumes.append(dict(life.loaded_from, wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), command=cmd,
                                 stream_regeneration_cpu=time.process_time() - f0))
    if life.cfg.get("birth") and life.done_visits == 0:
        if life.birth is None:
            b = life.run_birth()
            print(f"{stem} birth: {b['updates']} updates, complete {b['complete']}, end shares "
                  f"{ {t: round(x, 3) if x is not None else None for t, x in b['end_shares'].items()} } cpu {b['cpu']:.1f}s", flush=True)
        life.save_state(out / f"{stem}_state_v0.pt")             # the state at visit 0: the end of the birth
        if stop_after == 0:
            return None
    if life.is_lw() and life.done_visits == 0 and not life.list_summary:
        life.answer_list_before()
    for s in sched[life.done_visits:]:
        rec = life.visit(s, life.world.visit_batches(s["visit"], s["group"]), life_hash=lh)
        n = life.done_visits
        if life.is_lw():
            c0 = life.list_cpu; life.answer_list(s); rec["list_cpu"] = life.list_cpu - c0
        right = float(np.frombuffer(life.ev['correct'], dtype=np.uint8)[-16 * rec['updates']:].mean())     # no view may outlive the line
        print(f"{stem} visit {s['visit']} group {s['group']} new {s['new']} absence {s['absence']} loss {rec['loss_mean']:.4f} "
              f"right {right:.3f} cpu {rec['cpu']:.1f}s", flush=True)
        if life.wants_state(n) or (stop_after is not None and n == stop_after):
            life.save_state(out / f"{stem}_state_v{n}.pt")
        if stop_after is not None and n == stop_after:
            print(f"{stem}: stopped after visit {n - 1} (state {stem}_state_v{n}.pt)", flush=True)
            return None
    arrs = life.events_arrays()
    np.savez_compressed(epath, **arrs)
    extra = dict(schedule=sched, stream_sha256=lh.hexdigest(),
                 stream_note="sha256 over every stream item in order, as sw_world.digest; equals "
                             "sw_world.digest(World(world, visit_updates, p_new), visits)")
    if life.is_lw():
        extra.update(stream_note="sha256 over every stream item in order, as lw_world.digest; equals lw_world.digest(World(world, visit_updates, "
                                 "p_new, taught_share, lowest), visits, in_turn)", **list_record(life, lpath))
    rec = life.receipt("life", cmd, epath, arrs, extra)
    write_json(rpath, rec)
    print(f"{stem}: complete; receipt {rpath.name}", flush=True)
    return rec


def run_branch(state_path, groups, out, cmd):
    life = Life.load(state_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    n0, S = life.done_visits, life.cfg["world"]
    stem = life_stem(life.cfg) + f"_br{n0}_g" + "-".join(str(g) for g in groups)
    rpath, epath, lpath = out / f"{stem}.json", out / f"{stem}.npz", out / f"{stem}_list.npz"
    refuse_existing(rpath, epath, lpath, out / f"{stem}_state_end.pt")
    life.cpu0 = time.process_time(); life.cpu_before = 0.0
    trunk = dict(by_visit=len(life.by_visit), states=life.states, families_taken=list(life.families_taken))
    last = {}
    for r in life.by_visit:                                     # the trunk's visits (E44: an in-turn life's schedule is not world.schedule)
        last[r["group"]] = r["visit"]
    life.ev = {k: array(c) for k, c, _ in EVENT_COLUMNS}; life.by_visit = []; life.states = []
    life.lv = {k: array(c) for k, c, _ in LIST_COLUMNS}; life.list_summary = []; life.list_cpu = 0.0
    n_probe = LW.PROBE_FAMILIES if life.is_lw() else W.PROBE_GROUPS
    lh = hashlib.sha256()
    for k, g in enumerate(groups):
        assert 0 <= g < len(life.world.groups)
        v = n0 + k
        s = dict(visit=v, group=g, new=g not in last, absence=None if g not in last else v - last[g], branch_k=k,
                 generator=f"world|{S}|branch|{n0}|{k}", probe=g >= len(life.world.groups) - n_probe)
        rec = life.visit(s, batches_with(life.world, Random(f"world|{S}|branch|{n0}|{k}"), g), life_hash=lh)
        if life.is_lw():
            c0 = life.list_cpu; life.answer_list(s, decay=False); rec["list_cpu"] = life.list_cpu - c0
        last[g] = v
    arrs = life.events_arrays()
    np.savez_compressed(epath, **arrs)
    life.save_state(out / f"{stem}_state_end.pt")
    extra = dict(branch=dict(from_state=life.loaded_from, groups=groups, trunk=trunk), stream_sha256=lh.hexdigest())
    if life.is_lw():
        extra.update(list_record(life, lpath))
    rec = life.receipt("branch", cmd, epath, arrs, extra)
    write_json(rpath, rec)
    print(f"{stem}: complete", flush=True)
    return rec


def run_solo(arm, world, tid, H, updates, out, cmd):
    """E39 revision 4, job L1a: one task alone from the network's initial values (T.manual_seed(world) then the constructor), `updates`
    updates of the solo stream; asked items answered before each update; `visit` = 0 throughout."""
    assert arm in ("naive", "sweep")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stem = f"sw_solo_{arm}_w{world}_t{tid}_h{H}"
    rpath, epath, spath = out / f"{stem}.json", out / f"{stem}.npz", out / f"{stem}_state_end.pt"
    refuse_existing(rpath, epath, spath)
    life = Life(arm, world, 0, H=H, U=H)
    t = life.world.tasks[tid]
    hs, hl, h256 = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    losses, life.cpu0 = [], time.process_time()
    for i, (taught, asked) in enumerate(solo_batches(life.world, tid, updates)):
        if i < 256:
            for it in taught + asked:
                h256.update(stream_bytes(*it))
        lo, _ = life.update(0, taught, asked, i % 1024 == 0, hs, hl)
        losses.append(lo)
        if (i + 1) % 2048 == 0:
            c = float(np.frombuffer(life.ev["correct"], dtype=np.uint8)[-16 * 64:].mean())     # no view may outlive the line (the array grows)
            print(f"{stem} update {i + 1} loss {np.mean(losses[-64:]):.4f} right(last 64 updates) {c:.3f}", flush=True)
    arrs = life.events_arrays()
    np.savez_compressed(epath, **arrs)
    q = S.solo_quantities(arrs, tid, losses)
    life.save_state(spath)
    rec = life.receipt("solo", cmd, epath, arrs, dict(
        solo=dict(task=tid, op=t["op"][0] if t["op"][0] != "lin" else list(t["op"][1]), p=t["p"], symbol=chr(t["symbol"]), code=bytes(t["code"]).decode(),
                  taught_pairs=len(t["taught"]), asked_pairs=len(t["asked"]), updates=updates, generator=f"world|{world}|solo|{tid}"),
        stream_sha256=hs.hexdigest(), stream_first256_sha256=h256.hexdigest(), solo_learned_sha256=hl.hexdigest(),
        loss_per64=[float(np.mean(losses[k:k + 64])) for k in range(0, len(losses), 64)], quantities=q))
    rec["job"] = "L1a solo"
    write_json(rpath, rec)
    print(f"{stem}: complete; to criterion {q['updates_to_criterion']}, to 0.5 {q['updates_to_half']}, last 256 {q['last256']['all']['share']:.3f}, "
          f"loss {q['loss_last64']:.4f}", flush=True)
    return rec


def render(world, tid, a, b, letters=""):
    t = world.tasks[tid]
    return list(t["code"] + f"{a:02d}".encode() + bytes([t["symbol"]]) + letters.encode() + f"{b:02d}".encode() + b"=")


def run_ask(state_path, items, to, cmd):
    refuse_existing(to)
    life = Life.load(state_path)
    F, w = life.facts, life.world
    rows = []
    for it in items:
        tid, a, b = int(it[0]), int(it[1]), int(it[2]); letters = it[3] if len(it) > 3 else ""
        assert (a, b) in F.asked[tid], f"({a}, {b}) is not an asked pair of task {tid}"
        assert all(ch in W.LETTERS.decode() for ch in letters) and len(letters) <= W.KMAX
        rows.append((tid, a, b, letters, render(w, tid, a, b, letters), F.target(tid, a, b)))
    before = (tensors_sha(life.m.state_dict()), opt_sha(life.opt))
    gens = []
    for i in range(0, len(rows), W.BATCH):
        gens += N.answer(life.m, [r[4] for r in rows[i:i + W.BATCH]])
    after = (tensors_sha(life.m.state_dict()), opt_sha(life.opt))
    assert before == after
    out = dict(experiment="E39", kind="ask", state=life.loaded_from, command=cmd, host=host(), sources_sha256=source_hashes(),
               weights_tensors_sha256=before[0], optimizer_sha256=before[1], unchanged_after=before == after,
               items=[dict(task=tid, a=a, b=b, letters=l, prompt=bytes(p).decode(), answer=y, generated=g, correct=g == y)
                      for (tid, a, b, l, p, y), g in zip(rows, gens)])
    out["share_right"] = float(np.mean([r["correct"] for r in out["items"]])) if rows else None
    write_json(to, out)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("life", "branch", "ask", "solo"))
    ap.add_argument("--birth-groups"); ap.add_argument("--birth-cap", type=int, default=32768); ap.add_argument("--birth-need", type=int)
    ap.add_argument("--birth-from"); ap.add_argument("--updates", type=int, default=16384)
    ap.add_argument("pos", nargs="*")
    ap.add_argument("--rho", type=float); ap.add_argument("--H", type=int, default=None); ap.add_argument("--U", type=int, default=None)
    ap.add_argument("--visit-updates", type=int, default=None); ap.add_argument("--p-new", type=float, default=None)
    ap.add_argument("--world", choices=("sw", "lw"), default="sw"); ap.add_argument("--learner"); ap.add_argument("--in-turn", action="store_true")
    ap.add_argument("--taught-share", type=float); ap.add_argument("--lowest", type=int); ap.add_argument("--save-states")
    ap.add_argument("--lr", type=float, default=0.001); ap.add_argument("--wd", type=float, default=0.25); ap.add_argument("--tag")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--stop-after", type=int); ap.add_argument("--resume")
    ap.add_argument("--groups"); ap.add_argument("--items"); ap.add_argument("--to")
    a = ap.parse_args(argv)
    cmd = " ".join(["python3", "sw_run.py", *argv])
    dflt = LW_DEFAULTS if a.world == "lw" else DEFAULTS
    a.H = dflt["H"] if a.H is None else a.H
    a.U = (dflt["U"] if a.world == "lw" else a.H) if a.U is None else a.U
    a.visit_updates = dflt["visit_updates"] if a.visit_updates is None else a.visit_updates
    a.p_new = dflt["p_new"] if a.p_new is None else a.p_new
    ss = None if a.save_states is None else [int(x) for x in a.save_states.split(",")]
    if a.cmd == "life":
        if a.resume:
            assert not a.pos and not a.birth_from, "--resume takes the life's settings from the state"
            life = Life.load(a.resume)
        elif a.birth_from:
            assert not a.pos, "--birth-from takes the life's settings from the naive state at the end of its birth"
            life = Life.from_birth(a.birth_from)
        else:
            arm, world, visits = a.pos[0], int(a.pos[1]), int(a.pos[2])
            bg = [int(g) for g in a.birth_groups.split(",")] if a.birth_groups else None
            life = Life(arm, world, visits, a.rho, a.H, a.U, a.visit_updates, a.p_new, a.lr, a.wd, a.tag,
                        bg, a.birth_cap, a.birth_need, a.world, a.learner, a.in_turn, a.taught_share, a.lowest, ss)
        run_life(life, a.out, cmd, a.stop_after)
    elif a.cmd == "solo":
        assert a.world == "sw", "solo runs are E39's (the small world)"
        run_solo(a.pos[0], int(a.pos[1]), int(a.pos[2]), a.H, a.updates, a.out, cmd)
    elif a.cmd == "branch":
        run_branch(a.pos[0], [int(g) for g in a.groups.split(",")], a.out, cmd)
    else:
        run_ask(a.pos[0], json.loads(Path(a.items).read_text()), a.to, cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
