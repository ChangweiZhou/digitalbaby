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
import sw_net as N                    # noqa: E402

T.set_num_threads(1)
OUT = MDIR / "exp45" / "e39"
ARMS = {"naive": (False, False), "sweep": (True, False), "replay": (False, True), "sweep+replay": (True, True)}   # arm -> (local, replay)
DEFAULTS = dict(H=64, U=64, visit_updates=W.VISIT_UPDATES, p_new=W.P_NEW, lr=0.001, wd=0.25)
TAGS = dict(H="h", U="u", visit_updates="vu", p_new="pn", lr="lr", wd="wd")
ITEM_MAX = 15                         # bytes of the longest item: 12 of prompt, 2 digits and the end symbol
SOURCES = {"smallworld/" + s: HERE / s for s in ("sw_world.py", "sw_net.py", "sw_run.py", "sw_score.py")}
SOURCES.update({s: MDIR / s for s in ("arch26.py", "arch20.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py")})
EVENT_COLUMNS = (("update", "I", np.uint32), ("visit", "H", np.uint16), ("task", "H", np.uint16), ("correct", "B", np.uint8),
                 ("blind", "B", np.uint8), ("wrap", "B", np.uint8), ("mirror", "B", np.uint8), ("seen", "I", np.uint32),
                 ("replayed", "I", np.uint32))
STATE_FORMAT = "E39 small-world life state, format 1"
assert all(array(c).itemsize == np.dtype(d).itemsize for _, c, d in EVENT_COLUMNS)


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
    def __init__(self, world):
        self.tasks = world.tasks
        self.taught = [set(t["taught"]) for t in world.tasks]
        self.asked = [set(t["asked"]) for t in world.tasks]
        self.sym = [symmetric(t["op"]) for t in world.tasks]
        self.by_code = {bytes(t["code"]): t["id"] for t in world.tasks}

    def parse(self, prompt):
        """task, a, b from an item's own bytes: code(2) a(2) symbol d.. b(2) '='."""
        tid = self.by_code[bytes(prompt[:2])]
        t = self.tasks[tid]
        assert prompt[4] == t["symbol"] and prompt[-1] == ord("=")
        return tid, int(bytes(prompt[2:4])), int(bytes(prompt[-3:-1]))

    def mirror(self, tid, a, b):
        return int(self.sym[tid] and a != b and (b, a) in self.taught[tid])

    def wrap(self, tid, a, b):
        t = self.tasks[tid]
        return int(W.needs_wrap(t["op"], t["p"], a, b))

    def target(self, tid, a, b):
        t = self.tasks[tid]
        return list(f"{W.apply(t['op'], t['p'], a, b):02d}".encode()) + [W.EOS]


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
    for k, short in TAGS.items():
        if cfg[k] != DEFAULTS[k]:
            parts.append(f"{short}{cfg[k]:g}")
    if extra:
        parts.append(extra)
    return "_".join(parts)


def life_stem(cfg):
    tag = setting_tag(cfg, cfg.get("tag"))
    return f"sw_{cfg['arm']}_w{cfg['world']}_v{cfg['visits']}" + (f"_{tag}" if tag else "")


# ---------------------------------------------------------------- one life
class Life:
    def __init__(self, arm, world, visits, rho=None, H=64, U=64, visit_updates=W.VISIT_UPDATES, p_new=W.P_NEW, lr=0.001, wd=0.25, tag=None):
        assert arm in ARMS, arm
        self.local, self.uses_replay = ARMS[arm]
        if self.uses_replay:
            assert rho is not None and 0 <= rho < 1 and float(rho * W.TAUGHT_PER_BATCH).is_integer(), "replay arms need rho with rho*48 whole"
        else:
            assert rho is None, "rho is for the replay arms"
        self.cfg = dict(arm=arm, world=world, visits=visits, rho=rho, H=H, U=U, visit_updates=visit_updates, p_new=p_new, lr=lr, wd=wd, tag=tag)
        self._build()
        self.m = N.new_net(world, H, U)
        self.opt = N.make_opt(self.m, lr, wd)
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
        self.world = W.World(c["world"], c["visit_updates"], c["p_new"])
        self.facts = Facts(self.world)
        self.k_rep = int(round(c["rho"] * W.TAUGHT_PER_BATCH)) if self.uses_replay else 0

    # -------------------------------------------------- state
    def capacity(self):
        P = N.n_params(self.m)
        return P, (4 * P) // ITEM_MAX, 4 * P

    def events_arrays(self):
        return {k: np.frombuffer(self.ev[k], dtype=d).copy() if len(self.ev[k]) else np.zeros(0, d) for k, _, d in EVENT_COLUMNS}

    def state(self):
        bio = io.BytesIO(); np.savez_compressed(bio, **self.events_arrays())
        return dict(format=STATE_FORMAT, cfg=self.cfg, done_visits=self.done_visits, done_updates=self.done_updates,
                    model=self.m.state_dict(), opt=self.opt.state_dict(), buffer=self.buf, buffer_bytes=self.buf_bytes, offered=self.offered,
                    replay_rng=self.rrng.getstate() if self.rrng else None, seen=self.seen, replayed=self.replayed,
                    last_answer=self.last_answer, events_npz=bio.getvalue(), by_visit=self.by_visit, counters=self.counters,
                    cpu=self.cpu_before + time.process_time() - self.cpu0, resumes=self.resumes, states=self.states, initial=self.initial,
                    sources_sha256=source_hashes(), weights_tensors_sha256=tensors_sha(self.m.state_dict()), optimizer_sha256=opt_sha(self.opt))

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
        self.loaded_from = dict(path=str(path), file_sha256=sha_file(path), done_visits=self.done_visits)
        self.cpu0 = time.process_time()
        return self

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
                    new_learned_per_update_once_replaying=W.TAUGHT_PER_BATCH - self.k_rep)

    def receipt(self, kind, cmd, events_path, ev_arrays, extra):
        m = self.m
        return dict(experiment="E39", job="L0/L1 small world", kind=kind, settings=self.cfg, local=self.local, replay=self.uses_replay,
                    replay_items_per_update=self.k_rep,
                    model=dict(cls=f"{type(m).__module__}.{type(m).__name__}", H=self.cfg["H"], U=self.cfg["U"], parameters=N.n_params(m),
                               shapes={k: list(v.shape) for k, v in m.state_dict().items()}, initial=self.initial),
                    optimizer=dict(name="AdamW", lr=self.cfg["lr"], weight_decay=self.cfg["wd"], betas=[.9, .98], groups=N.opt_groups(m, self.opt)),
                    update="sw_net.step (exp26_candidate.step's text): exp45.batch_tensor, arch26.forward(m, x, local, True), answer-masked "
                           "cross entropy, zero_grad, backward, one AdamW step",
                    answering="sw_net.answer: greedy, 3 bytes, no_grad, through arch26.forward; right if all three are the answer",
                    world=dict(seed=self.cfg["world"], visit_updates=self.cfg["visit_updates"], p_new=self.cfg["p_new"],
                               probe_groups=W.PROBE_GROUPS, tasks=len(self.world.tasks), groups=len(self.world.groups)),
                    provenance=dict(command=cmd, host=host(), git=git_state(), sources_sha256=source_hashes(),
                                    env={k: v for k, v in os.environ.items() if k.startswith(("OMP_", "MKL_", "OPENBLAS_", "E16_", "E11_", "EXP45_", "PYTHONHASHSEED"))}),
                    ledger=self.ledger(), counters=self.counters,
                    events=dict(file=Path(events_path).name, file_sha256=sha_file(events_path), arrays_sha256=arrays_sha(ev_arrays),
                                entries=int(len(ev_arrays["update"])), columns=[k for k, _, _ in EVENT_COLUMNS]),
                    final=dict(visits=self.done_visits, updates=self.done_updates, weights_tensors_sha256=tensors_sha(m.state_dict()),
                               optimizer_sha256=opt_sha(self.opt)),
                    cpu=dict(total=self.cpu_before + time.process_time() - self.cpu0, per_visit=[r["cpu"] for r in self.by_visit],
                             mean_per_visit=float(np.mean([r["cpu"] for r in self.by_visit]))),
                    learned_sha256=hashlib.sha256("".join(r["learned_sha256"] for r in self.by_visit).encode()).hexdigest(),
                    by_visit=self.by_visit, resumes=self.resumes, states=self.states, complete=True, **extra)


# ---------------------------------------------------------------- commands
def run_life(life, out, cmd, stop_after=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stem = life_stem(life.cfg); V = life.cfg["visits"]
    rpath, epath = out / f"{stem}.json", out / f"{stem}.npz"
    refuse_existing(rpath, epath)
    life.cpu0 = time.process_time()
    sched = life.world.schedule(V)
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
    for s in sched[life.done_visits:]:
        rec = life.visit(s, life.world.visit_batches(s["visit"], s["group"]), life_hash=lh)
        n = life.done_visits
        print(f"{stem} visit {s['visit']} group {s['group']} new {s['new']} absence {s['absence']} loss {rec['loss_mean']:.4f} "
              f"right {np.frombuffer(life.ev['correct'], dtype=np.uint8)[-16 * rec['updates']:].mean():.3f} cpu {rec['cpu']:.1f}s", flush=True)
        if is_pow2(n) or n == V:
            life.save_state(out / f"{stem}_state_v{n}.pt")
        if stop_after is not None and n == stop_after:
            print(f"{stem}: stopped after visit {n - 1} (state {stem}_state_v{n}.pt)", flush=True)
            return None
    arrs = life.events_arrays()
    np.savez_compressed(epath, **arrs)
    rec = life.receipt("life", cmd, epath, arrs, dict(schedule=sched, stream_sha256=lh.hexdigest(),
                                                      stream_note="sha256 over every stream item in order, as sw_world.digest; equals "
                                                                  "sw_world.digest(World(world, visit_updates, p_new), visits)"))
    write_json(rpath, rec)
    print(f"{stem}: complete; receipt {rpath.name}", flush=True)
    return rec


def run_branch(state_path, groups, out, cmd):
    life = Life.load(state_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    n0, S = life.done_visits, life.cfg["world"]
    stem = life_stem(life.cfg) + f"_br{n0}_g" + "-".join(str(g) for g in groups)
    rpath, epath = out / f"{stem}.json", out / f"{stem}.npz"
    refuse_existing(rpath, epath, out / f"{stem}_state_end.pt")
    life.cpu0 = time.process_time(); life.cpu_before = 0.0
    trunk = dict(by_visit=len(life.by_visit), states=life.states)
    life.ev = {k: array(c) for k, c, _ in EVENT_COLUMNS}; life.by_visit = []; life.states = []
    last = {}
    for s in life.world.schedule(n0):
        last[s["group"]] = s["visit"]
    lh = hashlib.sha256()
    for k, g in enumerate(groups):
        assert 0 <= g < len(life.world.groups)
        v = n0 + k
        s = dict(visit=v, group=g, new=g not in last, absence=None if g not in last else v - last[g], branch_k=k,
                 generator=f"world|{S}|branch|{n0}|{k}", probe=g >= len(life.world.groups) - W.PROBE_GROUPS)
        life.visit(s, batches_with(life.world, Random(f"world|{S}|branch|{n0}|{k}"), g), life_hash=lh)
        last[g] = v
    arrs = life.events_arrays()
    np.savez_compressed(epath, **arrs)
    life.save_state(out / f"{stem}_state_end.pt")
    rec = life.receipt("branch", cmd, epath, arrs, dict(branch=dict(from_state=life.loaded_from, groups=groups, trunk=trunk),
                                                        stream_sha256=lh.hexdigest()))
    write_json(rpath, rec)
    print(f"{stem}: complete", flush=True)
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
    ap.add_argument("cmd", choices=("life", "branch", "ask"))
    ap.add_argument("pos", nargs="*")
    ap.add_argument("--rho", type=float); ap.add_argument("--H", type=int, default=64); ap.add_argument("--U", type=int, default=None)
    ap.add_argument("--visit-updates", type=int, default=W.VISIT_UPDATES); ap.add_argument("--p-new", type=float, default=W.P_NEW)
    ap.add_argument("--lr", type=float, default=0.001); ap.add_argument("--wd", type=float, default=0.25); ap.add_argument("--tag")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--stop-after", type=int); ap.add_argument("--resume")
    ap.add_argument("--groups"); ap.add_argument("--items"); ap.add_argument("--to")
    a = ap.parse_args(argv)
    cmd = " ".join(["python3", "sw_run.py", *argv])
    if a.cmd == "life":
        if a.resume:
            assert not a.pos, "--resume takes the life's settings from the state"
            life = Life.load(a.resume)
        else:
            arm, world, visits = a.pos[0], int(a.pos[1]), int(a.pos[2])
            life = Life(arm, world, visits, a.rho, a.H, a.U if a.U is not None else a.H, a.visit_updates, a.p_new, a.lr, a.wd, a.tag)
        run_life(life, a.out, cmd, a.stop_after)
    elif a.cmd == "branch":
        run_branch(a.pos[0], [int(g) for g in a.groups.split(",")], a.out, cmd)
    else:
        run_ask(a.pos[0], json.loads(Path(a.items).read_text()), a.to, cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
