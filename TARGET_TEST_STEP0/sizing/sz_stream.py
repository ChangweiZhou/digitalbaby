"""Sizing sweep, stage 1: the generator, parameterized (a copy of ../target_stream.py; brief OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3).

Every function takes one explicit configuration `cfg` (a dict with the keys of step 0's SIZES; the default configuration is step 0's SIZES
exactly). Nothing here reads a size from a module constant: the modulus, the counts and the repetitions all come from `cfg`. The format
constants (filler bytes, end byte, line length) are the stream's public format and do not vary. At the default configuration this file
must give step 0's bytes, panels and hashes (checked by sz_check.py against ../TARGET_STEP0_CHECK.json).

Changes from target_stream.py, all mechanical: SIZES -> cfg; M, NCTX, NCUE, WINDOW -> cfg values; the literal 16 of make_split -> cfg["per_set"];
the literal 12 of cue_byte -> cfg["cues_per_family"]; every built bundle records the configuration it was built from and its hash.
"""
import hashlib
import json
import random

FILL = b"!#%&*+,-"
END = b"\n"
LINE_LEN = 10
REQUIRED = ("contexts", "cues_per_family", "modulus", "window", "taught_per_family", "first_encounter_per_family", "first_encounter_sets",
            "per_set", "sealed_twins_per_family", "renderings_per_panel", "panel_filler", "teaching_filler", "first_encounter_filler",
            "changed_cues", "revision_pairs_with_changed_cue", "acquire_reps", "absent_reps", "return_reps", "revise_reps",
            "acquire_first_from", "retain_within", "revise_first_from", "change_at_fraction", "line_delays", "control_twins",
            "distinct_pair_share", "evidence_after_change")


def check_cfg(cfg):
    miss = [k for k in REQUIRED if k not in cfg]
    if miss:
        raise ValueError(f"sz_stream: configuration lacks {miss}")
    if not (2 <= cfg["modulus"] <= 100) or cfg["contexts"] > 26 or 2 * cfg["cues_per_family"] > 26:
        raise ValueError("sz_stream: configuration outside the byte format (two decimal digits, letters A-Z and a-z)")
    return cfg


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def ctx_byte(x):
    return 65 + x


def cue_byte(cfg, f, c):
    return 97 + cfg["cues_per_family"] * f + c


def rng(seed, tag):
    return random.Random(f"target-step0|{seed}|{tag}")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else json.dumps(b, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- world
def make_world(cfg, seed, shared_context_values=False):
    M, NCTX, NCUE = cfg["modulus"], cfg["contexts"], cfg["cues_per_family"]
    r = rng(seed, "values")
    v = [[r.randrange(M) for _ in range(NCTX)] for _ in range(2)]
    if shared_context_values:          # fixture only (check 9): the families share context values
        v[1] = list(v[0])
    u = [[r.randrange(M) for _ in range(NCUE)] for _ in range(2)]
    changed = sorted(r.sample(range(NCUE), cfg["changed_cues"]))
    u_new = {}
    for c in changed:                  # a revision must change the value
        n = r.randrange(M)
        while n == u[0][c]:
            n = r.randrange(M)
        u_new[c] = n
    return dict(seed=seed, v=v, u=u, changed=changed, u_new=u_new, shared_context_values=shared_context_values, modulus=M)


def true_outcome(cfg, world, f, x, c, era):
    """era 1 = after the change point. Used to write the stream's reveals; the oracle has its own implementation."""
    M = cfg["modulus"]
    uc = world["u_new"][c] if (f == 0 and era == 1 and c in world["u_new"]) else world["u"][f][c]
    return (world["v"][f][x] + uc) % M if f == 0 else (world["v"][f][x] - uc) % M


# ---------------------------------------------------------------- split, fixed before anything is generated
def components_identify(taught, targets):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for x, c in taught:
        parent[find(("x", x))] = find(("c", c))
    return all(find(("x", x)) == find(("c", c)) for x, c in targets)


def make_split(cfg, world, seed):
    NCTX, NCUE, K = cfg["contexts"], cfg["cues_per_family"], cfg["per_set"]
    r = rng(seed, "split"); S = {}; twin_redraws = [0, 0]; split_redraws = [0, 0]
    for f in (0, 1):
        while True:
            twins = []
            for c in range(NCUE):
                x1, x2 = r.sample(range(NCTX), 2)
                while world["v"][f][x1] == world["v"][f][x2]:      # redrawn, never dropped
                    twin_redraws[f] += 1; x1, x2 = r.sample(range(NCTX), 2)
                twins.append(((x1, c), (x2, c)))
            sealed = {p for tw in twins for p in tw}
            rest = [(x, c) for x in range(NCTX) for c in range(NCUE) if (x, c) not in sealed]
            r.shuffle(rest)
            if f == 0:
                rev = []
                for c in world["changed"]:
                    rev += [p for p in rest if p[1] == c][:cfg["revision_pairs_with_changed_cue"] // len(world["changed"])]
                rest = [p for p in rest if p not in rev]
                n = K - len(rev); rev += rest[:n]; rest = rest[n:]
            else:
                rev, rest = rest[:K], rest[K:]
            acq, ret, taught = rest[:K], rest[K:2 * K], rest[2 * K:]
            assert len(taught) == cfg["taught_per_family"] and len(rev) == K, (len(taught), len(rev))
            if components_identify(taught, sealed | set(acq) | set(ret) | set(rev)):
                break
            split_redraws[f] += 1
        S[f] = dict(twins=twins, acquire=acq, retain=ret, revise=rev, taught=taught)
    cand = [tw for tw in S[0]["twins"]]
    r.shuffle(cand); chosen = []
    for tw in cand:
        xs = {tw[0][0], tw[1][0]}
        if not xs & {p[0] for t in chosen for p in t}:
            chosen.append(tw)
        if len(chosen) == cfg["control_twins"]:
            break
    return dict(families=S, twin_redraws=twin_redraws, split_redraws=split_redraws,
                control_contexts=sorted({p[0] for tw in chosen for p in tw}), control_twins=chosen)


# ---------------------------------------------------------------- episodes
def make_episodes(cfg, world, split, seed):
    r = rng(seed, "episodes"); S = split["families"]

    def fill(lo, hi):
        return bytes(r.choice(FILL) for _ in range(r.randint(lo, hi)))

    def teach(fams, reps):
        out = [dict(f=f, x=x, c=c, kind="teach") for _ in range(reps) for f in fams for (x, c) in S[f]["taught"]]
        r.shuffle(out)
        for e in out:
            e["fill"] = fill(*cfg["teaching_filler"])
        return out

    def insert(seq, items, lo, hi):
        pos = sorted(r.randint(int(lo * len(seq)), int(hi * len(seq))) for _ in items)
        r.shuffle(items)
        out, k = [], 0                 # same result as step 0's repeated list.insert(p + k, e), built in one pass
        for j, (p, e) in enumerate(zip(pos, items)):
            out.extend(seq[k:p]); k = p; out.append(e)
        out.extend(seq[k:])
        return out

    def firsts(setname):
        return [dict(f=f, x=x, c=c, kind="first", set=setname, fill=fill(*cfg["first_encounter_filler"])) for f in (0, 1) for (x, c) in S[f][setname]]
    acquire = insert(teach((0, 1), cfg["acquire_reps"]), firsts("acquire"), cfg["acquire_first_from"], 1.0)
    absent = teach((1,), cfg["absent_reps"])
    back = insert(teach((0, 1), cfg["return_reps"]), firsts("retain"), 0.0, cfg["retain_within"])
    for attempt in range(1000):
        revise = teach((0, 1), cfg["revise_reps"])
        change_idx = int(cfg["change_at_fraction"] * len(revise))
        revise = insert(revise, firsts("revise"), cfg["revise_first_from"], 1.0)
        if revision_evidence_ok(cfg, world, revise, change_idx):
            break
    else:
        raise RuntimeError("no placement of the revision pairs met the evidence rule")
    eps = []
    for phase, segment, seg in (("acquire", "acquire", acquire), ("interfere", "absent", absent), ("interfere", "return", back), ("revise", "revise", revise)):
        for e in seg:
            e["phase"] = phase; e["segment"] = segment; e["era"] = 0; eps.append(e)
    n_acq, n_abs = len(acquire), len(absent)
    change_at = n_acq + n_abs + len(back) + change_idx
    for i, e in enumerate(eps):
        e["i"] = i
        if i >= change_at:
            e["era"] = 1
        e["o"] = true_outcome(cfg, world, e["f"], e["x"], e["c"], e["era"])
    probes_at = dict(A=n_acq, I=n_acq + n_abs, R=len(eps))
    return eps, dict(change_at=change_at, panel_probes=probes_at, segments=dict(acquire=n_acq, absent=n_abs, back=len(back), revise=len(revise)),
                     placement_attempts=attempt + 1)


def revision_evidence_ok(cfg, world, revise, change_idx):
    for k, e in enumerate(revise):
        if e["kind"] == "first" and e["f"] == 0 and e["c"] in world["changed"]:
            ctxs = {d["x"] for d in revise[change_idx:k] if d["kind"] == "teach" and d["f"] == 0 and d["c"] == e["c"]}
            if k < change_idx or len(ctxs) < cfg["evidence_after_change"]:
                return False
    return True


# ---------------------------------------------------------------- lines and streams
def episode_bytes(cfg, e, outcome=True):
    b = bytes([ctx_byte(e["x"])]) + e["fill"] + bytes([cue_byte(cfg, e["f"], e["c"])]) + b"="
    return b + (b"%02d" % e["o"] + END if outcome else b"")


def line_bytes(cfg, world, e):
    uc = world["u_new"][e["c"]] if (e["f"] == 0 and e["era"] == 1 and e["c"] in world["u_new"]) else world["u"][e["f"]][e["c"]]
    return bytes([ctx_byte(e["x"])]) + b":%02d " % world["v"][e["f"]][e["x"]] + bytes([cue_byte(cfg, e["f"], e["c"])]) + b":%02d" % uc + END


def make_slots(cfg, eps, seed):
    r = rng(seed, "lines"); slots = []
    seg = [(e["segment"], e["era"]) for e in eps]
    for i, e in enumerate(eps):
        d = r.choice(cfg["line_delays"])
        while i + d >= len(eps) or seg[i + d] != seg[i]:
            d -= 1
        slots.append(dict(describes=i, host=i + d, delay=d, phase=e["phase"], segment=e["segment"], era=e["era"]))
    slots.sort(key=lambda s: (s["host"], s["describes"]))
    return slots


def move_lines(slots, seed, eps, in_place=False):
    r = rng(seed, "move"); classes = {}
    free = [eps[s["describes"]]["kind"] == "teach" for s in slots]
    for k, s in enumerate(slots):
        if free[k]:
            classes.setdefault((s["segment"], s["delay"], s["era"]), []).append(k)
    content = [s["describes"] for s in slots]
    stayed = 0
    for key, ks in sorted(classes.items()):
        if len(ks) == 1:
            stayed += 1; continue
        if in_place:
            continue
        perm = [content[k] for k in ks]; r.shuffle(perm)
        for k, d in zip(ks, perm):
            content[k] = d
    moved = [dict(s, describes=content[k], original=s["describes"], free=free[k]) for k, s in enumerate(slots)]
    return moved, dict(classes=len(classes), class_sizes=sorted(len(v) for v in classes.values()), lines_that_stayed=stayed,
                       lines_free_to_move=sum(free), lines_fixed_first_encounter=len(slots) - sum(free),
                       lines_moved=sum(1 for k in range(len(slots)) if content[k] != slots[k]["describes"]))


def render(cfg, world, eps, slots, mode, seed):
    r = rng(seed, f"extra-filler|{mode}")
    out = bytearray(); starts, eqs, line_ranges = [], [], []
    by_host = {}
    for s in slots or []:
        by_host.setdefault(s["host"], []).append(s)
    for i, e in enumerate(eps):
        starts.append(len(out)); b = episode_bytes(cfg, e); eqs.append(len(out) + b.index(b"="))
        out += b
        for s in by_host.get(i, []) if mode != "ordinary" else []:
            a = len(out)
            out += line_bytes(cfg, world, eps[s["describes"]]) if mode == "lines" else bytes(r.choice(FILL) for _ in range(LINE_LEN))
            line_ranges.append((a, len(out), s["describes"], s["host"]))
    return bytes(out), dict(starts=starts, eqs=eqs, line_ranges=line_ranges, end=len(out))


def control_episodes(eps, split, seed):
    r = rng(seed, "control"); ctl = set(split["control_contexts"])
    taught = {}
    for (x, c) in split["families"][0]["taught"]:
        if x not in ctl:
            taught.setdefault(c, []).append(x)
    out, replaced = [], []
    for e in eps:
        if e["f"] == 0 and e["x"] in ctl:
            x2 = r.choice(sorted(taught[e["c"]]))
            n = dict(e, x=x2, kind="teach", replaced=dict(x=e["x"], kind=e["kind"], set=e.get("set")))
            n.pop("set", None)
            replaced.append(e["i"]); out.append(n)
        else:
            out.append(dict(e))
    return out, replaced


def fix_outcomes(cfg, world, eps):
    for e in eps:
        e["o"] = true_outcome(cfg, world, e["f"], e["x"], e["c"], e["era"])
    return eps


# ---------------------------------------------------------------- panels
def make_panels(cfg, world, split, seed):
    r = rng(seed, "panels"); P = {}
    for probe in ("A", "I", "R"):
        for f in (0, 1):
            tw = split["families"][f]["twins"]; out = []
            for k in range(cfg["renderings_per_panel"]):
                (x1, c), (x2, _) = tw[k % len(tw)]
                fl = bytes(r.choice(FILL) for _ in range(r.randint(*cfg["panel_filler"])))
                m1 = bytes([ctx_byte(x1)]) + fl + bytes([cue_byte(cfg, f, c)]) + b"="
                m2 = bytes([ctx_byte(x2)]) + fl + bytes([cue_byte(cfg, f, c)]) + b"="
                out.append(dict(pair=k % len(tw), f=f, c=c, x1=x1, x2=x2, m1=m1, m2=m2))
            P[(probe, f)] = out
    return P


def probes(eps, meta):
    pr = [dict(kind="first", at=e["i"], f=e["f"], x=e["x"], c=e["c"], set=e["set"], phase=e["phase"], era=e["era"]) for e in eps if e["kind"] == "first"]
    for name, at in meta["panel_probes"].items():
        pr.append(dict(kind="panel", name=name, at=at, era=1 if at >= meta["change_at"] else 0))
    return sorted(pr, key=lambda p: (p["at"], p["kind"] == "first"))


def score_twin(responder, m1, m2):
    s0 = responder.get_state()
    a1 = responder.answer(m1); responder.set_state(s0)
    a2 = responder.answer(m2); responder.set_state(s0)
    return a1, a2


# ---------------------------------------------------------------- everything for one world seed
def build(cfg, seed, shared_context_values=False, split_override=None, streams_wanted=None):
    """streams_wanted: None for all six of step 0, or a subset (a fit needs only its own stream; the checks need all six)."""
    check_cfg(cfg)
    world = make_world(cfg, seed, shared_context_values)
    split = split_override or make_split(cfg, world, seed)
    eps, meta = make_episodes(cfg, world, split, seed)
    slots = make_slots(cfg, eps, seed)
    moved, move_info = move_lines(slots, seed, eps)
    ceps, replaced = control_episodes(eps, split, seed)
    ceps = fix_outcomes(cfg, world, ceps)
    streams, offsets = {}, {}
    for name, (E_, sl, mode) in dict(ordinary=(eps, None, "ordinary"), wider=(eps, slots, "lines"), moved=(eps, moved, "lines"),
                                     extra=(eps, slots, "extra"), control=(ceps, None, "ordinary"),
                                     control_wider=(ceps, slots, "lines")).items():
        if streams_wanted is None or name in streams_wanted:
            streams[name], offsets[name] = render(cfg, world, E_, sl, mode, seed)
    panels = make_panels(cfg, world, split, seed)
    return dict(seed=seed, cfg=cfg, cfg_hash=cfg_hash(cfg), world=world, split=split, episodes=eps, meta=meta, slots=slots, moved_slots=moved,
                move_info=move_info, control_episodes=ceps, control_replaced=replaced, streams=streams, offsets=offsets, panels=panels,
                probes=probes(eps, meta))


def hashes(B):
    S = B["split"]["families"]
    h = dict(streams={k: sha(v) for k, v in B["streams"].items()},
             panels={f"{p}/family{f + 1}": sha([[d["m1"].decode(), d["m2"].decode()] for d in v]) for (p, f), v in B["panels"].items()},
             held_sets={f"{s}/family{f + 1}": sha(sorted(S[f][s])) for f in (0, 1) for s in ("acquire", "retain", "revise")})
    h["held_sets"].update({f"sealed/family{f + 1}": sha(sorted(S[f]["twins"])) for f in (0, 1)})
    return h
