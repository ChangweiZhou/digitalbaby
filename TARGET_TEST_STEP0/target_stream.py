"""Target test, step 0: the generator, the panels and the streams (brief OPUS_TASK_TARGET_TEST_STEP0_20261111.md, version 5).

Standard library only. No learner, no arithmetic harness. Every size in SIZES is the brief's proposal or, where the brief gives none, this
file's own proposal (marked "proposal here").

World. 24 contexts (A-X); two cue families of 12 (a-l, m-x); filler from 8 punctuation bytes; outcome two decimal digits; end byte "\\n".
    episode   X filler c = d d \\n            filler 0-14 bytes when teaching, 10-14 at a scored point (first encounter or panel)
    family 1  outcome = (v1(X) + u1(c)) mod 97        family 2  outcome = (v2(X) - u2(c)) mod 97
    two family-1 cues change value at the change point (an episode boundary inside the revise phase), each to a different value.
Streams built from one episode list:
    ordinary   episodes only
    wider      after an episode's end byte, d episodes later (d = 0..3, the line's delay stratum), a line "X:vv c:uu\\n" (10 bytes)
               stating that earlier episode's two values, as they were then. A line never crosses its phase or the change point.
    moved      the alignment control (version 6): first-encounter lines stay in place; teaching lines are permuted among their own
               slots of the same segment, delay stratum and value era, so no line names a pair before that pair's probe
    extra      no lines; at each place a line stands in `wider`, 10 ordinary filler bytes (the extra-observation control)
    control    the same four, with every family-1 episode of the control contexts (teaching and first-encounter reveals) replaced at
               the same position by a taught family-1 pair of another context with the same cue and the same filler bytes; lines
               follow the episodes they describe, so no line names a control context's family-1 value either.
Probes (scoring points; the scorer knows them, the learner does not): every first-encounter episode, at its own position, and the
sealed-twin panels at the end of acquire (A), at family 1's return (I) and at the end of the stream (R).
"""
import hashlib
import json
import random

M, NCTX, NCUE = 97, 24, 12
FILL = b"!#%&*+,-"
END = b"\n"
WINDOW = 10
LINE_LEN = 10
SIZES = dict(                         # proposals
    contexts=NCTX, cues_per_family=NCUE, modulus=M, window=WINDOW,
    taught_per_family=216, first_encounter_per_family=48, first_encounter_sets=("acquire", "retain", "revise"), per_set=16,
    sealed_twins_per_family=12, renderings_per_panel=150, panel_filler=(10, 14), teaching_filler=(0, 14),
    first_encounter_filler=(10, 14),                     # proposal here: scored episodes hide the context, as panels do
    changed_cues=2, revision_pairs_with_changed_cue=6,
    acquire_reps=4, absent_reps=2, return_reps=1, revise_reps=2,          # proposal here: times each taught pair is shown per segment
    acquire_first_from=0.5, retain_within=0.25, revise_first_from=0.4, change_at_fraction=0.10,   # proposal here: placements
    line_delays=(0, 1, 2, 3),                            # proposal here: a line's delay stratum, in episodes
    control_twins=2,                                     # proposal here: control contexts = the contexts of two family-1 twins
    distinct_pair_share=0.8,                             # proposal here: a distinct twin pair counts when this share of renderings is right
    evidence_after_change=3,                             # brief: three taught episodes of the cue, three contexts, before a revision probe
)


def ctx_byte(x):
    return 65 + x


def cue_byte(f, c):
    return 97 + 12 * f + c


def rng(seed, tag):
    return random.Random(f"target-step0|{seed}|{tag}")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else json.dumps(b, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- world
def make_world(seed, shared_context_values=False):
    r = rng(seed, "values")
    v = [[r.randrange(M) for _ in range(NCTX)] for _ in range(2)]
    if shared_context_values:          # fixture only (check 9): the families share context values
        v[1] = list(v[0])
    u = [[r.randrange(M) for _ in range(NCUE)] for _ in range(2)]
    changed = sorted(r.sample(range(NCUE), SIZES["changed_cues"]))
    u_new = {}
    for c in changed:                  # a revision must change the value
        n = r.randrange(M)
        while n == u[0][c]:
            n = r.randrange(M)
        u_new[c] = n
    return dict(seed=seed, v=v, u=u, changed=changed, u_new=u_new, shared_context_values=shared_context_values)


def true_outcome(world, f, x, c, era):
    """era 1 = after the change point. Used to write the stream's reveals; the oracle has its own implementation."""
    uc = world["u_new"][c] if (f == 0 and era == 1 and c in world["u_new"]) else world["u"][f][c]
    return (world["v"][f][x] + uc) % M if f == 0 else (world["v"][f][x] - uc) % M


# ---------------------------------------------------------------- split, fixed before anything is generated
def components_identify(taught, targets):
    """Teaching-graph argument (design draft 7a.1): a held pair is fixed by the taught outcomes iff its context and cue lie in one
    connected component of the graph whose edges are the taught pairs."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for x, c in taught:
        parent[find(("x", x))] = find(("c", c))
    return all(find(("x", x)) == find(("c", c)) for x, c in targets)


def make_split(world, seed):
    r = rng(seed, "split"); S = {}; twin_redraws = [0, 0]; split_redraws = [0, 0]
    for f in (0, 1):
        while True:
            twins = []
            for c in range(NCUE):                                  # one twin per cue: the same cue, two contexts
                x1, x2 = r.sample(range(NCTX), 2)
                while world["v"][f][x1] == world["v"][f][x2]:      # redrawn, never dropped
                    twin_redraws[f] += 1; x1, x2 = r.sample(range(NCTX), 2)
                twins.append(((x1, c), (x2, c)))
            sealed = {p for tw in twins for p in tw}
            rest = [(x, c) for x in range(NCTX) for c in range(NCUE) if (x, c) not in sealed]
            r.shuffle(rest)
            if f == 0:     # the revision set holds three pairs of each changed cue
                rev = []
                for c in world["changed"]:
                    rev += [p for p in rest if p[1] == c][:SIZES["revision_pairs_with_changed_cue"] // len(world["changed"])]
                rest = [p for p in rest if p not in rev]
                n = 16 - len(rev); rev += rest[:n]; rest = rest[n:]
            else:
                rev, rest = rest[:16], rest[16:]
            acq, ret, taught = rest[:16], rest[16:32], rest[32:]
            assert len(taught) == SIZES["taught_per_family"] and len(rev) == 16, (len(taught), len(rev))
            if components_identify(taught, sealed | set(acq) | set(ret) | set(rev)):
                break
            split_redraws[f] += 1
        S[f] = dict(twins=twins, acquire=acq, retain=ret, revise=rev, taught=taught)
    # control contexts: the contexts of two family-1 twins with four distinct contexts
    cand = [tw for tw in S[0]["twins"]]
    r.shuffle(cand); chosen = []
    for tw in cand:
        xs = {tw[0][0], tw[1][0]}
        if not xs & {p[0] for t in chosen for p in t}:
            chosen.append(tw)
        if len(chosen) == SIZES["control_twins"]:
            break
    return dict(families=S, twin_redraws=twin_redraws, split_redraws=split_redraws,
                control_contexts=sorted({p[0] for tw in chosen for p in tw}), control_twins=chosen)


# ---------------------------------------------------------------- episodes
def make_episodes(world, split, seed):
    """One list of episodes for every stream. Each: f, x, c, fill, kind (teach | first), set (acquire/retain/revise for first), phase, era."""
    r = rng(seed, "episodes"); S = split["families"]

    def fill(lo, hi):
        return bytes(r.choice(FILL) for _ in range(r.randint(lo, hi)))

    def teach(fams, reps):
        out = [dict(f=f, x=x, c=c, kind="teach") for _ in range(reps) for f in fams for (x, c) in S[f]["taught"]]
        r.shuffle(out)
        for e in out:
            e["fill"] = fill(*SIZES["teaching_filler"])
        return out

    def insert(seq, items, lo, hi):
        pos = sorted(r.randint(int(lo * len(seq)), int(hi * len(seq))) for _ in items)
        r.shuffle(items)
        for k, (p, e) in enumerate(zip(pos, items)):
            seq.insert(p + k, e)
        return seq

    def firsts(setname):
        items = [dict(f=f, x=x, c=c, kind="first", set=setname, fill=fill(*SIZES["first_encounter_filler"])) for f in (0, 1) for (x, c) in S[f][setname]]
        return items
    acquire = insert(teach((0, 1), SIZES["acquire_reps"]), firsts("acquire"), SIZES["acquire_first_from"], 1.0)
    absent = teach((1,), SIZES["absent_reps"])
    back = insert(teach((0, 1), SIZES["return_reps"]), firsts("retain"), 0.0, SIZES["retain_within"])
    for attempt in range(1000):        # revision probes need the declared evidence after the change; positions are redrawn until they have it
        revise = teach((0, 1), SIZES["revise_reps"])
        change_idx = int(SIZES["change_at_fraction"] * len(revise))
        revise = insert(revise, firsts("revise"), SIZES["revise_first_from"], 1.0)
        if revision_evidence_ok(world, revise, change_idx):
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
        e["o"] = true_outcome(world, e["f"], e["x"], e["c"], e["era"])
    probes_at = dict(A=n_acq, I=n_acq + n_abs, R=len(eps))       # panel probes: episode boundaries
    return eps, dict(change_at=change_at, panel_probes=probes_at, segments=dict(acquire=n_acq, absent=n_abs, back=len(back), revise=len(revise)),
                     placement_attempts=attempt + 1)


def revision_evidence_ok(world, revise, change_idx):
    """Before every revision probe with a changed cue, at least three taught episodes of that cue after the change, three contexts."""
    for k, e in enumerate(revise):
        if e["kind"] == "first" and e["f"] == 0 and e["c"] in world["changed"]:
            ctxs = {d["x"] for d in revise[change_idx:k] if d["kind"] == "teach" and d["f"] == 0 and d["c"] == e["c"]}
            if k < change_idx or len(ctxs) < SIZES["evidence_after_change"]:
                return False
    return True


# ---------------------------------------------------------------- lines and streams
def episode_bytes(e, outcome=True):
    b = bytes([ctx_byte(e["x"])]) + e["fill"] + bytes([cue_byte(e["f"], e["c"])]) + b"="
    return b + (b"%02d" % e["o"] + END if outcome else b"")


def line_bytes(world, e):
    """X:vv c:uu  -- the episode's context value and cue value in its family, as they were at the episode."""
    uc = world["u_new"][e["c"]] if (e["f"] == 0 and e["era"] == 1 and e["c"] in world["u_new"]) else world["u"][e["f"]][e["c"]]
    return bytes([ctx_byte(e["x"])]) + b":%02d " % world["v"][e["f"]][e["x"]] + bytes([cue_byte(e["f"], e["c"])]) + b":%02d" % uc + END


def make_slots(eps, seed):
    """Every revealed episode gets a line, hosted d episodes later within the same segment and value era (d capped at the segment end).
    Segments are the phases, with the interfere phase taken as two: the stretch where family 1 is absent, and its return. A line is never
    moved into the stretch where family 1 is absent from one where it is present, or the reverse."""
    r = rng(seed, "lines"); slots = []
    seg = [(e["segment"], e["era"]) for e in eps]
    for i, e in enumerate(eps):
        d = r.choice(SIZES["line_delays"])
        while i + d >= len(eps) or seg[i + d] != seg[i]:
            d -= 1
        slots.append(dict(describes=i, host=i + d, delay=d, phase=e["phase"], segment=e["segment"], era=e["era"]))
    slots.sort(key=lambda s: (s["host"], s["describes"]))
    return slots


def move_lines(slots, seed, eps, in_place=False):
    """The alignment control (version 6). The lines of first-encounter episodes stay where they are, so no line names a pair before that
    pair's probe. The lines of teaching episodes are permuted among their own slots within one class (segment of a phase, delay stratum,
    value era). Every symbol a line names has one value within a class, so a moved line is still true where it lands. A class with one
    free line keeps it, and is counted."""
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
        if in_place:                      # fixture (check 7): a "shuffle" that leaves every line where it was
            continue
        perm = [content[k] for k in ks]; r.shuffle(perm)
        for k, d in zip(ks, perm):
            content[k] = d
    moved = [dict(s, describes=content[k], original=s["describes"], free=free[k]) for k, s in enumerate(slots)]
    return moved, dict(classes=len(classes), class_sizes=sorted(len(v) for v in classes.values()), lines_that_stayed=stayed,
                       lines_free_to_move=sum(free), lines_fixed_first_encounter=len(slots) - sum(free),
                       lines_moved=sum(1 for k in range(len(slots)) if content[k] != slots[k]["describes"]))


def render(world, eps, slots, mode, seed):
    """mode: ordinary | lines (slots carry describes) | extra. Returns bytes, offsets of every episode (start, '='), the change point as a
    byte offset (the first byte of the first episode after the change), and the byte ranges where lines (or their filler) stand."""
    r = rng(seed, f"extra-filler|{mode}")
    out = bytearray(); starts, eqs, line_ranges = [], [], []
    by_host = {}
    for s in slots or []:
        by_host.setdefault(s["host"], []).append(s)
    for i, e in enumerate(eps):
        starts.append(len(out)); b = episode_bytes(e); eqs.append(len(out) + b.index(b"="))
        out += b
        for s in by_host.get(i, []) if mode != "ordinary" else []:
            a = len(out)
            out += line_bytes(world, eps[s["describes"]]) if mode == "lines" else bytes(r.choice(FILL) for _ in range(LINE_LEN))
            line_ranges.append((a, len(out), s["describes"], s["host"]))
    return bytes(out), dict(starts=starts, eqs=eqs, line_ranges=line_ranges, end=len(out))


def control_episodes(eps, split, seed):
    """Every family-1 episode (teaching or first-encounter reveal) of a control context replaced, at the same position, by a taught
    family-1 pair of a non-control context with the same cue and the same filler bytes."""
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


def fix_outcomes(world, eps):
    for e in eps:
        e["o"] = true_outcome(world, e["f"], e["x"], e["c"], e["era"])
    return eps


# ---------------------------------------------------------------- panels
def make_panels(world, split, seed):
    """For each panel probe (A, I, R) and family: 150 twin renderings drawn in turn from the 12 sealed twin pairs. The two members share
    the cue and the filler (10-14 bytes); they differ in context."""
    r = rng(seed, "panels"); P = {}
    for probe in ("A", "I", "R"):
        for f in (0, 1):
            tw = split["families"][f]["twins"]; out = []
            for k in range(SIZES["renderings_per_panel"]):
                (x1, c), (x2, _) = tw[k % len(tw)]
                fl = bytes(r.choice(FILL) for _ in range(r.randint(*SIZES["panel_filler"])))
                m1 = bytes([ctx_byte(x1)]) + fl + bytes([cue_byte(f, c)]) + b"="
                m2 = bytes([ctx_byte(x2)]) + fl + bytes([cue_byte(f, c)]) + b"="
                out.append(dict(pair=k % len(tw), f=f, c=c, x1=x1, x2=x2, m1=m1, m2=m2))
            P[(probe, f)] = out
    return P


def probes(eps, meta):
    """Scoring points by episode index: every first-encounter episode, and the three panel probes."""
    pr = [dict(kind="first", at=e["i"], f=e["f"], x=e["x"], c=e["c"], set=e["set"], phase=e["phase"], era=e["era"]) for e in eps if e["kind"] == "first"]
    for name, at in meta["panel_probes"].items():
        pr.append(dict(kind="panel", name=name, at=at, era=1 if at >= meta["change_at"] else 0))
    return sorted(pr, key=lambda p: (p["at"], p["kind"] == "first"))


# ---------------------------------------------------------------- the scorer's twin protocol
def score_twin(responder, m1, m2):
    """The protocol a learner will meet: save the state, present one member, restore, present the other, restore. Returns the answers."""
    s0 = responder.get_state()
    a1 = responder.answer(m1); responder.set_state(s0)
    a2 = responder.answer(m2); responder.set_state(s0)
    return a1, a2


# ---------------------------------------------------------------- everything for one world seed
def build(seed, shared_context_values=False, split_override=None):
    world = make_world(seed, shared_context_values)
    split = split_override or make_split(world, seed)
    eps, meta = make_episodes(world, split, seed)
    slots = make_slots(eps, seed)
    moved, move_info = move_lines(slots, seed, eps)
    ceps, replaced = control_episodes(eps, split, seed)
    ceps = fix_outcomes(world, ceps)
    streams, offsets = {}, {}
    for name, (E_, sl, mode) in dict(ordinary=(eps, None, "ordinary"), wider=(eps, slots, "lines"), moved=(eps, moved, "lines"),
                                     extra=(eps, slots, "extra"), control=(ceps, None, "ordinary"),
                                     control_wider=(ceps, slots, "lines")).items():
        streams[name], offsets[name] = render(world, E_, sl, mode, seed)
    panels = make_panels(world, split, seed)
    return dict(seed=seed, world=world, split=split, episodes=eps, meta=meta, slots=slots, moved_slots=moved, move_info=move_info,
                control_episodes=ceps, control_replaced=replaced, streams=streams, offsets=offsets, panels=panels, probes=probes(eps, meta))


def hashes(B):
    S = B["split"]["families"]
    h = dict(streams={k: sha(v) for k, v in B["streams"].items()},
             panels={f"{p}/family{f + 1}": sha([[d["m1"].decode(), d["m2"].decode()] for d in v]) for (p, f), v in B["panels"].items()},
             held_sets={f"{s}/family{f + 1}": sha(sorted(S[f][s])) for f in (0, 1) for s in ("acquire", "retain", "revise")})
    h["held_sets"].update({f"sealed/family{f + 1}": sha(sorted(S[f]["twins"])) for f in (0, 1)})
    return h
