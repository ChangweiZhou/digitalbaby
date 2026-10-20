"""BYTE-CORE-1 library.

Implements DESIGN.md (same directory): the byte-level relevance world, the
fixed front ends, the two frozen milestones (V50-REF, FULL_151 at S0), the
evaluator-side references, and the life driver.

Separation rule used throughout: learners receive only timed bytes (and, for
FULL_151, the output of the innate reinforcer detector, which the learner
computes itself from reserved byte values).  Everything else in a world
(identities, tags, committed truth, relevance) is evaluator data and is never
passed to a learner.  Oracle front ends are the single, labelled exception.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = "byte-core-1.0.0"
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
V88_RUNNER = PROJECT / "V88" / "minifly_rce_v87_followup_runner.py"
V51_SOURCE = PROJECT / "V51_Mac_M3" / "elm_v51_core_scope.py"
MINIFLY = PROJECT / "minifly"

# Hashes fixed in DESIGN.md sections 4.1/4.2 (verified before any science).
PRIMARY_SOURCE_HASHES = {
    "V88/minifly_rce_v87_followup_runner.py": "49b19beb9248c3a848c972659c7fe2fa714f313badecf086102048c5d898d435",
    "V51_Mac_M3/elm_v51_core_scope.py": "99e69d9b6e3b3cf7b05a394eb4248a60b819c9d7c42a3b8bcc8a43bea8b2316c",
    "minifly/V82E/src/model_evo.py": "ccdeaab99a889e6a74e47f95d641bba79ed95597b8bd902181e1c535942e25d7",
    "minifly/V82E/src/common_evo.py": "595211c5136bf5c13392d7390ce6c55ecb30f99fe21a537374eb3794cee68ad7",
}
OBJECT_HASHES = {
    "B": "32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964",
    "Q": "1aa98dbc6c7424b3f0ddc8cb2af6e02a4d405f718d2de0d829bde05f3575f202",
}
ENV_OF_RECORD = {"python": "3.11.5", "numpy": "2.2.6", "scipy": "1.14.1", "numba": "0.61.2"}

FULL_151 = [0.164547, 0.750000, -0.740848, -0.394336, 0.048862, 0.535677,
            0.170060, -0.048350, 0.304650, 0.848126, -0.176815, -0.047073,
            0.300916, 1.000000, -0.500000, 0.120990, 0.000000]
FULL_151_ID = "151ebd9e2d054ec9ec14"

# ---------------------------------------------------------------- world constants
EPISODE_S = 30.0
GAP_S = 135.0
DAY_S = 86400.0
FINAL_S = 604800.0
LIFE = {"block_pairs": [48, 24, 24, 24, 12, 12, 12, 12],
        "challenge_after_blocks": [0, 3, 4, 5, 6],
        "acquisition_bouts": 6, "challenge_bouts": 12,
        "stable_pairs": 4, "anomaly_pairs": 4, "revision_pairs": 4,
        "untrained_pairs": 24}
NOVEL_CHALLENGES = (2, 3, 4)
NOVEL_PER = 2
MISLEAD_CHALLENGES = (3, 4)
N_NEIGHBOURS = 24

# Reserved innate reinforcer bytes (amendment A1: 0xF1/0xF0 replace 'y'/'z',
# because V51's XOR mask maps key byte 'X' to 'z').
AVERSIVE = 0xF1
NEUTRAL = 0xF0
RESERVED = frozenset((AVERSIVE, NEUTRAL))
NEWLINE = 0x0A
ALPHABETS = {"R2": bytes((AVERSIVE, NEUTRAL)), "R8": b"abcdefgh",
             "R24": b"abcdefghijklmnopqrstuvwx"}
KEY_ALPHABET = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PREFIX = b"~|~|"
HEADERS = (b"@", b"#", b"?", b"!", b"%", b"&")
SEPARATORS = (b":", b"=", b">", b";", b"/", b"+")
XOR_MASKS = ((0x00, 0x00), (0x11, 0x22), (0x33, 0x55), (0x0F, 0x3C), (0x12, 0x34), (0x21, 0x43))
FORMATS = ("F1", "F2", "F3")
RUNGS = ("R2", "R8", "R24")

EPS = 1e-3
DELTA = 0.02
RECEPTOR = {"k": 6, "lo": 0.4, "hi": 1.0, "seed": "BYTE-CORE-1|RECEPTOR|v1"}
FE_PARAMS = {"FE0": {"tau_p": 10.0},
             "FE1": {"tau_p": 10.0, "kappa": 0.3, "tau_h": 1800.0},
             "FE2": {"tau_f": 5.0, "tau_s": 30.0, "w": 0.5},
             "FE3": {"tau_f": 5.0, "tau_s": 30.0, "w": 0.5, "kappa": 0.3, "tau_h": 1800.0},
             "FELAG": {"contacts": 8, "seed": "BYTE-CORE-1|LAG|v1"}}
NLMS_LRS = (0.1, 0.3, 1.0, 2.0)
IDREF_LAMBDAS = (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)

TAGS = ("acq", "stable", "anomaly", "anomaly_wrong", "revision", "react",
        "novel", "mislead", "mislead_wrong")
TAG = {t: i for i, t in enumerate(TAGS)}
PROBE_GROUPS = ("new", "relevant", "stable", "anomaly", "revision", "react", "novel",
                "mislead", "untrained", "neighbour", "f4", "trained")
PGROUP = {g: i for i, g in enumerate(PROBE_GROUPS)}
PHASES = ("immediate", "day", "final", "react", "revision_curve")
PHASE = {p: i for i, p in enumerate(PHASES)}


def protocol_constants():
    return {"version": VERSION, "episode_s": EPISODE_S, "gap_s": GAP_S, "day_s": DAY_S,
            "final_s": FINAL_S, "life": LIFE, "novel_challenges": NOVEL_CHALLENGES,
            "novel_per": NOVEL_PER, "mislead_challenges": MISLEAD_CHALLENGES,
            "neighbours": N_NEIGHBOURS, "aversive": AVERSIVE, "neutral": NEUTRAL,
            "alphabets": {k: v.hex() for k, v in ALPHABETS.items()},
            "key_alphabet": KEY_ALPHABET.decode(), "prefix": PREFIX.hex(),
            "headers": [h.hex() for h in HEADERS], "separators": [s.hex() for s in SEPARATORS],
            "xor_masks": XOR_MASKS, "eps": EPS, "delta": DELTA, "receptor": RECEPTOR,
            "fe_params": FE_PARAMS, "nlms_lrs": NLMS_LRS, "idref_lambdas": IDREF_LAMBDAS,
            "full_151": FULL_151, "full_151_id": FULL_151_ID}


class IntegrityError(RuntimeError):
    pass


class WorldError(RuntimeError):
    pass


# ---------------------------------------------------------------- utilities
def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).hex()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def canonical(x):
    return json.dumps(clean(x), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def sha_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def rng_for(*parts):
    return np.random.default_rng(int(digest(list(parts))[:16], 16))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".pending-", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, body):
    atomic_bytes(path, json.dumps(clean(body), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def save_npz(path, arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_bytes(path, buf.getvalue())


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- frozen sources
_MODS = {}


def load_module(name, path):
    if name in _MODS:
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def v88():
    return load_module("bc1_v88", V88_RUNNER)


def v51():
    return load_module("bc1_v51", V51_SOURCE)


_NATIVE = None


def native():
    """V88's native backend: loads V82E model_evo/common_evo with its own hash checks."""
    global _NATIVE
    if _NATIVE is None:
        _NATIVE = v88().NativeBackend({"backend": "native", "project_root": str(MINIFLY),
                                       "fixture_units": 64})
    return _NATIVE


def build_full151():
    """FULL_151 exactly as Stage 3A/V88: EvoLearner + fixed reference reader, own B."""
    be = native()
    a = be.model.EvoLearner(np.asarray(FULL_151, float))
    a.coupled_reader = a.reader
    a.reader = be.common.FrozenReader(be.ref_q.copy(), be.common.READER)
    return a


def object_hashes(a=None):
    a = a or build_full151()
    return {"B": v88().sparse_digest(a.m.B),
            "Q": hashlib.sha256(np.ascontiguousarray(a.reader.Q).tobytes()).hexdigest()}


def environment():
    import scipy
    import numba
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "numba": numba.__version__}


def loaded_project_files():
    native()
    v51()
    files = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        try:
            p.relative_to(PROJECT)
        except ValueError:
            continue
        if p.suffix == ".py":
            files.add(p)
    files.add((MINIFLY / "V82E" / "config.json").resolve())
    for extra in ("bytecore.py", "run_bytecore.py", "bc1_qualify.py", "bc1_analyze.py", "DESIGN.md"):
        if (HERE / extra).exists():
            files.add((HERE / extra).resolve())
    return sorted(files)


def compute_source_lock():
    files = {str(p.relative_to(PROJECT)): file_hash(p) for p in loaded_project_files()}
    for rel, h in PRIMARY_SOURCE_HASHES.items():
        if files.get(rel) != h:
            raise IntegrityError(f"primary source hash mismatch: {rel}")
    objs = object_hashes()
    if objs != OBJECT_HASHES:
        raise IntegrityError(f"FULL_151 object hash mismatch: {objs}")
    env = environment()
    if env != ENV_OF_RECORD:
        raise IntegrityError(f"environment differs from record: {env}")
    return {"files": files, "objects": objs, "environment": env,
            "protocol_sha256": digest(protocol_constants())}


def verify_source_lock(lock, root=PROJECT):
    bad = []
    for rel, h in lock["files"].items():
        p = Path(root) / rel
        if not p.is_file() or file_hash(p) != h:
            bad.append(rel)
    if bad:
        raise IntegrityError(f"source lock drift: {bad}")
    if digest(protocol_constants()) != lock["protocol_sha256"]:
        raise IntegrityError("protocol constants changed since lock")
    if environment() != lock["environment"]:
        raise IntegrityError("environment changed since lock")
    return True


# ---------------------------------------------------------------- rendering
def render_context(key, fmt, wrapper=0):
    key = bytes(key)
    if fmt == "F1":
        return key
    if fmt == "F2":
        return PREFIX + HEADERS[0] + key + SEPARATORS[0]
    if fmt in ("F3", "F4"):
        k = bytearray(key)
        m = XOR_MASKS[int(wrapper)]
        k[0] ^= m[0]
        k[1] ^= m[1]
        return PREFIX + HEADERS[int(wrapper)] + bytes(k) + SEPARATORS[int(wrapper)]
    raise ValueError(fmt)


def key_offset(fmt):
    return 0 if fmt == "F1" else 5


def render_record(key, fmt, wrapper, cons):
    return render_context(key, fmt, wrapper) + bytes((int(cons), NEWLINE))


# ---------------------------------------------------------------- world
def world_seed(ns, block, index):
    return sha_seed(f"BYTE-CORE-1|{ns}|{block}|{int(index)}")


def pick_fresh(rng, n, count, *, excluded, challenged, introduced):
    """V88 `pick_fresh`, verbatim semantics (stratified by introduction block)."""
    pool = [i for i in range(n) if i not in excluded and i not in challenged]
    if len(pool) < count:
        raise WorldError("insufficient challenge-fresh identities")
    pool.sort(key=lambda i: (introduced[i], i))
    strata = [list(map(int, x)) for x in np.array_split(np.asarray(pool, int), min(4, len(pool))) if len(x)]
    for s in strata:
        rng.shuffle(s)
    out = []
    while len(out) < count:
        progress = False
        for s in strata:
            if s:
                out.append(int(s.pop()))
                progress = True
                if len(out) == count:
                    break
        if not progress:
            break
    if len(out) != count:
        raise WorldError("fresh stratified selection failed")
    return out


def create_world(ns, block, index, rung, density="massed"):
    """Byte world (DESIGN.md section 3).  Keys and the challenge schedule depend
    only on (ns, block, index); consequences on the rung; acquisition order on
    the density.  Hence arms, formats and rungs are paired on identical worlds."""
    if rung not in ALPHABETS or density not in ("massed", "interleaved"):
        raise WorldError("bad world parameters")
    seed = world_seed(ns, block, index)
    K = len(ALPHABETS[rung])
    blocks = list(LIFE["block_pairs"])
    T = sum(blocks)
    U = LIFE["untrained_pairs"]
    NV = len(NOVEL_CHALLENGES) * NOVEL_PER
    npairs = T + U + NV
    ncues = 2 * npairs
    # keys
    krng = rng_for(seed, "keys")
    alph = np.frombuffer(KEY_ALPHABET, np.uint8)
    keys, seen = [], set()
    while len(keys) < ncues:
        k = bytes(krng.choice(alph, 6).astype(np.uint8))
        if k not in seen:
            seen.add(k)
            keys.append(k)
    nrng = rng_for(seed, "neighbours")
    sources = sorted(int(c) for c in nrng.choice(2 * T, N_NEIGHBOURS, replace=False))
    neighbours = []
    for c in sources:
        while True:
            k = bytearray(keys[c])
            for pos in nrng.choice([2, 3, 4, 5], 2, replace=False):
                others = [ch for ch in KEY_ALPHABET if ch != k[pos]]
                k[pos] = int(nrng.choice(others))
            kb = bytes(k)
            if kb not in seen:
                seen.add(kb)
                neighbours.append({"source": c, "key": kb})
                break
    # committed truth (index into the rung alphabet), per cue
    crng = rng_for(seed, "consequence", rung)
    if rung == "R2":
        roles = np.concatenate([crng.permutation(np.asarray([0, 1], int))
                                for _ in range((npairs + 1) // 2)])[:npairs].astype(int)
        truth0 = np.ones(ncues, int)
        for p in range(npairs):
            truth0[2 * p + int(roles[p])] = 0      # index 0 = AVERSIVE
    else:
        roles = None
        truth0 = crng.integers(0, K, ncues).astype(int)

    def changed(pair_truth):
        if rung == "R2":
            return [1 - int(pair_truth[0]), 1 - int(pair_truth[1])]
        return [int((int(t) + 1 + int(crng.integers(K - 1))) % K) for t in pair_truth]

    introduced = [bi for bi, size in enumerate(blocks) for _ in range(size)]
    srng = rng_for(seed, "schedule")
    events = []
    current = truth0.copy()
    challenged, ever_reactivated, ever_revised = set(), set(), set()
    future_reserved = set()
    react_plan = {}
    relevance = []
    novel_ids = {ci: [T + U + j * NOVEL_PER + k for k in range(NOVEL_PER)]
                 for j, ci in enumerate(NOVEL_CHALLENGES)}
    mislead_ids = {}
    groups_meta = {"challenges": []}

    def trial(p, tag, ctx, cycle, deliver):
        events.append({"kind": "trial", "pair": int(p), "deliver": [int(d) for d in deliver],
                       "tag": tag, "ctx": int(ctx), "cycle": int(cycle)})

    def probe(stage, phase, groups):
        events.append({"kind": "probe", "stage": stage, "phase": phase,
                       "groups": {g: [int(x) for x in v] for g, v in groups.items()}})

    def acquire(start, end, bi):
        arng = rng_for(seed, "acquisition", bi)
        pairs = [int(p) for p in arng.permutation(np.arange(start, end))]
        if density == "massed":
            order = [p for p in pairs for _ in range(LIFE["acquisition_bouts"])]
        else:
            order = list(pairs)
            for _ in range(LIFE["acquisition_bouts"] - 1):
                order += [int(p) for p in arng.permutation(np.asarray(pairs))]
        for p in order:
            trial(p, "acq", bi, -1, current[2 * p:2 * p + 2])
        g = {"new": range(start, end), "relevant": relevance}
        probe(f"block{bi}_immediate", "immediate", g)
        events.append({"kind": "rest", "seconds": DAY_S})
        probe(f"block{bi}_day", "day", {**g, "untrained": range(T, T + U)})

    def challenge(n, ci, after_block):
        nonlocal relevance, future_reserved
        excluded = set(future_reserved) | set(ever_reactivated)
        rev = pick_fresh(srng, n, LIFE["revision_pairs"], excluded=excluded,
                         challenged=challenged, introduced=introduced)
        excluded.update(rev)
        anomaly = pick_fresh(srng, n, LIFE["anomaly_pairs"], excluded=excluded,
                             challenged=challenged, introduced=introduced)
        excluded.update(anomaly)
        reactivated = list(react_plan.get(ci, []))
        if ci in (0, 1):
            stable = pick_fresh(srng, n, LIFE["stable_pairs"], excluded=excluded,
                                challenged=challenged, introduced=introduced)
        else:
            fresh = pick_fresh(srng, n, 2, excluded=excluded | set(reactivated),
                               challenged=challenged, introduced=introduced)
            stable = reactivated + fresh
        excluded.update(stable)
        novel = list(novel_ids.get(ci, []))
        mislead = []
        if ci in MISLEAD_CHALLENGES:
            mislead = pick_fresh(srng, n, 1, excluded=excluded | set(future_reserved),
                                 challenged=challenged, introduced=introduced)
            mislead_ids[ci] = mislead
        if any(p in ever_revised for p in reactivated):
            raise WorldError("reactivation pair was revised")
        challenged.update(rev + anomaly + stable + mislead)
        ever_reactivated.update(reactivated)
        future_reserved -= set(reactivated)
        targets = stable + anomaly + rev + novel + mislead
        if len(set(targets)) != len(targets):
            raise WorldError("challenge groups overlap")
        relevance = list(targets)
        events.append({"kind": "set_relevance", "pairs": list(targets), "challenge": ci})
        groups_meta["challenges"].append({"challenge": ci, "after_block": after_block, "stable": stable,
                                          "reactivated": reactivated, "anomaly": anomaly, "revision": rev,
                                          "novel": novel, "mislead": mislead})
        if reactivated or novel:
            events.append({"kind": "react_probe", "challenge": ci, "exposures": 0,
                           "react": reactivated, "novel": novel})
        wrong_bouts = {p: 1 + (j % 2) for j, p in enumerate(anomaly)}
        anomaly_wrong = {p: [changed(current[2 * p:2 * p + 2]) for _ in range(wrong_bouts[p])] for p in anomaly}
        mislead_wrong = {p: changed(current[2 * p:2 * p + 2]) for p in mislead}
        new_truth = {p: changed(current[2 * p:2 * p + 2]) for p in rev}
        committed = set()
        for cycle in range(LIFE["challenge_bouts"]):
            for p in srng.permutation(np.asarray(targets, int)):
                p = int(p)
                if p in rev and p not in committed:
                    events.append({"kind": "commit_truth", "pair": p, "truth": list(new_truth[p])})
                    current[2 * p:2 * p + 2] = new_truth[p]
                    committed.add(p)
                deliver = list(current[2 * p:2 * p + 2])
                if p in rev:
                    tag = "revision"
                elif p in anomaly:
                    if cycle < wrong_bouts[p]:
                        deliver, tag = anomaly_wrong[p][cycle], "anomaly_wrong"
                    else:
                        tag = "anomaly"
                elif p in mislead:
                    if cycle == 0:
                        deliver, tag = mislead_wrong[p], "mislead_wrong"
                    else:
                        tag = "mislead"
                elif p in novel:
                    tag = "novel"
                elif p in reactivated:
                    tag = "react"
                else:
                    tag = "stable"
                trial(p, tag, 100 + ci, cycle, deliver)
            ex = cycle + 1
            if (reactivated or novel) and ex in (1, 2, 4):
                events.append({"kind": "react_probe", "challenge": ci, "exposures": ex,
                               "react": reactivated, "novel": novel})
            if ex in (1, 2, 4, 8, LIFE["challenge_bouts"]):
                events.append({"kind": "revision_curve", "challenge": ci, "exposures": ex, "pairs": rev})
        ever_revised.update(rev)
        g = {"stable": stable, "anomaly": anomaly, "revision": rev, "react": reactivated,
             "novel": novel, "mislead": mislead, "relevant": targets}
        stage = f"challenge{ci}_after_block{after_block}"
        probe(stage + "_immediate", "immediate", g)
        events.append({"kind": "rest", "seconds": DAY_S})
        probe(stage + "_day", "day", {**g, "untrained": range(T, T + U)})
        if ci == 0:
            perm = [int(x) for x in srng.permutation(np.asarray(stable, int))]
            react_plan[2], react_plan[3] = perm[:2], perm[2:4]
            future_reserved.update(perm)
        elif ci == 1:
            perm = [int(x) for x in srng.permutation(np.asarray(stable, int))]
            react_plan[4] = perm[:2]
            future_reserved.update(perm[:2])

    start = 0
    ci = 0
    after = set(LIFE["challenge_after_blocks"])
    for bi, size in enumerate(blocks):
        end = start + size
        acquire(start, end, bi)
        if bi in after:
            challenge(end, ci, bi)
            ci += 1
        start = end
    events.append({"kind": "rest", "seconds": FINAL_S})
    probe("final_7day", "final", {"trained": range(T), "relevant": relevance})
    w = {"version": VERSION, "ns": ns, "block": block, "index": int(index), "seed": seed,
         "rung": rung, "density": density, "control": "normal", "keys": keys,
         "neighbours": neighbours, "truth0": truth0.tolist(),
         "roles": None if roles is None else roles.tolist(), "events": events,
         "introduced": introduced, "n_trained": T, "n_untrained": U, "novel_ids": novel_ids,
         "mislead_ids": mislead_ids, "groups": groups_meta, "npairs": npairs, "ncues": ncues}
    audit_world(w)
    return w


def unpaired_world(w):
    """V88 unpaired semantics: delivered consequences permuted across trials."""
    out = copy.deepcopy(w)
    trials = [e for e in out["events"] if e["kind"] == "trial"]
    rng = rng_for(w["seed"], "unpaired", w["rung"], w["density"])
    perm = rng.permutation(len(trials))
    delivers = [list(trials[i]["deliver"]) for i in perm]
    for e, d in zip(trials, delivers):
        e["deliver"] = d
    out["control"] = "unpaired"
    return out


def world_hash(w):
    return digest({k: w[k] for k in ("seed", "rung", "density", "control", "keys", "neighbours",
                                     "truth0", "events")})


def audit_world(w):
    """Structural audit (qualification Q8 and every life)."""
    T, U = w["n_trained"], w["n_untrained"]
    keys = w["keys"]
    if len(keys) != w["ncues"] or len(set(keys)) != len(keys):
        raise WorldError("keys not unique or wrong count")
    kset = set(keys)
    for k in keys:
        if len(k) != 6 or any(b not in KEY_ALPHABET for b in k):
            raise WorldError("bad key")
    if len(w["neighbours"]) != N_NEIGHBOURS:
        raise WorldError("neighbour count")
    for nb in w["neighbours"]:
        src = keys[nb["source"]]
        diff = [i for i in range(6) if src[i] != nb["key"][i]]
        if nb["key"] in kset or len(diff) != 2 or not set(diff) <= {2, 3, 4, 5}:
            raise WorldError("bad neighbour key")
    # reserved bytes never appear in any rendered context
    for k in keys + [nb["key"] for nb in w["neighbours"]]:
        ctxs = [render_context(k, "F1"), render_context(k, "F2")] + \
               [render_context(k, "F3", wr) for wr in (0, 1, 2)]
        for c in ctxs:
            if any(b in RESERVED for b in c):
                raise WorldError("reserved byte inside a context")
    ev = w["events"]
    trials = [e for e in ev if e["kind"] == "trial"]
    acq = [e for e in trials if e["tag"] == "acq"]
    if len(acq) != T * LIFE["acquisition_bouts"]:
        raise WorldError("acquisition trial count")
    cnt = defaultdict(int)
    for e in acq:
        cnt[e["pair"]] += 1
    if sorted(cnt) != list(range(T)) or set(cnt.values()) != {LIFE["acquisition_bouts"]}:
        raise WorldError("acquisition coverage")
    react4 = [e for e in ev if e["kind"] == "react_probe" and e["exposures"] == 4]
    if sum(len(e["react"]) for e in react4) != 6 or sum(len(e["novel"]) for e in react4) != 6:
        raise WorldError("reactivation / novel-match episodes")
    novel_all = [p for v in w["novel_ids"].values() for p in v]
    if any(e["pair"] in novel_all for e in acq):
        raise WorldError("novel pair acquired before its challenge")
    mw = [e for e in trials if e["tag"] == "mislead_wrong"]
    if len(mw) != len(MISLEAD_CHALLENGES) or any(e["cycle"] != 0 for e in mw):
        raise WorldError("misleading reminders")
    # every commit immediately precedes that pair's first revision trial
    commits = [(i, e) for i, e in enumerate(ev) if e["kind"] == "commit_truth"]
    if len(commits) != LIFE["revision_pairs"] * len(LIFE["challenge_after_blocks"]):
        raise WorldError("commit count")
    for i, e in commits:
        nxt = ev[i + 1]
        if nxt["kind"] != "trial" or nxt["pair"] != e["pair"] or nxt["tag"] != "revision" or nxt["cycle"] != 0:
            raise WorldError("commit is not immediately before the first revision trial")
    if any(e["tag"] not in TAG for e in trials):
        raise WorldError("unknown tag")
    return True


# ---------------------------------------------------------------- front ends
def receptor_table():
    rng = np.random.default_rng(sha_seed(RECEPTOR["seed"]))
    R = np.zeros((256, 88))
    for b in range(256):
        idx = rng.choice(88, RECEPTOR["k"], replace=False)
        R[b, idx] = rng.uniform(RECEPTOR["lo"], RECEPTOR["hi"], RECEPTOR["k"])
    return R


R_TABLE = receptor_table()


def _norm(p):
    m = float(p.max()) if p.size else 0.0
    return p / m if m > 0 else p.copy()


class FrontEnd:
    name = "base"
    oracle = False

    def __init__(self):
        self.t = 0.0

    def _dt(self, t):
        t = float(t)
        if t < self.t - 1e-9:
            raise IntegrityError(f"{self.name}: time went backwards")
        dt = max(0.0, t - self.t)
        self.t = max(self.t, t)
        return dt

    def advance(self, t):
        self._dt(t)

    def feed(self, b, t):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def set_oracle(self, **kw):
        raise IntegrityError(f"{self.name} is not an oracle front end")

    def state_arrays(self):
        return []

    def state_bytes(self):
        return int(sum(a.nbytes for a in self.state_arrays()))


class FE0(FrontEnd):
    name = "FE0"

    def __init__(self):
        super().__init__()
        self.p = np.zeros(88)
        self.tau = FE_PARAMS["FE0"]["tau_p"]

    def advance(self, t):
        dt = self._dt(t)
        if dt:
            self.p *= math.exp(-dt / self.tau)

    def feed(self, b, t):
        self.advance(t)
        self.p += R_TABLE[int(b)]

    def read(self):
        return _norm(self.p)

    def state_arrays(self):
        return [self.p]


class FE1(FrontEnd):
    name = "FE1"

    def __init__(self):
        super().__init__()
        prm = FE_PARAMS["FE1"]
        self.p = np.zeros(88)
        self.h = np.zeros(88)
        self.tau, self.kappa, self.tau_h = prm["tau_p"], prm["kappa"], prm["tau_h"]

    def advance(self, t):
        dt = self._dt(t)
        if dt:
            self.p *= math.exp(-dt / self.tau)
            self.h *= math.exp(-dt / self.tau_h)

    def feed(self, b, t):
        self.advance(t)
        r = R_TABLE[int(b)]
        self.p += r * (1.0 - self.h)
        self.h += self.kappa * (1.0 - self.h) * r
        np.clip(self.h, 0.0, 1.0, out=self.h)

    def read(self):
        return _norm(self.p)

    def state_arrays(self):
        return [self.p, self.h]


class FE2(FrontEnd):
    name = "FE2"

    def __init__(self):
        super().__init__()
        prm = FE_PARAMS["FE2"]
        self.pf = np.zeros(88)
        self.ps = np.zeros(88)
        self.tf, self.ts, self.w = prm["tau_f"], prm["tau_s"], prm["w"]

    def advance(self, t):
        dt = self._dt(t)
        if dt:
            self.pf *= math.exp(-dt / self.tf)
            self.ps *= math.exp(-dt / self.ts)

    def feed(self, b, t):
        self.advance(t)
        r = R_TABLE[int(b)]
        self.pf += r
        self.ps += r

    def read(self):
        return _norm(np.maximum(self.pf - self.w * self.ps, 0.0))

    def state_arrays(self):
        return [self.pf, self.ps]


class FE3(FrontEnd):
    name = "FE3"

    def __init__(self):
        super().__init__()
        prm = FE_PARAMS["FE3"]
        self.pf = np.zeros(88)
        self.ps = np.zeros(88)
        self.h = np.zeros(88)
        self.tf, self.ts, self.w = prm["tau_f"], prm["tau_s"], prm["w"]
        self.kappa, self.tau_h = prm["kappa"], prm["tau_h"]

    def advance(self, t):
        dt = self._dt(t)
        if dt:
            self.pf *= math.exp(-dt / self.tf)
            self.ps *= math.exp(-dt / self.ts)
            self.h *= math.exp(-dt / self.tau_h)

    def feed(self, b, t):
        self.advance(t)
        r0 = R_TABLE[int(b)]
        r = r0 * (1.0 - self.h)
        self.pf += r
        self.ps += r
        self.h += self.kappa * (1.0 - self.h) * r0
        np.clip(self.h, 0.0, 1.0, out=self.h)

    def read(self):
        return _norm(np.maximum(self.pf - self.w * self.ps, 0.0))

    def state_arrays(self):
        return [self.pf, self.ps, self.h]


class FELAG(FrontEnd):
    """Engineering reference: V51 lag code of the last 12 bytes -> 88 channels."""
    name = "FELAG"
    _IDX = None
    _SGN = None

    def __init__(self):
        super().__init__()
        self.window = bytearray()
        if FELAG._IDX is None:
            rng = np.random.default_rng(sha_seed(FE_PARAMS["FELAG"]["seed"]))
            c = FE_PARAMS["FELAG"]["contacts"]
            FELAG._IDX = np.stack([rng.choice(96, c, replace=False) for _ in range(88)])
            FELAG._SGN = (2 * rng.integers(0, 2, (88, c)) - 1) / math.sqrt(c)

    def feed(self, b, t):
        self.advance(t)
        self.window.append(int(b))
        if len(self.window) > 12:
            del self.window[0]

    def read(self):
        x = v51().context_bits(self.window)
        ch = np.tanh(np.einsum("pc,pc->p", x[FELAG._IDX], FELAG._SGN))
        return _norm(np.maximum(ch, 0.0))

    def state_arrays(self):
        return [np.frombuffer(bytes(self.window), np.uint8)]


class FEID(FrontEnd):
    """Oracle: fixed random 8-of-88 pattern per cue identity (V88 native cues)."""
    name = "FEID"
    oracle = True

    def __init__(self, world):
        super().__init__()
        rng = rng_for(world["seed"], "feid")
        pats, seen = [], set()
        n = world["ncues"] + len(world["neighbours"])
        while len(pats) < n:
            ix = tuple(sorted(int(i) for i in rng.choice(88, 8, replace=False)))
            if ix not in seen:
                seen.add(ix)
                pats.append(ix)
        self.patterns = pats
        self.cue = None

    def set_oracle(self, cue=None, key_offset=None, pattern=None):
        self.cue = cue
        self.override = pattern

    def feed(self, b, t):
        self.advance(t)

    def read(self):
        if getattr(self, "override", None) is not None:
            return np.asarray(self.override, float)
        if self.cue is None:
            raise IntegrityError("FEID read without oracle context")
        p = np.zeros(88)
        p[list(self.patterns[self.cue])] = 1.0
        return p


class FETAIL4(FE0):
    """Oracle: FE0 fed only key bytes 2..5 (V52 invariant-tail analogue)."""
    name = "FETAIL4"
    oracle = True

    def __init__(self):
        super().__init__()
        self.pos = 0
        self.lo = None

    def set_oracle(self, cue=None, key_offset=None, pattern=None):
        self.pos = 0
        self.lo = int(key_offset) + 2

    def feed(self, b, t):
        if self.lo is None:
            raise IntegrityError("FETAIL4 feed without oracle context")
        if self.lo <= self.pos < self.lo + 4:
            super().feed(b, t)
        else:
            self.advance(t)
        self.pos += 1


BIOLOGICAL_FE = ("FE0", "FE1", "FE2", "FE3")


def make_fe(name, world):
    return {"FE0": FE0, "FE1": FE1, "FE2": FE2, "FE3": FE3, "FELAG": FELAG,
            "FETAIL4": FETAIL4}[name]() if name != "FEID" else FEID(world)


# ---------------------------------------------------------------- learners
class F151Learner:
    """FULL_151 frozen; byte front end; innate reinforcer detector (S0)."""
    family = "F151"

    def __init__(self, fe, mode="episode", write=True, us_route=True):
        if mode not in ("episode", "stream"):
            raise ValueError(mode)
        be = native()
        self.common, self.model = be.common, be.model
        self.a = build_full151()
        self.fe, self.mode, self.write, self.us_route = fe, mode, bool(write), bool(us_route)
        self.N = len(self.a.m.adapt)

    # -- internal ------------------------------------------------------------
    def encode(self, p):
        return np.asarray(self.model.encode_sparse(self.a.m, np.asarray(p, float)), float)

    def responses(self, X, fast_lesion=False):
        """Read-only reader-relative (v) and baseline-relative (va) responses, Hz."""
        n = self.a.m.clone()
        if fast_lesion:
            n.fast[:, 1] = 0.0
        dx = self.common.observed_activity(n, np.atleast_2d(X))
        alpha = n.expression(dx)
        pred = self.a.reader.predict(dx)
        base = n.expression(dx, True)
        return (alpha - pred).mean(1), (alpha - base).mean(1)

    def _innate(self, b):
        return 1.0 if (self.us_route and int(b) == AVERSIVE) else 0.0

    # -- learner API (bytes and times only) ------------------------------------
    def present(self, record, t0, episode, gap):
        if not isinstance(record, (bytes, bytearray)):
            raise TypeError("records are bytes")
        L = len(record)
        spacing = float(episode) / L
        cpos = next((i for i, b in enumerate(record) if b in RESERVED), None)
        if cpos is None:
            raise IntegrityError("no innate reinforcer byte in record")
        out = None
        if self.mode == "episode":
            for i in range(cpos):
                self.fe.feed(record[i], t0 + i * spacing)
            self.fe.advance(t0 + cpos * spacing)
            x = self.encode(self.fe.read())
            v, va = self.responses(x)
            out = {"pos": cpos, "raw": np.asarray([v[0]]), "raw_actual": float(va[0]), "code": x}
            self.a.event(float(episode), x, self._innate(record[cpos]), self.write)
            for i in range(cpos, L):
                self.fe.feed(record[i], t0 + i * spacing)
            self.a.event(float(gap))
            self.fe.advance(t0 + float(episode) + float(gap))
        else:
            for i, b in enumerate(record):
                if i == cpos:
                    self.fe.advance(t0 + i * spacing)
                    x = self.encode(self.fe.read())
                    v, va = self.responses(x)
                    out = {"pos": cpos, "raw": np.asarray([v[0]]), "raw_actual": float(va[0]), "code": x}
                self.fe.feed(b, t0 + i * spacing)
                x = self.encode(self.fe.read())
                self.a.event(spacing, x, self._innate(b), self.write)
            self.a.event(float(gap))
            self.fe.advance(t0 + float(episode) + float(gap))
        return out

    def probe(self, context, t, fast_lesion=False):
        """Read-only: clone front end, feed context over one episode, read response."""
        fe = self.fe.clone()
        L = len(context) + 2
        spacing = EPISODE_S / L
        for i, b in enumerate(context):
            fe.feed(b, t + i * spacing)
        fe.advance(t + len(context) * spacing)
        x = self.encode(fe.read())
        v, va = self.responses(x, fast_lesion=fast_lesion)
        return np.asarray([v[0]]), float(va[0])

    def rest(self, seconds):
        self.a.rest(float(seconds))
        self.fe.advance(self.fe.t + float(seconds))

    def clock(self):
        return float(self.a.m.elapsed)

    def state_digest(self):
        h = hashlib.sha256()
        for arr in (self.a.m.fast, self.a.m.slow, self.a.m.adapt,
                    np.asarray([self.a.m.elapsed], float)) + tuple(self.fe.state_arrays()):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(repr(self.fe.t).encode())
        return h.hexdigest()

    def resources(self):
        m = self.a.m
        return {"mutable_learner_bytes": int(m.fast.nbytes + m.slow.nbytes + m.adapt.nbytes),
                "frontend_state_bytes": self.fe.state_bytes(),
                "observer_reader_bytes": int(np.asarray(self.a.reader.Q).nbytes),
                "fixed_B_nnz": int(m.B.nnz), "writable_kcs_expanded": int(self.a.expanded_cells),
                "diagnostic": clean(self.a.diagnostic())}


class V50Learner:
    """V50 raw-BANC core exactly as V51 byte mode (V50-REF), one arm."""
    family = "V50"

    def __init__(self, seed, alphabet, write=True):
        m = v51()
        self.core = m.ByteFeatureCore(int(seed))
        self.C = m.outcome_codebook(int(seed), m.BYTE_OUTPUTS, "byte-code")
        self.mem = m.Memory()
        self.hist = bytearray()
        self.Y = np.frombuffer(bytes(alphabet), np.uint8).astype(int)
        self.CY = self.C[self.Y]
        self.write = bool(write)
        self.N = m.KC_N

    def _h(self, z):
        m = v51()
        ed = m.active_km_edges(z)
        if len(ed):
            return np.bincount(m.KM_POST[ed], weights=self.mem.theta[ed] * z[m.KM_PRE[ed]], minlength=m.MBON_N)
        return np.zeros(m.MBON_N)

    def present(self, record, t0, episode, gap):
        if not isinstance(record, (bytes, bytearray)):
            raise TypeError("records are bytes")
        per_pos = []
        for y0 in record:
            y = int(y0)
            zr, _ze = self.core.features(self.hist, True)
            p = self.mem.predict(zr, self.C)
            per_pos.append((self.CY @ self._h(zr), math.log2(max(float(p[y]), 1e-300)), zr))
            if self.write:
                self.mem.update(zr, y, self.C, p)
            self.hist.append(y)
            if len(self.hist) > 12:
                del self.hist[0]
        return {"per_pos": per_pos}

    def probe(self, context, t, fast_lesion=False):
        zr, _ = self.core.features(bytes(context), use_cache=False)
        return self.CY @ self._h(zr), float("nan")

    def rest(self, seconds):
        return None                      # V50 has no time-dependent state

    def clock(self):
        return None

    def state_digest(self):
        h = hashlib.sha256()
        h.update(self.mem.theta.tobytes())
        h.update(bytes(self.hist))
        h.update(str(self.mem.update_events).encode())
        return h.hexdigest()

    def resources(self):
        return {"mutable_learner_bytes": int(self.mem.theta.nbytes),
                "frontend_state_bytes": 12, "feature_cache_bytes": int(self.core.approx_cache_bytes()),
                "codebook_bytes": int(self.C.nbytes), "adaptor_fixed_bytes": int(self.core.idx.nbytes + self.core.sgn.nbytes),
                "memory_stats": clean(self.mem.stats())}


# ---------------------------------------------------------------- arms
ARMS = {
    "V50-REF": {"family": "V50", "write": True, "nlms": True},
    "V50-NOLEARN": {"family": "V50", "write": False},
    "F151-E-FE0": {"family": "F151", "fe": "FE0", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FE1": {"family": "F151", "fe": "FE1", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FE2": {"family": "F151", "fe": "FE2", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FE3": {"family": "F151", "fe": "FE3", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FELAG": {"family": "F151", "fe": "FELAG", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FEID": {"family": "F151", "fe": "FEID", "mode": "episode", "write": True, "nlms": True},
    "F151-E-FETAIL4": {"family": "F151", "fe": "FETAIL4", "mode": "episode", "write": True, "nlms": True},
    "F151-S-FE0": {"family": "F151", "fe": "FE0", "mode": "stream", "write": True},
    "F151-E-FE0-NOLEARN": {"family": "F151", "fe": "FE0", "mode": "episode", "write": False},
    "F151-E-FE0-USCUT": {"family": "F151", "fe": "FE0", "mode": "episode", "write": True, "us_route": False},
    "F151-E-FEID-NOLEARN": {"family": "F151", "fe": "FEID", "mode": "episode", "write": False},
    "F151-E-FEID-USCUT": {"family": "F151", "fe": "FEID", "mode": "episode", "write": True, "us_route": False},
}
# decoder temperature is fitted on the parent arm; controls inherit it
PARENT = {"V50-NOLEARN": "V50-REF", "F151-E-FE0-NOLEARN": "F151-E-FE0",
          "F151-E-FE0-USCUT": "F151-E-FE0", "F151-E-FEID-NOLEARN": "F151-E-FEID",
          "F151-E-FEID-USCUT": "F151-E-FEID"}


def make_learner(arm_name, world):
    spec = ARMS[arm_name]
    A = ALPHABETS[world["rung"]]
    if spec["family"] == "V50":
        return V50Learner(world["seed"], A, write=spec["write"])
    if world["rung"] != "R2":
        raise ValueError("FULL_151 at S0 expresses only the R2 rung")
    return F151Learner(make_fe(spec["fe"], world), mode=spec["mode"], write=spec["write"],
                       us_route=spec.get("us_route", True))


# ---------------------------------------------------------------- evaluator shadows
class NLMSShadow:
    """CODE-NLMS reference: online softmax regression on the learner's code."""

    def __init__(self, K, N, lrs=NLMS_LRS):
        self.lrs = np.asarray(lrs, float)
        self.W = np.zeros((len(lrs), K, N))
        self.b = np.zeros((len(lrs), K))
        self.K = K

    def step(self, x, y):
        idx = np.flatnonzero(x)
        xv = x[idx]
        logits = self.W[:, :, idx] @ xv + self.b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        logp = np.log2(np.clip(P[:, y], 1e-300, None))
        ok = (P.argmax(axis=1) == y).astype(np.int8)
        g = -P
        g[:, y] += 1.0
        scale = self.lrs / (float(xv @ xv) + 1.0)
        self.W[:, :, idx] += (scale[:, None] * g)[:, :, None] * xv[None, None, :]
        self.b += self.lrs[:, None] * g
        return logp, ok


class MarkovShadow:
    """V51 MARKOV2 on the byte stream (evaluator baseline)."""

    def __init__(self, alphabet):
        self.mk = v51().Markov2()
        self.hist = bytearray()
        self.Y = np.frombuffer(bytes(alphabet), np.uint8).astype(int)

    def probs_Y(self):
        mk, h = self.mk, self.hist
        if len(h) >= 2:
            k = (int(h[-2]) << 8) | int(h[-1])
            cnt, tot = mk.c2[k, self.Y].astype(float), float(mk.t2[k])
        elif len(h) == 1:
            cnt, tot = mk.c1[int(h[-1]), self.Y].astype(float), float(mk.t1[int(h[-1])])
        else:
            cnt, tot = mk.c0[self.Y].astype(float), float(mk.t0)
        p = (cnt + 1.0) / (tot + 256.0)
        return p / p.sum()

    def feed_record(self, record, cpos):
        out = None
        framing = []
        for j, y0 in enumerate(record):
            y = int(y0)
            if j == cpos:
                out = self.probs_Y()
            else:
                framing.append((j, math.log2(self.mk.score(self.hist, y)["prob"])))
            self.mk.update(self.hist, y)
            self.hist.append(y)
            if len(self.hist) > 12:
                del self.hist[0]
        return out, framing


def byte_classes(fmt):
    """Position classes of a record: key positions, consequence, framing."""
    ko = key_offset(fmt)
    L = (6 if fmt == "F1" else 12) + 2
    cls = ["framing"] * L
    for i in range(ko, ko + 6):
        cls[i] = "key"
    cls[L - 2] = "consequence"
    return cls


# ---------------------------------------------------------------- life driver
def run_life(job, *, check_probes=True, learner_override=None):
    """One complete life.  `job` keys: arm, ns, block, index, rung, fmt, density,
    clock, control.  Returns (arrays, summary)."""
    t_start = time.perf_counter()
    arm = ARMS[job["arm"]]
    world = create_world(job["ns"], job["block"], job["index"], job["rung"], job["density"])
    if job["control"] == "unpaired":
        world = unpaired_world(world)
    elif job["control"] != "normal":
        raise ValueError(job["control"])
    fmt = job["fmt"]
    if fmt not in FORMATS:
        raise ValueError(fmt)
    A = ALPHABETS[job["rung"]]
    K = len(A)
    scale = float(job["clock"])
    episode, gap = EPISODE_S * scale, GAP_S * scale
    learner = learner_override(world) if learner_override else make_learner(job["arm"], world)
    fam = learner.family
    oracle = fam == "F151" and learner.fe.oracle
    inv0 = v88().inventory(learner.a) if fam == "F151" else None
    nlms = NLMSShadow(K, learner.N) if (arm.get("nlms") and job["control"] == "normal") else None
    markov = MarkovShadow(A) if (job["arm"] == "V50-REF" and job["control"] == "normal") else None
    keys = world["keys"]
    classes = byte_classes(fmt)
    timeline = TruthTimeline(world)
    clock = 0.0
    presentations = defaultdict(int)
    last_t = {}
    R = defaultdict(list)          # presentation rows
    P = defaultdict(list)          # probe rows
    stages = []
    reg = {"framing_logp": 0.0, "framing_n": 0, "key_logp": 0.0, "key_n": 0,
           "mk_framing_logp": 0.0, "mk_framing_n": 0}
    n_trials = 0

    def set_oracle(fe, cue, key_off, pattern=None):
        fe.set_oracle(cue=cue, key_offset=key_off, pattern=pattern)

    def probe_cues(stage_i, phase, group, cues, truth_of, fmt_eff, wrapper, lesion=False, extra=None):
        for c, tr in zip(cues, truth_of):
            if c >= 0:
                key = keys[c]
            else:
                key = world["neighbours"][-c - 1]["key"]
            ctx = render_context(key, fmt_eff, wrapper)
            if oracle:
                if learner.fe.name == "FEID":
                    pat = None
                    if c < 0:  # neighbours use their own oracle pattern
                        idx = world["ncues"] + (-c - 1)
                        pat = np.zeros(88)
                        pat[list(learner.fe.patterns[idx])] = 1.0
                    set_oracle(learner.fe, c if c >= 0 else None, key_offset(fmt_eff), pat)
                else:
                    set_oracle(learner.fe, c, key_offset(fmt_eff))
            raw, ra = learner.probe(ctx, clock)
            lraw = learner.probe(ctx, clock, fast_lesion=True)[0][0] if (lesion and fam == "F151") else np.nan
            P["stage"].append(stage_i)
            P["phase"].append(PHASE[phase])
            P["group"].append(PGROUP[group])
            P["cue"].append(c)
            P["truth"].append(tr)
            P["t"].append(clock)
            P["raw"].append(np.asarray(raw, float).ravel())
            P["raw_actual"].append(ra)
            P["lesion_raw"].append(lraw)
            P["expo"].append(-1 if extra is None else extra)

    for e in world["events"]:
        k = e["kind"]
        if k == "trial":
            n_trials += 1
            p = e["pair"]
            for side in (0, 1):
                cue = 2 * p + side
                y = int(e["deliver"][side])
                wr = presentations[cue] % 2 if fmt == "F3" else 0
                rec = render_record(keys[cue], fmt, wr, A[y])
                cpos = len(rec) - 2
                if oracle:
                    set_oracle(learner.fe, cue, key_offset(fmt))
                t_pred = clock + cpos * episode / len(rec)
                out = learner.present(rec, clock, episode, gap)
                if fam == "F151":
                    if out["pos"] != cpos:
                        raise IntegrityError("innate detector fired away from the consequence")
                    raw, ra, code = out["raw"], out["raw_actual"], out["code"]
                else:
                    pp = out["per_pos"]
                    raw, ra, code = pp[cpos][0], np.nan, pp[cpos][2]
                    for j, cl in enumerate(classes):
                        if cl == "framing":
                            reg["framing_logp"] += pp[j][1]
                            reg["framing_n"] += 1
                        elif cl == "key":
                            reg["key_logp"] += pp[j][1]
                            reg["key_n"] += 1
                R["cue"].append(cue)
                R["pair"].append(p)
                R["tag"].append(TAG[e["tag"]])
                R["ctx"].append(e["ctx"])
                R["cycle"].append(e["cycle"])
                R["expo"].append(presentations[cue])
                R["t"].append(t_pred)
                R["dt"].append(t_pred - last_t[cue] if cue in last_t else -1.0)
                R["y"].append(y)
                R["truth"].append(timeline.committed(cue))
                R["raw"].append(np.asarray(raw, float).ravel())
                R["raw_actual"].append(ra)
                if nlms is not None:
                    lp, ok = nlms.step(np.asarray(code, float), y)
                    R["nlms_logp"].append(lp)
                    R["nlms_ok"].append(ok)
                if markov is not None:
                    pm, fr = markov.feed_record(rec, cpos)
                    R["mk_logp"].append(math.log2(max(float(pm[y]), 1e-300)))
                    R["mk_ok"].append(int(int(np.argmax(pm)) == y))
                    for j, lpv in fr:
                        if classes[j] == "framing":
                            reg["mk_framing_logp"] += lpv
                            reg["mk_framing_n"] += 1
                presentations[cue] += 1
                last_t[cue] = t_pred
                clock += episode + gap
        elif k == "rest":
            learner.rest(e["seconds"])
            clock += float(e["seconds"])
        elif k == "commit_truth":
            timeline.apply(e)
        elif k == "set_relevance":
            pass
        elif k in ("probe", "react_probe", "revision_curve"):
            before = learner.state_digest() if check_probes else None
            if k == "probe":
                stages.append(e["stage"])
                si = len(stages) - 1
                for g, pairs in e["groups"].items():
                    cues = [2 * q + s for q in pairs for s in (0, 1)]
                    probe_cues(si, e["phase"], g, cues, [timeline.committed(c) for c in cues], fmt, 0,
                               lesion=(e["phase"] == "day" and g == "relevant"))
                if e["phase"] == "day":
                    nb = world["neighbours"]
                    probe_cues(si, "day", "neighbour", [-(j + 1) for j in range(len(nb))],
                               [timeline.committed(x["source"]) for x in nb], fmt, 0)
                    if fmt == "F3":
                        rel = e["groups"].get("relevant", [])
                        cues = [2 * q + s for q in rel for s in (0, 1)]
                        probe_cues(si, "day", "f4", cues, [timeline.committed(c) for c in cues], "F4", 2)
            elif k == "react_probe":
                stages.append(f"react_c{e['challenge']}_x{e['exposures']}")
                si = len(stages) - 1
                for g in ("react", "novel"):
                    cues = [2 * q + s for q in e[g] for s in (0, 1)]
                    probe_cues(si, "react", g, cues, [timeline.committed(c) for c in cues], fmt, 0,
                               extra=e["exposures"])
            else:
                stages.append(f"revision_c{e['challenge']}_x{e['exposures']}")
                si = len(stages) - 1
                cues = [2 * q + s for q in e["pairs"] for s in (0, 1)]
                probe_cues(si, "revision_curve", "revision", cues, [timeline.committed(c) for c in cues], fmt, 0,
                           extra=e["exposures"])
            if check_probes and learner.state_digest() != before:
                raise IntegrityError(f"probe changed learner state at {stages[-1]}")
        else:
            raise IntegrityError(f"unknown event {k}")
        if fam == "F151" and abs(learner.clock() - clock) > 1e-6 * max(1.0, clock):
            raise IntegrityError("learner clock diverged from world clock")
    if fam == "F151" and v88().inventory(learner.a) != inv0:
        raise IntegrityError("learner state shape/bytes changed during life")
    arrays = {}
    for name, vals in R.items():
        arrays["ev_" + name] = np.asarray(vals)
    for name, vals in P.items():
        arrays["pr_" + name] = np.asarray(vals)
    for kk in ("ev_raw", "pr_raw"):
        arrays[kk] = arrays[kk].astype(np.float64)
    n_rows = len(R["cue"])
    if n_rows != 2 * n_trials:
        raise IntegrityError("presentation rows do not match trials")
    summary = {"version": VERSION, "job": job, "arm_spec": arm, "family": fam,
               "world_hash": world_hash(world), "world_seed": world["seed"], "K": K,
               "n_trials": n_trials, "n_rows": n_rows, "n_probe_rows": len(P["cue"]),
               "stages": stages, "regularity": reg, "resources": learner.resources(),
               "final_state_digest": learner.state_digest(), "clock_end": clock,
               "elapsed_s": time.perf_counter() - t_start, "shadows": {"nlms": nlms is not None,
                                                                         "markov": markov is not None}}
    return arrays, summary


# ---------------------------------------------------------------- truth utility
class TruthTimeline:
    """Committed truth for diagnostics (DESIGN.md 3.4).  Revision truth switches
    at the commit event that the world places immediately before the pair's
    first revision presentation; anomaly/mislead truth never switches."""

    def __init__(self, world):
        self.truth = np.asarray(world["truth0"], int).copy()

    def apply(self, event):
        if event["kind"] != "commit_truth":
            raise ValueError("not a commit event")
        p = int(event["pair"])
        self.truth[2 * p:2 * p + 2] = np.asarray(event["truth"], int)

    def committed(self, cue):
        return int(self.truth[int(cue)])


# ---------------------------------------------------------------- scoring
def probs_from_raw(family, raw, T):
    """Decoder with one scalar temperature.  F151: raw = reader-relative response v
    (punished cue lower) -> p(AVERSIVE)=sigmoid(-v/T).  V50: logits raw/T over Y."""
    raw = np.asarray(raw, float)
    if family == "F151":
        v = raw.reshape(-1)
        z = np.clip(-v / T, -700, 700)
        pa = 1.0 / (1.0 + np.exp(-z))
        return np.stack([pa, 1.0 - pa], axis=1)
    L = raw / T
    L = L - L.max(axis=1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=1, keepdims=True)


def predicted_index(family, raw):
    raw = np.asarray(raw, float)
    if family == "F151":
        return np.where(raw.reshape(-1) < 0, 0, 1)
    return raw.argmax(axis=1)


def bits_saved(p_true, K):
    return np.log2(np.clip(p_true, EPS, 1.0 - EPS)) + math.log2(K)


def eta_of(p_true, K):
    s = bits_saved(p_true, K)
    return float(s.sum() / (len(s) * math.log2(K))) if len(s) else float("nan")


def idref_logp(cues, ys, K, lam, alpha0=0.5):
    """IDREF: per-identity Dirichlet with exponential forgetting (evaluator-only)."""
    counts = {}
    out = np.empty(len(cues))
    for i, (c, y) in enumerate(zip(cues, ys)):
        v = counts.get(int(c))
        if v is None:
            v = np.zeros(K)
        p = (v + alpha0) / (v.sum() + alpha0 * K)
        out[i] = p[int(y)]
        v = v * lam
        v[int(y)] += 1.0
        counts[int(c)] = v
    return out


# ---------------------------------------------------------------- interface metrics (Block B)
_ENC = None


def encoder():
    global _ENC
    if _ENC is None:
        _ENC = build_full151()
    return _ENC


def kc_code(p):
    a = encoder()
    return np.asarray(native().model.encode_sparse(a.m, np.asarray(p, float)), bool)


def jaccard(a, b):
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


def interface_metrics(fe_name, fmt, index, n_probe=32):
    """Block B: KC-code geometry of one front end on one life-length stream."""
    w = create_world("DEV", "B", 1000 + int(index), "R2", "massed")
    keys = w["keys"]
    A = ALPHABETS["R2"]
    prng = rng_for(w["seed"], "interface-probes")
    probe_cues = sorted(int(c) for c in prng.choice(2 * w["n_trained"], n_probe, replace=False))
    fe = FE0() if fe_name == "V50LAG" else make_fe(fe_name, w)
    wrappers = (0, 1, 2) if fmt == "F3" else (0,)

    def code_for(state, cue, wrapper, key=None):
        f = state.clone()
        key = keys[cue] if key is None else key
        fmt_eff = "F4" if wrapper == 2 else fmt
        ctx = render_context(key, fmt_eff, wrapper)
        if f.oracle:
            f.set_oracle(cue=cue, key_offset=key_offset(fmt_eff))
        L = len(ctx) + 2
        sp = EPISODE_S / L
        t0 = f.t
        for i, b in enumerate(ctx):
            f.feed(b, t0 + i * sp)
        f.advance(t0 + len(ctx) * sp)
        return kc_code(f.read())

    v50core = v51().ByteFeatureCore(w["seed"]) if fe_name == "V50LAG" else None

    def v50_code(cue, wrapper, key=None):
        key = keys[cue] if key is None else key
        ctx = render_context(key, "F4" if wrapper == 2 else fmt, wrapper)
        zr, _ = v50core.features(ctx, use_cache=False)
        return zr != 0

    if fe_name == "V50LAG":
        start = {(c, wr): v50_code(c, wr) for c in probe_cues for wr in wrappers}
        end = start
        filler = {c: v50_code(c, 0, key=b"000000") for c in probe_cues}
    else:
        start = {(c, wr): code_for(fe, c, wr) for c in probe_cues for wr in wrappers}
        # drive the front end through the whole life stream (no learner)
        clock = 0.0
        pres = defaultdict(int)
        for e in w["events"]:
            if e["kind"] == "trial":
                for side in (0, 1):
                    cue = 2 * e["pair"] + side
                    wr = pres[cue] % 2 if fmt == "F3" else 0
                    rec = render_record(keys[cue], fmt, wr, A[e["deliver"][side]])
                    if fe.oracle:
                        fe.set_oracle(cue=cue, key_offset=key_offset(fmt))
                    sp = EPISODE_S / len(rec)
                    for i, b in enumerate(rec):
                        fe.feed(b, clock + i * sp)
                    clock += EPISODE_S + GAP_S
                    fe.advance(clock)
                    pres[cue] += 1
            elif e["kind"] == "rest":
                clock += float(e["seconds"])
                fe.advance(clock)
        end = {(c, wr): code_for(fe, c, wr) for c in probe_cues for wr in wrappers}
        filler = ({c: code_for(fe, c, 0, key=b"000000") for c in probe_cues}
                  if fe_name != "FEID" else None)
    same = [jaccard(end[(c, 0)], end[(c, 1)]) for c in probe_cues] if fmt == "F3" else [1.0] * len(probe_cues)
    diff = [jaccard(end[(a, 0)], end[(b, 0)]) for i, a in enumerate(probe_cues) for b in probe_cues[i + 1:]]

    def retrieval(q):
        hits = 0
        for c in probe_cues:
            sims = [jaccard(end[(c, q)], end[(g, 0)]) for g in probe_cues]
            hits += int(probe_cues[int(np.argmax(sims))] == c)
        return hits / len(probe_cues)

    J_same, J_diff = float(np.mean(same)), float(np.mean(diff))
    res = {"fe": fe_name, "fmt": fmt, "index": int(index), "J_same": J_same, "J_diff": J_diff,
           "margin": J_same / J_diff if J_diff > 0 else float("inf"),
           "retrieval_w1": retrieval(1) if fmt == "F3" else 1.0,
           "retrieval_w2": retrieval(2) if fmt == "F3" else None,
           "stationarity": float(np.mean([jaccard(start[(c, 0)], end[(c, 0)]) for c in probe_cues])),
           "framing_share": (None if filler is None else float(np.mean(
               [np.count_nonzero(end[(c, 0)] & filler[c]) / max(1, np.count_nonzero(end[(c, 0)]))
                for c in probe_cues]))),
           "active_kcs": float(np.mean([np.count_nonzero(end[(c, 0)]) for c in probe_cues]))}
    return res

