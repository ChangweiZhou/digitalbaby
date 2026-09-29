#!/usr/bin/env python3
"""MiniFly RCE/V87 follow-up runner.

Two source-locked follow-up experiments:
  Round A: independent confirmation of RCE trace-dynamics restoration.
  Round B: feedback x KC-active-fraction causal screen on the Round-A baseline.

Scientific native mode uses the existing V82E MiniFly phenotype and inherited source tree.
The fixture backend is a bounded deterministic test double for engineering tests only.

This runner deliberately does NOT perform evolution and does NOT optimize blanket old-memory
retention. Relevance is evaluator-only behavioral demand; no relevance tag enters the learner.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import importlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

VERSION = "RCE-V87-FOLLOWUP-1.1-AUDITED"
ROOT = Path(__file__).resolve().parent
DESIGN_BASENAME = "DESIGN_RCE_V87_FOLLOWUP_AUDITED.md"
PRIOR_KEYS_BASENAME = "prior_pattern_keys_rce_v87.npz"
PRIOR_META_BASENAME = "prior_pattern_keys_meta.json"

GENES = ("b0", "b_adapt", "b_fast", "b_slow", "g_alpha_gain", "g_support",
         "g_teach_scale", "g_readout", "g_feedback", "g_mbon", "g_k0",
         "g_fast_tau", "g_slow_tau", "g_fraction", "g_sparsity",
         "g_pnkc_weight", "g_pnkc_rewire")
GENE_BOX = {
    "b0": (-0.5, 0.5), "b_adapt": (-0.75, 0.75), "b_fast": (-0.75, 0.75),
    "b_slow": (-0.75, 0.75), "g_alpha_gain": (-0.5, 0.5), "g_support": (0.0, 1.0),
    "g_teach_scale": (-0.5, 0.5), "g_readout": (-0.4, 0.4),
    "g_feedback": (-1.0, 1.0), "g_mbon": (-1.0, 1.0), "g_k0": (-0.3, 0.3),
    "g_fast_tau": (-1.0, 1.0), "g_slow_tau": (-1.0, 1.0), "g_fraction": (-1.0, 1.0),
    "g_sparsity": (-0.5, 0.5), "g_pnkc_weight": (0.0, 0.6), "g_pnkc_rewire": (0.0, 0.5),
}
GENE_LO = np.asarray([GENE_BOX[n][0] for n in GENES], float)
GENE_HI = np.asarray([GENE_BOX[n][1] for n in GENES], float)

REF = [0.0] * 17
V81 = [-.007703, -.612287, -.596517, 0., 0., 0., .197423, .4, .510573,
       -.146094, .000695, .070028, -.002734, 0., -.460409, .032964, 0.]
V84E = [.330853, -.75, -.349499, -.090123, -.152739, 1., -.233097, -.076526,
        .602103, .804988, .042987, .054654, 0., .035756, -.5, .008166, 0.]
RCE_FULL = [
    0.2769583684375018, 0.5060575709162071, 0.75, -0.5013699223698764,
    0.12733383481229846, 0.0, 0.0, 0.0, 0.33202778204816097,
    -0.2930507566579318, -0.18361794385481717, 0.3873791237411004,
    -0.2948909452871429, 0.5068632844705093, -0.4850183373885582,
    0.032964, 0.0,
]
RCE_TRACE = [
    0.2769583684375018, 0.5060575709162071, 0.75, -0.5013699223698764,
    0.0, 0.0, 0.0, 0.0, 0.33202778204816097,
    -0.2930507566579318, -0.18361794385481717, 0.0, 0.0, 0.0,
    -0.4850183373885582, 0.032964, 0.0,
]
STRUCTURAL_DONORS = {"REF": REF, "V81": V81, "V84E": V84E}
EXPECTED_SOURCE_IDS = {"RCE_FULL": "aadcb43cf170d1d018bf", "RCE_TRACE_RESTORED": "ca1b0a55a326a7d3e8e7"}
V82_HASHES = {
    "src/model_evo.py": "ccdeaab99a889e6a74e47f95d641bba79ed95597b8bd902181e1c535942e25d7",
    "src/common_evo.py": "595211c5136bf5c13392d7390ce6c55ecb30f99fe21a537374eb3794cee68ad7",
    "config.json": "685889038c91a156ed49bce45d07a5ea14c0331e9faebd666c9424f861f11d4d",
}
PRIMARY = ("P1_current_usefulness", "P2_relevant_after_write", "P3_revised_relevant_after_write",
           "P4_revision", "P5_acquisition", "P6_anomaly_recovery", "P7_reactivation_4")
LIFE = {
    "block_pairs": [48, 24, 24, 24, 12, 12, 12, 12],
    "challenge_after_blocks": [0, 3, 4, 5, 6],
    "acquisition_bouts": 6,
    "challenge_bouts": 12,
    "stable_pairs": 4,
    "anomaly_pairs": 4,
    "revision_pairs": 4,
    "untrained_pairs": 24,
    "rest_seconds": 86400.0,
    "final_rest_seconds": 604800.0,
}
ROUND_A_HISTORIES = 48
ROUND_B_HISTORIES = 24
BOOTSTRAP_DRAWS = 50000
NI_MARGIN = 0.02

class IntegrityError(RuntimeError):
    pass

class StopRequested(RuntimeError):
    pass


def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): raise ValueError("nonfinite value")
    return x


def canonical(x):
    return json.dumps(clean(x), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_bytes(path, data):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".pending-", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path, body):
    atomic_bytes(path, json.dumps(clean(body), indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def sealed_write(path, body):
    wrapper = {"sha256": digest(body), "body": clean(body)}
    atomic_bytes(path, gzip.compress(canonical(wrapper), compresslevel=3, mtime=0))


def sealed_read(path):
    wrapper = json.loads(gzip.decompress(Path(path).read_bytes()))
    if digest(wrapper["body"]) != wrapper["sha256"]:
        raise IntegrityError(f"corrupt sealed file: {path}")
    return wrapper["body"]


def rng_for(*parts):
    return np.random.default_rng(int(digest(parts)[:16], 16))


def sha_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def validate_genes(genes, context="genes"):
    x = np.asarray(genes, float)
    if x.shape != (17,) or not np.isfinite(x).all(): raise ValueError(f"{context}: 17 finite genes required")
    bad = np.flatnonzero((x < GENE_LO - 1e-12) | (x > GENE_HI + 1e-12))
    if len(bad):
        s = ", ".join(f"{GENES[i]}={x[i]:.12g} not in [{GENE_LO[i]:.12g},{GENE_HI[i]:.12g}]" for i in bad)
        raise ValueError(f"{context}: frozen V82E gene-box violation: {s}")
    return np.minimum(np.maximum(x, GENE_LO), GENE_HI).copy()


def legacy_candidate_id(genes, donor):
    x = validate_genes(genes)
    return digest([x.tolist(), donor])[:20]


def pattern_key_from_indices(indices):
    lo = 0; hi = 0
    for j in indices:
        j = int(j)
        if not 0 <= j < 88: raise ValueError("raw cue index outside 0..87")
        if j < 64: lo |= 1 << j
        else: hi |= 1 << (j - 64)
    return (lo, hi)


def pattern_key(row):
    return pattern_key_from_indices(np.flatnonzero(np.asarray(row)))


def load_prior_keys(package_root=ROOT):
    p = Path(package_root) / PRIOR_KEYS_BASENAME
    meta = Path(package_root) / PRIOR_META_BASENAME
    if not p.is_file() or not meta.is_file():
        raise FileNotFoundError(f"bundled prior-cue exclusion files missing beside runner: {p.name}, {meta.name}")
    m = json.loads(meta.read_text())
    if file_hash(p) != m["sha256"]: raise IntegrityError("prior-pattern bundle hash mismatch")
    with np.load(p, allow_pickle=False) as z:
        arr = np.asarray(z["keys"], np.uint64)
    if arr.ndim != 2 or arr.shape[1] != 2: raise IntegrityError("bad prior-pattern key file")
    keys = {(int(a), int(b)) for a, b in arr}
    if len(keys) != int(m["count"]): raise IntegrityError("prior-pattern key count mismatch")
    return keys, m


def numeric_arrays(obj):
    found, seen, buffers = {}, set(), set()
    def visit(o, path, depth):
        if id(o) in seen or depth > 10: return
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            base = o
            while isinstance(base.base, np.ndarray): base = base.base
            if id(base) not in buffers and base.dtype.kind in "biufc":
                buffers.add(id(base)); found[path] = base
            return
        if isinstance(o, dict):
            for k, v in sorted(o.items(), key=lambda kv: str(kv[0])): visit(v, path + "/" + str(k), depth + 1)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o): visit(v, path + f"/{i}", depth + 1)
        elif not isinstance(o, (type, str, bytes, int, float, bool, type(None))) and hasattr(o, "__dict__"):
            import types
            if not isinstance(o, (types.ModuleType, types.FunctionType, types.MethodType)):
                visit(vars(o), path, depth + 1)
    visit(obj, "agent", 0)
    return found


def inventory(obj):
    out = {}
    for name, arr in numeric_arrays(obj).items():
        if not np.isfinite(arr).all(): raise IntegrityError(f"nonfinite state in {name}")
        out[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "bytes": int(arr.nbytes)}
    return out


def array_digest(obj):
    h = hashlib.sha256()
    for name, arr in sorted(numeric_arrays(obj).items()):
        h.update(name.encode()); h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def sparse_digest(b):
    b = b.tocsr()
    return digest({"shape": b.shape, "data": b.data, "indices": b.indices, "indptr": b.indptr})


def arm_spec(name, genes, donor, *, active_fraction_override=None, source_candidate_id=None):
    x = validate_genes(genes, name)
    if donor not in STRUCTURAL_DONORS: raise ValueError(f"{name}: unknown donor {donor}")
    if not np.allclose(x[15:17], np.asarray(STRUCTURAL_DONORS[donor], float)[15:17], atol=1e-12, rtol=0):
        raise ValueError(f"{name}: structural genes do not match declared donor")
    intervention = {"active_fraction_override": None if active_fraction_override is None else float(active_fraction_override)}
    if active_fraction_override is not None and not 0 < float(active_fraction_override) < 1:
        raise ValueError("active-fraction override outside (0,1)")
    legacy = legacy_candidate_id(x, donor)
    if source_candidate_id is not None and legacy != source_candidate_id:
        raise IntegrityError(f"{name}: source candidate ID mismatch {legacy} != {source_candidate_id}")
    body = {"name": name, "genes": x.tolist(), "donor": donor, "intervention": intervention,
            "candidate_id": legacy, "source_candidate_id": source_candidate_id}
    body["arm_hash"] = digest({k: body[k] for k in ("genes", "donor", "intervention")})
    return body


def round_a_arms():
    return [
        arm_spec("REF", REF, "REF"),
        arm_spec("V84E", V84E, "V84E"),
        arm_spec("RCE_FULL", RCE_FULL, "V81", source_candidate_id=EXPECTED_SOURCE_IDS["RCE_FULL"]),
        arm_spec("RCE_TRACE_RESTORED", RCE_TRACE, "V81", source_candidate_id=EXPECTED_SOURCE_IDS["RCE_TRACE_RESTORED"]),
    ]


def round_b_arms(base):
    if base not in ("RCE_FULL", "RCE_TRACE_RESTORED"): raise ValueError("bad Round-B base")
    genes = list(RCE_FULL if base == "RCE_FULL" else RCE_TRACE)
    native_frac = 0.05 * 2 ** genes[14]
    out = []
    for f_name, f_gene in (("Fnative", genes[8]), ("Fref", 0.0)):
        for a_name, override in (("Anative", None), ("A3", 0.030), ("A2p5", 0.025)):
            g = genes.copy(); g[8] = f_gene
            out.append(arm_spec(f"{f_name}_{a_name}", g, "V81", active_fraction_override=override))
    out += [arm_spec("REF", REF, "REF"), arm_spec("V84E", V84E, "V84E")]
    # Exact six-cell contract: only feedback gene and explicit active override may differ.
    for a in out[:6]:
        for i in range(17):
            if i != 8 and abs(a["genes"][i] - genes[i]) > 1e-12:
                raise IntegrityError(f"Round-B arm {a['name']} changed non-feedback gene {GENES[i]}")
    return out


class NativeBackend:
    scientific = True
    def __init__(self, spec):
        self.spec = spec
        self.project = Path(spec["project_root"]).expanduser().resolve()
        for rel in ("iteration23/src/common78.py", "iteration24/src/common79.py"):
            if not (self.project / rel).is_file(): raise FileNotFoundError(f"native MiniFly dependency missing: {self.project / rel}")
        self.v82 = self.project / "V82E"
        for rel, expected in V82_HASHES.items():
            p = self.v82 / rel
            if not p.is_file(): raise FileNotFoundError(f"required source missing: {p}")
            if file_hash(p) != expected: raise IntegrityError(f"source-lock mismatch: {p}")
        if not (3, 10) <= sys.version_info[:2] <= (3, 12): raise RuntimeError("use Python 3.11 (3.10-3.12 supported)")
        try:
            import numba, scipy
        except Exception as e:
            raise RuntimeError("native dependencies missing; use bundled launcher") from e
        if tuple(map(int, np.__version__.split(".")[:2])) > (2, 2): raise RuntimeError("native source requires NumPy <=2.2")
        for k in ("pyarrow", "numexpr", "bottleneck"): sys.modules.setdefault(k, None)
        sys.path.insert(0, str(self.v82 / "src"))
        self.common = importlib.import_module("common_evo")
        self.model = importlib.import_module("model_evo")
        if Path(self.model.__file__).resolve() != (self.v82 / "src/model_evo.py").resolve():
            raise IntegrityError("wrong model_evo imported")
        if tuple(self.model.GENE_NAMES) != GENES: raise IntegrityError("native gene order changed")
        native_box = {str(n): tuple(map(float, self.model.BOX[n])) for n in self.model.GENE_NAMES}
        if native_box != GENE_BOX: raise IntegrityError("native frozen gene boxes changed")
        self.ref_q = np.asarray(self.model.EvoLearner(np.zeros(17)).m.Q, float).copy()
        self.donors = {k: self.model.EvoLearner(np.asarray(g, float)).m.B.copy().tocsr()
                       for k, g in STRUCTURAL_DONORS.items()}
        self.env = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                    "numba": numba.__version__, "platform": sys.platform}

    def make(self, arm):
        validate_genes(arm["genes"], arm["name"])
        a = self.model.EvoLearner(np.asarray(arm["genes"], float))
        a.m.B = self.donors[arm["donor"]].copy().tocsr()
        a.coupled_reader = a.reader
        a.reader = self.common.FrozenReader(self.ref_q.copy(), self.common.READER)
        if not np.array_equal(a.reader.Q, self.ref_q): raise IntegrityError("fixed REF-Q reader drift")
        genome_fraction = float(getattr(a.m, "active_fraction", 0.05))
        override = arm["intervention"].get("active_fraction_override")
        if override is not None:
            a.m.active_fraction = float(override)
        a.phenotype = dict(a.phenotype)
        a.phenotype["genome_active_fraction"] = genome_fraction
        a.phenotype["effective_active_fraction"] = float(getattr(a.m, "active_fraction", genome_fraction))
        a.phenotype["active_fraction_override"] = override
        return a

    def encode(self, a, pn):
        return np.asarray([self.model.encode_sparse(a.m, p) for p in pn], float)

    def teach(self, a, code_pair, role, write=True):
        a.learn_pair(code_pair, int(role), write=bool(write))

    def rest(self, a, seconds): a.rest(float(seconds))
    def clone(self, a): return a.clone()
    def elapsed(self, a): return float(a.m.elapsed)
    def stat_object(self, a): return a
    def fast_tau(self, a): return float(a.m.kernel["fast_tau"])
    def write_state(self, a): return np.asarray(a.m.fast[:, 1], float).copy(), np.asarray(a.m.slow, float).copy()

    def probe(self, a, codes, roles, lesion=None):
        n = a.m.clone()
        if lesion in ("fast", "both"): n.fast[:, 1] = 0.0
        if lesion in ("slow", "both"): n.slow[:] = 0.0
        dx = self.common.observed_activity(n, codes)
        alpha = n.expression(dx); baseline = n.expression(dx, True); pred_fixed = a.reader.predict(dx); pred_coupled = a.coupled_reader.predict(dx)
        direction = 2 * np.asarray(roles, float) - 1
        ca, rf, rc = [(alpha - p).mean(1) for p in (baseline, pred_fixed, pred_coupled)]
        actual = direction * (ca[::2] - ca[1::2]); fixed = direction * (rf[::2] - rf[1::2]); coupled = direction * (rc[::2] - rc[1::2])
        ok_a = actual >= 1.0
        out = {"actual": actual, "fixed": fixed, "coupled": coupled,
               "correct": ok_a & (fixed >= a.reader.pair_threshold), "correct_actual": ok_a,
               "clip_fraction": float(np.mean((alpha <= -11.2 + 1e-10) | (alpha >= 19.96 - 1e-10)))}
        if not all(np.isfinite(np.asarray(v)).all() for v in out.values()): raise IntegrityError("nonfinite native probe")
        return out

    def metadata(self, a):
        d = a.diagnostic()
        return {"phenotype": clean(dict(a.phenotype)), "B_sha256": sparse_digest(a.m.B),
                "expanded_cells": int(a.expanded_cells), "diagnostic": clean(d),
                "reader_Q_sha256": hashlib.sha256(np.ascontiguousarray(a.reader.Q).tobytes()).hexdigest(),
                "fast_bytes": int(a.m.fast.nbytes), "slow_bytes": int(a.m.slow.nbytes)}


class FixtureAgent:
    def __init__(self, arm, units=64):
        self.genes = np.asarray(arm["genes"], float)
        self.fast = np.zeros(units); self.slow = np.zeros(units); self.clock = 0.0
        self.B = rng_for("fixture-B", arm["donor"]).normal(size=(88, units))
        self.write_l1 = 0.0
        self.active_fraction = arm["intervention"].get("active_fraction_override")
        if self.active_fraction is None: self.active_fraction = 0.05 * 2 ** self.genes[14]
        self.phenotype = {"feedback_scale": float(2 ** self.genes[8]), "fast_tau": float(40000 * 2 ** self.genes[11]),
                          "slow_tau": float(2000000 * 2 ** self.genes[12]), "genome_active_fraction": float(0.05 * 2 ** self.genes[14]),
                          "effective_active_fraction": float(self.active_fraction), "active_fraction_override": arm["intervention"].get("active_fraction_override")}


class FixtureBackend:
    scientific = False
    def __init__(self, spec): self.spec = spec; self.units = int(spec.get("fixture_units", 64)); self.env = {"fixture": True, "numpy": np.__version__, "python": sys.version}
    def make(self, arm): return FixtureAgent(arm, self.units)
    def encode(self, a, pn):
        z = pn @ a.B
        k = max(2, int(math.ceil(self.units * a.active_fraction)))
        ix = np.argsort(z, axis=1, kind="stable")[:, -k:]
        out = np.zeros_like(z); np.put_along_axis(out, ix, 1 / math.sqrt(k), axis=1)
        return out
    def teach(self, a, code_pair, role, write=True):
        self.rest(a, 330.0)
        x = np.asarray(code_pair[0] - code_pair[1], float)
        if not write: return
        y = 2 * int(role) - 1
        pred = float(x @ (a.fast + a.slow))
        feedback = 2 ** a.genes[8]
        eta = 0.34 * 2 ** a.genes[6] / max(0.7, feedback)
        update = eta * (2.6 * y - pred) * x
        z = np.clip(.30 + .16 * a.genes[13] + .06 * a.genes[0], .03, .8)
        a.fast += (1 - z) * update; a.slow += z * update
        np.clip(a.fast, -3, 3, out=a.fast); np.clip(a.slow, -3, 3, out=a.slow)
        a.write_l1 += float(np.abs(update).sum())
    def rest(self, a, seconds):
        a.fast *= math.exp(-float(seconds) / a.phenotype["fast_tau"])
        a.slow *= math.exp(-float(seconds) / a.phenotype["slow_tau"])
        a.clock += float(seconds)
    def clone(self, a): return copy.deepcopy(a)
    def elapsed(self, a): return a.clock
    def stat_object(self, a): return a
    def fast_tau(self, a): return float(a.phenotype["fast_tau"])
    def write_state(self, a): return a.fast.copy(), a.slow.copy()
    def probe(self, a, codes, roles, lesion=None):
        f = np.zeros_like(a.fast) if lesion in ("fast", "both") else a.fast
        s = np.zeros_like(a.slow) if lesion in ("slow", "both") else a.slow
        w = f + s
        m = (2 * np.asarray(roles) - 1) * ((codes[::2] - codes[1::2]) @ w)
        return {"actual": m, "fixed": m.copy(), "coupled": m.copy(), "correct": m >= 1.0,
                "correct_actual": m >= 1.0, "clip_fraction": float(np.mean(np.abs(w) >= 3.0))}
    def metadata(self, a):
        return {"phenotype": clean(a.phenotype), "B_sha256": hashlib.sha256(np.ascontiguousarray(a.B).tobytes()).hexdigest(),
                "reader_Q_sha256": "fixed-fixture", "expanded_cells": 0,
                "diagnostic": {"fixture_write_L1": a.write_l1}, "fast_bytes": a.fast.nbytes, "slow_bytes": a.slow.nbytes}


_BACKENDS = {}
def get_backend(spec):
    key = digest(spec)
    if key not in _BACKENDS:
        _BACKENDS[key] = NativeBackend(spec) if spec["backend"] == "native" else FixtureBackend(spec)
    return _BACKENDS[key]


def pick_fresh(rng, n, count, *, excluded, challenged, introduced_block):
    pool = [i for i in range(n) if i not in excluded and i not in challenged]
    if len(pool) < count: raise IntegrityError("insufficient challenge-fresh identities")
    pool.sort(key=lambda i: (introduced_block[i], i))
    strata = [list(map(int, x)) for x in np.array_split(np.asarray(pool, int), min(4, len(pool))) if len(x)]
    for s in strata: rng.shuffle(s)
    out = []
    while len(out) < count:
        progress = False
        for s in strata:
            if s:
                out.append(int(s.pop())); progress = True
                if len(out) == count: break
        if not progress: break
    if len(out) != count: raise IntegrityError("fresh stratified selection failed")
    return out


def derive_history_seed(round_name, index):
    prefix = "MiniFly-RCEV87-FOLLOWUP-A|20260916|" if round_name == "A" else "MiniFly-RCEV87-FOLLOWUP-B|20260916|"
    return sha_seed(prefix + str(index))


def make_raw_rows(rng, count, excluded_keys):
    rows = []; keys = []
    while len(rows) < count:
        ix = tuple(sorted(map(int, rng.choice(88, 8, replace=False))))
        k = pattern_key_from_indices(ix)
        if k in excluded_keys: continue
        excluded_keys.add(k); keys.append(k)
        row = np.zeros(88, float); row[list(ix)] = 1.0; rows.append(row)
    return rows, keys


def create_world(round_name, index, excluded_keys):
    seed = derive_history_seed(round_name, index); rng = np.random.default_rng(seed)
    blocks = list(LIFE["block_pairs"]); total = sum(blocks); extra = LIFE["untrained_pairs"]
    rows, new_keys = make_raw_rows(rng, 2 * (total + extra), excluded_keys)
    nroles = total + extra
    roles = np.concatenate([rng.permutation(np.asarray([0, 1], int)) for _ in range((nroles + 1)//2)])[:nroles].astype(int)
    introduced = []
    for bi, size in enumerate(blocks): introduced.extend([bi] * size)
    current = roles.copy(); events = []; challenge_map = []; ever_revised = set(); challenged = set(); ever_reactivated = set()
    future_reserved = set(); react_plan = {}
    start = 0; challenge_index = 0; stable_history = {}

    def acquire(start_i, end_i, bi):
        for pair in rng.permutation(np.arange(start_i, end_i)):
            for _ in range(LIFE["acquisition_bouts"]):
                events.append({"kind": "teach", "pair": int(pair), "role": int(current[pair]), "context": "acquisition"})
        common = {"n": end_i, "new": list(range(start_i, end_i)), "revision": [], "anomaly": [], "stable": []}
        events.append({"kind": "probe", "stage": f"block{bi}_immediate", "phase": "immediate", "probe_type": "acquire", **common})
        events.append({"kind": "rest", "seconds": LIFE["rest_seconds"]})
        events.append({"kind": "probe", "stage": f"block{bi}_day", "phase": "day", "probe_type": "acquire", **common})

    def challenge(n, ci, after_block):
        nonlocal future_reserved
        excluded = set(future_reserved) | set(ever_reactivated)
        # All non-reactivation challenge targets are challenge-fresh. This prevents
        # accidental reuse from masquerading as planned reactivation or revision.
        rev = pick_fresh(rng, n, LIFE["revision_pairs"], excluded=excluded, challenged=challenged, introduced_block=introduced)
        excluded.update(rev)
        anomaly = pick_fresh(rng, n, LIFE["anomaly_pairs"], excluded=excluded, challenged=challenged, introduced_block=introduced)
        excluded.update(anomaly)
        reactivated = list(react_plan.get(ci, []))
        if ci in (0, 1):
            stable = pick_fresh(rng, n, LIFE["stable_pairs"], excluded=excluded, challenged=challenged, introduced_block=introduced)
        else:
            fresh = pick_fresh(rng, n, 2, excluded=excluded | set(reactivated), challenged=challenged, introduced_block=introduced)
            stable = reactivated + fresh
        if len(set(stable + anomaly + rev)) != 12: raise IntegrityError("challenge groups overlap")
        if any(p in ever_revised for p in reactivated): raise IntegrityError("reactivation cue was revised")
        challenged.update(rev); challenged.update(anomaly); challenged.update(stable)
        # Reused planned cues remain excluded from all later challenges after reactivation,
        # keeping the six episodes interpretable and unique.
        ever_reactivated.update(reactivated)
        future_reserved -= set(reactivated)
        stage = f"challenge{ci}_after_block{after_block}"
        for kind, ids in (("STABLE", stable), ("ANOMALY", anomaly), ("REV", rev)):
            for p in ids:
                challenge_map.append({"stage": stage, "challenge": ci, "kind": kind, "memory_index": int(p),
                                      "introduced_block": int(introduced[p]), "reactivated": bool(p in reactivated)})
        all_target = stable + anomaly + rev
        events.append({"kind": "set_relevance_context", "indices": list(map(int, all_target)), "challenge": ci})
        if reactivated:
            events.append({"kind": "reactivation_probe", "challenge": ci, "exposures": 0, "indices": reactivated})
        anomaly_wrong_bouts = {int(p): 1 + (j % 2) for j, p in enumerate(anomaly)}
        desired = [1 - int(current[p]) for p in rev]
        for cycle in range(LIFE["challenge_bouts"]):
            for pair in rng.permutation(np.asarray(all_target, int)):
                pair = int(pair); role = int(current[pair])
                if pair in rev: role = 1 - role
                elif pair in anomaly and cycle < anomaly_wrong_bouts[pair]: role = 1 - role
                events.append({"kind": "teach", "pair": pair, "role": role, "context": "challenge"})
            ex = cycle + 1
            if reactivated and ex in (1, 2, 4):
                events.append({"kind": "reactivation_probe", "challenge": ci, "exposures": ex, "indices": reactivated})
            if ex in (1, 2, 4, 8, LIFE["challenge_bouts"]):
                events.append({"kind": "revision_curve", "challenge": ci, "exposures": ex, "indices": list(rev), "desired": list(desired)})
        events.append({"kind": "change_truth", "indices": list(rev)})
        current[rev] = 1 - current[rev]; ever_revised.update(rev)
        common = {"probe_type": "challenge", "n": n, "new": [], "revision": list(rev), "anomaly": list(anomaly), "stable": list(stable)}
        events.append({"kind": "probe", "stage": stage + "_immediate", "phase": "immediate", **common})
        events.append({"kind": "rest", "seconds": LIFE["rest_seconds"]})
        events.append({"kind": "probe", "stage": stage + "_day", "phase": "day", **common})
        stable_history[ci] = list(stable)
        # Reserve exact future reactivation cues deterministically before any intervening challenge can reuse them.
        if ci == 0:
            perm = list(map(int, rng.permutation(np.asarray(stable, int))))
            react_plan[2] = perm[:2]; react_plan[3] = perm[2:4]; future_reserved.update(perm)
        elif ci == 1:
            perm = list(map(int, rng.permutation(np.asarray(stable, int))))
            react_plan[4] = perm[:2]; future_reserved.update(perm[:2])

    challenge_after = set(LIFE["challenge_after_blocks"])
    for bi, size in enumerate(blocks):
        end = start + size; acquire(start, end, bi)
        if bi in challenge_after:
            challenge(end, challenge_index, bi); challenge_index += 1
        start = end
    events.append({"kind": "rest", "seconds": LIFE["final_rest_seconds"]})
    events.append({"kind": "probe", "stage": "final_7day", "phase": "final", "probe_type": "time_only",
                   "n": total, "new": [], "revision": [], "anomaly": [], "stable": []})
    # Prespecified mechanism cues: current cue at 96 comes from challenge0; current cue at 168 from challenge4.
    mechanism = {"at96_current": int(stable_history[0][0]), "at168_current": int(stable_history[4][0]),
                 "untrained_pair": int(total)}
    world = {"round": round_name, "index": index, "seed": seed, "pn": np.asarray(rows), "roles": roles,
             "events": events, "introduced_block": introduced, "challenge_map": challenge_map,
             "reactivation_plan": {str(k): list(map(int, v)) for k, v in sorted(react_plan.items())},
             "total": total, "untrained_pairs": extra, "mechanism": mechanism,
             "pattern_keys": [[int(a), int(b)] for a, b in new_keys]}
    audit_world(world)
    return world


def audit_world(w):
    if int(w["total"]) != 168: raise IntegrityError("scientific world must contain 168 trained pairs")
    keys = [tuple(map(int, x)) for x in w["pattern_keys"]]
    if len(keys) != len(set(keys)): raise IntegrityError("raw cue duplicated within world")
    react_events = [e for e in w["events"] if e["kind"] == "reactivation_probe" and e["exposures"] == 4]
    if sum(len(e["indices"]) for e in react_events) != 6: raise IntegrityError("world does not contain exactly six reactivation episodes")
    seen_react = []
    challenge_contexts = {}
    for e in w["events"]:
        if e["kind"] == "set_relevance_context": challenge_contexts[int(e["challenge"])] = set(map(int, e["indices"]))
        if e["kind"] == "reactivation_probe" and e["exposures"] == 4: seen_react += list(map(int, e["indices"]))
    if len(set(seen_react)) != 6: raise IntegrityError("reactivation cue reused")
    # Planned reactivation must have been in an earlier context, absent from the immediately prior context.
    for ci in (2, 3, 4):
        ids = set(map(int, w["reactivation_plan"][str(ci)]))
        if not any(ids <= challenge_contexts[j] for j in range(ci - 1)): raise IntegrityError("reactivation cue was never previously relevant")
        if ids & challenge_contexts[ci - 1]: raise IntegrityError("reactivation cue did not deactivate before return")
    # Genuine revision identities unique across life.
    rev = [m["memory_index"] for m in w["challenge_map"] if m["kind"] == "REV"]
    if len(rev) != len(set(rev)): raise IntegrityError("revision cue revised twice")
    return True


def world_path(round_dir, index): return Path(round_dir) / "worlds" / f"world_{index:03d}.json.gz"


def prepare_worlds(round_name, round_dir, n_histories, prior_keys):
    round_dir = Path(round_dir); (round_dir / "worlds").mkdir(parents=True, exist_ok=True)
    prior_base = set(prior_keys)
    existing_keys = set(prior_keys)
    # Round B caller adds Round-A keys before entering here.
    manifest = []
    for i in range(n_histories):
        p = world_path(round_dir, i)
        if p.exists():
            w = sealed_read(p); audit_world(w)
            wk = {tuple(map(int, x)) for x in w["pattern_keys"]}
            if existing_keys & wk: raise IntegrityError(f"world {i} overlaps prior/earlier raw cues")
            existing_keys |= wk
        else:
            w = create_world(round_name, i, existing_keys)
            sealed_write(p, w)
        manifest.append({"index": i, "seed": int(w["seed"]), "world_hash": digest(w), "file": str(p.name),
                         "raw_pattern_count": len(w["pattern_keys"])})
    atomic_json(round_dir / "WORLD_MANIFEST.json", {"round": round_name, "histories": n_histories, "worlds": manifest})
    all_keys=set()
    for i in range(n_histories):
        ww=sealed_read(world_path(round_dir,i)); wk={tuple(map(int,x)) for x in ww["pattern_keys"]}
        if all_keys & wk: raise IntegrityError("raw cue overlap across worlds after preparation")
        all_keys |= wk
    overlap = len(prior_base & all_keys)
    if overlap: raise IntegrityError(f"world partition audit found {overlap} prior-cue overlaps")
    atomic_json(round_dir/"WORLD_PARTITION_AUDIT.json", {"round":round_name,"histories":n_histories,"new_raw_patterns":len(all_keys),
        "prior_pattern_count":len(prior_base),"overlap_with_prior":overlap,"within_round_duplicates":0,"status":"PASS"})
    return manifest


def load_world(path):
    w = sealed_read(path); w["pn"] = np.asarray(w["pn"], float); w["roles"] = np.asarray(w["roles"], int); audit_world(w); return w


def source_signature(backend_spec):
    sig = {"runner": file_hash(__file__), "version": VERSION,
           "design": file_hash(ROOT / DESIGN_BASENAME), "prior_keys": file_hash(ROOT / PRIOR_KEYS_BASENAME)}
    if backend_spec["backend"] == "native":
        project = Path(backend_spec["project_root"]).expanduser().resolve(); v82 = project / "V82E"
        sig["native"] = {}
        for rel, expected in V82_HASHES.items():
            p = v82 / rel; got = file_hash(p) if p.is_file() else None
            if got != expected: raise IntegrityError(f"native source-lock mismatch: {p}")
            sig["native"][f"V82E/{rel}"] = got
        sig["environment"] = get_backend(backend_spec).env
    else:
        sig["environment"] = get_backend(backend_spec).env
    return sig


def probe_subset(backend, a, codes, roles, indices, lesion=None):
    ids = list(map(int, indices))
    if not ids: return {"n": 0, "correct_n": 0, "accuracy": None, "correct": []}
    cc = np.asarray([codes[2*i:2*i+2] for i in ids]).reshape(2*len(ids), -1)
    rr = np.asarray([roles[i] for i in ids], int)
    v = backend.probe(a, cc, rr, lesion=lesion)
    return {"n": len(ids), "correct_n": int(np.sum(v["correct"])), "accuracy": float(np.mean(v["correct"])),
            "correct": clean(v["correct"]), "actual": clean(v["actual"]), "fixed": clean(v["fixed"]),
            "clip_fraction": float(v["clip_fraction"])}


def lesion_panel(backend, a, codes, roles, relevant, label, rest_seconds=0.0):
    before = array_digest(backend.stat_object(a))
    branch = backend.clone(a)
    if rest_seconds: backend.rest(branch, rest_seconds)
    out = {"label": label, "rest_seconds": float(rest_seconds), "relevant_indices": list(map(int, sorted(relevant)))}
    for lesion in (None, "fast", "slow", "both"):
        key = "intact" if lesion is None else lesion
        out[key] = probe_subset(backend, branch, codes, roles, sorted(relevant), lesion=lesion)
    if array_digest(backend.stat_object(a)) != before: raise IntegrityError("clone lesion panel modified continuing state")
    return out


def write_event(backend, branch, codes, roles, pair, role):
    f0, s0 = backend.write_state(branch); meta0 = backend.metadata(branch)
    backend.teach(branch, codes[2*pair:2*pair+2], int(role), write=True)
    f1, s1 = backend.write_state(branch); meta1 = backend.metadata(branch)
    return {"pair": int(pair), "role": int(role), "delta_fast": f1-f0, "delta_slow": s1-s0,
            "fast_l1": float(np.abs(f1-f0).sum()), "slow_l1": float(np.abs(s1-s0).sum()),
            "metadata_before": meta0, "metadata_after": meta1}


def cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if den == 0 else float(np.dot(a, b) / den)


def mechanism_panel(backend, a, codes, roles, current_pair, untrained_pair, label):
    before = array_digest(backend.stat_object(a)); true = int(roles[current_pair]); untrained_role = int(roles[untrained_pair])
    # M1 repeat-same sequentially on one clone.
    b1 = backend.clone(a); e1 = write_event(backend, b1, codes, roles, current_pair, true); e2 = write_event(backend, b1, codes, roles, current_pair, true)
    # M2 current-versus-never-trained from identical snapshots.
    b2a = backend.clone(a); b2b = backend.clone(a)
    e2a = write_event(backend, b2a, codes, roles, current_pair, true); e2b = write_event(backend, b2b, codes, roles, untrained_pair, untrained_role)
    # M3 correct-versus-reverse from identical snapshots.
    b3a = backend.clone(a); b3b = backend.clone(a)
    e3a = write_event(backend, b3a, codes, roles, current_pair, true); e3b = write_event(backend, b3b, codes, roles, current_pair, 1-true)
    if array_digest(backend.stat_object(a)) != before: raise IntegrityError("mechanism branch modified continuing state")
    def slim(e):
        return {k: clean(v) for k,v in e.items() if k not in ("delta_fast", "delta_slow")}
    return {"label": label, "current_pair": current_pair, "untrained_pair": untrained_pair,
            "repeat_same_slow_cosine": cosine(e1["delta_slow"], e2["delta_slow"]),
            "current_vs_new_slow_cosine": cosine(e2a["delta_slow"], e2b["delta_slow"]),
            "correct_vs_reverse_slow_cosine": cosine(e3a["delta_slow"], e3b["delta_slow"]),
            "repeat_same_fast_cosine": cosine(e1["delta_fast"], e2["delta_fast"]),
            "current_vs_new_fast_cosine": cosine(e2a["delta_fast"], e2b["delta_fast"]),
            "correct_vs_reverse_fast_cosine": cosine(e3a["delta_fast"], e3b["delta_fast"]),
            "events": {"repeat1": slim(e1), "repeat2": slim(e2), "current": slim(e2a), "new": slim(e2b),
                       "correct": slim(e3a), "reverse": slim(e3b)}}


def evaluate_life(backend, arm, world, *, control="normal", round_name="A", stop_path=None, diagnostics=True):
    a = backend.make(arm); start_inv = inventory(backend.stat_object(a)); codes = backend.encode(a, world["pn"]); roles = world["roles"].copy()
    current_relevant = set(); revised_ever = set(); checkpoints = []; react_events = []; revision_curves = []; context_changes = []
    lesions = []; mechanisms = []; teach_index = 0
    teach_roles = [e["role"] for e in world["events"] if e["kind"] == "teach"]
    unpaired = rng_for("followup-unpaired", world["round"], world["index"]).permutation(teach_roles)
    previous_clock = backend.elapsed(a)
    for e in world["events"]:
        if stop_path and Path(stop_path).exists(): raise StopRequested("stop requested")
        kind = e["kind"]
        if kind == "teach":
            role = int(e["role"] if control != "unpaired" else unpaired[teach_index])
            backend.teach(a, codes[2*int(e["pair"]):2*int(e["pair"])+2], role, write=(control != "no_learning")); teach_index += 1
        elif kind == "set_relevance_context":
            new = set(map(int, e["indices"])); context_changes.append({"challenge": int(e["challenge"]), "previous": sorted(current_relevant),
                "current": sorted(new), "continued": sorted(current_relevant & new), "deactivated": sorted(current_relevant-new), "activated": sorted(new-current_relevant)})
            current_relevant = new
        elif kind == "rest": backend.rest(a, float(e["seconds"]))
        elif kind == "change_truth":
            ids = list(map(int, e["indices"])); roles[ids] = 1 - roles[ids]; revised_ever.update(ids)
        elif kind == "revision_curve":
            ids = list(map(int, e["indices"])); rr = np.asarray(e["desired"], int)
            cc = np.asarray([codes[2*i:2*i+2] for i in ids]).reshape(2*len(ids), -1); v = backend.probe(a, cc, rr)
            revision_curves.append({"challenge": int(e["challenge"]), "exposures": int(e["exposures"]), "indices": ids,
                                    "correct_n": int(np.sum(v["correct"])), "n": len(ids), "accuracy": float(np.mean(v["correct"]))})
        elif kind == "reactivation_probe":
            ids = list(map(int, e["indices"])); r = probe_subset(backend, a, codes, roles, ids)
            react_events.append({"challenge": int(e["challenge"]), "exposures": int(e["exposures"]), "indices": ids,
                                 "correct_n": r["correct_n"], "n": r["n"], "accuracy": r["accuracy"]})
        elif kind == "probe":
            n = int(e["n"]); v = backend.probe(a, codes[:2*n], roles[:n])
            cp = {"stage": e["stage"], "phase": e["phase"], "probe_type": e["probe_type"], "n": n,
                  "elapsed": backend.elapsed(a), "truth": clean(roles[:n]), "correct": clean(v["correct"]),
                  "actual": clean(v["actual"]), "fixed": clean(v["fixed"]), "correct_actual": clean(v["correct_actual"]),
                  "new_indices": list(e["new"]), "revision_indices": list(e["revision"]), "anomaly_indices": list(e["anomaly"]),
                  "stable_indices": list(e["stable"]), "current_relevant_indices": sorted(i for i in current_relevant if i<n),
                  "revised_ever_indices": sorted(i for i in revised_ever if i<n), "clip_fraction": float(v["clip_fraction"]),
                  "teach_index": teach_index}
            checkpoints.append(cp)
            if diagnostics and e["stage"] == "challenge2_after_block4_immediate":
                lesions.append(lesion_panel(backend, a, codes, roles, current_relevant, "challenge2_immediate", 0.0))
                lesions.append(lesion_panel(backend, a, codes, roles, current_relevant, "challenge2_plus_one_fast_tau", backend.fast_tau(a)))
            if diagnostics and e["stage"] == "block5_immediate":
                lesions.append(lesion_panel(backend, a, codes, roles, current_relevant, "block5_immediate", 0.0))
            if diagnostics and e["stage"] == "final_7day":
                lesions.append(lesion_panel(backend, a, codes, roles, current_relevant, "final_7day", 0.0))
            if diagnostics and round_name == "B" and e["stage"] in ("block2_day", "block7_day"):
                pair = int(world["mechanism"]["at96_current" if e["stage"] == "block2_day" else "at168_current"])
                mechanisms.append(mechanism_panel(backend, a, codes, roles, pair, int(world["mechanism"]["untrained_pair"]), e["stage"]))
        else: raise IntegrityError(f"unknown event kind {kind}")
        now = backend.elapsed(a)
        if now < previous_clock: raise IntegrityError("life clock went backwards")
        previous_clock = now
    end_inv = inventory(backend.stat_object(a))
    if start_inv != end_inv: raise IntegrityError("numeric learner state shape/bytes changed during life")
    metrics, secondary = compute_metrics(checkpoints, react_events, backend, a, codes, roles, world)
    meta = backend.metadata(a)
    return clean({"version": VERSION, "round": round_name, "arm_name": arm["name"], "arm_hash": arm["arm_hash"], "candidate_id": arm["candidate_id"],
                  "control": control, "world_index": int(world["index"]), "world_hash": digest(world), "scientific_backend": backend.scientific,
                  "metrics": metrics, "secondary": secondary, "checkpoints": checkpoints, "reactivation_events": react_events,
                  "revision_curves": revision_curves, "context_changes": context_changes, "lesion_diagnostics": lesions,
                  "mechanism_diagnostics": mechanisms, "metadata": meta, "inventory": end_inv, "environment": clean(backend.env),
                  "scope": "native MiniFly" if backend.scientific else "ENGINEERING FIXTURE ONLY"})


def count_result(correct, ids):
    ids = list(map(int, ids))
    if not ids: return (0,0)
    c = np.asarray(correct, bool); return int(c[ids].sum()), len(ids)


def compute_metrics(checkpoints, react_events, backend, a, codes, roles, world):
    counts = {k: [0,0] for k in PRIMARY}
    write_rows = []; conditional_rows = []; collateral_rows = []; time_rows = []
    previous_day = None; pending_immediate = None
    # P1/P4/P5/P6 and write persistence. Only canonical checkpoints enter P1;
    # reactivation micro-probes are reserved for P7/curve and cannot overweight P1.
    for cp in checkpoints:
        if cp["phase"] == "immediate":
            relevant = set(map(int, cp["current_relevant_indices"]))
            if relevant:
                x,n = count_result(cp["correct"], sorted(relevant)); counts["P1_current_usefulness"][0]+=x; counts["P1_current_usefulness"][1]+=n
            if cp["revision_indices"]:
                x,n = count_result(cp["correct"], cp["revision_indices"]); counts["P4_revision"][0]+=x; counts["P4_revision"][1]+=n
            if cp["new_indices"]:
                x,n = count_result(cp["correct"], cp["new_indices"]); counts["P5_acquisition"][0]+=x; counts["P5_acquisition"][1]+=n
            if cp["anomaly_indices"]:
                x,n = count_result(cp["correct"], cp["anomaly_indices"]); counts["P6_anomaly_recovery"][0]+=x; counts["P6_anomaly_recovery"][1]+=n
            if previous_day is not None and int(cp["teach_index"]) > int(previous_day["teach_index"]):
                ncommon = min(int(previous_day["n"]), int(cp["n"])); ta=np.asarray(previous_day["truth"],int)[:ncommon]; tb=np.asarray(cp["truth"],int)[:ncommon]
                useful = set(map(int, previous_day["current_relevant_indices"])) & set(map(int, cp["current_relevant_indices"]))
                useful = {i for i in useful if i<ncommon and ta[i]==tb[i]}
                if useful:
                    x,n = count_result(cp["correct"], sorted(useful)); counts["P2_relevant_after_write"][0]+=x; counts["P2_relevant_after_write"][1]+=n
                    before=np.asarray(previous_day["correct"],bool); after=np.asarray(cp["correct"],bool); ids=np.asarray(sorted(useful),int)
                    prev_correct=int(before[ids].sum()); survived=int((before[ids] & after[ids]).sum())
                    conditional_rows.append({"from": previous_day["stage"], "to": cp["stage"], "kind": "useful", "eligible_n":len(ids),
                                             "previously_correct_n":prev_correct, "survived_n":survived,
                                             "conditional_survival": None if prev_correct==0 else survived/prev_correct})
                revised = useful & set(map(int, previous_day["revised_ever_indices"]))
                if revised:
                    x,n = count_result(cp["correct"], sorted(revised)); counts["P3_revised_relevant_after_write"][0]+=x; counts["P3_revised_relevant_after_write"][1]+=n
                    before=np.asarray(previous_day["correct"],bool); after=np.asarray(cp["correct"],bool); ids=np.asarray(sorted(revised),int)
                    prev_correct=int(before[ids].sum()); survived=int((before[ids] & after[ids]).sum())
                    conditional_rows.append({"from":previous_day["stage"],"to":cp["stage"],"kind":"revised_useful","eligible_n":len(ids),
                                             "previously_correct_n":prev_correct,"survived_n":survived,
                                             "conditional_survival":None if prev_correct==0 else survived/prev_correct})
                # Collateral diagnostic: prior-trained, truth-unchanged, not currently demanded.
                coll = [i for i in range(ncommon) if i not in set(cp["current_relevant_indices"]) and ta[i]==tb[i]]
                if coll:
                    before=np.asarray(previous_day["correct"],bool); after=np.asarray(cp["correct"],bool); ids=np.asarray(coll,int)
                    pc=int(before[ids].sum()); sv=int((before[ids]&after[ids]).sum())
                    collateral_rows.append({"from":previous_day["stage"],"to":cp["stage"],"eligible_n":len(ids),"post_correct_n":int(after[ids].sum()),
                                            "previously_correct_n":pc,"survived_n":sv,"conditional_survival":None if pc==0 else sv/pc})
                write_rows.append({"from":previous_day["stage"],"to":cp["stage"],"teach_events":int(cp["teach_index"])-int(previous_day["teach_index"]),
                                   "continued_relevant_n":len(useful),"continued_revised_relevant_n":len(revised)})
            pending_immediate = cp
        elif cp["phase"] == "day":
            if pending_immediate is None or pending_immediate["probe_type"] != cp["probe_type"]: raise IntegrityError("immediate/day checkpoint pairing broke")
            ids=set(map(int,pending_immediate["current_relevant_indices"])) & set(map(int,cp["current_relevant_indices"]))
            if ids:
                x,n=count_result(cp["correct"],sorted(ids)); time_rows.append({"from":pending_immediate["stage"],"to":cp["stage"],"duration":"24h","correct_n":x,"n":n,"accuracy":x/n})
            previous_day=cp; pending_immediate=None
        elif cp["phase"] == "final":
            if previous_day is None: raise IntegrityError("final checkpoint lacks previous day")
            ids=set(map(int,previous_day["current_relevant_indices"])) & set(map(int,cp["current_relevant_indices"]))
            if ids:
                x,n=count_result(cp["correct"],sorted(ids)); time_rows.append({"from":previous_day["stage"],"to":cp["stage"],"duration":"7day","correct_n":x,"n":n,"accuracy":x/n})
        else: raise IntegrityError("unknown checkpoint phase")
    # P7: exactly six cue decisions after four exposures per history.
    for r in react_events:
        if int(r["exposures"]) == 4:
            counts["P7_reactivation_4"][0] += int(r["correct_n"]); counts["P7_reactivation_4"][1] += int(r["n"])
    for k,(x,n) in counts.items():
        if n <= 0: raise IntegrityError(f"primary metric {k} has zero denominator")
    metrics = {k:{"correct_n":int(x),"n":int(n),"ratio":float(x/n)} for k,(x,n) in counts.items()}
    # Reactivation curves pooled within history.
    react_curve={}
    for ex in (0,1,2,4):
        rows=[r for r in react_events if int(r["exposures"])==ex]; x=sum(int(r["correct_n"]) for r in rows); n=sum(int(r["n"]) for r in rows)
        react_curve[str(ex)]={"correct_n":x,"n":n,"ratio":None if n==0 else float(x/n)}
    # Obsolete outcome diagnostic for all revised cues at final state.
    rev_ids=sorted({m["memory_index"] for m in world["challenge_map"] if m["kind"]=="REV"})
    obsolete=None
    if rev_ids:
        cc=np.asarray([codes[2*i:2*i+2] for i in rev_ids]).reshape(2*len(rev_ids),-1); old_roles=1-np.asarray([roles[i] for i in rev_ids],int)
        v=backend.probe(a,cc,old_roles); obsolete={"n":len(rev_ids),"old_outcome_preferred_n":int(np.sum(v["correct"])),"rate":float(np.mean(v["correct"]))}
    total=int(world["total"]); unseen=backend.probe(a,codes[2*total:],roles[total:])
    secondary={"write_intervals":write_rows,"conditional_survival":conditional_rows,"collateral":collateral_rows,"time_only":time_rows,
               "reactivation_curve":react_curve,"obsolete_outcome":obsolete,
               "untrained":{"n":len(unseen["correct"]),"accuracy":float(np.mean(unseen["correct"])),"actual_response_rate":float(np.mean(np.abs(unseen["actual"])>=1.0))}}
    return metrics, secondary


def worker_init(): signal.signal(signal.SIGINT, signal.SIG_IGN)

def worker_job(payload):
    backend=get_backend(payload["backend_spec"]); world=load_world(payload["world_file"])
    return evaluate_life(backend,payload["arm"],world,control=payload["control"],round_name=payload["round"],stop_path=payload.get("stop_path"),diagnostics=payload.get("diagnostics",True))


def job_key(round_name, arm, world_hash, control, source_sig):
    return digest({"version":VERSION,"round":round_name,"arm_hash":arm["arm_hash"],"world_hash":world_hash,"control":control,"source_signature":source_sig})


def receipt_path(round_dir, key): return Path(round_dir)/"receipts"/f"{key}.json.gz"


def read_cached(path, expected_key):
    if not Path(path).exists(): return None
    body=sealed_read(path)
    if body.get("job_key")!=expected_key: raise IntegrityError(f"cached receipt key mismatch: {path}")
    return body["result"]


def run_jobs(round_name, round_dir, arms, backend_spec, source_sig, workers, controls=False, diagnostics=True):
    round_dir=Path(round_dir); manifest=json.loads((round_dir/"WORLD_MANIFEST.json").read_text()); stop=round_dir/"STOP_REQUESTED"
    if stop.exists(): stop.unlink()
    jobs=[]; results={}
    for w in manifest["worlds"]:
        wp=world_path(round_dir,int(w["index"])); wh=w["world_hash"]
        for arm in arms:
            key=job_key(round_name,arm,wh,"normal",source_sig); rp=receipt_path(round_dir,key); cached=read_cached(rp,key)
            token=(arm["name"],int(w["index"]),"normal")
            if cached is not None: results[token]=cached
            else: jobs.append((token,key,rp,{"round":round_name,"arm":arm,"world_file":str(wp),"control":"normal","backend_spec":backend_spec,"stop_path":str(stop),"diagnostics":diagnostics}))
        if controls and int(w["index"])<24:
            full=next(a for a in arms if a["name"]=="RCE_FULL")
            for control in ("no_learning","unpaired"):
                key=job_key(round_name,full,wh,control,source_sig); rp=receipt_path(round_dir,key); cached=read_cached(rp,key)
                token=(full["name"],int(w["index"]),control)
                if cached is not None: results[token]=cached
                else: jobs.append((token,key,rp,{"round":round_name,"arm":full,"world_file":str(wp),"control":control,"backend_spec":backend_spec,"stop_path":str(stop),"diagnostics":False}))
    if not jobs: return results
    interrupted=False
    try:
        with ProcessPoolExecutor(max_workers=max(1,int(workers)),mp_context=mp.get_context("spawn"),initializer=worker_init) as pool:
            active={pool.submit(worker_job,payload):(token,key,rp) for token,key,rp,payload in jobs}
            done_count=0
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for fut in done:
                    token,key,rp=active.pop(fut); res=fut.result(); sealed_write(rp,{"job_key":key,"result":res}); results[token]=res; done_count+=1
                    if done_count%10==0 or not active: print(f"{round_name}: completed {len(results)} result records ({len(active)} active/pending)",flush=True)
    except KeyboardInterrupt:
        interrupted=True; atomic_bytes(stop,b"stop\n"); print("Stop requested; rerun same command to resume sealed receipts.",file=sys.stderr)
    if interrupted: raise StopRequested("interrupted")
    return results


def result_rows(results, control="normal"):
    return [r for (arm,idx,c),r in sorted(results.items(),key=lambda kv:(kv[0][1],kv[0][0],kv[0][2])) if c==control]


def paired_bootstrap(a, b, draws, seed, one_sided=False):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<2: raise ValueError("paired bootstrap needs matched 1D histories")
    d=a-b; rng=np.random.default_rng(seed); vals=np.empty(draws,float); n=len(d)
    for start in range(0,draws,5000):
        k=min(5000,draws-start); ix=rng.integers(0,n,size=(k,n)); vals[start:start+k]=d[ix].mean(1)
    if one_sided:
        return {"mean_difference":float(d.mean()),"lower_95_one_sided":float(np.quantile(vals,.05)),"history_n":n,
                "positive_history_fraction":float(np.mean(d>0)),"nonnegative_history_fraction":float(np.mean(d>=0))}
    return {"mean_difference":float(d.mean()),"lower_95":float(np.quantile(vals,.025)),"upper_95":float(np.quantile(vals,.975)),"history_n":n,
            "positive_history_fraction":float(np.mean(d>0)),"nonnegative_history_fraction":float(np.mean(d>=0))}


def aggregate_metric(rows, metric):
    ratios=np.asarray([r["metrics"][metric]["ratio"] for r in rows],float); x=sum(r["metrics"][metric]["correct_n"] for r in rows); n=sum(r["metrics"][metric]["n"] for r in rows)
    return {"history_mean":float(ratios.mean()),"history_sd":float(ratios.std(ddof=1)) if len(ratios)>1 else 0.0,"pooled_correct_n":int(x),"pooled_n":int(n),"pooled_ratio":float(x/n),"history_values":ratios}


def export_normal_results(round_dir, results):
    rows=[]
    for (arm,idx,control),result in sorted(results.items(), key=lambda kv:(kv[0][1],kv[0][0],kv[0][2])):
        if control == "normal": rows.append({"arm":arm,"history":idx,"control":control,"result":result})
    payload=b"".join(canonical(r)+b"\n" for r in rows)
    atomic_bytes(Path(round_dir)/"NORMAL_RESULTS.jsonl.gz", gzip.compress(payload, compresslevel=3, mtime=0))


def export_control_summary(round_dir, results):
    rows=[]
    for control in ("no_learning","unpaired"):
        rr=[r for (arm,idx,c),r in sorted(results.items(),key=lambda kv:kv[0][1]) if arm=="RCE_FULL" and c==control]
        if not rr: continue
        for m in ("P1_current_usefulness","P4_revision","P5_acquisition","P7_reactivation_4"):
            a=aggregate_metric(rr,m); rows.append({"control":control,"metric":m,"history_mean":a["history_mean"],"pooled_correct_n":a["pooled_correct_n"],"pooled_n":a["pooled_n"],"pooled_ratio":a["pooled_ratio"]})
    write_csv(Path(round_dir)/"CONTROL_SUMMARY.csv",rows)


def analyze_round_a(round_dir, results, source_sig, engineering=False):
    rows_by={name:[results[(name,i,"normal")] for i in sorted({k[1] for k in results if k[0]==name and k[2]=="normal"})] for name in ("REF","V84E","RCE_FULL","RCE_TRACE_RESTORED")}
    n=len(rows_by["RCE_FULL"]); expected=n if engineering else ROUND_A_HISTORIES
    if n!=expected or any(len(v)!=n for v in rows_by.values()): raise IntegrityError("Round-A paired result count mismatch")
    boot_seed=sha_seed("MiniFly-RCEV87-FOLLOWUP-A-bootstrap|20260916")
    table=[]; all_pass=True
    for j,m in enumerate(PRIMARY):
        full=aggregate_metric(rows_by["RCE_FULL"],m); trace=aggregate_metric(rows_by["RCE_TRACE_RESTORED"],m)
        b=paired_bootstrap(trace["history_values"],full["history_values"],BOOTSTRAP_DRAWS if not engineering else 2000,boot_seed+j,one_sided=True)
        passed=bool(b["lower_95_one_sided"]>=-NI_MARGIN); all_pass &= passed
        table.append({"metric":m,"full_mean":full["history_mean"],"trace_mean":trace["history_mean"],"paired_mean_difference":b["mean_difference"],
                      "lower_95_one_sided":b["lower_95_one_sided"],"ni_margin":NI_MARGIN,"pass":passed,"history_n":n,
                      "full_correct_n":full["pooled_correct_n"],"full_n":full["pooled_n"],"trace_correct_n":trace["pooled_correct_n"],"trace_n":trace["pooled_n"]})
    full_bytes=max(sum(v["bytes"] for v in r["inventory"].values()) for r in rows_by["RCE_FULL"])
    trace_bytes=max(sum(v["bytes"] for v in r["inventory"].values()) for r in rows_by["RCE_TRACE_RESTORED"])
    state_pass=trace_bytes<=full_bytes; all_pass &= state_pass
    label="TRACE_RESTORATION_CONFIRMED_NONINFERIOR_ON_RELEVANCE_SUITE" if all_pass else "TRACE_RESTORATION_NOT_CONFIRMED"
    decision={"version":VERSION,"decision":label,"base_for_round_b":"RCE_TRACE_RESTORED" if all_pass else "RCE_FULL","primary_all_pass":all_pass,
              "state_bytes_full":full_bytes,"state_bytes_trace":trace_bytes,"state_nonlarger":state_pass,"paired_noninferiority":table,
              "source_signature_hash":digest(source_sig),"engineering_only":bool(engineering)}
    decision["decision_hash"]=digest(decision)
    atomic_json(Path(round_dir)/"DECISION.json",decision)
    write_csv(Path(round_dir)/"PAIRED_NONINFERIORITY.csv",table)
    export_normal_results(round_dir, results)
    export_control_summary(round_dir, results)
    write_arm_summary(Path(round_dir)/"ARM_SUMMARY.csv",rows_by)
    export_diagnostics(round_dir,rows_by,round_name="A")
    write_round_a_report(round_dir,decision,rows_by)
    return decision


def write_csv(path, rows):
    rows=list(rows); Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not rows: atomic_bytes(path,b""); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with io.StringIO() as s:
        w=csv.DictWriter(s,fieldnames=fields); w.writeheader(); w.writerows([{k:clean(v) for k,v in r.items()} for r in rows]); atomic_bytes(path,s.getvalue().encode())


def write_arm_summary(path, rows_by):
    rows=[]
    for arm,rr in rows_by.items():
        for m in PRIMARY:
            a=aggregate_metric(rr,m); rows.append({"arm":arm,"metric":m,"history_mean":a["history_mean"],"history_sd":a["history_sd"],"pooled_correct_n":a["pooled_correct_n"],"pooled_n":a["pooled_n"],"pooled_ratio":a["pooled_ratio"]})
    write_csv(path,rows)


def export_diagnostics(round_dir, rows_by, round_name):
    react=[]; rel=[]; time_rows=[]; lesions=[]; mechanisms=[]
    for arm, rr in rows_by.items():
        for r in rr:
            hi=r["world_index"]
            for e in r["reactivation_events"]: react.append({"arm":arm,"history":hi,**e})
            for e in r["secondary"]["conditional_survival"]: rel.append({"arm":arm,"history":hi,**e})
            for e in r["secondary"]["time_only"]: time_rows.append({"arm":arm,"history":hi,**e})
            for e in r["lesion_diagnostics"]:
                for lesion in ("intact","fast","slow","both"):
                    q=e[lesion]; lesions.append({"arm":arm,"history":hi,"label":e["label"],"rest_seconds":e["rest_seconds"],"lesion":lesion,"n":q["n"],"correct_n":q["correct_n"],"accuracy":q["accuracy"]})
            for e in r.get("mechanism_diagnostics",[]): mechanisms.append({"arm":arm,"history":hi,**{k:v for k,v in e.items() if k!="events"}})
    write_csv(Path(round_dir)/"REACTIVATION_EVENTS.csv",react); write_csv(Path(round_dir)/"RELEVANCE_TRANSITIONS.csv",rel); write_csv(Path(round_dir)/"TIME_ONLY_DIAGNOSTICS.csv",time_rows); write_csv(Path(round_dir)/"LESION_DIAGNOSTICS.csv",lesions)
    if round_name=="B":
        write_csv(Path(round_dir)/"MECHANISM_EVENTS.csv",mechanisms)
        write_csv(Path(round_dir)/"WRITE_DIRECTION.csv",mechanisms)


def write_round_a_report(round_dir,decision,rows_by):
    lines=["# Round A report","",f"Status: **{decision['decision']}**","","> This is a relevance-centered parameter-restoration confirmation. It is not a minimal-core certificate.","","## Primary paired noninferiority"]
    lines += ["","| Metric | RCE full | Trace restored | Difference | One-sided lower 95% | Pass |","|---|---:|---:|---:|---:|:---:|"]
    for r in decision["paired_noninferiority"]:
        lines.append(f"| {r['metric']} | {r['full_mean']:.4f} | {r['trace_mean']:.4f} | {r['paired_mean_difference']:+.4f} | {r['lower_95_one_sided']:+.4f} | {'YES' if r['pass'] else 'NO'} |")
    lines += ["",f"State bytes: full={decision['state_bytes_full']}, trace={decision['state_bytes_trace']}; non-larger={'YES' if decision['state_nonlarger'] else 'NO'}.","","Collateral/unrelated retention, time-only survival, clipping, lesions, and untrained-cue response are diagnostics only and do not enter this decision."]
    atomic_bytes(Path(round_dir)/"REPORT.md",("\n".join(lines)+"\n").encode())


def analyze_round_b(round_dir,results,source_sig,engineering=False):
    names=["Fnative_Anative","Fnative_A3","Fnative_A2p5","Fref_Anative","Fref_A3","Fref_A2p5","REF","V84E"]
    rows_by={name:[results[(name,i,"normal")] for i in sorted({k[1] for k in results if k[0]==name and k[2]=="normal"})] for name in names}
    n=len(rows_by[names[0]]); expected=n if engineering else ROUND_B_HISTORIES
    if n!=expected or any(len(v)!=n for v in rows_by.values()): raise IntegrityError("Round-B paired result count mismatch")
    cells=[]
    for name in names:
        for m in PRIMARY:
            a=aggregate_metric(rows_by[name],m); cells.append({"arm":name,"metric":m,"history_mean":a["history_mean"],"history_sd":a["history_sd"],"pooled_correct_n":a["pooled_correct_n"],"pooled_n":a["pooled_n"],"pooled_ratio":a["pooled_ratio"]})
    write_csv(Path(round_dir)/"FACTORIAL_CELL_SUMMARY.csv",cells)
    export_normal_results(round_dir, results)
    boot_seed=sha_seed("MiniFly-RCEV87-FOLLOWUP-B-bootstrap|20260916"); contrasts=[]; ci=0
    def add(label,a_name,b_name,m):
        nonlocal ci
        aa=aggregate_metric(rows_by[a_name],m); bb=aggregate_metric(rows_by[b_name],m); b=paired_bootstrap(aa["history_values"],bb["history_values"],BOOTSTRAP_DRAWS if not engineering else 2000,boot_seed+ci,False); ci+=1
        contrasts.append({"contrast":label,"metric":m,"arm_a":a_name,"arm_b":b_name,**b,"a_correct_n":aa["pooled_correct_n"],"a_n":aa["pooled_n"],"b_correct_n":bb["pooled_correct_n"],"b_n":bb["pooled_n"]})
    for act in ("Anative","A3","A2p5"):
        for m in PRIMARY: add(f"feedback@{act}",f"Fref_{act}",f"Fnative_{act}",m)
    for fb in ("Fnative","Fref"):
        for act in ("A3","A2p5"):
            for m in PRIMARY: add(f"activity_{act}@{fb}",f"{fb}_{act}",f"{fb}_Anative",m)
    # Difference-in-differences paired at history level.
    for act in ("A3","A2p5"):
        for m in PRIMARY:
            a=np.asarray([r["metrics"][m]["ratio"] for r in rows_by[f"Fref_{act}"]]); b=np.asarray([r["metrics"][m]["ratio"] for r in rows_by[f"Fnative_{act}"]]); c=np.asarray([r["metrics"][m]["ratio"] for r in rows_by["Fref_Anative"]]); d=np.asarray([r["metrics"][m]["ratio"] for r in rows_by["Fnative_Anative"]])
            diff=(a-b)-(c-d); zero=np.zeros_like(diff); bs=paired_bootstrap(diff,zero,BOOTSTRAP_DRAWS if not engineering else 2000,boot_seed+ci,False); ci+=1
            contrasts.append({"contrast":f"interaction_{act}","metric":m,"arm_a":f"[(Fref,{act})-(Fnative,{act})]","arm_b":"[(Fref,Anative)-(Fnative,Anative)]",**bs})
    write_csv(Path(round_dir)/"FACTORIAL_PAIRED_CONTRASTS.csv",contrasts); write_arm_summary(Path(round_dir)/"ARM_SUMMARY.csv",rows_by); export_diagnostics(round_dir,rows_by,"B")
    report=["# Round B report","","> Causal feedback × activity screen. No automatic winner or persistent-core label is produced.","",f"Histories: {n}","","See `FACTORIAL_CELL_SUMMARY.csv` and `FACTORIAL_PAIRED_CONTRASTS.csv` for P1–P7 paired effects and intervals.","","Mechanism and lesion diagnostics are explanatory only."]
    atomic_bytes(Path(round_dir)/"REPORT.md",("\n".join(report)+"\n").encode())
    decision={"version":VERSION,"round":"B","status":"SCREEN_COMPLETE_NO_AUTOMATIC_WINNER","history_n":n,"source_signature_hash":digest(source_sig),"engineering_only":bool(engineering)}; decision["result_hash"]=digest(decision); atomic_json(Path(round_dir)/"RESULT.json",decision)
    return decision


def validate_arm_native_contract(backend,arms,round_name):
    meta=[]
    for arm in arms:
        backend.model.genome_array(np.asarray(arm["genes"],float)); a=backend.make(arm); m=backend.metadata(a); meta.append((arm,m))
        if abs(m["phenotype"]["effective_active_fraction"] - (arm["intervention"].get("active_fraction_override") if arm["intervention"].get("active_fraction_override") is not None else 0.05*2**arm["genes"][14]))>1e-12:
            raise IntegrityError(f"{arm['name']}: effective active fraction mismatch")
    if round_name=="A":
        x={a["name"]:m for a,m in meta}
        if x["RCE_FULL"]["B_sha256"]!=x["RCE_TRACE_RESTORED"]["B_sha256"]: raise IntegrityError("Round-A RCE arms have different B")
    else:
        hashes={m["B_sha256"] for a,m in meta[:6]}
        if len(hashes)!=1: raise IntegrityError("Round-B factorial B hashes differ")
        # Feedback factor must act exactly through native phenotype scale.
        for a,m in meta[:6]:
            expected=2**a["genes"][8]
            if abs(m["phenotype"]["feedback_scale"]-expected)>1e-12: raise IntegrityError("feedback scale mismatch")
    q={m["reader_Q_sha256"] for a,m in meta}
    if len(q)!=1: raise IntegrityError("REF-Q observer hash differs across arms")
    return [{"arm":a["name"],"B_sha256":m["B_sha256"],"reader_Q_sha256":m["reader_Q_sha256"],"phenotype":m["phenotype"]} for a,m in meta]


def preflight(round_name,round_dir,arms,backend_spec,source_sig):
    out=Path(round_dir)/"PREFLIGHT.json"
    key=digest({"version":VERSION,"round":round_name,"source":source_sig,"arms":arms})
    if out.exists():
        old=json.loads(out.read_text())
        if old.get("preflight_key")==key and old.get("passed") is True: return old
    backend=get_backend(backend_spec); checks=[]
    # Source/candidate construction invariants.
    meta=validate_arm_native_contract(backend,arms,round_name) if backend.scientific else []
    checks.append("arm_construction")
    # One complete life per arm on prespecified first world. Diagnostic branches included.
    w=load_world(world_path(round_dir,0))
    for arm in arms:
        a=backend.make(arm); before=array_digest(backend.stat_object(a)); codes=backend.encode(a,w["pn"]); _=probe_subset(backend,a,codes,w["roles"],range(4)); after=array_digest(backend.stat_object(a))
        if before!=after: raise IntegrityError("read-only probe modified main state")
        r=evaluate_life(backend,arm,w,round_name=round_name,diagnostics=True)
        for m in PRIMARY:
            if r["metrics"][m]["n"]<=0: raise IntegrityError(f"preflight zero denominator {arm['name']} {m}")
    checks.append("complete_life_all_arms")
    body={"passed":True,"preflight_key":key,"checks":checks,"native_contract":meta,"world0_hash":digest(w)}; atomic_json(out,body); return body


def write_manifests(round_name,round_dir,arms,backend_spec,source_sig,round_a_decision=None):
    atomic_json(Path(round_dir)/"SOURCE_HASHES.json",source_sig); atomic_json(Path(round_dir)/"ARM_MANIFEST.json",{"round":round_name,"arms":arms})
    run={"version":VERSION,"round":round_name,"backend_spec":backend_spec,"life":LIFE,"source_signature_hash":digest(source_sig),"design_sha256":source_sig["design"],"runner_sha256":source_sig["runner"],"prior_keys_sha256":source_sig["prior_keys"]}
    if round_a_decision is not None: run["round_a_decision_hash"]=round_a_decision["decision_hash"]; run["round_a_base_for_b"]=round_a_decision["base_for_round_b"]
    run["run_hash"]=digest(run); atomic_json(Path(round_dir)/"RUN_MANIFEST.json",run); return run


def validate_round_a_decision(decision):
    if not isinstance(decision, dict) or "decision_hash" not in decision:
        raise IntegrityError("Round-A decision lacks integrity hash")
    expected = decision["decision_hash"]
    body = {k:v for k,v in decision.items() if k != "decision_hash"}
    if digest(body) != expected:
        raise IntegrityError("Round-A decision hash mismatch")
    if decision.get("base_for_round_b") not in ("RCE_FULL", "RCE_TRACE_RESTORED"):
        raise IntegrityError("Round-A decision has invalid Round-B base")
    return decision


def load_round_a_keys(base_out):
    p=Path(base_out)/"round_A"/"WORLD_MANIFEST.json"
    if not p.is_file(): raise FileNotFoundError("Round-A world manifest missing")
    keys=set()
    d=json.loads(p.read_text())
    for w in d["worlds"]:
        ww=load_world(world_path(Path(base_out)/"round_A",int(w["index"])))
        keys |= {tuple(map(int,x)) for x in ww["pattern_keys"]}
    return keys


def run_round(args):
    round_name=args.round.upper(); base_out=Path(args.out).expanduser().resolve(); rd=base_out/("round_A" if round_name=="A" else "round_B"); rd.mkdir(parents=True,exist_ok=True)
    engineering=bool(args.engineering)
    n_histories=int(args.histories) if args.histories else (ROUND_A_HISTORIES if round_name=="A" else ROUND_B_HISTORIES)
    if not engineering and n_histories != (ROUND_A_HISTORIES if round_name=="A" else ROUND_B_HISTORIES): raise ValueError("scientific mode uses frozen history count; use --engineering for reduced tests")
    prior,prior_meta=load_prior_keys(ROOT)
    decision_a=None
    if round_name=="B":
        dp=base_out/"round_A"/"DECISION.json"
        if not dp.is_file(): raise FileNotFoundError("Round B requires completed sealed Round-A DECISION.json")
        decision_a=validate_round_a_decision(json.loads(dp.read_text()))
        if decision_a.get("engineering_only") and not engineering: raise IntegrityError("scientific Round B cannot use engineering-only Round-A decision")
        prior |= load_round_a_keys(base_out)
        arms=round_b_arms(decision_a["base_for_round_b"])
    else: arms=round_a_arms()
    backend_spec={"backend":args.backend,"project_root":str(Path(args.project_root).expanduser().resolve()) if args.project_root else None,"fixture_units":64}
    if args.backend=="native" and not args.project_root: raise ValueError("native mode requires --project-root")
    if args.backend=="fixture": backend_spec.pop("project_root",None)
    source_sig=source_signature(backend_spec); prepare_worlds(round_name,rd,n_histories,prior); write_manifests(round_name,rd,arms,backend_spec,source_sig,decision_a)
    if args.prepare_only:
        print(json.dumps({"prepared":True,"round":round_name,"output":str(rd),"histories":n_histories,"arms":[a["name"] for a in arms]},indent=2)); return
    preflight(round_name,rd,arms,backend_spec,source_sig)
    if args.preflight_only:
        print(json.dumps({"preflight":True,"round":round_name,"output":str(rd)},indent=2)); return
    results=run_jobs(round_name,rd,arms,backend_spec,source_sig,args.workers,controls=(round_name=="A"),diagnostics=True)
    if round_name=="A": decision=analyze_round_a(rd,results,source_sig,engineering)
    else: decision=analyze_round_b(rd,results,source_sig,engineering)
    atomic_json(rd/"STATUS.json",{"status":"COMPLETE","round":round_name,"version":VERSION,"result":decision})
    print(json.dumps({"status":"COMPLETE","round":round_name,"output":str(rd),"result":decision.get("decision",decision.get("status"))},indent=2))


def self_test():
    # Static candidate provenance.
    assert legacy_candidate_id(RCE_FULL,"V81")==EXPECTED_SOURCE_IDS["RCE_FULL"]
    assert legacy_candidate_id(RCE_TRACE,"V81")==EXPECTED_SOURCE_IDS["RCE_TRACE_RESTORED"]
    diff=np.flatnonzero(np.abs(np.asarray(RCE_FULL)-np.asarray(RCE_TRACE))>1e-12).tolist(); assert diff==[4,11,12,13]
    # Bound attacks fail.
    for i,name in enumerate(GENES):
        for val in (GENE_LO[i]-1e-4,GENE_HI[i]+1e-4):
            g=np.asarray(REF,float); g[i]=val
            try: validate_genes(g)
            except ValueError: pass
            else: raise AssertionError(f"bound attack accepted for {name}")
    # Round-B interventions do not mutate stored sparsity gene, and only feedback gene changes.
    arms=round_b_arms("RCE_TRACE_RESTORED"); native=next(a for a in arms if a["name"]=="Fnative_Anative")
    for a in arms[:6]:
        assert abs(a["genes"][14]-native["genes"][14])<1e-15
        assert all(abs(a["genes"][i]-native["genes"][i])<1e-15 for i in range(17) if i!=8)
    # Synthetic semantic gate: better primaries with worse collateral would pass because collateral is absent.
    assert "collateral" not in PRIMARY and "churn" not in " ".join(PRIMARY)
    # Create one world and verify exactly six reactivation episodes and no raw duplicates.
    w=create_world("A",0,set()); audit_world(w)
    assert sum(len(e["indices"]) for e in w["events"] if e["kind"]=="reactivation_probe" and e["exposures"]==4)==6
    # P1 must exclude reactivation microprobes by construction: compute_metrics only consumes checkpoints for P1.
    print(json.dumps({"self_test":"PASS","version":VERSION,"trace_diff_indices":diff,"world_hash":digest(w)},indent=2))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round",choices=["A","B","a","b"],help="run Round A or Round B")
    p.add_argument("--project-root",help="existing MiniFly project root containing V82E and inherited iterations")
    p.add_argument("--out",default=str(ROOT/"runs"/"RCE_V87_followup"),help="separate follow-up output root")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--backend",choices=["native","fixture"],default="native")
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--engineering",action="store_true",help="allow reduced fixture history counts; not scientific evidence")
    p.add_argument("--histories",type=int,help="history count override only with --engineering")
    p.add_argument("--self-test",action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv)
    if args.self_test: self_test(); return
    if not args.round: raise SystemExit("--round A or --round B is required (or use --self-test)")
    if args.histories and not args.engineering: raise SystemExit("--histories requires --engineering")
    if args.backend=="native" and args.engineering: raise SystemExit("--engineering is for fixture/software validation, not native scientific runs")
    run_round(args)

if __name__=="__main__": main()
