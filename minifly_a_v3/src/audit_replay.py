"""Independent replay reconstruction of mechanism equations (W branch, one store) from the SPEC alone.

Does NOT import stores.py, systems.py or runner.py. It drives the frozen native Full151 model directly
(birth via portable_birth, events via the inherited EvoLearner), re-implements each declared equation
(Z load/conflict gate + split patch, Z2_rand yoked permutation, P weight update + budget + concentration
statistics, R novelty, R shared-bank pre-answer values) and compares every per-event number with the receipt.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402,F401
import portable_birth as pb  # noqa: E402
import content_model  # noqa: E402
from fixture import DT, RECORD_SECONDS, make_world  # noqa: E402

CH = b"0123"
# SPEC_DRAFT constants, restated here independently of stores.py
TAU_L, KAPPA = 86400.0, 0.2
EPS, TAU_P, FLOOR = 0.05, 330.0, 1e-9
WINDOW = 24


class ReplayMismatch(AssertionError):
    pass


def close(a, b, rel=1e-9, what=""):
    if not (abs(a - b) <= rel * max(1.0, abs(a), abs(b))):
        raise ReplayMismatch(f"{what}: replay {a!r} vs receipt {b!r}")


def permitted_W(stage, domain, store):
    return domain == "fact" or store < 2      # W branch: every value write permitted on active stores


def schedule(world):
    """Yield (index, rec, begin) and the checkpoint flush times of the frozen platform life."""
    doc = make_world(world)
    for i, rec in enumerate(doc["records"]):
        yield i, rec, i * RECORD_SECONDS + (86400.0 if rec["stage"] == "new" else 0.0), doc


def drive(world, store_model, on_teach, *, feed_all=None):
    """Feed one continuing store exactly as the platform does for branch W; call on_teach at each answer."""
    models = feed_all or [store_model]
    doc = None
    for i, rec, begin, doc in schedule(world):
        cue = bytes.fromhex(rec["cue_hex"])
        for k, b in enumerate(cue):
            for m in models:
                m.byte(b, begin + k * DT, learn=False)
        t = begin + 12 * DT
        on_teach(i, rec, t)
        for m in models:
            m.byte(10, begin + 13 * DT, learn=False)
            m.flush(begin + RECORD_SECONDS)
        if i == 335:
            for m in models:
                m.flush(doc["new_start_s"])
        if i == 599:
            for m in models:
                m.flush(doc["final_s"])


def _finish_teach(m):
    m.brain_t = m.pending_t
    m.teach_seen += 1
    m.pending_x = None
    m.pending_t = None


def buckets_of(slow, u, side):
    return {f"{s}:{sign:+d}": np.flatnonzero((side == s) & (np.sign(u) == sign) & (slow != 0))
            for s in (0, 1) for sign in (-1, 1)}


def replay_z(receipt, z2_receipt=None, limit=None):
    arm, world = receipt["arm"], receipt["world"]
    rows = {r[1]: r for r in receipt["mechanism_events"]["W"] if r[0] == "Z" and r[2] == 0}
    led = {r[1]: r for r in receipt["mechanism_events"]["W"] if r[0] == "w" and r[3] == 0}
    ref = None
    if arm == "Z2_rand":
        ref = {r[1]: {k: v[2] for k, v in r[7].items()}
               for r in z2_receipt["mechanism_events"]["W"] if r[0] == "Z" and r[2] == 0}
    m, _ = pb.canonical_fresh_native()
    mask = np.abs(np.asarray(m.fly.m.T)[:, 2:4]).sum(axis=1) > 0
    side = np.asarray(m.fly.m.kc_side)
    state = {"L": np.zeros(len(mask)), "t": 0.0, "checked": 0}

    def on_teach(i, rec, t):
        if limit is not None and i >= limit:
            m.byte(rec["answer"], t, learn=False)
            m.fly.event(max(0.0, m.pending_t - m.brain_t), m.pending_x, float(CH[0] != rec["answer"]), False)
            _finish_teach(m)
            return
        m.byte(rec["answer"], t, learn=False)
        now = m.elapsed_base + m.pending_t
        state["L"] *= math.exp(-max(0.0, now - state["t"]) / TAU_L)
        state["t"] = now
        dt = max(0.0, m.pending_t - m.brain_t)
        write = permitted_W(rec["stage"], rec["domain"], 0)
        pre_fast, pre_slow = m.fly.m.fast[:, 1].copy(), m.fly.m.slow.copy()
        record = m.fly.event(dt, m.pending_x, float(CH[0] != rec["answer"]), write)
        alpha = float(np.abs(record["rawalpha"]).sum()) if record.get("active") else 0.0
        close(alpha, led[i][5], 1e-9, f"alpha {arm}/{i}")
        if record.get("active") and write:
            u = np.asarray(record["rawalpha"], float)
            old_slow, old_fast = np.asarray(record["rawslow"], float), np.asarray(record["rawfast"], float)
            L = state["L"]
            conflict = (u * pre_slow < 0) & mask
            g = np.where(mask, (1.0 - L) * (1.0 - conflict), 1.0)
            a = np.abs(old_slow)
            if arm == "Z0_resource":
                eff, target = np.ones_like(g), float(a.sum())
            elif arm == "Z2":
                eff, target = g, float(a @ g)
            else:
                eff = g.copy()
                key = f"A3-Z2RAND-v2|{world}|W|{i}|0"
                for name, ids in buckets_of(old_slow, u, side).items():
                    tgt = float(ref[i].get(name, 0.0))
                    if len(ids) == 0:
                        continue
                    aa = a[ids]
                    seed = int.from_bytes(hashlib.sha256(f"{key}|{name}".encode()).digest()[:8], "big")
                    pg = g[ids][np.random.default_rng(seed).permutation(len(ids))]
                    got = float(aa @ pg)
                    if got > tgt:
                        pg = pg * (tgt / got)
                    elif got < tgt:
                        pg = pg + (tgt - got) / float(aa @ (1.0 - pg)) * (1.0 - pg)
                    eff[ids] = np.clip(pg, 0.0, 1.0)
                target = float(sum(ref[i].values()))
            new_slow = eff * old_slow
            new_fast = old_fast + (old_slow - new_slow)
            df = math.exp(-dt / float(m.fly.m.kernel["fast_tau"]))
            ds = math.exp(-dt / float(m.fly.m.kernel["slow_tau"]))
            m.fly.m.fast[:, 1] += (new_fast - old_fast) * df
            m.fly.m.slow += (new_slow - old_slow) * ds
            moved = (u != 0) & mask
            L[moved] += KAPPA * eff[moved] * (1.0 - L[moved])
            z = rows[i]
            if z[3] != int(conflict.sum()):
                raise ReplayMismatch(f"conflicts {arm}/{i}: {int(conflict.sum())} vs {z[3]}")
            close(float(a.sum()), z[4], 1e-9, f"native slow {arm}/{i}")
            close(float(np.abs(new_slow).sum()), z[5], 1e-9, f"gated slow {arm}/{i}")
            close(target, z[6], 1e-9, f"target {arm}/{i}")
            state["checked"] += 1
        elif i in rows:
            raise ReplayMismatch(f"receipt has a Z event where replay has none {arm}/{i}")
        _finish_teach(m)

    drive(world, m, on_teach)
    return {"arm": arm, "store": 0, "branch": "W", "z_events_checked": state["checked"]}


def replay_p(receipt, limit=None, require_pd=True):
    arm, world = receipt["arm"], receipt["world"]
    prow = {r[1]: r for r in receipt["mechanism_events"]["W"] if r[0] == "P" and r[2] == 0}
    pd = {r[1]: r[3] for r in receipt["mechanism_events"]["W"] if r[0] == "PD"}
    m, _ = pb.canonical_fresh_native()
    B0 = m.fly.m.B.tocsr()
    rows_e = np.repeat(np.arange(B0.shape[0]), np.diff(B0.indptr))
    cols = B0.indices.copy()
    n_kc = B0.shape[1]
    deg = np.bincount(cols, minlength=n_kc)
    budget = np.bincount(cols, weights=B0.data, minlength=n_kc)
    mean = np.divide(budget, deg, out=np.zeros(n_kc), where=deg > 0)
    st = {"pre": np.zeros(B0.shape[0]), "post": np.zeros(n_kc), "t": None, "act": np.zeros(n_kc, np.int64),
          "n": 0, "checked": 0}

    def on_teach(i, rec, t):
        fe = m.fe.clone()
        fe.advance(t)
        a = fe.read()[m.fly.m.pn_type_index].astype(float)
        m.byte(rec["answer"], t, learn=False)
        x = np.asarray(m.pending_x, float)
        when = m.elapsed_base + m.pending_t
        dt = max(0.0, m.pending_t - m.brain_t)
        m.fly.event(dt, m.pending_x, float(CH[0] != rec["answer"]), permitted_W(rec["stage"], rec["domain"], 0))
        _finish_teach(m)
        B = m.fly.m.B
        decay = 0.0 if st["t"] is None else math.exp(-max(0.0, when - st["t"]) / TAU_P)
        st["pre"] *= decay
        st["post"] *= decay
        mk = mean[cols]
        v = B.data / mk
        xa = x[cols]
        if arm in ("P0", "P1"):
            dv = EPS * a[rows_e] * xa
        elif arm == "P2":
            dv = EPS * xa * (a[rows_e] - xa * v)
        else:
            dv = EPS * (st["pre"][rows_e] * xa - a[rows_e] * st["post"][cols])
        st["pre"] += a
        st["post"] += x
        st["t"] = when
        st["act"] += (x > 0)
        st["n"] += 1
        r = prow[i]
        close(float(np.abs(dv).sum()), r[3], 1e-8, f"P pre-norm delta {arm}/{i}")
        if arm == "P0":
            w = B.data
            close(0.0, r[4], 1e-12, f"P0 change {arm}/{i}")
        else:
            w = np.maximum(v + dv, FLOOR) * mk
            sums = np.bincount(cols, weights=w, minlength=n_kc)
            w = w * np.divide(budget, sums, out=np.ones_like(sums), where=sums > 0)[cols]
            close(float(np.abs(w - B.data).sum()), r[4], 1e-8, f"P change {arm}/{i}")
            m.fly.m.B = csr_matrix((w, B.indices, B.indptr), shape=B.shape)
        close(float(w.max()), r[5], 1e-9, f"P wmax {arm}/{i}")
        close(float(w.min()), r[6], 1e-6, f"P wmin {arm}/{i}")
        if st["n"] in (336, 600) and (require_pd or i in pd):
            d = pd[i]
            vv = w / mean[cols]
            if d["edges_rel_below_1e-3"] != int((vv < 1e-3).sum()) or d["edges_rel_below_1e-6"] != int((vv < 1e-6).sum()):
                raise ReplayMismatch(f"P concentration counts {arm}/{i}")
            s1 = np.bincount(cols, weights=w, minlength=n_kc)
            s2 = np.bincount(cols, weights=w * w, minlength=n_kc)
            multi = deg >= 2
            pr = s1[multi] ** 2 / s2[multi]
            for got, exp in zip(np.quantile(pr, [0.05, 0.5, 0.95]), d["kc_fanin_participation_q05_q50_q95"]):
                close(float(got), exp, 1e-9, f"P participation {arm}/{i}")
            if d["kc_active_every_update"] != int((st["act"] == st["n"]).sum()):
                raise ReplayMismatch(f"P activation count {arm}/{i}")
        st["checked"] += 1

    drive(world, m, on_teach)
    return {"arm": arm, "store": 0, "branch": "W", "p_updates_checked": st["checked"],
            "pd_checkpoints_checked": len(pd)}


def replay_r_novelty(receipt):
    """CONTENT pre-answer codes depend only on bytes and the fixed B: recompute Jaccard novelty for all records."""
    world = receipt["world"]
    rrows = {r[1]: r for r in receipt["mechanism_events"]["W"] if r[0] == "R"}
    base, _ = pb.canonical_fresh_native()
    m = content_model.from_native(base)
    buf = []
    n = {"checked": 0}

    def on_teach(i, rec, t):
        m.byte(rec["answer"], t, learn=False)
        x = np.asarray(m.pending_x) > 0
        if not buf:
            nov = 1.0
        else:
            best = 0.0
            for p in buf:
                union = np.count_nonzero(x | p)
                best = max(best, np.count_nonzero(x & p) / union if union else 1.0)
            nov = 1.0 - best
        close(nov, rrows[i][2], 1e-12, f"novelty {receipt['arm']}/{i}")
        buf.append(x)
        if len(buf) > WINDOW:
            buf.pop(0)
        m.fly.event(max(0.0, m.pending_t - m.brain_t), m.pending_x, 0.0, False)
        _finish_teach(m)
        n["checked"] += 1

    drive(world, m, on_teach)
    return {"arm": receipt["arm"], "novelty_checked": n["checked"]}


def replay_r_shared(receipt, scale_shared, limit=600):
    """Four native shared stores with native binary writes: recompute the cached pre-answer shared values and
    the signed coefficients of R0_signed / R3 (R3_randtarget's derangement is re-derived in audit_a)."""
    arm, world = receipt["arm"], receipt["world"]
    rrows = {r[1]: r for r in receipt["mechanism_events"]["W"] if r[0] == "R"}
    ms = [pb.canonical_fresh_native()[0] for _ in range(4)]
    n = {"checked": 0}

    def on_teach(i, rec, t):
        if i < limit:
            vals = [float(mm.association_value(t)) for mm in ms]
            for j in range(4):
                close(vals[j], rrows[i][5][j], 1e-9, f"shared value {arm}/{i}/{j}")
            k = 4 if rec["domain"] == "fact" else 2
            v = [vals[j] / scale_shared for j in range(k)]
            mx = max(v)
            e = [math.exp(z - mx) for z in v]
            p = [z / sum(e) for z in e]
            y = [1.0 if CH[j] == rec["answer"] else 0.0 for j in range(k)]
            if arm == "R3":
                for j in range(k):
                    close(p[j] - y[j], rrows[i][6][1 + j], 1e-9, f"R3 residual {i}/{j}")
            n["checked"] += 1
        for mm in ms:
            mm.byte(rec["answer"], t, learn=False)
        for j, mm in enumerate(ms):
            mm.fly.event(max(0.0, mm.pending_t - mm.brain_t), mm.pending_x, float(CH[j] != rec["answer"]),
                         rec["domain"] == "fact" or j < 2)
            _finish_teach(mm)

    drive(world, None, on_teach, feed_all=ms)
    return {"arm": arm, "records_checked": n["checked"]}
