"""Package A V3-CLAUDE store models: every store is born by its own canonical_fresh_native() call.

All classes add only what SPEC_DRAFT.md declares; the frozen parent sources are imported, never edited.
Every value teach returns the applied raw alpha L1 so the four-store wrapper can keep a per-record ledger.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
from scipy.sparse import csr_matrix

import paths  # noqa: F401
import portable_birth as pb
import brain_byte as bb
import content_model
import z1_model

F151 = bb.F151ByteBrain
EXPECTED_B = pb.EXPECTED_B


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _alpha_l1(record: dict) -> float:
    return float(np.abs(record["rawalpha"]).sum()) if record.get("active") else 0.0


class _Teach:
    """Logged native binary teach and the declared signed interface (mixin for F151ByteBrain subclasses)."""

    def _teach_prologue(self, t: float) -> float:
        if self.pending_x is None or self.pending_t is None or abs(float(t) - self.pending_t) > 1e-9:
            raise ValueError("Teach must immediately follow its outcome byte")
        dt = self.pending_t - self.brain_t
        if dt < -1e-9:
            raise ValueError("Teach time moved backwards")
        return max(0.0, dt)

    def _teach_epilogue(self) -> None:
        self.brain_t = self.pending_t
        self.teach_seen += 1
        self.pending_x = None
        self.pending_t = None

    def teach_logged(self, r: int, t: float, *, write: bool) -> float:
        if int(r) not in (0, 1):
            raise ValueError("r must be 0 or 1")
        dt = self._teach_prologue(t)
        record = self.fly.event(dt, self.pending_x, float(r), bool(write))
        self._teach_epilogue()
        return _alpha_l1(record)

    def teach(self, r: int, t: float, *, write: bool = True) -> None:
        self.teach_logged(r, t, write=write)

    def teach_signed(self, b: float, s: float, t: float, *, write: bool) -> float:
        """state <- S_n + b (W(0) - S_n) + s (W(1) - W(0)); exact native at b=1, s in {0,1}."""
        b, s = float(b), float(s)
        if not (math.isfinite(b) and math.isfinite(s)) or b not in (0.0, 1.0) or not -1.0 <= s <= 1.0:
            raise ValueError("invalid signed teacher coefficients")
        dt = self._teach_prologue(t)
        x = self.pending_x
        if not write or (b == 0.0 and s == 0.0):
            self.fly.event(dt, x, 0.0, False)
            self._teach_epilogue()
            return 0.0
        f0, f1 = self.fly.clone(), self.fly.clone()
        r0 = f0.event(dt, x, 0.0, True)
        r1 = f1.event(dt, x, 1.0, True)
        self.fly.event(dt, x, 0.0, False)
        m, m0, m1 = self.fly.m, f0.m, f1.m
        if not (np.array_equal(m.adapt, m0.adapt) and np.array_equal(m.adapt, m1.adapt)
                and float(m.elapsed) == float(m0.elapsed) == float(m1.elapsed)):
            raise AssertionError("signed interface: non-plastic state differs between evaluations")
        fast = m.fast + b * (m0.fast - m.fast) + s * (m1.fast - m0.fast)
        slow = m.slow + b * (m0.slow - m.slow) + s * (m1.slow - m0.slow)
        if not (np.isfinite(fast).all() and np.isfinite(slow).all()):
            raise FloatingPointError("nonfinite signed write")
        m.fast[...] = fast
        m.slow[...] = slow
        applied = 0.0
        if r0.get("active"):
            u0, u1 = np.asarray(r0["rawalpha"]), np.asarray(r1["rawalpha"])
            applied = float(np.abs(b * u0 + s * (u1 - u0)).sum())
        self._teach_epilogue()
        return applied


class NatBrain(_Teach, F151):
    """Native FE0 Full151 store (R shared bank)."""


class CtxBrain(_Teach, content_model.VisibleContextBrain):
    """Frozen CONTENT last-four-visible-byte store (R private bank)."""


def birth(kind: str):
    """One separate canonical newborn, converted to the requested store class."""
    model, raw = pb.canonical_fresh_native()
    canonical = bb.bc.v88().sparse_digest(model.fly.m.B)
    if canonical != EXPECTED_B:
        raise AssertionError("newborn B is not canonical")
    if kind == "native":
        model.__class__ = NatBrain
    elif kind == "content":
        model = content_model.from_native(model)
        model.__class__ = CtxBrain
    else:
        raise ValueError(kind)
    return model, {"raw_B_sha256": raw, "canonical_B_sha256": canonical,
                   "fly_id": id(model.fly)}


# ----------------------------------------------------------------------------- Z family
Z_ARMS = ("Z0_resource", "Z2", "Z2_rand")
Z_TAU_L = 86400.0
Z_KAPPA = 0.2


class Z2Learner(z1_model.Z1Learner):
    def event_gated(self, dt, x, punishment, plastic, gate_fn):
        """Frozen Z1 split certificate with a gate computed from pre-event state and the write's sign."""
        pre_fast = self.m.fast[:, 1].copy()
        pre_slow = self.m.slow.copy()
        record = z1_model._BASE.event(self, dt, x, punishment, bool(plastic))
        if not record.get("active", False) or not plastic:
            return record, None
        u = np.asarray(record["rawalpha"], dtype=np.float64)
        old_slow = np.asarray(record["rawslow"], dtype=np.float64).copy()
        old_fast = np.asarray(record["rawfast"], dtype=np.float64).copy()
        effective, info = gate_fn(u, old_slow, pre_slow)
        effective = np.asarray(effective, dtype=np.float64)
        if effective.shape != u.shape or not np.isfinite(effective).all() or (
                (effective < 0).any() or (effective > 1).any()):
            raise AssertionError("invalid Z gate")
        new_slow = effective * old_slow
        new_fast = old_fast + (old_slow - new_slow)
        df = math.exp(-float(dt) / float(self.m.kernel["fast_tau"]))
        ds = math.exp(-float(dt) / float(self.m.kernel["slow_tau"]))
        self.m.fast[:, 1] += (new_fast - old_fast) * df
        self.m.slow += (new_slow - old_slow) * ds
        split_error = float(np.max(np.abs(new_fast + new_slow - u)))
        patch_error = max(float(np.max(np.abs(self.m.fast[:, 1] - (pre_fast + new_fast) * df))),
                          float(np.max(np.abs(self.m.slow - (pre_slow + new_slow) * ds))))
        if (split_error > 1e-10 or patch_error > 1e-8 or np.any(new_slow * u < 0) or
                np.any(new_fast * u < 0) or np.any(new_slow[u == 0] != 0) or
                np.any(new_fast[u == 0] != 0)):
            raise AssertionError("Z2 violated the inherited alpha split certificate")
        info.update(split_error=split_error, patch_error=patch_error,
                    native_slow_l1=float(np.abs(old_slow).sum()),
                    gated_slow_l1=float(np.abs(new_slow).sum()))
        return record, (effective, info)


class ZDoseNotRepresentable(AssertionError):
    pass


def z_buckets(slow, u, side):
    """kc_side x sign(u) buckets over coordinates with a nonzero native slow share."""
    out = {}
    for s in (0, 1):
        for sign in (-1, 1):
            out[f"{s}:{sign:+d}"] = np.flatnonzero((side == s) & (np.sign(u) == sign) & (slow != 0))
    return out


def _z2rand_gate(g, slow, u, side, key: str, ref: dict | None):
    """Yoked dose diagnostic: permute this store's own Z2 gate within each bucket (SHA256 seed), then match the
    PAIRED Z2 arm's realised gated slow L1 for the same world/branch/record/store/bucket exactly. A reference
    target outside this store's capacity is a qualification failure, never clipped."""
    if ref is None:
        raise ZDoseNotRepresentable("Z2_rand has no paired Z2 reference for this event")
    out = g.copy()
    for name, ids in z_buckets(slow, u, side).items():
        target = float(ref.get(name, 0.0))
        if len(ids) == 0:
            if target > 0.0:
                raise ZDoseNotRepresentable(f"Z2 reference dose {target} in empty bucket {name}")
            continue
        a = np.abs(slow[ids])
        capacity = float(a.sum())
        if target > capacity * (1 + 1e-12):
            raise ZDoseNotRepresentable(f"Z2 reference dose {target} exceeds capacity {capacity} in {name}")
        seed = int.from_bytes(hashlib.sha256(f"{key}|{name}".encode()).digest()[:8], "big")
        pg = g[ids][np.random.default_rng(seed).permutation(len(ids))]
        got = float(a @ pg)
        if got > target:
            pg = pg * (target / got)
        elif got < target:
            room = float(a @ (1.0 - pg))
            pg = pg + (target - got) / room * (1.0 - pg)
        pg = np.clip(pg, 0.0, 1.0)
        if abs(float(a @ pg) - target) > 1e-10 * max(1.0, target):
            raise ZDoseNotRepresentable(f"Z2_rand missed the paired dose in {name}")
        out[ids] = pg
    return out


class ZBrain(_Teach, F151):
    """Native FE0 store with a per-writable-coordinate load L and the Z2 durable-write gate."""

    def _init_z(self, arm: str, world: int, store: int) -> None:
        if arm not in Z_ARMS:
            raise ValueError(arm)
        self.fly.__class__ = Z2Learner
        self.z_arm, self.z_world, self.z_store = arm, int(world), int(store)
        idx = np.flatnonzero(np.abs(np.asarray(self.fly.m.T)[:, 2:4]).sum(axis=1) > 0).astype(np.int32)
        idx.flags.writeable = False
        self.z_indices = idx
        self.z_mask = np.zeros(self.n_native_kc, dtype=bool)
        self.z_mask[idx] = True
        self.z_mask.flags.writeable = False
        self.z_load = np.zeros(self.n_native_kc, dtype=np.float64)
        self.z_t = 0.0
        self.z_events = 0

    def clone(self):
        out = copy.copy(self)
        out.fly = self.fly.clone()
        out.fe = self.fe.clone()
        out.w = self.w.copy()
        out.bias = self.bias.copy()
        out.pending_x = None if self.pending_x is None else self.pending_x.copy()
        out.z_load = self.z_load.copy()
        if out.state_digest() != self.state_digest():
            raise AssertionError("Z clone mismatch")
        return out

    def state_digest(self) -> str:
        return hashlib.sha256((F151.state_digest(self) + _sha(self.z_load) +
                               repr((self.z_t, self.z_events, self.z_arm))).encode()).hexdigest()

    def teach_logged(self, r: int, t: float, *, write: bool) -> float:
        if int(r) not in (0, 1):
            raise ValueError("r must be 0 or 1")
        dt = self._teach_prologue(t)
        now = self.elapsed_base + self.pending_t
        if now < self.z_t - 1e-9:
            raise ValueError("Z load time moved backwards")
        self.z_load *= math.exp(-max(0.0, now - self.z_t) / Z_TAU_L)
        self.z_t = now
        load = self.z_load
        arm = self.z_arm

        ref = getattr(self, "z_ref", None)
        key = f"A3-Z2RAND-v2|{self.z_world}|{getattr(self, 'z_ctx', None)}|{self.z_store}"
        side = np.asarray(self.fly.m.kc_side)

        def gate_fn(u, native_slow, pre_slow):
            conflict = (u * pre_slow < 0) & self.z_mask
            g = np.where(self.z_mask, (1.0 - load) * (1.0 - conflict), 1.0)
            if arm == "Z0_resource":
                eff = np.ones_like(g)
            elif arm == "Z2":
                eff = g
            else:
                eff = _z2rand_gate(g, native_slow, u, side, key, ref)
            a = np.abs(native_slow)
            buckets = {name: [int(len(ids)), float(a[ids].sum()), float(a[ids] @ eff[ids])]
                       for name, ids in z_buckets(native_slow, u, side).items()}
            target = (float(a.sum()) if arm == "Z0_resource" else float(a @ g) if arm == "Z2"
                      else float(sum(ref.values())))
            return eff, {"conflicts": int(conflict.sum()), "z_arm": arm, "gate_target_l1": target,
                         "buckets": buckets}

        record, applied = self.fly.event_gated(dt, self.pending_x, float(r), bool(write), gate_fn)
        if applied is not None:
            eff, info = applied
            moved = (np.asarray(record["rawalpha"]) != 0) & self.z_mask
            self.z_load[moved] += Z_KAPPA * eff[moved] * (1.0 - self.z_load[moved])
            if (self.z_load < 0).any() or (self.z_load > 1 + 1e-12).any():
                raise AssertionError("Z load left [0,1]")
            self.last_z = info
        else:
            self.last_z = None
        self.z_events += 1
        self._teach_epilogue()
        return _alpha_l1(record)


def birth_z(arm: str, world: int, store: int):
    model, receipt = birth("native")
    model.__class__ = ZBrain
    model._init_z(arm, world, store)
    return model, receipt


# ----------------------------------------------------------------------------- P family
P_ARMS = ("P0", "P1", "P2", "P4")
P_EPS = 0.05
P_TAU = 330.0
P_FLOOR = 1e-9


class PBrain(_Teach, F151):
    """Native FE0 store with mutable existing PN->KC weights on the canonical fixed support."""

    def _init_p(self, arm: str) -> None:
        if arm not in P_ARMS:
            raise ValueError(arm)
        B = self.fly.m.B.tocsr()
        indices, indptr = B.indices, B.indptr
        indices.flags.writeable = False
        indptr.flags.writeable = False
        data = B.data.copy()
        data.flags.writeable = False
        self.fly.m.B = csr_matrix((data, indices, indptr), shape=B.shape, copy=False)
        self.p_arm = arm
        self.p_rows = np.repeat(np.arange(B.shape[0]), np.diff(indptr)).astype(np.int32)
        self.p_cols = indices
        n_kc = B.shape[1]
        deg = np.bincount(indices, minlength=n_kc)
        self.p_budget = np.bincount(indices, weights=data, minlength=n_kc)
        self.p_mean = np.divide(self.p_budget, deg, out=np.zeros(n_kc), where=deg > 0)
        for a in (self.p_rows, self.p_budget, self.p_mean):
            a.flags.writeable = False
        self.p_support = (_sha(indices), _sha(indptr), B.shape)
        self.p_pre = np.zeros(B.shape[0], dtype=np.float64)
        self.p_post = np.zeros(n_kc, dtype=np.float64)
        self.p_t = None
        self.pending_p = None
        self._feat_p = None
        self.p_log = []
        self.p_act = np.zeros(n_kc, dtype=np.int64)
        self.p_updates = 0
        self.p_diag = []

    def clone(self):
        out = copy.copy(self)
        out.fly = self.fly.clone()
        out.fe = self.fe.clone()
        out.w = self.w.copy()
        out.bias = self.bias.copy()
        out.pending_x = None if self.pending_x is None else self.pending_x.copy()
        out.pending_p = None if self.pending_p is None else self.pending_p.copy()
        out.p_pre, out.p_post = self.p_pre.copy(), self.p_post.copy()
        out.p_log = []
        out.p_act = self.p_act.copy()
        out.p_diag = []
        if out.state_digest() != self.state_digest():
            raise AssertionError("P clone mismatch")
        return out

    def state_digest(self) -> str:
        B = self.fly.m.B
        return hashlib.sha256((F151.state_digest(self) + _sha(B.data) + _sha(self.p_pre) +
                               _sha(self.p_post) + _sha(self.p_act) + repr((self.p_t, self.p_arm, self.p_updates))).encode()).hexdigest()

    def _features(self, t, *, fe=None, fly_m=None):
        out = super()._features(t, fe=fe, fly_m=fly_m)
        if fe is None and fly_m is None:
            self._feat_p = self.fe.read().copy()
        return out

    def byte(self, b, t, learn=True, **kw):
        loss = super().byte(b, t, learn=learn, **kw)
        self.pending_p = self._feat_p
        return loss

    def teach_logged(self, r: int, t: float, *, write: bool) -> float:
        a_pn = None if self.pending_p is None else self.pending_p[self.fly.m.pn_type_index]
        x = self.pending_x
        when = self.elapsed_base + (self.pending_t if self.pending_t is not None else 0.0)
        applied = super().teach_logged(r, t, write=write)
        if a_pn is None or x is None:
            raise AssertionError("P update lacks pre-answer PN activity / KC code")
        self._p_update(np.asarray(a_pn, dtype=np.float64), np.asarray(x, dtype=np.float64), when)
        self.pending_p = None
        return applied

    def _p_update(self, a: np.ndarray, x: np.ndarray, when: float) -> None:
        B = self.fly.m.B
        if (_sha(B.indices), _sha(B.indptr), B.shape) != self.p_support:
            raise AssertionError("P support changed")
        decay = 0.0 if self.p_t is None else math.exp(-max(0.0, when - self.p_t) / P_TAU)
        self.p_pre *= decay
        self.p_post *= decay
        rows, cols = self.p_rows, self.p_cols
        mk = self.p_mean[cols]
        v = B.data / mk
        xa = x[cols]
        if self.p_arm in ("P0", "P1"):
            dv = P_EPS * a[rows] * xa
        elif self.p_arm == "P2":
            dv = P_EPS * xa * (a[rows] - xa * v)
        else:
            dv = P_EPS * (self.p_pre[rows] * xa - a[rows] * self.p_post[cols])
        self.p_pre += a
        self.p_post += x
        self.p_t = when
        self.p_act += (x > 0)
        self.p_updates += 1
        delta_l1 = float(np.abs(dv).sum())
        if self.p_arm == "P0":
            self.p_log.append((round(delta_l1, 9), 0.0, float(B.data.max()), float(B.data.min())))
            self._p_diagnostics(B.data)
            return
        w = np.maximum(v + dv, P_FLOOR) * mk
        sums = np.bincount(cols, weights=w, minlength=B.shape[1])
        scale = np.divide(self.p_budget, sums, out=np.ones_like(sums), where=sums > 0)
        w = w * scale[cols]
        if not np.isfinite(w).all() or (w <= 0).any():
            raise FloatingPointError("P weights nonfinite or nonpositive")
        change = float(np.abs(w - B.data).sum())
        w.flags.writeable = False
        self.fly.m.B = csr_matrix((w, B.indices, B.indptr), shape=B.shape, copy=False)
        if not (np.shares_memory(self.fly.m.B.indices, B.indices) and
                np.shares_memory(self.fly.m.B.indptr, B.indptr)):
            raise AssertionError("P support arrays were copied")
        self.p_log.append((round(delta_l1, 9), round(change, 9), float(w.max()), float(w.min())))
        self._p_diagnostics(w)

    def _p_diagnostics(self, w: np.ndarray) -> None:
        """Concentration diagnostics after update 336 (end of old stage) and 600 (end of life)."""
        if self.p_updates not in (336, 600):
            return
        cols = self.p_cols
        n_kc = len(self.p_budget)
        deg = np.bincount(cols, minlength=n_kc)
        v = w / self.p_mean[cols]
        s1 = np.bincount(cols, weights=w, minlength=n_kc)
        s2 = np.bincount(cols, weights=w * w, minlength=n_kc)
        wmax = np.zeros(n_kc)
        np.maximum.at(wmax, cols, w)
        multi = deg >= 2
        pr = s1[multi] ** 2 / s2[multi]
        eff = pr / deg[multi]
        act = self.p_act
        self.p_diag.append({
            "update": self.p_updates,
            "edges_rel_below_1e-3": int((v < 1e-3).sum()), "edges_rel_below_1e-6": int((v < 1e-6).sum()),
            "edges": int(len(w)),
            "kc_fanin_participation_q05_q50_q95": [float(q) for q in np.quantile(pr, [0.05, 0.5, 0.95])],
            "kc_effective_fraction_mean": float(eff.mean()),
            "kc_max_share_above_0.9": int((wmax[multi] / s1[multi] > 0.9).sum()), "kcs_multi_input": int(multi.sum()),
            "kc_activations_q50_q95_q99_max": [float(q) for q in np.quantile(act, [0.5, 0.95, 0.99])] + [int(act.max())],
            "kc_active_every_update": int((act == self.p_updates).sum())})


def birth_p(arm: str):
    model, receipt = birth("native")
    model.__class__ = PBrain
    model._init_p(arm)
    return model, receipt
