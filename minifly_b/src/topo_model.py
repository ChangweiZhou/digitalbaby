"""Full151 FE0 byte brain with a T-family static graph or an S-family living graph.

SPEC_LOCK.md §S is the authority for every constant and equation below.

Copy-on-write contract: ``fly.m.B`` and every S-state array are *replaced*,
never mutated in place, and all are flagged read-only.  A clone may therefore
share them; the first structural write of either copy installs new objects.
Structural state changes only inside ``teach(..., write=True)``; the byte feed,
value read and probe clones never touch it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
from scipy.sparse import csr_matrix

import paths  # noqa: F401
import portable_birth as pb
from common_platform import FourStore, bb
from t2_graph import CounterRng, legacy_sparse_digest

VERSION = "TOPO-MODEL-v1"
S_RULES = ("S1", "S2", "S3", "S4")
S_ARMS = ("S0",) + S_RULES + tuple(f"Srand_{r[1]}" for r in S_RULES)
T_ARMS = ("T1", "T0_1", "T3", "T0_3")
ALL_ARMS = T_ARMS + S_ARMS

# ---- locked S constants (SPEC_LOCK.md §S.2) --------------------------------
K_EVENT = 24          # write-enabled teach calls per store between structural events
R_MAX = 64            # maximum partner replacements per structural event per store
POOL = 16             # birth-fixed candidate PNs per KC (outside canonical partners)
TAU_S = 86400.0       # evidence decay time constant, model seconds
RHO_TARGET = 0.05     # S3 homeostatic target = Full151 active_fraction (source constant)
RHO_EPS = 1e-3
S3_BAND = math.log(2.0)
S4_TRIAL = 2          # tentative lifetime in structural events


class TopoError(RuntimeError):
    pass


def _ro(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def graph_bytes_digest(b: csr_matrix) -> str:
    h = hashlib.sha256(VERSION.encode())
    h.update(np.asarray(b.shape, dtype="<i8").tobytes())
    for a in (b.indptr, b.indices, b.data):
        h.update(str(a.dtype).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _hash_rank(key: bytes, i: int) -> bytes:
    return hashlib.sha256(key + int(i).to_bytes(4, "big")).digest()


class Support:
    """Birth-fixed per-KC support slots: canonical partners plus a POOL-PN pool.

    Identical for every S arm and store of one world (key: world only).
    """

    def __init__(self, b_canonical: csr_matrix, world: int):
        csc = b_canonical.tocsc()
        coo = b_canonical.tocoo()
        P, K = b_canonical.shape
        rng = CounterRng(f"SFAM-v1:{world}", "pool")
        kc_ptr = [0]
        pn, epos, w, on = [], [], [], []
        # canonical CSR data position of every (p,k) edge
        pos = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(coo.row, coo.col))}
        # tocoo of a CSR preserves data order, so i is the canonical data index
        for k in range(K):
            partners = [int(p) for p in csc.indices[csc.indptr[k]:csc.indptr[k + 1]]]
            members = set(partners)
            pool = []
            if partners:
                while len(pool) < POOL:
                    p = rng.randbelow(P)
                    if p not in members:
                        members.add(p)
                        pool.append(p)
            for p in sorted(set(partners) | set(pool)):
                pn.append(p)
                if p in partners:
                    e = pos[(p, k)]
                    epos.append(e)
                    w.append(float(b_canonical.data[e]))
                    on.append(True)
                else:
                    epos.append(-1)
                    w.append(0.0)
                    on.append(False)
            kc_ptr.append(len(pn))
        self.world = world
        self.shape = (P, K)
        self.kc_ptr = _ro(np.asarray(kc_ptr, dtype=np.int64))
        self.pn = _ro(np.asarray(pn, dtype=np.int32))
        self.kc = _ro(np.repeat(np.arange(K, dtype=np.int32), np.diff(self.kc_ptr)))
        self.init_epos = _ro(np.asarray(epos, dtype=np.int64))
        self.init_w = _ro(np.asarray(w, dtype=np.float64))
        self.init_on = _ro(np.asarray(on, dtype=bool))
        self.n = len(pn)
        tkey = hashlib.sha256(f"SFAM-v1|{world}|tie".encode()).digest()
        self.tie = _ro(np.asarray([int.from_bytes(_hash_rank(tkey, i)[:8], "big")
                                   for i in range(self.n)], dtype=np.uint64))
        self.tie_kc = _ro(np.asarray([int.from_bytes(_hash_rank(tkey + b"kc", k)[:8], "big")
                                      for k in range(K)], dtype=np.uint64))
        h = hashlib.sha256()
        for a in (self.kc_ptr, self.pn, self.init_epos, self.init_w, self.init_on):
            h.update(np.ascontiguousarray(a).tobytes())
        self.digest = h.hexdigest()


def build_csr(sup: Support, on: np.ndarray, epos: np.ndarray, w: np.ndarray) -> csr_matrix:
    idx = np.flatnonzero(on)
    rows = sup.pn[idx]
    order = np.lexsort((epos[idx], rows))
    idx = idx[order]
    P, K = sup.shape
    indptr = np.zeros(P + 1, dtype=np.int32)
    np.cumsum(np.bincount(sup.pn[idx], minlength=P), out=indptr[1:])
    b = csr_matrix((w[idx].copy(), sup.kc[idx].astype(np.int32), indptr), shape=(P, K))
    for a in (b.data, b.indices, b.indptr):
        a.flags.writeable = False
    return b


class TopoBrain(bb.F151ByteBrain):
    """One of the four output stores; ``rule`` selects T static or S living graph."""

    # ---------------------------------------------------------------- birth
    @classmethod
    def newborn(cls, arm: str, world: int, *, static_graph: csr_matrix | None = None,
                support: Support | None = None, replay: dict | None = None):
        if arm not in ALL_ARMS:
            raise TopoError("unknown arm")
        base, raw_digest = pb.canonical_fresh_native()
        if legacy_sparse_digest(base.fly.m.B) != pb.EXPECTED_B:
            raise TopoError("canonical B not installed")
        out = copy.copy(base)
        out.__class__ = cls
        out.arm, out.world, out.store = arm, int(world), None
        out.branch = None
        out.raw_b_digest = raw_digest
        out.canonical_b_digest = pb.EXPECTED_B
        out.rule = None
        if arm in T_ARMS:
            if static_graph is None:
                raise TopoError("T arm requires its birth graph")
            for a in (static_graph.data, static_graph.indices, static_graph.indptr):
                if a.flags.writeable:
                    raise TopoError("T graph must be read-only")
            out.fly.m.B = static_graph
        else:
            if support is None or support.world != world:
                raise TopoError("S arm requires the world's support")
            out.rule = arm if arm in S_RULES or arm == "S0" else "Srand"
            out.sup = support
            out.on, out.epos, out.sw = support.init_on, support.init_epos, support.init_w
            P, K = support.shape
            out.C = _ro(np.zeros(support.n))
            out.U = _ro(np.zeros(K))
            out.A = _ro(np.zeros(P))
            out.nev = 0.0
            out.last_ev_t = None
            out.writes = 0
            out.n_events = 0
            out.tentative = ()   # tuples (k, s_new, s_old, start_event, Cn, Co)
            out.replay = replay  # Srand only: {(branch, store, n): count}
            out.pending_p = None
            if arm.startswith("Srand") and replay is None:
                raise TopoError("Srand needs the candidate's realised schedule")
            b0 = build_csr(support, out.on, out.epos, out.sw)
            if legacy_sparse_digest(b0) != pb.EXPECTED_B:
                raise TopoError("S support does not reproduce canonical B")
            out.fly.m.B = b0
        out.new_events = []
        out._gdig = graph_bytes_digest(out.fly.m.B)
        out._sdig = None
        return out

    # ---------------------------------------------------------------- cloning
    def clone_round(self):
        out = copy.copy(self)
        out.fly = self.fly.clone()
        out.fe = self.fe.clone()
        out.w = self.w.copy()
        out.bias = self.bias.copy()
        out.pending_x = None if self.pending_x is None else self.pending_x.copy()
        if self.rule is not None:
            out.pending_p = None if self.pending_p is None else self.pending_p.copy()
        out.new_events = []
        if out.fly.m.B is not self.fly.m.B:
            raise TopoError("graph clone sharing contract failed")
        return out

    # ---------------------------------------------------------------- digests
    def _s_digest(self) -> str:
        if self._sdig is None:
            h = hashlib.sha256(VERSION.encode())
            if self.rule is not None:
                for a in (self.on, self.epos, self.sw, self.C, self.U, self.A):
                    h.update(np.ascontiguousarray(a).tobytes())
                h.update(json.dumps([self.nev.hex(), self.last_ev_t, self.writes, self.n_events,
                                     [list(x[:4]) + [float(x[4]).hex(), float(x[5]).hex()]
                                      for x in self.tentative]]).encode())
            self._sdig = h.hexdigest()
        return self._sdig

    _NATIVE_KEYS = ("w", "bias", "fly_fast", "fly_slow", "fly_adapt", "fe_p", "pending_x")

    def native_digest(self) -> str:
        """Exactly F151ByteBrain.state_digest over the parent snapshot (no topology)."""
        s = bb.F151ByteBrain.snapshot(self)
        arrays = tuple(s[k] for k in self._NATIVE_KEYS)
        meta = {k: v for k, v in s.items() if k not in self._NATIVE_KEYS}
        return bb._digest_arrays(arrays, meta)

    def state_digest(self) -> str:
        return hashlib.sha256(f"{self.native_digest()}|{self._gdig}|{self._s_digest()}|{self.arm}"
                              .encode()).hexdigest()

    # ---------------------------------------------------------------- checkpoint
    def snapshot(self) -> dict:
        state = super().snapshot()
        b = self.fly.m.B
        topo = {"version": VERSION, "arm": self.arm, "world": self.world, "store": self.store,
                "B_indptr": np.asarray(b.indptr).copy(), "B_indices": np.asarray(b.indices).copy(),
                "B_data": np.asarray(b.data).copy(), "graph_digest": self._gdig}
        if self.rule is not None:
            topo.update(support=self.sup.digest, on=self.on.copy(), epos=self.epos.copy(),
                        sw=self.sw.copy(), C=self.C.copy(), U=self.U.copy(), A=self.A.copy(),
                        nev=self.nev, last_ev_t=self.last_ev_t, writes=self.writes,
                        n_events=self.n_events, tentative=[list(x) for x in self.tentative],
                        pending_p=None if self.pending_p is None else self.pending_p.copy())
        state["topo"] = topo
        return state

    def restore(self, state: dict) -> None:
        topo = state["topo"]
        if (topo["version"], topo["arm"], topo["world"]) != (VERSION, self.arm, self.world):
            raise TopoError("checkpoint topology identity mismatch")
        b = csr_matrix((np.array(topo["B_data"]), np.array(topo["B_indices"]),
                        np.array(topo["B_indptr"])), shape=self.fly.m.B.shape)
        for a in (b.data, b.indices, b.indptr):
            a.flags.writeable = False
        if graph_bytes_digest(b) != topo["graph_digest"]:
            raise TopoError("checkpoint graph bytes do not match their digest")
        super().restore({k: v for k, v in state.items() if k != "topo"})
        self.fly.m.B = b
        self._gdig = topo["graph_digest"]
        self.store = topo["store"]
        if self.rule is not None:
            if topo["support"] != self.sup.digest:
                raise TopoError("checkpoint support mismatch")
            self.on, self.epos, self.sw = (_ro(np.array(topo[k])) for k in ("on", "epos", "sw"))
            self.C, self.U, self.A = (_ro(np.array(topo[k])) for k in ("C", "U", "A"))
            self.nev, self.last_ev_t = float(topo["nev"]), topo["last_ev_t"]
            self.writes, self.n_events = int(topo["writes"]), int(topo["n_events"])
            self.tentative = tuple(tuple(x) for x in topo["tentative"])
            self.pending_p = None if topo["pending_p"] is None else np.array(topo["pending_p"])
            if graph_bytes_digest(build_csr(self.sup, self.on, self.epos, self.sw)) != self._gdig:
                raise TopoError("checkpoint slot state does not rebuild its graph")
        self._sdig = None

    # ---------------------------------------------------------------- sensing
    def byte(self, b, t, learn=True, **kw):
        if self.rule is not None:
            # Same FE0 advance the parent performs first; dt=0 on its own re-advance.
            self.fe.advance(float(t))
            p = self.fe.read()
        loss = super().byte(b, t, learn, **kw)
        if self.rule is not None:
            self.pending_p = p  # PN-type activity that produced pending_x
        return loss

    # ---------------------------------------------------------------- teaching
    def teach(self, r, t, *, write=True):
        x = self.pending_x
        p = self.pending_p if self.rule is not None else None
        super().teach(r, t, write=write)
        if self.rule is None or not write:
            if self.rule is not None:
                self.pending_p = None
            return None
        if x is None or p is None:
            raise TopoError("write without pre-outcome code")
        self.pending_p = None
        self._evidence(np.flatnonzero(x > 0), p, float(t))
        self.writes += 1
        if self.writes % K_EVENT == 0:
            self.n_events += 1
            self._structural_event(float(t))
        self._sdig = None
        return None

    def _evidence(self, active: np.ndarray, p_type: np.ndarray, t: float) -> None:
        sup = self.sup
        d = 1.0 if self.last_ev_t is None else math.exp(-(t - self.last_ev_t) / TAU_S)
        a_pn = np.asarray(p_type, dtype=np.float64)[self.fly.m.pn_type_index]
        C = self.C * d
        lo, hi = sup.kc_ptr[active], sup.kc_ptr[active + 1]
        slots = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(active) else np.zeros(0, np.int64)
        C[slots] += a_pn[sup.pn[slots]]
        U = self.U * d
        U[active] += 1.0
        self.C, self.U = _ro(C), _ro(U)
        self.A = _ro(self.A * d + a_pn)
        self.nev = self.nev * d + 1.0
        self.last_ev_t = t
        if self.tentative:
            act = set(active.tolist())
            upd = []
            for (k, s_new, s_old, start, cn, co) in self.tentative:
                if k in act:
                    cn += float(a_pn[sup.pn[s_new]])
                    co += float(a_pn[sup.pn[s_old]])
                upd.append((k, s_new, s_old, start, cn, co))
            self.tentative = tuple(upd)

    # ---------------------------------------------------------------- structure
    def _move(self, on, epos, sw, s_old: int, s_new: int) -> None:
        if not on[s_old] or on[s_new] or self.sup.kc[s_old] != self.sup.kc[s_new]:
            raise TopoError("illegal partner move")
        on[s_old], on[s_new] = False, True
        epos[s_new], epos[s_old] = epos[s_old], -1
        sw[s_new], sw[s_old] = sw[s_old], 0.0

    def _structural_event(self, t: float) -> None:
        if self.rule == "S0":
            self.new_events.append({"n": self.n_events, "t": t, "rule": "S0", "changes": [],
                                    "graph": self._gdig})
            return
        sup = self.sup
        K = sup.shape[1]
        on, epos, sw = self.on.copy(), self.epos.copy(), self.sw.copy()
        rng = CounterRng(f"SFAM-v1:{self.world}:{self.arm}:{self.store}", f"event{self.n_events}")
        changes = []   # [kc, pn_old, pn_new, kind, evidence_old, evidence_new]
        nfree = np.bincount(sup.kc[~on], minlength=K)
        non = np.bincount(sup.kc[on], minlength=K)

        def move(s_old, s_new, kind, eo, en):
            self._move(on, epos, sw, int(s_old), int(s_new))
            changes.append([int(sup.kc[s_old]), int(sup.pn[s_old]), int(sup.pn[s_new]), kind,
                            float(eo), float(en)])

        def first_per_kc(mask, *keys):
            """Slot index of the first slot per KC under lexicographic keys (primary last)."""
            idx = np.flatnonzero(mask)
            order = np.lexsort(tuple(k[idx] for k in keys) + (sup.kc[idx],))
            idx = idx[order]
            kcs = sup.kc[idx]
            head = np.ones(len(idx), bool)
            head[1:] = kcs[1:] != kcs[:-1]
            out = np.full(K, -1, np.int64)
            out[kcs[head]] = idx[head]
            return out

        def free_slots(k):
            r = np.arange(sup.kc_ptr[k], sup.kc_ptr[k + 1])
            return r[~on[r]]

        elig_kc = (non > 0) & (nfree > 0)
        if self.rule == "S1":
            live = np.flatnonzero(on & elig_kc[sup.kc])
            order = np.lexsort((sup.tie[live], sw[live]))
            for s in live[order[:R_MAX]]:
                cand = free_slots(int(sup.kc[s]))
                move(s, cand[rng.randbelow(len(cand))], "set", sw[s], sw[s])
        elif self.rule == "S2":
            C = self.C
            weak = first_per_kc(on, sup.tie, C)                  # min C, tie rank
            best = first_per_kc(~on, sup.pn, -C)                 # max C, lower PN
            ks = np.flatnonzero(elig_kc)
            gain = C[best[ks]] - C[weak[ks]]
            ks, gain = ks[gain > 0], gain[gain > 0]
            order = np.lexsort((sup.tie_kc[ks], -gain))[:R_MAX]
            for k in ks[order]:
                move(weak[k], best[k], "coact", C[weak[k]], C[best[k]])
        elif self.rule == "S3":
            A = self.A[sup.pn]
            nev = max(self.nev, 1e-12)
            dev = np.log((self.U / nev + RHO_EPS) / RHO_TARGET)
            ks = np.flatnonzero(elig_kc & (np.abs(dev) > S3_BAND))
            order = np.lexsort((sup.tie_kc[ks], -np.abs(dev[ks])))[:R_MAX]
            lo_on = first_per_kc(on, sup.pn, A)
            hi_on = first_per_kc(on, sup.pn, -A)
            hi_free = first_per_kc(~on, sup.pn, -A)
            lo_free = first_per_kc(~on, sup.pn, A)
            for k in ks[order]:
                if dev[k] < 0:
                    so, sn = lo_on[k], hi_free[k]
                    ok = A[sn] > A[so]
                else:
                    so, sn = hi_on[k], lo_free[k]
                    ok = A[sn] < A[so]
                if ok:
                    move(so, sn, "homeo", A[so], A[sn])
        elif self.rule == "S4":
            keep = []
            for (k, s_new, s_old, start, cn, co) in self.tentative:
                if self.n_events - start >= S4_TRIAL:
                    if cn > co:
                        changes.append([k, int(sup.pn[s_old]), int(sup.pn[s_new]), "accept", co, cn])
                    else:
                        self._move(on, epos, sw, s_new, s_old)
                        changes.append([k, int(sup.pn[s_new]), int(sup.pn[s_old]), "rollback", cn, co])
                else:
                    keep.append((k, s_new, s_old, start, cn, co))
            nfree = np.bincount(sup.kc[~on], minlength=K)
            busy = np.zeros(K, bool)
            busy[[x[0] for x in keep]] = True
            elig = [int(k) for k in np.flatnonzero((non > 0) & (nfree > 0) & ~busy)]
            n = min(R_MAX, len(elig))
            for i in range(n):     # partial Fisher-Yates on the counter stream
                j = i + rng.randbelow(len(elig) - i)
                elig[i], elig[j] = elig[j], elig[i]
            C = self.C
            weak = first_per_kc(on, sup.tie, C)
            for k in elig[:n]:
                so = int(weak[k])
                cand = free_slots(k)
                sn = int(cand[rng.randbelow(len(cand))])
                move(so, sn, "tentative", C[so], C[sn])
                keep.append((k, sn, so, self.n_events, 0.0, 0.0))
            self.tentative = tuple(keep)
        elif self.rule == "Srand":
            want = self.replay.get((self.branch, self.store, self.n_events))
            if want is None:
                raise TopoError("Srand replay schedule missing an event")
            for _ in range(int(want)):
                live = np.flatnonzero(on & elig_kc[sup.kc])
                s = live[rng.randbelow(len(live))]
                cand = free_slots(int(sup.kc[s]))
                move(s, cand[rng.randbelow(len(cand))], "random", sw[s], sw[s])
        else:
            raise TopoError("unknown rule")
        self.on, self.epos, self.sw = _ro(on), _ro(epos), _ro(sw)
        if changes:
            self.fly.m.B = build_csr(sup, self.on, self.epos, self.sw)
            self._gdig = graph_bytes_digest(self.fly.m.B)
        if self.fly.m.B.nnz != int(sup.init_on.sum()) or int(on.sum()) != int(sup.init_on.sum()):
            raise TopoError("edge count changed")
        self.new_events.append({"n": self.n_events, "t": t, "rule": self.rule,
                                "changes": changes, "graph": self._gdig})




class TopoFourStore(FourStore):
    """Four independent stores; records structural events with their record context."""

    def __init__(self, stores):
        super().__init__(stores)
        for j, m in enumerate(self.stores):
            m.store = j

    def clone(self):
        out = TopoFourStore.__new__(TopoFourStore)
        FourStore.__init__(out, [m.clone_round() for m in self.stores])
        for j, (a, b) in enumerate(zip(out.stores, self.stores)):
            a.store = b.store
            if a.state_digest() != b.state_digest():
                raise TopoError("clone state digest mismatch")
        out.mechanism_events = list(self.mechanism_events)
        out.context = self.context
        out.is_base = False
        if getattr(self, "is_base", False):
            self.branch_children.append(out)   # the life driver's five branch births
        return out

    def mark_base(self):
        self.is_base = True
        self.branch_children = []
        return self

    def set_record_context(self, world, branch, index, stage, domain):
        super().set_record_context(world, branch, index, stage, domain)
        for m in self.stores:
            m.branch = branch

    def teach(self, answer, t, domain, *, write):
        super().teach(answer, t, domain, write=write)
        for j, m in enumerate(self.stores):
            for ev in m.new_events:
                self.mechanism_events.append({"context": list(self.context), "store": j, **ev})
            m.new_events = []
