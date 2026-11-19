"""E63, decision 66 of the coordinator (19 November 2026): a copy of tissue_step1.py that does work only where something is active.
tissue_step1.py is not edited. Three changes, each switchable for timing (Tissue(..., opts=...), default all three), and no other:
  (a) the search for a new source (formula 12) leaves out the cells flagged "no candidate" (all of their neighbourhood's members are
      already sources); the flag is set when a making fills a cell, cleared when one of its connections is given up, and computed
      from the sources at birth and on loading a saved state. No random number is drawn for such a cell in the original either.
  (b) formula 7, and the excitatory part of the coincidence rule, are applied only at connections whose receiving cell has x_prev != 0
      (both contain the factor x_prev); the inhibitory part of the coincidence rule (x - a*) is applied as before.
  (c) the arrival (formula 1) is summed over the connections whose sender's value zs is not zero.
The fifth tissue (withdrawal by use): its lines are timed as they are (Tissue.t_use); opts may hold "maxsort", which replaces
np.maximum.at by a maximum over a stable sort of the receivers (np.maximum.reduceat), the same numbers. Everything else is the original's.

The original's text follows.

E63, step 1 of the roadmap: the tissue reads the stream and stays informative.

Brief: TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md, revision 3 (on the reviewer's branch, in
PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/). The tissue is THE_TISSUE_DEFINED_REVISION_5_20261119.md ("the
definition"), with the errata and decisions of sections 5.2 to 5.8 of the brief, and nothing else. Python 3.13 and numpy only; the program
imports nothing from the repository.

    python3 tissue_step1.py run <first|coincidence|fixed> <seed> [--bytes N] [--stage A|B] [--formula11-as-written] [--out DIR]
                                [--ckpt-bytes N] [--resume] [--die-after-byte N]
    python3 tissue_step1.py check <1..10>
    python3 tissue_step1.py run first 73129 --withdraw-by-use     # the fifth tissue of stage A (the coordinator's decision 62)

The fifth tissue: formula 11 as changed at the head of revision 5 of the definition (01:18 UTC, 19 November). Each incoming connection
from a cell carries its use, c_ij(k + 1) = (127/128) c_ij(k) + (1/128) (e_ij(k) - ē_ij(k)) (x_i(k) - ā_i(k)), where ē_ij(k + 1) =
(127/128) ē_ij(k) + (1/128) e_ij(k) is the trace's running average over 128 steps and ā_i is the cell's own (formula 10). A connection is
withdrawn when |c_ij| has stayed below (1/128) of the largest |c| among the cell's connections from cells for 128 consecutive steps; and,
as before, when it is at least P_s steps old and its strength is below 1/128. Choices of the worker where the paragraph leaves one: the
trace's average has the same 128 steps as the activity's; the product uses the trace held before the step and x(k), computed in operation
6; the comparison is with the largest |c| among the cell's ordinary connections from cells at that step, and when that largest is 0
nothing is withdrawn by use; connections from feedback cells are exempt (at step 1 they are silent by construction, so their use cannot
be judged, and withdrawing them would remove the route of feedback before step 2); connections from lines are not concerned ("from a
cell"); a new connection starts with c = ē = 0 and a count of 0.

Section numbers below are the brief's unless "definition" is said. Formula numbers are the definition's (formulas 1 to 12).
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from random import Random

import numpy as np

HERE = Path(__file__).resolve().parent
BRIEF = "TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md"
REVIEWER_REF = "origin/claude/vibrant-cerf-sje726"

# ------------------------------------------------------------------------------------------------ numbers of section 5.7
N = 2048                      # cells
NFB = 64                      # feedback cells (32 yes, 32 no)
NIN = NOUT = 256              # input lines, output lines
R2 = (1 / 8) ** 2             # radius of a neighbourhood, squared (section 5.8, item 6)
START_N = 8                   # starting number of connections
START_W = 1 / 8               # starting strength; strength of a line connection
TAU_V = 2                     # formula 2
TAU = 32                      # formulas 4 and 6
ETA = 1 / 8                   # formula 7 and the coincidence rule
ETA_T = 1 / 128               # formula 9
ASTAR = 1 / 8                 # a*
PS = 128                      # P_s (formulas 10 to 12)
GIVE_UP = 1 / 128             # formula 11 (on the strength only, E10)
S = 32                        # steps a byte
EMIT = 1 / 2                  # emission level (sentence 75)
TAU_VT = 4096                 # sliding threshold of the coincidence rule (section 5.5)
LINE0 = N                     # input line of byte v is LINE0 + v
YES_LINE, NO_LINE = N + NIN, N + NIN + 1
NZ = N + NIN + 2              # length of the vector z of values reaching cells
REC_FLOOR = 0.01              # recorded outputs below this are stored as 0 (section 1)
SEEDS = (73129, 93683, 116089)

# receiver kinds and source kinds (section 7.1, item 4)
ENC, INNER, OUTE, FBK = 0, 1, 2, 3
RK_NAMES = {ENC: "encoder", INNER: "inner", OUTE: "output-end"}
SK_LINE_USED, SK_LINE_UNUSED, SK_EXC, SK_INH, SK_FBC, SK_FBLINE = 0, 1, 2, 3, 4, 5
SK_NAMES = {SK_LINE_USED: "input line of a byte value in use", SK_LINE_UNUSED: "input line of a value not in use",
            SK_EXC: "excitatory cell", SK_INH: "inhibitory cell", SK_FBC: "feedback cell"}


# ================================================================================================ formulas (definition, section 4)
def f1_arrival(sig, w, z):                       # formula 1, for one cell (the tissue uses numpy.bincount of the same products)
    return sum(s * ww * zz for s, ww, zz in zip(sig, w, z))


def f2_v(v_prev, A):                             # formula 2
    return (1 - 1 / TAU_V) * v_prev + (1 / TAU_V) * A


def f3_x(v, th):                                 # formula 3
    return np.minimum(1.0, np.maximum(0.0, v - th))


def f4_p(p, A):                                  # formula 4
    return (1 - 1 / TAU) * p + (1 / TAU) * A


def f5_m(A, p):                                  # formula 5
    return A - p


def f6_trace(e, z):                              # formula 6
    return (1 - 1 / TAU) * e + (1 / TAU) * z


def f7_dw(sig, m, x_prev, e):                    # formula 7
    return -ETA * sig * (m * x_prev) * e


def coin_exc_dw(e, x_prev, vt):                  # section 5.5, from an excitatory cell into an inner cell
    return ETA * e * x_prev * (x_prev - vt)


def coin_inh_dw(e, x_prev):                      # section 5.5, from an inhibitory cell into an inner cell
    return ETA * e * (x_prev - ASTAR)


def f8_bound(w):                                 # formula 8
    return np.minimum(1.0, np.maximum(0.0, w))


def f9_theta(th, x):                             # formula 9
    return np.maximum(0.0, th + ETA_T * (x - ASTAR))


def f10_abar(a, x):                              # formula 10
    return (1 - 1 / PS) * a + (1 / PS) * x


def vt_update(vt, x):                            # section 5.5, the sliding threshold
    return (1 - 1 / TAU_VT) * vt + (1 / TAU_VT) * x * x / ASTAR


def use_update(c, e, ebar, x, abar):            # the fifth tissue of stage A (decision 62): use of an input to the receiving cell
    return (1 - 1 / PS) * c + (1 / PS) * (e - ebar) * (x - abar)


def ubar_update(u, z):                           # revision 4's sending-side average (only the 512-byte tissue of stage A)
    return (1 - 1 / PS) * u + (1 / PS) * z


# ================================================================================================ the stream (section 4)
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
END = 10
KMAX = 12


def apply_op(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt + list(f"{apply_op(op, m, a, b):02d}".encode()) + [END]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    Random(seed * 1000 + round(frac * 100) + m * 7 + code[0]).shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def stream(seed, n):
    """The first n bytes of the stream of a seed (section 4), and the items begun (task, a, b, k, start)."""
    rng = Random(seed + 77)
    flat = [(t, pr) for t in DEV for pr in split(t, .5, seed)[0]]
    out, its = [], []
    while len(out) < n:
        order = list(range(len(flat))); rng.shuffle(order)
        for i in order:
            t, (a, b) = flat[i]
            k = rng.randint(0, KMAX)
            its.append((t[0], a, b, k, len(out)))
            out.extend(render(t, a, b, k, rng))
            if len(out) >= n:
                break
    return np.array(out[:n], dtype=np.uint8), its


# ================================================================================================ the birth (section 5.3)
def birth(seed):
    rb = Random(seed)
    P = [(rb.random(), rb.random()) for _ in range(N)]
    IN_Y = [rb.random() for _ in range(NIN)]
    OUT_Y = [rb.random() for _ in range(NOUT)]
    FB = rb.sample(range(N), NFB)
    YES, NO = FB[:32], FB[32:]
    KU = [rb.random() for _ in range(N)]
    fbset = set(FB)
    taken = set(FB)
    driver = []
    for v in range(NOUT):
        best = min((i for i in range(N) if i not in taken), key=lambda i: ((1 - P[i][0]) ** 2 + (P[i][1] - OUT_Y[v]) ** 2, i))
        driver.append(best); taken.add(best)
    outset = set(driver)
    members = {}
    for i in range(N):
        if i in fbset:
            members[i] = []
            continue
        xi, yi = P[i]
        mem = [j for j in range(N) if j != i and (P[j][0] - xi) ** 2 + (P[j][1] - yi) ** 2 <= R2]
        mem += [LINE0 + v for v in range(NIN) if xi ** 2 + (yi - IN_Y[v]) ** 2 <= R2]
        members[i] = mem
    src = {}
    for i in range(N):
        src[i] = [] if i in fbset else rb.sample(members[i], min(START_N, len(members[i])))
    inhib = [(KU[i] < 0.25) and i not in outset and i not in fbset for i in range(N)]
    kind = np.full(N, INNER, dtype=np.int8)
    for i in range(N):
        if i in fbset:
            kind[i] = FBK
        elif i in outset:
            kind[i] = OUTE
        elif any(mm >= LINE0 for mm in members[i]):
            kind[i] = ENC
    sig = np.ones(NZ, dtype=np.float64)                 # sigma of every sender: cells, input lines, feedback lines
    for i in range(N):
        if inhib[i]:
            sig[i] = -1.0
    for i in YES:                                       # sentence 89: yes-cells inhibitory, no-cells excitatory
        sig[i] = -1.0
    digests = dict(
        coordinates=hashlib.sha256(json.dumps(P).encode()).hexdigest()[:16],
        sources=hashlib.sha256(json.dumps([sorted(src[i]) for i in range(N)]).encode()).hexdigest()[:16],
        fb_drivers_kinds=hashlib.sha256(json.dumps([FB, driver, inhib]).encode()).hexdigest()[:16])
    return dict(P=P, IN_Y=IN_Y, OUT_Y=OUT_Y, FB=FB, YES=YES, NO=NO, driver=driver, members=members, src=src, inhib=inhib, kind=kind,
                sig=sig, digests=digests, fbset=fbset, outset=outset)


# ================================================================================================ the tissue
class Tissue:
    """One tissue (section 1). rule: 'first', 'coincidence' or 'fixed'. f11_written: formula 11 as written (512-byte tissue only)."""

    def __init__(self, seed, rule, f11_written=False, B=None, by_use=False, opts=("a", "b", "c")):
        assert rule in ("first", "coincidence", "fixed")
        assert not (f11_written and rule != "first")
        assert not (by_use and (rule != "first" or f11_written))
        self.seed, self.rule, self.f11w, self.by_use = seed, rule, f11_written, by_use
        B = B if B is not None else birth(seed)
        self.B = B
        self.kind = B["kind"]; self.sig = B["sig"]
        self.isfb = self.kind == FBK
        self.isinner = self.kind == INNER
        self.members = B["members"]
        self.driver = np.array(B["driver"], dtype=np.int64)
        snd, rcv, w = [], [], []
        for i in range(N):
            for j in B["src"][i]:
                snd.append(j); rcv.append(i); w.append(START_W)
        for i in B["YES"]:
            snd.append(YES_LINE); rcv.append(i); w.append(1.0)
        for i in B["NO"]:
            snd.append(NO_LINE); rcv.append(i); w.append(1.0)
        n = len(snd)
        cap = max(4096, 2 * n)
        self.cap, self.n = cap, n
        self.snd = np.zeros(cap, np.int64); self.snd[:n] = snd
        self.rcv = np.zeros(cap, np.int64); self.rcv[:n] = rcv
        self.w = np.zeros(cap, np.float64); self.w[:n] = w
        self.e = np.zeros(cap, np.float64)
        self.ub = np.zeros(cap, np.float64) if f11_written else None
        if by_use:                                       # the fifth tissue: per connection, the trace's running average, the use, and
            self.eb = np.zeros(cap, np.float64)          # the number of consecutive steps the use has stayed below its level
            self.cu = np.zeros(cap, np.float64)
            self.nb = np.zeros(cap, np.int64)
        self.k0 = np.zeros(cap, np.int64)
        self.alive = np.zeros(cap, bool); self.alive[:n] = True
        self.dead = 0
        self.srcset = [set(B["src"][i]) for i in range(N)]
        # cell state (sentence 123: all 0 at birth, except the sliding threshold, which starts at a*)
        self.v = np.zeros(N); self.x_prev = np.zeros(N); self.p = np.zeros(N); self.th = np.zeros(N); self.abar = np.zeros(N)
        self.vt = np.full(N, ASTAR) if rule == "coincidence" else None
        self.klast = np.zeros(N, np.int64)
        self.k = 0                                       # the next step
        self.rl = Random(seed + 500000)                  # the life generator (formula 12)
        self.z = np.zeros(NZ)
        self.given_up = 0; self.made = 0
        self._dirty = True
        self.x = np.zeros(N)
        self.audit = None
        self.withdrawn_by_use = 0
        self.opts = frozenset(opts)                      # decision 66: the changes in use
        assert self.opts <= {"a", "b", "c", "maxsort"}, self.opts
        self._set_nocand()
        self.t_use = [0.0, 0.0]                          # CPU-seconds of the fifth tissue's lines 360-363 and 382-395

    def _set_nocand(self):                               # change (a): all of the neighbourhood's members are already sources
        self.nocand = np.array([(not self.isfb[i]) and len(self.srcset[i]) >= len(self.members[i]) for i in range(N)], bool)

    # ---------------------------------------------------------------------------------------- helpers
    def _index(self):
        n = self.n
        a = self.alive[:n]
        self.ci = np.nonzero(a & (self.snd[:n] < N))[0]          # alive connections from a cell (into a non-feedback cell)
        if self.rule == "coincidence":
            inner = self.isinner[self.rcv[self.ci]]
            self.ci_f7 = self.ci[~inner]
            self.ci_ce = self.ci[inner & (self.sig[self.snd[self.ci]] > 0)]
            self.ci_ci = self.ci[inner & (self.sig[self.snd[self.ci]] < 0)]
        self.not_fbline = a & (self.snd[:n] < YES_LINE)
        if self.by_use:
            sn = self.snd[:n]
            self.ci_use = np.nonzero(a & (sn < N) & ~self.isfb[np.minimum(sn, N - 1)])[0]
        self._dirty = False

    def _grow(self):
        cap = self.cap * 2
        for name in ("snd", "rcv", "w", "e", "k0", "alive") + (("ub",) if self.f11w else ()) + (("eb", "cu", "nb") if self.by_use else ()):
            old = getattr(self, name)
            new = np.zeros(cap, old.dtype); new[:self.cap] = old[:self.cap]
            setattr(self, name, new)
        self.cap = cap

    def _compact(self):
        n = self.n
        keep = np.nonzero(self.alive[:n])[0]
        m = len(keep)
        for name in ("snd", "rcv", "w", "e", "k0") + (("ub",) if self.f11w else ()) + (("eb", "cu", "nb") if self.by_use else ()):
            arr = getattr(self, name)
            arr[:m] = arr[keep]; arr[m:n] = 0
        self.alive[:m] = True; self.alive[m:n] = False
        self.n, self.dead = m, 0
        self._dirty = True

    # ---------------------------------------------------------------------------------------- one step (section 5.4)
    def step(self, byte, hook=None):
        k = self.k
        phi = k % S
        n = self.n
        if self._dirty:
            self._index()
        z = self.z
        # 1. input lines; both feedback lines 0 at every step of step 1
        if phi == 0:
            z[LINE0:LINE0 + NIN] = 0.0
            z[LINE0 + int(byte)] = 1.0
        z[YES_LINE] = 0.0; z[NO_LINE] = 0.0
        z[:N] = self.x_prev
        snd, rcv = self.snd[:n], self.rcv[:n]
        w, e = self.w[:n], self.e[:n]
        zs = z[snd]
        # 2. arrival (formula 1)
        if "c" in self.opts:                             # change (c): only the senders whose value is not zero
            nzc = np.nonzero(zs)[0]
            A = np.bincount(rcv[nzc], weights=self.sig[snd[nzc]] * w[nzc] * zs[nzc], minlength=N)
        else:
            A = np.bincount(rcv, weights=self.sig[snd] * w * zs, minlength=N)
        # 3. mismatch (formula 5)
        m = f5_m(A, self.p)
        # 4. state and output (formulas 2, 3)
        v = f2_v(self.v, A)
        x = f3_x(v, self.th)
        xp = self.x_prev
        # 5. strength changes (formula 7 or the coincidence rule; formula 8), from traces held before the step and x(k - 1)
        ob = "b" in self.opts                            # change (b): only where the receiving cell's x_prev is not zero
        if self.rule == "first":
            ci = self.ci
            r = rcv[ci]
            if ob:
                q = np.nonzero(xp[r])[0]; ci = ci[q]; r = r[q]
            w[ci] = f8_bound(w[ci] + f7_dw(self.sig[snd[ci]], m[r], xp[r], e[ci]))
        elif self.rule == "coincidence":
            ci = self.ci_f7
            r = rcv[ci]
            if ob:
                q = np.nonzero(xp[r])[0]; ci = ci[q]; r = r[q]
            w[ci] = f8_bound(w[ci] + f7_dw(self.sig[snd[ci]], m[r], xp[r], e[ci]))
            ci = self.ci_ce
            r = rcv[ci]
            if ob:
                q = np.nonzero(xp[r])[0]; ci = ci[q]; r = r[q]
            w[ci] = f8_bound(w[ci] + coin_exc_dw(e[ci], xp[r], self.vt[r]))
            ci = self.ci_ci
            r = rcv[ci]
            w[ci] = f8_bound(w[ci] + coin_inh_dw(e[ci], xp[r]))
        # 6. traces, prediction, threshold (not at feedback cells, E4), activity average; ū (512-byte tissue); ϑ (coincidence)
        if self.by_use:
            tu0 = time.process_time()
            r_ = rcv
            self.cu[:n] = use_update(self.cu[:n], e, self.eb[:n], x[r_], self.abar[r_])
            self.eb[:n] = (1 - 1 / PS) * self.eb[:n] + (1 / PS) * e
            self.t_use[0] += time.process_time() - tu0
        self.e[:n] = f6_trace(e, zs)
        if self.f11w:
            self.ub[:n] = ubar_update(self.ub[:n], zs)
        self.p = f4_p(self.p, A)
        th = f9_theta(self.th, x)
        th[self.isfb] = 0.0
        self.th = th
        self.abar = f10_abar(self.abar, x)
        if self.rule == "coincidence":
            self.vt = vt_update(self.vt, x)
        self.v = v
        # 7. giving up (formula 11), then making (formula 12)
        if self.rule != "fixed":
            old = (k - self.k0[:n]) >= PS
            if self.f11w:
                gone = self.not_fbline & old & ((self.w[:n] < GIVE_UP) | (self.ub[:n] < GIVE_UP))
            else:
                gone = self.not_fbline & old & (self.w[:n] < GIVE_UP)
            if self.by_use:
                tu0 = time.process_time()
                ci = self.ci_use                                 # alive connections from ordinary cells (not feedback cells)
                mag = np.abs(self.cu[ci])
                mx = np.zeros(N)
                if "maxsort" in self.opts:                       # the same maxima, by a stable sort of the receivers
                    rr = self.rcv[ci]
                    if len(rr):
                        o = np.argsort(rr, kind="stable")
                        rs = rr[o]
                        starts = np.concatenate(([0], np.nonzero(rs[1:] != rs[:-1])[0] + 1))
                        mx[rs[starts]] = np.maximum.reduceat(mag[o], starts)
                        np.maximum(mx, 0.0, out=mx)
                else:
                    np.maximum.at(mx, self.rcv[ci], mag)
                low = mag < GIVE_UP * mx[self.rcv[ci]]
                nbv = self.nb[ci]
                nbv = np.where(low, nbv + 1, 0)
                self.nb[ci] = nbv
                byuse = np.zeros(n, bool)
                byuse[ci[nbv >= PS]] = True
                self.gone_by_use = int((byuse & ~gone).sum())
                self.withdrawn_by_use += self.gone_by_use
                gone = gone | byuse
                self.t_use[1] += time.process_time() - tu0
            if gone.any():
                idx = np.nonzero(gone)[0]
                if self.audit is not None:                   # check 6 only: what is given up (no effect on the tissue)
                    self.audit.append((k, self.snd[idx].copy(), self.w[idx].copy(), self.rcv[idx].copy()))
                for c in idx:
                    self.srcset[int(self.rcv[c])].discard(int(self.snd[c]))
                self.nocand[self.rcv[idx]] = False               # change (a): a cell that gives up a connection has a candidate again
                self.alive[idx] = False
                self.w[idx] = 0.0
                self.given_up += len(idx); self.dead += len(idx)
                self._dirty = True
            if "a" in self.opts:                                  # change (a): cells with no candidate are left out (no draw for them)
                elig = np.nonzero((~self.isfb) & (~self.nocand) & (self.th == 0.0) & (self.abar < ASTAR) & ((k - self.klast) >= PS))[0]
            else:
                elig = np.nonzero((~self.isfb) & (self.th == 0.0) & (self.abar < ASTAR) & ((k - self.klast) >= PS))[0]
            for i in elig:
                i = int(i)
                ss = self.srcset[i]
                cand = [mm for mm in self.members[i] if mm not in ss]
                if not cand:
                    continue
                j = self.rl.choice(cand)
                if self.n >= self.cap:
                    self._grow()
                c = self.n
                self.snd[c] = j; self.rcv[c] = i; self.w[c] = START_W; self.e[c] = 0.0; self.k0[c] = k; self.alive[c] = True
                if self.f11w:
                    self.ub[c] = 0.0
                if self.by_use:
                    self.eb[c] = 0.0; self.cu[c] = 0.0; self.nb[c] = 0
                self.n += 1
                ss.add(j)
                if len(ss) >= len(self.members[i]):
                    self.nocand[i] = True                        # change (a): this making filled the cell
                self.klast[i] = k
                self.made += 1
                self._dirty = True
            if self.dead > self.n // 4 and self.dead > 1024:
                self._compact()
        self.x = x
        self.x_prev = x
        if hook is not None:
            hook(self, k, A, m, v, x)
        self.k = k + 1
        # 8. emission at the reading step
        if phi == S - 1:
            xo = x[self.driver]
            b = int(np.argmax(xo))
            return b if xo[b] > EMIT else -1
        return None

    def run_byte(self, byte, hook=None):
        em = None
        for _ in range(S):
            r = self.step(byte, hook)
            if r is not None:
                em = r
        return em

    # ---------------------------------------------------------------------------------------- readings
    def live(self):
        n = self.n
        a = np.nonzero(self.alive[:n])[0]
        return self.snd[a], self.rcv[a], self.w[a], self.k0[a]

    def distances(self, present_values):
        """Distance of every cell from the input lines of the byte values present (section 1), along the present connections."""
        snd, rcv, _, _ = self.live()
        lines = set(LINE0 + int(v) for v in present_values)
        dist = np.zeros(N, np.int16)                     # 0: not reached
        isl = np.array([s in lines for s in snd.tolist()]) if len(snd) else np.zeros(0, bool)
        frontier = np.zeros(N, bool)
        frontier[rcv[isl]] = True
        frontier &= ~self.isfb
        d = 1
        dist[frontier] = 1
        cellc = snd < N
        cs, cr = snd[cellc], rcv[cellc]
        while frontier.any():
            nxt = np.zeros(N, bool)
            nxt[cr[frontier[cs]]] = True
            nxt &= (dist == 0)
            d += 1
            dist[nxt] = d
            frontier = nxt
        return dist

    def counts(self, used_values):
        """The records of section 7.1, item 4."""
        snd, rcv, w, k0 = self.live()
        used = set(int(v) for v in used_values)
        sk = np.empty(len(snd), np.int8)
        isline = (snd >= LINE0) & (snd < YES_LINE)
        sk[isline] = [SK_LINE_USED if int(s) - LINE0 in used else SK_LINE_UNUSED for s in snd[isline]]
        isfbl = snd >= YES_LINE
        sk[isfbl] = SK_FBLINE
        isc = snd < N
        cs = snd[isc]
        sk_c = np.where(self.isfb[cs], SK_FBC, np.where(self.sig[cs] < 0, SK_INH, SK_EXC))
        sk[isc] = sk_c
        rk = self.kind[rcv]
        table = {}
        for r_, rn in RK_NAMES.items():
            for s_, sn in SK_NAMES.items():
                msk = (rk == r_) & (sk == s_)
                table[f"{rn} <- {sn}"] = dict(n=int(msk.sum()), mean_w=float(w[msk].mean()) if msk.any() else None)
        ordinary = sk != SK_FBLINE
        per = np.bincount(rcv[ordinary], minlength=N)
        nonfb = ~self.isfb
        exhausted = sum(1 for i in range(N) if nonfb[i] and len(self.srcset[i]) >= len(self.members[i]))
        line_vals = set(int(s) - LINE0 for s in snd[isline])
        yes = set(self.B["YES"]); no = set(self.B["NO"])
        outc = set(self.B["driver"])
        out_yes = sum(1 for i in outc if self.srcset[i] & yes)
        out_no = sum(1 for i in outc if self.srcset[i] & no)
        exc_inner = (rk == INNER) & (sk == SK_EXC)
        return dict(step=self.k - 1, connections=int(ordinary.sum()), by_kind=table,
                    starting_remaining=int(((k0 == 0) & ordinary).sum()),
                    given_up_total=self.given_up, made_total=self.made, withdrawn_by_use_total=self.withdrawn_by_use,
                    per_cell_mean=float(per.mean()), per_cell_max=int(per.max()), per_cell_min_nonfb=int(per[nonfb].min()),
                    exhausted=exhausted, line_values=len(line_vals), line_values_in_use=len(line_vals & used),
                    from_feedback_cells=dict(n=int((sk == SK_FBC).sum()), w_min=float(w[sk == SK_FBC].min()) if (sk == SK_FBC).any() else None,
                                             w_max=float(w[sk == SK_FBC].max()) if (sk == SK_FBC).any() else None),
                    output_end_taking_yes=out_yes, output_end_taking_no=out_no,
                    exc_into_inner_mean_w=float(w[exc_inner].mean()) if exc_inner.any() else None)

    # ---------------------------------------------------------------------------------------- state
    def state_arrays(self):
        n = self.n
        d = dict(snd=self.snd[:n].copy(), rcv=self.rcv[:n].copy(), w=self.w[:n].copy(), e=self.e[:n].copy(), k0=self.k0[:n].copy(),
                 alive=self.alive[:n].copy(), v=self.v, x_prev=self.x_prev, p=self.p, th=self.th, abar=self.abar, klast=self.klast,
                 meta=np.array([self.k, self.dead, self.given_up, self.made, self.cap], np.int64))
        if self.f11w:
            d["ub"] = self.ub[:n].copy()
        if self.by_use:
            d["eb"] = self.eb[:n].copy(); d["cu"] = self.cu[:n].copy(); d["nb"] = self.nb[:n].copy()
            d["wbu"] = np.array([self.withdrawn_by_use], np.int64)
        if self.vt is not None:
            d["vt"] = self.vt
        return d

    def load_arrays(self, d):
        k, dead, gu, made, cap = (int(q) for q in d["meta"])
        n = len(d["snd"])
        self.cap = cap
        for name in ("snd", "rcv", "w", "e", "k0", "alive") + (("ub",) if self.f11w else ()) + (("eb", "cu", "nb") if self.by_use else ()):
            arr = np.zeros(cap, d[name].dtype); arr[:n] = d[name]
            setattr(self, name, arr)
        self.n, self.k, self.dead, self.given_up, self.made = n, k, dead, gu, made
        for name in ("v", "x_prev", "p", "th", "abar", "klast"):
            setattr(self, name, d[name].copy())
        self.x = self.x_prev.copy()
        if self.by_use:
            self.withdrawn_by_use = int(d["wbu"][0])
        if self.vt is not None:
            self.vt = d["vt"].copy()
        self.srcset = [set() for _ in range(N)]
        for s_, r_, a_ in zip(self.snd[:n].tolist(), self.rcv[:n].tolist(), self.alive[:n].tolist()):
            if a_ and s_ < YES_LINE:
                self.srcset[r_].add(s_)
        self._set_nocand()
        self._dirty = True
        self.z = np.zeros(NZ)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine(), cores=os.cpu_count())


# ================================================================================================ runs (stage A; stage B is not released)
STAGE = {"A": dict(windows={"A1": 2048, "A2": 12288}, wlen=4096, stretch=1024, ckpt=4096, length=16384),
         "B": dict(windows={"W0": 0, "W1": 24576, "W2": 122880, "W3": 253952}, wlen=8192, stretch=None, ckpt=16384, length=262144)}
BLOCK = 128


class Run:
    """A run of one tissue on its stream, with the records of section 7.1 (stage A: and the stretch readings of section 11.2)."""

    def __init__(self, rule, seed, nbytes, stage, f11w, out, by_use=False):
        self.rule, self.seed, self.nbytes, self.stage, self.f11w, self.out = rule, seed, nbytes, stage, f11w, Path(out)
        self.cfg = STAGE[stage]
        self.st, _ = stream(seed, nbytes)
        self.used = sorted(set(self.st.tolist()))
        self.T = Tissue(seed, rule, f11w, by_use=by_use)
        self.emis = np.full(nbytes, -1, np.int16)
        self.block_dist = {}                                 # block start position -> distances (first step of the block)
        self.win = {}                                        # window name -> float16 records [wlen, N]
        self.win_theta = {}                                  # window name -> sum of θ at reading steps
        self.stretch_buf = None
        self.stretch = []                                    # stage A stretch readings
        self.counts = []
        self.cpu = []                                        # per stretch (stage A) or per 1,024 bytes
        self.resumes = []
        self.pos = 0
        self.cpu_total = 0.0

    # which counts (section 11.2, item 1) after operation 7 of a step
    def count_steps(self):
        if self.f11w:
            return {127, 129} | {32 * b + 31 for b in (128, 256, 384, 511)}
        return {127, 129, 1024} | {32 * b + 31 for b in range(127, self.nbytes, 128)}

    def state_path(self):
        return self.out / "saved_state.npz"

    def save(self):
        d = self.T.state_arrays()
        d["emis"] = self.emis
        d["rl_state"] = np.frombuffer(pickle.dumps(self.T.rl.getstate()), np.uint8)
        d["pos"] = np.array([self.pos])
        d["stream_sha"] = np.frombuffer(hashlib.sha256(self.st[:self.pos].tobytes()).digest(), np.uint8)
        for wn, arr in self.win.items():
            d["win_" + wn] = arr
            d["winth_" + wn] = self.win_theta[wn]
        if self.stretch_buf is not None:
            d["stretch_buf"] = self.stretch_buf
        d["block_keys"] = np.array(sorted(self.block_dist), np.int64)
        d["block_vals"] = np.array([self.block_dist[b] for b in sorted(self.block_dist)], np.int16).reshape(-1, N)
        extra = dict(stretch=self.stretch, counts=self.counts, cpu=self.cpu, resumes=self.resumes, cpu_total=self.cpu_total)
        d["extra"] = np.frombuffer(json.dumps(extra).encode(), np.uint8)
        tmp = self.out / "saved_state.tmp.npz"
        np.savez(tmp, **d)
        os.replace(tmp, self.state_path())

    def load(self):
        d = np.load(self.state_path(), allow_pickle=False)
        self.T.load_arrays(d)
        self.T.rl.setstate(pickle.loads(d["rl_state"].tobytes()))
        self.pos = int(d["pos"][0])
        assert hashlib.sha256(self.st[:self.pos].tobytes()).digest() == d["stream_sha"].tobytes(), "the stream read so far differs"
        self.emis = d["emis"].copy()
        for key in d.files:
            if key.startswith("win_"):
                self.win[key[4:]] = d[key].copy()
            if key.startswith("winth_"):
                self.win_theta[key[6:]] = d[key].copy()
        self.stretch_buf = d["stretch_buf"].copy() if "stretch_buf" in d.files else None
        for b, v in zip(d["block_keys"].tolist(), d["block_vals"]):
            self.block_dist[b] = v.copy()
        extra = json.loads(d["extra"].tobytes().decode())
        self.stretch, self.counts, self.cpu, self.resumes, self.cpu_total = (extra["stretch"], extra["counts"], extra["cpu"],
                                                                             extra["resumes"], extra["cpu_total"])

    def present_in(self, lo, hi):
        return sorted(set(self.st[lo:hi].tolist()))

    def run(self, ckpt, die_after=None, resume=False, log=None):
        T, cfg = self.T, self.cfg
        if resume and self.state_path().exists():
            self.load()
            self.resumes.append(dict(byte=self.pos, wall_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        cst = self.count_steps()
        recording = not self.f11w
        stretch = cfg["stretch"] if recording else None
        c0 = time.process_time()
        cstretch = time.process_time()
        while self.pos < self.nbytes:
            t = self.pos
            byte = int(self.st[t])
            if recording and t % BLOCK == 0:
                blo = (t // BLOCK) * BLOCK
                self.block_dist[t] = T.distances(self.present_in(self._dist_lo(t), self._dist_hi(t)))
            em = None
            for _ in range(S):
                k = T.k
                r = T.step(byte)
                if r is not None:
                    em = r
                if k in cst:
                    rec = T.counts(self.used); rec["at"] = f"after operation 7 of step {k}"
                    self.counts.append(rec)
            self.emis[t] = em
            if recording:
                x = T.x
                rec = x.astype(np.float16)
                rec[x < REC_FLOOR] = 0
                for wn, w0 in cfg["windows"].items():
                    if w0 <= t < w0 + cfg["wlen"]:
                        if wn not in self.win:
                            self.win[wn] = np.zeros((cfg["wlen"], N), np.float16)
                            self.win_theta[wn] = np.zeros(N)
                        self.win[wn][t - w0] = rec
                        self.win_theta[wn] += T.th
                if stretch:
                    if t % stretch == 0:
                        self.stretch_buf = np.zeros((stretch, N), np.float16)
                    self.stretch_buf[t % stretch] = rec
            self.pos = t + 1
            if recording and stretch and self.pos % stretch == 0:
                now = time.process_time()
                lo = self.pos - stretch
                self.stretch.append(self.read_stretch(lo, self.pos))
                self.cpu.append(dict(bytes=[lo, self.pos - 1], cpu_s=now - cstretch, per_byte=(now - cstretch) / stretch,
                                     mean_connections=T.counts(self.used)["per_cell_mean"],
                                     emitted_share=float((self.emis[lo:self.pos] >= 0).mean())))
                cstretch = now
                if log:
                    print(f"byte {self.pos} cpu/byte {self.cpu[-1]['per_byte']:.4f} conns/cell {self.cpu[-1]['mean_connections']:.2f} "
                          + " ".join(f"d{d}: {self.stretch[-1]['d'][str(d)]['n']} {self.stretch[-1]['d'][str(d)]['informative_share']}"
                                     for d in (1, 2, 4)), flush=True)
            for wn, w0 in cfg["windows"].items():
                if recording and self.pos == w0 + cfg["wlen"]:
                    self.save_window(wn, w0)
            if ckpt and self.pos % ckpt == 0 and self.pos < self.nbytes:
                self.cpu_total += time.process_time() - c0; c0 = time.process_time()
                self.save()
            if die_after is not None and self.pos == die_after + 1:
                os._exit(9)
        self.cpu_total += time.process_time() - c0

    def _dist_lo(self, t):
        cfg = self.cfg
        for wn, w0 in cfg["windows"].items():
            if w0 <= t < w0 + cfg["wlen"]:
                return w0
        s = cfg["stretch"] or cfg["wlen"]
        return (t // s) * s

    def _dist_hi(self, t):
        cfg = self.cfg
        for wn, w0 in cfg["windows"].items():
            if w0 <= t < w0 + cfg["wlen"]:
                return w0 + cfg["wlen"]
        s = cfg["stretch"] or cfg["wlen"]
        return (t // s) * s + s

    def mode_distance(self, lo, hi):
        """The distance d of every cell over [lo, hi) (section 1): the most frequent of its block distances, ties to the smaller."""
        blocks = [self.block_dist[b] for b in range(lo, hi, BLOCK)]
        M = np.stack(blocks)                              # [nblocks, N], 0 = not reached
        out = np.zeros(N, np.int16); share = np.zeros(N)
        for i in range(N):
            vals, cnt = np.unique(M[:, i], return_counts=True)
            reached = vals > 0
            if not reached.any():
                continue
            vals, cnt = vals[reached], cnt[reached]
            j = int(np.argmax(cnt))                       # np.unique sorts, so the first maximum is the smaller distance
            out[i] = vals[j]; share[i] = cnt[j] / len(blocks)
        return out, share

    def read_stretch(self, lo, hi):
        """Section 11.2, item 2: for a stretch, cells at d = 1, 2, 4 by the mode over its blocks; shares neither silent nor constant."""
        d, _ = self.mode_distance(lo, hi)
        R = self.stretch_buf.astype(np.float64)
        npos = R.shape[0]
        above = (R > REC_FLOOR).sum(0)
        silent = above < npos / 128
        sd = R.std(0)
        const = (~silent) & (sd < REC_FLOOR)
        out = dict(bytes=[lo, hi - 1], d={})
        for dd in (1, 2, 4):
            cells = d == dd
            nn = int(cells.sum())
            inf = cells & ~silent & ~const
            out["d"][str(dd)] = dict(n=nn, informative=int(inf.sum()), silent=int((cells & silent).sum()), constant=int((cells & const).sum()),
                                     informative_share=round(float(inf.sum() / nn), 4) if nn else None,
                                     mean_output=float(R[:, cells].mean()) if nn else None, sd_output=float(R[:, cells].std()) if nn else None)
        return out

    def save_window(self, wn, w0):
        wl = self.cfg["wlen"]
        d, share = self.mode_distance(w0, w0 + wl)
        R = self.win[wn]
        np.savez_compressed(self.out / f"{wn}.npz", x=R, positions=np.arange(w0, w0 + wl), bytes=self.st[w0:w0 + wl],
                            distance=d, distance_share=share,
                            block_distances=np.stack([self.block_dist[b] for b in range(w0, w0 + wl, BLOCK)]),
                            theta_mean=self.win_theta[wn] / wl)

    def summary(self):
        out = {}
        for wn, R in self.win.items():
            Rf = R.astype(np.float64)
            out[wn] = dict(above=(Rf > REC_FLOOR).sum(0), mean=Rf.mean(0), sd=Rf.std(0), theta_mean=self.win_theta[wn] / Rf.shape[0])
        return out


def git_head():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def cmd_run(a, argv):
    out = Path(a.out) if a.out else HERE / "runs" / f"{a.stage}_{a.rule}{'_f11written' if a.formula11_as_written else ''}{'_withdrawbyuse' if a.withdraw_by_use else ''}_{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    nbytes = a.bytes or STAGE[a.stage]["length"]
    R = Run(a.rule, a.seed, nbytes, a.stage, a.formula11_as_written, out, by_use=a.withdraw_by_use)
    c0, w0 = time.process_time(), time.time()
    ckpt = STAGE[a.stage]["ckpt"] if a.ckpt_bytes is None else a.ckpt_bytes
    R.run(ckpt, a.die_after_byte, a.resume, log=True)
    T = R.T
    np.save(out / "emissions.npy", R.emis)
    np.savez(out / "final_state.npz", **T.state_arrays(), rl_state=np.frombuffer(pickle.dumps(T.rl.getstate()), np.uint8),
             pos=np.array([R.pos]))
    sm = R.summary()
    if sm:
        np.savez(out / "summary.npz", **{f"{wn}_{k}": v for wn, d in sm.items() for k, v in d.items()})
    rec = dict(experiment="E63", brief=BRIEF, brief_ref=REVIEWER_REF, command=" ".join(["python3", "tissue_step1.py"] + argv),
               program_sha256=sha_file(__file__), read_sha256=sha_file(HERE / "step1_read.py") if (HERE / "step1_read.py").exists() else None,
               git_head=git_head(), versions=versions(), stage=a.stage, tissue=a.rule, seed=a.seed,
               formula11="as written" if a.formula11_as_written else ("with E10 and withdrawal by use (decision 62)" if a.withdraw_by_use else "with E10"),
               opening=None,
               stream=dict(sha256=hashlib.sha256(R.st.tobytes()).hexdigest(), length=nbytes, used_values=len(R.used)),
               birth=T.B["digests"], counts=R.counts, stretches=R.stretch, cpu_by_stretch=R.cpu, cpu_total_s=R.cpu_total,
               wall_s=time.time() - w0, beside=a.beside, resumes=R.resumes,
               emissions=dict(share=float((R.emis >= 0).mean()), commonest=[[int(v), int(c)] for v, c in
                                                                         sorted(zip(*np.unique(R.emis[R.emis >= 0], return_counts=True)),
                                                                                key=lambda q: -q[1])[:5]] if (R.emis >= 0).any() else []),
               windows=sorted(R.win))
    (out / "receipt.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if R.state_path().exists():
        R.state_path().unlink()
    print("complete", out, flush=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "check"))
    ap.add_argument("pos", nargs="+")
    ap.add_argument("--bytes", type=int); ap.add_argument("--stage", default="A", choices=("A", "B"))
    ap.add_argument("--formula11-as-written", action="store_true"); ap.add_argument("--out")
    ap.add_argument("--withdraw-by-use", action="store_true")      # the fifth tissue of stage A (decision 62)
    ap.add_argument("--ckpt-bytes", type=int); ap.add_argument("--resume", action="store_true"); ap.add_argument("--die-after-byte", type=int)
    ap.add_argument("--opening", type=int); ap.add_argument("--beside", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        assert a.stage == "A", "stage B is not released"
        assert a.opening is None, "no opening is released"
        a.rule, a.seed = a.pos[0], int(a.pos[1])
        cmd_run(a, argv)
    else:
        import step1_checks
        step1_checks.run_check(int(a.pos[0]), a)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1:])
