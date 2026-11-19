"""Check of the two parameters (T) a* and (W) withdrawal by use: the core against a dense reference.

The reference is E63's Tissue.step (TISSUE_STEP1/tissue_step1.py, imported, not edited), copied here into a subclass with (W) added in
the reviewer's words; (T) is E63's own constant ASTAR set to the value under test (every use in E63's program reads the module constant
at call time). Every switch of the new program is off. Compares connections made and given up, emissions, and the state at the end.

    python3 tw_check.py <first|coincidence> <astar> <bytes> [--use] [--exempt-fb]
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import evt
from evt import ts

HERE = Path(__file__).resolve().parent


class RefTW(ts.Tissue):
    def __init__(self, seed, rule, use, exempt_fb, floor=0.0):
        super().__init__(seed, rule)
        self.use, self.exempt_fb, self.floor = use, exempt_fb, floor
        self.cu = np.zeros(self.cap); self.nb = np.zeros(self.cap, np.int64)
        self.events_made, self.events_gone = [], []
        self.wbu = 0

    def _grow(self):
        super()._grow()
        for name in ("cu", "nb"):
            old = getattr(self, name); new = np.zeros(self.cap, old.dtype); new[:len(old)] = old; setattr(self, name, new)

    def _compact(self):
        n = self.n
        keep = np.nonzero(self.alive[:n])[0]; m = len(keep)
        for name in ("cu", "nb"):
            arr = getattr(self, name); arr[:m] = arr[keep]; arr[m:n] = 0
        super()._compact()

    def step(self, byte, hook=None):
        k = self.k; phi = k % ts.S; n = self.n
        if self._dirty:
            self._index()
        z = self.z
        if phi == 0:
            z[ts.LINE0:ts.LINE0 + ts.NIN] = 0.0; z[ts.LINE0 + int(byte)] = 1.0
        z[ts.YES_LINE] = 0.0; z[ts.NO_LINE] = 0.0
        z[:ts.N] = self.x_prev
        snd, rcv = self.snd[:n], self.rcv[:n]
        w, e = self.w[:n], self.e[:n]
        zs = z[snd]
        A = np.bincount(rcv, weights=self.sig[snd] * w * zs, minlength=ts.N)
        m = ts.f5_m(A, self.p)
        v = ts.f2_v(self.v, A)
        x = ts.f3_x(v, self.th)
        if self.floor:
            x = np.where(x < self.floor, 0.0, x)      # the output floor, as the new program applies it
        xp = self.x_prev
        if self.rule == "first":
            ci = self.ci; r = rcv[ci]
            w[ci] = ts.f8_bound(w[ci] + ts.f7_dw(self.sig[snd[ci]], m[r], xp[r], e[ci]))
        elif self.rule == "coincidence":
            ci = self.ci_f7; r = rcv[ci]
            w[ci] = ts.f8_bound(w[ci] + ts.f7_dw(self.sig[snd[ci]], m[r], xp[r], e[ci]))
            ci = self.ci_ce; r = rcv[ci]
            w[ci] = ts.f8_bound(w[ci] + ts.coin_exc_dw(e[ci], xp[r], self.vt[r]))
            ci = self.ci_ci; r = rcv[ci]
            w[ci] = ts.f8_bound(w[ci] + ts.coin_inh_dw(e[ci], xp[r]))
        # (W), in the reviewer's words: only where x(k - 1) of the receiving cell is not zero; e is the trace before the step
        if self.use:
            al = np.nonzero(self.alive[:n] & (snd < ts.N))[0]
            act = al[xp[rcv[al]] != 0.0]
            r = rcv[act]
            self.cu[act] = (1 - 1 / 128) * self.cu[act] + (1 / 128) * e[act] * xp[r]
            mx = np.zeros(ts.N); np.maximum.at(mx, r, self.cu[act])
            low = self.cu[act] < (1 / 128) * mx[r]
            self.nb[act] = np.where(low, self.nb[act] + 1, 0)
        self.e[:n] = ts.f6_trace(e, zs)
        self.p = ts.f4_p(self.p, A)
        th = ts.f9_theta(self.th, x); th[self.isfb] = 0.0; self.th = th
        self.abar = ts.f10_abar(self.abar, x)
        if self.rule == "coincidence":
            self.vt = ts.vt_update(self.vt, x)
        self.v = v
        old = (k - self.k0[:n]) >= ts.PS
        gone = self.not_fbline & old & (self.w[:n] < ts.GIVE_UP)
        if self.use:
            byuse = self.alive[:n] & (snd < ts.N) & old & (self.nb[:n] >= ts.PS)
            if self.exempt_fb:
                byuse &= ~self.isfb[np.minimum(snd, ts.N - 1)]
            self.wbu += int((byuse & ~gone).sum())
            gone = gone | byuse
        if gone.any():
            idx = np.nonzero(gone)[0]
            for c in idx:
                self.srcset[int(self.rcv[c])].discard(int(self.snd[c]))
                self.events_gone.append((k, int(self.snd[c]), int(self.rcv[c])))
            self.alive[idx] = False; self.w[idx] = 0.0
            self.given_up += len(idx); self.dead += len(idx); self._dirty = True
        elig = np.nonzero((~self.isfb) & (self.th == 0.0) & (self.abar < ts.ASTAR) & ((k - self.klast) >= ts.PS))[0]
        for i in elig:
            i = int(i); ss = self.srcset[i]
            cand = [mm for mm in self.members[i] if mm not in ss]
            if not cand:
                continue
            j = self.rl.choice(cand)
            if self.n >= self.cap:
                self._grow()
            c = self.n
            self.snd[c] = j; self.rcv[c] = i; self.w[c] = ts.START_W; self.e[c] = 0.0; self.k0[c] = k; self.alive[c] = True
            self.cu[c] = 0.0; self.nb[c] = 0
            self.n += 1; ss.add(j); self.klast[i] = k; self.made += 1; self._dirty = True
            self.events_made.append((k, j, i))
        if self.dead > self.n // 4 and self.dead > 1024:
            self._compact()
        self.x = x; self.x_prev = x
        self.k = k + 1
        if phi == ts.S - 1:
            xo = x[self.driver]; b = int(np.argmax(xo))
            return b if xo[b] > ts.EMIT else -1
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=("first", "coincidence"))
    ap.add_argument("astar", type=float)
    ap.add_argument("bytes", type=int)
    ap.add_argument("--use", action="store_true")
    ap.add_argument("--exempt-fb", action="store_true")
    ap.add_argument("--floor", type=float, default=0.0)
    a = ap.parse_args()
    st, _ = ts.stream(73129, a.bytes)
    ts.ASTAR = a.astar                                  # (T) in the reference: E63's constant, read at call time
    c0 = time.process_time()
    R = RefTW(73129, a.rule, a.use, a.exempt_fb, a.floor)
    em_r = np.array([R.run_byte(int(b)) for b in st], np.int64)
    cpu_r = time.process_time() - c0
    c1 = time.process_time()
    T = evt.EvTissue(73129, a.rule, astar=a.astar, withdraw_by_use=a.use, use_exempt_feedback=a.exempt_fb, floor=a.floor)
    em_n = T.run_bytes(st).astype(np.int64)
    cpu_n = time.process_time() - c1
    made, gone = T.events()
    rm = np.array(R.events_made, np.int64).reshape(-1, 3); rg = np.array(R.events_gone, np.int64).reshape(-1, 3)
    gone = gone[:, :3]

    def first_diff(x, y):
        n = min(len(x), len(y))
        bad = np.nonzero((x[:n] != y[:n]).any(1))[0]
        if len(bad):
            return dict(index=int(bad[0]), mine=x[bad[0]].tolist(), reference=y[bad[0]].tolist())
        return None if len(x) == len(y) else dict(index=n, lengths=[len(x), len(y)])
    snd, rcv, w, e, k0, ub = T.live_full()
    nn = R.n; al = np.nonzero(R.alive[:nn])[0]
    same_conn = len(snd) == len(al) and bool(np.array_equal(snd, R.snd[al]) and np.array_equal(rcv, R.rcv[al]) and np.array_equal(k0, R.k0[al]))
    cs = T.cells()
    diffs = dict(v=float(np.abs(cs["v"] - R.v).max()), p=float(np.abs(cs["p"] - R.p).max()), th=float(np.abs(cs["th"] - R.th).max()),
                 abar=float(np.abs(cs["abar"] - R.abar).max()))
    if same_conn:
        diffs["w"] = float(np.abs(w - R.w[al]).max()); diffs["e"] = float(np.abs(e - R.e[al]).max())
    if cs["vt"] is not None:
        diffs["vt"] = float(np.abs(cs["vt"] - R.vt).max())
    res = dict(rule=a.rule, astar=a.astar, bytes=a.bytes, use=a.use, exempt_fb=a.exempt_fb, floor=a.floor,
               made=[len(made), len(rm)], given_up=[len(gone), len(rg)], withdrawn_by_use=[T.withdrawn_by_use, R.wbu],
               first_difference_made=first_diff(made, rm), first_difference_given_up=first_diff(gone, rg),
               emissions_identical=bool(np.array_equal(em_n, em_r)), generator_identical=T.rng_state() == R.rl.getstate(),
               connections_identical=same_conn, largest_differences=diffs, cpu_reference_s=cpu_r, cpu_new_s=cpu_n,
               core_sha256=hashlib.sha256(evt.SRC.read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    res["pass"] = bool(res["first_difference_made"] is None and res["first_difference_given_up"] is None and res["emissions_identical"]
                       and res["generator_identical"] and same_conn and max(diffs.values()) <= 1e-9)
    name = f"tw_{a.rule}_a{a.astar:g}_{a.bytes}{'_use' if a.use else ''}{'_exfb' if a.exempt_fb else ''}{'_floor' if a.floor else ''}"
    (HERE / "qual" / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(name, json.dumps({k: res[k] for k in ("pass", "made", "given_up", "withdrawn_by_use", "first_difference_made",
                                                "first_difference_given_up", "largest_differences")}), flush=True)


if __name__ == "__main__":
    main()
