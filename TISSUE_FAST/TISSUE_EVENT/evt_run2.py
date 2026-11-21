"""Part (b) of the task: a tissue of stage A computed by the new program, with E63's records and readings (tissue_step1.Run and
step1_read.read_run, unchanged).

    python3 evt_run.py <first|coincidence|fixed> <name> [--bytes N] [--floor F] [--tol T] [--trace-sender] [--strength-byte]

Writes TISSUE_EVENT/runs/<name>/: receipt.json (as E63's, with the switches, the counters of the core and the cost), emissions.npy,
summary.npz and the windows A1.npz and A2.npz when the run reaches their end; the scores are then read by step1_read.read_run. After
the recorded run the same tissue is run again by the core alone, without any recording, for the cost of the computation itself.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import evt
from evt import ts

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TISSUE_STEP1"))
import step1_read  # noqa: E402


class EvRun(ts.Run):
    def __init__(self, rule, seed, nbytes, out, sw):
        self.rule, self.seed, self.nbytes, self.stage, self.f11w, self.out = rule, seed, nbytes, "A", False, Path(out)
        self.cfg = ts.STAGE["A"]
        self.st, _ = ts.stream(seed, nbytes)
        self.used = sorted(set(self.st.tolist()))
        self.T = evt.EvTissue(seed, rule, **sw)
        self.emis = np.full(nbytes, -1, np.int16)
        self.block_dist, self.win, self.win_theta = {}, {}, {}
        self.stretch_buf, self.stretch, self.counts, self.cpu, self.resumes = None, [], [], [], []
        self.pos, self.cpu_total = 0, 0.0


# ---------------------------------------------------------------------------------------- readings computed with arrays
# The same quantities as tissue_step1.Tissue.counts and .distances, and evt.EvTissue.srcset, without Python loops over every
# connection. The live connections are read from the core once per step.
_live_cache = {}
def _live_once(self):
    k = self.k
    hit = _live_cache.get(id(self))
    if hit is not None and hit[0] == k and hit[1] == (self.made, self.given_up):
        return hit[2]
    r = evt.EvTissue.live(self)
    _live_cache[id(self)] = (k, (self.made, self.given_up), r)
    return r


def _distances_fast(self, present_values):
    N = ts.N
    snd, rcv, _, _ = _live_once(self)
    lines = np.array(sorted(ts.LINE0 + int(v) for v in present_values), np.int64)
    dist = np.zeros(N, np.int16)
    isl = np.isin(snd, lines) if len(snd) else np.zeros(0, bool)
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


class _SrcStats:
    """What tissue_step1.Tissue.counts reads from srcset: the number of distinct senders below the yes line into each cell, and
    whether a cell has a sender in a given set."""
    def __init__(self, snd, rcv):
        m = snd < ts.YES_LINE
        self.s, self.r = snd[m], rcv[m]
        key = np.sort(self.r * (ts.NZ + 1) + self.s)
        pairs = key[np.concatenate(([True], key[1:] != key[:-1]))] if len(key) else key
        self.size = np.bincount(pairs // (ts.NZ + 1), minlength=ts.N)
    def any_from(self, cells, senders):
        sel = np.isin(self.s, np.array(sorted(senders), np.int64))
        has = np.zeros(ts.N, bool); has[self.r[sel]] = True
        return int(sum(1 for i in cells if has[i]))


def _counts_fast(self, used_values):
    N, LINE0, YES_LINE = ts.N, ts.LINE0, ts.YES_LINE
    snd, rcv, w, k0 = _live_once(self)
    used = set(int(v) for v in used_values)
    sk = np.empty(len(snd), np.int8)
    isline = (snd >= LINE0) & (snd < YES_LINE)
    usedarr = np.array(sorted(LINE0 + u for u in used), np.int64)
    sk[isline] = np.where(np.isin(snd[isline], usedarr), ts.SK_LINE_USED, ts.SK_LINE_UNUSED)
    isfbl = snd >= YES_LINE
    sk[isfbl] = ts.SK_FBLINE
    isc = snd < N
    cs = snd[isc]
    sk[isc] = np.where(self.isfb[cs], ts.SK_FBC, np.where(self.sig[cs] < 0, ts.SK_INH, ts.SK_EXC))
    rk = self.kind[rcv]
    table = {}
    for r_, rn in ts.RK_NAMES.items():
        for s_, sn in ts.SK_NAMES.items():
            msk = (rk == r_) & (sk == s_)
            table[f"{rn} <- {sn}"] = dict(n=int(msk.sum()), mean_w=float(w[msk].mean()) if msk.any() else None)
    ordinary = sk != ts.SK_FBLINE
    per = np.bincount(rcv[ordinary], minlength=N)
    nonfb = ~self.isfb
    st = _SrcStats(snd, rcv)
    nmem = np.array([len(self.members[i]) for i in range(N)])
    exhausted = int((nonfb & (st.size >= nmem)).sum())
    line_vals = set((np.unique(snd[isline]) - LINE0).tolist())
    yes = set(self.B["YES"]); no = set(self.B["NO"])
    outc = set(self.B["driver"])
    out_yes = st.any_from(outc, yes)
    out_no = st.any_from(outc, no)
    exc_inner = (rk == ts.INNER) & (sk == ts.SK_EXC)
    return dict(step=self.k - 1, connections=int(ordinary.sum()), by_kind=table,
                starting_remaining=int(((k0 == 0) & ordinary).sum()),
                given_up_total=self.given_up, made_total=self.made, withdrawn_by_use_total=self.withdrawn_by_use,
                per_cell_mean=float(per.mean()), per_cell_max=int(per.max()), per_cell_min_nonfb=int(per[nonfb].min()),
                exhausted=exhausted, line_values=len(line_vals), line_values_in_use=len(line_vals & used),
                from_feedback_cells=dict(n=int((sk == ts.SK_FBC).sum()), w_min=float(w[sk == ts.SK_FBC].min()) if (sk == ts.SK_FBC).any() else None,
                                         w_max=float(w[sk == ts.SK_FBC].max()) if (sk == ts.SK_FBC).any() else None),
                output_end_taking_yes=out_yes, output_end_taking_no=out_no,
                exc_into_inner_mean_w=float(w[exc_inner].mean()) if exc_inner.any() else None)


class FastRun(EvRun):
    """E63's recording loop (tissue_step1.Run.run, stage A, no checkpoints) with one change: a byte whose steps hold none of the
    counting steps is computed by one call into the core instead of one call a step. Every record is taken at the same moment as
    in E63's loop (after the byte, or after operation 7 of a counting step), so the records are the same."""

    def count_steps(self):
        """E63's counting steps (after operation 7 of steps 127, 129, 1,024 and the last step of every 128th byte from byte 127),
        written for any number of steps a byte; with count_every other than 128, the last step of every count_every-th byte."""
        S, e = ts.S, getattr(self, "count_every", 128)
        return {127, 129, 1024} | {S * b + S - 1 for b in range(e - 1, self.nbytes, e)}

    def run(self, ckpt=0, die_after=None, resume=False, log=None):
        T, cfg = self.T, self.cfg
        S, BLOCK, N, REC_FLOOR = ts.S, ts.BLOCK, ts.N, ts.REC_FLOOR
        cst = self.count_steps()
        stretch = cfg["stretch"]
        c0 = time.process_time(); cstretch = c0
        one = np.zeros(1, np.uint8); em1 = np.zeros(1, np.int16)
        while self.pos < self.nbytes:
            t = self.pos
            byte = int(self.st[t])
            if t % BLOCK == 0:
                self.block_dist[t] = T.distances(self.present_in(self._dist_lo(t), self._dist_hi(t)))
            k0 = T.k
            if any(k in cst for k in range(k0, k0 + S)):
                em = None
                for _ in range(S):
                    k = T.k
                    r = T.step(byte)
                    if r is not None:
                        em = r
                    if k in cst:
                        rec = T.counts(self.used); rec["at"] = f"after operation 7 of step {k}"
                        self.counts.append(rec)
            else:
                one[0] = byte
                evt.L.evt_run_bytes(one, 1, em1)
                em = int(em1[0])
            self.emis[t] = em
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
            if stretch and self.pos % stretch == 0:
                now = time.process_time()
                lo = self.pos - stretch
                self.stretch.append(self.read_stretch(lo, self.pos))
                self.cpu.append(dict(bytes=[lo, self.pos - 1], cpu_s=now - cstretch, per_byte=(now - cstretch) / stretch,
                                     mean_connections=T.counts(self.used)["per_cell_mean"],
                                     emitted_share=float((self.emis[lo:self.pos] >= 0).mean()), active_end=int((T.x > 0).sum())))
                cstretch = now
                if log:
                    print(f"byte {self.pos} cpu/byte {self.cpu[-1]['per_byte']:.4f} conns/cell {self.cpu[-1]['mean_connections']:.2f} active {self.cpu[-1]['active_end']}", flush=True)
            for wn, w0 in cfg["windows"].items():
                if self.pos == w0 + cfg["wlen"]:
                    self.save_window(wn, w0)
        self.cpu_total += time.process_time() - c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rule", choices=("first", "coincidence", "fixed"))
    ap.add_argument("name")
    ap.add_argument("--bytes", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=73129)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--trace-sender", action="store_true")
    ap.add_argument("--strength-byte", action="store_true")
    ap.add_argument("--astar", type=float, default=0.125)
    ap.add_argument("--withdraw-by-use", action="store_true")
    ap.add_argument("--use-exempt-feedback", action="store_true")
    ap.add_argument("--steps", type=int, default=32, help="steps a byte (a number of the tissue; E63: 32)")
    ap.add_argument("--cap", type=int, default=0, help="at most this many incoming connections a cell (0: no cap)")
    ap.add_argument("--turnover", action="store_true", help="plan 2: make a connection every --make-period steps whatever the activity; trial of new connections")
    ap.add_argument("--make-period", type=int, default=128, help="steps between connections a cell makes (E63: 128)")
    ap.add_argument("--trial", type=int, default=512, help="steps a new connection from a cell is on trial (plan 2)")
    ap.add_argument("--trial-theta", type=float, default=0.25, help="kept if its use is at least this share of the mean use of the cell's other connections")
    ap.add_argument("--norm", action="store_true", help="plan 3: a cell's learnable incoming strengths held at --c-e (excitatory) and --c-i (inhibitory)")
    ap.add_argument("--c-e", type=float, default=0.75)
    ap.add_argument("--c-i", type=float, default=0.25)
    ap.add_argument("--gain", action="store_true", help="plan 5: each cell's own gain, moved once a byte toward rho times its sources' mean value")
    ap.add_argument("--eta-g", type=float, default=1 / 128)
    ap.add_argument("--rho", type=float, default=0.875)
    ap.add_argument("--gmax", type=float, default=8.0)
    ap.add_argument("--long-frac", type=float, default=0.0, help="plan 5: share of excitatory cells with long axons (given at birth)")
    ap.add_argument("--long-r", type=float, default=0.5, help="their reach (E63's neighbourhood radius: 1/8)")
    ap.add_argument("--spike", action="store_true", help="plan 7: all-or-none output (1 when potential less threshold is at least --fire)")
    ap.add_argument("--fire", type=float, default=0.125)
    ap.add_argument("--inh-adapt", action="store_true", help="plan 7: each cell's inhibitory total moved once a byte toward an activity target")
    ap.add_argument("--eta-i", type=float, default=0.01)
    ap.add_argument("--a-i", type=float, default=1 / 32)
    ap.add_argument("--ci-max", type=float, default=4.0)
    ap.add_argument("--count-every", type=int, default=128, help="bytes between counts of connections (E63: 128)")
    ap.add_argument("--slow-records", action="store_true", help="the recording loop as E63 wrote it (a call into the core every step)")
    a = ap.parse_args()
    ts.S = a.steps
    if not a.slow_records:
        evt.EvTissue.distances = _distances_fast
        evt.EvTissue.counts = _counts_fast
    if a.cap:
        evt.L.evt_set_cap(a.cap)
    assert not a.turnover or a.withdraw_by_use, 'plan 2 runs with withdrawal by use (the use is measured by it)'
    evt.L.evt_set_turnover(int(a.turnover), a.make_period, a.trial, a.trial_theta)
    evt.L.evt_set_norm(int(a.norm), a.c_e, a.c_i)
    evt.L.evt_set_gain(int(a.gain), a.eta_g, a.rho, a.gmax)
    evt.L.evt_set_spike(int(a.spike), a.fire)
    evt.L.evt_set_inh_adapt(int(a.inh_adapt), a.eta_i, a.a_i, a.ci_max)
    sw = dict(floor=a.floor, tol=a.tol, trace_sender=a.trace_sender, strength_byte=a.strength_byte, astar=a.astar, long_frac=a.long_frac, long_r=a.long_r,
              withdraw_by_use=a.withdraw_by_use, use_exempt_feedback=a.use_exempt_feedback)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    c0, w0 = time.process_time(), time.time()
    R = (EvRun if a.slow_records else FastRun)(a.rule, a.seed, a.bytes, out, sw)
    R.count_every = a.count_every
    R.run(0, None, False, log=True)
    cpu_recorded = time.process_time() - c0
    T = R.T
    cnt = T.counters()
    snd, rcv, w, k0 = T.live()
    conns = int((snd < ts.YES_LINE).sum())
    np.save(out / "emissions.npy", R.emis)
    sm = R.summary()
    if sm:
        np.savez(out / "summary.npz", **{f"{wn}_{k}": v for wn, d in sm.items() for k, v in d.items()})
    # the same tissue by the core alone
    c1 = time.process_time()
    T2 = evt.EvTissue(a.seed, a.rule, **sw)
    em2 = T2.run_bytes(R.st)
    cpu_core = time.process_time() - c1
    nb = a.bytes
    steps = cnt["steps"]
    tc = np.zeros(2, np.int64); evt.L.evt_trial_counts(tc)
    sw = dict(sw, steps=a.steps, cap=a.cap, turnover=a.turnover, make_period=a.make_period, trial=a.trial, trial_theta=a.trial_theta,
              trial_withdrawn=int(tc[0]), trial_kept=int(tc[1]), norm=a.norm, c_e=a.c_e, c_i=a.c_i, gain=a.gain, eta_g=a.eta_g, rho=a.rho, gmax=a.gmax, spike=a.spike, fire=a.fire, inh_adapt=a.inh_adapt, eta_i=a.eta_i, a_i=a.a_i, ci_max=a.ci_max, norm_calls=int(evt.L.evt_norm_calls()), records="slow" if a.slow_records else "fast")
    rec = dict(experiment="E63, the tissue's program (event-driven)", task="TASK_E63_THE_TISSUE_PROGRAM_EVENT_DRIVEN_20261119.md",
               command=" ".join(["python3", "evt_run.py"] + sys.argv[1:]), switches=sw,
               core_sha256=hashlib.sha256(evt.SRC.read_bytes()).hexdigest(),
               wrapper_sha256=hashlib.sha256((HERE / "evt.py").read_bytes()).hexdigest(),
               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               reference_program_sha256=ts.sha_file(HERE.parent / "TISSUE_STEP1" / "tissue_step1.py"),
               versions=ts.versions(), stage="A", tissue=a.rule, seed=a.seed,
               stream=dict(sha256=hashlib.sha256(R.st.tobytes()).hexdigest(), length=nb, used_values=len(R.used)),
               birth=T.B["digests"], counts=R.counts, stretches=R.stretch, cpu_by_stretch=R.cpu,
               cpu_total_s=cpu_recorded, cpu_per_byte_with_records=cpu_recorded / nb, wall_s=time.time() - w0,
               cpu_core_only_s=cpu_core, cpu_per_byte_core_only=cpu_core / nb,
               core_only_emissions_identical=bool(np.array_equal(em2, R.emis)),
               counters=cnt, counters_per_byte={k: v / nb for k, v in cnt.items()},
               mean_cells_sending_a_step=cnt["senders_passing_on"] / steps,
               connection_visits_a_byte=(cnt["visits_passing_on"] + cnt["visits_strengths"] + cnt["visits_life"]) / nb,
               cell_updates_a_byte=cnt["cell_updates"] / nb,
               connections_a_cell_end=conns / ts.N, made_total=T.made, given_up_total=T.given_up, withdrawn_by_use_total=T.withdrawn_by_use,
               emissions=dict(share=float((R.emis >= 0).mean()), commonest=[[int(v), int(c)] for v, c in
                                                                          sorted(zip(*np.unique(R.emis[R.emis >= 0], return_counts=True)),
                                                                                 key=lambda q: -q[1])[:5]] if (R.emis >= 0).any() else []),
               windows=sorted(n for n in R.win if (out / f"{n}.npz").exists()))
    (out / "receipt.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if rec["windows"]:
        step1_read.read_run(out)
    print("complete", out, f"cpu/byte {rec['cpu_per_byte_with_records']:.5f} core {rec['cpu_per_byte_core_only']:.6f}", flush=True)


if __name__ == "__main__":
    main()
