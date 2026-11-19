"""Step 1, second round (TASK_E63_STEP_1_SECOND_ROUND_20261119.md on the reviewer's branch): the runner.

Given a list of tissue settings, it runs them side by side (at most four processes, one thread each) on the fast program of the tissue
(TISSUE_EVENT/evt.py and evt_core.c, by E57's worker, not edited), with E63's stream, records and readings (TISSUE_STEP1/tissue_step1.py,
its class Run, and step1_read.py, not edited), and writes one table.

    python3 round2.py                 # every tissue of the round that the program can run; the others are listed with what is missing
    python3 round2.py one '<json>'    # one tissue (used by the runner; writes PREP/round2_runs/<name>.json)

A setting is {"rule": "first" | "coincidence" | "fixed", "F": 0 | 1, "T": 8 | 32 | 128, "W": 0 | 1}:
  F  an output below 1/128 is zero        -> evt.EvTissue(floor=1/128)                     (in the program)
  T  the target of activity a* = 1/T       -> no parameter: a* is the constant ASTAR 0.125 of evt_core.c, line 41
  W  withdrawal by use (the round's form)  -> no parameter: evt_core.c has no use per connection and no withdrawal by it
Everything else is stage A's (seed 73129, 16,384 bytes, the windows A1 = bytes 2,048-6,143 and A2 = 12,288-16,383, formula 11 with E10,
tol 0, a trace for each connection, strengths at every step).

Each tissue: tissue_step1.Run with its tissue replaced by an evt.EvTissue (the interface E57's worker gave it for this use), run without
saving (a tissue of 16,384 bytes costs about a minute; an interrupted one is run again from the start). Measured at the end and in each
window: connections a cell (Tissue.counts, per_cell_mean, as in stage A); at d = 1, 2, 4 connections from the input lines (the mode of
the block distances over the window, Run.mode_distance) the share of cells neither silent nor constant (step1_read.shares) and the score
S(d, L) for L = 0, 1, 2 (step1_read.cell_scores), beside the score of the present byte alone C0(L) (step1_read.c0); the mean number of
cells with non-zero output at a step (counted after every step) and the core's count of senders that passed a change on, a step; the
CPU-seconds a byte of the run with its records, and of the readings. Step 1 is reached by a tissue in this round when, in A2, at d = 2 and
d = 4, the share is at least 1/2 and S(d, 0) at least the level 0.1103 (step1_tables.stays, the rule of stage A's first outcome).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "round2_runs"
SEED, NBYTES, LEVEL = 73129, 16384, 0.1103
MAXP = 4


def all_settings():
    """The round's tissues: for each strength rule every combination of F (0, 1), T (8, 32, 128) and W (0, 1); and the fixed tissue."""
    out = []
    for rule in ("first", "coincidence"):
        for F in (0, 1):
            for T in (8, 32, 128):
                for W in (0, 1):
                    out.append(dict(rule=rule, F=F, T=T, W=W))
    out.append(dict(rule="fixed", F=0, T=8, W=0))
    return out


def name(s):
    return "fixed" if s["rule"] == "fixed" else f"{s['rule']}_F{s['F']}_T{s['T']}_W{s['W']}"


def missing(s):
    m = []
    if s["T"] != 8:
        m.append(f"T: a* = 1/{s['T']} (evt_core.c has a* as the constant ASTAR 0.125; it is used in formula 9, in the condition of "
                 f"formula 12, in the coincidence rule's inhibitory part and in the sliding threshold ϑ)")
    if s["W"]:
        m.append("W: withdrawal by use, the round's form (a use per connection, the running product of the input's trace and the "
                 "receiving cell's activity, brought up to date only when the cell is active; withdrawn below 1/128 of the cell's "
                 "largest for 128 steps in which the cell was active)")
    return m


# ------------------------------------------------------------------------------------------------ one tissue (a child process)
def one(s):
    sys.path.insert(0, str(ROOT / "TISSUE_EVENT")); sys.path.insert(0, str(ROOT / "TISSUE_STEP1"))
    import numpy as np
    import evt
    import tissue_step1 as ts
    import step1_read as RD
    assert not missing(s), missing(s)
    nm = name(s)
    work = Path(os.environ.get("ROUND2_WORK", "/tmp")) / nm
    work.mkdir(parents=True, exist_ok=True)

    class Counted(evt.EvTissue):
        """evt.EvTissue with a count of the cells whose output is not zero, after every step (nothing else is changed)."""
        nz_sum = 0; nz_steps = 0

        def step(self, byte, hook=None):
            r = super().step(byte, hook)
            self.nz_sum += int(np.count_nonzero(self.x)); self.nz_steps += 1
            return r

    c0 = time.process_time()
    R = ts.Run(s["rule"], SEED, NBYTES, "A", False, work)
    T = Counted(SEED, s["rule"], floor=(1 / 128 if s["F"] else 0.0), B=R.T.B)
    R.T = T
    c_birth = time.process_time() - c0
    c1 = time.process_time()
    R.run(0, None, False, log=False)
    c_run = time.process_time() - c1
    cnt = T.counts(R.used)
    ctr = T.counters()
    c2 = time.process_time()
    RD.LAGS = (0, 1, 2)
    res = dict(name=nm, setting=s, seed=SEED, bytes=NBYTES, connections_a_cell_end=cnt["per_cell_mean"], live_connections_end=cnt["connections"],
               made=T.made, given_up=T.given_up, nonzero_cells_a_step=T.nz_sum / T.nz_steps,
               senders_passing_on_a_step=ctr["senders_passing_on"] / max(1.0, ctr["steps"]), counters=ctr,
               emitted_share=float((R.emis >= 0).mean()), windows={})
    cfg = ts.STAGE["A"]
    for wn, w0 in cfg["windows"].items():
        wl = cfg["wlen"]
        Rw = R.win[wn]
        dist, _ = R.mode_distance(w0, w0 + wl)
        sh = RD.shares(Rw, dist)
        sc = RD.cell_scores(Rw, R.st, w0, wl, dist, SEED)
        C0 = {str(L): RD.c0(R.st, w0, wl, L)[0] for L in (0, 1, 2)}
        res["windows"][wn] = dict(bytes=[w0, w0 + wl - 1], shares={d: v["share"] for d, v in sh.items()}, shares_full=sh,
                                  S={d: {L: (v["S"] if v else None) for L, v in r["by_lag"].items()} for d, r in sc.items()},
                                  cells_at_d={d: r["cells_at_d"] for d, r in sc.items()}, C0=C0)
    a2 = res["windows"]["A2"]
    res["reached"] = all(a2["shares"][d] is not None and a2["shares"][d] >= 0.5 and a2["S"][d]["0"] is not None and a2["S"][d]["0"] >= LEVEL
                         for d in ("2", "4"))
    res["cpu_s"] = dict(birth=c_birth, run_with_records=c_run, readings=time.process_time() - c2)
    res["cpu_s_a_byte_run_with_records"] = c_run / NBYTES
    res["versions"] = dict(python=sys.version.split()[0], numpy=np.__version__)
    for f in work.glob("*.npz"):
        f.unlink()
    OUT.mkdir(exist_ok=True)
    (OUT / f"{nm}.json").write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ the runner and the table
def table(results, notrun):
    def f(x, n=3):
        return "–" if x is None else f"{x:.{n}f}"
    L = ["| Tissue | Connections a cell, end | Window | Share neither silent nor constant, d = 1 / 2 / 4 | S(d, 0) d = 1 / 2 / 4 | "
         "S(d, 1) d = 1 / 2 / 4 | S(d, 2) d = 1 / 2 / 4 | Present byte alone C0(0) / C0(1) / C0(2) | Cells with non-zero output a step | "
         "Senders passing on a step | CPU-s a byte (run with records) | Reached |", "|" + "---|" * 12]
    for r in results:
        for wn in ("A1", "A2"):
            w = r["windows"][wn]
            L.append(f"| {r['name']} | {r['connections_a_cell_end']:.1f} | {wn} | " + " / ".join(f(w["shares"][d]) for d in ("1", "2", "4")) + " | "
                     + " | ".join(" / ".join(f(w["S"][d][Lg], 4) for d in ("1", "2", "4")) for Lg in ("0", "1", "2")) + " | "
                     + " / ".join(f(w["C0"][Lg], 4) for Lg in ("0", "1", "2")) + f" | {r['nonzero_cells_a_step']:.0f} | {r['senders_passing_on_a_step']:.0f} | "
                     f"{r['cpu_s_a_byte_run_with_records']:.4f} | {('yes' if r['reached'] else 'no') if wn == 'A2' else ''} |")
    L.append("")
    if notrun:
        L.append("**Not run: the program has no parameter for**\n")
        for nm, m in notrun:
            L.append(f"- {nm}: " + "; ".join(x.split(" (")[0] for x in m))
    return "\n".join(L) + "\n"


def main():
    sys.path.insert(0, str(ROOT / "TISSUE_EVENT")); sys.path.insert(0, str(ROOT / "TISSUE_STEP1"))
    import evt  # noqa: F401  builds evt_core.so once, before the children start
    settings = all_settings()
    run = [s for s in settings if not missing(s)]
    notrun = [(name(s), missing(s)) for s in settings if missing(s)]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    OUT.mkdir(exist_ok=True)
    todo = [s for s in run if not (OUT / f"{name(s)}.json").exists()]
    live = []
    log = open(OUT / "LAUNCHES.txt", "a")
    while todo or live:
        for p, nm in live[:]:
            if p.poll() is not None:
                live.remove((p, nm)); log.write(f"{time.strftime('%H:%M:%S', time.gmtime())} exit {p.returncode} {nm}\n"); log.flush()
        while todo and len(live) < MAXP:
            s = todo.pop(0)
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "one", json.dumps(s)], env=env,
                                 stdout=open(OUT / f"{name(s)}.log", "w"), stderr=subprocess.STDOUT)
            live.append((p, name(s))); log.write(f"{time.strftime('%H:%M:%S', time.gmtime())} start {name(s)} pid {p.pid}\n"); log.flush()
        time.sleep(2)
    results = [json.loads((OUT / f"{name(s)}.json").read_text()) for s in run if (OUT / f"{name(s)}.json").exists()]
    md = table(results, notrun)
    (HERE / "ROUND2_TABLE.md").write_text(md)
    print(md)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "one":
        one(json.loads(sys.argv[2]))
    else:
        main()
