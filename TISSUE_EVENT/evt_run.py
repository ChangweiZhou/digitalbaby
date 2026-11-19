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
    a = ap.parse_args()
    sw = dict(floor=a.floor, tol=a.tol, trace_sender=a.trace_sender, strength_byte=a.strength_byte, astar=a.astar,
              withdraw_by_use=a.withdraw_by_use, use_exempt_feedback=a.use_exempt_feedback)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    c0, w0 = time.process_time(), time.time()
    R = EvRun(a.rule, a.seed, a.bytes, out, sw)
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
