"""Part (a) of the task, the reference side: run E63's program (TISSUE_STEP1/tissue_step1.py, imported unchanged) on one tissue of
stage A for its first bytes, and keep what the comparison needs: every connection made and given up with its step, the emissions, the
state of the life generator and the full state at the end.

    python3 ref_trace.py <first|coincidence|fixed|f11w> [--bytes N] [--seed 73129]

Writes TISSUE_EVENT/ref/<tissue>_<seed>_<bytes>.npz and a receipt beside it.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "TISSUE_STEP1"))
import tissue_step1 as ts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tissue", choices=("first", "coincidence", "fixed", "f11w"))
    ap.add_argument("--bytes", type=int)
    ap.add_argument("--seed", type=int, default=73129)
    a = ap.parse_args()
    rule = "first" if a.tissue == "f11w" else a.tissue
    nbytes = a.bytes or (512 if a.tissue == "f11w" else 2048)
    st, _ = ts.stream(a.seed, nbytes)
    T = ts.Tissue(a.seed, rule, f11_written=(a.tissue == "f11w"))
    T.audit = []
    made = []
    prev = [T.made]

    def hook(self, k, A, m, v, x):
        d = self.made - prev[0]
        if d:
            n = self.n
            al = np.nonzero(self.alive[:n])[0][-d:]
            for c in al:
                assert self.k0[c] == k
                made.append((k, int(self.snd[c]), int(self.rcv[c])))
        prev[0] = self.made

    emis = np.full(nbytes, -1, np.int16)
    c0 = time.process_time()
    for t in range(nbytes):
        em = T.run_byte(int(st[t]), hook)
        emis[t] = em
    cpu = time.process_time() - c0
    gone = [(int(k), int(s), int(r)) for k, ss, ww, rr in T.audit for s, r in zip(ss.tolist(), rr.tolist())]
    out = HERE / "ref"
    out.mkdir(exist_ok=True)
    name = f"{a.tissue}_{a.seed}_{nbytes}"
    d = T.state_arrays()
    d["x"] = T.x
    d["rl_state"] = np.frombuffer(pickle.dumps(T.rl.getstate()), np.uint8)
    d["emis"] = emis
    d["made"] = np.array(made, np.int64).reshape(-1, 3)
    d["gone"] = np.array(gone, np.int64).reshape(-1, 3)
    np.savez(out / f"{name}.npz", **d)
    rec = dict(tissue=a.tissue, seed=a.seed, bytes=nbytes, cpu_s=cpu, cpu_per_byte=cpu / nbytes,
               program_sha256=hashlib.sha256((HERE.parent / "TISSUE_STEP1" / "tissue_step1.py").read_bytes()).hexdigest(),
               tracer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), versions=ts.versions(),
               made=len(made), given_up=len(gone), emitted=int((emis >= 0).sum()),
               stream_sha256=hashlib.sha256(st.tobytes()).hexdigest())
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print("complete", name, f"cpu {cpu:.1f}s", flush=True)


if __name__ == "__main__":
    main()
