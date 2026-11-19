"""E63, decision 66 of the coordinator: runs of the original program (tissue_step1.py) and of its copy (tissue_step1_active.py) on seed
73129, for the five tissues of stage A over their first 2,048 bytes, for the comparison and the timings that the decision asks for.

    python3 e63_active_runs.py run <tissue> <variant> [--resume]
      tissue:  fixed | first | coincidence | first_f11written | first_withdrawbyuse
      variant: orig | abc | a | b | c | abcm      (abcm: the three changes and the sorted maximum, fifth tissue only)

orig and abc run bytes 0 to 2,047 from birth; a, b, c and abcm start from the original's saved state at byte 1,024 (runs/<tissue>/orig/
state1024.npz) and run bytes 1,024 to 2,047. CPU-seconds a byte are measured over bytes 1,024 to 2,047 (time.process_time, one
thread). At every step the cells with non-zero output are counted by phase. A checkpoint is saved every 256 bytes; --resume continues
from it (Tissue.load_arrays, as in E63's own resuming, which E63's check 8 showed exact).
"""
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tissue_step1 as O            # noqa: E402  the original, not edited
import tissue_step1_active as C     # noqa: E402  the copy

SEED = 73129
NB, MID, CK = 2048, 1024, 256
OUT = HERE / "runs_active"
TISSUES = {"fixed": dict(rule="fixed"), "first": dict(rule="first"), "coincidence": dict(rule="coincidence"),
           "first_f11written": dict(rule="first", f11_written=True), "first_withdrawbyuse": dict(rule="first", by_use=True)}
VARIANTS = {"orig": None, "abc": ("a", "b", "c"), "a": ("a",), "b": ("b",), "c": ("c",), "abcm": ("a", "b", "c", "maxsort")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make(tissue, variant, B):
    kw = TISSUES[tissue]
    rule = kw["rule"]; f11 = kw.get("f11_written", False); bu = kw.get("by_use", False)
    if variant == "orig":
        return O.Tissue(SEED, rule, f11, B=B, by_use=bu)
    return C.Tissue(SEED, rule, f11, B=B, by_use=bu, opts=VARIANTS[variant])


def save_state(T, path, emis, acc):
    d = T.state_arrays()
    d["rl_state"] = np.frombuffer(pickle.dumps(T.rl.getstate()), np.uint8)
    d["emis"] = emis
    d["acc"] = np.frombuffer(json.dumps(acc).encode(), np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, path)


def load_state(T, path):
    d = np.load(path, allow_pickle=False)
    T.load_arrays(d)
    T.rl.setstate(pickle.loads(d["rl_state"].tobytes()))
    if hasattr(T, "_set_nocand"):
        T._set_nocand()
    return d["emis"].copy(), json.loads(d["acc"].tobytes().decode())


def run(tissue, variant, resume):
    assert tissue in TISSUES and variant in VARIANTS
    assert variant != "abcm" or tissue == "first_withdrawbyuse"
    out = OUT / tissue / variant
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "checkpoint.npz"
    if (out / "receipt.json").exists():
        print("complete already", out); return
    st, _ = O.stream(SEED, NB)
    B = O.birth(SEED)
    T = make(tissue, variant, B)
    start = 0 if variant in ("orig", "abc") else MID
    emis = np.full(NB, -1, np.int16)
    acc = dict(pos=start, cpu_mid=0.0, cpu_all=0.0, act_phase_mid=[0] * O.S, steps_mid=0, act_phase_all=[0] * O.S, steps_all=0,
               live_sum_mid=0, t_use_mid=[0.0, 0.0], resumes=[], wall0=time.time())
    if resume and ck.exists():
        emis, acc = load_state(T, ck)
        acc["resumes"].append(dict(byte=acc["pos"], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    elif start == MID:
        src = OUT / tissue / "orig" / "state1024.npz"
        assert src.exists(), f"the original's state at byte 1,024 is needed: {src}"
        emis, _ = load_state(T, src)
        acc["state1024_sha256"] = sha(src)
    pos = acc["pos"]
    while pos < NB:
        byte = int(st[pos])
        tu = list(T.t_use) if hasattr(T, "t_use") else None
        c0 = time.process_time()
        em = None
        nz = []
        for _ in range(O.S):
            r = T.step(byte)
            if r is not None:
                em = r
            nz.append(int(np.count_nonzero(T.x)))
        dt = time.process_time() - c0
        emis[pos] = em
        acc["cpu_all"] += dt; acc["steps_all"] += O.S
        acc["act_phase_all"] = [a + b for a, b in zip(acc["act_phase_all"], nz)]
        if pos >= MID:
            acc["cpu_mid"] += dt; acc["steps_mid"] += O.S
            acc["act_phase_mid"] = [a + b for a, b in zip(acc["act_phase_mid"], nz)]
            acc["live_sum_mid"] += int(T.alive[:T.n].sum())
            if tu is not None:
                acc["t_use_mid"] = [a + (b - c) for a, b, c in zip(acc["t_use_mid"], T.t_use, tu)]
        pos += 1
        acc["pos"] = pos
        if pos == MID and variant == "orig":
            save_state(T, out / "state1024.npz", emis, acc)
        if pos % CK == 0 and pos < NB:
            save_state(T, ck, emis, acc)
            print(f"{tissue} {variant} byte {pos} cpu/byte(mid) "
                  f"{acc['cpu_mid'] / max(1, pos - MID) if pos > MID else acc['cpu_all'] / pos:.4f}", flush=True)
    save_state(T, out / "final.npz", emis, acc)
    nmid = NB - MID
    rec = dict(task="E63, decision 66 of the coordinator (19 November 2026)", tissue=tissue, variant=variant, seed=SEED,
               options=VARIANTS[variant], start_byte=start, end_byte=NB,
               cpu_s_per_byte_1024_2047=acc["cpu_mid"] / nmid, cpu_s_bytes_measured=acc["cpu_all"],
               mean_live_connections_1024_2047=acc["live_sum_mid"] / nmid,
               live_connections_at_2048=int(T.alive[:T.n].sum()),
               mean_nonzero_cells_per_step_1024_2047=sum(acc["act_phase_mid"]) / acc["steps_mid"],
               nonzero_cells_by_phase_1024_2047=[a / nmid for a in acc["act_phase_mid"]],
               fifth_tissue_lines_cpu_s_per_byte_1024_2047=[t / nmid for t in acc["t_use_mid"]] if hasattr(T, "t_use") else None,
               made=T.made, given_up=T.given_up, withdrawn_by_use=T.withdrawn_by_use,
               emissions_share=float((emis[start:] >= 0).mean()), resumes=acc["resumes"], wall_s=time.time() - acc["wall0"],
               beside=os.environ.get("E63A_BESIDE"), state1024_sha256=acc.get("state1024_sha256"),
               sources=dict(tissue_step1=sha(HERE / "tissue_step1.py"), tissue_step1_active=sha(HERE / "tissue_step1_active.py"),
                            runner=sha(__file__)),
               versions=dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine(), cores=os.cpu_count()),
               env={k: v for k, v in os.environ.items() if k.startswith(("OMP_", "OPENBLAS_", "MKL_"))})
    (out / "receipt.json").write_text(json.dumps(rec, indent=1))
    if ck.exists():
        ck.unlink()
    print("complete", out, f"{rec['cpu_s_per_byte_1024_2047']:.4f} s/byte", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    assert a[0] == "run", __doc__
    run(a[1], a[2], "--resume" in a)
