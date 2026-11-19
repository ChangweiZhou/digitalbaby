"""Part (a) of the task: with every switch off, the new program against E63's program.

    python3 part_a.py example                 # the definition's worked example, E63's check 1 parts (a) to (d)
    python3 part_a.py trace <fixed|f11w|coincidence>   # against the reference trace in ref/ (ref_trace.py, E63's program)

Each writes a JSON receipt in qual/ and prints pass or the first difference.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import evt
from evt import L, ts

HERE = Path(__file__).resolve().parent
QUAL = HERE / "qual"
sys.path.insert(0, str(HERE.parent / "TISSUE_STEP1"))
import step1_checks as C  # noqa: E402  (E63's check 1: its example and expected rows, unchanged)


def receipt(name, obj):
    QUAL.mkdir(exist_ok=True)
    obj["core_sha256"] = hashlib.sha256(evt.SRC.read_bytes()).hexdigest()
    obj["wrapper_sha256"] = hashlib.sha256((HERE / "evt.py").read_bytes()).hexdigest()
    obj["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    obj["reference_sha256"] = hashlib.sha256((HERE.parent / "TISSUE_STEP1" / "tissue_step1.py").read_bytes()).hexdigest()
    obj["versions"] = ts.versions()
    (QUAL / f"{name}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(name, "pass =", obj["pass"], flush=True)


# ================================================================================================ the worked example
def example_core(variant):
    """The worked example of check 1, computed by the core with every switch off."""
    names_c = ["F", "E", "I", "O"]
    ext = ["s1", "h", "c", "q", "g", "u", "f", "L3", "L5", "NO"]
    vals = {"s1": lambda k: 0.1, "h": lambda k: 0.25, "c": lambda k: 0.125, "q": lambda k: 0.25, "g": lambda k: 0.125,
            "u": lambda k: 0.5 if k < 16 else 0.125, "f": lambda k: 0.25, "L3": lambda k: 1.0, "L5": lambda k: 0.0,
            "NO": lambda k: 1.0 if k == 0 else 0.0}
    sign = {"s1": 1, "h": -1, "c": 1, "q": 1, "g": -1, "u": 1, "f": -1, "L3": 1, "L5": 1, "NO": 1, "E": 1, "F": 1}
    lines = {"L3", "L5", "NO"}
    r4 = variant == "r4"
    cells = {
        "F": dict(src={"NO": [1.0, 0.0], **({"s1": [0.25, 0.10]} if r4 else {})}, v=0.0 if variant == "r5" else 0.02, p=0.02,
                  th=0.05 if r4 else 0.0, xp=0.0, plastic=r4, theta=r4, rule=1),
        "E": dict(src={"L3": [0.125, 0.02], "L5": [0.125, 0.40], "h": [0.25, 0.25], "c": [0.5, 0.125], "F": [0.25, 0.0]},
                  v=0.04, p=0.05, th=0.02, xp=0.03, plastic=True, theta=True, rule=1),
        "I": dict(src={"E": [0.5, 0.06], "q": [0.25, 0.25], "g": [0.25, 0.125]}, v=0.10, p=0.12, th=0.05, xp=0.06, plastic=True,
                  theta=True, rule=2 if variant == "K" else 1),
        "O": dict(src={"u": [0.75, 0.50], "f": [0.25, 0.25], "F": [0.25, 0.0]}, v=0.70, p=0.3125, th=0.10, xp=0.60, plastic=True,
                  theta=True, rule=1)}
    idx = {n: q for q, n in enumerate(names_c + ext)}
    nc, nz = len(names_c), len(names_c) + len(ext)
    L.evt_init(nc, nz, 0.0, 0.0, 0, 0, 0, 0, 1, 32, -1, 0)
    sig = np.array([sign.get(n, 1) for n in names_c + ext], np.float64)
    lrn = np.array([0 if n in lines else 1 for n in names_c + ext], np.uint8)
    L.evt_set_senders(sig, lrn, np.zeros(nz, np.uint8))
    L.evt_set_cells(np.array([cells[n]["rule"] if cells[n]["plastic"] else 0 for n in names_c], np.uint8),
                    np.array([1 if cells[n]["theta"] else 0 for n in names_c], np.uint8))
    L.evt_set_members(np.zeros(nc + 1, np.int32), np.zeros(1, np.int32))
    vt0 = np.full(nc, ts.ASTAR)
    L.evt_set_cell_state(np.array([cells[n]["v"] for n in names_c]), np.array([cells[n]["p"] for n in names_c]),
                         np.array([cells[n]["th"] for n in names_c]), np.array([cells[n]["xp"] for n in names_c]), vt0.ctypes.data)
    slots = {}
    for n in names_c:
        for j, (w, e) in cells[n]["src"].items():
            slots[(n, j)] = L.evt_add_connection(idx[j], idx[n], w, e, 0)
    L.evt_start()
    x = evt.view(L.evt_ptr_x(), nc); A = evt.view(L.evt_ptr_A(), nc); m = evt.view(L.evt_ptr_m(), nc); v = evt.view(L.evt_ptr_v(), nc)
    rows = {}
    for k in range(32):
        for n in ext:
            L.evt_set_line(idx[n], vals[n](k))
        L.evt_step(-1)
        rows[k] = {n: (float(A[q]), float(m[q]), float(v[q]), float(x[q])) for q, n in enumerate(names_c)}
    t = evt.EvTissue.__new__(evt.EvTissue)
    snd, rcv, w, e, k0, ub = evt.EvTissue.live_full(t)
    fin = {}
    for s_, r_, ww in zip(snd.tolist(), rcv.tolist(), w.tolist()):
        n, j = (names_c + ext)[r_], (names_c + ext)[s_]
        if j not in lines:
            fin.setdefault(n, {})[j] = ww
    return rows, fin


def part_example():
    out, ok = {}, True
    for part, variant in (("a", "r4"), ("b", "E4"), ("c", "K"), ("d", "r5")):
        rows, fin = example_core(variant)
        ref_rows, ref_fin = C.example(variant)                     # E63's own computation of the example
        diff_ref = max(abs(a - b) for k in rows for n in rows[k] for a, b in zip(rows[k][n], ref_rows[k][n]))
        diff_ref_fin = max(abs(fin[c][j] - ref_fin[c][j]) for c in ref_fin for j in ref_fin[c] if ref_fin[c])
        worst = 0.0
        for k, exp in C.EXPECT_ROWS[variant].items():              # the expected rows of the definition, as check 1 reads them
            r = rows[k]
            vals = (r["F"][3], r["E"][0], r["E"][1], r["E"][3], r["I"][0], r["I"][1], r["I"][3], r["O"][0], r["O"][1], r["O"][3])
            worst = max(worst, max(abs(round(v, 4) - e) for v, e in zip(vals, exp)))
        fw = max(abs(fin[c][j] - vv) for c, d in C.EXPECT_FINAL[variant].items() for j, vv in d.items())
        good = worst <= 1e-9 and fw < 0.6e-4 and diff_ref <= 1e-9 and diff_ref_fin <= 1e-9
        out[part] = dict(variant=variant, largest_row_difference_after_rounding=worst, largest_final_difference=fw,
                         largest_difference_from_E63_rows=diff_ref, largest_difference_from_E63_final=diff_ref_fin, passed=good)
        ok &= good
        print(part, variant, f"rows vs expected {worst:.3g}, final vs expected {fw:.3g}, vs E63's computation rows {diff_ref:.3g} final {diff_ref_fin:.3g}",
              flush=True)
    receipt("part_a_example", dict(**{"pass": bool(ok)}, parts=out))


# ================================================================================================ the reference traces
def part_trace(tissue):
    seed = 73129
    nbytes = 512 if tissue == "f11w" else 2048
    R = np.load(HERE / "ref" / f"{tissue}_{seed}_{nbytes}.npz", allow_pickle=False)
    rule = "first" if tissue == "f11w" else tissue
    st, _ = ts.stream(seed, nbytes)
    c0 = time.process_time()
    T = evt.EvTissue(seed, rule, f11_written=(tissue == "f11w"))
    em = T.run_bytes(st)
    cpu = time.process_time() - c0
    res = dict(tissue=tissue, seed=seed, bytes=nbytes, cpu_s=cpu, cpu_per_byte=cpu / nbytes, counters=T.counters())
    first = None
    # emissions
    eq_em = bool(np.array_equal(em.astype(np.int64), R["emis"].astype(np.int64)))
    res["emissions_identical"] = eq_em
    if not eq_em:
        first = first or f"emission at byte {int(np.nonzero(em != R['emis'])[0][0])}"
    # connections made and given up (step, sender, receiver)
    made, gone = T.events()
    rm, rg = R["made"], R["gone"]
    res["made"] = [len(made), len(rm)]; res["given_up"] = [len(gone), len(rg)]
    for name, mine, ref in (("made", made[:, :3], rm), ("given up", gone[:, :3], rg)):
        n = min(len(mine), len(ref))
        bad = np.nonzero((mine[:n] != ref[:n]).any(1))[0]
        if len(bad) or len(mine) != len(ref):
            q = int(bad[0]) if len(bad) else n
            step = int(min(mine[q][0] if q < len(mine) else 1 << 60, ref[q][0] if q < len(ref) else 1 << 60))
            res[f"first_difference_{name}"] = dict(index=q, step=step, mine=mine[q].tolist() if q < len(mine) else None,
                                                   reference=ref[q].tolist() if q < len(ref) else None)
            first = first or f"connection {name} differs first at step {step}"
    res["events_identical"] = "first_difference_made" not in res and "first_difference_given up" not in res
    # the life generator
    rl = pickle.loads(R["rl_state"].tobytes())
    res["generator_identical"] = T.rng_state() == rl
    if not res["generator_identical"]:
        first = first or "state of the life generator"
    # the connections present, and the quantities within 1e-9
    snd, rcv, w, e, k0, ub = T.live_full()
    al = R["alive"].astype(bool)
    rs, rr, rw, re_, rk0 = R["snd"][al], R["rcv"][al], R["w"][al], R["e"][al], R["k0"][al]
    same_conn = len(snd) == len(rs) and bool(np.array_equal(snd, rs) and np.array_equal(rcv, rr) and np.array_equal(k0, rk0))
    res["connections_identical"] = same_conn
    res["connections"] = [int(len(snd)), int(len(rs))]
    if not same_conn:
        first = first or "connections present at the end"
    diffs = {}
    if same_conn:
        diffs["strength"] = float(np.abs(w - rw).max())
        diffs["trace"] = float(np.abs(e - re_).max())
        if tissue == "f11w":
            diffs["sending_average_ub"] = float(np.abs(ub - R["ub"][al]).max())
    cs = T.cells()
    diffs["state_v"] = float(np.abs(cs["v"] - R["v"]).max())
    diffs["output_x"] = float(np.abs(cs["x"] - R["x"]).max())
    diffs["prediction_p"] = float(np.abs(cs["p"] - R["p"]).max())
    diffs["threshold"] = float(np.abs(cs["th"] - R["th"]).max())
    diffs["average_abar"] = float(np.abs(cs["abar"] - R["abar"]).max())
    if cs["vt"] is not None:
        diffs["sliding_threshold_vt"] = float(np.abs(cs["vt"] - R["vt"]).max())
    res["largest_differences"] = diffs
    res["largest_difference"] = max(diffs.values())
    res["klast_identical"] = bool(np.array_equal(cs["klast"], R["klast"]))
    ok = (first is None) and res["largest_difference"] <= 1e-9 and res["klast_identical"]
    res["first_difference"] = first
    receipt(f"part_a_{tissue}", dict(**{"pass": bool(ok)}, **res))
    print(json.dumps({k: res[k] for k in ("emissions_identical", "events_identical", "generator_identical", "connections_identical",
                                          "largest_difference", "first_difference", "made", "given_up", "cpu_per_byte")}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "example":
        part_example()
    else:
        part_trace(sys.argv[2])
