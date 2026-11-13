"""The sizing learner's span (brief OPUS_TASK_SIZING_SPAN_20261113.md, revision 2, commit 06cf6b0e).

Stage 1's copy fixture (sz_crosscut.fixture_stream: episodes  K <filler of D - 2> = K \\n, K one of 24 letters, chance 1/24) fed to
sz_learner.StreamLearner (every next byte a target, credit cut every 256 targets, state carried, backprop, AdamW 0.001, decay 0.3). Both
modules are imported unedited. Predictions are read from each chunk's forward pass before its update (sz_learner's observe hook).

    python3 sz_span.py admit                       admission items 1, 2, 3, 5 (item 4 is sz_verify_born.py); writes runs/span/admission.json
    python3 sz_span.py run <block> <dist> <seed>   one roster run; <dist> is a distance or "M"; writes runs/span/<block>_<dist>_<seed>.json
    python3 sz_span.py tables                      the per-block tables, spans and forecast outcomes; writes runs/span/SPAN_TABLES.json

Roster (stream and model seeds 1104, 1105, 1106; births 73129, 93683, 116089 paired with them in that order):
    A random   fixed 4, 12, 16, 18, 20, 24, 40, 60   235 x (D + 1) chunks
    B random   mixed M = {4, 12, 24, 40}             20,000 chunks
    C born     fixed 4, 12, 24, 40                   235 x (D + 1) chunks
    P permuted fixed 12, 24                          235 x (D + 1) chunks
    D born     mixed M                               20,000 chunks

Choices where the brief leaves room, fixed here before any roster run:
  * Mixed stream. Keys and filler come from a generator seeded "sizing-crosscut|<label>|<seed>" exactly as stage 1's (label = the set's
    distances joined by commas, so a one-element set {12} gives label "12", stage 1's own seed string); each episode's D is drawn first, by
    random.choice over the set in ascending order, from a separate generator seeded "sizing-span-D|<label>|<seed>". The stream is generated
    to at least chunks x 256 bytes and cut there, as stage 1 does; an instance is kept when its repeat byte lies inside the stream.
  * Born start: the birth's bare state dict loaded IN PLACE (lr.m.load_state_dict, strict) after the learner is built, so the optimizer,
    built fresh on lr.m in the constructor, holds the loaded parameters. Permuted start: the same state dict, each tensor's entries permuted
    within that tensor (Python random.Random("sizing-span-perm|<birth seed>").shuffle over the flat indices, tensors in state-dict order),
    then loaded the same way. The padding row of the byte table is permuted like every other entry.
  * An instance CROSSES a cut when its key was read as input in an earlier chunk than the one holding the repeat target (stage 1's rule).
  * Windows of 250 chunks (the last window may be shorter); "chunk 0" is the first chunk's forward pass, before any update.
  * Accuracy for classes, windows and repeats to 50% counts every instance of that D, crossing or not; crossing and not-crossing figures are
    recorded beside it and given in the tables. "Repeats to 50%": the number of instances of that D read up to the end of the first window
    whose accuracy is at least 50% with at least 100 instances (None if never).
  * Classes on the last tenth of the stream (the instances whose repeat byte lies in the last 10% of the stream's bytes): pass (>= 100
    instances and >= 50% right), partial (>= 10%, below 50%), chance (below 10%); a run that passed in an earlier window and does not pass in
    the last tenth is "not held" and counts as not passing; fewer than 100 instances at >= 50% is "too few" and does not pass.
  * Block span: the largest D such that it and every smaller D of that block pass in at least two of three seeds; a D that passes in exactly
    one seed is "mixed". Distances where the block span stops are given with their pass counts.
"""
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_crosscut as C    # noqa: E402
import sz_learner as L     # noqa: E402

OUT = HERE / "runs" / "span"
MOD = HERE.parents[1] / "MOD97_LOCAL_RULE_20261106"
SEEDS = (1104, 1105, 1106)
BIRTH_OF = {1104: 73129, 1105: 93683, 1106: 116089}
M_SET = (4, 12, 24, 40)
WINDOW = 250
BLOCKS = dict(A=("random", (4, 12, 16, 18, 20, 24, 40, 60)), B=("random", ("M",)), C=("born", (4, 12, 24, 40)),
              P=("permuted", (12, 24)), D=("born", ("M",)))
FILL = C.FILL


def chunks_for(dist):
    return 20000 if dist == "M" else 235 * (int(dist) + 1)


# ---------------------------------------------------------------- streams
def mixed_stream(ds, n_bytes, seed):
    """Episodes K <filler of D - 2> = K \\n with D drawn per episode from ds. Returns bytes and [(key pos, repeat pos, key, D)]."""
    ds = tuple(sorted(ds)); label = ",".join(str(d) for d in ds)
    r = random.Random(f"sizing-crosscut|{label}|{seed}")
    rd = random.Random(f"sizing-span-D|{label}|{seed}")
    out = bytearray(); keys = []
    while len(out) < n_bytes:
        D = rd.choice(ds)
        k = 65 + r.randrange(24)
        keys.append((len(out), len(out) + D, k, D))
        out += bytes([k]) + bytes(r.choice(FILL) for _ in range(D - 2)) + b"=" + bytes([k]) + b"\n"
    return bytes(out), keys


def stream_for(dist, seed, chunks):
    n = chunks * L.CHUNK
    if dist == "M":
        data, keys = mixed_stream(M_SET, n, seed)
    else:
        data, k3 = C.fixture_stream(int(dist), n, seed)            # stage 1's generator, unedited
        keys = [(p, r, k, int(dist)) for p, r, k in k3]
    data = data[:n]
    return data, [k for k in keys if k[1] < len(data)]


# ---------------------------------------------------------------- starts
def birth_state(seed):
    pt = MOD / "exp45" / "e26" / f"e26_dev_B_{BIRTH_OF[seed]}.pt"
    return T.load(pt, map_location="cpu", weights_only=True), pt.name


def permuted(sd, birth):
    r = random.Random(f"sizing-span-perm|{birth}")
    out = {}
    for name, t in sd.items():
        idx = list(range(t.numel())); r.shuffle(idx)
        out[name] = t.flatten()[T.tensor(idx)].reshape(t.shape).clone()
    return out


def make_learner(start, seed, data):
    lr = L.StreamLearner(seed, data, [])
    info = dict(start=start, model_seed=seed)
    if start in ("born", "permuted"):
        sd, name = birth_state(seed)
        if start == "permuted":
            sd = permuted(sd, BIRTH_OF[seed])
        res = lr.m.load_state_dict(sd, strict=True)                 # in place: the optimizer holds these very parameters
        assert not res.missing_keys and not res.unexpected_keys
        opt_params = {id(p) for g in lr.opt.param_groups for p in g["params"]}
        assert opt_params == {id(p) for p in lr.m.parameters()} and not lr.opt.state
        info.update(birth=BIRTH_OF[seed], birth_file=name, permuted=start == "permuted")
    return lr, info


# ---------------------------------------------------------------- one run
def run(block, dist, seed, chunks=None, quarters=False):
    T.set_num_threads(1)
    start = BLOCKS[block][0] if block in BLOCKS else "random"
    chunks = chunks or chunks_for(dist)
    t0 = time.process_time()
    data, keys = stream_for(dist, seed, chunks)
    t_gen = time.process_time() - t0
    rep = {r: (p, k, D) for p, r, k, D in keys}
    Ds = sorted({D for *_, D in keys})
    lr, info = make_learner(start, seed, data)
    n = len(data); tenth_lo = n - n // 10
    win = {}                       # window index -> D -> [cross_right, cross_n, not_right, not_n]
    chunk0 = {D: [0, 0, 0, 0] for D in Ds}; last = {D: [0, 0, 0, 0] for D in Ds}
    quart = [{D: [0, 0, 0, 0] for D in Ds} for _ in range(4)]
    state = dict(chunk=0)

    def observe(a, e, lg):
        ci = state["chunk"]; state["chunk"] += 1
        pred = lg.argmax(1).tolist()
        w = win.setdefault(ci // WINDOW, {D: [0, 0, 0, 0] for D in Ds})
        for r in range(a, e):
            if r in rep:
                p, k, D = rep[r]
                right = pred[r - a] == k; cross = p + 1 < a
                j = 0 if cross else 2
                for acc in (w[D], chunk0[D] if ci == 0 else None, last[D] if r >= tenth_lo else None, quart[min(3, r * 4 // n)][D] if quarters else None):
                    if acc is not None:
                        acc[j] += right; acc[j + 1] += 1
    lr.observe = observe
    lr.advance_to(n)
    cpu_train = time.process_time() - t0 - t_gen
    # windows, repeats so far, repeats to 50%, earlier pass
    windows = []; seen = {D: 0 for D in Ds}; to50 = {D: None for D in Ds}; passed_before = {D: False for D in Ds}
    for wi in sorted(win):
        row = dict(window=wi + 1, chunks=[wi * WINDOW, min(chunks, (wi + 1) * WINDOW)], by_D={})
        for D in Ds:
            cr, cn, nr, nn = win[wi][D]; seen[D] += cn + nn
            acc = (cr + nr) / (cn + nn) if cn + nn else None
            if to50[D] is None and acc is not None and cn + nn >= 100 and acc >= 0.5:
                to50[D] = seen[D]
            if acc is not None and cn + nn >= 100 and acc >= 0.5 and wi < len(win) - 1:
                passed_before[D] = True
            row["by_D"][str(D)] = dict(crossing_right=cr, crossing=cn, not_crossing_right=nr, not_crossing=nn, repeats_so_far=seen[D])
        windows.append(row)
    classes = {}
    for D in Ds:
        cr, cn, nr, nn = last[D]; tot = cn + nn; acc = (cr + nr) / tot if tot else 0.0
        cls = ("pass" if tot >= 100 else "too few") if acc >= 0.5 else ("partial" if acc >= 0.1 else "chance")
        if cls != "pass" and passed_before[D]:
            cls = "not held"
        classes[str(D)] = dict(cls=cls, last_tenth=dict(right=cr + nr, n=tot, accuracy=acc, crossing_right=cr, crossing=cn,
                                                        not_crossing_right=nr, not_crossing=nn),
                               repeats_to_50=to50[D], repeats_total=seen[D], chunk0=dict(zip(("crossing_right", "crossing", "not_crossing_right", "not_crossing"), chunk0[D])))
    out = dict(brief="OPUS_TASK_SIZING_SPAN_20261113.md rev. 2 (06cf6b0e)", block=block, distance=dist, seed=seed, chunks=chunks, bytes=n,
               updates=lr.updates, start=info, distances=Ds, classes=classes, windows=windows,
               instances_per_D={str(D): sum(1 for *_, d in keys if d == D) for D in Ds},
               cpu=dict(generation=round(t_gen, 2), training=round(cpu_train, 1), per_chunk=cpu_train / lr.updates),
               python=platform.python_version(), torch=T.__version__, machine=platform.machine())
    if quarters:
        out["quarters"] = [{str(D): q[D] for D in Ds} for q in quart]
    return out


# ---------------------------------------------------------------- admission (items 1, 2, 3, 5)
def admit():
    rec = dict(item1=dict(python=platform.python_version(), torch=T.__version__, machine=platform.machine(), platform=platform.platform()))
    print("item 1:", rec["item1"], flush=True)
    # item 2: fixed 12, seed 1104, 3,000 chunks, last quarter against stage 1's record
    s1 = json.loads((HERE / "runs" / "crosscut_diag_D12.json").read_text())["by_quarter"][-1]
    r = run("A", 12, 1104, chunks=3000, quarters=True)
    cr, cn, nr, nn = r["quarters"][-1]["12"]
    same = (cr, cn, nr, nn) == (s1["crossing_right"], s1["crossing"], s1["not_crossing_right"], s1["not_crossing"])
    acc = (cr + nr) / (cn + nn)
    rec["item2"] = dict(last_quarter=dict(crossing_right=cr, crossing=cn, not_crossing_right=nr, not_crossing=nn),
                        stage1=s1, equal_to_stage1=same, accuracy=acc, cpu_per_chunk=r["cpu"]["per_chunk"], cpu_training=r["cpu"]["training"],
                        rule="same host class as stage 1 (Linux x86_64, torch 2.15.0+cpu): counts must equal; another class: accuracy >= 95%",
                        passed=same or (acc >= 0.95 and platform.machine() != "x86_64"))
    print("item 2:", rec["item2"]["last_quarter"], "equal to stage 1:", same, flush=True)
    # item 3: the mixed stream
    data, keys = mixed_stream(M_SET, 20000 * L.CHUNK, 1104)
    ok_rep = all(data[r] == k and data[p] == k and r - p == D for p, r, k, D in keys if r < len(data))
    per_D = {D: sum(1 for *_, d in keys if d == D) for D in M_SET}
    one = {}
    for D in (12, 24):
        a, _ = mixed_stream((D,), 3000 * L.CHUNK, 1104); b, _ = C.fixture_stream(D, 3000 * L.CHUNK, 1104)
        one[str(D)] = a == b
    rec["item3"] = dict(every_repeat_equals_its_key_at_its_D=ok_rep, episodes_per_D=per_D, one_element_set_byte_identical_to_stage1=one,
                        passed=ok_rep and all(one.values()) and all(v > 0 for v in per_D.values()))
    print("item 3:", rec["item3"], flush=True)
    # item 5: the permuted start
    res = {}
    for seed in SEEDS:
        sd, _ = birth_state(seed); pm = permuted(sd, BIRTH_OF[seed])
        res[str(BIRTH_OF[seed])] = {n: dict(sorted_equal=bool(T.equal(T.sort(sd[n].flatten())[0], T.sort(pm[n].flatten())[0])),
                                            order_changed=not T.equal(sd[n], pm[n]),
                                            share_of_entries_moved=float((sd[n] != pm[n]).float().mean())) for n in sd}
    rec["item5"] = dict(tensors=res, passed=all(v["sorted_equal"] and v["order_changed"] for b in res.values() for v in b.values()))
    print("item 5:", rec["item5"]["passed"], flush=True)
    vb = json.loads((OUT / "verify_born.json").read_text())
    rec["item4"] = dict(file="runs/span/verify_born.json", passed=vb["passed"],
                        max_abs_logit_difference={k: v["max_abs_logit_difference"] for k, v in vb["births"].items()})
    rec["admitted"] = all(rec[f"item{i}"].get("passed", True) for i in (2, 3, 4, 5))
    # forecast of the roster's CPU from item 2's measured cost per chunk
    total = sum(chunks_for(d) for b, (_, ds) in BLOCKS.items() for d in ds for _ in SEEDS)
    pc = r["cpu"]["per_chunk"]
    rec["cost_forecast"] = dict(runs=sum(len(ds) for _, ds in BLOCKS.values()) * 3, chunks=total, cpu_per_chunk=pc, cpu_hours=total * pc / 3600,
                                wall_hours_at_three_processes=total * pc / 3600 / 3, cap_cpu_hours=12, within_cap=total * pc / 3600 <= 12)
    print("cost forecast:", rec["cost_forecast"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "admission.json").write_text(json.dumps(rec, indent=1))
    return rec


# ---------------------------------------------------------------- tables
def tables():
    res = {}
    for f in sorted(OUT.glob("[ABCDP]_*_*.json")):
        r = json.loads(f.read_text()); res[(r["block"], str(r["distance"]), r["seed"])] = r
    out = dict(blocks={}, spans={}, cpu={})
    for b, (start, dists) in BLOCKS.items():
        rows = {}
        for d in dists:
            for s in SEEDS:
                r = res.get((b, str(d), s))
                if r is None:
                    continue
                for D, c in r["classes"].items():
                    rows.setdefault(D, {})[s] = dict(cls=c["cls"], last_tenth=f"{c['last_tenth']['right']}/{c['last_tenth']['n']}",
                                                     accuracy=round(c["last_tenth"]["accuracy"], 3),
                                                     crossing=f"{c['last_tenth']['crossing_right']}/{c['last_tenth']['crossing']}",
                                                     repeats_to_50=c["repeats_to_50"], repeats_total=c["repeats_total"])
        out["blocks"][b] = dict(start=start, rows=rows)
        span = None; stop = None
        for D in sorted(rows, key=int):
            k = sum(v["cls"] == "pass" for v in rows[D].values())
            if k >= 2 and stop is None:
                span = int(D)
            elif stop is None:
                stop = dict(D=int(D), seeds_passing=k, reading="mixed" if k == 1 else "fails")
        out["spans"][b] = dict(span=span, stops_at=stop, passes_by_D={D: sum(v["cls"] == "pass" for v in rows[D].values()) for D in rows})
        out["cpu"][b] = round(sum(r["cpu"]["training"] + r["cpu"]["generation"] for (bb, _, _), r in res.items() if bb == b), 1)

    def passes(b, D):
        return sum(v["cls"] == "pass" for v in out["blocks"][b]["rows"].get(str(D), {}).values())

    def to50(b, D, s):
        return out["blocks"][b]["rows"].get(str(D), {}).get(s, {}).get("repeats_to_50")
    f4 = [D for D in (24, 40) if passes("B", D) >= 2 and passes("A", D) < 2]
    # F5 per pair: the born start needed more repeats to 50% than random weights. A born run that never reaches 50% while its random
    # partner does counts as more; a pair where random never reaches 50% cannot be scored (None).
    f5 = []
    for s in SEEDS:
        a, c = to50("A", 12, s), to50("C", 12, s)
        f5.append(None if a is None else (True if c is None else c > a))
    out["forecasts"] = dict(
        F1=dict(text="Block A passes at 18 in at least two of three seeds", held=passes("A", 18) >= 2),
        F2=dict(text="Block A passes at 24 in at least two of three", held=passes("A", 24) >= 2),
        F3=dict(text="Block A passes at 60 in at least two of three", held=passes("A", 60) >= 2),
        F4=dict(text="Block B passes at 24 or at 40 in at least two of three seeds where block A does not", held=bool(f4), distances=f4),
        F5=dict(text="At D = 12, repeats to 50% are more from the born start than from random weights in at least two of three pairs",
                pairs=[dict(seed=s, random=to50("A", 12, s), born=to50("C", 12, s), born_more=x) for s, x in zip(SEEDS, f5)],
                held=sum(x is True for x in f5) >= 2),
        F6=dict(text="Block C passes at 40 in at least two of three", held=passes("C", 40) >= 2))
    out["runs"] = len(res); out["cpu_total"] = round(sum(out["cpu"].values()), 1)
    (OUT / "SPAN_TABLES.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "admit":
        sys.exit(0 if admit()["admitted"] else 1)
    elif cmd == "run":
        block, dist, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        dist = dist if dist == "M" else int(dist)
        assert block in BLOCKS and dist in BLOCKS[block][1] and seed in SEEDS
        path = OUT / f"{block}_{dist}_{seed}.json"
        if path.exists():
            raise SystemExit(f"refused: {path.name} exists")
        r = run(block, dist, seed)
        path.write_text(json.dumps(r, indent=1))
        print(block, dist, seed, {D: (c["cls"], round(c["last_tenth"]["accuracy"], 3), c["repeats_to_50"]) for D, c in r["classes"].items()},
              "cpu", r["cpu"]["training"], flush=True)
    elif cmd == "tables":
        print(json.dumps(tables()["spans"], indent=1))
    else:
        print(__doc__)
