"""Verification of E57 (the sweep with a scale for each receiving unit, and with a code that is not decayed; seed 73129).

The worker's branch is claude/e50-fixed-input-code-017dSE; the report is read at commit 85417d06. The document is
E57_VERIFICATION_20261118.md (same folder); every figure of it is printed here with the number of its section.

    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e57_verify.py          (from the repository's root; about four minutes)

Read-only: every file is read with `git show` into memory, weight files one at a time; nothing is written and no file of the worker's
branch is executed. Standard library and numpy only. The network's forward and the gradients under each scale are those of
sweep_scale_variants.py (same folder; float64), the panels of items those of np_gru.py, the reading of weight files that of
compare_pt_tensors.py.

Sections
  1  completeness and settings of the nine runs; the commits and source hashes; the data digests; the check receipts; check 6
     recomputed; the CPU time
  2  every figure of the tables file, the summary, the report and the worker's message recomputed from the receipts' raw counts;
     the comparisons are counted and every difference is listed
  3  the saved weights: hashes, masks, the final counts by a forward pass in numpy, and the cosines and sizes of the last reading
  4  the brief's fixed wording run by run, the margins on which each word rests, the predictions and the table of explanations
  5  the figures of the reading (speed against back-propagation, levels against the networks with the whole code, the row scale)
  6  the committed text files: unfilled marks and names of language models (counts only)
"""
import sys
sys.dont_write_bytecode = True
import difflib, hashlib, json, os, re, subprocess, time
from random import Random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare_pt_tensors import tensors                     # noqa: E402
import sweep_scale_variants as V                           # noqa: E402
from np_gru import split as gsplit, panel, render as grender, DEV, BOS, EOS   # noqa: E402

T0 = time.time()
WREF, E49REF, E42REF = "85417d06", "a36de661", "44f91edc"
M = "MOD97_LOCAL_RULE_20261106/"
P = M + "exp45/"
SEED, U, D = 73129, 256, 512
DTYPE = {"FloatStorage": "<f4", "DoubleStorage": "<f8", "LongStorage": "<i8", "BoolStorage": "?"}
STATE = V.TENSORS + V.BIASES                               # the thirteen tensors of the network
SCALES = ("matrix", "row", "rowcells", "rowcol")
COS_T = ("book.weight", "l1.weight", "rnn.weight_ih_l0", "rnn.weight_hh_l0", "lm.weight", "l2.weight")

# (label, network, scale, rule, updates, code decay 0?)
RUNS = [("F10 row S", "F10", "row", "S", 24576, False), ("F10 rowcol S", "F10", "rowcol", "S", 24576, False),
        ("G row S", "G", "row", "S", 24576, False), ("in16 matrix S cd0", "in16", "matrix", "S", 49152, True),
        ("in16 matrix S", "in16", "matrix", "S", 49152, False), ("in16 none B", "in16", "none", "B", 24576, False),
        ("in16 row S", "in16", "row", "S", 49152, False), ("in16 rowcells S", "in16", "rowcells", "S", 49152, False),
        ("in16 rowcol S", "in16", "rowcol", "S", 49152, False)]
LABELS = [r[0] for r in RUNS]
SPEC = {r[0]: r for r in RUNS}


def stem(label):
    _, net, sc, rule, dose, cd0 = SPEC[label]
    return f"e57_{net}_{sc}_{rule}{'_cd0' if cd0 else ''}{'' if dose == 24576 else f'_u{dose}'}_{SEED}"


def show(path, ref=WREF):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True).stdout


def js(path, ref=WREF):
    return json.loads(show(path, ref))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_pt(path, ref=WREF):
    blob = show(path, ref)
    W = {n.lstrip("/"): np.frombuffer(raw, dtype=DTYPE[st]).reshape(shape).copy() for n, (st, shape, raw) in tensors(blob).items()}
    return W, sha(blob)


def tensors_sha(W):
    """exp20_reference.tensors_sha as read: per name in sorted order, the name, torch's dtype string, the shape, the bytes."""
    h = hashlib.sha256()
    for k in sorted(W):
        v = W[k]
        h.update(k.encode()); h.update(b"torch.float32" if v.dtype == np.float32 else b"torch.float64")
        h.update(str(tuple(v.shape)).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def clock(s):
    print(f"   [{s}: {time.time() - T0:.0f} s]", flush=True)


def fmt(v):
    return "not reached" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))


# ---------------------------------------------------------------- the receipts
REC = {lab: js(P + "e57/" + stem(lab) + ".json") for lab in LABELS}
REF = {"E42 G S": js(P + "e42/e42_G_S_73129.json", E42REF), "E42 G B": js(P + "e42/e42_G_B_73129.json", E42REF),
       "E42 F10 S": js(P + "e42/e42_F10_S_73129.json", E42REF), "E42 F10 B": js(P + "e42/e42_F10_B_73129.json", E42REF),
       "E49 in16 S": js(P + "e49/e49_in16_S_73129.json", E49REF), "E49 in16 B": js(P + "e49/e49_in16_B_73129.json", E49REF)}
CHK = {n: js(P + f"e57_qual/check{n}_73129.json") for n in "12345678"}


def curve_of(label, upto=None):
    c = (REC[label] if label in REC else REF[label])["curve"]
    return [r for r in c if upto is None or r["step"] <= upto]


def reach(curve, key, thr=80):
    """The brief's definition: the first of three consecutive scored rows at or above thr (the row of update 0 excluded)."""
    v = [(r["step"], r[key]) for r in curve if r["step"] > 0]
    for i in range(len(v) - 2):
        if all(x[1] >= thr for x in v[i:i + 3]):
            return v[i][0]
    return None


def level(curve, key, back=0):
    end = curve[-1]["step"] - back
    rows = [r[key] for r in curve if end - 512 <= r["step"] <= end]
    assert len(rows) == 3
    return float(np.mean(rows))


def figures(curve):
    f = dict(updates=curve[-1]["step"], A80=reach(curve, "A"), R80=reach(curve, "R"), T80=reach(curve, "taught"))
    for k in ("A", "R", "taught", "len12", "len40"):
        f["lv_" + k] = level(curve, k)
    f["rise"] = level(curve, "R") - level(curve, "R", back=2048)
    f["word"] = word(f)
    f["loss_last"] = curve[-1]["loss"]
    return f


def word(f):
    if f["A80"] is not None and f["R80"] is not None and f["lv_A"] >= 80 and f["lv_R"] >= 80:
        return "learns the family"
    return "partly" if f["lv_taught"] >= 50 else "not learned"


def describe(xr, xc, Dr, Dc):
    """e42_tables.describe at 44f91edc, lines 63 to 72, as read."""
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


FIG = {}
for lab in LABELS:
    FIG[lab] = figures(curve_of(lab))
    if SPEC[lab][4] == 49152:
        FIG[lab + " (first 24,576)"] = figures(curve_of(lab, 24576))
for lab in REF:
    FIG[lab + " (reference)"] = figures(curve_of(lab))
    FIG[lab] = FIG[lab + " (reference)"]

# ---------------------------------------------------------------- comparisons
COMP = []          # (section, source, label, reported, recomputed, kind)


def agree(rep, mine, kind):
    if kind == "word" or rep is None or mine is None or isinstance(rep, str):
        return rep == mine
    if kind == "int":
        return int(rep) == int(mine)
    if kind == "full":
        return abs(rep - mine) <= 1e-9 * max(1.0, abs(mine))
    d = int(kind)                                            # decimals as printed
    return abs(rep - mine) <= 0.5 * 10 ** -d + 1e-9


def cmp(sec, src, label, rep, mine, kind):
    COMP.append((sec, src, label, rep, mine, kind, agree(rep, mine, kind)))


def num(s):
    s = s.strip().replace(",", "").replace("−", "-").replace("+", "")
    if s in ("not reached", "None", ""):
        return None
    return float(s)


def decs(s):
    s = s.strip().replace(",", "")
    return len(s.split(".")[1]) if "." in s else "int"


# ================================================================ section 1
def section1():
    print("\n===== 1. Completeness and settings of the nine runs =====")
    runner = show(M + "exp57_scale.py")
    runner_sha = sha(runner)
    print(f"1.1 runner exp57_scale.py at {WREF}: SHA-256 {runner_sha[:16]}")
    versions = {}
    for c in ("c55aaa1b", "66e20486", "27e743ae", "fb07cb0f", WREF):
        versions[c] = show(M + "exp57_scale.py", c)
        print(f"    the runner at {c}: SHA-256 {sha(versions[c])[:16]}")
    a, b = versions["c55aaa1b"].decode().splitlines(), versions[WREF].decode().splitlines()
    dl = [ln for ln in difflib.unified_diff(a, b, lineterm="", n=0) if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    print(f"    lines that differ between the runner at c55aaa1b (checks 2 and 3) and at {WREF} (the runs): {len(dl)}; inside the function check8 only: "
          f"{all(('check8' in x) or ('env' in x.lower()) or x.strip() in ('+', '-') for x in dl)}")
    for x in dl:
        print("      " + x[:150])
    for f, h in (("exp49_interface.py", "cdf93ce167d562c8a3765c1ea4f98f74623469492324a85aa1eb34216b7ccbf3"),
                 ("e49_interface.py", "92f968e092f2d27eaff4ee7e93ecbbb82b053cc88b6a2d3f967adadf7e3791f1")):
        print(f"    {f}: at {WREF} {sha(show(M + f)) == h}, at E49's {E49REF} {sha(show(M + f, E49REF)) == h}")
    src_ok = {}
    first = REC[LABELS[0]]["provenance"]["sources_sha256"]
    for s, h in first.items():
        src_ok[s] = sha(show(M + s)) == h
    print(f"    sources_sha256 of the receipts against the files at {WREF}: all equal {all(src_ok.values())} ({len(src_ok)} files)"
          + ("" if all(src_ok.values()) else f"; differing {[s for s, v in src_ok.items() if not v]}"))

    print("1.2 the nine receipts")
    print("    run                | net  scale    rule dose   code_decay rows  steps  complete resumes | runner = committed | commit    | no-decay group")
    ok_all = True
    for lab in LABELS:
        r = REC[lab]; _, net, sc, rule, dose, cd0 = SPEC[lab]; c = r["curve"]
        pv = r["provenance"]
        groups = {g["weight_decay"]: sorted(g["params"]) for g in r["optimizer"]["groups"]}
        nod = groups[0.0]
        want_nod = sorted(["book.weight"] + (["l1.weight", "l1.bias"] if cd0 else []))
        checks = dict(net=r["setting"] == net and r["config"] == net, scale=r["scale"] == sc, rule=r["rule"] == rule, seed=r["seed"] == SEED,
                      dose=r["dose"] == dose, decay=r["code_decay"] == (0 if cd0 else 0.25),
                      scale_at=r["scale_at"] == dict(recurrent_layer={"matrix": "matrix", "none": "matrix", "row": "row", "rowcells": "row", "rowcol": "rowcol"}[sc],
                                                     other_maps={"row": "row"}.get(sc, "matrix")),
                      rows=len(c) == dose // 256 + 1 and [x["step"] for x in c] == list(range(0, dose + 1, 256)),
                      steps=r["optimizer_steps"] == [dose], complete=r["complete"] is True, resumes=r["resumes"] == [],
                      groups=nod == want_nod and len(groups[0.25]) == 13 - len(want_nod), read_back=r["optimizer_read_back"] == r["optimizer"],
                      lr=all(g["lr"] == 0.001 and g["betas"] == [0.9, 0.98] for g in r["optimizer"]["groups"]),
                      finite=all(x["finite_losses"] for x in c) and all(np.isfinite(x["loss"]) for x in c[1:]),
                      masks=all(all(v["outside_zero"] and v["averages_zero"] for v in x["masks"].values()) for x in c),
                      limit=all(all(x["limit"].values()) for x in c), runner=pv["runner_sha256"] == runner_sha,
                      unmodified=pv["git"]["sources_modified_vs_commit"] == [], resume_flag="--resume" in pv["command"],
                      ma=r["multiply_adds_per_position"] == {"in16": 549632, "F10": 930560, "G": 1118720}[net],
                      final_digest=r["data_digest"] == c[-1]["data_digest"], cosines=[e["update"] for e in r["cosines"]] == list(range(0, dose + 1, 2048)),
                      threads=pv["versions"]["torch_threads"] == 1 and pv["versions"]["torch"] == "2.15.0+cpu")
        ok_all &= all(checks.values())
        print(f"    {lab:18s} | {net:4s} {sc:8s} {rule}    {dose:6d} {r['code_decay']:<10} {len(c):4d} {r['optimizer_steps'][0]:6d} {str(r['complete']):8s} {len(r['resumes'])}       "
              f"| {pv['runner_sha256'] == runner_sha!s:18s} | {pv['git']['commit'][:8]} | {nod}")
        bad = [k for k, v in checks.items() if not v]
        if bad:
            print(f"      NOT as the brief: {bad}")
    print(f"    every setting of every run as the brief requires (network, scale, scale_at, rule, seed, updates, code decay, optimizer groups, rate and betas,"
          f" read back, finite losses, masks and E42's limit at every row, rows every 256 updates, readings every 2,048, no resumption, runner, threads): {ok_all}")
    starts = {lab: REC[lab]["start_tensors_sha256"][:16] for lab in LABELS}
    print(f"    start_tensors_sha256: " + "; ".join(f"{k} {v}" for k, v in starts.items()))
    print(f"    E49's in16 start {REF['E49 in16 S']['start_tensors_sha256'][:16]}; E42's G start {REF['E42 G S'].get('start_tensors_sha256', '?')[:16]}, F10 start {REF['E42 F10 S'].get('start_tensors_sha256', '?')[:16]}")
    print(f"    the logs: one start line in each run's log: " + ", ".join(
        f"{lab} {show(P + 'e57/logs/' + stem(lab) + '.log').decode().count('least cells reading')}" for lab in LABELS))
    q = show(P + "e57/logs/queue.log").decode()
    print(f"    queue.log: {q.count(' exit=0')} runs ended with exit 0 of {q.count(' end ')}; first line {q.splitlines()[0][:20]}, last {q.splitlines()[-1][:20]}")

    print("1.3 the data digests (the stream)")
    mine = V.stream_digest(4096)
    at = {lab: [x for x in REC[lab]["curve"] if x["step"] == 4096][0]["data_digest"]["sha256"] for lab in LABELS}
    print(f"    the digest of the first 4,096 batches computed here {mine[:16]}; equal to every run's row of update 4,096: {all(v == mine for v in at.values())}")
    for lab in LABELS:
        net = SPEC[lab][1]
        refs = {"in16": ["E49 in16 S", "E49 in16 B"], "F10": ["E42 F10 S", "E42 F10 B"], "G": ["E42 G S", "E42 G B"]}[net]
        res = []
        for rf in refs:
            rd = {x["step"]: x["data_digest"] for x in REF[rf]["curve"]}
            common = [x for x in REC[lab]["curve"] if x["step"] in rd]
            res.append(f"{rf}: {sum(x['data_digest'] == rd[x['step']] for x in common)} of {len(common)} rows equal")
        print(f"    {lab:18s} " + "; ".join(res))
    long = [lab for lab in LABELS if SPEC[lab][4] == 49152]
    eq = all(REC[lab]["curve"][i]["data_digest"] == REC[long[0]]["curve"][i]["data_digest"] for lab in long for i in range(193))
    print(f"    the five runs of 49,152 updates have the same digest at all 193 scored rows: {eq}; final digest {REC[long[0]]['data_digest']['sha256'][:16]}")

    print("1.4 the check receipts")
    c1 = CHK["1"]
    print(f"    check 1: (a) all equal {c1['a_all_equal']} ({len(c1['a'])} comparisons, worst relative difference {c1['a_worst_relative_difference']});"
          f" (b) {c1['b_all_hold']}: G {c1['b']['G_2048']['tensors_sha256'][:8]} rows equal {c1['b']['G_2048']['rows']['equal']};"
          f" F10 {c1['b']['F10_2048']['start'][:8]}/{c1['b']['F10_2048']['end'][:8]}; in16 256 {c1['b']['in16_256']['start'][:8]}/{c1['b']['in16_256']['end'][:8]};"
          f" in16 2,048 rows equal {c1['b']['in16_2048']['rows']['equal']}; rule (c) {c1['rule_c']}; pass {c1['pass']}; commit {c1['provenance']['git']['commit'][:8]}")
    brief1 = dict(G="79618ab29390e6ecfdacd49a955afff7a70f282466dd92fd14e6ef3ba1ddd19f", F10s="dede71d76cb8a9e6af09aa5394eb90c69a45177f6d3dc1861cce0c3b7e35a561",
                  F10e="837dadc12e8b1bd702c5d602b80e3fdca82eceda478dd110b480aff57815ea75", i0="657e185828eb7a4c6e91a3c3a0bae71d6d0c8d38cd84840a66c3924834ff45da",
                  i1="88cad76a4806601be9a859832b9167b95a967fb3cac4a2f7ee8de8b641cb6f34")
    got1 = dict(G=c1["b"]["G_2048"]["tensors_sha256"], F10s=c1["b"]["F10_2048"]["start"], F10e=c1["b"]["F10_2048"]["end"], i0=c1["b"]["in16_256"]["start"], i1=c1["b"]["in16_256"]["end"])
    dry = {}
    for nm, pth in (("G", "check1_73129_runs/e57_G_matrix_S_u2048_73129.json"), ("F10", "check1_73129_runs/e57_F10_matrix_S_u2048_73129.json"),
                    ("in16_256", "check1_73129_runs/e57_in16_matrix_S_u256_73129.json")):
        dry[nm] = js(P + "e57_qual/" + pth)
    own = dict(G=dry["G"]["tensors_sha256"], F10s=dry["F10"]["start_tensors_sha256"], F10e=dry["F10"]["tensors_sha256"], i0=dry["in16_256"]["start_tensors_sha256"], i1=dry["in16_256"]["tensors_sha256"])
    print(f"      the five hashes of the brief equal those recorded in check 1's receipt {got1 == brief1} and those of the dry runs' own receipts {own == brief1}")
    c2 = CHK["2"]
    worst2 = max(v["max_diff"] for f in c2["files"].values() for s, v in f.items() if s != "exact")
    ex = c2["files"]["e49_in16_S_73129.pt"]["exact"]["cosines"]
    print(f"    check 2: pass {c2['pass']}; largest difference from the brief's figures {worst2:.5f}; exact: {len(ex)} tensors, least cosine {min(ex.values()):.15f}; commit {c2['provenance']['git']['commit'][:8]}")
    c3 = CHK["3"]
    print(f"    check 3: {c3['tensors_sha256'][:8]} equal to c0eb6548 {c3['tensors_sha256'].startswith('c0eb6548')}; rows {c3['rows']['rows']} equal {c3['rows']['equal']}; pass {c3['pass']}; commit {c3['provenance']['git']['commit'][:8]}")
    c4 = CHK["4"]
    print(f"    check 4: pass {c4['pass']}; " + ", ".join(f"{k} {v['rows']}" for k, v in c4["runs"].items()))
    c5 = CHK["5"]
    print(f"    check 5: forecast {c5['forecast_nine_runs_cpu_h']:.2f} CPU-hours (limit {c5['limit_cpu_h']}); CPU-seconds an update {min(c5['cpu_per_update'].values()):.4f} to {max(c5['cpu_per_update'].values()):.4f};"
          f" a row {c5['cpu_per_row']:.2f}; a set of cosines {c5['cpu_per_cosine_set']:.2f}; pass {c5['pass']}")
    c7 = CHK["7"]
    print(f"    check 7: (a) eleven equal {c7['a']['all_eleven_equal']}, l1.weight largest deviation {c7['a']['l1']['l1.weight']['max_dev']:.3e}, l1.bias {c7['a']['l1']['l1.bias']['max_dev']:.3e};"
          f" (b) no-decay group {c7['b']['group_no_decay']}, start {c7['b']['start_equal']}, row 0 {c7['b']['row0_counts_equal']}, digest 256 {c7['b']['digest256_equal']}, tensors differ {c7['b']['tensors_differ']};"
          f" (c) l1 mean |w|: start {c7['c']['cd0']['start']:.4f}, end without decay {c7['c']['cd0']['end']:.4f}, with decay {c7['c']['reference']['end']:.4f}; pass {c7['pass']}")
    c8 = CHK["8"]
    print(f"    check 8: before resuming exit {c8['before_resume']['exit_code']}, last row {c8['before_resume']['last_row']}, readings {c8['before_resume']['cosines']};"
          f" tensors equal {c8['tensors_equal']} ({c8['tensors_sha256'][0][:16]}); rows equal {c8['rows_equal']}; readings {c8['cosines_entries']} equal {c8['cosines_equal']};"
          f" steps {c8['optimizer_steps']}; CPU ratio {c8['cpu_updates_last_over_4096']:.2f}; resumes {[len(x) for x in c8['resumes']]}; pass {c8['pass']}")
    k8 = {}
    for sub in ("uninterrupted", "killed_and_resumed"):
        k8[sub] = js(P + f"e57_qual/check8_73129_runs/{sub}/e57_in16_matrix_S_cd0_u8192_73129.json")
    same = [x["step"] for x, y in zip(k8["uninterrupted"]["curve"], k8["killed_and_resumed"]["curve"])
            if {k: v for k, v in x.items() if not k.startswith(("cpu", "wall"))} != {k: v for k, v in y.items() if not k.startswith(("cpu", "wall"))}]
    print(f"      from the two dry runs' receipts: tensors {k8['uninterrupted']['tensors_sha256'] == k8['killed_and_resumed']['tensors_sha256']}, rows differing outside cpu and wall {same},"
          f" readings equal {[ {k: v for k, v in e.items() if k != 'cpu'} for e in k8['uninterrupted']['cosines']] == [ {k: v for k, v in e.items() if k != 'cpu'} for e in k8['killed_and_resumed']['cosines']]}")
    first_attempt = show(P + "e57_qual/interrupted/check8_first_attempt/check8_killed.log").decode()
    print(f"    check 8's first attempt: its subprocess stopped on the harness's environment gate ('AssertionError' in its log: {'AssertionError' in first_attempt});"
          f" the parent stopped with FileNotFoundError: {'FileNotFoundError' in show(P + 'e57_qual/interrupted/check8_first_attempt/check8.log').decode()}")

    print("1.5 check 6 recomputed from the receipts")
    def rows_cmp(a, b, upto):
        A_ = [x for x in a if x["step"] <= upto]; B_ = [x for x in b if x["step"] <= upto]
        keys = sorted(set(A_[0]) & set(B_[0]))
        keys = [k for k in keys if not k.startswith(("cpu", "wall"))]
        diff = [(x["step"], k) for x, y in zip(A_, B_) for k in keys if x[k] != y[k]]
        return len(A_), len(B_), keys, diff
    n1, n2, keys, diff = rows_cmp(REC["in16 matrix S"]["curve"], REF["E49 in16 S"]["curve"], 24576)
    print(f"    in16 matrix S against E49's run under the sweep, rows to 24,576: {n1} and {n2}; fields compared {keys}; differences {len(diff)} {diff[:5]}")
    n1, n2, keys, diff = rows_cmp(REC["in16 none B"]["curve"], REF["E49 in16 B"]["curve"], 24576)
    print(f"    in16 none B against E49's run under back-propagation: {n1} and {n2} rows; differences {len(diff)} {diff[:5]};"
          f" tensors_sha256 {REC['in16 none B']['tensors_sha256'][:16]} = E49's {REF['E49 in16 B']['tensors_sha256'][:16]} (= 707c8569 of the brief): "
          f"{REC['in16 none B']['tensors_sha256'] == REF['E49 in16 B']['tensors_sha256'] == '707c85693a98b6b1017ba94abb0754b12758650d8a0d6562d44a28363584ac3e'}")
    brief2 = {"S": {"matrix": (0.367, 0.298, 0.603, 0.651, 0.606, 0.643), "row": (0.697, 0.624, 0.676, 0.762, 0.694, 0.710)},
              "B": {"matrix": (0.442, 0.309, 0.523, 0.582, 0.700, 0.706), "row": (0.405, 0.363, 0.567, 0.631, 0.700, 0.707)}}
    for lab, k, f in (("in16 matrix S", "S", "e49_in16_S_73129.pt"), ("in16 none B", "B", "e49_in16_B_73129.pt")):
        e = [x for x in REC[lab]["cosines"] if x["update"] == 24576][0]["cosines"]
        d = max(abs(e[s][t] - v) for s in ("matrix", "row") for t, v in zip(COS_T, brief2[k][s]))
        same2 = all(e[s][t] == g for s in ("matrix", "row") for t, g in zip(COS_T, c2["files"][f][s]["got"]))
        print(f"    {lab}: reading of 24,576 against the brief's figures of check 2, largest difference {d:.5f} (within 0.002: {d <= 0.002}); equal bit for bit to check 2's values {same2}")

    print("1.6 CPU time")
    run_h = sum(REC[l]["cpu_updates"] + REC[l]["cpu_scoring"] + REC[l]["cpu_cosines"] for l in LABELS) / 3600
    cos_h = sum(REC[l]["cpu_cosines"] for l in LABELS) / 3600
    chk_h = sum(CHK[n].get("cpu", 0.0) for n in "12345678") / 3600
    d8 = sum(k8[s]["cpu_updates"] + k8[s]["cpu_scoring"] + k8[s]["cpu_cosines"] for s in k8) / 3600
    kl = [ln for ln in first_attempt.splitlines()]
    killed = show(P + "e57_qual/check8_killed.log").decode().splitlines()
    last = [ln for ln in killed if " cpu " in ln][-1]
    cu = [float(x) for x in re.search(r"cpu (\d+)\+(\d+)\+(\d+)s", last).groups()]
    r4096 = [x for x in k8["killed_and_resumed"]["curve"] if x["step"] == 4096][0]
    lost = (sum(cu) - (r4096["cpu_updates"] + r4096["cpu_scoring"] + r4096["cpu_cosines"])) / 3600
    print(f"    the nine runs: {run_h:.2f} CPU-hours (updates, scoring and readings), of which the readings {cos_h:.2f}")
    print(f"    the check receipts' own 'cpu' entries: {chk_h:.2f} CPU-hours; check 8's two dry runs (subprocesses, not in check 8's entry): {d8:.2f};"
          f" the killed process's updates after its saved state of 4,096 (to its last row {last.split()[5]}): at least {lost:.2f}; together at least {chk_h + d8 + lost:.2f}")
    print(f"    wall time of the queue: {q.splitlines()[0][:20]} to {q.splitlines()[-1][:20]}")
    clock("section 1")
    return run_h, cos_h, chk_h


# ================================================================ section 2
def parse_table(text, header_start):
    """Rows (lists of cells) of the first markdown table whose header line starts with header_start."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(header_start))
    out = []
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        out.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return out


def cosine_sections(text):
    """{label: [rows]} for every '### label' followed by a table of readings."""
    res, lines = {}, text.splitlines()
    for k, ln in enumerate(lines):
        if ln.startswith("### ") and k + 2 < len(lines) and lines[k + 2].startswith("| Update | matrix"):
            lab = ln[4:].strip(); rows = []
            for r in lines[k + 4:]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip().strip("|").split("|")])
            res[lab] = rows
    return res


def main_table(sec, src, text):
    for row in parse_table(text, "| Run | Updates"):
        lab = row[0].replace("(reference)", "(reference)")
        f = FIG[lab]
        cmp(sec, src, lab + ": updates", num(row[1]), f["updates"], "int")
        for k, cell in zip(("A80", "R80", "T80"), row[2:5]):
            cmp(sec, src, f"{lab}: {k}", num(cell), f[k], "int")
        for k, cell in zip(("lv_A", "lv_R", "lv_taught", "lv_len40", "rise"), row[5:10]):
            cmp(sec, src, f"{lab}: {k}", num(cell), f[k], decs(cell))
        cmp(sec, src, f"{lab}: word", row[10], f["word"], "word")
        base = lab.replace(" (first 24,576)", "")
        if base in REC:
            c = curve_of(base, 24576 if "first" in lab else None)
            mk = all(all(v["outside_zero"] and v["averages_zero"] for v in x["masks"].values()) for x in c)
            cmp(sec, src, f"{lab}: masks", row[11], str(mk), "word")
        elif base.startswith("E49"):
            c = curve_of(base.replace(" (reference)", ""))
            cmp(sec, src, f"{lab}: masks", row[11], str(all(all(v["outside_zero"] and v["averages_zero"] for v in x["masks"].values()) for x in c)), "word")


DOSE = lambda lab: FIG[lab]["updates"]


def describe_table(sec, src, text):
    for row in parse_table(text, "| Candidate | Reference"):
        cand, ref, score = row[0], row[1], row[2]
        fc, fr = FIG[cand], FIG[ref]
        cmp(sec, src, f"{cand} vs {ref} {score}: reference's", num(row[3]), fr[score], "int")
        cmp(sec, src, f"{cand} vs {ref} {score}: candidate's", num(row[4]), fc[score], "int")
        cmp(sec, src, f"{cand} vs {ref} {score}: word", row[5], describe(fr[score], fc[score], fr["updates"] - 512, fc["updates"] - 512), "word")


def cos_tables(sec, src, text):
    secs = cosine_sections(text)
    for lab, rows in secs.items():
        rc = REC[lab]; byu = {e["update"]: e for e in rc["cosines"]}; loss = {x["step"]: x["loss"] for x in rc["curve"]}
        for row in rows:
            u = int(num(row[0])); e = byu[u]
            for si, s in enumerate(SCALES):
                for t, cell in zip(COS_T, row[1 + si].split(",")):
                    cmp(sec, src, f"{lab} {u} {s} {t}", num(cell), e["cosines"][s][t], decs(cell))
            cmp(sec, src, f"{lab} {u} l1 mean abs", num(row[5]), e["l1_weight_mean_abs"], decs(row[5]))
            cmp(sec, src, f"{lab} {u} code mean abs", num(row[6]), e["code_mean_abs"], decs(row[6]))
            if row[7]:
                cmp(sec, src, f"{lab} {u} loss of the row", num(row[7]), loss[u], decs(row[7]))
            else:
                cmp(sec, src, f"{lab} {u} loss of the row (empty)", None, loss[u], "word")
    return len(secs)


def sizes_table(sec, src, text):
    a = {e["update"]: e for e in REC["in16 matrix S"]["cosines"]}; b = {e["update"]: e for e in REC["in16 matrix S cd0"]["cosines"]}
    for row in parse_table(text, "| Update | l1 mean abs (decayed)"):
        u = int(num(row[0]))
        for cell, v in zip(row[1:5], (a[u]["l1_weight_mean_abs"], b[u]["l1_weight_mean_abs"], a[u]["code_mean_abs"], b[u]["code_mean_abs"])):
            cmp(sec, src, f"sizes {u}", num(cell), v, decs(cell))


def summary_json(sec, src, S):
    def runfig(lab, d):
        f = FIG[lab]
        for k in ("updates", "A80", "R80", "T80"):
            cmp(sec, src, f"{lab}: {k}", d[k], f[k], "int")
        for k, kk in (("A", "lv_A"), ("R", "lv_R"), ("taught", "lv_taught"), ("len12", "lv_len12"), ("len40", "lv_len40")):
            cmp(sec, src, f"{lab}: level {k}", d["levels"][k], f[kk], "full")
        cmp(sec, src, f"{lab}: R rise", d["R_rise_2048"], f["rise"], "full")
        cmp(sec, src, f"{lab}: loss_last", d["loss_last"], f["loss_last"], "full")
        cmp(sec, src, f"{lab}: word", d["word"], f["word"], "word")
    for lab, d in S["runs"].items():
        runfig(lab, d)
    for lab, d in S["first_24576"].items():
        runfig(lab + " (first 24,576)", d)
    for lab, d in S["references"].items():
        runfig(lab, d)
    for k, w in S["describe"].items():
        m_ = re.match(r"(.*) against (.*) (A80|R80)$", k)
        cand, ref, score = m_.groups()
        cmp(sec, src, f"describe {k}", w, describe(FIG[ref][score], FIG[cand][score], FIG[ref]["updates"] - 512, FIG[cand]["updates"] - 512), "word")
    for lab, d in S["seed_range"].items():
        ref = "E42 F10 S" if lab.startswith("F10") else "E42 G S"
        for k in ("A80", "R80"):
            cmp(sec, src, f"seed range {lab} {k}", d[k], FIG[lab][k] - FIG[ref][k], "int")
    fb = first_below()
    for lab in ("in16 matrix S", "in16 none B"):
        cmp(sec, src, f"first below 0.5 {lab}: update", S["first_below"][lab][0], fb[lab][0], "int")
        cmp(sec, src, f"first below 0.5 {lab}: loss", S["first_below"][lab][1], fb[lab][1], "full")
    for lab, w in S["words"].items():
        cmp(sec, src, f"word {lab}", w, FIG[lab]["word"], "word")


def first_below():
    out = {}
    for lab in ("in16 matrix S", "in16 none B", "in16 matrix S cd0", "in16 row S", "in16 rowcells S", "in16 rowcol S"):
        e = next((x for x in REC[lab]["cosines"] if x["cosines"]["matrix"]["l1.weight"] < 0.5), None)
        loss = {x["step"]: x["loss"] for x in REC[lab]["curve"]}
        out[lab] = (e["update"], loss[e["update"]]) if e else (None, None)
    return out


def section2():
    print("\n===== 2. Every figure recomputed from the receipts =====")
    print("2.1 the nine runs and the six references, recomputed (updates to 80 by the brief's definition; levels: mean of the last three rows;")
    print("    rise: level of R minus the mean of the three rows ending 2,048 updates earlier)")
    print("    run                               | upd    | A80    | R80    | T80    | lvA    lvR    lvT    len40  | rise  | word")
    for lab, f in FIG.items():
        if lab in REF:
            continue
        print(f"    {lab:33s} | {f['updates']:6,d} | {fmt(f['A80']):>6s} | {fmt(f['R80']):>6s} | {fmt(f['T80']):>6s} | {f['lv_A']:6.2f} {f['lv_R']:6.2f} {f['lv_taught']:6.2f} {f['lv_len40']:6.2f} | {f['rise']:+5.2f} | {f['word']}")
    tables = show(P + "e57/E57_TABLES_73129.md").decode()
    report = show(P + "e57/E57_REPORT.md").decode()
    summary = js(P + "e57/E57_SUMMARY_73129.json")
    main_table("2.2", "tables", tables); describe_table("2.2", "tables", tables)
    n_t = cos_tables("2.2", "tables", tables); sizes_table("2.2", "tables", tables)
    summary_json("2.3", "summary", summary)
    main_table("2.4", "report", report); describe_table("2.4", "report", report)
    n_r = cos_tables("2.4", "report", report); sizes_table("2.4", "report", report)
    print(f"    tables of readings parsed: {n_t} in the tables file, {n_r} in the report")
    # the readings table of the report
    for row in parse_table(report, "| Scale | Code decayed"):
        sc, cd = row[0].strip("`"), row[1]
        lab = f"in16 {sc} S" + (" cd0" if cd == "no" else "")
        cmp("2.4", "report", f"readings {lab} word 24,576", row[2], FIG[lab + " (first 24,576)"]["word"], "word")
        cmp("2.4", "report", f"readings {lab} word 49,152", row[3].strip("*"), FIG[lab]["word"], "word")
        parts = row[4].split("(")[0].split(";")
        cmp("2.4", "report", f"readings {lab} A80", num(parts[0]), FIG[lab]["A80"], "int")
        cmp("2.4", "report", f"readings {lab} R80", num(parts[1]), FIG[lab]["R80"], "int")
        cmp("2.4", "report", f"readings {lab} describe", row[4].split("*")[1], describe(FIG["E49 in16 B"]["A80"], FIG[lab]["A80"], 24064, FIG[lab]["updates"] - 512), "word")
    # predictions table of the report
    pred = predictions()
    for row in parse_table(report, "| | Statement | Chance | Outcome"):
        cmp("2.4", "report", f"{row[0]} outcome", row[3].split()[0].strip(":"), "true" if pred[row[0]][0] else "false", "word")
    # figures in the report's text (transcribed here with the place they come from)
    f, fb = FIG, first_below()
    cz = {e["update"]: e for e in REC["in16 matrix S cd0"]["cosines"]}; cd = {e["update"]: e for e in REC["in16 matrix S"]["cosines"]}
    run_h = sum(REC[l]["cpu_updates"] + REC[l]["cpu_scoring"] + REC[l]["cpu_cosines"] for l in LABELS) / 3600
    l1min = min((e["l1_weight_mean_abs"], e["update"]) for e in REC["in16 matrix S cd0"]["cosines"])
    text = [
        ("The runs: CPU-hours of the nine runs", 17.19, run_h, 2),
        ("The runs: of which the readings", 0.56, sum(REC[l]["cpu_cosines"] for l in LABELS) / 3600, 2),
        ("The runs: CPU-hours of the checks", 0.42, sum(CHK[n].get("cpu", 0.0) for n in "12345678") / 3600, 2),
        ("Explanation: row/B on A", 2.17, f["in16 row S"]["A80"] / f["E49 in16 B"]["A80"], 2),
        ("Explanation: row/B on R", 2.19, f["in16 row S"]["R80"] / f["E49 in16 B"]["R80"], 2),
        ("Explanation: matrix level A at 49,152", 1.2, f["in16 matrix S"]["lv_A"], 1),
        ("Explanation: matrix level R at 49,152", 1.2, f["in16 matrix S"]["lv_R"], 1),
        ("Explanation: matrix level T at 49,152", 1.9, f["in16 matrix S"]["lv_taught"], 1),
        ("Code: A", 51.06, f["in16 matrix S cd0"]["lv_A"], 2), ("Code: R", 46.05, f["in16 matrix S cd0"]["lv_R"], 2),
        ("Code: T", 71.00, f["in16 matrix S cd0"]["lv_taught"], 2),
        ("Code: code_mean_abs at the start", 0.407, cz[0]["code_mean_abs"], 3),
        ("Code: code_mean_abs at 49,152, not decayed", 0.159, cz[49152]["code_mean_abs"], 3),
        ("Code: code_mean_abs at 49,152, decayed", 0.033, cd[49152]["code_mean_abs"], 3),
        ("Code: l1_weight_mean_abs at the start", 0.1773, cz[0]["l1_weight_mean_abs"], 4),
        ("Code: least l1_weight_mean_abs, not decayed", 0.1222, l1min[0], 4), ("Code: its update", 18432, l1min[1], "int"),
        ("Code: l1_weight_mean_abs at the end, not decayed", 0.1674, cz[49152]["l1_weight_mean_abs"], 4),
        ("Code: l1_weight_mean_abs at the end, decayed", 0.0343, cd[49152]["l1_weight_mean_abs"], 4),
        ("Cosine: first below 0.5, sweep", 4096, fb["in16 matrix S"][0], "int"), ("Cosine: loss there", 1.040, fb["in16 matrix S"][1], 3),
        ("Cosine: first below 0.5, back-propagation", 4096, fb["in16 none B"][0], "int"), ("Cosine: loss there (B)", 0.842, fb["in16 none B"][1], 3),
        ("F10: row A80", 11520, f["F10 row S"]["A80"], "int"), ("F10: row R80", 13824, f["F10 row S"]["R80"], "int"),
        ("F10: row - E42 S on A", -1792, f["F10 row S"]["A80"] - f["E42 F10 S"]["A80"], "int"),
        ("F10: row - E42 S on R", -1280, f["F10 row S"]["R80"] - f["E42 F10 S"]["R80"], "int"),
        ("F10: row/E42 B on A", 1.32, f["F10 row S"]["A80"] / f["E42 F10 B"]["A80"], 2),
        ("F10: row/E42 B on R", 1.46, f["F10 row S"]["R80"] / f["E42 F10 B"]["R80"], 2),
        ("F10: rowcol A80", 12800, f["F10 rowcol S"]["A80"], "int"), ("F10: rowcol R80", 15616, f["F10 rowcol S"]["R80"], "int"),
        ("F10: rowcol - E42 S on A", -512, f["F10 rowcol S"]["A80"] - f["E42 F10 S"]["A80"], "int"),
        ("F10: rowcol - E42 S on R", 512, f["F10 rowcol S"]["R80"] - f["E42 F10 S"]["R80"], "int"),
        ("F10: level A row", 92.9, f["F10 row S"]["lv_A"], 1), ("F10: level A rowcol", 92.4, f["F10 rowcol S"]["lv_A"], 1),
        ("F10: level R row", 92.3, f["F10 row S"]["lv_R"], 1), ("F10: level R rowcol", 91.4, f["F10 rowcol S"]["lv_R"], 1),
        ("G: A80", 6144, f["G row S"]["A80"], "int"), ("G: R80", 6144, f["G row S"]["R80"], "int"),
        ("G: - E42 S on A", 512, f["G row S"]["A80"] - f["E42 G S"]["A80"], "int"), ("G: - E42 S on R", -512, f["G row S"]["R80"] - f["E42 G S"]["R80"], "int"),
        ("G: level A", 98.9, f["G row S"]["lv_A"], 1), ("G: level R", 98.8, f["G row S"]["lv_R"], 1),
        ("G: E42 S level A", 97.8, f["E42 G S"]["lv_A"], 1), ("G: E42 S level R", 96.3, f["E42 G S"]["lv_R"], 1),
        ("Cannot show: rise rowcells", 0.10, f["in16 rowcells S"]["rise"], 2), ("Cannot show: rise rowcol", 0.55, f["in16 rowcol S"]["rise"], 2),
        ("Cannot show: rise cd0", -3.90, f["in16 matrix S cd0"]["rise"], 2),
        ("Cannot show: row level at 40 letters", 48.1, f["in16 row S"]["lv_len40"], 1), ("Cannot show: row level A", 86.1, f["in16 row S"]["lv_A"], 1),
        ("Cannot show: B level at 40 letters", 66.1, f["in16 none B"]["lv_len40"], 1),
        ("Checks: check 5 forecast", 18.3, CHK["5"]["forecast_nine_runs_cpu_h"], 1),
        ("Checks: check 5 CPU-seconds an update (dry run of in16, 2,048)", 0.155, CHK["5"]["cpu_per_update"]["G"], 3),
        ("Checks: check 7 start mean", 0.1773, CHK["7"]["c"]["cd0"]["start"], 4), ("Checks: check 7 end without decay", 0.1764, CHK["7"]["c"]["cd0"]["end"], 4),
        ("Checks: check 7 end with decay", 0.1655, CHK["7"]["c"]["reference"]["end"], 4),
        ("Checks: check 7 largest deviation (2 significant digits)", 2.5, CHK["7"]["a"]["l1"]["l1.weight"]["max_dev"] * 1e8, 1),
        ("Checks: check 8 CPU ratio", 2.01, CHK["8"]["cpu_updates_last_over_4096"], 2),
        ("Checks: check 2 within 0.0005", 0.0005, max(0.0005, max(v["max_diff"] for fl in CHK["2"]["files"].values() for s, v in fl.items() if s != "exact")), "full"),
        ("Checks: check 6 rows equal to 24,576", 97, len([x for x in REC["in16 matrix S"]["curve"] if x["step"] <= 24576]), "int"),
    ]
    for lab, rep, mine, kind in text:
        cmp("2.5", "report text", lab, rep, mine, kind)
    # the worker's message
    msg = [
        ("row A80", 36096, f["in16 row S"]["A80"], "int"), ("row R80", 44800, f["in16 row S"]["R80"], "int"),
        ("row A at 49,152", 86.1, f["in16 row S"]["lv_A"], 1), ("row R at 49,152", 83.8, f["in16 row S"]["lv_R"], 1), ("row T at 49,152", 96.1, f["in16 row S"]["lv_taught"], 1),
        ("row A at 24,576", 53.5, f["in16 row S (first 24,576)"]["lv_A"], 1), ("row R at 24,576", 44.9, f["in16 row S (first 24,576)"]["lv_R"], 1),
        ("row T at 24,576", 72.7, f["in16 row S (first 24,576)"]["lv_taught"], 1),
        ("rowcells A", 53.7, f["in16 rowcells S"]["lv_A"], 1), ("rowcells R", 40.2, f["in16 rowcells S"]["lv_R"], 1), ("rowcells T", 68.3, f["in16 rowcells S"]["lv_taught"], 1),
        ("rowcol A", 69.8, f["in16 rowcol S"]["lv_A"], 1), ("rowcol R", 58.7, f["in16 rowcol S"]["lv_R"], 1), ("rowcol T", 85.9, f["in16 rowcol S"]["lv_taught"], 1),
        ("matrix A", 1.2, f["in16 matrix S"]["lv_A"], 1), ("matrix R", 1.2, f["in16 matrix S"]["lv_R"], 1), ("matrix T", 1.9, f["in16 matrix S"]["lv_taught"], 1),
        ("cd0 A", 51.06, f["in16 matrix S cd0"]["lv_A"], 2), ("cd0 R", 46.05, f["in16 matrix S cd0"]["lv_R"], 2), ("cd0 T", 71.00, f["in16 matrix S cd0"]["lv_taught"], 2),
        ("cd0 code_mean_abs end", 0.159, cz[49152]["code_mean_abs"], 3), ("decayed code_mean_abs end", 0.033, cd[49152]["code_mean_abs"], 3),
        ("code_mean_abs start", 0.407, cz[0]["code_mean_abs"], 3),
        ("B A80", 16640, f["in16 none B"]["A80"], "int"), ("B R80", 20480, f["in16 none B"]["R80"], "int"),
        ("F10 row A80", 11520, f["F10 row S"]["A80"], "int"), ("F10 row R80", 13824, f["F10 row S"]["R80"], "int"),
        ("F10 rowcol A80", 12800, f["F10 rowcol S"]["A80"], "int"), ("F10 rowcol R80", 15616, f["F10 rowcol S"]["R80"], "int"),
        ("G row A80", 6144, f["G row S"]["A80"], "int"), ("G row R80", 6144, f["G row S"]["R80"], "int"),
        ("first below 0.5, sweep", 4096, fb["in16 matrix S"][0], "int"), ("first below 0.5, B", 4096, fb["in16 none B"][0], "int"),
        ("CPU-hours runs", 17.19, run_h, 2), ("CPU-hours checks", 0.42, sum(CHK[n].get("cpu", 0.0) for n in "12345678") / 3600, 2),
    ]
    for lab, rep, mine, kind in msg:
        cmp("2.6", "message", lab, rep, mine, kind)
    words = [("row learns the family", "learns the family", f["in16 row S"]["word"]), ("rowcells partly", "partly", f["in16 rowcells S"]["word"]),
             ("rowcol partly", "partly", f["in16 rowcol S"]["word"]), ("matrix not learned", "not learned", f["in16 matrix S"]["word"]),
             ("cd0 partly", "partly", f["in16 matrix S cd0"]["word"])]
    for x in ("F10 row S", "F10 rowcol S"):
        for sc in ("A80", "R80"):
            words.append((f"{x} {sc} about the same against E42 F10 S", "about the same", describe(f["E42 F10 S"][sc], f[x][sc], 24064, 24064)))
    for sc in ("A80", "R80"):
        words.append((f"G row S {sc} about the same against E42 G S", "about the same", describe(f["E42 G S"][sc], f["G row S"][sc], 24064, 24064)))
    words.append(("in16 matrix S rows equal E49's to 24,576", "True", "True"))
    words.append(("B ends on E49's hash", "True", str(REC["in16 none B"]["tensors_sha256"] == REF["E49 in16 B"]["tensors_sha256"])))
    words.append(("all nine runs exited 0", "True", str(show(P + "e57/logs/queue.log").decode().count(" exit=0") == 9)))
    words.append(("none resumed", "True", str(all(REC[l]["resumes"] == [] for l in LABELS))))
    for k, (v, _) in pred.items():
        claimed = "true" if k in ("P57.4", "P57.5", "P57.6", "P57.7", "P57.8", "P57.10", "P57.11", "P57.12") else "false"
        words.append((f"{k}", claimed, "true" if v else "false"))
    for lab, rep, mine in words:
        cmp("2.6", "message", lab, rep, mine, "word")
    # the count
    print("2.7 the comparisons")
    for src in ("tables", "summary", "report", "report text", "message"):
        rows = [c for c in COMP if c[1] == src]
        nw = sum(c[5] == "word" for c in rows)
        print(f"    {src:12s}: {len(rows)} compared ({len(rows) - nw} figures, {nw} words or outcomes); differing {sum(not c[6] for c in rows)}")
    bad = [c for c in COMP if not c[6]]
    print(f"    all sources: {len(COMP)} compared, {sum(c[5] != 'word' for c in COMP)} of them figures; differing {len(bad)}")
    for c in bad:
        print(f"      differs [{c[0]} {c[1]}] {c[2]}: reported {c[3]!r}, recomputed {c[4]!r}")
    clock("section 2")


# ================================================================ section 3
def scorer_panels():
    held = {t: gsplit(t, .5, SEED) for t in DEV}
    nm = lambda t: f"{t[0]}{t[1]}"
    AP = {nm(t): panel(t, held[t][1], SEED, ks=(12,), n=300)[12] for t in DEV}
    TA = {nm(t): panel(t, held[t][0], SEED + 1, ks=(12,), n=300)[12] for t in DEV}
    RP = {}
    for t in DEV[:2]:
        red = [(a, b) for a, b in held[t][1] if (a + b >= 89 if t[0] == "add" else a - b < 0)]
        RP[nm(t)] = [grender(t, a, b, 0, None) for a, b in Random(f"e41|{SEED}|50|{t[0]}").sample(sorted(red), 1000)]
    LT = ("add", 89, b"p")
    LP = panel(LT, gsplit(LT, .5, SEED)[1], SEED, ks=(12, 40), n=300)
    return AP, RP, TA, {str(k): v for k, v in LP.items()}


def right(W, masks, items):
    """Whole answers right under teacher forcing (= under greedy generation), float32, masked matrices."""
    f32 = np.float32
    x = np.array([[BOS] + p + a[:-1] for p, a in items], dtype=np.int64); tgt = np.array([a for _, a in items], dtype=np.int64)
    z1 = np.tanh(W["book.weight"] @ W["l1.weight"].T + W["l1.bias"]).astype(f32)
    GI = (z1 @ (W["rnn.weight_ih_l0"] * masks["ih"]).T + W["rnn.bias_ih_l0"]).astype(f32)
    WhT = (W["rnn.weight_hh_l0"] * masks["hh"]).T.astype(f32); bh = W["rnn.bias_hh_l0"]
    LmT = (W["lm.weight"] * masks["lm"]).T.astype(f32)
    B, L = x.shape; n = tgt.shape[1]; h = np.zeros((B, U), f32); out = []
    sg = lambda v: 1 / (1 + np.exp(-v))
    for t in range(L):
        gi = GI[x[:, t]]; gh = h @ WhT + bh
        r = sg(gi[:, :U] + gh[:, :U]); z = sg(gi[:, U:2 * U] + gh[:, U:2 * U]); nn_ = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
        h = ((1 - z) * nn_ + z * h).astype(f32)
        if t >= L - n:
            z2 = np.tanh(z1[x[:, t]] @ W["l2.weight"].T + W["l2.bias"] + h @ LmT + W["lm.bias"])
            out.append((z2 @ W["l3.weight"].T + W["l3.bias"]).argmax(-1))
    return int((np.stack(out, 1) == tgt).all(1).sum())


def make_net(Wt, name):
    net = V.Net.__new__(V.Net)
    net.name, net.dt, net.coded, net.code, net.when, net.what, net.receipt_path = name, np.float64, False, None, "end", name, ""
    ones = lambda s: np.ones(s, dtype=bool)
    net.M = {"l1": ones((D, 8)), "l2": ones((D, D)), "l3": ones((257, D)), "hh": np.tile(Wt["mask"].astype(bool), (3, 1)),
             "ih": np.tile(Wt["mask_in"].astype(bool), (3, 1)), "lm": Wt["mask_out"].astype(bool)}
    net.P = {k: Wt[k].astype(np.float64) for k in STATE}
    net.trained, net.maps, net.raw = list(STATE), list(V.MAPS), Wt
    net.outside_zero = {k: bool((net.P[V.WNAME[k]][~net.M[k]] == 0).all()) for k in V.MAPS}
    return net


FEED = {"matrix": ("V0", "all"), "row": ("V2", "all"), "rowcells": ("V2", "gru"), "rowcol": ("V4", "gru")}


def cosines_of(net):
    Fx = V.exact_feedback(net); Fs = {s: V.feedback(net, *FEED[s]) for s in SCALES}
    acc = {s: {t: 0.0 for t in COS_T} for s in SCALES}
    bts = V.stream_batches(17)[1:]
    for x, y, mask in bts:
        c = V.forward(net, x, y, mask); ge = V.backward(net, c, Fx)
        for s in SCALES:
            gs = V.backward(net, c, Fs[s])
            for t in COS_T:
                acc[s][t] += V.cos(gs[t], ge[t]) / len(bts)
    tot = cnt = 0.0
    z = np.tanh(net.P["book.weight"] @ net.P["l1.weight"].T + net.P["l1.bias"])
    for x, _, _ in bts:
        keep = x != BOS
        tot += np.abs(z[x[keep]]).sum(); cnt += keep.sum() * D
    return acc, dict(l1_weight_mean_abs=float(np.abs(net.P["l1.weight"]).mean()), code_mean_abs=float(tot / cnt))


def row_scale_spread(net):
    """Coefficient of variation (standard deviation over mean) of the row means of |W| over kept entries, map by map."""
    out = {}
    for k in V.MAPS:
        A = np.abs(net.P[V.WNAME[k]]) * net.M[k]
        row = A.sum(1) / np.maximum(net.M[k].sum(1), 1)
        out[k] = float(row.std() / row.mean())
    return out


SPREAD = {}


def section3():
    print("\n===== 3. The saved weights =====")
    AP, RP, TA, LP = scorer_panels()
    print("3.1 hashes, masks, the final counts by a forward pass in numpy (here / the receipt's last row)")
    E49B, _ = load_pt(P + "e49/e49_in16_B_73129.pt", E49REF)
    e49b = {k: E49B[k] for k in STATE}; e49masks = {k: E49B[k].copy() for k in ("mask", "mask_in", "mask_out")}
    del E49B
    cos_rows = []
    for lab in LABELS:
        rc = REC[lab]; last = rc["curve"][-1]
        S0, s0 = load_pt(P + "e57/" + stem(lab) + "_start.pt")
        start_ok = (tensors_sha({k: S0[k] for k in STATE}) == rc["start_tensors_sha256"], s0 == rc["start_sha256"])
        if SPEC[lab][1] == "in16":
            start_ok += (all(np.array_equal(S0[k], e49masks[k]) for k in e49masks),)
        del S0
        Wt, s1 = load_pt(P + "e57/" + stem(lab) + ".pt")
        W = {k: Wt[k] for k in STATE}
        masks = {"ih": np.tile(Wt["mask_in"].astype(bool), (3, 1)), "hh": np.tile(Wt["mask"].astype(bool), (3, 1)), "lm": Wt["mask_out"].astype(bool)}
        outside = max(float(np.abs(W[n][~masks[k]]).max()) if (~masks[k]).any() else 0.0 for k, n in (("ih", "rnn.weight_ih_l0"), ("hh", "rnn.weight_hh_l0"), ("lm", "lm.weight")))
        kept = {k: int(masks[k].sum()) for k in masks}
        mine = dict(A={k: right(W, masks, v) for k, v in AP.items()}, R={k: right(W, masks, v) for k, v in RP.items()},
                    T={k: right(W, masks, v) for k, v in TA.items()}, L={k: right(W, masks, v) for k, v in LP.items()})
        theirs = dict(A=last["A_counts"], R=last["R_counts"], T=last["taught_counts"], L=last["length_counts"])
        eq = {k: mine[k] == theirs[k] for k in mine}
        extra = ""
        if lab == "in16 none B":
            extra = f"; tensors equal to E49's file under back-propagation bit for bit: {all(np.array_equal(W[k], e49b[k]) for k in STATE)}"
        print(f"    {lab:18s} file SHA-256 = receipt {s1 == rc['weights_sha256']}; tensors_sha256 = receipt {tensors_sha(W) == rc['tensors_sha256']};"
              f" start file (tensors, file{', masks = E49' if len(start_ok) == 3 else ''}) {start_ok}; kept entries {kept}; largest weight outside a mask {outside}{extra}")
        print(f"        A {list(mine['A'].values())} / {list(theirs['A'].values())} {eq['A']} | R {list(mine['R'].values())} / {list(theirs['R'].values())} {eq['R']}"
              f" | T {list(mine['T'].values())} / {list(theirs['T'].values())} {eq['T']} | length 12, 40 {list(mine['L'].values())} / {list(theirs['L'].values())} {eq['L']}", flush=True)
        net = make_net(Wt, lab)
        acc, sz = cosines_of(net)
        SPREAD[lab] = row_scale_spread(net)
        e = rc["cosines"][-1]
        d = max(abs(acc[s][t] - e["cosines"][s][t]) for s in SCALES for t in COS_T)
        cos_rows.append((lab, e["update"], acc, e, d, sz))
        del Wt, W, net
    print("3.2 the cosines of the last reading on the saved weights (16 batches: the 2nd to the 17th of the stream; float64 numpy, sweep_scale_variants.py)")
    print("    run                update | largest |here - receipt| over 24 cosines | l1 mean abs here / receipt | code mean abs here / receipt")
    for lab, u, acc, e, d, sz in cos_rows:
        print(f"    {lab:18s} {u:6d} | {d:.2e} | {sz['l1_weight_mean_abs']:.6f} / {e['l1_weight_mean_abs']:.6f} | {sz['code_mean_abs']:.6f} / {e['code_mean_abs']:.6f}")
    for lab, u, acc, e, d, sz in cos_rows:
        if lab in ("in16 row S", "in16 matrix S"):
            print(f"    {lab} at {u:,}, each scale: byte table, l1, input map, recurrent map, lm, l2 (here; receipt)")
            for s in SCALES:
                print(f"      {s:8s} " + ", ".join(f"{acc[s][t]:.3f}" for t in COS_T) + " ; " + ", ".join(f"{e['cosines'][s][t]:.3f}" for t in COS_T))
    clock("section 3")


# ================================================================ section 4
def predictions():
    f, fb = FIG, first_below()
    c1, c8 = CHK["1"], CHK["8"]
    first_attempt_failed = "AssertionError" in show(P + "e57_qual/interrupted/check8_first_attempt/check8_killed.log").decode()
    l1m = {e["update"]: e["cosines"]["matrix"]["l1.weight"] for e in REC["in16 matrix S"]["cosines"]}
    l1b = {e["update"]: e["cosines"]["matrix"]["l1.weight"] for e in REC["in16 none B"]["cosines"]}
    return {
        "P57.1": (all(CHK[n]["pass"] for n in "12378") and not first_attempt_failed and not c1["rule_c"], "check 8 passed at its second attempt; checks 1 and 3 without rule (c)"),
        "P57.2": (f["in16 row S (first 24,576)"]["word"] == "learns the family", f["in16 row S (first 24,576)"]["word"]),
        "P57.3": (f["in16 rowcol S (first 24,576)"]["word"] == "learns the family", f["in16 rowcol S (first 24,576)"]["word"]),
        "P57.4": (f["in16 row S (first 24,576)"]["lv_taught"] >= 50, f"{f['in16 row S (first 24,576)']['lv_taught']:.2f}"),
        "P57.5": (describe(13312, f["F10 row S"]["A80"], 24064, 24064) == "about the same", f"x = {f['F10 row S']['A80']}, ratio {f['F10 row S']['A80'] / 13312:.3f}"),
        "P57.6": (describe(5632, f["G row S"]["A80"], 24064, 24064) == "about the same", f"x = {f['G row S']['A80']}, ratio {f['G row S']['A80'] / 5632:.3f}"),
        "P57.7": (any(l1m[u] < 0.5 for u in (2048, 4096, 6144)), ", ".join(f"{u}: {l1m[u]:.3f}" for u in (2048, 4096, 6144))),
        "P57.8": (f["in16 row S"]["word"] == "learns the family", f["in16 row S"]["word"]),
        "P57.9": (f["in16 rowcells S"]["word"] == "learns the family", f["in16 rowcells S"]["word"]),
        "P57.10": (f["in16 matrix S"]["word"] == "not learned", f["in16 matrix S"]["word"]),
        "P57.11": (any(l1b[u] < 0.5 for u in l1b if u <= 12288), ", ".join(f"{u}: {l1b[u]:.3f}" for u in sorted(l1b) if u <= 12288)),
        "P57.12": (f["in16 matrix S cd0"]["lv_taught"] >= 50, f"{f['in16 matrix S cd0']['lv_taught']:.2f}"),
        "P57.13": (f["in16 matrix S cd0"]["word"] == "learns the family", f["in16 matrix S cd0"]["word"]),
    }


def flip_rows(xr, xc):
    """For two numbers of updates to 80 that both exist: how many scored rows (256 updates) the candidate's number must move for
    describe's word to change, and in which direction."""
    w = describe(xr, xc, 0, 0)
    for k in range(1, 400):
        for sgn in (1, -1):
            x = xc + sgn * 256 * k
            if x > 0 and describe(xr, x, 0, 0) != w:
                return k, ("later" if sgn > 0 else "earlier")
    return None, None


def section4():
    print("\n===== 4. The fixed wording, the margins, the predictions =====")
    print("4.1 each run's word and the margins on which it rests (level minus threshold; the three rows of the level)")
    for lab, f in FIG.items():
        if lab in REF or "(reference)" in lab:
            continue
        base = lab.replace(" (first 24,576)", ""); c = curve_of(base, 24576 if "first" in lab else None)
        last3 = c[-3:]
        print(f"    {lab:33s} {f['word']:18s} | A {f['lv_A'] - 80:+6.2f}, R {f['lv_R'] - 80:+6.2f}, T {f['lv_taught'] - 50:+6.2f} (against 80, 80, 50)"
              f" | rows of T {[round(x['taught'], 2) for x in last3]}, of R {[round(x['R'], 2) for x in last3]}")
    c = curve_of("in16 row S")
    tailR = [(x["step"], round(x["R"], 2)) for x in c if x["step"] >= 44032]
    tailA = [(x["step"], round(x["A"], 2)) for x in c if 35328 <= x["step"] <= 37376]
    windows = sum(1 for i in range(len(c) - 2) if c[i]["step"] > 0 and all(c[i + j]["R"] >= 80 for j in range(3)))
    print(f"    in16 row S, R from 44,032: {tailR}")
    print(f"    in16 row S, A around its 80: {tailA}; windows of three rows at or above 80 on R in the whole run: {windows};"
          f" rows at or above 80 on R: {sum(x['R'] >= 80 for x in c)} of 193, after 44,800: {sum(x['R'] >= 80 for x in c if x['step'] >= 44800)} of {sum(1 for x in c if x['step'] >= 44800)}")
    single = next((x["step"] for x in c if x["step"] > 0 and x["R"] >= 80), None)
    print(f"    in16 row S, first single row at or above 80 on R: {single:,}; on A: {next(x['step'] for x in c if x['step'] > 0 and x['A'] >= 80):,}")
    print("4.2 the words of describe and the number of scored rows by which the candidate's updates to 80 would have to move to change them")
    for cand, ref in (("F10 row S", "E42 F10 S"), ("F10 row S", "E42 F10 B"), ("F10 rowcol S", "E42 F10 S"), ("F10 rowcol S", "E42 F10 B"),
                      ("G row S", "E42 G S"), ("G row S", "E42 G B"), ("in16 none B", "E49 in16 B")):
        for sc in ("A80", "R80"):
            xr, xc = FIG[ref][sc], FIG[cand][sc]
            k, d = flip_rows(xr, xc)
            print(f"    {cand:13s} vs {ref:10s} {sc}: {xc:,}/{xr:,} = {xc / xr:.3f} {describe(xr, xc, 24064, 24064):15s} changes after {k} rows {d}")
    for lab in ("in16 row S", "in16 rowcells S", "in16 rowcol S", "in16 matrix S", "in16 matrix S cd0"):
        for sc, xr in (("A80", 16640), ("R80", 20480)):
            for L_ in (lab, lab + " (first 24,576)"):
                xc = FIG[L_][sc]
                print(f"    {L_:33s} {sc} against E49 B ({xr:,}): {fmt(xc):>11s} -> {describe(xr, xc, 24064, FIG[L_]['updates'] - 512)}"
                      + (f" (recordable {FIG[L_]['updates'] - 512:,} = x{(FIG[L_]['updates'] - 512) / xr:.2f})" if xc is None else f" (x{xc / xr:.2f})"))
    print("4.3 the predictions")
    pr = predictions()
    for k, (v, why) in pr.items():
        print(f"    {k}: {'true' if v else 'false'} ({why})")
    print(f"    true: {[k for k, (v, _) in pr.items() if v]}; false: {[k for k, (v, _) in pr.items() if not v]}")
    print("4.4 the brief's table of explanations (words over 49,152 updates, the four runs with a decayed code)")
    w = {s: FIG[f"in16 {s} S"]["word"] for s in ("matrix", "row", "rowcells", "rowcol")}
    if w["matrix"] == "learns the family":
        row = 1
    elif w["rowcells"] == "learns the family":
        row = 2
    elif w["row"] == "learns the family":
        row = 3
    elif w["rowcells"] == "not learned" and w["row"] == "not learned":
        row = 4
    else:
        row = 5
    print(f"    words {w}; row {row} of the table" + (": Part 1 alone is not supported; a scale for each receiving unit at every map was sufficient" if row == 3 else ""))
    print(f"    condition of the table (the matrix run repeats E49's rows to 24,576; B ends on E49's hash): see 1.5")
    cz = FIG["in16 matrix S cd0"]
    print(f"4.5 the code not decayed: the run with a decayed code is '{w['matrix']}', so the three sentences apply; the run without decay is '{cz['word']}'"
          f" (A {cz['lv_A']:.2f}, R {cz['lv_R']:.2f}, T {cz['lv_taught']:.2f})")
    clock("section 4")


# ================================================================ section 5
def section5():
    print("\n===== 5. The figures of the reading =====")
    f = FIG
    print(f"5.1 in16 under row against in16 under back-propagation: A {f['in16 row S']['A80']:,}/{f['E49 in16 B']['A80']:,} = {f['in16 row S']['A80'] / f['E49 in16 B']['A80']:.2f};"
          f" R {f['in16 row S']['R80']:,}/{f['E49 in16 B']['R80']:,} = {f['in16 row S']['R80'] / f['E49 in16 B']['R80']:.2f}; E42's F10 sweep/back-propagation on three seeds: A 1.53, 1.89, 1.36; R 1.59, 1.81, 1.47 (the brief)")
    print(f"    F10 under row against E42's F10 under back-propagation: A {f['F10 row S']['A80'] / f['E42 F10 B']['A80']:.2f}, R {f['F10 row S']['R80'] / f['E42 F10 B']['R80']:.2f};"
          f" G under row against E42's G under back-propagation: A {f['G row S']['A80'] / f['E42 G B']['A80']:.2f}, R {f['G row S']['R80'] / f['E42 G B']['R80']:.2f}")
    for lab in ("in16 row S", "in16 row S (first 24,576)", "in16 none B", "F10 row S", "F10 rowcol S", "E42 F10 S", "E42 F10 B", "G row S", "E42 G S"):
        x = f[lab]
        print(f"5.2 {lab:26s} levels A {x['lv_A']:.2f}, R {x['lv_R']:.2f}, T {x['lv_taught']:.2f}, 12 letters {x['lv_len12']:.2f}, 40 letters {x['lv_len40']:.2f} (after {x['updates']:,} updates)")
    bB = {x["step"]: x for x in curve_of("in16 row S")}
    print(f"    in16 row S at 24,576 updates (the length of the run under back-propagation): A {f['in16 row S (first 24,576)']['lv_A']:.2f}, R {f['in16 row S (first 24,576)']['lv_R']:.2f} against 85.22, 82.53")
    print("5.3 the row scale: coefficient of variation of the row means of |W| over kept entries at the end (map: l1, input map, recurrent map, l2, lm, l3)")
    kept = {"in16": {"l1": 8, "ih": 16, "hh": 11, "l2": 512, "lm": 256, "l3": 512}}
    print(f"    kept entries a row in in16 (entries a receiving unit receives): {kept['in16']}")
    for lab in ("in16 matrix S", "in16 row S", "in16 rowcells S", "in16 none B", "F10 row S", "G row S"):
        print(f"    {lab:18s} " + ", ".join(f"{k} {SPREAD[lab][k]:.3f}" for k in V.MAPS))
    clock("section 5")


# ================================================================ section 6
def section6():
    print("\n===== 6. The committed text files: unfilled marks and names of language models (counts only) =====")
    files = [p for p in subprocess.run(["git", "ls-tree", "-r", "--name-only", WREF, "--", P + "e57"], capture_output=True, text=True, check=True).stdout.split()
             if not p.endswith(".pt")] + [M + "exp57_scale.py"]
    marks = ["TO" + "DO", "TB" + "D", "FIX" + "ME", "XX" + "X", "<" + "fill", "?" + "??", "PLACE" + "HOLDER", "[" + "...]", "{" + "{"]
    names = ["cl" + "aude", "an" + "thropic", "op" + "us", "son" + "net", "hai" + "ku", "g" + "pt", "co" + "dex", "gem" + "ini", "lla" + "ma", "open" + "ai"]
    tot_m = tot_n = 0; per = {}
    for p in files:
        t = show(p).decode(errors="replace")
        nm = sum(t.count(m) for m in marks)
        nn = 0
        for n in names:
            for mo in re.finditer(r"(?i)(?<![a-z])" + n + r"(?![a-z])", t):
                s = t[max(0, mo.start() - 8):mo.end() + 1]
                if t[mo.end():mo.end() + 1] == "/" or "origin/" in s:          # a branch name (claude/..., codex/...)
                    continue
                nn += 1
        tot_m += nm; tot_n += nn
        if nm or nn:
            per[p.split("/")[-1]] = (nm, nn)
    print(f"6.1 text files searched: {len(files)} (exp45/e57 without weight files, and the runner); unfilled marks {tot_m}; names of language models outside branch names {tot_n}")
    print(f"    files with a hit (marks, names): {per}")
    all_e57 = subprocess.run(["git", "ls-tree", "-r", "--name-only", WREF, "--", P + "e57", P + "e57_qual", M + "exp57_scale.py", M + "e57_tables.py", M + "E49_COPIES_FOR_E57.txt"],
                             capture_output=True, text=True, check=True).stdout.split()
    print(f"6.2 E57's committed files at {WREF}: {len(all_e57)} (the report lists 109 and itself)")
    rep = show(P + "e57/E57_REPORT.md").decode()
    listed = re.findall(r"^(exp45/\S+|exp57_scale\.py|e57_tables\.py|E49_COPIES_FOR_E57\.txt)", rep, re.M)
    have = {p.replace(M, "") for p in all_e57}
    print(f"    paths in the report's list: {len(listed)}; listed but not committed {sorted(set(listed) - have)}; committed but not listed {sorted(have - set(listed))}")
    draft = js(P + "e57/E57_SUMMARY_draft.json")
    print(f"6.3 E57_SUMMARY_draft.json: runs in it {len(draft.get('runs', {}))}; equal to the final summary {draft == js(P + 'e57/E57_SUMMARY_73129.json')}")
    clock("section 6")


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
