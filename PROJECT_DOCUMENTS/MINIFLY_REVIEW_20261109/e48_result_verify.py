"""Verification of the result of E48 (a new family learned only in connections gated by its own code), 17 November 2026.

Read-only.  Nothing is trained, torch is not used, and no code of another branch is run.  Files are read with `git show`:
the worker's branch claude/sweet-turing-lwi3no at 2d9d1726 (receipts, check receipts, answer files, 108 term files), the
default branch at 7cabb13a (E40's receipts, the life-end weights, the brief), and claude/vibrant-allen-8hziln at 6922a3a4
(E40's weights at rows 64 and 8,192).  The forward is the numpy reader of the audit (np_gru.py, e48_gated_reader.py).

  A  the receipts: 18 runs complete; the flags of checks 2, 3 and 4 at every row; the seven check receipts; the sources
     named in the receipts against the committed blobs; the six required changes of the audit in what ran
  B  the reader on the committed files: for every committed term file (108) the life-end network with the term and the four
     code rows reproduces the 18 recorded counts of that row; the bytes generated for the reduction pairs equal the worker's
     answer file item for item; the digests of the earlier panels' answers equal the receipts'; file and tensor hashes;
     the scores with the gate forced off at rows 256 and 1,024
  C  keeping: every earlier family with and without the term loaded (answers item for item, scores, and logits value for
     value on a sample), on a wide set of earlier items (the twelve tasks at every length, the compositions, every held-out
     reduction pair); the same with the gate forced on; E40's own losses on the six panels at the same rows
  D  the reading: R (mean of the reduction score at rows 128, 256, 512) for the 18 runs beside E40's; the count "16 of 18 by
     row 1,024"; first rows at 80% and 90% and at E40's level; the order of the ranks; the two rules
  E  the predictions P48.1 to P48.7, scored from the receipts
  F  what a term costs in weights beside the family's taught set; how many directions the trained products use; the size of
     the product by row beside the size of E40's change; what the wrong answers on the new family are
  G  the gate and the position: the new family scored with the term applied from the code byte's position on, and from the
     position after it, in place of from the start symbol on (nothing retrained)

Run from the repository root:
    python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e48_result_verify.py [A B C D E F G] [quick]
(`quick`: part B reads rows 256 and 1,024 of the rank-16 runs only, part C two networks at rank 16, part G rank 16.
 `file=<text>`: part B reads only the term files whose name contains the text, for example file=e48_S_73129_r16_row16.)
"""
import hashlib, json, os, re, subprocess, sys, time
from random import Random
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):      # one thread: the host is shared
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import (load_weights, forward as plain_forward, split, panel, render, batch, apply, comp_panels, git_bytes, tensors,
                    exact_teacher_forced, sigmoid, SYM, DEV, LIFE, BOS, EOS, P)
import e48_gated_reader as G

WB = "2d9d1726"                      # claude/sweet-turing-lwi3no: the E48 worker's report
DREF = "7cabb13a"                    # the default branch, codex/minifly-consolidated-20261023, on 17 November
E40W = "6922a3a4"                    # claude/vibrant-allen-8hziln: E40's weights at rows 64 and 8,192
M = "MOD97_LOCAL_RULE_20261106/"
E48 = M + "exp45/e48/"
SEEDS = (73129, 93683, 116089)
RULES = ("S", "B")
RANKS = (4, 16, 64)
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
EARLIER6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
KS_LIFE = (0, 8, 12, 20, 40)         # exp16_integrated.panels under E16_ARCH=bank2, the runner's setting
RULE_NAME = {"S": "sweep", "B": "backpropagation"}
want = {a for a in sys.argv[1:] if a in set("ABCDEFG")} or set("ABCDEFG")
quick = "quick" in sys.argv[1:]
only = next((a[5:] for a in sys.argv[1:] if a.startswith("file=")), None)
T0 = time.time()

show = lambda path, ref: subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout
name = lambda t: f"{t[0]}{t[1]}"
_cache = {}


def jload(path, ref):
    if (path, ref) not in _cache:
        _cache[(path, ref)] = json.loads(show(path, ref))
    return _cache[(path, ref)]


run_name = lambda rule, seed, rank: f"e48_{rule}_{seed}_r{rank}"
receipt = lambda rule, seed, rank: jload(E48 + run_name(rule, seed, rank) + ".json", WB)
e40_receipt = lambda rule, seed: jload(M + f"exp45/e40/e40_lived_{rule}_{seed}.json", DREF)
RUNS = [(rule, seed, rank) for rule in RULES for seed in SEEDS for rank in RANKS]
sha_json = lambda obj: hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def tensors_sha(d):
    """exp20_reference.tensors_sha for float32 tensors: names in order, dtype, shape and values."""
    h = hashlib.sha256()
    for k in sorted(d):
        v = np.ascontiguousarray(d[k], dtype="<f4")
        h.update(k.encode()); h.update(b"torch.float32"); h.update(str(tuple(v.shape)).encode()); h.update(v.tobytes())
    return h.hexdigest()


def life_end(rule, seed):
    key = ("W", rule, seed)
    if key not in _cache:
        _cache[key] = load_weights(P + f"e26/e26_life_{rule}_{seed}.pt", DREF)
    return _cache[key]


def e40_panels(seed):
    """E40's Scorer, rebuilt from its text: held and taught at 12 letters (300 pairs), every held-out add and sub pair without
    letters split by the reduction, and the earlier tasks as exp16_integrated.panels builds them."""
    key = ("panels", seed)
    if key in _cache:
        return _cache[key]
    held, taught, plain = {}, {}, {}
    for t in F4:
        tp, hp = split(t, .10, seed)
        held[name(t)] = panel(t, hp, seed, (12,), 300)[12]
        taught[name(t)] = panel(t, tp, seed + 1, (12,), 300)[12]
        if t[0] in ("add", "sub"):
            red, nored = [], []
            for a, b in hp:
                need = (a + b >= 83) if t[0] == "add" else (a - b < 0)
                (red if need else nored).append(render(t, a, b, 0, Random(0)))
            plain[name(t)] = dict(reduction=red, no_reduction=nored)
    earlier = {name(t): panel(t, split(t, .25, seed)[1], seed, KS_LIFE, 400) for t in LIFE + F3}
    earlier.update({name(t): panel(t, split(t, .5, seed)[1], seed, (12,), 300) for t in DEV})
    _cache[key] = (held, taught, plain, earlier)
    return _cache[key]


def gen(W, term, items):
    outs = []
    for j in range(0, len(items), 4096):
        outs += G.generate(W, term, [p for p, _ in items[j:j + 4096]])
    return outs


def answers(W, term, prompts, gate=None, first=0, cap=6):
    """e48_gated_reader.generate with two things made explicit: `gate` (None: read from the code byte; True or False: forced) and
    `first`, the first position at which the term is applied (0: from the start symbol on, as the runner does)."""
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_plain = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']
    gi_gated = gi_plain + (z1 @ term["down_i"].T) @ term["up_i"].T
    l2 = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']
    U = Whh.shape[1]
    for _, idx_all in groups.items():
        for j in range(0, len(idx_all), 4096):
            idx = idx_all[j:j + 4096]
            x = np.array([[BOS] + list(prompts[i]) for i in idx], dtype=np.int64)
            on = G.gate_of(x) if gate is None else gate
            h = np.zeros((len(idx), U), dtype=np.float32)

            def step(h, tok, t):
                use = on and t >= first
                gi = (gi_gated if use else gi_plain)[tok]; gh = h @ Whh.T + bhh
                if use:
                    gh = gh + (h @ term["down_h"].T) @ term["up_h"].T
                r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U])
                n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:])
                return (1 - z) * n + z * h

            L = x.shape[1]
            for t in range(L):
                h = step(h, x[:, t], t)
            tok = x[:, -1]
            gen_ = [[] for _ in idx]; done = [False] * len(idx)
            for c in range(cap):
                y = (np.tanh(l2[tok] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1)
                for q in range(len(idx)):
                    if not done[q]:
                        gen_[q].append(int(y[q])); done[q] = int(y[q]) == EOS
                if all(done) or c == cap - 1:
                    break
                tok = y; h = step(h, tok, L + c)
            for q, i in enumerate(idx):
                out[i] = gen_[q]
    return out


def cnt(outs, items):
    return dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)), final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))


def score_new(W, term, seed):
    """The new family's twelve panels as E40's Scorer scores them; also the bytes generated for the reduction pairs."""
    held, taught, plain, _ = e40_panels(seed)
    s = dict(held_k12={n: cnt(gen(W, term, it), it) for n, it in held.items()},
             taught_k12={n: cnt(gen(W, term, it), it) for n, it in taught.items()}, held_plain_k0={})
    red_out = {}
    for n, v in plain.items():
        s["held_plain_k0"][n] = {}
        for w, it in v.items():
            o = gen(W, term, it); s["held_plain_k0"][n][w] = cnt(o, it)
            if w == "reduction":
                red_out[n] = o
    return s, red_out


def score_earlier(W, term, seed):
    _, _, _, earlier = e40_panels(seed)
    outs = {n: gen(W, term, earlier[n][12]) for n in EARLIER6}
    return {n: cnt(outs[n], earlier[n][12]) for n in EARLIER6}, outs


def summary(s):
    pct = lambda d: 100.0 * d["exact"] / d["n"]
    return dict(held_add_sub_mean=(pct(s["held_k12"]["add83"]) + pct(s["held_k12"]["sub83"])) / 2,
                taught_add_sub_mean=(pct(s["taught_k12"]["add83"]) + pct(s["taught_k12"]["sub83"])) / 2,
                reduction_score=(pct(s["held_plain_k0"]["add83"]["reduction"]) + pct(s["held_plain_k0"]["sub83"]["reduction"])) / 2,
                no_reduction_score=(pct(s["held_plain_k0"]["add83"]["no_reduction"]) + pct(s["held_plain_k0"]["sub83"]["no_reduction"])) / 2)


def red(rule, seed, rank, row):
    return next(r for r in receipt(rule, seed, rank)["rows"] if r["updates"] == row)["scores"]["summary"]["reduction_score"]


def e40_row(rule, seed, row):
    return next(r for r in e40_receipt(rule, seed)["rows"] if r["updates"] == row)


e40_red = lambda rule, seed, row: e40_row(rule, seed, row)["scores"]["summary"]["reduction_score"]
R48 = lambda rule, seed, rank: float(np.mean([red(rule, seed, rank, u) for u in (128, 256, 512)]))
R40 = lambda rule, seed: float(np.mean([e40_red(rule, seed, u) for u in (128, 256, 512)]))


def reading(diffs):
    if sum(d >= -10 for d in diffs) >= 2:
        return "no price"
    if sum(d <= -30 for d in diffs) >= 2:
        return "a price"
    return "some price"


# =========================================================================================================== A
if "A" in want:
    print("A. the receipts on", WB)
    files = subprocess.run(["git", "ls-tree", "-r", "-l", WB, E48], capture_output=True, text=True, check=True).stdout.splitlines()
    blobs = {ln.split("\t")[1][len(E48):]: (ln.split()[2], int(ln.split()[3])) for ln in files}
    pts = sorted(f for f in blobs if f.endswith(".pt"))
    print(f"   files under exp45/e48: {len(blobs)}; term files: {len(pts)}, {sum(blobs[f][1] for f in pts):,} bytes;"
          f" run receipts: {sum(bool(re.fullmatch(r'e48_[SB]_\d+_r\d+\.json', f)) for f in blobs)}; answer files: {sum(f.endswith('_reduction_answers.json') and '/' not in f for f in blobs)}")
    ok_all = True
    for rule, seed, rank in RUNS:
        d = receipt(rule, seed, rank)
        rows = d["rows"]
        flags = dict(complete=d["complete"], rows=[r["updates"] for r in rows] == list(ROWS),
                     finite=all(r["finite_losses"] for r in rows),
                     check2=rows[0]["check2"]["scores_equal_e40_row0"] and rows[0]["scores"] == e40_row(rule, seed, 0)["scores"],
                     check3_digest=all(r["check3"]["earlier_digest_equals_row0"] and r["earlier_answers_sha256"] == rows[0]["earlier_answers_sha256"] for r in rows),
                     check3_counts=all(r["check3"]["earlier_counts"] == {n: e40_row(rule, seed, 0)["scores"]["earlier_k12"][n]["exact"] for n in EARLIER6}
                                       and r["scores"]["earlier_k12"] == rows[0]["scores"]["earlier_k12"] for r in rows),
                     check3_lived=all(r["check3"]["equals_life_end"] and r["check3"]["lived_tensors_sha256_codes_reset"] == d["start"]["receipt_tensors_sha256"] for r in rows),
                     check4=all(r["data_digest"]["sha256"] == e40_row(rule, seed, r["updates"])["data_digest"]["sha256"] for r in rows if r["updates"] >= 64),
                     presented=all(r["presented"]["held_out_presented"] == 0 and r["presented"]["not_taught_presented"] == 0 and r["presented"]["earlier_code_items"] == 0
                                   and r["presented"]["items"] == 64 * r["updates"] for r in rows),
                     count=d["numbers_that_may_change"] == 2304 * rank + 32,
                     trainable=d["trainable"] == ["book.weight", "term.down_i", "term.up_i", "term.down_h", "term.up_h"],
                     optimizer=d["optimizer"]["groups"] == [dict(params="term (4 factors)", weight_decay=0.25), dict(params="book.weight", weight_decay=0.0)]
                     and d["optimizer"]["lr"] == 0.001 and d["optimizer"]["betas"] == [0.9, 0.98],
                     first_grad_zero=d["grads_updates_1_2"][0]["down_i"] == 0 and d["grads_updates_1_2"][0]["down_h"] == 0
                     and d["grads_updates_1_2"][1]["down_i"] > 0 and d["grads_updates_1_2"][1]["down_h"] > 0,
                     source=d["provenance"]["git"]["commit"].startswith("f8f62df6") and d["provenance"]["git"]["sources_modified_vs_commit"] == []
                     and d["provenance"]["e48_gated_is_spec"],
                     term_rows=[r["updates"] for r in rows if "term_file" in r] == ([16, 32, 64] if rank in (4, 16) else []) + [128, 256, 512, 1024],
                     forced_rows=[r["updates"] for r in rows if r["scores_gate_forced_off"]] == [256, 1024],
                     answers=sorted(map(int, jload(E48 + run_name(rule, seed, rank) + "_reduction_answers.json", WB)["rows"])) == [0, 64, 128, 256, 512, 1024])
        bad = [k for k, v in flags.items() if not v]
        ok_all &= not bad
        if bad:
            print(f"   {run_name(rule, seed, rank)}: FAILS {bad}")
    print(f"   every one of the 18 receipts: complete, rows {ROWS}, finite losses, row 0 equal to E40's row 0 on every score (check 2), the earlier panels' digest and six counts"
          f" at every row equal to row 0's and to E40's row 0 (check 3), the lived tensors' hash with the code rows reset equal to the life-end file's (check 3), the batch digest at rows"
          f" 64 to 1,024 equal to E40's (check 4), no held-out, untaught or earlier-code item presented, the count 2,304 r + 32, the optimizer groups, the first-update gradients: {ok_all}")
    print("   the seven check receipts:")
    for c in ("check1", "check2", "check3", "check3a_check5_S", "check3a_check5_B", "check4", "check6_dry_and_forecast"):
        d = jload(E48 + f"checks/E48_{c}.json", WB); r = d["result"]
        extra = ""
        if c == "check1":
            extra = f"copy identical {r['identical']}, stdout equals the Mac output exactly {r['stdout_equals_mac_output_exactly']}"
        if c == "check2":
            extra = f"{sum(v['ok'] for v in r['rows'].values())} of {len(r['rows'])} networks equal"
        if c == "check3":
            extra = f"{sum(v['ok'] for v in r['runs'].values())} of {r['n']} runs"
        if c == "check4":
            extra = f"{sum(v['ok'] for v in r['runs'].values())} of {len(r['runs'])} runs"
        if c.startswith("check3a"):
            cm = r["check3a"]["comparisons"]
            extra = "; ".join(f"{k}: gate {v['gate']}, bit-equal {v['bit_equal']}, largest difference {v['max_abs_diff']:.1f}" for k, v in cm.items()) + \
                    f"; update 1 gradient of down {r['check5']['update1']['down_i']:.0e}/{r['check5']['update1']['down_h']:.0e}, update 2 {r['check5']['update2']['down_i']:.1e}/{r['check5']['update2']['down_h']:.1e}"
        if c.startswith("check6"):
            extra = f"forecast {r['forecast_18_runs_cpu_hours_rank16_basis']:.2f} CPU-hours; actual {sum(receipt(*x)['cpu_total'] for x in RUNS) / 3600:.2f}"
        print(f"      {c}: ok {r.get('ok')} | {extra}")
    # the sources
    blob = lambda path, ref: subprocess.run(["git", "rev-parse", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout.strip()
    sha_blob = lambda path, ref: hashlib.sha256(git_bytes(path, ref)).hexdigest()
    src = receipt("S", 73129, 16)["provenance"]
    same_spec = blob(M + "e48_gated.py", WB) == blob("PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py", DREF)
    own = {s: sha_blob(M + s, WB) == h for s, h in src["e48_sources_sha256"].items()}
    base = {s: sha_blob(M + s, WB) == h for s, h in src["sources_sha256"].items()}
    all_same = all(receipt(*x)["provenance"]["e48_sources_sha256"] == src["e48_sources_sha256"] and receipt(*x)["provenance"]["sources_sha256"] == src["sources_sha256"] for x in RUNS)
    unchanged = [s for s in list(src["sources_sha256"]) + ["e40_new_family.py"] if blob(M + s, WB) == blob(M + s, "f8f62df6")]
    print(f"   sources: e48_gated.py on the worker's branch is the same blob as the specification file on the default branch: {same_spec};"
          f" the committed runner, copy and e40_new_family.py have the SHA-256 the receipts name: {all(own.values())}; the ten harness sources too: {all(base.values())};"
          f" the 18 receipts name the same sources: {all_same}; harness files identical to the brief's commit f8f62df6: {len(unchanged)} of {len(src['sources_sha256']) + 1}")
    runner = show(M + "exp48_gated_run.py", WB)
    spec = show(M + "e48_gated.py", WB)
    report = show(E48 + "E48_REPORT.md", WB)
    six = {
        "1 the reading on the mean of rows 128, 256, 512": "R = mean reduction score at rows 128, 256, 512" in report and "82.65, 78.93, 81.45" in report,
        "2 the gate read from x[:, 1], asserted for the call, installed forward, keep_code_rows before the step, check 3a":
            all(s in runner for s in ("g = (x[:, 1] >= G.CODES[0]) & (x[:, 1] <= G.CODES[-1])", "assert bool((g == on).all())", "E.forward = gated_forward",
                                      "loss.backward()\n    G.keep_code_rows(m)", "G.freeze(m); opt = G.make_opt(m, term)")) and jload(E48 + "checks/E48_check3a_check5_S.json", WB)["result"]["check3a"]["ok"],
        "3 digests of the earlier answers at every row; batch digest; answers at six rows":
            all("earlier_answers_sha256" in r for x in RUNS for r in receipt(*x)["rows"]) and "ANSWER_ROWS = (0, 64, 128, 256, 512, 1024)" in runner,
        "4 the term files (five plain float32 tensors; rows 128 to 1,024 for every run)": len(pts) == 108 and "TERM_ROWS_ALL = (128, 256, 512, 1024)" in runner
            and "v.dtype == T.float32 and not isinstance(v, T.nn.Parameter)" in runner,
        "5 the replay comparison on stream entries": "12,330 / 49,194 / 196,650" in report and "4.5 / 17.9 / 71.4" in report,
        "6 what decay acts on, and a term under the standing optimizer": "only the\nterm's factors decay" in report and "0.36 after 2,048 updates and 0.02 after 8,192" in report,
    }
    for k, v in six.items():
        print(f"   audit change {k}: {v}")
    gate_lines = [ln.strip() for ln in spec.splitlines() if "gate" in ln and ("if gate" in ln or "def forward" in ln)]
    print(f"   the specification's forward uses the gate in these lines only: {gate_lines}")
    print(f"   [{time.time() - T0:.0f} s]", flush=True)

# =========================================================================================================== B
if "B" in want:
    print("B. the reader on the committed files (life-end weights from the default branch, term files from the worker's branch)")
    # the life-end files and row 0
    for rule in RULES:
        for seed in SEEDS:
            W, fsha = life_end(rule, seed)
            st = receipt(rule, seed, 16)["start"]
            zero = G.new_term(16, np.random.default_rng(1))
            s, red_out = score_new(W, zero, seed); e, eouts = score_earlier(W, zero, seed)
            row0 = receipt(rule, seed, 16)["rows"][0]
            dig = dict(per_panel={n: sha_json(eouts[n]) for n in EARLIER6}, all=sha_json([eouts[n] for n in EARLIER6]))
            _cache[("dig", rule, seed)] = dig
            ans = jload(E48 + run_name(rule, seed, 16) + "_reduction_answers.json", WB)
            plain = e40_panels(seed)[2]
            prompts_same = all(ans["prompts"][n] == [[list(p), list(t)] for p, t in plain[n]["reduction"]] for n in ("add83", "sub83"))
            print(f"   lived {rule} {seed}: file SHA-256 equals the receipt's {fsha == st['weights_sha256']}; tensors hash (recomputed in numpy) equals the life-end receipt's {tensors_sha(W) == st['receipt_tensors_sha256']};"
                  f" row 0 from the reader: the 12 counts of the new family equal {all(s[k] == row0['scores'][k] for k in s)}, the 6 earlier counts equal {e == row0['scores']['earlier_k12']},"
                  f" the reduction answers equal the answer file's row 0 item for item {all(red_out[n] == ans['rows']['0'][n] for n in red_out)}, the prompts and targets of the answer file are the reader's {prompts_same},"
                  f" the digests of the six earlier panels' answers equal the receipt's {dig == row0['earlier_answers_sha256']}", flush=True)
    n_files = n_ok = n_items = n_same = n_rows_ans = 0
    bad = []
    forced_ok = forced_n = 0
    for rule, seed, rank in RUNS:
        if quick and rank != 16:
            continue
        d = receipt(rule, seed, rank); W, _ = life_end(rule, seed)
        ans = jload(E48 + run_name(rule, seed, rank) + "_reduction_answers.json", WB)
        for row in d["rows"]:
            if "term_file" not in row or (quick and row["updates"] not in (256, 1024)) or (only and only not in row["term_file"]["file"]):
                continue
            u = row["updates"]; path = E48 + row["term_file"]["file"]
            raw = git_bytes(path, WB)
            term = G.load_term(path, WB)
            shapes = (term["up_i"].shape == (768, rank) and term["down_i"].shape == (rank, 512) and term["up_h"].shape == (768, rank)
                      and term["down_h"].shape == (rank, 256) and term["code_rows"].shape == (4, 8))
            hashes = (hashlib.sha256(raw).hexdigest() == row["term_file"]["sha256"] and len(raw) == row["term_file"]["bytes"]
                      and tensors_sha(term) == row["term_file"]["tensors_sha256"]
                      and tensors_sha({"code_rows": term["code_rows"]}) == row["code_rows_tensors_sha256"])
            norms = all(abs(float(np.linalg.norm(term[k])) - row["term_norms"][k]) < 1e-3 for k in ("down_i", "up_i", "down_h", "up_h"))
            V = G.with_code_rows(W, term["code_rows"])
            s, red_out = score_new(V, term, seed); e, eouts = score_earlier(V, term, seed)
            sm = summary(s)
            counts = all(s[k] == row["scores"][k] for k in s) and e == row["scores"]["earlier_k12"] and all(abs(sm[k] - row["scores"]["summary"][k]) < 1e-9 for k in sm)
            digest = dict(per_panel={n: sha_json(eouts[n]) for n in EARLIER6}, all=sha_json([eouts[n] for n in EARLIER6])) == row["earlier_answers_sha256"]
            items = True
            if str(u) in ans["rows"]:
                n_rows_ans += 1
                for n in ("add83", "sub83"):
                    same = sum(a == b for a, b in zip(red_out[n], ans["rows"][str(u)][n])); n_items += len(red_out[n]); n_same += same
                    items &= same == len(red_out[n]) == len(ans["rows"][str(u)][n])
            fo = True
            if row["scores_gate_forced_off"]:
                off = dict(term, up_i=np.zeros_like(term["up_i"]), up_h=np.zeros_like(term["up_h"]))      # a term whose product is zero: the gate forced off
                so, _ = score_new(V, off, seed); f = row["scores_gate_forced_off"]
                fo = all(so[k] == f[k] for k in so) and f["earlier_k12"] == row["scores"]["earlier_k12"]
                forced_n += 1; forced_ok += fo
            n_files += 1
            good = shapes and hashes and norms and counts and digest and items and fo
            n_ok += good
            if not good:
                bad.append((row["term_file"]["file"], dict(shapes=shapes, hashes=hashes, norms=norms, counts=counts, digest=digest, items=items, forced=fo)))
            if not counts:                                            # which count, by how much, and how close the two largest outputs are there
                held, taught, plain, _ = e40_panels(seed)
                for kind, mine, theirs, panels_ in (("held_k12", s["held_k12"], row["scores"]["held_k12"], held), ("taught_k12", s["taught_k12"], row["scores"]["taught_k12"], taught)) + tuple(
                        (f"held_plain_k0 {n}", s["held_plain_k0"][n], row["scores"]["held_plain_k0"][n], plain[n]) for n in plain):
                    for n in mine:
                        if mine[n] != theirs[n]:
                            x, tgt = batch(panels_[n])
                            lg = G.forward(V, term, x)[1][:, -3:, :].astype(np.float64); srt = np.sort(lg, axis=-1); margin = (srt[..., -1] - srt[..., -2]).min(1)
                            i = int(margin.argmin())
                            print(f"      {row['term_file']['file']}: {kind} {n}: the reader counts {mine[n]['exact']} of {mine[n]['n']}, the receipt {theirs[n]['exact']};"
                                  f" on the item `{bytes(panels_[n][i][0]).decode()}` the two largest outputs differ by {margin[i]:.1e} (single-precision rounding); the next smallest difference on the panel is {np.sort(margin)[1]:.1e}")
        print(f"   {run_name(rule, seed, rank)}: term files read so far {n_files}, in full agreement {n_ok}  [{time.time() - T0:.0f} s]", flush=True)
    print(f"   term files read: {n_files}; for each: the five tensors' shapes, the file's SHA-256 and size, the tensors hash and the code rows' hash as in the receipt, the four norms,"
          f" the 18 counts and the four summary scores of the row from the reader, and the digests of the six earlier panels' answers: in full agreement {n_ok} of {n_files}")
    print(f"   reduction answers, reader against the worker's answer files: {n_same:,} of {n_items:,} items the same bytes, over {n_rows_ans} rows of answers")
    print(f"   the scores with the gate forced off (rows 256 and 1,024), reader against receipt: {forced_ok} of {forced_n} rows equal on all 12 counts")
    for b in bad:
        print("   not in full agreement:", b)
    print(f"   [{time.time() - T0:.0f} s]", flush=True)

# =========================================================================================================== C
if "C" in want:
    print("C. keeping: the earlier families with and without the term loaded")

    def wide_set(seed):
        """Earlier items beyond E40's six panels: the twelve tasks of the three earlier families at every length, the compositions,
        and every held-out add and sub pair that needs the reduction, without letters."""
        _, _, _, earlier = e40_panels(seed)
        groups = {}
        for t in LIFE + F3:
            for k in KS_LIFE:
                groups[f"{name(t)} k{k}"] = earlier[name(t)][k]
        for t in DEV:
            pn = panel(t, split(t, .5, seed)[1], seed, (0, 12, 40), 300)
            for k in (0, 12, 40):
                groups[f"{name(t)} k{k}"] = pn[k]
        comp, _ = comp_panels(seed)
        for k, v in comp.items():
            if k != "taught_sample":
                groups[f"composition {k}"] = v
        for t, frac in [(t, .5) for t in DEV[:2]] + [(t, .25) for t in LIFE[:2] + F3[:2]]:
            hp = split(t, frac, seed)[1]
            groups[f"{name(t)} reduction pairs"] = [render(t, a, b, 0, Random(0)) for a, b in hp if ((a + b >= t[1]) if t[0] == "add" else (a - b < 0))]
        return groups

    right = lambda outs, items: sum(o == t for o, (_, t) in zip(outs, items))
    fam = lambda g: "birth family, mod 89" if "89" in g.split()[0] else "life family, mod 97" if "97" in g.split()[0] else "life family, mod 73" if "73" in g.split()[0] else "compositions (mod 97)"
    checked = False
    for rule in RULES:
        for seed in (SEEDS if not quick else SEEDS[:1]):
            W, _ = life_end(rule, seed)
            groups = wide_set(seed)
            used = sorted({b for it in groups.values() for p, t in it for b in list(p) + list(t)})
            pos1 = sorted({p[0] for it in groups.values() for p, _ in it})
            for rank in ((64,) if not quick else (16,)):
                term = G.load_term(E48 + f"{run_name(rule, seed, rank)}_row1024.pt", WB)
                V = G.with_code_rows(W, term["code_rows"])
                moved = float(np.abs(V["book.weight"] - W["book.weight"]).max())
                zero = G.new_term(4, np.random.default_rng(0))
                if not checked:                                       # the local generator against the audit's reader, once
                    it = groups["add97 k12"]; new = e40_panels(seed)[0]["add83"]
                    same = (answers(V, term, [p for p, _ in it]) == G.generate(V, term, [p for p, _ in it]) and answers(V, term, [p for p, _ in new]) == G.generate(V, term, [p for p, _ in new])
                            and answers(V, term, [p for p, _ in new[:64]]) == G.generate_slow(V, term, [p for p, _ in new[:64]]))
                    print(f"   the generator of this script (explicit gate) gives the audit reader's bytes on an earlier panel and on a new-family panel: {same}"); checked = True
                tot = {}; per = {}
                equal = True; gen_codes = 0; same_answers = True; biggest = 0.0; n_logit = 0
                for g, it in groups.items():
                    pr = [p for p, _ in it]
                    o0 = answers(W, zero, pr, gate=False)              # the life-end network
                    o1 = answers(V, term, pr)                          # the term and its code rows loaded; the gate read from the code byte, as the runner reads it
                    o2 = answers(V, term, pr, gate=True)               # the same term applied to the earlier item
                    same_answers &= o0 == o1; gen_codes += sum(b in G.CODES for o in o1 for b in o)
                    x, _ = batch(it[:48]); n_logit += len(x)
                    ref = plain_forward(W, x)[3]
                    equal &= np.array_equal(G.forward(V, term, x)[1], ref) and not G.gate_of(x)
                    biggest = max(biggest, float(np.abs(G.forward(V, term, x, gate=True)[1] - ref).max()))
                    f = fam(g) + (", reduction pairs without letters" if "reduction" in g else "")
                    a = tot.setdefault(f, [0, 0, 0, 0]); a[0] += len(it); a[1] += right(o0, it); a[2] += right(o1, it); a[3] += right(o2, it)
                    per[g] = (100 * right(o0, it) / len(it), 100 * right(o2, it) / len(it))
                print(f"   lived {rule} {seed}, the rank-{rank} term of row 1,024 and its code rows (largest change of a code-row entry {moved:.2f}); {sum(len(v) for v in groups.values()):,} earlier items in {len(groups)} groups;"
                      f" their bytes: {len(used)} values, any of 101 to 104: {bool(set(used) & set(G.CODES))}; bytes after the start symbol: {bytes(pos1).decode()}")
                print(f"      answers with the term loaded and the gate read from the code byte equal the life-end network's item for item: {same_answers}; bytes e f g h among the generated bytes: {gen_codes};"
                      f" logits equal value for value on {n_logit:,} sampled items: {equal}; with the gate forced on the sampled logits move by up to {biggest:.1f}")
                for f, (n, b, g_, fo) in tot.items():
                    print(f"      {f:58s} {n:6,} items | life-end {100 * b / n:5.1f}% | term loaded, gate from the code {100 * g_ / n:5.1f}% | gate forced on {100 * fo / n:5.1f}%")
                up = sorted(((v[1] - v[0], g) for g, v in per.items() if v[1] - v[0] >= 5), reverse=True)
                print(f"      groups that are answered better with the gate forced on, by 5 points or more ({len(up)} of {len(per)}): " + "; ".join(f"{g} {per[g][0]:.0f} to {per[g][1]:.0f}" for _, g in up[:10]))
                print(f"      [{time.time() - T0:.0f} s]", flush=True)
    print("   E40's side, from its receipts: the six earlier panels (2,200 items at 12 letters), percent right at rows 0, 64, 128, 256, 512, 1,024, 8,192; E48 is row 0 at every row")
    for rule in RULES:
        for seed in SEEDS:
            v = []
            for u in (0, 64, 128, 256, 512, 1024, 8192):
                e = e40_row(rule, seed, u)["scores"]["earlier_k12"]
                v.append(100 * sum(e[n]["exact"] for n in EARLIER6) / sum(e[n]["n"] for n in EARLIER6))
            life4 = lambda u: 100 * sum(e40_row(rule, seed, u)["scores"]["earlier_k12"][n]["exact"] for n in EARLIER6[2:]) / 1600
            print(f"      E40 lived {rule} {seed}: " + ", ".join(f"{x:.1f}" for x in v) + f" | the four life panels (mod 97 and mod 73) alone: {life4(0):.1f} at row 0, {life4(64):.1f} at row 64, {life4(256):.1f} at row 256,"
                  f" {life4(1024):.1f} at row 1,024, {life4(8192):.1f} at row 8,192")
    print(f"   [{time.time() - T0:.0f} s]", flush=True)

# =========================================================================================================== D
if "D" in want:
    print("D. the reading, from the receipts")
    print("   R = mean of the reduction score at rows 128, 256 and 512; E48 / E40 / difference, seeds 73129, 93683, 116089")
    for rule in RULES:
        print(f"   E40 lived {rule}: R = " + ", ".join(f"{R40(rule, s):.2f}" for s in SEEDS))
        for rank in RANKS:
            diffs = [R48(rule, s, rank) - R40(rule, s) for s in SEEDS]
            print(f"      {RULE_NAME[rule]:15s} rank {rank:2d}: " + " | ".join(f"{R48(rule, s, rank):.2f} / {R40(rule, s):.2f} / {d:+.2f}" for s, d in zip(SEEDS, diffs)) + f" | {reading(diffs)}")
    print("   the count at row 1,024: the reduction score of the E48 run against that of E40's run of the same seed and rule at the same row")
    for u in (64, 128, 256, 512, 1024):
        above = [(x, red(*x, u) - e40_red(x[0], x[1], u)) for x in RUNS]
        below = [f"{RULE_NAME[x[0]]} {x[1]} rank {x[2]} ({d:+.1f})" for x, d in above if d < 0]
        print(f"      row {u:5,}: E48 at or above E40 in {sum(d >= 0 for _, d in above):2d} of 18 runs" + (f"; below: {', '.join(below)}" if len(below) <= 4 else f"; below in {len(below)}"))
    first = lambda vals, thr: next((u for u, v in vals if v >= thr), None)
    fmt = lambda u: "never" if u is None else f"{u:,}"
    print("   first scored row at which a run is at or above a level (rows 0, 16, 32, 64, 128, 256, 512, 1,024; E40 has no rows 16 and 32)")
    print("   run | taught panel (add and sub, 12 letters) 80%, 90% | never-taught reduction pairs 80%, 90% | E40's R of the seed | E40's score at the same row")
    for rule in RULES:
        for seed in SEEDS:
            e40 = e40_receipt(rule, seed)["rows"]
            et = [(r["updates"], r["scores"]["summary"]["taught_add_sub_mean"]) for r in e40]; er = [(r["updates"], r["scores"]["summary"]["reduction_score"]) for r in e40]
            print(f"      E40 lived {rule} {seed}: taught {fmt(first(et, 80))}, {fmt(first(et, 90))} | reduction {fmt(first(er, 80))}, {fmt(first(er, 90))} (highest in 8,192 updates {max(v for _, v in er):.1f})")
            for rank in RANKS:
                rows = receipt(rule, seed, rank)["rows"]
                t = [(r["updates"], r["scores"]["summary"]["taught_add_sub_mean"]) for r in rows]; rr = [(r["updates"], r["scores"]["summary"]["reduction_score"]) for r in rows]
                same = next((u for u, v in rr if u >= 64 and v >= e40_red(rule, seed, u)), None)
                print(f"         E48 rank {rank:2d}: taught {fmt(first(t, 80))}, {fmt(first(t, 90))} | reduction {fmt(first(rr, 80))}, {fmt(first(rr, 90))} | {fmt(first(rr, R40(rule, seed)))} | {fmt(same)}"
                      f" | at 1,024: taught {t[-1][1]:.1f}, reduction {rr[-1][1]:.1f} (E40 {e40_red(rule, seed, 1024):.1f})")
    print("   the order of the ranks: of the 42 cells (rule, seed, row from 16 on), the reduction score rises with the rank (4 <= 16 <= 64) in", end=" ")
    cells = [(rule, seed, u) for rule in RULES for seed in SEEDS for u in ROWS[1:]]
    mono = [c for c in cells if red(c[0], c[1], 4, c[2]) <= red(c[0], c[1], 16, c[2]) <= red(c[0], c[1], 64, c[2])]
    print(f"{len(mono)}; the exceptions: " + ", ".join(f"{RULE_NAME[r]} {s} row {u} ({red(r, s, 4, u):.1f}, {red(r, s, 16, u):.1f}, {red(r, s, 64, u):.1f})" for r, s, u in cells if (r, s, u) not in mono))
    for u in (64, 256, 1024):
        print(f"      row {u:5,}: mean over seeds of the reduction score, ranks 4 / 16 / 64: " + "; ".join(
            f"{RULE_NAME[rule]} " + " / ".join(f"{np.mean([red(rule, s, rank, u) for s in SEEDS]):.1f}" for rank in RANKS) + f" (E40 {np.mean([e40_red(rule, s, u) for s in SEEDS]):.1f})" for rule in RULES))
    print("   the two rules: difference from E40 in R, sweep minus backpropagation, by seed; and the two rules' own R")
    for rank in RANKS:
        d = [(R48("S", s, rank) - R40("S", s)) - (R48("B", s, rank) - R40("B", s)) for s in SEEDS]
        print(f"      rank {rank:2d}: " + ", ".join(f"{x:+.2f}" for x in d) + f" | R under the sweep " + ", ".join(f"{R48('S', s, rank):.1f}" for s in SEEDS)
              + " | under backpropagation " + ", ".join(f"{R48('B', s, rank):.1f}" for s in SEEDS))
    print("      E40 itself, R under backpropagation minus R under the sweep: " + ", ".join(f"{R40('B', s) - R40('S', s):+.2f}" for s in SEEDS))
    print("   row-to-row movement of the reduction score from row 128 on (how even the curves are): largest fall between neighbouring rows")
    for lab, getter, rows_ in (("E40 (every weight free)", lambda rule, s, rank, u: e40_red(rule, s, u), (128, 256, 512, 1024)), ("E48", red, (128, 256, 512, 1024))):
        falls = [getter(rule, s, rank, a) - getter(rule, s, rank, b) for rule in RULES for s in SEEDS for rank in (RANKS if lab == "E48" else (0,)) for a, b in zip(rows_, rows_[1:])]
        print(f"      {lab}: {sum(f > 0 for f in falls)} of {len(falls)} steps fall; the largest fall {max(falls):.1f} points")
    print(f"   [{time.time() - T0:.0f} s]", flush=True)

# =========================================================================================================== E
if "E" in want:
    print("E. the predictions, scored from the receipts")
    rd = {(rule, rank): reading([R48(rule, s, rank) - R40(rule, s) for s in SEEDS]) for rule in RULES for rank in RANKS}
    df = {(rule, rank): [R48(rule, s, rank) - R40(rule, s) for s in SEEDS] for rule in RULES for rank in RANKS}
    f3 = lambda v: ", ".join(f"{x:+.2f}" for x in v)
    p1 = all(receipt(*x)["rows"][0]["check2"]["scores_equal_e40_row0"] and all(r["check3"]["earlier_digest_equals_row0"] and r["check3"]["earlier_counts_equal_e40_row0"]
                                                                                 and r["check3"]["equals_life_end"] for r in receipt(*x)["rows"]) for x in RUNS)
    print(f"   P48.1 (90%) checks 2 and 3 pass for all six networks: {p1}")
    print(f"   P48.2 (50%) rank 16, backpropagation, no price: {rd[('B', 16)] == 'no price'} ({f3(df[('B', 16)])})")
    print(f"   P48.3 (35%) rank 16, sweep, no price: {rd[('S', 16)] == 'no price'} ({f3(df[('S', 16)])})")
    print(f"   P48.4 (65%) rank 64, backpropagation, no price: {rd[('B', 64)] == 'no price'} ({f3(df[('B', 64)])})")
    print(f"   P48.5 (65%) rank 4, some price or a price under both rules: {rd[('S', 4)] != 'no price' and rd[('B', 4)] != 'no price'} (sweep: {rd[('S', 4)]}, {f3(df[('S', 4)])}; backpropagation: {rd[('B', 4)]}, {f3(df[('B', 4)])})")
    per_rank = {rank: [df[("S", rank)][i] - df[("B", rank)][i] for i in range(3)] for rank in RANKS}
    holds = {rank: sum(x <= -10 for x in v) >= 2 for rank, v in per_rank.items()}
    print(f"   P48.6 (40%) the difference from E40 lower under the sweep by 10 points or more in two of three seeds, at two of three ranks: {sum(holds.values()) >= 2}"
          f" (" + "; ".join(f"rank {rank}: {f3(v)}, holds {holds[rank]}" for rank, v in per_rank.items()) + ")")
    m2 = lambda f, *a: float(np.mean([f(*a, u) for u in (512, 1024)]))
    p7 = {}
    for rule in RULES:
        d = [m2(red, rule, s, 16) - m2(e40_red, rule, s) for s in SEEDS]
        p7[rule] = (sum(abs(x) <= 10 for x in d) >= 2, d, [m2(e40_red, rule, s) for s in SEEDS])
    print(f"   P48.7 (55%) the mean of rows 512 and 1,024 at rank 16 within 10 points of E40's, two of three seeds, each rule: {p7['S'][0] and p7['B'][0]}"
          f" (sweep {f3(p7['S'][1])} against E40's {', '.join(f'{x:.2f}' for x in p7['S'][2])}; backpropagation {f3(p7['B'][1])} against {', '.join(f'{x:.2f}' for x in p7['B'][2])})")
    chance = [.90, .50, .35, .65, .65, .40, .55]
    out = [p1, rd[("B", 16)] == "no price", rd[("S", 16)] == "no price", rd[("B", 64)] == "no price", rd[("S", 4)] != "no price" and rd[("B", 4)] != "no price", sum(holds.values()) >= 2, p7["S"][0] and p7["B"][0]]
    brier = float(np.mean([(c - o) ** 2 for c, o in zip(chance, out)]))
    print(f"   {sum(out)} of 7 true; mean squared error of the stated chances {brier:.3f} (0.25 for a chance of one half throughout)")
    print("   rank 4 under backpropagation, the miss of P48.5: the three differences are " + f3(df[("B", 4)]) + "; at rows 128 / 256 / 512 the E48 run is at "
          + "; ".join(f"{s}: " + " / ".join(f"{red('B', s, 4, u):.1f}" for u in (128, 256, 512)) + " against E40's " + " / ".join(f"{e40_red('B', s, u):.1f}" for u in (128, 256, 512)) for s in SEEDS))

# =========================================================================================================== F
if "F" in want:
    print("F. cost in weights, directions used, and what the wrong answers are")
    total = receipt("S", 73129, 4)["network_parameters"]
    taught_pairs = round(.10 * 83 * 83); entries = 4 * taught_pairs
    print(f"   the family's taught set: {taught_pairs} pairs an operation, {entries:,} entries of 3 bytes = {3 * entries:,} bytes; the network has {total:,} numbers")
    for r in RANKS:
        n = receipt("S", 73129, r)["numbers_that_may_change"]
        print(f"      rank {r:2d}: {n:7,} numbers = {100 * n / total:5.2f}% of the network = {4 * n:7,} bytes at 4 a number = {4 * n / (3 * entries):5.1f} times the taught set as entries;"
              f" {n / entries:5.1f} numbers a taught entry; {n / (4 * 83 * 83):5.2f} numbers for each of the family's {4 * 83 * 83:,} pairs")

    def spectrum(up, down):
        qa, ra = np.linalg.qr(up.astype(np.float64)); qb, rb = np.linalg.qr(down.astype(np.float64).T)
        return np.linalg.svd(ra @ rb.T, compute_uv=False)

    print("   directions used by the trained products at row 1,024 (singular values of up . down): number at or above a tenth of the largest, and the participation number (sum s^2)^2 / sum s^4")
    for rule in RULES:
        for rank in RANKS:
            cells = []
            for seed in SEEDS:
                t = G.load_term(E48 + f"{run_name(rule, seed, rank)}_row1024.pt", WB)
                W, _ = life_end(rule, seed)
                si, sh = spectrum(t["up_i"], t["down_i"]), spectrum(t["up_h"], t["down_h"])
                pn = lambda s: float((s ** 2).sum() ** 2 / (s ** 4).sum())
                rel_i = float(np.linalg.norm(t["up_i"] @ t["down_i"]) / np.linalg.norm(W["rnn.weight_ih_l0"])); rel_h = float(np.linalg.norm(t["up_h"] @ t["down_h"]) / np.linalg.norm(W["rnn.weight_hh_l0"]))
                cells.append(f"{seed}: input map {int((si >= si[0] / 10).sum())} ({pn(si):.1f}), size {rel_i:.2f} of the matrix; recurrent {int((sh >= sh[0] / 10).sum())} ({pn(sh):.1f}), size {rel_h:.2f}")
            print(f"      {RULE_NAME[rule]:15s} rank {rank:2d}: " + " | ".join(cells))

    print("   the size of the product up . down relative to the frozen matrix (Frobenius norms), input map / recurrent matrix, by row; beside it the size of E40's change of the same matrix")
    for rule in RULES:
        for seed in SEEDS:
            W, _ = life_end(rule, seed)
            ni, nh = np.linalg.norm(W["rnn.weight_ih_l0"]), np.linalg.norm(W["rnn.weight_hh_l0"])
            e40 = []
            for u in (64, 8192):
                Wu, _ = load_weights(M + f"exp45/e40/e40_lived_{rule}_{seed}_row{u}.pt", E40W)
                e40.append(f"row {u:,}: {np.linalg.norm(Wu['rnn.weight_ih_l0'] - W['rnn.weight_ih_l0']) / ni:.2f} / {np.linalg.norm(Wu['rnn.weight_hh_l0'] - W['rnn.weight_hh_l0']) / nh:.2f}")
            print(f"      lived {rule} {seed}: E40's change, " + "; ".join(e40))
            for rank in RANKS:
                cells = []
                for u in ((16, 64, 256, 1024) if rank != 64 else (128, 256, 1024)):
                    t = G.load_term(E48 + f"{run_name(rule, seed, rank)}_row{u}.pt", WB)
                    cells.append(f"row {u:,}: {np.linalg.norm(t['up_i'] @ t['down_i']) / ni:.2f} / {np.linalg.norm(t['up_h'] @ t['down_h']) / nh:.2f}")
                print(f"         E48 rank {rank:2d}: " + "; ".join(cells))

    n_cos = ok_cos = 0; lo, hi = 1.0, 0.0
    for rule, seed, rank in RUNS:
        if rule != "S":
            continue
        for row in receipt(rule, seed, rank)["rows"]:
            if "term_file" in row:
                t = G.load_term(E48 + row["term_file"]["file"], WB)
                for k in ("down_i", "up_i", "down_h", "up_h"):
                    F = t[k].astype(np.float64).ravel(); c = float(np.abs(F).sum() / (np.linalg.norm(F) * np.sqrt(F.size)))      # cosine of F with sign(F) x mean|F|
                    n_cos += 1; ok_cos += abs(c - row["sweep_cosines"][k]) < 1e-6
                    if row["updates"] >= 64:
                        lo, hi = min(lo, c), max(hi, c)
    print(f"   under the sweep, the cosine of each factor with what the sweep uses in its place (the signs times the mean size), recomputed from the term files: {ok_cos} of {n_cos} equal the receipts' to six decimals;"
          f" from row 64 on they lie between {lo:.2f} and {hi:.2f}")

    def classify(prompt, target, out):
        """What a wrong answer to a reduction pair is.  prompt: code, a a, symbol, b b, '='."""
        code = prompt[0]; a = int(bytes(prompt[1:3])); b = int(bytes(prompt[4:6])); op = "add" if code == ord("e") else "sub"
        c = int(bytes(target[:2]))
        well = len(out) == 3 and out[2] == EOS and all(48 <= v <= 57 for v in out[:2])
        if not well:
            return "not two digits and the end symbol"
        o = int(bytes(out[:2]))
        raw = a + b if op == "add" else a - b
        tags = []
        if op == "add" and o == raw % 100:
            tags.append("the sum without the reduction (its last two digits)")
        if op == "sub" and o == (b - a) % 100:
            tags.append("b - a (the difference the other way round)")
        for mod in (89, 97, 73):
            if o == raw % mod:
                tags.append(f"the answer under modulus {mod}")
        if tags:
            return tags[0]
        if o // 10 == c // 10:
            return "tens digit right, units digit wrong"
        if o % 10 == c % 10:
            return "units digit right, tens digit wrong"
        return "both digits wrong"

    print("   what the wrong answers on the never-taught reduction pairs are (the worker's answer files; add and sub together), percent of the wrong answers")
    order = ["units digit right, tens digit wrong", "tens digit right, units digit wrong", "both digits wrong", "the sum without the reduction (its last two digits)", "b - a (the difference the other way round)",
             "the answer under modulus 89", "the answer under modulus 97", "the answer under modulus 73", "not two digits and the end symbol"]
    for u in (0, 64, 256, 1024):
        for rule in RULES:
            for rank in ((16,) if u == 0 else RANKS):
                tally = {}; wrong = total_n = 0; near = 0; sg = {}
                for seed in SEEDS:
                    ans = jload(E48 + run_name(rule, seed, rank) + "_reduction_answers.json", WB)
                    for n in ("add83", "sub83"):
                        for (p, t), o in zip(ans["prompts"][n], ans["rows"][str(u)][n]):
                            total_n += 1
                            if o != t:
                                wrong += 1; k = classify(p, t, o); tally[k] = tally.get(k, 0) + 1
                                if len(o) == 3 and all(48 <= v <= 57 for v in o[:2]):
                                    dlt = int(bytes(o[:2])) - int(bytes(t[:2])); near += abs(dlt) in (1, 10)
                                    toward73 = dlt * (1 if n == "add83" else -1)        # +10: the answer under modulus 73, for both operations
                                    sg[toward73] = sg.get(toward73, 0) + 1
                print(f"      row {u:5,} {RULE_NAME[rule]:15s} rank {rank:2d}{' (the life-end network, any rank)' if u == 0 else ''}: {wrong:6,} wrong of {total_n:,} ({100 * wrong / total_n:4.1f}%) | "
                      + "; ".join(f"{k} {100 * tally.get(k, 0) / max(wrong, 1):.0f}" for k in order if tally.get(k, 0)) + f" | off by exactly 1 or 10: {100 * near / max(wrong, 1):.0f}"
                      + f" | by 10 towards the answer under modulus 73: {100 * sg.get(10, 0) / max(wrong, 1):.0f}, by 10 the other way: {100 * sg.get(-10, 0) / max(wrong, 1):.0f},"
                      f" by +1: {100 * sg.get(1, 0) / max(wrong, 1):.0f}, by -1: {100 * sg.get(-1, 0) / max(wrong, 1):.0f} (signs for addition; reversed for subtraction)")
    print(f"   [{time.time() - T0:.0f} s]", flush=True)

# =========================================================================================================== G
if "G" in want:
    print("G. the gate and the position (row 1,024; nothing retrained): the reduction score with the term applied from the start symbol on (as run),")
    print("   from the code byte's position on, and from the position after the code byte on")
    for rule in RULES:
        for rank in ((16, 64) if not quick else (16,)):
            cells = []
            for seed in SEEDS:
                W, _ = life_end(rule, seed)
                term = G.load_term(E48 + f"{run_name(rule, seed, rank)}_row1024.pt", WB)
                V = G.with_code_rows(W, term["code_rows"])
                plain = e40_panels(seed)[2]
                v = []
                for first in (0, 1, 2):
                    v.append(float(np.mean([100 * sum(o == t for o, (_, t) in zip(answers(V, term, [p for p, _ in plain[n]["reduction"]], first=first), plain[n]["reduction"])) / len(plain[n]["reduction"])
                                            for n in ("add83", "sub83")])))
                assert abs(v[0] - red(rule, seed, rank, 1024)) < 1e-9
                cells.append(f"{seed}: {v[0]:.1f} / {v[1]:.1f} / {v[2]:.1f}")
            print(f"      {RULE_NAME[rule]:15s} rank {rank:2d}: " + " | ".join(cells) + f"  [{time.time() - T0:.0f} s]", flush=True)
