"""E20-B runner (brief OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md, Job B; amendments 1-16 govern).

Births and lives of the windowless reference networks G0-G3 (arch20.py) under backprop through time, on the protocol of
exp16_integrated.develop17 (plain path) and exp16_integrated.life arm D, re-implemented here so that nothing on file is reused or
overwritten (amendment 5). exp16_integrated is imported with E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 (amendment 11), so that the
windowless settings hold (k = 40 panels, no store, embedding table without decay); its panels, strict_panels, score, adamw16,
family_entries, entry_stream, joint_stream, zero_grad_report, book_shift are used unchanged.

    python3 exp20_reference.py dev  <net> <seed> [--out DIR]
    python3 exp20_reference.py life <net> <seed> [--out DIR]                                   # birth: DIR/e20_dev_<net>[dose]_<seed>.pt
    python3 exp20_reference.py life G0 <seed> --out DIR --parity-birth FILE --parity-sha SHA   # amendment 9b: start from the reference birth
    python3 exp20_reference.py check [--qual DIR]                                              # writes DIR/E20B_CHECK.json

Dose: E11_DEV_UPDATES and E11_STAGE_UPDATES, read by exp11_cell exactly as the reference reads them (full dose when unset).
No other E16_ or E11_ variable may be set. Every output path must not exist (refused otherwise).
Lives admit only an E20 birth whose complete receipt matches network, seed (receipt, file name and life), dose, optimizer settings,
source hashes and weights hash; there is no legacy admission (amendment 12).
"""
import os
import sys

# ---------------------------------------------------------------- environment, before any harness import (amendments 9, 11)
ALLOWED_ENV = {"E11_DEV_UPDATES", "E11_STAGE_UPDATES"}
REQUIRED_ENV = {"E16_ARCH": "bank2", "E16_BOOKDECAY": "off", "E16_COMPGAP": "0"}
_bad = {k: v for k, v in os.environ.items() if k.startswith(("E16_", "E11_")) and k not in ALLOWED_ENV and REQUIRED_ENV.get(k) != v}
assert not _bad, f"E20-B runs with no other E16_/E11_ variable set: {_bad}"
ENV_AT_START = {k: v for k, v in os.environ.items() if k.startswith(("E16_", "E11_", "OMP_", "OPENBLAS_", "MKL_", "EXP45_", "PYTHONHASHSEED"))}
os.environ.update(REQUIRED_ENV)

import hashlib
import json
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from random import Random

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp16_integrated as X16        # noqa: E402  (installs arch17 bank2 at import)
import exp11_cell as E                # noqa: E402
import exp45 as X                     # noqa: E402
import arch17                         # noqa: E402
import arch20                         # noqa: E402

assert X16.ARCH16 == "bank2" and X16.BOOKDECAY16 == "off" and X16.COMPGAP16 == 0 and X16.STORE16 == "off"
assert X16.HEADS16 == "top" and X16.LOSS16 == "answer" and X16.BIRTHRULE16 == "D" and X16.FIXEDCODES16 == "off" and X16.STREAM16 == "blocks"
LR, WD, BETAS = .001, .3, (.9, .98)
SOURCES = ("arch20.py", "exp20_reference.py", "exp16_integrated.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py", "mod97_local.py")
TAUGHT_N = 50


# ---------------------------------------------------------------- provenance
def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def tensors_sha(sd):
    """Hash of the tensors themselves (names, dtypes, shapes, values), independent of the file name that torch.save embeds."""
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().contiguous().cpu()
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def source_hashes():
    return {s: X16.sha_file(HERE / s) for s in SOURCES}


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES], cwd=HERE, capture_output=True, text=True).stdout.strip()
        return dict(commit=head, sources_modified_vs_commit=dirty.splitlines())
    except Exception as e:      # noqa: BLE001
        return dict(error=str(e))


def host():
    return dict(hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(), cores=os.cpu_count(),
                python=sys.version.split()[0], torch=T.__version__, torch_threads=T.get_num_threads())


def provenance(cmd):
    return dict(command=cmd, environment_at_start=ENV_AT_START, environment_set_by_runner=REQUIRED_ENV, host=host(), git=git_state(),
                sources_sha256=source_hashes())


def opt_groups(m, opt):
    names = {id(p): n for n, p in m.named_parameters()}
    return [dict(weight_decay=g["weight_decay"], lr=g["lr"], betas=list(g["betas"]), params=[names[id(p)] for p in g["params"]]) for g in opt.param_groups]


def expected_optimizer(m):
    rest = [n for n, _ in m.named_parameters() if not n.startswith("book.")]
    return [dict(weight_decay=WD, lr=LR, betas=list(BETAS), params=rest), dict(weight_decay=0.0, lr=LR, betas=list(BETAS), params=["book.weight"])]


def make_opt(m):
    opt = X16.adamw16(m, LR, WD, ())
    g = opt_groups(m, opt)
    assert g == expected_optimizer(m), f"optimizer groups are not the expected ones: {g}"
    return opt


def opt_steps(opt):
    st = [int(s["step"]) for s in opt.state.values() if "step" in s]
    return sorted(set(st))


def dose_tag_dev():
    return E.DEVTAG


def dose_tag_life():
    return E.DEVTAG + E.DRYTAG


def refuse_existing(*paths):
    for p in paths:
        assert not Path(p).exists(), f"output path exists, refused: {p}"


def save_weights(m, path):
    tmp = Path(str(path) + ".tmp"); T.save(m.state_dict(), tmp); os.replace(tmp, path)
    return X16.sha_file(path)


# ---------------------------------------------------------------- the taught panel (amendment 10): a dictionary of its own
def taught_panel(seed):
    return [X16.render_pair(o1, o2, *tr) for o1, o2 in X16.TAUGHT_PAIRS for tr in X16.triples((o1, o2), seed, 2000)[:TAUGHT_N]]


def taught_panel_check(seed, P, comp, taught):
    """Every taught-panel item is in the training stream's taught set; no item of `within` or a held pair is."""
    f2 = {X16.render_pair(e[1], e[2], *e[3:])[0].__repr__() for e in X16.family_entries("F2", seed)}
    key = lambda it: it[0].__repr__()
    return dict(taught_items=len(taught), taught_in_F2_entries=sum(key(it) in f2 for it in taught),
                within_in_F2_entries=sum(key(it) in f2 for it in comp["within"]),
                held_in_F2_entries={n: sum(key(it) in f2 for it in v) for n, v in comp.items() if n != "within"},
                sha256=sha_bytes(json.dumps(taught).encode()))


# ---------------------------------------------------------------- birth (develop17's plain path)
def birth_path(out, net, seed):
    return Path(out) / f"e20_dev_{net}{dose_tag_dev()}_{seed}.pt"


def dev(net, seed, out, cmd):
    arch20.install(net)
    path = birth_path(out, net, seed); rec_path = path.with_suffix(".json")
    refuse_existing(path, rec_path, str(path) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    m = E.new_net(seed)
    desc = arch20.describe(m)
    book0 = m.book.weight.detach().clone()
    opt = make_opt(m)
    census = None
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    bt = E.batches(tp, seed, E.KMAX["M"])
    rec = dict(experiment="E20-B", kind="birth", net=net, seed=seed, kmax=E.KMAX["M"], dev_updates=E.DEV_DOSE, block=X.BLOCK, batch=X.BATCH,
               optimizer=dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=opt_groups(m, opt)),
               protocol="exp16_integrated.develop17 plain path (backprop through time, answer loss, four mod-89 operations, k 0..12)",
               model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
               provenance=provenance(cmd), complete=False, curve=[])
    t0 = time.process_time(); curve = rec["curve"]; row = None
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        losses = []
        for _ in range(X.BLOCK):
            items = next(bt); losses.append(E.update(m, opt, items, "D"))
            if census is None:
                census = X16.zero_grad_report(m, X.batch_tensor(items)[0]); print("e20 dev credit census:", census["by_role"], flush=True)
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}, book_shift_by_role=X16.book_shift(m, book0))
            curve.append(row)
            print("e20 dev", net, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            rec_path.write_text(json.dumps(dict(rec, credit_census_first_update=census), indent=1))
    assert row is not None, "the birth loop works in whole blocks of 512 updates: E11_DEV_UPDATES must be a positive multiple of 512"
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    wsha = save_weights(m, path)
    rec.update(credit_census_first_update=census, optimizer_steps=opt_steps(opt), weights=path.name, weights_sha256=wsha,
               tensors_sha256=tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, complete=True)
    rec_path.write_text(json.dumps(rec, indent=1))
    print("e20 dev", net, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- admission (amendment 12)
def admit_e20(bfile, net, seed):
    """Admit an E20 birth for a life of `net` and `seed`, from its complete receipt. Raises AssertionError on any mismatch."""
    bfile = Path(bfile); rp = bfile.with_suffix(".json")
    assert bfile.exists() and rp.exists(), f"{bfile.name}: birth or its receipt is missing"
    r = json.loads(rp.read_text())
    assert r.get("experiment") == "E20-B" and r.get("kind") == "birth", f"{bfile.name}: not an E20-B birth receipt (no legacy admission)"
    assert r.get("complete") is True, f"{bfile.name}: receipt is not complete"
    fseed = bfile.stem.rsplit("_", 1)[1]
    assert str(r.get("seed")) == fseed == str(seed), f"{bfile.name}: receipt seed {r.get('seed')}, file-name seed {fseed}, life seed {seed} must agree"
    assert r.get("net") == net and bfile.stem.startswith(f"e20_dev_{net}"), f"{bfile.name}: receipt network {r.get('net')}, life network {net}"
    assert r.get("dev_updates") == E.DEV_DOSE and r["curve"] and r["curve"][-1]["step"] == E.DEV_DOSE, f"{bfile.name}: birth dose {r.get('dev_updates')}, expected {E.DEV_DOSE}"
    m = E.new_net(seed)
    exp_opt = dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=expected_optimizer(m))
    assert r.get("optimizer") == exp_opt, f"{bfile.name}: optimizer settings differ"
    assert r["model"]["cls"] == arch20.describe(m)["cls"] and r["model"]["params"] == arch20.describe(m)["params"], f"{bfile.name}: network class or size differs"
    srcs = r.get("provenance", {}).get("sources_sha256", {})
    now = source_hashes()
    assert srcs == now, f"{bfile.name}: source files differ from the birth's: {sorted(k for k in now if srcs.get(k) != now[k])}"
    sha = X16.sha_file(bfile)
    assert r.get("weights_sha256") == sha, f"{bfile.name}: weights hash {sha[:12]} is not the receipt's {str(r.get('weights_sha256'))[:12]}"
    return m, sha, "E20 receipt: network, seed (receipt, file name, life), dose, optimizer, sources and weights hash checked"


def admit_parity(bfile, sha_expected, seed):
    """Amendment 9b: the reference birth itself, loaded by path, its hash checked against the hash given on the command line."""
    bfile = Path(bfile); r = json.loads(bfile.with_suffix(".json").read_text())
    sha = X16.sha_file(bfile)
    assert sha == sha_expected == r.get("weights_sha256"), f"{bfile.name}: hash {sha[:12]}, expected {sha_expected[:12]}, receipt {str(r.get('weights_sha256'))[:12]}"
    assert r.get("complete") and str(r.get("seed")) == bfile.stem.rsplit("_", 1)[1] == str(seed), "reference birth seed mismatch"
    want = dict(arch="bank2", bookdecay="off", heads="top", birthrule="D", fixedcodes="off", loss="answer", updates=E.DEV_DOSE)
    bad = {k: (r.get(k), v) for k, v in want.items() if r.get(k) != v}
    assert not bad, f"{bfile.name}: not the reference birth for this dose {bad}"
    return E.new_net(seed), sha, "parity: reference birth by path, hash given on the command line and in its receipt"


# ---------------------------------------------------------------- life (exp16_integrated.life, arm D)
def life(net, seed, out, cmd, parity_birth=None, parity_sha=None):
    arch20.install(net)
    tag = "_parity" if parity_birth else ""
    path = Path(out) / f"e20_life_{net}{tag}{dose_tag_life()}_{seed}.json"
    refuse_existing(path, path.with_suffix(".pt"), str(path.with_suffix(".pt")) + ".tmp")
    Path(out).mkdir(parents=True, exist_ok=True)
    if parity_birth:
        assert net == "G0", "parity is defined for G0 only"
        bfile = Path(parity_birth); m, birth_sha, admission = admit_parity(bfile, parity_sha, seed)
    else:
        bfile = birth_path(out, net, seed); m, birth_sha, admission = admit_e20(bfile, net, seed)
    m.load_state_dict(T.load(bfile))
    desc = arch20.describe(m)
    opt = make_opt(m)
    P, comp = X16.panels(seed)
    SP, strict_counts = X16.strict_panels(seed)
    taught = taught_panel(seed)
    tcheck = taught_panel_check(seed, P, comp, taught)
    assert tcheck["taught_in_F2_entries"] == len(taught) and tcheck["within_in_F2_entries"] == 0 and not any(tcheck["held_in_F2_entries"].values())
    f1c, f3c = {t[2][0] for t in X16.F1}, {t[2][0] for t in X16.F3}
    code_of = lambda p_: next((b_ for b_ in p_ if not 65 <= b_ <= 90), p_[0])
    fam_of = lambda p_: "F2" if code_of(p_) == ord("c") else "F1" if code_of(p_) in f1c else "F3" if code_of(p_) in f3c else "other"
    entries = {f: X16.family_entries(f, seed) for f in ("F1", "F2", "F3")}
    taught_keys = {tuple(p) for p, _ in taught}
    unseen_keys = {tuple(p) for n, v in comp.items() for p, _ in v}
    presented = dict(taught_panel_items_presented=0, taught_panel_distinct_presented=set(), within_or_held_presented=0)
    seen_streams = []
    t0 = time.process_time(); stages = []
    score = X16.score

    def evaluate():
        res = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, None) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
        res["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
        res["taught"] = score(m, taught, None)                       # E20-B: taught instances (kept out of `compose`)
        return res

    c0 = time.process_time(); scores0 = evaluate(); cpu_score = [time.process_time() - c0]
    settings = dict(net=net, model=desc, installed=dict(E_forward=f"{E.forward.__module__}.{E.forward.__name__}", E_new_net_class=desc["cls"]),
                    optimizer=dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=opt_groups(m, opt)),
                    stage_updates=E.STAGE_UPDATES, dev_updates=E.DEV_DOSE, batch=X.BATCH, arch16=X16.ARCH16, bookdecay=X16.BOOKDECAY16, compgap=X16.COMPGAP16,
                    store=False, panel_sha256=X16.panel_sha(P, comp, SP), strict_panel_counts=strict_counts, taught_panel=tcheck,
                    birth=str(bfile), birth_sha256=birth_sha, birth_admission=admission, protocol="exp16_integrated.life arm D (joint stream of families admitted so far, backprop, decay 0.3)",
                    provenance=provenance(cmd))
    stages.append(dict(stage=0, task=None, scores=scores0, shortcut_ceilings=X16.shortcut_ceilings(comp), entries={f: len(v) for f, v in entries.items()}, settings=settings))
    for si, (fam, block) in enumerate(X16.SCHEDULE, 1):
        if block == 1:
            seen_streams.append(X16.entry_stream(entries[fam], seed + 100 + si))
        joint = X16.joint_stream(seen_streams, seed + si)
        losses, expo = [], {}
        for u in range(E.STAGE_UPDATES):
            batch = [next(joint) for _ in range(X.BATCH)]
            for p_, t_ in batch:
                e_ = expo.setdefault(f"joint/{fam_of(p_)}", [0, 0]); e_[0] += 1; e_[1] += len(t_)
                k_ = tuple(p_)
                if k_ in taught_keys:
                    presented["taught_panel_items_presented"] += 1; presented["taught_panel_distinct_presented"].add(k_)
                if k_ in unseen_keys:
                    presented["within_or_held_presented"] += 1
            losses.append(E.update(m, opt, batch, "D", None))
            if si == 1 and u == 0:
                stages[0]["settings"]["zero_grad_first_update"] = X16.zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e20 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"]["by_role"], flush=True)
        cb = time.process_time(); sc = evaluate(); ca = time.process_time(); cpu_score[0] += ca - cb
        stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)), cpu=cb - t0,
                           cpu_scoring_so_far=cpu_score[0], cpu_updates_so_far=ca - t0 - cpu_score[0], optimizer_steps=opt_steps(opt),
                           exposure_rows_tokens=expo, exposure_note="[rows, bytes under the loss mask]; loss=answer",
                           exposure_panels=dict(presented, taught_panel_distinct_presented=len(presented["taught_panel_distinct_presented"]))))
        print("e20", net, seed, "after", f"{fam}.{block}",
              "F1@k", {n: [sc[n][k]["exact"] for k in X16.KS] for n in sc if n.endswith("97")},
              "comp.exact", {n: v["exact"] for n, v in sc["compose"].items()}, "taught", sc["taught"]["exact"],
              f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        path.write_text(json.dumps(dict(experiment="E20-B", kind="life", net=net, arm="D", seed=seed, schedule=X16.SCHEDULE, stages=stages), indent=1))
    assert presented["within_or_held_presented"] == 0, "a within or held-pair panel item was presented in training"
    wsha = save_weights(m, path.with_suffix(".pt"))
    path.write_text(json.dumps(dict(experiment="E20-B", kind="life", net=net, arm="D", seed=seed, schedule=X16.SCHEDULE, stages=stages,
                                    complete=dict(stages=len(stages), weights=path.with_suffix(".pt").name, weights_sha256=wsha,
                                                  tensors_sha256=tensors_sha(m.state_dict()), optimizer_steps=opt_steps(opt),
                                                  cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0])), indent=1))
    print("e20 life", net, seed, "complete", wsha[:16], flush=True)


# ---------------------------------------------------------------- qualification checks (amendments 9-12)
def mixed_items(seed, n_ops=24, n_comp=24):
    rng = Random(5)
    ops = [E.render(t, a, b, rng.randint(0, 12), rng) for t in E.DEV + X16.F1 for a, b in rng.sample(E.split(t, .5, seed)[0], n_ops // 8 or 1)]
    comps = [X16.render_pair(o1, o2, *tr) for o1, o2 in rng.sample(X16.TAUGHT_PAIRS, 6) for tr in X16.triples((o1, o2), seed, 2000)[:n_comp // 6]]
    items = ops + comps
    rng.shuffle(items)
    return items


def arch_checks(net, seed):
    """Amendment 3 and 11, on a batch of mixed lengths, before any training."""
    arch20.install(net)
    m = E.new_net(seed); m.eval()
    out = dict(model=arch20.describe(m), installed_forward=f"{E.forward.__module__}.{E.forward.__name__}")
    out["model"].pop("params_by_tensor")
    out["params_within_5pct_of_bank2"] = abs(out["model"]["params"] / arch20.TARGET_PARAMS - 1) <= .05
    items = mixed_items(seed)
    x, y, mask = X.batch_tensor(items)
    lens = [len(p) + len(t) for p, t in items]
    out["batch"] = dict(rows=x.shape[0], width=x.shape[1], row_lengths=sorted(set(lens)))
    with T.no_grad():
        lg = E.forward(m, x, False, True)[0]
    out["shapes"] = dict(x=list(x.shape), logits=list(lg.shape), ok=list(lg.shape) == [x.shape[0], x.shape[1], 257])
    g = T.Generator().manual_seed(11)
    # causality: bytes after position j changed -> logits at positions <= j unchanged
    worst, n = 0.0, 0
    for i in range(x.shape[0]):
        for j in (2, lens[i] // 2, lens[i] - 1):
            x2 = x.clone(); x2[i, j + 1:] = T.randint(0, 256, (x.shape[1] - j - 1,), generator=g)
            with T.no_grad():
                l2 = E.forward(m, x2, False, True)[0]
            worst = max(worst, float((l2[i, :j + 1] - lg[i, :j + 1]).abs().max())); n += 1
    out["causality"] = dict(cases=n, max_abs_diff_earlier_logits=worst, exact=worst == 0.0)
    # rows: other rows changed, and rows reordered
    x3 = x.clone(); x3[1:] = T.randint(0, 256, x3[1:].shape, generator=g)
    perm = T.randperm(x.shape[0], generator=g)
    with T.no_grad():
        l3 = E.forward(m, x3, False, True)[0]; l4 = E.forward(m, x[perm], False, True)[0]
    out["rows"] = dict(other_rows_changed_max_abs_diff=float((l3[0] - lg[0]).abs().max()),
                       reordered_max_abs_diff=float((l4 - lg[perm]).abs().max()))
    out["rows"]["exact"] = out["rows"]["other_rows_changed_max_abs_diff"] == 0.0 and out["rows"]["reordered_max_abs_diff"] == 0.0
    # state reset: no state kept in the module; the same batch after another batch gives the same logits; each row alone gives its row
    with T.no_grad():
        E.forward(m, x3, False, True)
        l5 = E.forward(m, x, False, True)[0]
        alone = max(float((E.forward(m, x[i:i + 1, :lens[i]], False, True)[0][0] - lg[i, :lens[i]]).abs().max()) for i in range(x.shape[0]))
    bufs = [n_ for n_, _ in m.named_buffers()]
    out["state_reset"] = dict(module_buffers=bufs, same_batch_after_another_max_abs_diff=float((l5 - lg).abs().max()),
                              each_row_alone_vs_in_batch_max_abs_diff=alone, tolerance_for_row_alone=1e-5)
    if net in ("G1", "G2"):     # a step-by-step loop from the textbook equations, every state (the LSTM's cell state too) starting at zero
        with T.no_grad():
            z1 = T.tanh(m.l1(m.book(x))); W_ih, W_hh, b_ih, b_hh = m.rnn.weight_ih_l0, m.rnn.weight_hh_l0, m.rnn.bias_ih_l0, m.rnn.bias_hh_l0
            h = T.zeros(x.shape[0], m.units); c = T.zeros(x.shape[0], m.units); hs = []
            for t in range(x.shape[1]):
                gi, gh = z1[:, t] @ W_ih.T + b_ih, h @ W_hh.T + b_hh
                if net == "G1":
                    r_, z_, n_ = gi.chunk(3, 1); hr, hz, hn = gh.chunk(3, 1)
                    r = T.sigmoid(r_ + hr); z = T.sigmoid(z_ + hz); nn_ = T.tanh(n_ + r * hn); h = (1 - z) * nn_ + z * h
                else:
                    i_, f_, g_, o_ = (gi + gh).chunk(4, 1)
                    c = T.sigmoid(f_) * c + T.sigmoid(i_) * T.tanh(g_); h = T.sigmoid(o_) * T.tanh(c)
                hs.append(h)
            ref = m.l3(T.tanh(m.l2(z1) + m.lm(T.stack(hs, 1))))
        out["state_reset"]["manual_zero_state_loop_max_abs_diff"] = float((ref - lg).abs().max())
        # the initial state matters: a non-zero cell (or hidden) state would change the first outputs, so zero is a real reset
        with T.no_grad():
            st = (T.ones(1, x.shape[0], m.units),) if net == "G1" else ((T.zeros(1, x.shape[0], m.units), T.ones(1, x.shape[0], m.units)),)
            hs_nz, _ = m.rnn(z1, *st); hs_z = m.rnn(z1)[0]
        out["state_reset"]["nonzero_initial_state_changes_first_step"] = float((hs_nz[:, 0] - hs_z[:, 0]).abs().max())
    out["state_reset"]["ok"] = (not [b for b in bufs if b != "R"] and out["state_reset"]["same_batch_after_another_max_abs_diff"] == 0.0
                                and alone <= 1e-5 and out["state_reset"].get("manual_zero_state_loop_max_abs_diff", 0.0) <= 1e-5)
    # gradients on a first batch: every parameter; active embedding rows (bytes at or before a row's last supervised position)
    m.train()
    birth_batch = next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, 12))
    opt = make_opt(m)
    grads = {}
    for name_, its in (("birth_batch", birth_batch), ("mixed_batch", items)):
        xb, yb, mb = X.batch_tensor(its)
        lgb = E.forward(m, xb, False, True)[0]
        loss = E.cross_entropy(lgb[mb], yb[mb]); opt.zero_grad(); loss.backward()
        last = [int(mb[i].nonzero().max()) for i in range(xb.shape[0])]
        active = sorted({int(xb[i, j]) for i in range(xb.shape[0]) for j in range(last[i] + 1)} - {E.BOS})
        gb = m.book.weight.grad
        rec = dict(params_without_finite_nonzero_grad=[n_ for n_, p_ in m.named_parameters() if p_.grad is None or not bool(T.isfinite(p_.grad).all()) or not bool(p_.grad.any())],
                   active_rows=len(active), active_rows_without_grad=[r for r in active if not bool(gb[r].any())],
                   bos_row_grad_zero=not bool(gb[E.BOS].any()))
        rec_names = {"G1": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"], "G2": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"],
                     "G3": ["lb.weight", "lc.weight", "lw.weight", "lc2.weight", "lw2.weight"], "G0": ["lc.weight", "lw.weight", "lc2.weight", "lw2.weight"]}[net]
        rec["recurrent_weight_grad_abs_max"] = {n_: float(m.get_parameter(n_).grad.abs().max()) for n_ in rec_names}
        grads[name_] = rec
    out["gradients"] = grads
    out["gradients_ok"] = all(not v["params_without_finite_nonzero_grad"] and not v["active_rows_without_grad"] and v["bos_row_grad_zero"]
                              and all(a > 0 for a in v["recurrent_weight_grad_abs_max"].values()) for v in grads.values())
    out["optimizer_groups"] = [dict(weight_decay=g_["weight_decay"], n_params=len(g_["params"]), names=(g_["params"] if len(g_["params"]) < 4 else g_["params"][:3] + ["..."]))
                               for g_ in opt_groups(m, opt)]
    out["embedding_without_decay"] = [g_["params"] for g_ in opt_groups(m, opt) if g_["weight_decay"] == 0.0] == [["book.weight"]]
    out["bos_eos"] = dict(vocab=m.book.num_embeddings, padding_idx=m.book.padding_idx, BOS=E.BOS, EOS=E.EOS, outputs=m.l3.out_features,
                          ok=m.book.num_embeddings == 258 and m.book.padding_idx == E.BOS and m.l3.out_features == 257)
    if net == "G3":
        out["g3"] = g3_checks(seed, x)
    out["ok"] = (out["params_within_5pct_of_bank2"] and out["shapes"]["ok"] and out["causality"]["exact"] and out["rows"]["exact"]
                 and out["state_reset"]["ok"] and out["gradients_ok"] and out["embedding_without_decay"] and out["bos_eos"]["ok"]
                 and (net != "G3" or out["g3"]["ok"]))
    return out


def g3_checks(seed, x):
    g0 = arch20.new_net(seed, "G0"); g3 = arch20.new_net(seed, "G3")
    s0, s3 = g0.state_dict(), g3.state_dict()
    shared_equal = all(T.equal(s0[k], s3[k]) for k in s0)
    added = sorted(set(s3) - set(s0))
    with T.no_grad():
        l0 = arch17.forward(g0, x, False, True)[0]
        l3, it = arch20.forward(g3, x, False, True)
        lb_saved = g3.lb.weight.clone(); g3.lb.weight.zero_()
        l3z = arch20.forward(g3, x, False, True)[0]
        g3.lb.weight.copy_(lb_saved)
        # p_t recomputed from the returned states: tanh(lp z1_t + lb h2_{t-1}), h2_{-1} = 0
        h2prev = T.cat([T.zeros_like(it["hs2"][0])[:, None], T.stack(it["hs2"], 1)[:, :-1]], 1)
        p_re = T.tanh(g3.lp(it["z1"]) + g3.lb(h2prev))
    out = dict(added=added, added_shape=list(g3.lb.weight.shape), added_bias=g3.lb.bias is not None, added_weights=g3.lb.weight.numel(),
               params_G0=sum(p.numel() for p in g0.parameters()), params_G3=sum(p.numel() for p in g3.parameters()),
               shared_initial_values_equal_G0=shared_equal,
               lb_zero_outputs_equal_G0_exact=bool(T.equal(l3z, l0)), lb_zero_max_abs_diff=float((l3z - l0).abs().max()),
               lb_enabled_first_step_equal_G0_exact=bool(T.equal(l3[:, 0], l0[:, 0])),
               lb_enabled_later_steps_differ_from_G0=float((l3[:, 1:] - l0[:, 1:]).abs().max()),
               p_t_recomputed_from_previous_h2_max_abs_diff=float((p_re - it["p"]).abs().max()))
    out["ok"] = (added == ["lb.weight"] and out["added_shape"] == [64, 512] and not out["added_bias"] and shared_equal
                 and out["lb_zero_outputs_equal_G0_exact"] and out["lb_enabled_first_step_equal_G0_exact"]
                 and out["lb_enabled_later_steps_differ_from_G0"] > 0 and out["p_t_recomputed_from_previous_h2_max_abs_diff"] <= 1e-6)
    return out


def scoring_fixtures(seed):
    """Amendment 10: the six-byte life scorer, with the responder replaced by a stub; and the birth scorer's four-byte cap."""
    arch20.install("G0")
    P, comp = X16.panels(seed)
    taught = taught_panel(seed)
    panels_ = dict(comp, taught=taught)
    real = E.respond
    res = {}
    try:
        for fx, make in (("fully_correct", lambda t: list(t)),
                         ("wrong_intermediate_right_ending", lambda t: [(t[0] - 48 + 1) % 10 + 48] + list(t[1:])),
                         ("missing_end_symbol", lambda t: list(t[:-1]) + [t[-2]])):
            res[fx] = {}
            for name, its in panels_.items():
                E.respond = (lambda its_: (lambda m_, prompts, store=None, cap=4, silence=False: [make(t) for _, t in its_]))(its)
                res[fx][name] = X16.score(None, its)
    finally:
        E.respond = real
    ok = all(v["exact"] == v["n"] and v["final"] == v["n"] for v in res["fully_correct"].values()) \
        and all(v["exact"] == 0 and v["final"] == v["n"] for v in res["wrong_intermediate_right_ending"].values()) \
        and all(v["exact"] == 0 and v["final"] == 0 for v in res["missing_end_symbol"].values())
    m = E.new_net(seed)
    its = comp["+-"][:64]
    gen4 = E.respond(m, [p for p, _ in its])
    try:
        X16.shortcut_ceilings({"taught": taught}); sc_taught = "accepted"
    except ValueError as e:
        sc_taught = f"ValueError: {e}"
    return dict(fixtures=res, fixtures_ok=ok, life_scorer_cap=X16.CAP, composition_target_bytes=sorted({len(t) for _, t in its}),
                birth_scorer_max_generated=max(len(g) for g in gen4), birth_scorer_used_for="births only (single operations, 3-byte answers)",
                shortcut_ceilings_on_compose_ok=sorted(X16.shortcut_ceilings(comp)), shortcut_ceilings_with_taught_key=sc_taught,
                taught_kept_out_of_compose="taught" not in comp, taught_panel=taught_panel_check(seed, P, comp, taught))


def admission_fixtures(seed):
    """Amendment 12: a birth under another seed, a changed weights file and an existing output path are refused; a reference (exp16)
    birth is not admitted as an E20 birth."""
    arch20.install("G0")
    res = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        m = E.new_net(seed)
        own = td / f"e20_dev_G0{dose_tag_dev()}_{seed}.pt"
        wsha = save_weights(m, own)
        opt = make_opt(m)
        rec = dict(experiment="E20-B", kind="birth", net="G0", seed=seed, dev_updates=E.DEV_DOSE, curve=[dict(step=E.DEV_DOSE)],
                   optimizer=dict(name="AdamW", lr=LR, weight_decay=WD, betas=list(BETAS), groups=opt_groups(m, opt)),
                   model=arch20.describe(m), provenance=dict(sources_sha256=source_hashes()), weights=own.name, weights_sha256=wsha, complete=True)
        own.with_suffix(".json").write_text(json.dumps(rec))
        def attempt(f, net, s):
            try:
                return "admitted: " + admit_e20(f, net, s)[2]
            except AssertionError as e:
                return f"refused: {e}"
        res["own_birth_own_seed"] = attempt(own, "G0", seed)
        other = 93683 if seed != 93683 else 73129
        fake = td / f"e20_dev_G0{dose_tag_dev()}_{other}.pt"
        shutil.copy(own, fake); shutil.copy(own.with_suffix(".json"), fake.with_suffix(".json"))
        res["birth_copied_under_another_seed_for_that_seed"] = attempt(fake, "G0", other)
        res["birth_of_seed_presented_to_another_seeds_life"] = attempt(own, "G0", other)
        res["birth_presented_to_another_network"] = attempt(own, "G3", seed)
        changed = td / "chg" / own.name; changed.parent.mkdir()
        sd = m.state_dict(); sd["l3.bias"] = sd["l3.bias"].clone(); sd["l3.bias"][0] += 1e-3
        T.save(sd, changed); shutil.copy(own.with_suffix(".json"), changed.with_suffix(".json"))
        res["changed_weights_file"] = attempt(changed, "G0", seed)
        bad_rec = dict(rec, net="G1"); badd = td / "badnet" / own.name; badd.parent.mkdir(); shutil.copy(own, badd)
        badd.with_suffix(".json").write_text(json.dumps(bad_rec)); res["receipt_names_another_network"] = attempt(badd, "G0", seed)
        inc = td / "inc" / own.name; inc.parent.mkdir(); shutil.copy(own, inc)
        inc.with_suffix(".json").write_text(json.dumps(dict(rec, complete=False))); res["incomplete_receipt"] = attempt(inc, "G0", seed)
        legacy = td / "legacy" / own.name; legacy.parent.mkdir(); shutil.copy(own, legacy)
        legacy.with_suffix(".json").write_text(json.dumps(dict(arch="bank2", seed=seed, bookdecay="off", complete=True, weights_sha256=wsha, curve=[dict(step=E.DEV_DOSE)])))
        res["exp16_style_receipt_no_legacy_admission"] = attempt(legacy, "G0", seed)
        for what, p in (("birth_output_exists", own), ("life_output_exists", td / "e20_life_x.json")):
            p.touch()
            try:
                refuse_existing(p); res[what] = "not refused"
            except AssertionError as e:
                res[what] = f"refused: {e}".replace(str(td), "<tmp>")
    res = {k: v.replace(str(td), "<tmp>") for k, v in res.items()}
    res["ok"] = res["own_birth_own_seed"].startswith("admitted") and all(v.startswith("refused") for k, v in res.items() if k not in ("own_birth_own_seed", "ok"))
    return res


def compare_curves(a, b):
    diffs = []
    if len(a) != len(b):
        return [f"curve lengths {len(a)} vs {len(b)}"]
    for ra, rb in zip(a, b):
        for k in ("step", "loss", "sealed", "book_shift_by_role", "silenced_k12"):
            if (k in ra or k in rb) and ra.get(k) != rb.get(k):
                diffs.append(f"step {ra.get('step')}: {k}")
    return diffs


def per_tensor_maxdiff(fa, fb):
    a, b = T.load(fa), T.load(fb)
    return {k: float((a[k].float() - b[k].float()).abs().max()) for k in a} if set(a) == set(b) else dict(keys_differ=sorted(set(a) ^ set(b)))


def resave_sha(f):
    """The file hash of these tensors saved under one fixed name (torch.save embeds the file's name in the archive)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "weights.pt"; T.save(T.load(f), p); return X16.sha_file(p)


def parity(qual):
    q = Path(qual); ref = q / "reference"
    rb = sorted(ref.glob("e17_dev_bank2_nbd_dry*_73129.json")); rl = sorted(ref.glob("e17_life_D_bank2_nbd_e20bref_dry*_73129.json"))
    mb = sorted(q.glob("e20_dev_G0_dry*_73129.json")); ml = sorted(q.glob("e20_life_G0_parity_dry*_73129.json"))
    if not (len(rb) == len(rl) == len(mb) == len(ml) == 1):
        return dict(status="NOT_RUN", found=dict(ref_birth=[p.name for p in rb], ref_life=[p.name for p in rl], e20_birth=[p.name for p in mb], e20_life=[p.name for p in ml]))
    RB, RL, MB, ML = (json.loads(p.read_text()) for p in (rb[0], rl[0], mb[0], ml[0]))
    out = dict(files=dict(reference_birth=rb[0].name, reference_life=rl[0].name, e20_birth=mb[0].name, e20_parity_life=ml[0].name))
    # a. birth
    bd = compare_curves(RB["curve"], MB["curve"])
    rw, mw = rb[0].with_suffix(".pt"), mb[0].with_suffix(".pt")
    out["a_birth"] = dict(curve_rows=len(RB["curve"]), curve_fields_compared=["step", "loss", "sealed", "book_shift_by_role", "silenced_k12"],
                          curve_differences=bd, census_equal=RB.get("credit_census_first_update") == MB.get("credit_census_first_update"),
                          reference_file_sha256=RB["weights_sha256"], e20_file_sha256=MB["weights_sha256"],
                          reference_tensors_sha256=tensors_sha(T.load(rw)), e20_tensors_sha256=tensors_sha(T.load(mw)),
                          reference_resaved_sha256=resave_sha(rw), e20_resaved_sha256=resave_sha(mw),
                          file_sha_note="torch.save embeds the file's own name in the archive, so files of identical tensors under different names have different hashes; the weights are compared by their tensors and by re-saving both under one name")
    ab = out["a_birth"]
    ab["weights_equal"] = ab["reference_tensors_sha256"] == ab["e20_tensors_sha256"] and ab["reference_resaved_sha256"] == ab["e20_resaved_sha256"]
    if not ab["weights_equal"]:
        ab["per_tensor_max_abs_diff"] = per_tensor_maxdiff(rw, mw)
    ab["pass"] = not bd and ab["census_equal"] and ab["weights_equal"]
    # b. life from the reference birth file itself
    st_r, st_m = RL["stages"], ML["stages"]
    rows = []
    for a, b in zip(st_r, st_m):
        shared = sorted(set(a["scores"]) & set(b["scores"]))
        diff = [k for k in shared if a["scores"][k] != b["scores"][k]]
        r = dict(stage=a["stage"], task=a.get("task"), shared_score_keys=len(shared), score_keys_only_in_e20=sorted(set(b["scores"]) - set(a["scores"])),
                 score_keys_only_in_reference=sorted(set(a["scores"]) - set(b["scores"])), differing_score_keys=diff)
        if a["stage"] > 0:
            r["loss_reference"], r["loss_e20"], r["loss_equal"] = a["loss"], b["loss"], a["loss"] == b["loss"]
            r["exposure_equal"] = a["exposure_rows_tokens"] == b["exposure_rows_tokens"]
            r["updates_reference_from_rows"] = sum(v[0] for v in a["exposure_rows_tokens"].values()) // X.BATCH
            r["optimizer_steps_e20"] = b["optimizer_steps"]
        rows.append(r)
    s0r, s0m = st_r[0]["settings"], st_m[0]["settings"]
    cum = 0; steps_ok = True
    for r in rows[1:]:
        cum += r["updates_reference_from_rows"]; steps_ok &= r["optimizer_steps_e20"] == [cum]
    lw_r, lw_m = rl[0].with_suffix(".pt"), ml[0].with_suffix(".pt")
    out["b_life"] = dict(stages_reference=len(st_r), stages_e20=len(st_m), per_stage=rows,
                         panel_sha256=dict(reference=s0r["panel_sha256"], e20=s0m["panel_sha256"], equal=s0r["panel_sha256"] == s0m["panel_sha256"]),
                         strict_counts_equal=s0r["strict_panel_counts"] == s0m["strict_panel_counts"],
                         zero_grad_first_update_equal=s0r.get("zero_grad_first_update") == s0m.get("zero_grad_first_update"),
                         shortcut_ceilings_equal=st_r[0]["shortcut_ceilings"] == st_m[0]["shortcut_ceilings"],
                         birth_sha256=dict(reference=s0r["birth_sha256"], e20=s0m["birth_sha256"], equal=s0r["birth_sha256"] == s0m["birth_sha256"]),
                         optimizer_steps_match_reference_updates=steps_ok, total_updates=cum,
                         reference_file_sha256=RL["complete"]["weights_sha256"], e20_file_sha256=ML["complete"]["weights_sha256"],
                         reference_tensors_sha256=tensors_sha(T.load(lw_r)), e20_tensors_sha256=tensors_sha(T.load(lw_m)),
                         reference_resaved_sha256=resave_sha(lw_r), e20_resaved_sha256=resave_sha(lw_m))
    bl = out["b_life"]
    bl["weights_equal"] = bl["reference_tensors_sha256"] == bl["e20_tensors_sha256"] and bl["reference_resaved_sha256"] == bl["e20_resaved_sha256"]
    if not bl["weights_equal"]:
        bl["per_tensor_max_abs_diff"] = per_tensor_maxdiff(lw_r, lw_m)
    bl["pass"] = (len(st_r) == len(st_m) == 7 and all(not r["differing_score_keys"] and not r["score_keys_only_in_reference"] and r.get("loss_equal", True)
                                                     and r.get("exposure_equal", True) for r in rows)
                  and bl["panel_sha256"]["equal"] and bl["birth_sha256"]["equal"] and steps_ok and bl["weights_equal"] and bl["strict_counts_equal"])
    out["pass"] = ab["pass"] and bl["pass"]
    return out


def qualification_runs(qual):
    q = Path(qual); runs = {}
    for f in sorted(q.glob("e20_*.json")):
        r = json.loads(f.read_text())
        if r.get("kind") == "birth":
            runs[f.name] = dict(kind="birth", net=r["net"], seed=r["seed"], complete=r["complete"], dev_updates=r["dev_updates"], cpu_total=r.get("cpu_total"),
                                weights_sha256=r.get("weights_sha256"), command=r["provenance"]["command"], environment=r["provenance"]["environment_at_start"],
                                host=r["provenance"]["host"], git=r["provenance"]["git"], params=r["model"]["params"], cls=r["model"]["cls"],
                                final_hard_k12={n: r["curve"][-1]["sealed"][n]["12"]["exact"] for n in ("add89", "sub89")} if r["curve"] else None)
        elif r.get("kind") == "life":
            s0 = r["stages"][0]["settings"]; last = r["stages"][-1]
            runs[f.name] = dict(kind="life", net=r["net"], seed=r["seed"], stages=len(r["stages"]), complete=r.get("complete"), stage_updates=s0["stage_updates"],
                                birth=s0["birth"], birth_admission=s0["birth_admission"], command=s0["provenance"]["command"],
                                environment=s0["provenance"]["environment_at_start"], host=s0["provenance"]["host"], git=s0["provenance"]["git"],
                                cpu_updates=last.get("cpu_updates_so_far"), cpu_scoring=last.get("cpu_scoring_so_far"),
                                final_compose_exact={n: v["exact"] for n, v in last["scores"]["compose"].items()}, final_taught_exact=last["scores"]["taught"]["exact"])
    return runs


def check(qual, cmd):
    seed = X.SEEDS[0]
    out = dict(experiment="E20-B", stage=1, what="implementation and reduced-dose qualification (amendments 8-12)", provenance=provenance(cmd),
               dose=dict(dev_updates=E.DEV_DOSE, stage_updates=E.STAGE_UPDATES),
               dispatch=dict(arch16=X16.ARCH16, bookdecay16=X16.BOOKDECAY16, compgap16=X16.COMPGAP16, store16=X16.STORE16, prefix=X16.PFX,
                             k40_panels_present=40 in next(iter(X16.panels(seed)[0].values()))))
    out["architecture"] = {net: arch_checks(net, seed) for net in arch20.NETS}
    out["scoring"] = scoring_fixtures(seed)
    out["admission"] = admission_fixtures(seed)
    out["parity"] = parity(qual)
    out["qualification_runs"] = qualification_runs(qual)
    out["summary"] = dict(architecture_ok={n: v["ok"] for n, v in out["architecture"].items()}, scoring_ok=out["scoring"]["fixtures_ok"],
                          admission_ok=out["admission"]["ok"], parity_birth=out["parity"].get("a_birth", {}).get("pass"),
                          parity_life=out["parity"].get("b_life", {}).get("pass"), parity=out["parity"].get("pass", out["parity"].get("status")))
    p = Path(qual) / "E20B_CHECK.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    cmd = " ".join([f"{k}={v}" for k, v in sorted(ENV_AT_START.items()) if k.startswith(("E16_", "E11_"))] + ["python3"] + sys.argv)
    a = sys.argv[1:]
    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default
    outdir = opt_("--out", str(X16.OUT))
    if a[0] == "dev":
        dev(a[1], int(a[2]), outdir, cmd)
    elif a[0] == "life":
        life(a[1], int(a[2]), outdir, cmd, opt_("--parity-birth"), opt_("--parity-sha"))
    elif a[0] == "check":
        check(opt_("--qual", str(X16.OUT / "e20b_qual")), cmd)
    else:
        raise SystemExit(__doc__)
