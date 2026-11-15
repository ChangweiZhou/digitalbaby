"""E38 runner (brief OPUS_TASK_E38_WHAT_ERASES_THE_BIRTH_FAMILY_20261115.md, revision 2): what erases the birth family in E26's first two
blocks, and does an anchor to the birth values protect it? Seed 73129, rules S (sweep) and B (backpropagation), five arms each.

The life is exp26_candidate.life's loop (lines 196-279) over X16.SCHEDULE[:2], copied here (not called, not patched) with the one
difference per arm the brief names. The step is exp26_candidate.step's four lines, copied, with the anchor pull between backward and the
optimizer step:

    zero      E26 unchanged (AdamW decay 0.25 toward zero on group 0)
    none      group 0 weight_decay = 0.0, no pull
    anchor    group 0 weight_decay = 0.0, p -= LR * 0.25 * (p - p_birth) before opt.step()
    anchor10  the same at strength 2.5
    rehearse  as zero; the birth family's taught items join the stream as a second family (stream seed + 100, before F1's)

    python3 exp38_erase.py life  <arm> <S|B> <seed> [--updates N] [--out DIR]    # N = 12288 (default) or a dry 256
    python3 exp38_erase.py check <2|3|4|5> <S|B> <seed> [--out DIR]             # 64-update checks; check 1 and 6 read the life receipts
    python3 exp38_erase.py check1 <S|B> <seed> [--out DIR]                      # zero's rows 6144 and 12288 against E26's stages 1 and 2

Every output path must not exist (exp20_reference.refuse_existing). arch26.install() is called first.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (environment gate via exp20_reference; not edited)

R, X16, E, X, arch26, T = C.R, C.X16, C.E, C.X, C.arch26, C.T
LR = C.LR
ARMS38 = {"zero": dict(target="zero", kappa=C.WD, rehearse=False),
          "none": dict(target="none", kappa=0.0, rehearse=False),
          "anchor": dict(target="birth", kappa=0.25, rehearse=False),
          "anchor10": dict(target="birth", kappa=2.5, rehearse=False),
          "rehearse": dict(target="zero", kappa=C.WD, rehearse=True)}
FULL = 2 * E.STAGE_UPDATES
ROWS = [0, 64, 128, 256, 512] + list(range(1024, FULL + 1, 1024))
COMMIT_ROWS = (6144, 12288)
OUT38 = X.OUT / "e38"
E26DIR = X.OUT / "e26"
BIRTH_TASKS = [E.NAME(*t) for t in E.DEV]          # add89 sub89 max89 min89
NEW_TASKS = ("add97", "sub97")
SOURCES38 = C.SOURCES + ("exp38_erase.py",)


EXP16_AT_BIRTH = ("a301ae8e", "f4559971a6c1770523d641431fe5d4bd56def3f5ee3cafc87a5b869509be50ae")
EXP16_NOW = ("deea398c", "c9fcedcfe5f53ff7ac480e020caa5de8fc1a5163a1fd3d276b402132d73e6350")


def admit(bfile, rule, seed):
    """exp26_candidate.admit_e26, with exp31_switch.admit_s's one recorded exception: the E26 births (and lives) were made with
    exp16_integrated.py at a301ae8e; the branch has it at deea398c (E21b resume and receipt machinery in exp16's own life/develop17, which
    exp26_candidate.life's loop does not call). Only that file, only that exact pair of hashes; every other check runs unchanged. Check 1 then
    tests arm zero against E26 bit for bit."""
    try:
        return C.admit_e26(bfile, rule, seed)
    except AssertionError as e:
        if "made by other sources: ['exp16_integrated.py']" not in str(e):
            raise
    rec = json.loads(Path(bfile).with_suffix(".json").read_text())["provenance"]["sources_sha256"]
    keep = C.source_hashes
    assert rec["exp16_integrated.py"] == EXP16_AT_BIRTH[1] and keep()["exp16_integrated.py"] == EXP16_NOW[1], "exp16_integrated.py differs in an unexpected way"
    C.source_hashes = lambda: dict(keep(), **{"exp16_integrated.py": rec["exp16_integrated.py"]})
    try:
        r = C.admit_e26(bfile, rule, seed)
    finally:
        C.source_hashes = keep
    return r[0], r[1], r[2] + "; exception: exp16_integrated.py a301ae8e (f4559971) -> deea398c (c9fcedcf), E21b resume machinery only, as exp31_switch.admit_s"


def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES38}
    import subprocess
    p["git"]["sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES38], cwd=HERE, capture_output=True,
                                                            text=True).stdout.strip().splitlines()
    return p


# ---------------------------------------------------------------- the step (exp26_candidate.step, with the pull before opt.step)
def step(m, opt, items, local, pull=None):
    x, y, mask = X.batch_tensor(items)
    logits = arch26.forward(m, x, local, True)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    if pull is not None:
        pull()
    opt.step()
    return float(loss.detach())


def make_pull(decayed, birth, kappa, form):
    if form == 1:
        def pull():
            with T.no_grad():
                for p, p0 in zip(decayed, birth):
                    p.sub_(LR * kappa * (p - p0))
    else:                                            # check 3's second form, written separately
        def pull():
            with T.no_grad():
                for p, p0 in zip(decayed, birth):
                    p.copy_(p0 + (1 - LR * kappa) * (p - p0))
    return pull


# ---------------------------------------------------------------- score 2: all pairs, no letters, split by wrap
def all_pairs(task):
    op, mod, code = task
    items, wrap = [], []
    rng = None
    for a in range(mod):
        for b in range(mod):
            items.append(E.render(task, a, b, 0, rng)); wrap.append(a + b >= mod if op == "add" else a - b < 0)
    return items, wrap


def score_split(m, items, wrap):
    outs = E.respond(m, [p for p, _ in items], None, cap=X16.CAP)
    ok = [o == t for o, (_, t) in zip(outs, items)]
    w = [o for o, f in zip(ok, wrap) if f]; nw = [o for o, f in zip(ok, wrap) if not f]
    return dict(wrap=dict(n=len(w), exact=sum(w), pct=100 * sum(w) / len(w)), nowrap=dict(n=len(nw), exact=sum(nw), pct=100 * sum(nw) / len(nw)))


# ---------------------------------------------------------------- held-out pairs of each family (check 6)
def held_keys(seed):
    keys = {}
    for t in E.DEV:
        keys[t[2][0]] = (t, set(E.split(t, .5, seed)[1]))
    for t in X16.F1:
        keys[t[2][0]] = (t, set(E.split(t, .25, seed)[1]))
    return keys


def parse_op(prompt):
    """(code byte, a, b) of a rendered single-operation prompt: code, two digits, operator, letters, two digits, '='."""
    code = prompt[0]
    a = int(bytes(prompt[1:3])); b = int(bytes(prompt[-3:-1]))
    assert prompt[-1] == ord("=") and chr(prompt[3]) in "+-><"
    return code, a, b


# ---------------------------------------------------------------- the life (exp26_candidate.life's loop, two blocks, with E38's arms)
def life(arm, rule, seed, out, cmd, updates=FULL, check=None):
    """check: None for a life; or dict(birth_zero, kappa, form, wd_override, target) for the 64-update checks (returns the model)."""
    local = C.ARMS[rule]
    arch26.install()
    cfg = dict(ARMS38[arm]) if arm in ARMS38 else {}
    if check is not None:
        cfg.update(check)
    form = cfg.get("form", 1)
    tag = f"e38_{arm}_{rule}_{seed}" if updates == FULL else f"e38_{arm}_{rule}_{seed}_dry{updates}"
    out = Path(out); path = out / f"{tag}.json"
    if check is None:
        R.refuse_existing(path, out / f"{tag}_row{updates}.pt")
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows_uncommitted").mkdir(exist_ok=True)
    bfile = C.birth_path(E26DIR, rule, seed); m, birth_sha, admission = admit(bfile, rule, seed)
    bsd = T.load(bfile)
    m.load_state_dict(bsd)
    desc = arch26.describe(m)
    opt = C.make_opt(m)
    decayed = list(opt.param_groups[0]["params"])
    names = {id(p): n for n, p in m.named_parameters()}
    pull, birth = None, None
    if cfg["target"] in ("none", "birth") or "wd_override" in cfg:
        opt.param_groups[0]["weight_decay"] = cfg.get("wd_override", 0.0)
    if cfg["target"] == "birth":
        birth = [p.detach().clone() for p in decayed]                     # taken once, right after the birth file is loaded
        if cfg.get("birth_zero"):
            birth = [T.zeros_like(p) for p in decayed]
        pull = make_pull(decayed, birth, cfg["kappa"], form)
    n_birth = sum(p.numel() for p in birth) if birth is not None else 0
    groups_read_back = R.opt_groups(m, opt)

    P, comp = X16.panels(seed)
    SP, strict_counts = X16.strict_panels(seed)
    taught = R.taught_panel(seed)
    tcheck = R.taught_panel_check(seed, P, comp, taught)
    assert tcheck["taught_in_F2_entries"] == len(taught) and tcheck["within_in_F2_entries"] == 0 and not any(tcheck["held_in_F2_entries"].values())
    f1c, f3c = {t[2][0] for t in X16.F1}, {t[2][0] for t in X16.F3}
    code_of = lambda p_: next((b_ for b_ in p_ if not 65 <= b_ <= 90), p_[0])
    fam_of = lambda p_: "F2" if code_of(p_) == ord("c") else "F1" if code_of(p_) in f1c else "F3" if code_of(p_) in f3c else "other"
    entries = {f: X16.family_entries(f, seed) for f in ("F1", "F2", "F3")}
    taught_keys = {tuple(p) for p, _ in taught}
    unseen_keys = {tuple(p) for n, v in comp.items() for p, _ in v}
    presented = dict(taught_panel_items_presented=0, taught_panel_distinct_presented=set(), within_or_held_presented=0)
    hk = held_keys(seed)
    held_presented = {}
    seen_streams = []
    if cfg["rehearse"]:
        birth_entries = [("op", t, a, b) for t in E.DEV for a, b in E.split(t, .5, seed)[0]]
        seen_streams = [X16.entry_stream(birth_entries, seed + 100)]       # then block 1 appends F1's stream as E26 does
    dg = C.Digest()
    t0 = time.process_time(); stages = []; rows = []
    score = X16.score
    s2 = {n: all_pairs(t) for t in E.DEV if (n := E.NAME(*t)) in ("add89", "sub89")}

    def evaluate():
        res = {n: {k: score(m, it, None) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, None) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], None, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, None, silence=True) for n, it in comp.items() if n != "within"}
        res["strict_k12"] = {n: score(m, it, None) for n, it in SP.items()}
        res["taught"] = score(m, taught, None)
        return res

    cpu_score = [0.0]

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g)
        r["birth_heldout_k12"] = {n: score(m, P[n][12], None) for n in BIRTH_TASKS}
        r["birth_allpairs_k0"] = {n: score_split(m, *v) for n, v in s2.items()}
        r["new_heldout_k12"] = {n: score(m, P[n][12], None) for n in NEW_TASKS}
        cpu_score[0] += time.process_time() - cb
        if check is None:
            wp = (out if g in COMMIT_ROWS else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
            r["weights"] = str(wp.relative_to(out)); r["weights_sha256"] = R.save_weights(m, wp); r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        r["cpu"] = time.process_time() - t0
        rows.append(r)
        pct = lambda d: round(100 * d["exact"] / d["n"], 1)
        print(tag, "row", g, "birth k12", {n: pct(v) for n, v in r["birth_heldout_k12"].items()},
              "wrap", {n: round(v["wrap"]["pct"], 1) for n, v in r["birth_allpairs_k0"].items()},
              "new k12", {n: pct(v) for n, v in r["new_heldout_k12"].items()}, f"cpu {r['cpu']:.0f}s", flush=True)

    birth_checks = []

    def check_birth(g):
        if birth is None or cfg.get("birth_zero"):
            return
        bit_equal = {names[id(p)]: bool(T.equal(p0, bsd[names[id(p)]])) for p, p0 in zip(decayed, birth)}
        differs = {names[id(p)]: not bool(T.equal(p.detach(), p0)) for p, p0 in zip(decayed, birth)}
        birth_checks.append(dict(updates=g, stored_bit_equal_to_birth_file=all(bit_equal.values()), decayed_differs_from_stored=all(differs.values()),
                                 not_equal=[k for k, v in bit_equal.items() if not v], unchanged=[k for k, v in differs.items() if not v]))

    settings = dict(experiment="E38", arm=arm, rule=rule, local=local, seed=seed, updates=updates, net="G1", model=desc,
                    decay_target=cfg["target"], decay_strength=cfg["kappa"], pull_form=form if birth is not None else None,
                    rehearse=cfg["rehearse"], stored_birth_values=n_birth, stored_birth_tensors=len(birth) if birth is not None else 0,
                    optimizer_groups_read_back=groups_read_back, optimizer_record_exp26=C.opt_record(m, opt),
                    optimizer_record_note="exp26_candidate.opt_record writes weight_decay=0.25 at its top level whatever the groups hold; "
                                          "decay_target and decay_strength are this arm's", stage_updates=E.STAGE_UPDATES, batch=X.BATCH,
                    panel_sha256=X16.panel_sha(P, comp, SP), birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_admission=admission,
                    rows=[r for r in ROWS if r <= updates], check=check, provenance=provenance(cmd))
    stages.append(dict(stage=0, task=None, settings=settings))
    if check is None:
        score_row(0)

    def write(complete=None):
        if check is None:
            path.write_text(json.dumps(dict(experiment="E38", kind="life", arm=arm, rule=rule, seed=seed, schedule=X16.SCHEDULE[:2], stages=stages,
                                            rows=rows, birth_checks=birth_checks, held_pairs_presented=held_presented, complete=complete), indent=1))

    g = 0
    for si, (fam, block) in enumerate(X16.SCHEDULE[:2], 1):
        if g >= updates:
            break
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
                if p_[0] in hk:                                        # check 6: single-operation items of the birth family and F1
                    code, a, b = parse_op(p_)
                    if (a, b) in hk[code][1]:
                        nm = E.NAME(*hk[code][0]); held_presented[nm] = held_presented.get(nm, 0) + 1
            dg.add(batch)
            losses.append(step(m, opt, batch, local, pull))
            g += 1
            if si == 1 and u == 0:
                stages[0]["settings"]["zero_grad_first_update"] = X16.zero_grad_report(m, X.batch_tensor(batch)[0])
            if check is not None and g == updates:
                return m, dict(settings=settings, birth=birth, decayed=decayed, names=names, bsd=bsd, opt=opt)
            if g == 64:
                check_birth(g)
            if g in ROWS and g <= updates:
                score_row(g); write()
            if g >= updates:
                break
        if u + 1 == E.STAGE_UPDATES:
            cb = time.process_time(); sc = evaluate(); ca = time.process_time(); cpu_score[0] += ca - cb
            stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)),
                               finite_losses=all(v == v and abs(v) != float("inf") for v in losses), cpu=cb - t0, scores=sc,
                               cpu_scoring_so_far=cpu_score[0], cpu_updates_so_far=ca - t0 - cpu_score[0], optimizer_steps=R.opt_steps(opt),
                               data_digest=dg.value(), exposure_rows_tokens=expo, exposure_note="[rows, bytes under the loss mask]; loss=answer",
                               exposure_panels=dict(presented, taught_panel_distinct_presented=len(presented["taught_panel_distinct_presented"]))))
            print(tag, "after", f"{fam}.{block}", "loss", stages[-1]["loss"], "digest", dg.value()["sha256"][:16], "expo", expo, flush=True)
        else:
            stages.append(dict(stage=si, task=f"{fam}.{block}", partial_updates=u + 1, data_digest=dg.value(), exposure_rows_tokens=expo,
                               loss_last512=sum(losses[-512:]) / min(512, len(losses))))
        write()
    check_birth(g)
    assert presented["within_or_held_presented"] == 0, "a within or held-pair panel item was presented in training"
    assert not held_presented, f"held-out single-operation pairs were presented: {held_presented}"
    write(dict(updates=g, optimizer_steps=R.opt_steps(opt), optimizer_groups_read_back=R.opt_groups(m, opt), data_digest=dg.value(),
               tensors_sha256=R.tensors_sha(m.state_dict()), cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
               cpu_updates=time.process_time() - t0 - cpu_score[0]))
    print(tag, "complete", f"cpu {time.process_time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- checks 2 to 5 (64 updates each)
def max_rel(a, b):
    """Per tensor: largest absolute difference over the tensor's largest absolute entry."""
    return {n: float((a[n] - b[n]).abs().max() / a[n].abs().max().clamp_min(1e-30)) for n in a}


def sd_of(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def run_check(which, rule, seed, out, cmd):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rp = out / f"e38_check{which}_{rule}_{seed}.json"
    R.refuse_existing(rp)
    t0 = time.process_time()
    res = {}
    if which == "2":
        zero = sd_of(life("zero", rule, seed, out, cmd, 64, check={})[0])
        a25 = sd_of(life("anchor", rule, seed, out, cmd, 64, check=dict(birth_zero=True))[0])
        own25 = sd_of(life("zero", rule, seed, out, cmd, 64, check=dict(wd_override=2.5))[0])
        a250 = sd_of(life("anchor10", rule, seed, out, cmd, 64, check=dict(birth_zero=True))[0])
        r1, r2 = max_rel(zero, a25), max_rel(own25, a250)
        res = dict(kappa_0_25_vs_torch_decay_0_25=r1, kappa_2_5_vs_torch_decay_2_5=r2, tolerance=1e-5,
                   embedding_table_equal=dict(k025=bool(T.equal(zero["book.weight"], a25["book.weight"])), k25=bool(T.equal(own25["book.weight"], a250["book.weight"]))),
                   **{"pass": max(r1.values()) <= 1e-5 and max(r2.values()) <= 1e-5})
    elif which == "3":
        res = {}
        ok = True
        for arm in ("anchor", "anchor10"):
            f1 = sd_of(life(arm, rule, seed, out, cmd, 64, check=dict(form=1))[0])
            f2 = sd_of(life(arm, rule, seed, out, cmd, 64, check=dict(form=2))[0])
            r = max_rel(f1, f2); res[arm] = r; ok &= max(r.values()) <= 1e-6
        res.update(tolerance=1e-6, **{"pass": ok})
    elif which == "4":
        res = {}; ok = True
        none = sd_of(life("none", rule, seed, out, cmd, 64, check={})[0])
        for arm in ("anchor", "anchor10"):
            m, ctx = life(arm, rule, seed, out, cmd, 64, check={})
            bit = {ctx["names"][id(p)]: bool(T.equal(p0, ctx["bsd"][ctx["names"][id(p)]])) for p, p0 in zip(ctx["decayed"], ctx["birth"])}
            dif = {ctx["names"][id(p)]: not bool(T.equal(p.detach(), p0)) for p, p0 in zip(ctx["decayed"], ctx["birth"])}
            alias = any(p.data_ptr() == p0.data_ptr() for p, p0 in zip(ctx["decayed"], ctx["birth"]))
            vs_none = {n: not bool(T.equal(v, none[n])) for n, v in sd_of(m).items() if n != "book.weight"}
            res[arm] = dict(stored_values=sum(p.numel() for p in ctx["birth"]), stored_tensors=len(ctx["birth"]), stored_bit_equal=bit,
                            decayed_differs_from_stored=dif, any_storage_aliased=alias, differs_from_none_after_64=vs_none)
            ok &= all(bit.values()) and all(dif.values()) and not alias and all(vs_none.values()) and res[arm]["stored_values"] == 1122049 and len(ctx["birth"]) == 12
        res["note"] = "the end-of-life part of check 4 is in each anchor life's receipt (birth_checks at 64 and 12288)"
        res["pass"] = ok
    elif which == "5":
        m, ctx = life("none", rule, seed, out, cmd, 64, check={})
        g = R.opt_groups(m, ctx["opt"])
        res = dict(groups=g, **{"pass": [x["weight_decay"] for x in g] == [0.0, 0.0]})
    rec = dict(experiment="E38", check=which, rule=rule, seed=seed, command=cmd, provenance=provenance(cmd), cpu_seconds=time.process_time() - t0, result=res)
    rp.write_text(json.dumps(rec, indent=1))
    print("check", which, rule, "pass" if res.get("pass") else "FAIL", f"cpu {rec['cpu_seconds']:.0f}s", flush=True)


def check1(rule, seed, out, cmd):
    """Check 1: arm zero's stages 1 and 2 against E26's: batch digest, loss of the last 512 updates, every score of evaluate()."""
    out = Path(out); rp = out / f"e38_check1_{rule}_{seed}.json"; R.refuse_existing(rp)
    mine = json.loads((out / f"e38_zero_{rule}_{seed}.json").read_text())
    e26 = json.loads((E26DIR / f"e26_life_{rule}_{seed}.json").read_text())
    res = {}
    for i in (1, 2):
        a, b = mine["stages"][i], e26["stages"][i]
        res[f"stage{i}"] = dict(task=[a["task"], b["task"]], digest_equal=a["data_digest"] == b["data_digest"], loss_equal=a["loss"] == b["loss"],
                                loss=[a["loss"], b["loss"]], scores_equal=a["scores"] == b["scores"],
                                exposure_equal=a["exposure_rows_tokens"] == b["exposure_rows_tokens"])
        if not res[f"stage{i}"]["scores_equal"]:
            res[f"stage{i}"]["score_differences"] = [k for k in b["scores"] if a["scores"].get(k) != b["scores"][k]]
    res["pass"] = all(v["digest_equal"] and v["loss_equal"] and v["scores_equal"] for v in res.values())
    rec = dict(experiment="E38", check="1", rule=rule, seed=seed, command=cmd, provenance=provenance(cmd), result=res)
    rp.write_text(json.dumps(rec, indent=1))
    print("check 1", rule, "pass" if res["pass"] else "FAIL", res, flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    outdir = opt_("--out", str(OUT38))
    if a[0] == "life":
        assert a[1] in ARMS38 and a[2] in C.ARMS
        life(a[1], a[2], int(a[3]), outdir, cmd, int(opt_("--updates", FULL)))
    elif a[0] == "check":
        assert a[1] in ("2", "3", "4", "5") and a[2] in C.ARMS
        run_check(a[1], a[2], int(a[3]), outdir, cmd)
    elif a[0] == "check1":
        check1(a[1], int(a[2]), outdir, cmd)
    else:
        raise SystemExit(__doc__)
