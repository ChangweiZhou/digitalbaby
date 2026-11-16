"""E43 runner (brief OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md, revision 2): is the first family lost when the learner
then learns something unrelated? Four arms from each of the six E26 births, 4,096 updates:

    conflict          the second family alone (mod 97), X16.entry_stream(X16.family_entries("F1", seed), seed + 101); weight_decay 0 in group 0
    conflict-decay    the same stream; weight_decay 0.25 (E26's life, E38's arm zero)
    unrelated         e43_task.batches(seed) (the coordinator's specification file, copied byte for byte); weight_decay 0
    unrelated-decay   the same stream; 0.25

Update: exp26_candidate.step's text (as exp38_erase.step with no pull); a fresh exp26_candidate.make_opt(m). Births admitted by
exp26_candidate.admit_e26 through exp38_erase.admit (the one recorded exception for exp16_integrated.py, checked by check 1).

    python3 exp43_unrelated.py run    <arm> <S|B> <seed> [--updates N] [--out DIR]     # N = 4096 (default) or a dry 64
    python3 exp43_unrelated.py extend <arm> <S|B> <seed> [--out DIR]                   # unrelated arms: 4,096 -> 16,384 (rows 8,192, 16,384)

Scores at each row (the brief's numbering): 1 first family, all pairs without letters (add/sub split by reduction, max/min on all pairs);
2 first family held out at k=12; 3 the new task; 4 weight movement per group (relative norm of the difference from birth, cosine);
6 the change of the state at '=' on the panel of score 2. Score 5 (decay alone) is e43_checks.py's, no run.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp38_erase as E38            # noqa: E402  (not edited; admit, step and provenance helpers)
import e43_task as TK                # noqa: E402  (byte-identical copy of the coordinator's specification file)

C, R, X16, E, X, arch26, T = E38.C, E38.R, E38.X16, E38.E, E38.X, E38.arch26, E38.T
ARMS43 = {"conflict": dict(stream="conflict", decay=0.0), "conflict-decay": dict(stream="conflict", decay=0.25),
          "unrelated": dict(stream="unrelated", decay=0.0), "unrelated-decay": dict(stream="unrelated", decay=0.25)}
DOSE, LONG = 4096, 16384
ROWS = (0, 64, 256, 1024, 2048, 4096)
EXT_ROWS = (8192, 16384)
OUT43 = X.OUT / "e43"
GROUPS = {"embedding": ["book.weight"], "l1": ["l1.weight", "l1.bias"], "input map": ["rnn.weight_ih_l0", "rnn.bias_ih_l0"],
          "recurrent matrix": ["rnn.weight_hh_l0", "rnn.bias_hh_l0"], "lm": ["lm.weight", "lm.bias"], "l2": ["l2.weight", "l2.bias"],
          "l3": ["l3.weight", "l3.bias"]}
SOURCES43 = E38.SOURCES38 + ("exp43_unrelated.py", "e43_task.py")


def provenance(cmd):
    p = C.provenance(cmd)
    p["sources_sha256"] = {s: X16.sha_file(HERE / s) for s in SOURCES43}
    import subprocess
    p["git"]["sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES43], cwd=HERE, capture_output=True,
                                                            text=True).stdout.strip().splitlines()
    return p


def keep_rows(arm, rule, seed):
    """Rows whose weights are committed: 4,096 (and the extension's 16,384) for every run; 64 and 1,024 of seed 73129's unrelated arm."""
    k = {DOSE, LONG}
    if arm == "unrelated" and seed == 73129:
        k |= {64, 1024}
    return k


# ---------------------------------------------------------------- scores
def first_family_items():
    """Score 1: all 7,921 pairs of each mod-89 operation at k = 0; add and sub split by reduction (a + b >= 89, a - b < 0)."""
    out = {}
    for t in E.DEV:
        op, mod, _ = t
        items, red = [], []
        for a in range(mod):
            for b in range(mod):
                items.append(E.render(t, a, b, 0, None)); red.append(a + b >= mod if op == "add" else a - b < 0 if op == "sub" else None)
        out[E.NAME(*t)] = (items, red)
    return out


def score1(m, s1):
    res = {}
    for n, (items, red) in s1.items():
        outs = E.respond(m, [p for p, _ in items], None, cap=X16.CAP)
        ok = [o == t for o, (_, t) in zip(outs, items)]
        if n in ("add89", "sub89"):
            r = [o for o, f in zip(ok, red) if f]; nr = [o for o, f in zip(ok, red) if not f]
            res[n] = dict(reduction=dict(n=len(r), exact=sum(r), pct=100 * sum(r) / len(r)), other=dict(n=len(nr), exact=sum(nr), pct=100 * sum(nr) / len(nr)))
        else:
            res[n] = dict(all=dict(n=len(ok), exact=sum(ok), pct=100 * sum(ok) / len(ok)))
    res["reduction_score"] = (res["add89"]["reduction"]["pct"] + res["sub89"]["reduction"]["pct"]) / 2
    return res


def movement(m, bsd):
    """Score 4: per group, |w - w_birth| / |w_birth| and cosine(w, w_birth), over the group's tensors concatenated; and per tensor."""
    sd = {n: p.detach() for n, p in m.named_parameters()}
    per = {}
    for g, names in list(GROUPS.items()) + [(n, [n]) for n in sd]:
        w = T.cat([sd[n].flatten().double() for n in names]); w0 = T.cat([bsd[n].flatten().double() for n in names])
        per[g] = dict(rel_diff=float((w - w0).norm() / w0.norm()), cosine=float(w @ w0 / (w.norm() * w0.norm())))
    return dict(groups={g: per[g] for g in GROUPS}, tensors={n: per[n] for n in sd})


@T.no_grad()
def eq_states(m, items):
    """The recurrent state at the '=' position (the last prompt byte) for each item, no letters added."""
    out = []
    groups = {}
    for i, (p, _) in enumerate(items):
        groups.setdefault(len(p), []).append(i)
    res = [None] * len(items)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            x = T.tensor([[E.BOS, *items[i][0]] for i in ids])
            hs = arch26.forward(m, x, False, True)[1]["h"]
            for q, i in enumerate(ids):
                res[i] = hs[-1][q].clone()
    return T.stack(res)


# ---------------------------------------------------------------- one run
def run(arm, rule, seed, out, cmd, updates=DOSE, extend=False):
    local = C.ARMS[rule]
    arch26.install()
    cfg = ARMS43[arm]
    tag = f"e43_{arm}_{rule}_{seed}" + ("" if updates in (DOSE, LONG) else f"_dry{updates}")
    out = Path(out); path = out / (f"{tag}_ext{LONG}.json" if extend else f"{tag}.json")
    R.refuse_existing(path)
    out.mkdir(parents=True, exist_ok=True); (out / "rows_uncommitted").mkdir(exist_ok=True)
    bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
    bsd = T.load(bfile)
    m.load_state_dict(bsd)
    birth_tsha = R.tensors_sha(m.state_dict())
    opt = C.make_opt(m)
    opt.param_groups[0]["weight_decay"] = cfg["decay"]
    groups_read_back = R.opt_groups(m, opt)

    P, comp = X16.panels(seed)
    held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
    s1 = first_family_items()
    s2_items = [it for n in held for it in held[n]]
    for items in [s2_items] + [v[0] for v in s1.values()]:       # check 4, first-family side: prompt bytes and answer bytes (EOS aside) below 128
        assert all(b < 128 for p, t in items for b in p) and all(b < 128 for p, t in items for b in t[:-1]) and all(t[-1] == E.EOS for _, t in items)
    if cfg["stream"] == "conflict":
        st = X16.entry_stream(X16.family_entries("F1", seed), seed + 101)
        nxt = lambda: [next(st) for _ in range(X.BATCH)]
        new_panels = {n: P[n][12] for n in ("add97", "sub97")}
    else:
        bt = TK.batches(seed, X.BATCH)
        nxt = lambda: next(bt)
        new_panels = {"held": TK.panel(seed, "held"), "taught": TK.panel(seed, "taught")}
    dg = C.Digest()
    h0 = eq_states(m, s2_items); h0_mean = float(h0.abs().mean())
    t0 = time.process_time(); cpu_score = [0.0]; rows = []
    kr = keep_rows(arm, rule, seed)

    def score_row(g):
        cb = time.process_time()
        r = dict(updates=g, data_digest=dg.value())
        r["score1_first_family_all_pairs_k0"] = score1(m, s1)
        r["score2_first_family_heldout_k12"] = {n: X16.score(m, it, None) for n, it in held.items()}
        r["score3_new_task"] = {n: X16.score(m, it, None) for n, it in new_panels.items()}
        r["score4_movement"] = movement(m, bsd)
        h = eq_states(m, s2_items)
        r["score6_state_change"] = dict(mean_abs_change=float((h - h0).abs().mean()), mean_abs_state_row0=h0_mean,
                                        ratio=float((h - h0).abs().mean()) / h0_mean)
        cpu_score[0] += time.process_time() - cb
        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        wp = (out if g in kr else out / "rows_uncommitted") / f"{tag}_row{g}.pt"
        if updates != 64 or g == 64:
            r["weights"] = str(wp.relative_to(out)); r["weights_sha256"] = R.save_weights(m, wp)
        r["cpu"] = time.process_time() - t0
        rows.append(r)
        s = r["score1_first_family_all_pairs_k0"]; pc = lambda d: round(100 * d["exact"] / d["n"], 1)
        print(tag, "row", g, "reduction", round(s["reduction_score"], 1), "other", round(s["add89"]["other"]["pct"], 1), round(s["sub89"]["other"]["pct"], 1),
              "max/min", round(s["max89"]["all"]["pct"], 1), round(s["min89"]["all"]["pct"], 1),
              "held k12", {n: pc(v) for n, v in r["score2_first_family_heldout_k12"].items()}, "new", {n: pc(v) for n, v in r["score3_new_task"].items()},
              "state", round(r["score6_state_change"]["ratio"], 3), f"cpu {r['cpu']:.0f}s", flush=True)

    settings = dict(experiment="E43", arm=arm, rule=rule, local=local, seed=seed, updates=updates, stream=cfg["stream"], decay=cfg["decay"],
                    optimizer_groups_read_back=groups_read_back, optimizer_record_exp26=C.opt_record(m, opt),
                    optimizer_record_note="exp26_candidate.opt_record writes weight_decay=0.25 at its top level whatever the groups hold; `decay` is this arm's",
                    model=arch26.describe(m), birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_tensors_sha256=birth_tsha,
                    birth_admission=admission, batch=X.BATCH, rows=[r for r in ROWS if r <= updates] + (list(EXT_ROWS) if extend else []),
                    committed_rows=sorted(kr), task_file_sha256=X16.sha_file(HERE / "e43_task.py"), extend=extend, provenance=provenance(cmd))
    g = 0
    if extend:                                                     # resume from the 4,096 checkpoint, the stream fast-forwarded
        base = json.loads((out / f"{tag}.json").read_text())
        assert base.get("complete") and base["complete"]["updates"] == DOSE and cfg["stream"] == "unrelated"
        ck = T.load(out / "rows_uncommitted" / f"{tag}_ckpt{DOSE}.pt")
        assert X16.sha_file(out / "rows_uncommitted" / f"{tag}_ckpt{DOSE}.pt") == base["complete"]["checkpoint_sha256"]
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        assert R.tensors_sha(m.state_dict()) == base["rows"][-1]["tensors_sha256"]
        assert R.opt_groups(m, opt) == groups_read_back
        for _ in range(DOSE):
            dg.add(nxt())
        g = DOSE
        assert dg.value() == base["rows"][-1]["data_digest"], "the fast-forwarded stream is not the one the run saw"
        settings["resumed_from"] = dict(receipt=base["experiment"] + " " + tag, tensors_sha256=base["rows"][-1]["tensors_sha256"], data_digest=dg.value())
    else:
        score_row(0)
    target = LONG if extend else updates
    stop_rows = EXT_ROWS if extend else ROWS

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E43", kind="run", arm=arm, rule=rule, seed=seed, settings=settings, rows=rows, complete=complete), indent=1))

    while g < target:
        batch = nxt()
        if cfg["stream"] == "unrelated":                           # check 4: every prompt byte of the unrelated task is 128 or above
            assert all(b >= 128 for p, _ in batch for b in p)
        dg.add(batch)
        E38.step(m, opt, batch, local)
        g += 1
        if g in stop_rows:
            score_row(g); write()
    ckpt_sha = None
    if not extend and updates == DOSE and cfg["stream"] == "unrelated":
        cp = out / "rows_uncommitted" / f"{tag}_ckpt{DOSE}.pt"
        T.save(dict(model=m.state_dict(), opt=opt.state_dict()), cp); ckpt_sha = X16.sha_file(cp)
    write(dict(updates=g, optimizer_steps=R.opt_steps(opt), optimizer_groups_read_back=R.opt_groups(m, opt), data_digest=dg.value(),
               tensors_sha256=R.tensors_sha(m.state_dict()), checkpoint_sha256=ckpt_sha, cpu_total=time.process_time() - t0, cpu_scoring=cpu_score[0],
               cpu_updates=time.process_time() - t0 - cpu_score[0]))
    print(tag, "complete", f"cpu {time.process_time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    outdir = opt_("--out", str(OUT43))
    assert a[1] in ARMS43 and a[2] in C.ARMS
    if a[0] == "run":
        run(a[1], a[2], int(a[3]), outdir, cmd, int(opt_("--updates", DOSE)))
    elif a[0] == "extend":
        assert a[1].startswith("unrelated")
        run(a[1], a[2], int(a[3]), outdir, cmd, LONG, extend=True)
    else:
        raise SystemExit(__doc__)
