"""E51, Part B (brief TASK_E51_A_NEW_TASK_LEARNED_IN_CELLS_HELD_IN_RESERVE_20261117.md, revision 2): how much rehearsal keeps the first
family, at what times, and with what steps. E43's own setting: the E26 birth of the seed at the present size, the unrelated stream, no
decay, no embedding and no reserve; arch26.forward and exp38_erase.step (exp26_candidate.step's text), as exp43_unrelated.run.

    B1  share     the unrelated stream with each item drawn from the first family's taught half with chance 1/k (k = 4, 8, 16, 32): an item
                  is a rehearsal item when Random(f"e51|{seed}|joint|{k}").randrange(k) == 0; the two streams are consumed in order as
                  e43_addendum2.rehearse does. 4,096 updates; E43's scores at its rows; the probe every 64 updates to 1,024, every 256 after.
                  `k2addendum` is check 8: k = 2 with the addendum's own choice (Random(f"e43|{seed}|joint").randrange(2), 1 = rehearsal).
    B2  intervals one optimizer; eight cycles of a gap (256 updates of the unrelated stream alone) and a review (the first family's taught
                  stream alone, continued from cycle to cycle; the probe every 16 updates; until the probe's first-family share is at least
                  90% of its value at update 0, or 1,024 review updates).
    B3  steps     E43's `unrelated` arm at learning rate 0.001, 0.0003, 0.0001 for 4,096, 13,824, 40,960 updates; the probe every 16 updates
                  to 256 and every 64 after; E43's scores at update 0, at update 64 and at the end (and at 4,096).

The probe: 400 reduction pairs of the first family without letters (200 of add89, then 200 of sub89, Random(f"e51|{seed}|probe").sample from
the reduction pairs of each task in first_family_items()'s order) and E43's held-out panel of the unrelated task (300 items).

    python3 exp51_partb.py share     <4|8|16|32|k2addendum> <S|B> <seed> [--out DIR]
    python3 exp51_partb.py intervals <S|B> <seed> [--out DIR]
    python3 exp51_partb.py steps     <0.001|0.0003|0.0001> <S|B> <seed> [--out DIR]
    python3 exp51_partb.py check     <8|9|10|11> <seed>

Part A's runner (exp51_reserve.py) is not imported: nothing here uses the keyed forward. Every output path must not exist. One thread.
"""
import json
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp43_unrelated as U          # noqa: E402  (not edited)
import e43_addendum2 as A2           # noqa: E402  (not edited)

E38, C, R, X16, E, X, arch26, T, TK = U.E38, U.C, U.R, U.X16, U.E, U.X, U.arch26, U.T, U.TK
OUTB = X.OUT / "e51" / "partB"
OWN = ("exp51_partb.py", "exp43_unrelated.py", "e43_addendum2.py", "e43_task.py", "exp38_erase.py")
STEPS = {"0.001": (0.001, 4096), "0.0003": (0.0003, 13824), "0.0001": (0.0001, 40960)}
GAP, REVIEW_MAX, CYCLES, REVIEW_PROBE = 256, 1024, 8, 16


def provenance(cmd):
    p = U.provenance(cmd)
    p["e51_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    import subprocess
    p["git"]["e51_sources_modified_vs_commit"] = subprocess.run(["git", "status", "--porcelain", "--", *OWN], cwd=HERE, capture_output=True,
                                                                text=True).stdout.strip().splitlines()
    return p


def family_stream(seed):
    """E38's rehearsal stream: the first family's taught half."""
    return X16.entry_stream([("op", t, a, b) for t in E.DEV for a, b in E.split(t, .5, seed)[0]], seed + 100)


def probe_items(seed, s1):
    rng = Random(f"e51|{seed}|probe"); out = {}
    for n in ("add89", "sub89"):
        items, red = s1[n]
        idx = rng.sample([i for i, f in enumerate(red) if f], 200)
        out[n] = idx
    return out


class Bench:
    """The birth, the optimizer, the panels, E43's scores and the probe, for one run."""

    def __init__(self, rule, seed, lr=None):
        self.rule, self.seed, self.local = rule, seed, C.ARMS[rule]
        arch26.install()
        bfile = C.birth_path(E38.E26DIR, rule, seed); m, birth_sha, admission = E38.admit(bfile, rule, seed)
        self.bsd = T.load(bfile); m.load_state_dict(self.bsd); self.m = m
        self.opt = C.make_opt(m); self.opt.param_groups[0]["weight_decay"] = 0.0
        if lr is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr
        self.groups = R.opt_groups(m, self.opt)
        P, comp = X16.panels(seed)
        self.held = {n: P[n][12] for n in ("add89", "sub89", "max89", "min89")}
        self.s1 = U.first_family_items(); self.s2_items = [it for n in self.held for it in self.held[n]]
        self.new_panels = {"held": TK.panel(seed, "held"), "taught": TK.panel(seed, "taught")}
        self.pidx = probe_items(seed, self.s1)
        self.probe_ff = [self.s1[n][0][i] for n in ("add89", "sub89") for i in self.pidx[n]]
        self.h0 = U.eq_states(m, self.s2_items); self.h0_mean = float(self.h0.abs().mean())
        self.start = dict(birth=str(bfile.relative_to(HERE)), birth_sha256=birth_sha, birth_tensors_sha256=R.tensors_sha(m.state_dict()), birth_admission=admission)
        self.t0 = time.process_time(); self.cpu_score = 0.0; self.cpu_probe = 0.0

    def e43_row(self, g, dg):
        cb = time.process_time(); m = self.m
        r = dict(updates=g, data_digest=dg.value())
        r["score1_first_family_all_pairs_k0"] = U.score1(m, self.s1)
        r["score2_first_family_heldout_k12"] = {n: X16.score(m, it, None) for n, it in self.held.items()}
        r["score3_new_task"] = {n: X16.score(m, it, None) for n, it in self.new_panels.items()}
        r["score4_movement"] = U.movement(m, self.bsd)
        h = U.eq_states(m, self.s2_items)
        r["score6_state_change"] = dict(mean_abs_change=float((h - self.h0).abs().mean()), mean_abs_state_row0=self.h0_mean, ratio=float((h - self.h0).abs().mean()) / self.h0_mean)
        r["tensors_sha256"] = R.tensors_sha(m.state_dict())
        self.cpu_score += time.process_time() - cb; r["cpu"] = time.process_time() - self.t0
        s = r["score1_first_family_all_pairs_k0"]
        print("  E43 scores at", g, "reduction", round(s["reduction_score"], 1), "max/min", round(s["max89"]["all"]["pct"], 1), round(s["min89"]["all"]["pct"], 1),
              "new", {n: round(100 * v["exact"] / v["n"], 1) for n, v in r["score3_new_task"].items()}, f"cpu {r['cpu']:.0f}s", flush=True)
        return r

    def probe(self, g, **extra):
        cb = time.process_time()
        outs = E.respond(self.m, [p for p, _ in self.probe_ff], None, cap=X16.CAP)
        ff = sum(o == t for o, (_, t) in zip(outs, self.probe_ff))
        un = X16.score(self.m, self.new_panels["held"], None)["exact"]
        self.cpu_probe += time.process_time() - cb
        return dict(updates=g, first_family_exact=ff, first_family_pct=100 * ff / len(self.probe_ff), unrelated_held_exact=un, unrelated_held_pct=100 * un / 300, **extra)

    def step(self, batch):
        return E38.step(self.m, self.opt, batch, self.local)

    def finish(self):
        t = time.process_time() - self.t0
        return dict(optimizer_steps=R.opt_steps(self.opt), optimizer_groups_read_back=R.opt_groups(self.m, self.opt), tensors_sha256=R.tensors_sha(self.m.state_dict()),
                    cpu_total=t, cpu_scoring=self.cpu_score, cpu_probe=self.cpu_probe, cpu_updates=t - self.cpu_score - self.cpu_probe)


def stream_run(tag, b, nxt, dose, score_rows, probe_rows, settings, out, cmd):
    """Updates from a stream of single items (batches of 64 consecutive items); E43's scores at score_rows, the probe at probe_rows."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); path = out / f"{tag}.json"; R.refuse_existing(path)
    dg = C.Digest(); counts = {"unrelated": 0, "first_family": 0}; rows, probes, losses = [], [], []
    settings = dict(settings, experiment="E51", part="B", rule=b.rule, local=b.local, seed=b.seed, updates=dose, decay=0.0, start=b.start, batch=X.BATCH,
                    optimizer_groups_read_back=b.groups, score_rows=sorted(score_rows), probe_rows=sorted(probe_rows), probe_items_first_family=b.pidx,
                    update="exp38_erase.step (exp26_candidate.step's text; arch26.forward)", provenance=provenance(cmd))

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E51", kind="run", tag=tag, settings=settings, rows=rows, probe=probes, complete=complete), indent=1))

    def at(g):
        if g in probe_rows:
            probes.append(b.probe(g, loss_since_last_probe=(sum(losses) / len(losses)) if losses else None, items_so_far=dict(counts))); losses.clear()
            p = probes[-1]; print(tag, "probe", g, "first family", p["first_family_pct"], "unrelated held", round(p["unrelated_held_pct"], 1), flush=True)
        if g in score_rows:
            rows.append(b.e43_row(g, dg))
        if g in probe_rows or g in score_rows:
            write()

    at(0); g = 0
    while g < dose:
        batch = []
        for _ in range(X.BATCH):
            kind, it = nxt(); counts[kind] += 1
            assert all(v >= 128 for v in it[0]) if kind == "unrelated" else all(v < 128 for v in it[0])
            batch.append(it)
        dg.add(batch); losses.append(b.step(batch)); g += 1
        at(g)
    wp = out / f"{tag}_row{dose}.pt"; R.save_weights(b.m, wp)
    write(dict(updates=g, items=counts, data_digest=dg.value(), weights=wp.name, weights_sha256=X16.sha_file(wp), **b.finish()))
    print(tag, "complete", flush=True)


def share(k, rule, seed, out, cmd):
    b = Bench(rule, seed)
    unrel = TK.stream(seed); fam = family_stream(seed)
    if k == "k2addendum":                                   # check 8: the addendum's own choice
        rj = Random(f"e43|{seed}|joint"); note = f"Random('e43|{seed}|joint').randrange(2): 0 -> the unrelated stream, 1 -> the first family's taught half"
        nxt = lambda: ("unrelated", next(unrel)) if rj.randrange(2) == 0 else ("first_family", next(fam))
    else:
        kk = int(k); assert kk in (4, 8, 16, 32)
        rj = Random(f"e51|{seed}|joint|{kk}"); note = f"Random('e51|{seed}|joint|{kk}').randrange({kk}): 0 -> the first family's taught half, otherwise the unrelated stream"
        nxt = lambda: ("first_family", next(fam)) if rj.randrange(kk) == 0 else ("unrelated", next(unrel))
    probe_rows = set(range(0, 1025, 64)) | set(range(1024, U.DOSE + 1, 256))
    stream_run(f"e51_B1_share_{k}_{rule}_{seed}", b, nxt, U.DOSE, set(U.ROWS), probe_rows, dict(set="B1", k=k, stream=note), out, cmd)


def steps(key, rule, seed, out, cmd):
    lr, dose = STEPS[key]
    b = Bench(rule, seed, lr=lr)
    assert all(g["lr"] == lr for g in b.groups)
    unrel = TK.stream(seed)
    probe_rows = set(range(0, 257, 16)) | set(range(256, dose + 1, 64))
    stream_run(f"e51_B3_steps_{key}_{rule}_{seed}", b, lambda: ("unrelated", next(unrel)), dose, {0, 64, 4096, dose}, probe_rows,
               dict(set="B3", learning_rate=lr, stream="e43_task.stream(seed) alone (E43's arm `unrelated`)"), out, cmd)


def intervals(rule, seed, out, cmd):
    b = Bench(rule, seed); tag = f"e51_B2_intervals_{rule}_{seed}"
    out = Path(out); out.mkdir(parents=True, exist_ok=True); path = out / f"{tag}.json"; R.refuse_existing(path)
    unrel = TK.batches(seed, X.BATCH); fam = family_stream(seed)
    dg = C.Digest(); rows, cycles, trace = [], [], []
    settings = dict(experiment="E51", part="B", set="B2", rule=rule, local=b.local, seed=seed, decay=0.0, start=b.start, batch=X.BATCH, gap=GAP, review_max=REVIEW_MAX,
                    cycles=CYCLES, review_probe_every=REVIEW_PROBE, optimizer_groups_read_back=b.groups, probe_items_first_family=b.pidx,
                    streams="gap: e43_task.batches(seed); review: the first family's taught half (X16.entry_stream, seed + 100), 64 consecutive items a batch, continued from cycle to cycle",
                    update="exp38_erase.step (exp26_candidate.step's text; arch26.forward); one optimizer for the whole run", provenance=provenance(cmd))

    def write(complete=None):
        path.write_text(json.dumps(dict(experiment="E51", kind="run", tag=tag, settings=settings, rows=rows, cycles=cycles, probe=trace, complete=complete), indent=1))

    g = 0
    p0 = b.probe(0, phase="start"); trace.append(p0); rows.append(b.e43_row(0, dg)); write()
    goal = 0.9 * p0["first_family_pct"]
    settings["probe_first_family_pct_at_update_0"] = p0["first_family_pct"]; settings["review_goal_pct"] = goal
    for c in range(1, CYCLES + 1):
        for i in range(GAP):
            batch = next(unrel); assert all(v >= 128 for p, _ in batch for v in p)
            dg.add(batch); b.step(batch); g += 1
            if (i + 1) % 64 == 0:
                trace.append(b.probe(g, phase="gap", cycle=c, updates_in_phase=i + 1))
        pg = trace[-1]; n = 0; pr = pg
        while pr["first_family_pct"] < goal and n < REVIEW_MAX:
            batch = [next(fam) for _ in range(X.BATCH)]; assert all(v < 128 for p, _ in batch for v in p)
            dg.add(batch); b.step(batch); g += 1; n += 1
            if n % REVIEW_PROBE == 0:
                pr = b.probe(g, phase="review", cycle=c, updates_in_phase=n); trace.append(pr)
        cycles.append(dict(cycle=c, updates_at_end_of_gap=pg["updates"], probe_at_end_of_gap=dict(first_family_pct=pg["first_family_pct"], unrelated_held_pct=pg["unrelated_held_pct"]),
                           review_updates=n, review_reached_goal=pr["first_family_pct"] >= goal,
                           probe_at_end_of_review=dict(first_family_pct=pr["first_family_pct"], unrelated_held_pct=pr["unrelated_held_pct"]), updates_at_end_of_review=g))
        print(tag, "cycle", c, cycles[-1], flush=True); write()
    rows.append(b.e43_row(g, dg))
    wp = out / f"{tag}_end.pt"; R.save_weights(b.m, wp)
    write(dict(updates=g, data_digest=dg.value(), weights=wp.name, weights_sha256=X16.sha_file(wp), **b.finish()))
    print(tag, "complete", flush=True)


# ---------------------------------------------------------------- checks 8 to 11
def check(n, seed, cmd):
    d = OUTB / "checks"; d.mkdir(parents=True, exist_ok=True); res = {}
    if n == "8":                                            # B1's runner is the addendum's at k = 2
        for rule in ("S", "B"):
            mine = json.loads((OUTB / "check8" / f"e51_B1_share_k2addendum_{rule}_{seed}.json").read_text())
            ref = json.loads((A2.OUT / f"e43_unrelated-rehearse_{rule}_{seed}.json").read_text())
            a = [r for r in mine["rows"] if r["updates"] == 64][0]; bb = [r for r in ref["rows"] if r["updates"] == 64][0]
            res[rule] = dict(first_family_counts_row64_equal=a["score1_first_family_all_pairs_k0"] == bb["score1_first_family_all_pairs_k0"]
                             and a["score2_first_family_heldout_k12"] == bb["score2_first_family_heldout_k12"],
                             tensors_sha256_4096_here=mine["complete"]["tensors_sha256"], tensors_sha256_4096_addendum=ref["complete"]["tensors_sha256"],
                             data_digest_equal=mine["complete"]["data_digest"] == ref["complete"]["data_digest"])
            res[rule]["ok"] = res[rule]["first_family_counts_row64_equal"] and mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"]
    elif n == "9":                                          # B3 at 0.001 is E43's run
        for rule in ("S", "B"):
            mine = json.loads((OUTB / f"e51_B3_steps_0.001_{rule}_{seed}.json").read_text()); ref = json.loads((U.OUT43 / f"e43_unrelated_{rule}_{seed}.json").read_text())
            eq = {}
            for g in (64, 4096):
                a = [r for r in mine["rows"] if r["updates"] == g][0]; bb = [r for r in ref["rows"] if r["updates"] == g][0]
                eq[g] = a["score1_first_family_all_pairs_k0"] == bb["score1_first_family_all_pairs_k0"] and a["score2_first_family_heldout_k12"] == bb["score2_first_family_heldout_k12"]
            res[rule] = dict(first_family_counts_equal_at_rows=eq, tensors_sha256_4096_here=mine["complete"]["tensors_sha256"], tensors_sha256_4096_e43=ref["complete"]["tensors_sha256"],
                             learning_rate_read_back=[g["lr"] for g in mine["complete"]["optimizer_groups_read_back"]])
            res[rule]["ok"] = all(eq.values()) and mine["complete"]["tensors_sha256"] == ref["complete"]["tensors_sha256"]
    elif n == "10":                                         # the probe at update 0 against the same 400 items in E43's score 1
        T.set_num_threads(1)
        for rule in ("S", "B"):
            b = Bench(rule, seed); p = b.probe(0)
            exact = 0
            for name in ("add89", "sub89"):
                items = b.s1[name][0]; outs = E.respond(b.m, [q for q, _ in items], None, cap=X16.CAP)      # score 1's own generation, all 7,921 pairs
                exact += sum(outs[i] == items[i][1] for i in b.pidx[name])
            res[rule] = dict(probe_exact=p["first_family_exact"], same_items_in_score1_exact=exact, n=400, all_are_reduction_pairs=all(b.s1[n_][1][i] for n_ in b.pidx for i in b.pidx[n_]),
                             distinct=all(len(set(v)) == 200 for v in b.pidx.values()), ok=p["first_family_exact"] == exact)
    elif n == "11":                                         # cost, from Part A's measured CPU-seconds an update (arm open, the unrelated task)
        per = {}
        for rule in ("S", "B"):
            c = json.loads((X.OUT / "e51" / f"e51_unrelated_open_{rule}_{seed}.json").read_text())["complete"]; per[rule] = c["cpu_updates"] / c["updates"]
        e43row, probe = 22.0, 1.5                           # CPU-seconds of a full E43 row (E43's receipts, row 0) and of a probe (an estimate; measured in the runs)
        runs = {}
        for rule in ("S", "B"):
            for k in (4, 8, 16, 32):
                runs[f"B1 k={k} {rule}"] = 4096 * per[rule] + 6 * e43row + 29 * probe
            runs[f"B2 {rule}"] = CYCLES * (GAP + REVIEW_MAX) * per[rule] + 2 * e43row + CYCLES * (4 + 64) * probe
            for key, (lr, dose) in STEPS.items():
                runs[f"B3 {key} {rule}"] = dose * per[rule] + 4 * e43row + (16 + (dose - 256) // 64 + 1) * probe
        res = dict(cpu_seconds_an_update=per, forecast_cpu_seconds=runs, runs=len(runs), forecast_cpu_hours=sum(runs.values()) / 3600,
                   wall_hours_at_three_processes=sum(runs.values()) / 3600 / 3, note="B2 at its upper limit (every review 1,024 updates)", ok=True)
    names = {"8": "b1_is_addendum", "9": "b3_is_e43", "10": "probe", "11": "cost"}
    p = d / f"e51_check_{n}_{names[n]}_{seed}.json"; R.refuse_existing(p)
    ok = res.pop("ok") if n == "11" else all(v["ok"] for v in res.values())
    p.write_text(json.dumps(dict(experiment="E51", kind=f"check {n}", seed=seed, ok=ok, result=res, provenance=provenance(cmd)), indent=1))
    print("check", n, json.dumps(res)[:1500], "ok" if ok else "FAILED", flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    outdir = a[a.index("--out") + 1] if "--out" in a else str(OUTB)
    if a[0] == "share":
        assert a[2] in C.ARMS; share(a[1], a[2], int(a[3]), outdir, cmd)
    elif a[0] == "intervals":
        assert a[1] in C.ARMS; intervals(a[1], int(a[2]), outdir, cmd)
    elif a[0] == "steps":
        assert a[1] in STEPS and a[2] in C.ARMS; steps(a[1], a[2], int(a[3]), outdir, cmd)
    elif a[0] == "check":
        check(a[1], int(a[2]), cmd)
    else:
        raise SystemExit(__doc__)
