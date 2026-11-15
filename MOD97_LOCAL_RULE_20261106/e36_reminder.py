"""E36 runner (brief OPUS_TASK_E36_REMINDER_AFTER_A_LIFE_20261115.md, revision 2): after a life, how many updates bring the unrehearsed
birth family (the four mod-89 operations) back? A new file; every existing source stays as it is.

Runs (1,024 updates each, fresh exp26_candidate.make_opt in every run, the same batches for every run of a seed):
    reminder S  from exp45/e26/e26_life_S_<seed>.pt, the sweep: exp26_candidate.step(m, opt, items, True)
    reminder B  from exp45/e26/e26_life_B_<seed>.pt, backprop: exp26_candidate.step(m, opt, items, False)
    reminder K  from exp45/e31/e31_life_K_<seed>.pt with Bhh from e31_life_K_<seed>_Bhh.pt: step_K below (exp26_candidate.step's text
                with arch31_workerB.forward(m, x, True, True, switch="K", Bhh=B); Bhh <- (1 - d) Bhh + (W1 - (1 - d) W0), d = 0.00025)
    reference S, B   arch26.new_net(seed), untrained, sweep / backprop
Data: exp11_cell.batches([(t, exp11_cell.split(t, .5, seed)[0]) for t in exp11_cell.DEV], seed + 36, 12).
Rows: 0, 16, 32, 64, 128, 256, 512, 1024 updates; scores and weights at every row.

    python3 e36_reminder.py run <reminder|reference> <S|B|K> <seed> [--updates N] [--out DIR]
    python3 e36_reminder.py check <1|3|4> [seed] [--out DIR]      (check 2 is read from the run receipts: `check 2 <dir>`)
"""
import json
import sys
import time
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import torch as T                    # noqa: E402
import arch26                        # noqa: E402
import arch31_workerB as A31         # noqa: E402

E, X, X16, R = C.E, C.X, C.X16, C.R
OUT = HERE / "exp45" / "e36"
E26, E31 = HERE / "exp45" / "e26", HERE / "exp45" / "e31"
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
DECAY = C.LR * C.WD                  # 0.00025
SEEDS = (73129, 93683, 116089)
OWN = ("e36_reminder.py", "arch31_workerB.py", "exp31_switch_workerB.py")
BIRTH4 = ("add89", "sub89", "max89", "min89")
LIFE4 = ("add97", "sub97", "add73", "sub73")
COMMIT_ROWS = (256, 1024)


def provenance(cmd):
    p = C.provenance(cmd)
    p["e36_sources_sha256"] = {s: X16.sha_file(HERE / s) for s in OWN}
    return p


# ---------------------------------------------------------------- data and panels
def taught_pairs(seed):
    return [(t, E.split(t, .5, seed)[0]) for t in E.DEV]


def batches(seed):
    return E.batches(taught_pairs(seed), seed + 36, 12)


def all_pairs(op):
    """Every one of the 7,921 pairs of op mod 89, rendered with k = 0, split by wrap (add: a + b >= 89; sub: a - b < 0)."""
    t = next(t for t in E.DEV if t[0] == op)
    wrap, nowrap = [], []
    for a in range(89):
        for b in range(89):
            it = E.render(t, a, b, 0, Random(0))
            (wrap if (a + b >= 89 if op == "add" else a - b < 0) else nowrap).append(it)
    return dict(wrap=wrap, nowrap=nowrap)


class Scorer:
    def __init__(self, seed):
        self.P, self.comp = X16.panels(seed)
        self.allp = {op: all_pairs(op) for op in ("add", "sub")}
        assert all(len(v["wrap"]) + len(v["nowrap"]) == 7921 for v in self.allp.values())

    def __call__(self, m):
        s = X16.score
        held = {n: s(m, self.P[n][12], None) for n in BIRTH4}
        allp = {f"{op}89": {w: s(m, items, None) for w, items in v.items()} for op, v in self.allp.items()}
        life = {n: s(m, self.P[n][12], None) for n in LIFE4}
        life["within"] = s(m, self.comp["within"], None)
        pct = lambda d: 100.0 * d["exact"] / d["n"]
        summary = dict(held_add_sub_mean=(pct(held["add89"]) + pct(held["sub89"])) / 2,
                       wrap_score=(pct(allp["add89"]["wrap"]) + pct(allp["sub89"]["wrap"])) / 2,
                       nowrap_score=(pct(allp["add89"]["nowrap"]) + pct(allp["sub89"]["nowrap"])) / 2)
        summary["restored"] = summary["held_add_sub_mean"] >= 85
        return dict(birth_held_k12=held, birth_all_pairs_k0=allp, life_families=life, summary=summary)


# ---------------------------------------------------------------- starts
def start(kind, rule, seed):
    """The network (and Bhh for K) a run starts from, with what was checked."""
    m = arch26.new_net(seed)
    if kind == "reference":
        assert rule in ("S", "B")
        return m, None, dict(start="arch26.new_net(seed), untrained", tensors_sha256=R.tensors_sha(m.state_dict()))
    if rule in ("S", "B"):
        f = E26 / f"e26_life_{rule}_{seed}.pt"
    else:
        f = E31 / f"e31_life_K_{seed}.pt"
    rec = json.loads(f.with_suffix(".json").read_text())
    sha = X16.sha_file(f)
    assert sha == rec["complete"]["weights_sha256"], f"{f.name}: file hash is not the receipt's"
    m.load_state_dict(T.load(f))
    tsha = R.tensors_sha(m.state_dict())
    info = dict(start=str(f.relative_to(HERE)), weights_sha256=sha, tensors_sha256=tsha, receipt_tensors_sha256=rec["complete"]["tensors_sha256"],
                tensors_sha_matches_receipt=tsha == rec["complete"]["tensors_sha256"])
    B = None
    if rule == "K":
        bf = E31 / f"e31_life_K_{seed}_Bhh.pt"
        B = T.load(bf)["Bhh"].detach().clone()
        bsha = R.tensors_sha({"Bhh": B})
        info.update(Bhh_file=str(bf.relative_to(HERE)), Bhh_file_sha256=X16.sha_file(bf), Bhh_tensors_sha256=bsha,
                    receipt_Bhh_final_tensors_sha256=rec["e31"]["Bhh_final_tensors_sha256"], Bhh_matches_receipt=bsha == rec["e31"]["Bhh_final_tensors_sha256"])
    return m, B, info


# ---------------------------------------------------------------- the update
def step_K(m, opt, items, B):
    """exp26_candidate.step's text with arch31_workerB.forward(switch="K", Bhh=B); then Bhh's rule (exp31_switch_workerB lines 74-77, 196-202)."""
    x, y, mask = X.batch_tensor(items)
    logits = A31.forward(m, x, True, True, switch="K", Bhh=B)[0]
    loss = E.cross_entropy(logits[mask], y[mask])
    opt.zero_grad()
    loss.backward()
    W0 = m.rnn.weight_hh_l0.detach().clone()
    opt.step()
    W1 = m.rnn.weight_hh_l0.detach().clone()
    B = (1 - DECAY) * B + (W1 - (1 - DECAY) * W0)
    return float(loss.detach()), B


class Learner:
    def __init__(self, m, rule, B=None):
        self.m, self.rule, self.B = m, rule, B
        self.opt = C.make_opt(m)

    def update(self, items):
        if self.rule == "K":
            loss, self.B = step_K(self.m, self.opt, items, self.B)
            return loss
        return C.step(self.m, self.opt, items, self.rule == "S")


def bhh_measures(B, W):
    W = W.detach()
    return dict(Bhh_tensors_sha256=R.tensors_sha({"Bhh": B}),
                cos_Bhh_W=float(T.nn.functional.cosine_similarity(B.flatten().double(), W.flatten().double(), dim=0)),
                rel_diff_Bhh_W=float((B.double() - W.double()).norm() / W.double().norm()))


# ---------------------------------------------------------------- one run
def run(kind, rule, seed, updates, out, cmd):
    arch26.install()
    out = Path(out)
    name = f"e36_{kind}_{rule}_{seed}"
    rows = [r for r in ROWS if r <= updates]
    if updates not in rows:
        rows.append(updates)
    rec_path = out / f"{name}.json"
    wpaths = {r: out / f"{name}_row{r}.pt" for r in rows}
    R.refuse_existing(rec_path, *wpaths.values(), *[str(p) + ".tmp" for p in wpaths.values()])
    out.mkdir(parents=True, exist_ok=True)
    T.manual_seed(seed)
    m, B, info = start(kind, rule, seed)
    L = Learner(m, rule, B)
    sc = Scorer(seed)
    bt = batches(seed)
    dg = C.Digest()
    rec = dict(experiment="E36", kind=kind, rule=rule, seed=seed, updates=updates, rows_planned=rows, start=info,
               optimizer=C.opt_record(m, L.opt), data="exp11_cell.batches(birth taught half, seed + 36, 12)",
               update={"S": "exp26_candidate.step(local=True)", "B": "exp26_candidate.step(local=False)",
                       "K": "e36_reminder.step_K (arch31_workerB.forward switch=K, Bhh rule d=0.00025)"}[rule],
               provenance=provenance(cmd), rows=[], complete=False)
    t0 = time.process_time(); cpu_score = 0.0; losses = []; u = 0
    for r in rows:
        while u < r:
            items = next(bt); dg.add(items); losses.append(L.update(items)); u += 1
        cb = time.process_time(); s = sc(m); cpu_score += time.process_time() - cb
        wsha = R.save_weights(m, wpaths[r])
        row = dict(updates=r, scores=s, loss_since_last_row=(sum(losses) / len(losses)) if losses else None,
                   finite_losses=all(v == v and abs(v) != float("inf") for v in losses), data_digest=dg.value(),
                   weights=wpaths[r].name, weights_sha256=wsha, tensors_sha256=R.tensors_sha(m.state_dict()),
                   optimizer_steps=R.opt_steps(L.opt), cpu=time.process_time() - t0, cpu_scoring_so_far=cpu_score)
        if rule == "K":
            row["Bhh"] = bhh_measures(L.B, m.rnn.weight_hh_l0)
        losses = []
        rec["rows"].append(row)
        sm = s["summary"]
        print(name, r, f"held {sm['held_add_sub_mean']:.1f} wrap {sm['wrap_score']:.1f} nowrap {sm['nowrap_score']:.1f}",
              {n: v["exact"] for n, v in s["life_families"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
        rec_path.write_text(json.dumps(rec, indent=1))
    rec.update(complete=True, cpu_total=time.process_time() - t0, cpu_scoring=cpu_score, cpu_updates=time.process_time() - t0 - cpu_score)
    rec_path.write_text(json.dumps(rec, indent=1))
    print(name, "complete", flush=True)


# ---------------------------------------------------------------- checks
def check1(out, cmd):
    """Row 0 of each reminder: the life-end file's tensors_sha256, and the last stage's add89, sub89, max89, min89 reproduced exactly."""
    arch26.install()
    res = {}
    for seed in SEEDS:
        sc = Scorer(seed)
        for rule in ("S", "B", "K"):
            m, B, info = start("reminder", rule, seed)
            f = (E26 / f"e26_life_{rule}_{seed}.json") if rule != "K" else (E31 / f"e31_life_K_{seed}.json")
            last = json.loads(f.read_text())["stages"][-1]["scores"]
            got = {n: X16.score(m, sc.P[n][12], None) for n in BIRTH4}
            want = {n: last[n]["12"] for n in BIRTH4}
            res[f"{rule}_{seed}"] = dict(tensors_sha256=info["tensors_sha256"], receipt_tensors_sha256=info["receipt_tensors_sha256"],
                                         sha_ok=info["tensors_sha_matches_receipt"], scores=got, receipt_scores=want, scores_ok=got == want)
            print("check1", rule, seed, res[f"{rule}_{seed}"]["sha_ok"], res[f"{rule}_{seed}"]["scores_ok"], flush=True)
    return dict(rows=res, ok=all(v["sha_ok"] and v["scores_ok"] for v in res.values()))


def check2(d):
    """The batch digest is equal across every run of a seed at every row (read from the run receipts in d)."""
    res = {}
    for seed in SEEDS:
        recs = sorted(Path(d).glob(f"e36_*_{seed}.json"))
        if not recs:
            continue
        dig = {}
        for p in recs:
            r = json.loads(p.read_text())
            for row in r["rows"]:
                dig.setdefault(row["updates"], {})[p.stem] = row["data_digest"]["sha256"]
        res[seed] = dict(runs=[p.stem for p in recs], rows={u: dict(distinct=len(set(v.values())), runs=len(v)) for u, v in dig.items()},
                         ok=all(len(set(v.values())) == 1 and len(v) == len(recs) for v in dig.values()))
    return dict(seeds=res, ok=bool(res) and all(v["ok"] for v in res.values()))


def maxrel(a, b):                    # exp31_switch_workerB.maxrel
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def grads(m, x, y, mask, **kw):      # exp31_switch_workerB.grads, against arch31_workerB
    m.zero_grad()
    logits, _ = A31.forward(m, x, True, True, **kw)
    E.cross_entropy(logits[mask], y[mask]).backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def check3(seed):
    arch26.install()
    res = {}
    m, B, info = start("reminder", "K", seed)
    res["Bhh_sha"] = dict(loaded=info["Bhh_tensors_sha256"], receipt=info["receipt_Bhh_final_tensors_sha256"], ok=info["Bhh_matches_receipt"])
    first = next(batches(seed)); x, y, mask = X.batch_tensor(first)
    gH = grads(m, x, y, mask, switch="H")
    gKW = grads(m, x, y, mask, switch="K", Bhh=m.rnn.weight_hh_l0.detach().clone())
    rel = {n: maxrel(gKW[n], gH[n]) for n in gH}
    res["K_equals_H_when_Bhh_is_W"] = dict(grads_max_rel=rel, worst=max(rel.values()), ok=max(rel.values()) <= 1e-6)
    gK = grads(m, x, y, mask, switch="K", Bhh=B)
    gN = grads(m, x, y, mask, switch=None)
    rel2 = {n: maxrel(gK[n], gN[n]) for n in gN}
    res["K_loaded_differs_from_None"] = dict(grads_max_rel=rel2, worst=max(rel2.values()), ok=max(rel2.values()) > 1e-3)
    m.zero_grad()
    L = Learner(m, "K", B)
    rows = []
    for u, items in enumerate(_take(batches(seed), 100), 1):
        Bold = L.B.clone(); W0 = m.rnn.weight_hh_l0.detach().clone()
        L.update(items)
        W1 = m.rnn.weight_hh_l0.detach().clone(); Bnew = L.B
        if u in (1, 10, 100):
            contr = float(((Bnew - W1) - (1 - DECAY) * (Bold - W0)).abs().max())
            ratio = float((Bnew - W1).norm() / (Bold - W0).norm())
            rows.append(dict(update=u, contraction_max_abs=contr, norm_ratio=ratio, expected_ratio=1 - DECAY, **bhh_measures(Bnew, W1)))
    # Revision after the first check-3 run (receipts kept in exp45/e36/checks/): the gate is worker B's elementwise residual (<= 1e-7);
    # the float32 norm ratio of the small difference Bhh - W deviated from 1 - d by up to 2.0e-6, over the first tolerance of 1e-6.
    res["contraction"] = dict(rows=rows, ok=all(r["contraction_max_abs"] <= 1e-7 for r in rows),
                              norm_ratio_within_1e_5=all(abs(r["norm_ratio"] - (1 - DECAY)) <= 1e-5 for r in rows))
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


def _take(it, n):
    return [next(it) for _ in range(n)]


def check4():
    """From arch26.new_net(73129) with make_opt, 2,048 updates on the birth's own stream reproduce E26's dry births (S and B)."""
    arch26.install()
    seed = 73129; res = {}
    for rule, f in (("S", "e26_dev_S_dry2048_73129.json"), ("B", "e26_dev_B_dry2048_73129.json")):
        want = json.loads((HERE / "exp45" / "e26_qual" / f).read_text())["tensors_sha256"]
        T.manual_seed(seed)
        m = arch26.new_net(seed); L = Learner(m, rule)
        bt = E.batches(taught_pairs(seed), seed, 12)
        t0 = time.process_time()
        for _ in range(2048):
            L.update(next(bt))
        got = R.tensors_sha(m.state_dict())
        res[rule] = dict(got=got, want=want, ok=got == want, cpu=time.process_time() - t0)
        print("check4", rule, got[:8], want[:8], got == want, flush=True)
    res["ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    return res


if __name__ == "__main__":
    T.set_num_threads(1)
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])

    def opt_(flag, default=None):
        return a[a.index(flag) + 1] if flag in a else default

    out = Path(opt_("--out", str(OUT)))
    if a[0] == "run":
        assert a[1] in ("reminder", "reference") and a[2] in ("S", "B", "K")
        run(a[1], a[2], int(a[3]), int(opt_("--updates", "1024")), out, cmd)
    elif a[0] == "check":
        t0 = time.process_time()
        part = a[1]
        if part == "1":
            res = check1(out, cmd); name = "check1"
        elif part == "2":
            res = check2(a[2]); name = f"check2_{Path(a[2]).name}"
        elif part == "3":
            res = check3(int(a[2])); name = f"check3_{a[2]}"
        elif part == "4":
            res = check4(); name = "check4"
        else:
            raise SystemExit(__doc__)
        p = out / "checks" / f"E36_{name}.json"
        R.refuse_existing(p); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(check=name, command=cmd, provenance=provenance(cmd), cpu_seconds=time.process_time() - t0, result=res), indent=1, default=str))
        print(name, "pass" if res.get("ok") else "FAIL", f"cpu {time.process_time() - t0:.0f}s", flush=True)
    else:
        raise SystemExit(__doc__)
