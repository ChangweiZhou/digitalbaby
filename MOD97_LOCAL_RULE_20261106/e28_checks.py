"""E28 stage 1 checks (brief OPUS_TASK_E28_GRU_EVERY_BYTE_20261113.md revision 2, items 1 to 6). Writes exp45/e28_qual/check_<item>.json.

    python3 e28_checks.py host        item 1: the host-check births (exp45/e28/host_check) against E26's receipts and weights, seed 73129
    python3 e28_checks.py refusal     item 2: exp26_candidate.admit_e26 on E28 birth files of every targets value
    python3 e28_checks.py mask        item 3: the `all` mask on the first birth batch; one answer-target step against exp26_candidate.step
    python3 e28_checks.py balanced    item 4: the balanced loss against an independently written weighted form (rel. 1e-5 of each tensor's max)
    python3 e28_checks.py onestep     item 5: one real step per new arm against AdamW given an independently computed gradient
    python3 e28_checks.py dry         item 6: the dry births' receipts (exp45/e28_qual/dry), every record present; CPU per update

exp26_candidate.step is called here only as the reference item 3 names; the runner (exp28_everybyte.py) never calls it. The independent
gradient is e25_alignment.forward_rnn (the coordinator's written-out GRU, not arch26) with torch's unreduced cross entropy.
"""
import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp28_everybyte as W          # noqa: E402  (imports exp26_candidate, exp20_reference: environment set)

import torch as T                    # noqa: E402
from torch.nn.functional import cross_entropy  # noqa: E402

C, R, X16, E, X, arch26 = W.C, W.R, W.X16, W.E, W.X, W.arch26
import e25_alignment as REF          # noqa: E402

SEED = 73129
QUAL = X.OUT / "e28_qual"
HOST = X.OUT / "e28" / "host_check"
DRY = QUAL / "dry"
TOL = 1e-5
NEW = ("S-all", "B-all", "S-bal", "B-bal")


def first_batch(seed=SEED):
    return next(E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"]))


def tcmp(sa, sb):
    return {k: dict(equal=bool(T.equal(sa[k], sb[k])), max_abs_diff=float((sa[k].double() - sb[k].double()).abs().max())) for k in sa}


def relcmp(ga, gb):
    """max |a - b| / max |a|, per tensor (a: the independent form)."""
    return {k: float((ga[k].double() - gb[k].double()).abs().max()) / max(float(ga[k].double().abs().max()), 1e-30) for k in ga}


# ---------------------------------------------------------------- item 1
def item_host():
    out = {}; ok = True
    for a in ("S", "B"):
        e26r = json.loads((W.E26_DIR / f"e26_dev_{a}_{SEED}.json").read_text())
        e28r = json.loads((HOST / f"e28_dev_{a}-ans_{SEED}.json").read_text())
        sd26 = T.load(W.E26_DIR / f"e26_dev_{a}_{SEED}.pt"); sd28 = T.load(HOST / f"e28_dev_{a}-ans_{SEED}.pt")
        tc = tcmp(sd26, sd28)
        rows = []
        for r26, r28 in zip(e26r["curve"], e28r["curve"]):
            rows.append(dict(step=(r26["step"], r28["step"]), sealed_equal=r26["sealed"] == json.loads(json.dumps(r28["sealed"])),
                             data_digest_equal=r26["data_digest"] == r28["data_digest"], data_digest=r28["data_digest"],
                             loss_e26=r26["loss"], loss_e28=r28["loss"], loss_equal=r26["loss"] == r28["loss"],
                             sealed_differences=[(n, k, f, r26["sealed"][n][k][f], r28["sealed"][n][k][f]) for n in r26["sealed"] for k in r26["sealed"][n]
                                                 for f in ("n", "exact", "final") if r26["sealed"][n][k][f] != r28["sealed"][n][k][f]]))
        sil = e26r["curve"][-1].get("silenced_k12") == e28r["curve"][-1].get("silenced_k12")
        res = dict(e26_receipt=f"exp45/e26/e26_dev_{a}_{SEED}.json", e28_receipt=str((HOST / f"e28_dev_{a}-ans_{SEED}.json").relative_to(HERE)),
                   tensors_sha256=dict(e26=e26r["tensors_sha256"], e28=e28r["tensors_sha256"], equal=e26r["tensors_sha256"] == e28r["tensors_sha256"]),
                   every_tensor_equal=all(v["equal"] for v in tc.values()), per_tensor=tc,
                   rows=len(rows) == 6 and rows, all_rows_sealed_equal=len(rows) == 6 and all(r["sealed_equal"] for r in rows),
                   all_rows_data_digest_equal=len(rows) == 6 and all(r["data_digest_equal"] for r in rows), silenced_k12_equal=sil,
                   final_data_digest_equal=e26r["data_digest"] == e28r["data_digest"], optimizer_steps=(e26r["optimizer_steps"], e28r["optimizer_steps"]),
                   cpu=dict(e26_cpu_updates=e26r["cpu_updates"], e28_cpu_updates=e28r["cpu_updates"], e28_cpu_per_update=e28r["cpu_per_update"]),
                   e26_host=e26r["provenance"]["host"], e28_host=e28r["provenance"]["host"])
        res["pass"] = (res["tensors_sha256"]["equal"] and res["every_tensor_equal"] and res["all_rows_sealed_equal"] and res["all_rows_data_digest_equal"]
                       and res["final_data_digest_equal"])
        ok = ok and res["pass"]
        out[f"{a}-ans"] = res
    return dict(arms=out, **{"pass": ok})


# ---------------------------------------------------------------- item 2
def item_refusal():
    files = {"answer": [HOST / f"e28_dev_{a}-ans_{SEED}.pt" for a in ("S", "B")],
             "all": [DRY / f"e28_dev_{a}-all_dry2048_{SEED}.pt" for a in ("S", "B")],
             "balanced": [DRY / f"e28_dev_{a}-bal_dry2048_{SEED}.pt" for a in ("S", "B")]}
    out = {}; ok = True
    with tempfile.TemporaryDirectory() as td:
        for targets, fs in files.items():
            for f in fs:
                arm = f.name.split("_")[2][0]          # S or B: the life arm the file might be offered to
                rec = json.loads(f.with_suffix(".json").read_text())
                assert rec["targets"] == targets and rec["experiment"] == "E28"
                for how, p in (("as written", f), ("renamed to E26's file name", None)):
                    if p is None:
                        p = Path(td) / f"{targets}_{arm}" / f"e26_dev_{arm}{E.DEVTAG}_{SEED}.pt"; p.parent.mkdir()
                        shutil.copy(f, p); shutil.copy(f.with_suffix(".json"), p.with_suffix(".json"))
                    try:
                        C.admit_e26(p, arm, SEED); r = "ADMITTED"
                    except AssertionError as e:
                        r = f"refused: AssertionError: {str(e).replace(td, '<tmp>')}"
                    ok = ok and r.startswith("refused")
                    out[f"{targets} | {f.name} | {how} | offered to an arm {arm} life"] = r
    return dict(cases=out, note="admit_e26 refuses at its first receipt assertion (experiment == 'E26'); it does not read `targets`", **{"pass": ok})


# ---------------------------------------------------------------- item 3
def item_mask():
    items = first_batch()
    x, y, amask, real, fine = W.masks(items)
    per = []
    for i, (p, t) in enumerate(items):
        r = p + t; n = len(r)
        per.append(dict(real=int(real[i].sum()), bytes_of_prompt_and_answer=n, first_n_positions_all_real=bool(real[i, :n].all()),
                        padding_positions=int(y.shape[1] - n), padding_positions_in_mask=int(real[i, n:].sum()),
                        targets_equal_bytes=y[i, :n].tolist() == r, padding_is_bos=bool((y[i, n:] == E.BOS).all())))
    m_ok = all(d["real"] == d["bytes_of_prompt_and_answer"] and d["first_n_positions_all_real"] and d["padding_positions_in_mask"] == 0
               and d["targets_equal_bytes"] and d["padding_is_bos"] for d in per)
    counts = {n: int((fine == i).sum()) for i, n in enumerate(W.FINE)}
    coarse = {r_: sum(c for n, c in counts.items() if W.FINE_TO_ROLE[n] == r_) for r_ in W.ROLES}
    part = dict(fine_counts=counts, role_counts=coarse, sum_of_roles=sum(coarse.values()), real_positions=int(real.sum()),
                roles_disjoint="each position carries one label (a single integer tensor)", partition=sum(coarse.values()) == int(real.sum()),
                answer_role_equals_answer_mask=bool(T.equal((fine == W.FINE.index("answer_digit")) | (fine == W.FINE.index("eos")), amask)),
                example_item=dict(prompt=bytes(items[0][0]).decode("latin1"), target=list(items[0][1]),
                                  roles=[W.FINE[v] for v in fine[0].tolist() if v >= 0]))
    arch26.install()
    steps = {}
    for a, local in (("S", True), ("B", False)):
        m1 = E.new_net(SEED); o1 = C.make_opt(m1); m2 = copy.deepcopy(m1); o2 = C.make_opt(m2)
        assert all(T.equal(u, v) for u, v in zip(m1.state_dict().values(), m2.state_dict().values()))
        l1 = W.step(m1, o1, items, local, "answer", W.new_roles()); l2 = C.step(m2, o2, items, local)
        tc = tcmp(m2.state_dict(), m1.state_dict())
        st = all(T.equal(o1.state[p1][k], o2.state[p2][k]) for p1, p2 in zip(m1.parameters(), m2.parameters()) for k in o1.state[p1])
        steps[a] = dict(loss_e28=l1, loss_e26_step=l2, loss_equal=l1 == l2, weights_bit_equal=all(v["equal"] for v in tc.values()),
                        max_abs_diff=max(v["max_abs_diff"] for v in tc.values()), optimizer_state_bit_equal=st)
    s_ok = all(v["weights_bit_equal"] and v["loss_equal"] and v["optimizer_state_bit_equal"] for v in steps.values())
    return dict(batch=f"first birth batch of seed {SEED}", width=int(x.shape[1]), items=per, all_mask_ok=m_ok, roles_partition=part,
                answer_step_vs_exp26_candidate_step=steps, **{"pass": m_ok and part["partition"] and part["answer_role_equals_answer_mask"] and s_ok})


# ---------------------------------------------------------------- items 4 and 5: the independent forms
def independent_grads(m, x, y, amask, real, local, targets):
    """e25_alignment's written-out forward; for `all` its own grads(); for `balanced`, unreduced CE weighted 0.5/n_a and 0.5/n_o, summed."""
    if targets == "all":
        l, gp, _, _ = REF.grads(m, x, y, real, local)
        return l, gp
    m.zero_grad()
    logits = REF.forward_rnn(m, x, local)[0]
    ce = cross_entropy(logits[real], y[real], reduction="none")
    a = amask[real]; na = int(a.sum()); no = int((~a).sum())
    w = T.where(a, T.full_like(ce, 0.5 / na), T.full_like(ce, 0.5 / no))
    loss = (ce * w).sum(); loss.backward()
    return float(loss.detach()), {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}


def runner_grads(m, x, y, amask, real, local, targets):
    m.zero_grad()
    loss = W.loss_of(arch26.forward(m, x, local, True)[0], y, amask, real, targets); loss.backward()
    return float(loss.detach()), {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}


def states():
    arch26.install()
    out = [("untrained G1, seed 73129", E.new_net(SEED))]
    for a in ("S", "B"):
        m = E.new_net(SEED); f = W.E26_DIR / f"e26_dev_{a}_{SEED}.pt"; m.load_state_dict(T.load(f))
        out.append((f"E26 born {a} seed 73129 ({f.name}, sha {X16.sha_file(f)[:12]})", m))
    return out


def item_balanced():
    items = first_batch(); x, y, amask, real, _ = W.masks(items)
    out = {}; ok = True
    for name, m in states():
        for path, local in (("sweep", True), ("backprop", False)):
            li, gi = independent_grads(m, x, y, amask, real, local, "balanced")
            lr_, gr = runner_grads(m, x, y, amask, real, local, "balanced")
            rel = relcmp(gi, gr)
            c = dict(loss_independent=li, loss_runner=lr_, loss_rel_diff=abs(li - lr_) / abs(li), grad_rel_diff_per_tensor=rel,
                     grad_rel_diff_max=max(rel.values()), bitwise_equal_grads=all(T.equal(gi[k], gr[k]) for k in gi))
            c["pass"] = c["loss_rel_diff"] <= TOL and c["grad_rel_diff_max"] <= TOL
            ok = ok and c["pass"]
            out[f"{name} | {path}"] = c
    return dict(batch=f"first birth batch of seed {SEED}", tolerance=f"relative {TOL} of each tensor's largest entry (loss: of the loss)", cases=out,
                **{"pass": ok})


def item_onestep():
    items = first_batch(); x, y, amask, real, _ = W.masks(items)
    arch26.install()
    out = {}; ok = True
    for arm in NEW:
        local, targets = W.ARMS[arm]
        m1 = E.new_net(SEED); o1 = C.make_opt(m1); m2 = copy.deepcopy(m1); o2 = C.make_opt(m2)
        W.step(m1, o1, items, local, targets, W.new_roles())                       # the runner's real step
        _, gp = independent_grads(m2, x, y, amask, real, local, targets)
        o2.zero_grad()
        for n, p in m2.named_parameters():
            p.grad = gp[n].clone()
        o2.step()
        sa, sb = m2.state_dict(), m1.state_dict()
        tc = tcmp(sa, sb)
        delta = {k: float((sb[k].double() - E.new_net(SEED).state_dict()[k].double()).abs().max()) for k in sb}
        rel = relcmp(sa, sb)
        c = dict(targets=targets, local=local, bit_equal=all(v["equal"] for v in tc.values()), max_abs_diff=max(v["max_abs_diff"] for v in tc.values()),
                 weights_rel_diff_per_tensor=rel, weights_rel_diff_max=max(rel.values()),
                 tensors_differing=sorted(k for k, v in tc.items() if not v["equal"]),
                 largest_step_taken=max(delta.values()))
        c["criterion"] = "bit for bit" if targets == "all" else f"relative {TOL} of each tensor's largest entry"
        c["pass"] = c["bit_equal"] if targets == "all" else c["weights_rel_diff_max"] <= TOL
        ok = ok and c["pass"]
        out[arm] = c
    return dict(start="new_net(73129), fresh optimizer, the seed's first birth batch", arms=out, **{"pass": ok})


# ---------------------------------------------------------------- item 6
def item_dry():
    out = {}; ok = True
    need_top = ("experiment", "targets", "finite", "role_curve", "data_digest", "row_weights", "weights_sha256", "tensors_sha256", "optimizer_read_back",
                "optimizer_steps", "credit_census_first_update", "provenance", "cpu_per_update")
    for arm in NEW:
        f = DRY / f"e28_dev_{arm}_dry2048_{SEED}.json"
        r = json.loads(f.read_text())
        pv = r["provenance"]
        present = {k: k in r for k in need_top}
        rows_ok = all(all(k in row for k in ("data_digest", "weights_sha256", "sealed", "finite_cumulative", "hard_mean_add89_sub89_k12")) for row in r["curve"])
        role_ok = len(r["role_curve"]) == 2048 // 512 and all(set(b["by_role"]) == set(W.ROLES) and all(v["mean"] is not None for v in b["by_role"].values())
                                                            for b in r["role_curve"])
        wf = DRY / r["weights"]
        c = dict(complete=r["complete"], experiment=r["experiment"], targets=r["targets"], records_present=present, rows_have_records=rows_ok,
                 role_blocks=len(r["role_curve"]), role_curve_ok=role_ok, finite=r["finite"],
                 weights_hash_matches_file=X16.sha_file(wf) == r["weights_sha256"],
                 host=pv["host"], interpreter=pv["interpreter"], command=pv["command"], environment_at_start=pv["environment_at_start"],
                 git=pv["git"], optimizer_groups_read_back_equal_expected=r["optimizer_read_back"]["groups"] == C.expected_optimizer(arch26.new_net(SEED)),
                 cpu_per_update=r["cpu_per_update"], cpu_total=r["cpu_total"], cpu_scoring=r["cpu_scoring"],
                 last_row=dict(step=r["curve"][-1]["step"], hard=r["curve"][-1]["hard_mean_add89_sub89_k12"], data_digest_check=r["curve"][-1]["data_digest_check"]),
                 role_curve_last_block=r["role_curve"][-1]["by_role"])
        c["pass"] = (r["complete"] is True and all(present.values()) and rows_ok and role_ok and r["finite"]["updates_covered"] == 2048
                     and c["weights_hash_matches_file"] and c["optimizer_groups_read_back_equal_expected"])
        ok = ok and c["pass"]
        out[arm] = c
    cpu = [out[a]["cpu_per_update"] for a in NEW]
    return dict(arms=out, cpu_per_update=dict(min=min(cpu), max=max(cpu)), **{"pass": ok})


ITEMS = dict(host=item_host, refusal=item_refusal, mask=item_mask, balanced=item_balanced, onestep=item_onestep, dry=item_dry)

if __name__ == "__main__":
    T.set_num_threads(1)
    part = sys.argv[1]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *sys.argv[1:]])
    QUAL.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    res = ITEMS[part]()
    rec = dict(experiment="E28", stage=1, item=part, command=cmd, provenance=W.provenance(cmd),
               extra_sources_sha256={s: X16.sha_file(HERE / s) for s in ("e28_checks.py", "e25_alignment.py")}, cpu_seconds=time.process_time() - t0, result=res)
    X16.write_json(QUAL / f"check_{part}.json", rec)
    print(part, "PASS" if res.get("pass") else "FAIL", f"cpu {rec['cpu_seconds']:.0f}s", flush=True)
