"""E24 admission check (11 Nov 2026): the balanced every-byte loss (E16_LOSSW=balanced) is what the brief says, and nothing else moved.
For the backprop birth rule and the sign sweep, on one birth batch of the two-bank learner with one head at the top:
  1. with E16_LOSS=all and the option unset, update16's gradient equals the every-byte gradient computed here by hand;
  2. with E16_LOSSW=balanced, it equals one half the answer-position gradient plus one half the other-position gradient, each computed by hand,
     and the reported loss equals the same combination of the two losses;
  3. under E16_LOSS=all every operator, digit, distractor letter and task code present in the batch receives a gradient (the end symbol as an
     input comes only after the last supervised position and receives none, as in the harness's own census); the two updates give the same loss
     and different gradients below the readout;
  4. the option is refused outside its definition (answer loss; local heads; a life).
Trains nothing (the optimizer's rate is zero). Writes exp45/E24_CHECK.json; exits nonzero on any failure.  python3 e24_check.py"""
import os, sys, json, subprocess, pathlib
HERE = pathlib.Path(__file__).parent
BASE = dict(E16_ARCH="bank2", E16_BOOKDECAY="off", OMP_NUM_THREADS="1")
RULES = {"D": {}, "T": dict(E16_BIRTHRULE="T", E16_TRACEUP="all")}


def child(rule, lossw):
    import torch as T
    from random import Random
    import exp16_integrated as G, exp11_cell as E, exp45 as X
    T.set_num_threads(1); seed = 73129; rng = Random(seed + 24)
    items = [E.render(E.DEV[i % 4], rng.randrange(89), rng.randrange(89), rng.randint(0, 12), rng) for i in range(64)]
    x, y, amask = X.batch_tensor(items); allmask = y != E.BOS; other = allmask & ~amask
    m = E.new_net(seed)
    def by_hand(mask):
        logits, it = E.forward(m, x, local=(rule != "D"), through_time=(rule == "D"), keep=(rule == "T"))
        loss = E.cross_entropy(logits[mask], y[mask]); m.zero_grad(); loss.backward()
        if rule == "T":
            E.trace_grads_scan(m, it)
        return float(loss.detach()), {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    l_all, g_all = by_hand(allmask); l_ans, g_ans = by_hand(amask); l_oth, g_oth = by_hand(other)
    opt = T.optim.SGD(m.parameters(), lr=0.0)
    w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
    l_u = G.update16(m, opt, items, rule, None)
    g_u = {n: (p.grad.clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    moved = max(float((p.detach() - w0[n]).abs().max()) for n, p in m.named_parameters())
    want_l, want_g = (l_all, g_all) if lossw == "uniform" else (.5 * l_ans + .5 * l_oth, {n: .5 * g_ans[n] + .5 * g_oth[n] for n in g_ans})
    scale = max(float(v.abs().max()) for v in want_g.values())
    worst = max(float((g_u[n] - want_g[n]).abs().max()) for n in want_g)
    wrong = max(float((g_u[n] - (g_ans[n] if lossw == "uniform" else g_all[n])).abs().max()) for n in want_g)   # able to fail: it must differ from the other candidate
    roles = {}
    gb = g_u["book.weight"].abs().sum(1)
    for ch in sorted(set(int(v) for v in x.flatten().tolist())):
        if ch == E.BOS: continue
        role = G.byte_role(ch); r = roles.setdefault(role, dict(present=0, with_gradient=0)); r["present"] += 1; r["with_gradient"] += int(float(gb[ch]) > 0)
    print(json.dumps(dict(rule=rule, lossw=lossw, loss_reported=l_u, loss_expected=want_l, loss_answer=l_ans, loss_other=l_oth, loss_all=l_all,
                          positions=dict(answer=int(amask.sum()), other=int(other.sum())), largest_gradient=scale, worst_difference=worst,
                          difference_from_the_other_candidate=wrong, weights_moved=moved, census=roles,
                          first_layer_gradient=[round(float(v), 8) for v in g_u["l1.weight"].flatten()[:6].tolist()], first_layer_gradient_norm=float(g_u["l1.weight"].norm()), stem_tag=("_bal" if G.LOSSW16 == "balanced" else ""))))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3]); sys.exit(0)
    out = dict(cells={}, refusals={}); ok = True
    env0 = {k: v for k, v in os.environ.items() if not k.startswith(("E16_", "E11_"))}
    for rule, extra in RULES.items():
        for lossw in ("uniform", "balanced"):
            env = dict(env0, **BASE, **extra, E16_LOSS="all", **({"E16_LOSSW": "balanced"} if lossw == "balanced" else {}))
            r = subprocess.run([sys.executable, __file__, "child", rule, lossw], env=env, capture_output=True, text=True, cwd=HERE)
            assert r.returncode == 0, r.stderr[-2000:]
            c = json.loads(r.stdout.strip().splitlines()[-1]); out["cells"][f"{rule} {lossw}"] = c
            tol = 1e-5 * max(1.0, c["largest_gradient"])
            good = c["worst_difference"] <= tol and abs(c["loss_reported"] - c["loss_expected"]) <= 1e-5 and c["difference_from_the_other_candidate"] > 100 * tol \
                and c["weights_moved"] == 0.0 and all(v["present"] == v["with_gradient"] for k, v in c["census"].items() if k not in ("separator", "BOS/EOS"))
            c["pass"] = good; ok = ok and good
    # the two updates must really be two updates: the same loss, different gradients below the readout
    for lossw in ("uniform", "balanced"):
        d, s = out["cells"][f"D {lossw}"], out["cells"][f"T {lossw}"]
        out[f"rules_differ_below_the_readout {lossw}"] = d["first_layer_gradient"] != s["first_layer_gradient"] and abs(d["loss_reported"] - s["loss_reported"]) < 1e-6
        ok = ok and out[f"rules_differ_below_the_readout {lossw}"]
    for name, env in (("answer loss", dict(E16_LOSSW="balanced")), ("local heads", dict(E16_LOSSW="balanced", E16_LOSS="all", E16_HEADS="local", E16_BIRTHRULE="T"))):
        e = dict(env0, **BASE); e.update(env)
        r = subprocess.run([sys.executable, "-c", "import exp16_integrated"], env=e, capture_output=True, text=True, cwd=HERE)
        out["refusals"][name] = r.returncode != 0 and "E16_LOSSW" in r.stderr; ok = ok and out["refusals"][name]
    r = subprocess.run([sys.executable, "exp16_integrated.py", "D", "73129"], env=dict(env0, **BASE, E16_LOSS="all", E16_LOSSW="balanced", E11_STAGE_UPDATES="4"), capture_output=True, text=True, cwd=HERE)
    out["refusals"]["a life"] = r.returncode != 0 and ("births only (E24)" in r.stderr or "No such file" in r.stderr); out["refusals"]["a life, message"] = r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else ""
    ok = ok and out["refusals"]["a life"]
    out["pass"] = ok
    (HERE / "exp45").mkdir(exist_ok=True); (HERE / "exp45" / "E24_CHECK.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); sys.exit(0 if ok else 1)
