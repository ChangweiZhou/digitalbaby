"""Measurements answering the 10 November source audit (PROJECT_DOCUMENTS/MINIFLY_E16_E19_HOSTILE_SOURCE_AUDIT_20261110.md). Read-only: no training.

  census <arm>   E18 birth arms at their random initialisation: which embedding rows present in a development batch get any gradient
  projfb         E17 Phase 3/4: the trace to the write projections with raw sign(W) feedback (as run) against sign(W) * mean|W| (the map the
                 every-layer arm uses): cosine of each with the true gradient at the three bank2 births
  split          how many held single-operation pairs have their reversal in the taught pool; how many are supervised as sub-steps of taught compositions
Run with no argument to do all and write exp45/AUDIT_RESPONSE_CHECKS.json.
"""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ARMS = {"B0": dict(E16_BIRTHRULE="T", E16_TRACEUP="all"), "B1": dict(E16_HEADS="local"),
        "B2": dict(E16_HEADS="local", E16_BIRTHRULE="T", E16_TRACEUP="all"), "B3": dict(E16_HEADS="local", E16_BIRTHRULE="T")}


def role(r):
    if r >= 256: return "BOS/EOS"
    c = chr(r)
    return "digit" if c.isdigit() else "equals" if c == "=" else "operator" if c in "+-<>" else "task code" if c.islower() else "distractor letter" if c.isupper() else "separator" if c == ";" else "other"


if len(sys.argv) > 1:
    mode = sys.argv[1]
    os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E16_REPLAY="raw", E16_CAP="65536")
    if mode == "census":
        os.environ.update(ARMS[sys.argv[2]])
    import torch as T
    from torch.nn.functional import cross_entropy
    import exp16_integrated as G
    import arch17, exp11_cell as E, exp45 as X
    out = {}
    if mode == "census":
        seed = 73129
        m = E.new_net(seed)
        opt = G.adamw16(m, .001, .3 if G.BIRTHRULE16 == "D" else E.WD)
        bt = E.batches([(t, E.split(t, .5, seed)[0]) for t in E.DEV], seed, E.KMAX["M"])
        seen, got = {}, {}
        for _ in range(8):                                  # eight development batches at the initial weights; no step is taken
            items = next(bt); x, y, mask = X.batch_tensor(items)
            logits, it = E.forward(m, x, local=(G.BIRTHRULE16 != "D"), through_time=(G.BIRTHRULE16 == "D"), keep=(G.BIRTHRULE16 == "T"))
            loss = cross_entropy(logits[mask], y[mask])
            if G.HEADS16 == "local":
                loss = loss + cross_entropy(it["logits1"][mask], y[mask])
            opt.zero_grad(); loss.backward()
            if G.BIRTHRULE16 == "T":
                E.trace_grads_scan(m, it)
            g = m.book.weight.grad
            for r in sorted(set(x.unique().tolist())):
                seen[role(r)] = seen.get(role(r), set()) | {r}
                if g is not None and bool(g[r].any()):
                    got[role(r)] = got.get(role(r), set()) | {r}
            xm = x[mask]                                    # the input byte at each supervised position
            out["input_bytes_at_supervised_positions"] = sorted(set(role(int(v)) for v in xm.unique().tolist()))
        out["rows_present"] = {k: len(v) for k, v in seen.items()}
        out["rows_with_gradient"] = {k: len(got.get(k, ())) for k in seen}
    elif mode == "projfb":
        from random import Random
        res = {}
        for seed in (73129, 93683, 116089):
            m = arch17.new_net(seed, "bank2"); m.load_state_dict(T.load(G.OUT / f"e17_dev_bank2_nbd_{seed}.pt"))
            rng = Random(seed + 9); ent = G.family_entries("F1", seed)
            x, y, mask = X.batch_tensor([G.render(rng.choice(ent), rng) for _ in range(256)])
            lg, _ = arch17.forward(m, x, False, True); m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward()
            ref = {n: m.get_parameter(n).grad.clone() for n in ("lp.weight", "lp2.weight")}
            row = {"mean_abs": {n: float(m.get_parameter(n).abs().mean()) for n in ("lc.weight", "lw.weight", "lc2.weight", "lw2.weight")}}
            grads = {}
            for name, scaled in (("raw_sign_as_run", False), ("scaled_sign", True)):
                arch17.TRACEUP, arch17.FB_SIGN, arch17.PROJ_SCALED = True, True, scaled
                lg, it = arch17.forward(m, x, True, False, keep=True); m.zero_grad(); cross_entropy(lg[mask], y[mask]).backward(); arch17.trace_grads_scan(m, it)
                grads[name] = {n: m.get_parameter(n).grad.clone() for n in ref}
                row[name] = {n: float(T.dot(grads[name][n].flatten(), ref[n].flatten()) / (grads[name][n].norm() * ref[n].norm())) for n in ref}
            row["cosine_raw_vs_scaled"] = {n: float(T.dot(grads["raw_sign_as_run"][n].flatten(), grads["scaled_sign"][n].flatten()) / (grads["raw_sign_as_run"][n].norm() * grads["scaled_sign"][n].norm())) for n in ref}
            res[seed] = row
        out = res
    elif mode == "split":
        res = {}
        for seed in (73129, 400009, 1200007):
            row = {}
            for fam, frac, tasks in (("F1 (mod 97), quarter taught", .25, G.F1), ("F3 (mod 73), quarter taught", .25, G.F3), ("development (mod 89), half taught", .5, E.DEV)):
                for t in tasks:
                    taught, held = E.split(t, frac, seed); ts = set(taught)
                    off = [(a, b) for a, b in held if a != b]
                    row[f"{fam}: {E.NAME(*t)}"] = dict(held=len(held), held_off_diagonal=len(off), reversal_taught=sum((b, a) in ts for a, b in off),
                                                      share=round(sum((b, a) in ts for a, b in off) / len(off), 4), commutative=t[0] in ("add", "max", "min"))
            # sub-steps of the taught compositions: (operator, x, y) computed as the first step or as the second step of a taught scratchpad
            sub = {}
            for o1, o2 in G.TAUGHT_PAIRS:
                for a, b, c in G.triples((o1, o2), seed, 2000):
                    i = E.apply(G.SYM_OP[o1], 97, a, b)
                    sub.setdefault(G.SYM_OP[o1], set()).add((a, b)); sub.setdefault(G.SYM_OP[o2], set()).add((i, c))
            for t in G.F1:
                held = E.split(t, .25, seed)[1]
                row[f"F1 held {E.NAME(*t)} supervised as a sub-step of a taught composition"] = dict(held=len(held), exposed=sum(p in sub.get(t[0], ()) for p in held),
                                                                                                       share=round(sum(p in sub.get(t[0], ()) for p in held) / len(held), 4))
            res[seed] = row
        out = res
    print(json.dumps(out))
else:
    res = {"census_at_init": {}}
    def run(*a):
        r = subprocess.run([sys.executable, __file__, *a], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-3000:]
        return json.loads(r.stdout.strip().splitlines()[-1])
    for arm in ARMS:
        res["census_at_init"][arm] = run("census", arm)
    res["projection_feedback_map"] = run("projfb")
    res["split_overlap"] = run("split")
    (HERE / "exp45" / "AUDIT_RESPONSE_CHECKS.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
