"""Coordinator's own check of E26 stage 1 on the owner's machine: the worker's written-out network against E25's helper and against E20-B's receipts.
Reads saved files; trains nothing."""
import os, json, hashlib, pathlib, torch as T
import exp16_integrated as G, exp11_cell as E, exp45 as X, arch20, arch26, e25_alignment as A
T.set_num_threads(1); out = {}
for seed in (73129, 93683, 116089):
    f = pathlib.Path(f"exp45/e20b/e20_life_G1_{seed}.pt"); rec = json.loads(f.with_suffix(".json").read_text())
    m = arch20.new_net(seed, "G1"); m.load_state_dict(T.load(f))
    P, comp = G.panels(seed); items = P["add97"][12][:64] + comp["within"][:64]; x, y, mask = X.batch_tensor(items)
    with T.no_grad():
        a = arch26.forward(m, x, False, True)[0]; b = A.forward_rnn(m, x, False)[0]; lib = arch20._forward_rnn(m, x, False)[0]
    r = dict(arch26_equals_e25_helper=bool(T.equal(a, b)), max_difference_from_library=float((a - lib).abs().max()), largest_logit=float(lib.abs().max()),
             relative=float((a - lib).abs().max() / lib.abs().max()), top_byte_differs=int((a.argmax(-1) != lib.argmax(-1))[mask].sum()))
    # gradients under the sweep: the worker's network against my helper
    def g(fn):
        m.zero_grad(); E.cross_entropy(fn()[mask], y[mask]).backward(); return {n: p.grad.clone() for n, p in m.named_parameters()}
    g1 = g(lambda: arch26.forward(m, x, True, True)[0]); g2 = g(lambda: A.forward_rnn(m, x, True)[0])
    r["sweep_gradients_equal_e25_helper"] = all(T.equal(g1[n], g2[n]) for n in g1)
    # the production scorer through the worker's network, against the life's own receipt
    E.forward = arch26.forward
    sc = rec["stages"][-1]["scores"]["compose"]; mine = {k: G.score(m, comp[k]) for k in ("within", "+-", "-+", "><", "<>")}
    r["compose_counts_equal_receipt"] = {k: (mine[k]["exact"], sc[k]["exact"], mine[k]["exact"] == sc[k]["exact"] and mine[k]["final"] == sc[k]["final"]) for k in mine}
    out[seed] = r
ok = all(v["arch26_equals_e25_helper"] and v["sweep_gradients_equal_e25_helper"] and v["top_byte_differs"] == 0 and all(c[2] for c in v["compose_counts_equal_receipt"].values()) for v in out.values())
out["pass"] = ok; print(json.dumps(out, indent=1)); pathlib.Path("MY_E26_CHECK.json").write_text(json.dumps(out, indent=1))
