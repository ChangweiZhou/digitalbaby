"""E25: how well does the sign sweep follow the true gradient inside a GRU, an LSTM and the two-bank network? (11 Nov 2026, review 9.102)
A measurement on saved weights. Nothing is trained: no optimizer exists in this file.

Why. The working rule sends the error back through weight signs (scaled by the matrix's mean magnitude) wherever it crosses a weight matrix.
In the two-bank network the step back in time crosses no matrix (h_t = R h_{t-1} + w_t c_t). In a GRU or an LSTM it crosses the recurrent
matrix at every step. Before any such network is born under the rule, measure what the sweep's update looks like on E20-B's saved weights.

The GRU and LSTM steps are written out here from exp3_state.lin, so that every linear map can run either exactly or by weight signs:
  GRU   r = s(Wir x + Whr h), z = s(Wiz x + Whz h), n = tanh(Win x + r * (Whn h)), h' = (1 - z) n + z h           (torch's own form)
  LSTM  i, f, g, o from Wi x + Wh h;  c' = s(f) c + s(i) tanh(g);  h' = s(o) tanh(c')
The gates' own nonlinearities and products are local to a unit and are differentiated exactly, as in the harness's cell. One scale per whole
matrix (the stacked input matrix, the stacked recurrent matrix), as exp3_state.SignLinear does for every layer.

  E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 python3 e25_alignment.py fixtures <seed>     untrained nets only
  ... python3 e25_alignment.py run <seed>                                                    E20-B's saved births and lives of that seed

Measured, for each network (G0 two-bank, G1 GRU, G2 LSTM), each saved file (birth, end of life) and each objective (answer bytes, every byte):
  per parameter tensor   cosine between the sweep's gradient and the true gradient, and the ratio of their norms
  per state (answer objective)   cosine between the sweep's error and the true error at the state, by steps back from `=` (0 = at `=`)."""
import os, sys, json, math, pathlib, hashlib
assert os.environ.get("E16_ARCH") == "bank2" and os.environ.get("E16_BOOKDECAY") == "off" and os.environ.get("E16_COMPGAP", "0") == "0", "run with E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0"
import torch as T
import exp16_integrated as G, exp11_cell as E, exp45 as X, arch17, arch20
from exp3_state import lin
NETS = ("G0", "G1", "G2"); NITEMS = 128      # items per panel in a batch; fixed before any saved file was read
MAXBACK = 20


class Mat:
    """One of an nn.GRU / nn.LSTM's parameter pairs, seen as a linear layer by lin()."""
    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias


def forward_rnn(m, x, local):
    """arch20._forward_rnn with the recurrent layer written out. local=False: exact. local=True: weight signs on the way back."""
    u = m.book(x); z1 = T.tanh(lin(u, m.l1, local))
    B, L = x.shape; n = m.units
    gi = lin(z1, Mat(m.rnn.weight_ih_l0, m.rnn.bias_ih_l0), local)            # the input map, all steps at once
    hh = Mat(m.rnn.weight_hh_l0, m.rnn.bias_hh_l0)
    h = T.zeros(B, n); c = T.zeros(B, n); hs, cs = [], []
    for t in range(L):
        gh = lin(h, hh, local)                                                 # the recurrent map: crossed again at every step back
        if m.kind == "gru":
            i_r, i_z, i_n = gi[:, t].chunk(3, 1); h_r, h_z, h_n = gh.chunk(3, 1)
            r = T.sigmoid(i_r + h_r); z = T.sigmoid(i_z + h_z); nn_ = T.tanh(i_n + r * h_n)
            h = (1 - z) * nn_ + z * h
        else:
            i, f, g, o = (gi[:, t] + gh).chunk(4, 1)
            c = T.sigmoid(f) * c + T.sigmoid(i) * T.tanh(g); h = T.sigmoid(o) * T.tanh(c)
            if c.requires_grad: c.retain_grad()
            cs.append(c)
        if h.requires_grad: h.retain_grad()
        hs.append(h)
    Hs = T.stack(hs, 1)
    z2 = T.tanh(lin(z1, m.l2, local) + lin(Hs, m.lm, local))
    return lin(z2, m.l3, local), dict(h=hs, c=cs)


def forward_any(m, x, local):
    if hasattr(m, "kind"):
        return forward_rnn(m, x, local)
    logits, it = arch17.forward(m, x, local, True, keep=True)                  # the two-bank network: the harness's own forward, credit through time
    return logits, dict(h1=it["hs"], h2=it["hs2"])


def grads(m, x, y, mask, local):
    """Loss, every parameter's gradient, and the error at every state at every step, exactly (local=False) or by the sweep (local=True)."""
    m.zero_grad(); logits, st = forward_any(m, x, local)
    loss = E.cross_entropy(logits[mask], y[mask]); loss.backward()
    gp = {n: (p.grad.detach().clone() if p.grad is not None else T.zeros_like(p)) for n, p in m.named_parameters()}
    gs = {k: T.stack([(v.grad if v.grad is not None else T.zeros_like(v)) for v in vs], 1).detach().clone() for k, vs in st.items() if vs}
    return float(loss.detach()), gp, gs, logits.detach()


def cosine(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else None


def alignment(m, items, objective):
    x, y, amask = X.batch_tensor(items); mask = amask if objective == "answer" else (y != E.BOS)
    l_true, gp_t, gs_t, _ = grads(m, x, y, mask, False); l_sign, gp_s, gs_s, _ = grads(m, x, y, mask, True)
    assert abs(l_true - l_sign) < 1e-6, "the forward pass must not depend on the path back"
    out = dict(loss=l_true, items=len(items), parameters={}, all_parameters=cosine(T.cat([v.flatten() for v in gp_s.values()]), T.cat([v.flatten() for v in gp_t.values()])))
    for n in gp_t:
        nt = float(gp_t[n].norm())
        out["parameters"][n] = dict(cosine=cosine(gp_s[n], gp_t[n]), norm_ratio=(float(gp_s[n].norm()) / nt if nt > 0 else None), true_norm=nt)
    if objective == "answer":       # the error at each state, by steps back from `=`: per item the first supervised position is where `=` is the input
        first = amask.float().argmax(1); out["states"] = {}
        for k in gs_t:
            by = {}
            for b in range(x.shape[0]):
                e = int(first[b])
                for t in range(e + 1):
                    back = e - t
                    if back > MAXBACK: continue
                    c_ = cosine(gs_s[k][b, t], gs_t[k][b, t])
                    if c_ is not None:
                        s = by.setdefault(back, [0.0, 0]); s[0] += c_; s[1] += 1
            out["states"][k] = {str(bk): dict(mean_cosine=by[bk][0] / by[bk][1], n=by[bk][1]) for bk in sorted(by)}
    return out


def load(net, seed, which):
    m = arch20.new_net(seed, net)
    if which != "untrained":
        f = pathlib.Path(f"exp45/e20b/e20_{'dev' if which == 'birth' else 'life'}_{net}_{seed}.pt")
        rec = json.loads(f.with_suffix(".json").read_text()); c = rec.get("complete")
        want = c.get("weights_sha256") if isinstance(c, dict) else rec.get("weights_sha256")
        assert want == hashlib.sha256(f.read_bytes()).hexdigest(), f"{f.name}: the weights are not the ones its receipt names"
        m.load_state_dict(T.load(f))
    return m


def batches(seed):
    """Birth files are read on what a birth has seen (mod-89 operations at k = 12); life files on mod-97 operations at k = 12 and on new
    instances of taught compositions. Sealed panels: the items are held out, which does not matter for a gradient comparison."""
    P, comp = G.panels(seed)
    dev = P["add89"][12][:NITEMS] + P["sub89"][12][:NITEMS]
    ops = P["add97"][12][:NITEMS] + P["sub97"][12][:NITEMS]
    return dict(birth=dict(operations_mod89=dev), life=dict(operations_mod97=ops, compositions=comp["within"][:2 * NITEMS]))


def fixtures(seed):
    T.manual_seed(seed); out = {}; B = batches(seed); items = B["life"]["operations_mod97"][:64] + B["life"]["compositions"][:64]
    x, y, amask = X.batch_tensor(items); ok = True
    for net in ("G1", "G2"):
        m = load(net, seed, "untrained")
        with T.no_grad():
            lib = arch20._forward_rnn(m, x, False)[0]; mine = forward_rnn(m, x, False)[0]
        d_out = float((lib - mine).abs().max())
        # gradients through the library layer against gradients through the written-out step, exact path
        m.zero_grad(); loss = E.cross_entropy(arch20._forward_rnn(m, x, False)[0][amask], y[amask]); loss.backward()
        g_lib = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
        _, g_mine, _, _ = grads(m, x, y, amask, False)
        rel = max(float((g_lib[n] - g_mine[n]).abs().max()) / max(float(g_lib[n].abs().max()), 1e-12) for n in g_lib)
        # the sweep: the same forward values; the top layer's own gradient is exact; everything below it differs
        l_t, g_t, s_t, lo_t = grads(m, x, y, amask, False); l_s, g_s, s_s, lo_s = grads(m, x, y, amask, True)
        top = float((g_t["l3.weight"] - g_s["l3.weight"]).abs().max()); below = float((g_t["rnn.weight_hh_l0"] - g_s["rnn.weight_hh_l0"]).abs().max())
        fwd = float((lo_t - lo_s).abs().max())
        # able to fail: the written-out step with two gates' recurrent weights swapped must not match the library layer
        w = m.rnn.weight_hh_l0.data.clone(); n_ = m.units; m.rnn.weight_hh_l0.data = T.cat([w[n_:2 * n_], w[:n_], w[2 * n_:]])
        with T.no_grad():
            wrong = float((forward_rnn(m, x, False)[0] - lib).abs().max())
        m.rnn.weight_hh_l0.data = w
        a = alignment(m, items, "answer")
        out[net] = dict(outputs_max_difference=d_out, gradients_max_relative_difference=rel, sweep_forward_difference=fwd, sweep_top_layer_gradient_difference=top,
                        sweep_recurrent_gradient_difference=below, written_out_step_with_two_gates_swapped_differs_from_the_library_by=wrong, parameters=sum(p.numel() for p in m.parameters()),
                        untrained_all_parameters_cosine=a["all_parameters"], untrained_recurrent_cosine=a["parameters"]["rnn.weight_hh_l0"]["cosine"],
                        untrained_state_cosine_at_0_and_12=[a["states"]["h"].get("0", {}).get("mean_cosine"), a["states"]["h"].get("12", {}).get("mean_cosine")])
        good = d_out < 1e-5 and rel < 1e-4 and fwd == 0.0 and top < 1e-7 and below > 1e-6 and wrong > 1e-3
        out[net]["pass"] = good; ok = ok and good
    # the two-bank network through the harness's own forward: the exact path equals the gradient the harness's backprop arm takes
    m = load("G0", seed, "untrained"); l_t, g_t, s_t, _ = grads(m, x, y, amask, False)
    m.zero_grad(); logits, _ = arch17.forward(m, x, False, True); loss = E.cross_entropy(logits[amask], y[amask]); loss.backward()
    relb = max(float(((p.grad if p.grad is not None else T.zeros_like(p)) - g_t[n]).abs().max()) for n, p in m.named_parameters())
    a = alignment(m, items, "answer")
    out["G0"] = dict(exact_path_equals_harness_backprop=relb, untrained_all_parameters_cosine=a["all_parameters"], states=list(a["states"]),
                     untrained_state_cosine_at_0_and_12={k: [v.get("0", {}).get("mean_cosine"), v.get("12", {}).get("mean_cosine")] for k, v in a["states"].items()}, **{"pass": relb == 0.0})
    ok = ok and out["G0"]["pass"]
    # a cosine of a gradient with itself is one, and with its negative minus one
    v = g_t["l1.weight"]; out["cosine_self_and_negative"] = [cosine(v, v), cosine(v, -v)]; ok = ok and abs(out["cosine_self_and_negative"][0] - 1) < 1e-6 and abs(out["cosine_self_and_negative"][1] + 1) < 1e-6
    out["pass"] = ok
    pathlib.Path(f"exp45/E25_FIXTURES_{seed}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); sys.exit(0 if ok else 1)


def run(seed):
    import time
    t0 = time.process_time(); B = batches(seed); out = dict(seed=seed, items_per_panel=NITEMS, script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), cells={})
    for net in NETS:
        for which in ("birth", "life"):
            m = load(net, seed, which)
            for panel, items in B[which].items():
                for objective in ("answer", "all"):
                    out["cells"][f"{net} {which} {panel} {objective}"] = alignment(m, items, objective)
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    pathlib.Path(f"exp45/E25_ALIGNMENT_{seed}.json").write_text(json.dumps(out, indent=1)); print("wrote", f"exp45/E25_ALIGNMENT_{seed}.json", out["cpu_seconds"], "cpu-s")


if __name__ == "__main__":
    T.set_num_threads(1)
    if sys.argv[1] == "fixtures": fixtures(int(sys.argv[2]))
    else: run(int(sys.argv[2]))
