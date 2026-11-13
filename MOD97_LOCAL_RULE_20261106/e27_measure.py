"""E27: two measurements on saved weights, no training (13 Nov 2026, review 9.113). Suggested by the audit session's note of 13 November.

Revised by the Opus worker under brief revision 2 (OPUS_TASK_E27_FLOOD_AND_ALIGNMENT_20261113.md) and the amendment of E27_PREDICTIONS.md.
The coordinator's draft (sha256 c5eb5fdd...) is preserved byte for byte in E27_REPORT.md's appendix by its hash; what changed and why is listed
there. Nothing here trains: no update rule is built, no parameter is assigned from a gradient, and the only weight edits are the positive
control's row scaling on an UNTRAINED network inside `fixtures`.

One hypothesis for two failures. Weight signs carry no magnitudes. (1) Under every-byte targets the error from bytes that cannot be predicted
might then flood back undamped and swamp the answer's error (E24's collapse). (2) Credit might not favour the pathways that already do the
work, which would show as worse alignment on composition items than on single operations (E26's weaker carry-over).

A. Flood. Under every-byte targets the loss is a weighted sum of an answer part and a part on all other bytes. Both parts' gradients are computed
   separately, under the exact path and under the sign path, on birth items (mod-89 operations at k = 12). Over a set of tensors:
     flood = |other part's gradient| * w_other / (|answer part's gradient| * w_answer)   (w = share of supervised positions)
     amplification = flood under signs / flood under the exact path
     usable = cosine(every-byte gradient under that path, the EXACT answer-only gradient)
   Sets: all parameters (registered), all parameters except the top layer l3.* (S1), every single tensor including biases and the GRU's two
   recurrent matrices (S1). "Other" is also split into distractor letters (targets A..Z) and structural bytes (task code, operands, operator,
   `=`) (S2). A null: twenty random splits of the supervised positions at the answer's share (S4). Two baselines for usable (S5).
B. Alignment by item kind. On end-of-life files of the GRU (E26 and its crossover), answer targets: cosine between the sweep's gradient and the
   true gradient, over all parameters and per tensor, on single operations (mod 97, k = 12), on new instances of taught operator pairs drawn
   evenly from all twelve taught pairs (S6), and on untaught plus-minus pairs.

Every saved file is checked against the sha256 its receipt names before it is loaded; a mismatch stops the run.

  E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 python3 e27_measure.py fixtures <seed>      untrained nets only, with the controls
  ... python3 e27_measure.py run <seed> [<seed> ...]     one process; writes exp45/E27_MEASURE_<seed>.json; absent files are listed"""
import sys, json, hashlib, pathlib, time
import torch as T
import e25_alignment as A
G, E, X, arch20 = A.G, A.E, A.X, A.arch20
N = 128          # items per panel half, as in E25; fixed before any saved file was read
NULL_DRAWS = 20  # S4, registered in the amendment
TOP = ("l3.",)   # the top layer: its own gradient is the same under both paths
ANSWER_BYTES = frozenset(range(48, 58)) | {256}   # digits and EOS: every byte an answer can hold (birth and life items)

FILES_A = {   # label: (network, path)
    "two-bank untrained": ("G0", None),
    "two-bank born by backprop, answer targets": ("G0", "exp45/e20b/e20_dev_G0_{s}.pt"),
    "two-bank born by the sweep, answer targets": ("G0", "exp45/e17_dev_bank2_nbd_bT_tall_{s}.pt"),
    "two-bank born by backprop, every byte": ("G0", "exp45/e24/e17_dev_bank2_nbd_lossall_{s}.pt"),
    "two-bank born by the sweep, every byte (collapsed)": ("G0", "exp45/e24/e17_dev_bank2_nbd_bT_tall_lossall_{s}.pt"),
    "two-bank born by the sweep, balanced": ("G0", "exp45/e24/e17_dev_bank2_nbd_bT_tall_lossall_bal_{s}.pt"),
    "GRU untrained": ("G1", None),
    "GRU born by backprop": ("G1", "exp45/e26/e26_dev_B_{s}.pt"),
    "GRU born by the sweep": ("G1", "exp45/e26/e26_dev_S_{s}.pt"),
}
FILES_B = {   # birth -> update during the life
    "sweep -> sweep": "exp45/e26/e26_life_S_{s}.pt", "backprop -> backprop": "exp45/e26/e26_life_B_{s}.pt",
    "sweep -> backprop": "exp45/e26x/e26x_life_SB_{s}.pt", "backprop -> sweep": "exp45/e26x/e26x_life_BS_{s}.pt",
}


# ---------------------------------------------------------------- receipts
def receipt_sha(p):
    """The sha256 the file's receipt names: its own .json (top level or under `complete`), else exp45/BIRTH_MANIFEST.json (births that
    predate hashed receipts, admitted there by the coordinating session)."""
    p = pathlib.Path(p); r = p.with_suffix(".json")
    if r.exists():
        rec = json.loads(r.read_text()); c = rec.get("complete")
        want = c.get("weights_sha256") if isinstance(c, dict) else rec.get("weights_sha256")
        if want:
            return want, str(r)
    man = json.loads(pathlib.Path("exp45/BIRTH_MANIFEST.json").read_text())
    if p.name in man and man[p.name].get("sha256"):
        return man[p.name]["sha256"], "exp45/BIRTH_MANIFEST.json"
    return None, None


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def net(kind, path, seed):
    m = arch20.new_net(seed, kind); rec = None
    if path is not None:
        want, src = receipt_sha(path); got = sha(path)
        assert want is not None, f"{path}: no receipt names a hash for it"
        assert want == got, f"{path}: sha256 {got} is not the {want} named by {src}"
        m.load_state_dict(T.load(path))          # strict: every parameter name must match
        rec = dict(file=path, sha256=got, receipt=src, matches_receipt=True)
    return m, rec


# ---------------------------------------------------------------- vectors
def flat(g, keep=None):
    return T.cat([v.flatten() for n, v in g.items() if keep is None or keep(n)])


def is_top(n):
    return n.startswith(TOP)


SETS = {"all_parameters": None, "all_parameters_except_top": (lambda n: not is_top(n))}


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum()) / (na * nb) if na > 0 and nb > 0 else None


def ratio(num, den):
    return num / den if den > 0 else None


# ---------------------------------------------------------------- A. flood
def masks(items):
    x, y, amask = X.batch_tensor(items); allmask = y != E.BOS; other = allmask & ~amask
    letters = other & (y >= 65) & (y <= 90); struct = other & ~letters
    return x, y, dict(answer=amask, other=other, letters=letters, structural=struct, every=allmask)


def flood(m, items, null_draws=NULL_DRAWS, null_seed=0):
    x, y, M = masks(items); n_all = float(M["every"].sum())
    w = {k: float(M[k].sum()) / n_all for k in ("answer", "other", "letters", "structural")}
    out = dict(positions={k: int(v.sum()) for k, v in M.items()}, shares=w, loss={}, decomposition_error={}, per_tensor={})
    g = {}
    for path, local in (("exact", False), ("sign", True)):
        g[path] = {}; out["loss"][path] = {}
        for part in ("answer", "other", "letters", "structural", "every"):
            l, gp, _, _ = A.grads(m, x, y, M[part], local); g[path][part] = gp; out["loss"][path][part] = l
        ga, go, gl, gs, ge = (g[path][k] for k in ("answer", "other", "letters", "structural", "every"))
        out["decomposition_error"][path] = dict(
            answer_plus_other=max(float((ge[n] - (w["answer"] * ga[n] + w["other"] * go[n])).abs().max()) for n in ge),
            answer_plus_letters_plus_structural=max(float((ge[n] - (w["answer"] * ga[n] + w["letters"] * gl[n] + w["structural"] * gs[n])).abs().max()) for n in ge))
    assert out["loss"]["exact"] == out["loss"]["sign"], "the forward pass must not depend on the path back"
    out["loss"] = out["loss"]["exact"]

    def fl(path, part, keep, den_part="answer", wd=None):
        num = w[part] * float(flat(g[path][part], keep).norm()); den = (wd if wd is not None else w[den_part]) * float(flat(g[path][den_part], keep).norm())
        return ratio(num, den)

    for sname, keep in SETS.items():
        s = {}
        for path in ("exact", "sign"):
            s[path] = dict(flood=fl(path, "other", keep), flood_letters=fl(path, "letters", keep), flood_structural=fl(path, "structural", keep),
                           usable=cos(flat(g[path]["every"], keep), flat(g["exact"]["answer"], keep)))
        for k in ("flood", "flood_letters", "flood_structural"):
            s[k.replace("flood", "amplification")] = ratio(s["sign"][k], s["exact"][k]) if s["sign"][k] is not None and s["exact"][k] is not None else None
        s["usable_baselines"] = dict(sign_answer_vs_exact_answer=cos(flat(g["sign"]["answer"], keep), flat(g["exact"]["answer"], keep)),   # S5
                                     sign_every_vs_sign_answer=cos(flat(g["sign"]["every"], keep), flat(g["sign"]["answer"], keep)))
        out[sname] = s
    for n in g["exact"]["answer"]:     # every tensor, biases and the GRU's recurrent matrices included (S1)
        f = {p: fl(p, "other", lambda k, n=n: k == n) for p in ("exact", "sign")}
        fL = {p: fl(p, "letters", lambda k, n=n: k == n) for p in ("exact", "sign")}
        out["per_tensor"][n] = dict(flood_exact=f["exact"], flood_sign=f["sign"],
                                    amplification=(ratio(f["sign"], f["exact"]) if f["sign"] is not None and f["exact"] is not None else None),
                                    amplification_letters=(ratio(fL["sign"], fL["exact"]) if fL["sign"] is not None and fL["exact"] is not None else None),
                                    usable_exact=A.cosine(g["exact"]["every"][n], g["exact"]["answer"][n]), usable_sign=A.cosine(g["sign"]["every"][n], g["exact"]["answer"][n]))
    # S4: the null. Random splits of the supervised positions with the answer's share; "answer" and "other" replaced by the two random parts.
    if null_draws:
        gen = T.Generator().manual_seed(null_seed); idx = M["every"].nonzero(as_tuple=False); na = int(M["answer"].sum())
        draws = {s: [] for s in SETS}
        for _ in range(null_draws):
            pick = idx[T.randperm(len(idx), generator=gen)[:na]]
            pa = T.zeros_like(M["every"]); pa[pick[:, 0], pick[:, 1]] = True; po = M["every"] & ~pa
            gn = {}
            for path, local in (("exact", False), ("sign", True)):
                _, gA, _, _ = A.grads(m, x, y, pa, local); _, gO, _, _ = A.grads(m, x, y, po, local); gn[path] = (gA, gO)
            for sname, keep in SETS.items():
                fe = ratio(w["other"] * float(flat(gn["exact"][1], keep).norm()), w["answer"] * float(flat(gn["exact"][0], keep).norm()))
                fs = ratio(w["other"] * float(flat(gn["sign"][1], keep).norm()), w["answer"] * float(flat(gn["sign"][0], keep).norm()))
                draws[sname].append(ratio(fs, fe))
        out["null"] = {s: dict(amplification=v, min=min(v), max=max(v), mean=sum(v) / len(v)) for s, v in draws.items()}
        out["null"]["draws"] = null_draws; out["null"]["generator_seed"] = null_seed
    return out


# ---------------------------------------------------------------- B. alignment by kind
def by_kind(m, panels):
    out = {}
    for name, items in panels.items():
        a = A.alignment(m, items, "answer")
        keep = SETS["all_parameters_except_top"]
        out[name] = dict(all_parameters=a["all_parameters"], loss=a["loss"], items=len(items),
                         per_tensor={n: v["cosine"] for n, v in a["parameters"].items()})        # every tensor (the draft kept only *weight)
    return out


def taught_even(comp, n):
    """S6: the taught panel drawn evenly from all twelve taught pairs, round robin over comp['within'] (50 items per pair, in TAUGHT_PAIRS
    order). 256 = 12 * 21 + 4: the first four pairs give 22 items, the rest 21."""
    per = len(comp["within"]) // len(G.TAUGHT_PAIRS)
    groups = [comp["within"][i * per:(i + 1) * per] for i in range(len(G.TAUGHT_PAIRS))]
    for p, grp in zip(G.TAUGHT_PAIRS, groups):     # each block is one pair: its prompts carry that pair's two operator bytes
        assert all(it[0][3] == ord(p[0]) and it[0][6] == ord(p[1]) for it in grp), f"within block for {p} is not that pair"
    out, i = [], 0
    while len(out) < n:
        out.append(groups[i % len(groups)][i // len(groups)]); i += 1
    return out


def panels(seed):
    P, comp = G.panels(seed)
    birth = P["add89"][12][:N] + P["sub89"][12][:N]
    life = dict(single_operations=P["add97"][12][:N] + P["sub97"][12][:N], taught_pairs_new_instances=taught_even(comp, 2 * N),
                untaught_plus_minus=comp["+-"][:N] + comp["-+"][:N])
    return birth, life


def panel_census(birth, life):
    def ops(items):
        c = {}
        for p, _ in items:
            k = chr(p[3]) + chr(p[6]) if len(p) == 10 else chr(p[3]); c[k] = c.get(k, 0) + 1
        return c
    return dict(birth=ops(birth), **{k: ops(v) for k, v in life.items()})


# ---------------------------------------------------------------- fixtures and controls (untrained networks only)
def positive_control(m, birth):
    """Revision 2, item 2: multiply by 0.01 the top layer's rows for output bytes that are never answer bytes. The exact path then damps the
    other bytes' error on its way down (their logits' rows are small); the sign path sends sign(W) * mean|W| and does not damp by row."""
    with T.no_grad():
        rows = [b for b in range(m.l3.weight.shape[0]) if b not in ANSWER_BYTES]
        m.l3.weight[rows] *= 0.01
    return rows


def fixtures(seed):
    out = {}; ok = True; birth, life = panels(seed)
    out["panel_census"] = panel_census(birth, life)
    for kind in ("G0", "G1"):
        m, _ = net(kind, None, seed); f = flood(m, birth, null_draws=NULL_DRAWS)
        x, y, M = masks(birth)
        wa = float(M["answer"].sum()) / float(M["every"].sum()); _, ga, _, _ = A.grads(m, x, y, M["answer"], True)
        _, go, _, _ = A.grads(m, x, y, M["other"], True); _, gall, _, _ = A.grads(m, x, y, M["every"], True)
        wrong = max(float((gall[n] - ((1 - wa) * ga[n] + wa * go[n])).abs().max()) for n in gall)   # able to fail: shares swapped
        scale = max(float(v.abs().max()) for v in gall.values())
        k = by_kind(m, life)
        top_amp = [f["per_tensor"][n]["amplification"] for n in f["per_tensor"] if is_top(n)]
        r = dict(decomposition_error=f["decomposition_error"], with_shares_swapped=wrong, largest_gradient=scale, positions=f["positions"],
                 untrained={s: f[s] for s in SETS}, untrained_null=f["null"], untrained_top_layer_amplification=top_amp,
                 untrained_alignment={n: v["all_parameters"] for n, v in k.items()},
                 per_tensor_names=sorted(f["per_tensor"]))
        dec = max(max(v.values()) for v in f["decomposition_error"].values())
        good = (dec < 1e-5 * max(1.0, scale) and wrong > 1e-3 * scale and all(v["all_parameters"] is not None for v in k.values())
                and all(abs(a - 1) < 1e-6 for a in top_amp)
                and (kind != "G1" or {"rnn.weight_ih_l0", "rnn.weight_hh_l0", "rnn.bias_ih_l0", "rnn.bias_hh_l0"} <= set(f["per_tensor"]))
                and f["positions"]["letters"] == 12 * len(birth) and f["positions"]["structural"] == 7 * len(birth))
        # positive control: same untrained net, non-answer rows of the top layer times 0.01
        rows = positive_control(m, birth); fp = flood(m, birth, null_draws=NULL_DRAWS)
        pc = dict(rows_scaled=len(rows), all_parameters_amplification=fp["all_parameters"]["amplification"],
                  except_top_amplification=fp["all_parameters_except_top"]["amplification"],
                  first_layer_amplification=fp["per_tensor"]["l1.weight"]["amplification"],
                  top_layer_amplification={n: v["amplification"] for n, v in fp["per_tensor"].items() if is_top(n)},
                  per_tensor_amplification={n: v["amplification"] for n, v in fp["per_tensor"].items()},
                  null_except_top=fp["null"]["all_parameters_except_top"], null_all=fp["null"]["all_parameters"])
        # admission, fixed before any saved file is read: the measurement moves in the positive control (except-top >= 1.5 and above the
        # null's maximum), and does not move where no matrix lies between the loss and the tensor (top layer exactly 1) nor under the null
        # (the null's mean within 0.9 to 1.1 on the control network and on the untrained one).
        pc["moves"] = (pc["except_top_amplification"] >= 1.5 and pc["except_top_amplification"] > pc["null_except_top"]["max"]
                       and pc["all_parameters_amplification"] > pc["null_all"]["max"])
        pc["negative_controls_hold"] = (all(abs(a - 1) < 1e-6 for a in pc["top_layer_amplification"].values())
                                        and 0.9 <= pc["null_except_top"]["mean"] <= 1.1 and 0.9 <= f["null"]["all_parameters_except_top"]["mean"] <= 1.1)
        r["positive_control"] = pc; r["admitted"] = pc["moves"] and pc["negative_controls_hold"]
        good = good and r["admitted"]
        r["pass"] = good; ok = ok and good; out[kind] = r
    v = {"a": T.ones(4)}; out["cosine_self"] = cos(flat(v), flat(v)); ok = ok and abs(out["cosine_self"] - 1) < 1e-6
    out["script_sha256"] = sha(__file__); out["pass"] = ok
    pathlib.Path(f"exp45/E27_FIXTURES_{seed}.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
    return ok


# ---------------------------------------------------------------- the run
def run(seed):
    t0 = time.process_time(); birth, life = panels(seed)
    out = dict(seed=seed, items=2 * N, script_sha256=sha(__file__),
               scorer_sha256_at_run=(sha("e27_score.py") if pathlib.Path("e27_score.py").exists() else None),
               torch=T.__version__, panel_census=panel_census(birth, life), A={}, B={}, absent=[])
    for label, (kind, path) in FILES_A.items():
        p = None if path is None else path.format(s=seed)
        if p is not None and not pathlib.Path(p).exists(): out["absent"].append(p); continue
        m, rec = net(kind, p, seed)
        out["A"][label] = dict(file=p, receipt=rec, **flood(m, birth, null_seed=seed))
        print(seed, label, round(time.process_time() - t0, 1), "cpu-s", flush=True)
    for label, path in FILES_B.items():
        p = path.format(s=seed)
        if not pathlib.Path(p).exists(): out["absent"].append(p); continue
        m, rec = net("G1", p, seed)
        out["B"][label] = dict(file=p, receipt=rec, **by_kind(m, life))
        print(seed, label, round(time.process_time() - t0, 1), "cpu-s", flush=True)
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    pathlib.Path(f"exp45/E27_MEASURE_{seed}.json").write_text(json.dumps(out, indent=1)); print("wrote", f"exp45/E27_MEASURE_{seed}.json", "absent:", out["absent"], flush=True)


if __name__ == "__main__":
    T.set_num_threads(1)
    if sys.argv[1] == "fixtures":
        sys.exit(0 if fixtures(int(sys.argv[2])) else 1)
    for s in sys.argv[2:]:
        run(int(s))
