"""E20-P: probes on saved E20-A checkpoints (11 Nov 2026 design, after Codex's survey memo; review 9.92). Scoring only: no update is made.
Where does a twin fail: at the intermediate, after it, or because the hidden operand is no longer in the state?

  E16_COMPGAP=6 python3 e20_probe.py fixtures <seed>          the checks below, on an untrained net and stub responders (no trained weights)
  E16_COMPGAP=6 python3 e20_probe.py run <weights.pt> <seed>  the three probes on one checkpoint; writes exp45/E20P_<stem>.json

Window architecture only (the E20-A window lives: arms D, T, M). The two-bank lives need their own reset and state code, to be added when they are complete.

Probe 1, intermediate. For each item: the answer written alone (six bytes), and the ending written after the TRUE intermediate `ii;` is supplied.
  Also: is the written ending the right second step of the learner's OWN written intermediate?
Probe 2, reset. The state is set to zero (a) just before `=`, where everything read afterwards is the ten bytes the twin members share: both-correct
  must then be exactly zero and the members' outputs identical (a check of the assay, zero by construction); (b) just before the first of those ten
  bytes, Codex's wording: the window still reaches back to the first operand for a few steps, so this one is not zero by construction.
Probe 3, readout. A ridge readout of each operand's two digits and the intermediate's two digits from the state after `=`, fitted on fresh items of
  the held operator pairs and tested on the twin items. Controls: shuffled labels; an untrained net of the same seed. A linear readout is a lower
  bound on what the state holds."""
import os, sys, json, hashlib, pathlib
from random import Random
assert os.environ.get("E16_COMPGAP") == "6" and os.environ.get("E16_ARCH", "window") == "window", "run with E16_COMPGAP=6 and the window architecture"
import torch as T
import exp16_integrated as X, exp11_cell as E
EOS, BOS, W = E.EOS, E.BOS, E.W
LAMBDA, NTRAIN = 1.0, 1500      # ridge penalty on standardised features; fresh items per held operator pair. Fixed before any trained checkpoint was probed.


def parse(prompt):
    q = [ch for ch in prompt if ch not in E.DISTRACT]
    assert len(q) == 10 and q[0] == ord("c") and q[-1] == ord("="), "not a composition prompt"
    j = prompt.index(q[6], prompt.index(q[1]) + 3)   # the second operator: searched for after `aa o1` (a prompt may begin with letters and the task byte)
    return dict(a=int(bytes(q[1:3])), o1=chr(q[3]), b=int(bytes(q[4:6])), o2=chr(q[6]), c=int(bytes(q[7:9])), g2=len(prompt) - j - 4)


def digits(v):
    return [ord(ch) for ch in f"{v:02d}"]


# ---------------------------------------------------------------- probe 1
def probe_intermediate(m, items, respond=None):
    respond = respond or (lambda prompts, cap: E.respond(m, prompts, None, cap=cap))
    auto = respond([p for p, _ in items], X.CAP)
    forced = respond([p + t[:3] for p, t in items], 3)
    rows = []
    for (p, t), o, f in zip(items, auto, forced):
        d = parse(p); own = None
        if len(o) == 6 and all(48 <= ch <= 57 for ch in o[:2]) and o[2] == ord(";"):
            fo = E.apply(X.SYM_OP[d["o2"]], 97, int(bytes(o[:2])), d["c"])
            own = o[3:] == digits(fo) + [EOS]
        rows.append(dict(inter=o[:2] == t[:2], whole=o == t, forced=f == t[3:], own=own, g2=d["g2"], out=o))
    return rows


def rate(xs):
    xs = list(xs)
    return dict(n=len(xs), k=sum(bool(x) for x in xs))


def summarise(rows, twin=False):
    s = dict(intermediate=rate(r["inter"] for r in rows), whole=rate(r["whole"] for r in rows), ending_given_true_intermediate=rate(r["forced"] for r in rows),
             ending_follows_own_intermediate=rate(r["own"] for r in rows if r["own"] is not None),
             ending_follows_own_wrong_intermediate=rate(r["own"] for r in rows if r["own"] is not None and not r["inter"]), answers_with_a_readable_intermediate=sum(r["own"] is not None for r in rows),
             ending_given_true_intermediate_o2_in_view=rate(r["forced"] for r in rows if r["g2"] <= 3),
             ending_given_true_intermediate_o2_out_of_view=rate(r["forced"] for r in rows if r["g2"] > 3),
             whole_given_intermediate_right=rate(r["whole"] for r in rows if r["inter"]))
    if twin:
        prs = [(rows[2 * i], rows[2 * i + 1]) for i in range(len(rows) // 2)]
        s["both"] = dict(pairs=len(prs), intermediate=sum(x["inter"] and y["inter"] for x, y in prs), whole=sum(x["whole"] and y["whole"] for x, y in prs),
                         ending_given_true_intermediate=sum(x["forced"] and y["forced"] for x, y in prs))
        fails = [(x, y) for x, y in prs if not (x["whole"] and y["whole"])]
        s["failed_twins"] = dict(n=len(fails), an_intermediate_wrong=sum(not (x["inter"] and y["inter"]) for x, y in fails),
                                 intermediates_right_ending_wrong=sum(x["inter"] and y["inter"] for x, y in fails),
                                 would_pass_with_true_intermediate=sum(x["forced"] and y["forced"] for x, y in fails))
    return s


# ---------------------------------------------------------------- probe 2
def forward_reset(m, x, r):
    """exp11_cell.forward for the window net (single gate), with the state set to zero just before position r of x (r = None: never)."""
    lin = T.nn.functional.linear
    u = m.window(x); z1 = T.tanh(m.l1(u)); p = T.tanh(m.lp(z1)); c = T.tanh(m.lc(p)); rr = T.sigmoid(m.lr(p)); w = 1 - rr
    h = T.zeros(x.shape[0], E.C); hs = []
    for t in range(x.shape[1]):
        if r is not None and t == r:
            h = T.zeros_like(h)
        h = rr[:, t] * h + w[:, t] * c[:, t]; hs.append(h)
    return m.l3(T.tanh(m.l2(z1) + m.lm(T.stack(hs, 1))))


@T.no_grad()
def respond_reset(m, prompts, cap, where):
    """E.respond with the reset. x = [BOS] + prompt, so `=` is at index len(prompt) and the first of the last ten bytes at len(prompt) - 9."""
    out = [None] * len(prompts); groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        r = dict(eq=L, suffix=L - 9, none=None)[where]
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]; p = T.tensor([[BOS, *prompts[i]] for i in ids]); gen = [[] for _ in ids]; done = [False] * len(ids)
            for _ in range(cap):
                yy = forward_reset(m, p, r)[:, -1].argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q])); done[q] = done[q] or int(yy[q]) == EOS
                p = T.cat([p, yy[:, None]], 1)
                if all(done): break
            for q, i in enumerate(ids):
                out[i] = gen[q]
    return out


def probe_reset(m, twins):
    flat = [it for xy in twins for it in xy]; res = {}
    for where in ("none", "suffix", "eq"):
        outs = respond_reset(m, [p for p, _ in flat], X.CAP, where); ok = [o == t for o, (_, t) in zip(outs, flat)]
        n = len(twins)
        res[where] = dict(pairs=n, items_correct=sum(ok), both_correct=sum(ok[2 * i] and ok[2 * i + 1] for i in range(n)),
                          members_give_the_same_output=sum(outs[2 * i] == outs[2 * i + 1] for i in range(n)))
        if where == "suffix":      # by the number of steps for which the window still reaches the first operand's second digit after the reset
            by = {}
            for i in range(n):
                steps = max(0, 12 - (len(flat[2 * i][0]) - 10)); b = by.setdefault(steps, dict(pairs=0, both_correct=0))
                b["pairs"] += 1; b["both_correct"] += ok[2 * i] and ok[2 * i + 1]
            res[where]["by_steps_operand_still_in_window"] = {str(k): by[k] for k in sorted(by)}
    return res


# ---------------------------------------------------------------- probe 3
@T.no_grad()
def state_after_eq(m, prompts):
    out = [None] * len(prompts); groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for L, idx in groups.items():
        for j in range(0, len(idx), 256):
            ids = idx[j:j + 256]
            h = E.forward(m, T.tensor([[BOS, *prompts[i]] for i in ids]), False, True)[1]["hs"][-1]
            for q, i in enumerate(ids):
                out[i] = h[q]
    return T.stack(out).double()


def fresh_items(seed, all_taught, panel_triples):
    g = Random(f"probe|{seed}"); items = []
    def letters():
        while True:
            a, b = (bytes(g.choice(E.DISTRACT) for _ in range(g.randint(0, X.COMPGAP16))) for _ in range(2))
            if len(a) + len(b) >= 3:
                return a, b
    for o1, o2 in X.HELD_PAIRS:
        n = 0
        while n < NTRAIN:
            tr = (g.randrange(97), g.randrange(97), g.randrange(97))
            if tr in all_taught or (o1 + o2, tr) in panel_triples:
                continue
            items.append(X.render_pair(o1, o2, *tr, inside=letters())); n += 1
    return items


def labels(items):
    L = {k: [] for k in ("a", "b", "c", "i")}
    for p, t in items:
        d = parse(p)
        for k in ("a", "b", "c"):
            L[k].append(d[k])
        L["i"].append(int(bytes(t[:2])))
    return L


def ridge_readout(Ftr, Fte, ytr, yte, shuffle=None):
    """Two ten-way ridge classifiers (tens digit, units digit) on standardised features; returns accuracy of each digit and of both, and per-item hits."""
    mu, sd = Ftr.mean(0), Ftr.std(0).clamp_min(1e-8)
    A = T.cat([(Ftr - mu) / sd, T.ones(len(Ftr), 1, dtype=T.double)], 1); B = T.cat([(Fte - mu) / sd, T.ones(len(Fte), 1, dtype=T.double)], 1)
    G = A.T @ A + LAMBDA * T.eye(A.shape[1], dtype=T.double); hits = []
    if shuffle is not None:
        ytr = [ytr[i] for i in shuffle]
    for place in (10, 1):
        tr = T.tensor([v // place % 10 for v in ytr]); te = T.tensor([v // place % 10 for v in yte])
        Wt = T.linalg.solve(G, A.T @ T.nn.functional.one_hot(tr, 10).double())
        hits.append((B @ Wt).argmax(1) == te)
    both = hits[0] & hits[1]
    return dict(n=len(yte), tens=int(hits[0].sum()), units=int(hits[1].sum()), both=int(both.sum())), both.tolist()


def probe_readout(m, seed, twin_items, train_items, m0=None):
    ytr, yte = labels(train_items), labels(twin_items); res = {}; per_item = {}
    Ftr, Fte = state_after_eq(m, [p for p, _ in train_items]), state_after_eq(m, [p for p, _ in twin_items])
    perm = list(range(len(train_items))); Random(f"shuffle|{seed}").shuffle(perm)
    for k in ("a", "b", "c", "i"):
        res[k], per_item[k] = ridge_readout(Ftr, Fte, ytr[k], yte[k])
        res[k + "_shuffled_labels"] = ridge_readout(Ftr, Fte, ytr[k], yte[k], shuffle=perm)[0]
    if m0 is not None:
        F0tr, F0te = state_after_eq(m0, [p for p, _ in train_items]), state_after_eq(m0, [p for p, _ in twin_items])
        for k in ("a", "b", "c", "i"):
            res[k + "_untrained_net"] = ridge_readout(F0tr, F0te, ytr[k], yte[k])[0]
    return res, per_item


# ---------------------------------------------------------------- panels and the two commands
def build(seed):
    P, comp = X.panels(seed); mem = X.MEMORY[0]
    sha = hashlib.sha256(json.dumps(mem, sort_keys=True, default=list).encode()).hexdigest()
    taught = {p: set(X.triples(p, seed, 2000)) for p in X.TAUGHT_PAIRS}; all_taught = frozenset().union(*taught.values())
    panel_triples = {(pair, (parse(p)["a"], parse(p)["b"], parse(p)["c"])) for pair, tw in mem["twins"].items() for xy in tw for p, _ in xy}
    return mem, sha, all_taught, panel_triples


def fixtures(seed):
    out = {}; mem, sha, all_taught, panel_triples = build(seed); m = E.new_net(seed)
    tw = mem["twins"]["+-"][:20]; flat = [it for xy in tw for it in xy]
    # 1. scoring, through stub responders: an oracle; a responder whose intermediate is one too high and whose ending follows it; one without the end symbol
    def stub(kind):
        ans = {}
        for p, t in flat:
            d = parse(p)
            if kind == "oracle": o = list(t)
            elif kind == "consistent_wrong":
                iw = (int(bytes(t[:2])) + 1) % 97; o = digits(iw) + [ord(";")] + digits(E.apply(X.SYM_OP[d["o2"]], 97, iw, d["c"])) + [EOS]
            else: o = list(t[:-1]) + [t[-2]]
            ans[tuple(p)] = o
        def respond(prompts, cap):
            res = []
            for p in prompts:
                if tuple(p) in ans: res.append(ans[tuple(p)][:cap])
                else: res.append(ans[tuple(p[:-3])][3:] if kind != "no_eos" else ans[tuple(p[:-3])][3:5] + [ans[tuple(p[:-3])][4]])   # the forced call: prompt + ii;
            return res
        return summarise(probe_intermediate(None, flat, respond), twin=True)
    o, cw, ne = stub("oracle"), stub("consistent_wrong"), stub("no_eos")
    out["1_oracle"] = dict(intermediate=o["intermediate"]["k"], whole=o["whole"]["k"], forced=o["ending_given_true_intermediate"]["k"], own=o["ending_follows_own_intermediate"]["k"], both=o["both"])
    out["1_wrong_intermediate_consistent_ending"] = dict(intermediate=cw["intermediate"]["k"], whole=cw["whole"]["k"], own=cw["ending_follows_own_intermediate"]["k"], failed=cw["failed_twins"])
    out["1_no_end_symbol"] = dict(intermediate=ne["intermediate"]["k"], whole=ne["whole"]["k"], forced=ne["ending_given_true_intermediate"]["k"], readable=ne["answers_with_a_readable_intermediate"])
    n = len(flat)
    ok1 = (out["1_oracle"]["intermediate"], out["1_oracle"]["whole"], out["1_oracle"]["forced"], out["1_oracle"]["own"]) == (n, n, n, n) \
        and (cw["intermediate"]["k"], cw["whole"]["k"], cw["ending_follows_own_intermediate"]["k"]) == (0, 0, n) and cw["failed_twins"]["an_intermediate_wrong"] == n // 2 \
        and (ne["intermediate"]["k"], ne["whole"]["k"], ne["ending_given_true_intermediate"]["k"]) == (n, 0, 0)
    # 2. the reset forward equals the harness forward when no reset is asked, and differs when one is; outputs under the `=` reset are identical within twins
    x = T.tensor([[BOS, *p] for p, _ in flat if len(p) == len(flat[0][0])])
    with T.no_grad():
        ref = E.forward(m, x, False, True)[0]; same = float((forward_reset(m, x, None) - ref).abs().max()); diff = float((forward_reset(m, x, x.shape[1] - 1) - ref).abs().max())
        early = float((forward_reset(m, x, x.shape[1] - 1)[:, :-1] - ref[:, :-1]).abs().max())
    pr = probe_reset(m, tw)
    twx = T.tensor([[BOS, *flat[0][0]], [BOS, *flat[1][0]]]); L = len(flat[0][0])
    with T.no_grad():
        d_eq = float((forward_reset(m, twx, L)[0, -1] - forward_reset(m, twx, L)[1, -1]).abs().max()); d_suf = float((forward_reset(m, twx, L - 9)[0, -1] - forward_reset(m, twx, L - 9)[1, -1]).abs().max())
        d_none = float((forward_reset(m, twx, None)[0, -1] - forward_reset(m, twx, None)[1, -1]).abs().max())
    out["2_reset_forward"] = dict(max_abs_difference_without_reset=same, max_abs_difference_with_reset_at_last_position=diff, positions_before_the_reset_unchanged=early == 0.0,
                                  twin_logit_gap_at_eq=dict(no_reset=d_none, reset_before_suffix=d_suf, reset_before_eq=d_eq),
                                  untrained_net_same_output_under_eq_reset=pr["eq"]["members_give_the_same_output"], pairs=pr["eq"]["pairs"])
    ok2 = same == 0.0 and diff > 0 and early == 0.0 and d_eq == 0.0 and d_none > 0 and pr["eq"]["members_give_the_same_output"] == pr["eq"]["pairs"] and pr["eq"]["both_correct"] == 0
    # 3. the readout: features that hold the first operand as two one-hot digits plus noise must read out; shuffled labels must not; a held-out split
    g = T.Generator().manual_seed(seed); tr = fresh_items(seed, all_taught, panel_triples)[:2000]; ytr, yte = labels(tr), labels(flat)
    def feat(vals):
        f = T.randn(len(vals), 64, generator=g, dtype=T.double) * .1
        for i, v in enumerate(vals):
            f[i, v // 10] += 1; f[i, 10 + v % 10] += 1
        return f
    good = ridge_readout(feat(ytr["a"]), feat(yte["a"]), ytr["a"], yte["a"])[0]
    perm = list(range(len(tr))); Random("fx").shuffle(perm)
    shuf = ridge_readout(feat(ytr["a"]), feat(yte["a"]), ytr["a"], yte["a"], shuffle=perm)[0]
    absent = ridge_readout(feat(ytr["c"]), feat(yte["c"]), ytr["a"], yte["a"])[0]     # features hold the third operand; the first must not read out
    out["3_readout"] = dict(operand_present=good, shuffled_labels=shuf, operand_absent=absent)
    ok3 = good["both"] == good["n"] and shuf["both"] <= .15 * shuf["n"] and absent["both"] <= .15 * absent["n"]
    overlap = sum((parse(p)["o1"] + parse(p)["o2"], (parse(p)["a"], parse(p)["b"], parse(p)["c"])) in panel_triples for p, _ in fresh_items(seed, all_taught, panel_triples))
    out["3_fresh_items"] = dict(n=4 * NTRAIN, triples_shared_with_the_twin_panel=overlap, taught_triples=sum((parse(p)["a"], parse(p)["b"], parse(p)["c"]) in all_taught for p, _ in fresh_items(seed, all_taught, panel_triples)))
    ok3 = ok3 and overlap == 0 and out["3_fresh_items"]["taught_triples"] == 0
    # 4. the whole run path on the untrained net, at a small size (20 twins per pair, 200 fresh items per pair): it must execute and score nothing
    import time; t0 = time.process_time()
    pa = probe_all(m, E.new_net(seed + 1), seed, mem, all_taught, panel_triples, n_twins=20, ntrain=200)
    out["4_run_path_on_untrained_net"] = dict(cpu_seconds=round(time.process_time() - t0, 1), twins_both_whole={k: v["both"]["whole"] for k, v in pa["probe1"].items() if k.startswith("twins")},
                                              first_operand_readout=pa["probe3"]["twin_items"]["a"], third_operand_readout=pa["probe3"]["twin_items"]["c"], fitted_on=pa["probe3"]["fitted_on"])
    ok4 = all(v == 0 for v in out["4_run_path_on_untrained_net"]["twins_both_whole"].values())
    out["panel_sha256"] = sha; out["pass"] = dict(scoring=ok1, reset=ok2, readout=ok3, run_path=ok4)
    pathlib.Path(f"exp45/E20P_FIXTURES_{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)); sys.exit(0 if all(out["pass"].values()) else 1)


def probe_all(m, m0, seed, mem, all_taught, panel_triples, life_twins=None, n_twins=None, ntrain=NTRAIN):
    """The three probes on one net. life_twins: the life's own both-correct counts per operator pair, which this script must reproduce."""
    out = dict(probe1={}, probe2={}, probe3={}); all_tw, all_rows = [], []
    for pair, tw in mem["twins"].items():
        tw = tw[:n_twins]; flat = [it for xy in tw for it in xy]
        rows = probe_intermediate(m, flat); out["probe1"][f"twins {pair}"] = summarise(rows, twin=True); all_tw += flat; all_rows += rows
        ld = [it for xy in mem["lead"][pair][:n_twins] for it in xy]; out["probe1"][f"lead {pair}"] = summarise(probe_intermediate(m, ld), twin=True)
        out["probe2"][pair] = probe_reset(m, tw)
        assert out["probe2"][pair]["none"]["both_correct"] == out["probe1"][f"twins {pair}"]["both"]["whole"], "the reset forward without a reset disagrees with the harness responder"
        assert out["probe2"][pair]["eq"]["both_correct"] == 0 and out["probe2"][pair]["eq"]["members_give_the_same_output"] == len(tw), "a twin differs after the reset at =: the assay leaks"
        if life_twins is not None:     # this script must score what the life scored
            assert out["probe1"][f"twins {pair}"]["both"]["whole"] == life_twins[pair]["both_correct"], f"twins {pair}: this script and the life disagree"
    for key in ("within_hidden", "taught_hidden"):
        out["probe1"][key] = summarise(probe_intermediate(m, mem[key][:None if n_twins is None else 2 * n_twins]))
    train = fresh_items(seed, all_taught, panel_triples)
    if ntrain != NTRAIN:
        train = [it for k in range(4) for it in train[k * NTRAIN:k * NTRAIN + ntrain]]
    res, per_item = probe_readout(m, seed, all_tw, train, m0); out["probe3"]["twin_items"] = res; out["probe3"]["fitted_on"] = len(train)
    a = per_item["a"]
    out["probe3"]["first_operand_read_out_by_intermediate_written"] = {
        "read out, intermediate right": sum(x and r["inter"] for x, r in zip(a, all_rows)), "read out, intermediate wrong": sum(x and not r["inter"] for x, r in zip(a, all_rows)),
        "not read out, intermediate right": sum((not x) and r["inter"] for x, r in zip(a, all_rows)), "not read out, intermediate wrong": sum((not x) and not r["inter"] for x, r in zip(a, all_rows))}
    return out


def run(weights, seed):
    import time
    wp = pathlib.Path(weights); mem, sha, all_taught, panel_triples = build(seed)
    rec = json.loads(wp.with_suffix(".json").read_text())
    assert isinstance(rec.get("complete"), dict), "the life has no completion record"
    assert rec["complete"]["weights_sha256"] == X.sha_file(wp), "the weights file is not the one named in the life's completion record"
    life_sha = rec["stages"][0]["settings"]["memory_panels"]["sha256"]
    assert life_sha == sha, f"the memory panels built here ({sha[:12]}) are not the ones the life scored ({life_sha[:12]})"
    m = E.new_net(seed); m.load_state_dict(T.load(wp)); m.eval(); m0 = E.new_net(seed); t0 = time.process_time()
    out = dict(weights=wp.name, weights_sha256=X.sha_file(wp), seed=seed, panel_sha256=sha, lambda_=LAMBDA, fresh_items_per_pair=NTRAIN,
               script_sha256=X.sha_file(__file__), **probe_all(m, m0, seed, mem, all_taught, panel_triples, life_twins=rec["stages"][-1]["scores"]["memory"]["twins"]))
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    pathlib.Path(f"exp45/E20P_{wp.stem}.json").write_text(json.dumps(out, indent=1)); print("wrote", f"exp45/E20P_{wp.stem}.json", out["cpu_seconds"], "cpu-s")


if __name__ == "__main__":
    T.set_num_threads(1)
    if sys.argv[1] == "fixtures": fixtures(int(sys.argv[2]))
    else: run(sys.argv[2], int(sys.argv[3]))
