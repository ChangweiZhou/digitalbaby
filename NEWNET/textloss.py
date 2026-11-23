"""Where is the text lost in the t23 life (net13)? Diagnosis only, no new mechanism (23 November 2026).
Runs t23_life_rules.py unchanged (its source is executed with one line added: a hook after each probe), so the life, its probes and
its log are the same as t23's. At line 17000 (after the text) the whole net is copied. At the end (29000), on a copy of the end net,
one part at a time is put back as it was at 17000 and the t23 text probe (1,000 never-read bytes, learning off) is run; each test
starts from a fresh copy of the end net, so nothing put back stays. Also: the overlap of active context cells and holder expansion
cells between text and rule lines (learning off), at 17000 and at the end, and how many text-active cells changed their connections.
Usage: python3 textloss.py [t23 options, e.g. net=13 struct=K8 cue=1 seed=1 rc_all=1] diag=FILE.json"""
import sys, copy, json, time, numpy as np

diag = None
for x in list(sys.argv[1:]):
    if x.startswith("diag="): diag = x.split("=", 1)[1]; sys.argv.remove(x)
SRC = open("/home/user/digitalbaby/NEWNET/t23_life_rules.py").read()
OLD = "probes.append(dict(line=ln, **r)); print(json.dumps(probes[-1]), flush=True)"
assert SRC.count(OLD) == 1
SRC = SRC.replace(OLD, OLD + "; HOOK(ln, globals())")
SNAP = {}
import os, pickle
HA, HB = (int(v) for v in os.environ.get("TEXTLOSS_HOOKS", "17000,29000").split(","))
PK = os.environ.get("TEXTLOSS_PICKLE")                                             # if set: the two nets are also saved there (to rerun the diagnosis)
CTXK = ("rsrc", "rw", "gsrc", "use", "rel")


def textprobe(net, SH_T):                                                          # the t23 text probe (learning off)
    ok = 0
    for t in range(len(SH_T) - 1):
        net.step(int(SH_T[t]), False); ok += net.answer() == SH_T[t + 1]
    return round(ok / (len(SH_T) - 1), 4)


def put(net, old, part):
    """put part back into net (the end net's copy) from old (the net at 17000)"""
    def ctx(keys):
        for k in keys: getattr(net.ctx, k)[...] = getattr(old.ctx, k)
    if part == "ctx: all (rsrc rw gsrc use rel)": ctx(CTXK)
    elif part == "ctx: connections (rsrc rw gsrc)": ctx(("rsrc", "rw", "gsrc"))
    elif part == "ctx: recycling state (use)": ctx(("use",))
    elif part == "ctx: reliability (rel)": ctx(("rel",))
    elif part == "Ac (context compartments' readout)": net.Ac[...] = old.Ac
    elif part.startswith("holder ") and part.endswith(" A"):
        m = int(part.split()[1]); net.holds[m].A[...] = old.holds[m].A
    elif part == "holders' A (all)":
        for h, o in zip(net.holds, old.holds): h.A[...] = o.A
    elif part == "holders' internal (vy, m)":
        for h, o in zip(net.holds, old.holds): h.vy[...] = o.vy; h.m[...] = o.m
    elif part == "holders' vy": [h.vy.__setitem__(Ellipsis, o.vy) for h, o in zip(net.holds, old.holds)]
    elif part == "holders' m (adaptation)": [h.m.__setitem__(Ellipsis, o.m) for h, o in zip(net.holds, old.holds)]
    elif part == "B (lag readout)": net.B[...] = old.B
    elif part == "G (relay gate)": net.G[...] = old.G
    elif part == "rule cells: all (Wr ruse Gr Gc)":
        for k in ("Wr", "ruse", "Gr", "Gc"): getattr(net, k)[...] = getattr(old, k)
    elif part == "rule cells: Gc (gains)": net.Gc[...] = old.Gc
    elif part == "rule cells: Gr (relay gate part)": net.Gr[...] = old.Gr
    elif part == "rule cells: prototypes (Wr ruse)": net.Wr[...] = old.Wr; net.ruse[...] = old.ruse
    elif part == "all readouts (Ac, holders' A, B, G, Gr, Gc)":
        for p in ("Ac (context compartments' readout)", "holders' A (all)", "B (lag readout)", "G (relay gate)"): put(net, old, p)
        net.Gr[...] = old.Gr; net.Gc[...] = old.Gc
    elif part == "everything plastic":
        for p in ("ctx: all (rsrc rw gsrc use rel)", "all readouts (Ac, holders' A, B, G, Gr, Gc)", "holders' internal (vy, m)",
                  "rule cells: all (Wr ruse Gr Gc)"): put(net, old, p)
    else: raise KeyError(part)


GROUPS = ["ctx: all (rsrc rw gsrc use rel)", "Ac (context compartments' readout)", "holders' A (all)", "holders' internal (vy, m)",
          "B (lag readout)", "G (relay gate)", "rule cells: all (Wr ruse Gr Gc)"]


def activity(net, G):
    """learning off: per-cell counts of active steps on rule probe items (fixed items) and on the text probe; also mean gains"""
    nc = net.xc.size; ng = sum(h.gran for h in net.holds)
    cR, cT, hR, hT = np.zeros(nc), np.zeros(nc), np.zeros(ng), np.zeros(ng)
    gR, gT = [], []
    def rec(c, h, g):
        c += net.xc.ravel() > 0; h += np.concatenate([h_.z > 0 for h_ in net.holds]); g.append(np.array(net.gc))
    r = np.random.default_rng(12345)
    for rule in ("copy", "rev", "succ"):
        for _ in range(60):
            w, W = G["item"](rule, 40, r)
            for b in w + [G["DEL"][rule]] + W + [10]: net.step(b, False); rec(cR, hR, gR)
    SH_T = G["SH_T"]
    for t in range(len(SH_T) - 1): net.step(int(SH_T[t]), False); rec(cT, hT, gT)
    return cR, cT, hR, hT, np.mean(gR, 0), np.mean(gT, 0)


def overlap(cR, cT):
    T, R = cT > 0, cR > 0
    return dict(n_text=int(T.sum()), n_rule=int(R.sum()), text_cells_also_rule=round(float((T & R).sum() / max(T.sum(), 1)), 4),
                text_events_on_rule_cells=round(float(cT[R].sum() / max(cT.sum(), 1)), 4),
                jaccard=round(float((T & R).sum() / max((T | R).sum(), 1)), 4))


def HOOK(ln, G):
    if ln not in (HA, HB): return
    net = G["net"]; print(f"[textloss] hook at line {ln}", flush=True)
    if ln == HA: SNAP["n17"] = copy.deepcopy(net); return
    t0 = time.time(); n17 = SNAP["n17"]; n29 = copy.deepcopy(net); SH_T = G["SH_T"]
    if PK: pickle.dump((n17, n29), open(PK, "wb"), protocol=4)
    diagnose(n17, n29, G)


def diagnose(n17, n29, G):
    t0 = time.time(); net = n29; SH_T = G["SH_T"]
    res = dict(opts=G["a"], seed=G["seed"], putback={}, without={})
    res["text_17000_copy"] = textprobe(copy.deepcopy(n17), SH_T)
    res["text_end"] = textprobe(copy.deepcopy(n29), SH_T)
    print(f"[textloss] text probe: 17000 copy {res['text_17000_copy']}, end {res['text_end']}", flush=True)
    parts = (["ctx: all (rsrc rw gsrc use rel)", "ctx: connections (rsrc rw gsrc)", "ctx: recycling state (use)", "ctx: reliability (rel)",
              "Ac (context compartments' readout)", "holders' A (all)"] + [f"holder {m} A" for m in range(len(net.holds))]
             + ["holders' internal (vy, m)", "holders' vy", "holders' m (adaptation)", "B (lag readout)", "G (relay gate)",
                "rule cells: all (Wr ruse Gr Gc)", "rule cells: Gc (gains)", "rule cells: Gr (relay gate part)", "rule cells: prototypes (Wr ruse)",
                "all readouts (Ac, holders' A, B, G, Gr, Gc)", "everything plastic"])
    for p in parts:
        n = copy.deepcopy(n29); put(n, n17, p); res["putback"][p] = textprobe(n, SH_T)
        print(f"[textloss] put back {p}: {res['putback'][p]}", flush=True)
    for q in GROUPS:                                                               # everything plastic put back except one group
        n = copy.deepcopy(n29)
        for p in GROUPS:
            if p != q: put(n, n17, p)
        res["without"][q] = textprobe(n, SH_T)
        print(f"[textloss] everything put back except {q}: {res['without'][q]}", flush=True)
    # overlaps (learning off, fixed rule items, the text probe), at 17000 and at the end
    A17 = activity(copy.deepcopy(n17), G); A29 = activity(copy.deepcopy(n29), G)
    ov = {}
    for nm, A in (("17000", A17), ("end", A29)):
        ov[nm] = dict(ctx=overlap(A[0], A[1]), holders=overlap(A[2], A[3]))
        if G["a"].get("rc_all"): ov[nm]["gain_rule_lines"] = [round(float(v), 3) for v in A[4]]; ov[nm]["gain_text"] = [round(float(v), 3) for v in A[5]]
    T17, T29 = A17[1] > 0, A29[1] > 0
    ov["ctx_text_cells_17000_vs_end_jaccard"] = round(float((T17 & T29).sum() / max((T17 | T29).sum(), 1)), 4)
    H17, H29 = A17[3] > 0, A29[3] > 0
    ov["holder_text_cells_17000_vs_end_jaccard"] = round(float((H17 & H29).sum() / max((H17 | H29).sum(), 1)), 4)
    # what changed, between 17000 and the end, in the context cells that were text-active at 17000 (cell = region x column x cell)
    L, C, M = n17.ctx.rsrc.shape[:3]
    srcchg = (n17.ctx.rsrc != n29.ctx.rsrc).any(-1).reshape(L, C * M).ravel()
    dw = np.abs(n17.ctx.rw - n29.ctx.rw).sum(-1).reshape(-1); w17 = np.abs(n17.ctx.rw).sum(-1).reshape(-1)
    usechg = (n17.ctx.use != n29.ctx.use).reshape(-1)
    R29 = A29[0] > 0
    for nm, msk in (("text_active_17000", T17), ("not_text_active_17000", ~T17)):
        ov[f"ctx_{nm}"] = dict(n=int(msk.sum()), new_connections=round(float(srcchg[msk].mean()), 4),
                               learned_as_winner_use_up=round(float(usechg[msk & np.isfinite(n17.ctx.use.reshape(-1))].mean()), 4),
                               rw_rel_change=round(float(dw[msk].sum() / max(w17[msk].sum(), 1e-9)), 4))
    ov["ctx_text17_cells_active_on_rule_lines_at_end"] = round(float((T17 & R29).sum() / max(T17.sum(), 1)), 4)
    # holders' expansion: connections fixed; how much the readout of text-active expansion cells changed
    A17h = np.concatenate([h.A for h in n17.holds], 1); A29h = np.concatenate([h.A for h in n29.holds], 1)
    da = np.linalg.norm(A29h - A17h, axis=0); a17 = np.linalg.norm(A17h, axis=0)
    for nm, msk in (("text_active_17000", H17), ("not_text_active_17000", ~H17)):
        ov[f"holders_{nm}"] = dict(n=int(msk.sum()), A_rel_change=round(float(da[msk].sum() / max(a17[msk].sum(), 1e-9)), 4))
    dAc = np.linalg.norm(n29.Ac - n17.Ac, axis=1).reshape(-1); aAc = np.linalg.norm(n17.Ac, axis=1).reshape(-1)
    for nm, msk in (("text_active_17000", T17), ("not_text_active_17000", ~T17)):
        ov[f"Ac_{nm}"] = dict(A_rel_change=round(float(dAc[msk].sum() / max(aAc[msk].sum(), 1e-9)), 4))
    res["overlap"] = ov; res["diag_s"] = round(time.time() - t0)
    print("[textloss] " + json.dumps(res), flush=True)
    if diag: open(diag, "a").write(json.dumps(res) + "\n")


exec(compile(SRC, "t23_life_rules.py", "exec"), dict(__name__="__main__", HOOK=HOOK))
