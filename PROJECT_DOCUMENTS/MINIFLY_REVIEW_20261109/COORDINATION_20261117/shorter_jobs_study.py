"""Figures of SHORTER_TRAINING_JOBS_STUDY_20261118.md (how the planned training runs can be made shorter or smaller).

Run from the repository's root:   python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/shorter_jobs_study.py

Reads receipts (JSON) through `git show` from the default branch and from the branch that holds E57's receipts; writes no file.
Needs the remote-tracking ref origin/claude/e50-fixed-input-code-017dSE (fetch it as the study's head says); without it, E57 is left out
and the script says so. Uses only the standard library and numpy. Every figure is printed with the number of the section it belongs to.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict

import numpy as np

MAIN = "origin/codex/minifly-consolidated-20261023"
E57REF = "origin/claude/e50-fixed-input-code-017dSE"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/"
ROW = 256
ROWS_KEPT = 0.76    # share of scored rows kept by the adaptive schedule at theta = 40 (section 3.2)
RNG = np.random.default_rng(20261118)
# E61's revision 3 (decision 39 (i)): limits of "sufficient" (1.25 times the record's updates to 80 on E50's C46-one), per rule
LIM = {"R": {"B": 19200, "S": 42880}, "A": {"B": 17600, "S": 44480}}


def git(*a):
    p = subprocess.run(["git", *a], capture_output=True, text=True)
    return p.returncode, p.stdout


def ls(ref, folder):
    rc, out = git("ls-tree", "-r", "--name-only", ref, "--", BASE + folder + "/")
    return [p for p in out.split() if p.endswith(".json")] if rc == 0 else []


def show(ref, path):
    rc, out = git("show", f"{ref}:{path}")
    return json.loads(out) if rc == 0 else None


def P(sec, *a):
    print(f"[{sec}]", *a)


# ------------------------------------------------------------------------------------------------ loading
def skip(path):
    low = path.lower()
    return any(s in low for s in ("answers", "structure", "summary", "/qual/", "interrupted", "reference", "predictions",
                                   "tables", "driver_state", "/checks/", "/dry/", "lost_at_restart", "_dry2048", "_dry1024", "_dry64", "_dry256"))


def runs_of(ref, folder, group):
    out = []
    for p in ls(ref, folder):
        if skip(p):
            continue
        r = show(ref, p)
        if not isinstance(r, dict) or not isinstance(r.get("curve"), list) or not r["curve"]:
            continue
        name = p.split("/")[-1][:-5]
        c = r["curve"]
        g = lambda k: [row.get(k) for row in c]
        two = lambda k, a, b: [((row.get(k) or {}).get(a), (row.get(k) or {}).get(b)) for row in c]
        cells = r.get("cells") or r.get("U") or 256
        if group == "E41":
            arm = f"h{r['H']}_f{int(round(100 * r['fraction']))}"
        elif group == "E52":
            arm = f"h{r['H']}_f{int(round(100 * r['fraction']))}" + ("_product" if r.get("product_control") else "")
        elif group == "E54":
            arm = (r.get("config") or "") + (f"_f{int(round(100 * r['fraction']))}" if r.get("fraction") else "")
            if name.startswith("e41_"):
                arm = f"h{r['H']}_f{int(round(100 * r['fraction']))}_to122880"
        elif group == "E57":
            arm = name.replace("e57_", "").replace(f"_{r['seed']}", "")
        else:
            arm = r.get("setting") or r.get("config") or r.get("arm") or "?"
        lr = r.get("rate") or (r.get("optimizer") or {}).get("lr") if isinstance(r.get("optimizer"), dict) else r.get("rate")
        out.append(dict(key=folder + "/" + name, group=group, arm=str(arm), rule=r.get("rule"), seed=r.get("seed"), dose=r.get("dose"),
                        lr=lr or 0.001, cells=cells, frac=r.get("fraction") or 0.5, complete=r.get("complete"), diverged=r.get("diverged"),
                        beside=str(r.get("processes_beside")), wall=r.get("wall"),
                        steps=np.array(g("step"), float), R=np.array(g("R"), float), A=np.array(g("A"), float),
                        T=np.array(g("taught"), float),
                        L12=np.array([np.nan if v is None else v for v in g("len12")], float),
                        L40=np.array([np.nan if v is None else v for v in g("len40")], float),
                        loss=np.array([np.nan if v is None else v for v in g("loss")], float),
                        cu=np.array([np.nan if v is None else v for v in g("cpu_updates")], float),
                        cs=np.array([np.nan if v is None else v for v in g("cpu_scoring")], float),
                        Rc=two("R_counts", "add89", "sub89"), Ac=two("A_counts", "add89", "sub89"), Tc=two("taught_counts", "add89", "sub89"),
                        Lc12=[(row.get("length_counts") or {}).get("12") for row in c],
                        cpu_up=r.get("cpu_updates"), cpu_sc=r.get("cpu_scoring"), name=name))
    return out


def load():
    runs = []
    for folder, group in (("e41", "E41"), ("e42", "E42"), ("e47", "E47"), ("e49", "E49"), ("e50", "E50"), ("e52", "E52"),
                          ("e52_seeds", "E52"), ("e54", "E54")):
        runs += runs_of(MAIN, folder, group)
    e57 = runs_of(E57REF, "e57", "E57")
    if not e57:
        print("E57's branch is not fetched; E57 is left out")
    runs += e57
    other = []
    for folder, group in (("e58", "E58"), ("e59", "E59")):
        for p in ls(MAIN, folder):
            if skip(p):
                continue
            r = show(MAIN, p)
            if not isinstance(r, dict):
                continue
            fin = r.get("complete") if isinstance(r.get("complete"), dict) else r
            cu, cs = fin.get("cpu_updates"), fin.get("cpu_scoring")
            rows = r.get("rows")
            nrows = len(rows) if isinstance(rows, list) else None
            if cu and r.get("updates") and nrows:
                other.append(dict(group=group, rule=r.get("rule"), updates=r["updates"], rows=nrows, cu=cu, cs=cs))
    return runs, other


# ------------------------------------------------------------------------------------------------ readings
def reach(steps, v, thr=80.0):
    """First of three consecutive scored rows at or above thr, or None."""
    ok = v >= thr
    for i in range(len(v) - 2):
        if ok[i] and ok[i + 1] and ok[i + 2]:
            return int(steps[i])
    return None


def level(steps, v, u):
    """Mean of the three scored rows ending at update u (None if they are not all present)."""
    idx = np.where(steps == u)[0]
    if not len(idx) or idx[0] < 2:
        return None
    i = idx[0]
    if steps[i] - steps[i - 2] != 2 * ROW:
        return None
    return float(np.mean(v[i - 2:i + 1]))


def word(u, rule, panel, dose):
    if u is None:
        return "not reached"
    return "sufficient" if u <= LIM[panel][rule] else "slower"


def reading(r, upto=None):
    """Reached on R and on A (updates to 80), and the E61 words, from the rows up to update `upto` (all rows if None)."""
    m = np.ones(len(r["steps"]), bool) if upto is None else r["steps"] <= upto
    s = r["steps"][m]
    uR, uA = reach(s, r["R"][m]), reach(s, r["A"][m])
    return dict(R=uR, A=uA, wR=word(uR, r["rule"], "R", r["dose"]), wA=word(uA, r["rule"], "A", r["dose"]))


def fmt(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.{d}f}"


# ------------------------------------------------------------------------------------------------ section 1: where the time goes
def ma_counts():
    """Multiply-adds a position of E50's network (C46-one), from e50_sparse_code.forward and E49.masked: every map is a dense matrix
    product with the mask multiplied into the weights, so the work done is that of the dense maps."""
    U, D, H, O = 256, 512, 512, 257
    f = dict(input=D * 3 * U, cells=U * 3 * U, lm=U * H, l2=D * H, l3=H * O)
    used = dict(input=3 * U * 46, cells=3 * U * 11, lm=U * H, l2=D * H, l3=H * O)
    # backward: weight gradient for every map; input gradient where the input carries a gradient (not z1, which is the fixed table)
    b = dict(input=f["input"], cells=2 * f["cells"], lm=2 * f["lm"], l2=f["l2"], l3=2 * f["l3"])
    return f, b, used


def section1(runs, other):
    P("1.0", f"receipts with a curve read: {len(runs)}; groups:", dict(sorted(defaultdict(int, {g: sum(1 for r in runs if r['group'] == g) for g in {r['group'] for r in runs}}).items())))
    rows = defaultdict(list)
    for r in runs:
        if not r["complete"] or not r["cpu_up"]:
            continue
        n_up = r["steps"][-1]
        n_rows = len(r["steps"])
        key = (r["group"], r["arm"] if r["group"] not in ("E41", "E52") else r["arm"].split("_")[0], r["rule"])
        rows[key].append((r["cpu_up"] / n_up, r["cpu_sc"] / n_rows, r["cpu_sc"] / (r["cpu_up"] + r["cpu_sc"]),
                          (r["cpu_up"] + r["cpu_sc"]) / 3600, n_up))
    P("1.1", "group | network or arm | rule | runs | CPU-s an update (median; range) | CPU-s a scored row (median) | scoring share | CPU-h a run (median; updates)")
    for k in sorted(rows):
        v = np.array(rows[k])
        P("1.1", f"{k[0]} | {k[1]} | {k[2]} | {len(v)} | {np.median(v[:, 0]):.3f} ({v[:, 0].min():.3f}-{v[:, 0].max():.3f}) | {np.median(v[:, 1]):.2f} | "
                 f"{np.median(v[:, 2]):.3f} | {np.median(v[:, 3]):.2f} ({int(np.median(v[:, 4])):,})")
    for o in sorted({(o["group"], o["rule"]) for o in other}):
        v = np.array([(x["cu"] / x["updates"], x["cs"] / x["rows"], x["cs"] / (x["cu"] + x["cs"])) for x in other if (x["group"], x["rule"]) == o])
        P("1.1", f"{o[0]} (rows at an uneven schedule) | rule {o[1]} | {len(v)} runs | {np.median(v[:, 0]):.3f} s an update | {np.median(v[:, 1]):.2f} s a row | scoring share {np.median(v[:, 2]):.3f}")
    allv = np.array([v for k, vs in rows.items() for v in vs if k[0] in ("E42", "E49", "E50", "E54", "E57")])
    P("1.1", f"256-cell networks with a fixed dose schedule (E42, E49, E50, E54, E57): {len(allv)} runs; scoring share median {np.median(allv[:, 2]):.3f} "
             f"(range {allv[:, 2].min():.3f}-{allv[:, 2].max():.3f}); CPU-s a row median {np.median(allv[:, 1]):.2f}")

    # the sweep against the exact gradient, matched by group, arm and seed
    by = {(r["group"], r["arm"], r["seed"], r["lr"], r["dose"], r["rule"]): r for r in runs if r["complete"] and r["cpu_up"]}
    rat = [by[k[:5] + ("S",)]["cpu_up"] / by[k[:5] + ("S",)]["steps"][-1] / (by[k]["cpu_up"] / by[k]["steps"][-1])
           for k in by if k[5] == "B" and k[:5] + ("S",) in by]
    P("1.2", f"CPU of an update, sweep over exact gradient, {len(rat)} matched pairs: median {np.median(rat):.3f}, range {min(rat):.3f}-{max(rat):.3f}")

    # processes beside: the same computation run beside 3 others (E57) and beside fewer (E49)
    e49 = [r for r in runs if r["key"] == "e49/e49_in16_B_73129"]
    e57 = [r for r in runs if r["key"] == "e57/e57_in16_none_B_73129"]
    if e49 and e57:
        a, b = e49[0], e57[0]
        same = len(a["R"]) == len(b["R"]) and np.array_equal(a["R"], b["R"]) and np.array_equal(a["A"], b["A"]) and np.allclose(a["loss"][1:], b["loss"][1:], rtol=0, atol=1e-12)
        ca, cb = a["cpu_up"] / a["steps"][-1], b["cpu_up"] / b["steps"][-1]
        sa, sb = a["cpu_sc"] / len(a["steps"]), b["cpu_sc"] / len(b["steps"])
        P("1.3", f"E49 in16 B (beside: {a['beside'][:60]}...) and E57 in16 none B (beside: 3): curves identical in R, A and loss: {same}; "
                 f"CPU-s an update {ca:.4f} and {cb:.4f} (ratio {cb / ca:.3f}); CPU-s a row {sa:.2f} and {sb:.2f} (ratio {sb / sa:.3f})")
    hb = defaultdict(list)
    for r in runs:
        if r["group"] in ("E41", "E52") and r["cells"] == 128 and r["complete"]:
            m = re.match(r"(\d) at start", r["beside"])
            if m:
                hb[(r["rule"], int(m.group(1)))].append(r["cpu_up"] / r["steps"][-1])
    for k in sorted(hb):
        P("1.3", f"E41/E52, 128 cells, rule {k[0]}, {k[1]} processes beside at the start: {len(hb[k])} runs, CPU-s an update median {np.median(hb[k]):.4f}")
    wr = [r["wall"] / (r["cpu_up"] + r["cpu_sc"]) for r in runs if r["wall"] and r["cpu_up"] and r["group"] in ("E41", "E42", "E50", "E52", "E54", "E57")]
    P("1.3", f"wall time over CPU time of updates and scoring, {len(wr)} runs: median {np.median(wr):.3f}, 10th-90th percentile {np.percentile(wr, 10):.3f}-{np.percentile(wr, 90):.3f}")

    f, b, used = ma_counts()
    F, Bk = sum(f.values()), sum(b.values())
    P("1.4", "multiply-adds a position, E50's network (C46-one), as the source computes them (masked maps are dense products):")
    for k in f:
        P("1.4", f"  {k}: forward {f[k]:,} (in use {used[k]:,}), backward {b[k]:,}, share of forward+backward {(f[k] + b[k]) / (F + Bk):.3f}")
    rd = sum(f[k] + b[k] for k in ("lm", "l2", "l3"))
    P("1.4", f"  forward {F:,} (in use {sum(used.values()):,}); backward {Bk:,}; backward share {Bk / (F + Bk):.3f}; readout share {rd / (F + Bk):.3f} "
             f"(l2 {(f['l2'] + b['l2']) / (F + Bk):.3f}); cells {(f['cells'] + b['cells']) / (F + Bk):.3f}; input map {(f['input'] + b['input']) / (F + Bk):.3f}")
    P("1.4", f"  readout share of the forward as counted in use: {sum(used[k] for k in ('lm', 'l2', 'l3')) / sum(used.values()):.3f}, l2 {used['l2'] / sum(used.values()):.3f}; "
             f"as computed: {sum(f[k] for k in ('lm', 'l2', 'l3')) / F:.3f}, l2 {f['l2'] / F:.3f}")

    # a model of the CPU of an update: a + b x (forward multiply-adds a position, in millions), fitted on E41/E52 (arch26 at three widths)
    def arch26_fwd(Hh, Uu):
        return 3 * Uu * Hh + 3 * Uu * Uu + Uu * Hh + Hh * Hh + 257 * Hh + 8 * Hh
    pts = []
    for r in runs:
        if r["group"] in ("E41", "E52") and r["complete"] and r["rule"] == "B":
            Hh = int(r["arm"].split("_")[0][1:])
            pts.append((arch26_fwd(Hh, Hh // 2) / 1e6, r["cpu_up"] / r["steps"][-1]))
    x, y = np.array(pts).T
    bb, aa = np.polyfit(x, y, 1)
    P("1.5", f"fit on {len(pts)} E41/E52 runs under the exact gradient (widths 128, 256, 512 cells): CPU-s an update = {aa:.4f} + {bb:.4f} x (forward multiply-adds a position, millions)")
    for Hh in (256, 512, 1024):
        sel = y[np.isclose(x, arch26_fwd(Hh, Hh // 2) / 1e6)]
        P("1.5", f"  {Hh // 2} cells: forward {arch26_fwd(Hh, Hh // 2):,} a position; observed median {np.median(sel):.4f}, model {aa + bb * arch26_fwd(Hh, Hh // 2) / 1e6:.4f}")
    obs = {k: np.median([r["cpu_up"] / r["steps"][-1] for r in runs if r["group"] == "E50" and r["arm"] == "C46-one" and r["rule"] == k]) for k in "SB"}
    P("1.5", f"  E50 C46-one: forward {F:,}; model {aa + bb * F / 1e6:.4f}; observed B {obs['B']:.4f}, S {obs['S']:.4f}")
    srat = obs["S"] / obs["B"]
    rowc = np.median([r["cpu_sc"] / len(r["steps"]) for r in runs if r["group"] == "E50" and r["arm"] == "C46-one"])

    def run_cost(fwd, rule="B", updates=49152, rows=193, row_s=None, letters_scale=1.0):
        t = (aa + bb * fwd * letters_scale / 1e6) * (srat if rule == "S" else 1.0)
        rs = rowc * fwd / F if row_s is None else row_s
        return (t * updates + rs * rows) / 3600, t, rs
    variants = [("the record's network (C46-one)", F),
                ("readout without l2", F - f["l2"]),
                ("readout without l2, units reading 32 or 8 cells, masked dense as now", F - f["l2"]),
                ("readout without l2, units reading 32 cells, computed sparsely", F - f["l2"] - f["lm"] + 32 * 512),
                ("readout without l2, units reading 8 cells, computed sparsely", F - f["l2"] - f["lm"] + 8 * 512),
                ("no hidden layer: 257 output units reading 32 cells, masked dense", f["input"] + f["cells"] + 256 * 257),
                ("R = 2 exchanges a byte", F + f["cells"]), ("R = 4", F + 3 * f["cells"]), ("R = 8", F + 7 * f["cells"])]
    for name, fw in variants:
        for rule in "BS":
            h, t, rs = run_cost(fw, rule)
            P("1.6", f"{name}, rule {rule}: forward {fw:,} a position; {t:.3f} CPU-s an update; {rs:.1f} CPU-s a row; 49,152 updates and 193 rows: {h:.2f} CPU-h")
    e47 = [r for r in runs if r["group"] == "E47" and r["arm"] == "one" and r["complete"]]
    if e47:
        cu = np.mean([r["cpu_up"] / r["steps"][-1] for r in e47]); cs = np.mean([r["cpu_sc"] / len(r["steps"]) for r in e47])
        base_u = np.median([r["cpu_up"] / r["steps"][-1] for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
        base_s = np.median([r["cpu_sc"] / len(r["steps"]) for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
        P("1.7", f"E47 (1,024 cells, arm one): {cu:.3f} CPU-s an update, {cs:.1f} CPU-s a row; E42 F10 (256 cells): {base_u:.3f} and {base_s:.1f}")
        for Rx in (4, 8):
            # if the part of E47's cost above F10's is the cells' part and grows with R (an assumption; E47's runner was not read)
            u = base_u + Rx * (cu - base_u); s = base_s + Rx * (cs - base_s)
            for dose in (24576, 49152):
                P("1.7", f"  E47's network at R = {Rx}, if the cells' part ({cu - base_u:.3f} s an update, {cs - base_s:.1f} s a row) grows with R: "
                         f"{u:.2f} s an update, {s:.0f} s a row; {dose:,} updates: {(u * dose + s * (dose // 256 + 1)) / 3600:.1f} CPU-h a run")
    return dict(aa=aa, bb=bb, F=F, f=f, srat=srat, rowc=rowc)


# ------------------------------------------------------------------------------------------------ section 2: stopping early
def section2(runs, cm):
    cur = [r for r in runs if r["complete"] and len(r["steps"]) > 3 and r["steps"][1] - r["steps"][0] == ROW]
    full = {r["key"]: reading(r) for r in cur}
    P("2.0", f"curves used: {len(cur)} complete runs scored every 256 updates; reached 80 on R: {sum(full[r['key']]['R'] is not None for r in cur)}; "
             f"on A: {sum(full[r['key']]['A'] is not None for r in cur)}; on both: {sum(full[r['key']]['R'] is not None and full[r['key']]['A'] is not None for r in cur)}")
    up_total = sum(r["steps"][-1] for r in cur)
    cpu_u = {r["key"]: r["cpu_up"] / r["steps"][-1] for r in cur}
    cpu_s = {r["key"]: r["cpu_sc"] / len(r["steps"]) for r in cur}
    P("2.0", f"updates in these runs: {int(up_total):,}; CPU-h: {sum(r['cpu_up'] + r['cpu_sc'] for r in cur) / 3600:.1f}")

    def apply(stop_at, label, show=8):
        """stop_at(r) -> (update at which the run stops, verdict) or None. verdict 'reached' keeps the readings so far; 'not' reads not reached."""
        saved = 0; cpu = 0.0; changed = []; nstop = 0
        for r in cur:
            s = stop_at(r)
            if s is None:
                continue
            u, verdict = s
            if u >= r["steps"][-1]:
                continue
            nstop += 1
            saved += r["steps"][-1] - u
            cpu += (r["steps"][-1] - u) * cpu_u[r["key"]] + ((r["steps"][-1] - u) / ROW) * cpu_s[r["key"]]
            st = reading(r, u)
            fu = full[r["key"]]
            if verdict == "not":
                st = dict(R=None, A=None, wR="not reached", wA="not reached")
            diff = [k for k in ("R", "A", "wR", "wA") if st[k] != fu[k]]
            if diff:
                changed.append((r["key"], r["rule"], u, {k: (fu[k], st[k]) for k in diff}))
        P("2", f"{label}: stops {nstop} runs; saves {int(saved):,} updates ({saved / up_total:.3f} of all) and {cpu / 3600:.1f} CPU-h; readings changed in {len(changed)} runs")
        for c in changed[:show]:
            P("2", f"    changed: {c[0]} ({c[1]}) stopped at {c[2]:,}: full -> stopped {c[3]}")
        return saved, cpu, changed

    # (a) settled: 80 held on R and on A for three rows, then n further rows
    def settled(n):
        def f(r):
            uR, uA = full[r["key"]]["R"], full[r["key"]]["A"]
            if uR is None or uA is None:
                return None
            return max(uR, uA) + (2 + n) * ROW, "reached"
        return f
    res = {}
    for n in (0, 4, 8, 16, 32):
        res[("settled", n)] = apply(settled(n), f"2.1 stop when R and A have each held 80 for three rows, then {n} further rows")

    def settled24(n):
        def f(r):
            s = settled(n)(r)
            return None if s is None else (max(s[0], 24576), "reached")
        return f
    for n in (8,):
        res[("settled24", n)] = apply(settled24(n), f"2.1 the same with {n} further rows, but never before update 24,576 (the level at 24,576 kept)")
    sub = [r for r in cur if r["cells"] == 256 and r["dose"] == 49152]
    tot = sum(r["steps"][-1] for r in sub)
    for n in (0, 8):
        sv = sum(max(0, r["steps"][-1] - settled24(n)(r)[0]) for r in sub if settled24(n)(r))
        P("2.1", f"  on the {len(sub)} 256-cell runs of 49,152 updates (E50, E54, E57), {n} further rows, not before 24,576: saves {int(sv):,} of {int(tot):,} updates ({sv / tot:.3f})")
    # what such a stop loses
    lost24 = lostend = drop2 = drop5 = relapse = len_lost = 0; nst = 0; d_end = []
    for r in cur:
        fu = full[r["key"]]
        if fu["R"] is None or fu["A"] is None:
            continue
        u = max(fu["R"], fu["A"]) + 2 * ROW
        if u >= r["steps"][-1]:
            continue
        nst += 1
        if u < 24576 <= r["steps"][-1]:
            lost24 += 1
        i = np.where(r["steps"] == u)[0][0]
        at_stop = np.mean(r["R"][i - 2:i + 1]); at_end = np.mean(r["R"][-3:])
        d_end.append(at_end - at_stop)
        drop2 += abs(at_end - at_stop) > 2; drop5 += abs(at_end - at_stop) > 5
        after = r["R"][i + 1:]
        if any(after[j] < 80 and after[j + 1] < 80 and after[j + 2] < 80 for j in range(len(after) - 2)):
            relapse += 1
            P("2.2", f"  R fell below 80 for three rows after reaching it: {r['key']} ({r['rule']}); reached {fu['R']:,}; lowest later row {after.min():.1f}; level at the end {np.mean(r['R'][-3:]):.1f}")
        if not np.isnan(r["L40"][-1]):
            len_lost += 1
    d_end = np.array(d_end)
    P("2.2", f"runs that reach 80 on R and A before their end: {nst}; stopped at the third row: the level of R at the end exceeds the level at the stop by "
             f"median {np.median(d_end):.1f} points (10th-90th percentile {np.percentile(d_end, 10):.1f} to {np.percentile(d_end, 90):.1f}); differs by more than 2 points in {drop2}, by more than 5 in {drop5}")
    P("2.2", f"  runs in which R later fell below 80 for three rows after reaching it: {relapse}; runs whose level at 24,576 would be lost: {lost24}; runs whose length test at the end would be lost: {len_lost}")
    g40 = [(np.mean(r["L40"][-3:]) - np.mean(r["L40"][np.where(r["steps"] == max(full[r['key']]['R'], full[r['key']]['A']) + 2 * ROW)[0][0] - 2:np.where(r["steps"] == max(full[r['key']]['R'], full[r['key']]['A']) + 2 * ROW)[0][0] + 1]))
           for r in cur if full[r["key"]]["R"] is not None and full[r["key"]]["A"] is not None and not np.isnan(r["L40"][-1])
           and max(full[r["key"]]["R"], full[r["key"]]["A"]) + 2 * ROW < r["steps"][-1]]
    if g40:
        g40 = np.array(g40)
        P("2.2", f"  length test at 40 letters, end minus stop (mean of three rows), {len(g40)} runs: median {np.median(g40):.1f}, 10th-90th {np.percentile(g40, 10):.1f} to {np.percentile(g40, 90):.1f}, more than 5 points in {int((np.abs(g40) > 5).sum())}")

    # (b) not reached: the level of R (or T) at update u below x
    best = []
    for panel in ("R", "T"):
        for u in (4096, 8192, 12288, 16384, 24576):
            for x in (2, 5, 10, 20, 30, 40, 50, 60):
                def f(r, u=u, x=x, panel=panel):
                    if r["steps"][-1] <= u:
                        return None
                    lv = level(r["steps"], r[panel], u)
                    if lv is None or lv >= x:
                        return None
                    if full[r["key"]]["R"] is not None and full[r["key"]]["R"] <= u and full[r["key"]]["A"] is not None and full[r["key"]]["A"] <= u:
                        return None
                    return u, "not"
                saved, cpu, ch = 0, 0.0, []
                for r in cur:
                    s = f(r)
                    if s is None:
                        continue
                    saved += r["steps"][-1] - s[0]
                    cpu += (r["steps"][-1] - s[0]) * cpu_u[r["key"]] + (r["steps"][-1] - s[0]) / ROW * cpu_s[r["key"]]
                    fu = full[r["key"]]
                    if fu["R"] is not None or fu["A"] is not None:
                        ch.append(r["key"])
                best.append((len(ch), -saved, panel, u, x, saved, cpu, ch))
    best.sort(key=lambda t: (t[0], t[1]))
    P("2.3", "stop as 'not reached' when the level (mean of three rows) of a panel at update u is below x; best rules by changed readings, then saving:")
    for t in best[:12]:
        P("2.3", f"  {t[2]} below {t[4]} at {t[3]:,}: saves {int(t[5]):,} updates ({t[5] / up_total:.3f}), {t[6] / 3600:.1f} CPU-h; runs whose full reading reached 80 on R or A: {t[0]} {t[7][:4]}")
    zero = [t for t in best if t[0] == 0]
    zbest = max(zero, key=lambda t: t[5]) if zero else None
    if zbest:
        P("2.3", f"  largest saving with no changed reading: {zbest[2]} below {zbest[4]} at {zbest[3]:,}: {int(zbest[5]):,} updates, {zbest[6] / 3600:.1f} CPU-h")
        # margin: the lowest level at u among runs that later reached
        lv = [(level(r["steps"], r[zbest[2]], zbest[3]), r["key"], full[r["key"]]["R"], full[r["key"]]["A"]) for r in cur
              if r["steps"][-1] > zbest[3] and not (full[r["key"]]["R"] and full[r["key"]]["A"] and max(full[r["key"]]["R"], full[r["key"]]["A"]) <= zbest[3])
              and (full[r["key"]]["R"] is not None or full[r["key"]]["A"] is not None)]
        lv = sorted([v for v in lv if v[0] is not None])[:5]
        P("2.3", f"  margin: the lowest levels of {zbest[2]} at {zbest[3]:,} among runs not settled by then that later reached 80 on R or A: "
                 + "; ".join(f"{v[1]} {v[0]:.1f} (80 on R at {fmt(v[2])}, on A at {fmt(v[3])})" for v in lv))
    # which runs a moderate rule stops and what end level they had
    for (panel, u, x) in (("R", 16384, 20), ("R", 24576, 40), ("T", 16384, 50)):
        hit = [r for r in cur if r["steps"][-1] > u and (level(r["steps"], r[panel], u) or 999) < x]
        ends = sorted((round(float(np.mean(r["R"][-3:])), 1), r["key"], fmt(full[r["key"]]["R"])) for r in hit)
        P("2.4", f"rule {panel} below {x} at {u:,}: stops {len(hit)} runs; their levels of R at the end (and full updates to 80 on R): " + "; ".join(f"{e[1]} {e[0]} ({e[2]})" for e in ends[-6:]))

    # (c) loss: not fallen by a share y over the last w updates
    lres = []
    for w in (2048, 4096, 8192):
        for y in (0.0, 0.01, 0.02, 0.05):
            saved, ch, nst = 0, [], 0
            for r in cur:
                k = w // ROW
                L = r["loss"]
                fu = full[r["key"]]
                for i in range(2 * k + 1, len(L)):
                    u = r["steps"][i]
                    if u < 8192:
                        continue
                    if (fu["R"] is not None and fu["R"] + 2 * ROW <= u) and (fu["A"] is not None and fu["A"] + 2 * ROW <= u):
                        break
                    a = np.nanmean(L[i - k + 1:i + 1]); b = np.nanmean(L[i - 2 * k + 1:i - k + 1])
                    if b > 0 and (b - a) / b < y:
                        nst += 1; saved += r["steps"][-1] - u
                        if fu["R"] is not None or fu["A"] is not None:
                            ch.append(r["key"])
                        break
            lres.append((w, y, nst, saved, ch))
    for t in lres:
        P("2.5", f"loss not fallen by {t[1]:.0%} over the last {t[0]:,} updates (checked from 8,192): stops {t[2]} runs, saves {int(t[3]):,} updates; of them reached 80 later: {len(t[4])} {t[4][:3]}")

    # (d) the limit of 'sufficient': stop at 1.25 times the record's updates when not reached by then
    def at_limit(r):
        if r["group"] not in ("E50", "E57", "E49") or r["dose"] != 49152:
            return None
        lim = max(LIM["R"][r["rule"]], LIM["A"][r["rule"]])
        lim = int(np.ceil(lim / ROW) * ROW) + 2 * ROW
        fu = full[r["key"]]
        if fu["R"] is not None and fu["A"] is not None and max(fu["R"], fu["A"]) + 2 * ROW <= lim:
            return None
        return lim, "keep"
    saved, cpu, ch = apply(at_limit, "2.6 E49/E50/E57 runs of 49,152 updates: stop at the limit of 'sufficient' (E61's 1.25 times) plus two rows when not settled by then", show=12)
    return dict(full=full, cur=cur, up_total=up_total, zbest=zbest, settled=res)


# ------------------------------------------------------------------------------------------------ section 3: scoring less
def resample(counts, n_have, n_take, reps):
    """Hypergeometric draws of the right answers among n_take of n_have items, for each recorded count."""
    c = np.array([0 if v is None else v for v in counts], int)
    return RNG.hypergeometric(c[None, :].repeat(reps, 0), (n_have - c)[None, :].repeat(reps, 0), n_take)


def section3(runs, s2):
    cur, full = s2["cur"], s2["full"]
    # dense to coarse rows
    for g in (512, 1024):
        diffs, flips, flipsA, n = [], 0, 0, 0
        for r in cur:
            m = (r["steps"] % g == 0)
            s = r["steps"][m]
            uR, uA = reach(s, r["R"][m]), reach(s, r["A"][m])
            fu = full[r["key"]]
            n += 1
            if (uR is None) != (fu["R"] is None):
                flips += 1
            elif uR is not None:
                diffs.append(uR - fu["R"])
            if (uA is None) != (fu["A"] is None):
                flipsA += 1
        d = np.array(diffs)
        P("3.1", f"rows every {g:,} (three consecutive rows at that spacing): runs {n}; reached/not reached on R changed in {flips}, on A in {flipsA}; "
                 f"updates to 80 on R later by median {np.median(d):,.0f}, mean {d.mean():,.0f}, max {d.max():,.0f}, earlier in {int((d < 0).sum())} runs; saves {1 - ROW / g:.0%} of the rows")
    # an adaptive schedule: rows every 1,024 updates while R and A are both below theta, every 256 once either is at or above it;
    # plus the rows that the levels need (24,064 to 24,576, and the last three)
    for theta in (30, 40, 50, 60, 70):
        rows_full = rows_ad = 0; ch = []
        for r in cur:
            st = r["steps"]; last = st[-1]
            keep = set([0.0, last - 512, last - 256, last, 24064.0, 24320.0, 24576.0])
            u = 0.0; idx = {v: i for i, v in enumerate(st)}
            while u < last:
                i = idx[u]
                step = ROW if (r["R"][i] >= theta or r["A"][i] >= theta) else 1024
                u = min(u + step, last)
                if u not in idx:
                    u = st[st > u - step][0]
                keep.add(u)
            keep = sorted(k for k in keep if k in idx)
            sel = np.array([idx[k] for k in keep])
            s = st[sel]
            # a reach is read only where the three rows are 256 apart
            def rch(v):
                ok = v[sel] >= 80
                for j in range(len(sel) - 2):
                    if ok[j] and ok[j + 1] and ok[j + 2] and s[j + 2] - s[j] == 2 * ROW:
                        return int(s[j])
                return None
            uR, uA = rch(r["R"]), rch(r["A"])
            fu = full[r["key"]]
            if uR != fu["R"] or uA != fu["A"]:
                ch.append((r["key"], fu["R"], uR, fu["A"], uA))
            rows_full += len(st); rows_ad += len(sel)
        P("3.2", f"adaptive rows (every 1,024 while R and A are below {theta}, every 256 after; the rows of the levels kept): rows {rows_ad:,} of {rows_full:,} "
                 f"({rows_ad / rows_full:.3f}); readings changed in {len(ch)} runs " + "; ".join(f"{c[0]} R {fmt(c[1])}->{fmt(c[2])} A {fmt(c[3])}->{fmt(c[4])}" for c in ch[:4]))
    # smaller panels, by resampling the recorded counts row by row
    reps = 200
    res = {}
    for label, nR, nA in (("R 500 items (250 a task)", 250, None), ("A 300 items (150 a task)", None, 150), ("R 500 and A 300", 250, 150)):
        dR, fR, dA, fA, lvl_sd, nrun = [], 0, [], 0, [], 0
        word_ch = 0
        for r in cur:
            if any(a is None for a, b in r["Rc"]) or any(a is None for a, b in r["Ac"]):
                continue
            nrun += 1
            fu = full[r["key"]]
            if nR:
                Ra = resample([a for a, b in r["Rc"]], 1000, nR, reps); Rb = resample([b for a, b in r["Rc"]], 1000, nR, reps)
                Rv = 100 * (Ra + Rb) / (2 * nR)
            else:
                Rv = np.repeat(r["R"][None, :], reps, 0)
            if nA:
                Aa = resample([a for a, b in r["Ac"]], 300, nA, reps); Ab = resample([b for a, b in r["Ac"]], 300, nA, reps)
                Av = 100 * (Aa + Ab) / (2 * nA)
            else:
                Av = np.repeat(r["A"][None, :], reps, 0)
            lvl_sd.append(np.std(Rv[:, -3:].mean(1)))
            for k in range(reps):
                uR, uA = reach(r["steps"], Rv[k]), reach(r["steps"], Av[k])
                if (uR is None) != (fu["R"] is None):
                    fR += 1
                elif uR is not None:
                    dR.append(uR - fu["R"])
                if (uA is None) != (fu["A"] is None):
                    fA += 1
                elif uA is not None:
                    dA.append(uA - fu["A"])
                if word(uR, r["rule"], "R", r["dose"]) != fu["wR"] or word(uA, r["rule"], "A", r["dose"]) != fu["wA"]:
                    word_ch += 1
        dR, dA = np.array(dR), np.array(dA)
        P("3.3", f"{label}: {nrun} runs x {reps} draws; reached/not reached changed on R in {fR / (nrun * reps):.4f} of draws, on A in {fA / (nrun * reps):.4f}; "
                 f"E61's words changed in {word_ch / (nrun * reps):.4f}; |error| in updates to 80 on R: median {np.median(np.abs(dR)) if len(dR) else 0:,.0f}, "
                 f"90th percentile {np.percentile(np.abs(dR), 90) if len(dR) else 0:,.0f}, 99th {np.percentile(np.abs(dR), 99) if len(dR) else 0:,.0f}; "
                 f"on A: 90th percentile {np.percentile(np.abs(dA), 90) if len(dA) else 0:,.0f}; s.d. of the level of R at the end {np.median(lvl_sd):.2f} points (median over runs)")
    # the full panels' own sampling error, for scale: the binomial s.d. of R at 80 with 2,000 items and of A with 600
    P("3.3", f"for scale, binomial s.d. of a share near 80: R with 2,000 items {100 * np.sqrt(.16 / 2000):.2f} points, with 500 {100 * np.sqrt(.16 / 500):.2f}; "
             f"A with 600 {100 * np.sqrt(.16 / 600):.2f}, with 300 {100 * np.sqrt(.16 / 300):.2f}")
    # differences between arms that readings rest on: pairs in one experiment with the same seed and rule
    pairs = []
    groups = defaultdict(list)
    for r in cur:
        groups[(r["group"], r["seed"], r["rule"], r["lr"] if r["group"] != "E52" else 0, r["cells"])].append(r)
    for k, rs in groups.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = full[rs[i]["key"]]["R"], full[rs[j]["key"]]["R"]
                if a is not None and b is not None:
                    pairs.append((abs(a - b), max(a, b) / min(a, b), rs[i]["key"], rs[j]["key"]))
    d = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
    P("3.4", f"pairs of runs (same experiment, seed, rule) that both reach 80 on R: {len(pairs)}; difference in updates to 80: median {np.median(d):,.0f}, "
             f"10th percentile {np.percentile(d, 10):,.0f}; pairs differing by less than 512 updates {int((d < 512).sum())}, less than 1,024 {int((d < 1024).sum())}; "
             f"ratio of the two within 1.25: {int((q <= 1.25).sum())}")
    # how often coarse rows change the side of 1.25 for a pair
    for g in (512, 1024):
        flip = 0
        for (dd, qq, ka, kb) in pairs:
            ra = [r for r in cur if r["key"] == ka][0]; rb = [r for r in cur if r["key"] == kb][0]
            ua = reach(ra["steps"][ra["steps"] % g == 0], ra["R"][ra["steps"] % g == 0]); ub = reach(rb["steps"][rb["steps"] % g == 0], rb["R"][rb["steps"] % g == 0])
            if ua and ub and ((max(ua, ub) / min(ua, ub) <= 1.25) != (qq <= 1.25)):
                flip += 1
        P("3.4", f"rows every {g:,}: pairs whose ratio moves across 1.25: {flip} of {len(pairs)}")
    # duplication in the panels and their cost
    eq = tot = 0
    for r in runs:
        for lc, ac in zip(r["Lc12"], r["Ac"]):
            if lc is not None and ac[0] is not None:
                tot += 1; eq += (lc == ac[0])
    P("3.5", f"rows in which the length test at 12 letters has the count of A's add89 panel: {eq} of {tot}")
    return pairs


# scoring work, in positions passed through the network, from exp42_cells.panels and exp11_cell.respond
PANELS = dict(A=(1200, 19), T=(1200, 19), R=(2000, 7), L12=(300, 19), L40=(300, 47))   # items, prompt length (without the start symbol)


def scoring_work(f, gen=3, incremental=False, readout_last=False, lookup=False, dedupe=False, drop_mm=False, only=None):
    cells_in = f["cells"] + (0 if lookup else f["input"])
    rd = f["lm"] + f["l2"] + f["l3"]
    tot = {}
    for k, (n, L0) in PANELS.items():
        if only and k not in only:
            continue
        if dedupe and k == "L12":
            continue
        if drop_mm and k in ("A", "T"):
            n //= 2
        if incremental:
            w = (L0 + gen) * cells_in + gen * rd
        else:
            pos = sum(L0 + 1 + s for s in range(gen))
            w = pos * (cells_in + (0 if readout_last else rd)) + (gen * rd if readout_last else 0)
        tot[k] = n * w
    return tot


def section7(cm, runs):
    f, aa, bb, F, srat, rowc = cm["f"], cm["aa"], cm["bb"], cm["F"], cm["srat"], cm["rowc"]
    now = scoring_work(f)
    tn = sum(now.values())
    P("7.1", "scoring work now by panel (share): " + ", ".join(f"{k} {v / tn:.3f}" for k, v in now.items()) + f"; total {tn / 1e9:.2f} G multiply-adds a row (three generated bytes)")
    cases = [("the length test at 12 letters taken from A's add89 (the same items)", dict(dedupe=True)),
             ("the readout only at the last position", dict(readout_last=True)),
             ("the state carried: each generated byte one new position", dict(incremental=True)),
             ("carried state, the input map as a table of 258 rows", dict(incremental=True, lookup=True)),
             ("carried state, table, and the length test at 12 from A", dict(incremental=True, lookup=True, dedupe=True)),
             ("the same, and A and T without max and min (changes the panels' content)", dict(incremental=True, lookup=True, dedupe=True, drop_mm=True))]
    for name, kw in cases:
        w = sum(scoring_work(f, **kw).values())
        P("7.2", f"{name}: work {w / tn:.3f} of now; a row about {rowc * w / tn:.2f} CPU-s against {rowc:.2f}; 193 rows {193 * rowc * w / tn / 3600:.2f} CPU-h against {193 * rowc / 3600:.2f}")
    # training: the readout only at the three answer positions; the input map as a table
    Lpad, ans, B = 22, 3, 64
    cells_in = f["cells"] + f["input"]
    rd = f["lm"] + f["l2"] + f["l3"]
    fwd_now = Lpad * (cells_in + rd)
    for name, fw in (("now", fwd_now), ("readout only at the answer positions", Lpad * cells_in + ans * rd),
                     ("and the input map as a table (258 rows an update)", Lpad * f["cells"] + ans * rd + 258 * f["input"] / B),
                     ("no l2, readout at the answer positions, table", Lpad * f["cells"] + ans * (f["lm"] + f["l3"]) + 258 * f["input"] / B)):
        t = (aa + bb * fw / Lpad / 1e6)
        P("7.3", f"training update, {name}: forward {fw / Lpad:,.0f} multiply-adds a padded position; model {t:.4f} CPU-s an update under the exact gradient "
                 f"({t * srat:.4f} under the sweep); 49,152 updates {t * 49152 / 3600:.2f} / {t * srat * 49152 / 3600:.2f} CPU-h")
    # padding: the length of the batch is the longest item
    rng = np.random.default_rng(1)
    k = rng.integers(0, 13, size=(20000, 64))
    P("7.4", f"items of 10 to 22 positions (0 to 12 letters): mean length of an item {10 + k.mean():.1f}, mean longest item of a batch of 64 {10 + k.max(1).mean():.2f}; "
             f"share of padded positions {1 - (10 + k.mean()) / (10 + k.max(1).mean()):.3f}")
    k4 = rng.integers(0, 5, size=(20000, 64))
    P("7.4", f"0 to 4 letters: mean longest item {10 + k4.max(1).mean():.2f} positions, against 22")


# ------------------------------------------------------------------------------------------------ section 4: smaller task
def section4(runs, s2, cm):
    full = s2["full"]
    cur = s2["cur"]
    P("4.1", "E41 (dense network; 128, 256, 512 cells; pairs taught 10, 25, 50 per cent): updates to 80 on R and A (median over seeds; reached/runs), level of R at the end, CPU-s an update")
    tab = defaultdict(list)
    for r in cur:
        if r["group"] == "E41" or (r["group"] == "E54" and "to122880" in r["arm"]):
            tab[(r["arm"].replace("_to122880", "") + ("_to122880" if "to122880" in r["arm"] else ""), r["rule"])].append(r)
    for k in sorted(tab):
        rs = tab[k]
        uR = [full[r["key"]]["R"] for r in rs]; uA = [full[r["key"]]["A"] for r in rs]
        P("4.1", f"  {k[0]} {k[1]}: dose {int(rs[0]['dose']):,}; R reached {sum(u is not None for u in uR)}/{len(rs)} (median {fmt(np.median([u for u in uR if u]) if any(uR) else None)}); "
                 f"A {sum(u is not None for u in uA)}/{len(rs)} (median {fmt(np.median([u for u in uA if u]) if any(uA) else None)}); R at end {np.mean([np.mean(r['R'][-3:]) for r in rs]):.1f}; "
                 f"{np.median([r['cpu_up'] / r['steps'][-1] for r in rs]):.3f} s an update")
    P("4.2", "E54 (fewer taught pairs, 49,152 updates) beside E42 (half the pairs, 24,576 updates), seed 73129: updates to 80 on R / A; level of R at 24,576 and at the end")
    for r in sorted([r for r in cur if (r["group"] == "E54" and "to122880" not in r["arm"]) or (r["group"] == "E42" and r["arm"] in ("G", "F10", "P8") and r["seed"] == 73129)], key=lambda r: (r["arm"].split("_")[0], r["rule"], r["frac"])):
        fu = full[r["key"]]
        P("4.2", f"  {r['group']} {r['arm']} {r['rule']} pairs {r['frac']:.2f}: {fmt(fu['R'])} / {fmt(fu['A'])}; R {fmt(level(r['steps'], r['R'], 24576), 1)} at 24,576, {np.mean(r['R'][-3:]):.1f} at end")
    # length: what the record says about items longer than those taught
    l40 = [(np.mean(r["L40"][-3:]), np.mean(r["L12"][-3:]), r["key"]) for r in cur if not np.isnan(r["L40"][-1])]
    a = np.array([x[0] for x in l40]); b = np.array([x[1] for x in l40])
    good = b >= 80
    P("4.3", f"length test at the end (taught 0 to 12 letters; tested at 12 and at 40): runs {len(l40)}; with 12 letters at or above 80: {int(good.sum())}; "
             f"among them 40 letters: median {np.median(a[good]):.1f}, 10th-90th {np.percentile(a[good], 10):.1f}-{np.percentile(a[good], 90):.1f}; "
             f"at or above 80 at 40 letters: {int((a[good] >= 80).sum())}")
    dl = [(full[r["key"]]["A"] - full[r["key"]]["R"]) for r in cur if full[r["key"]]["A"] is not None and full[r["key"]]["R"] is not None]
    dl = np.array(dl)
    P("4.3", f"updates to 80 on A (12 letters) minus on R (no letters), runs reaching both: {len(dl)}; median {np.median(dl):,.0f}; A first in {int((dl < 0).sum())}, R first in {int((dl > 0).sum())}")
    aa, bb, f, srat = cm["aa"], cm["bb"], cm["f"], cm["srat"]
    F = cm["F"]
    for L in (14, 22):
        lo = aa + bb * F / 1e6 * L / 22; hi = (aa + bb * F / 1e6) * L / 22
        P("4.4", f"training on items with at most {L - 10} letters (batches of {L} positions): model {hi:.4f} to {lo:.4f} CPU-s an update (exact gradient), against {aa + bb * F / 1e6:.4f}")


# ------------------------------------------------------------------------------------------------ section 5: the rate
def section5(runs, s2):
    full = s2["full"]
    by = {}
    for r in s2["cur"]:
        if r["group"] in ("E41", "E52") and "product" not in r["arm"]:
            by[(r["arm"], r["rule"], r["seed"], r["lr"])] = r
    P("5.1", "dense network (E41 at 0.001; E52 at 0.0003, 0.001, 0.003, 0.01): arm, rule, seed: updates to 80 on R and A at 0.001 -> 0.003 (and 0.01); level of R at the end; most R fell from its best")
    rows = []
    for (arm, rule, seed, lr), r in sorted(by.items()):
        if lr != 0.003:
            continue
        base = by.get((arm, rule, seed, 0.001))
        if base is None:
            continue
        hi = by.get((arm, rule, seed, 0.01))
        fb, f3 = full[base["key"]], full[r["key"]]
        n = min(len(base["steps"]), len(r["steps"]))
        drop3 = float(np.max(r["R"]) - np.mean(r["R"][-3:]))
        rows.append((arm, rule, seed, fb, f3, np.mean(base["R"][n - 3:n]), np.mean(r["R"][n - 3:n]), drop3, hi, base["dose"], r["dose"]))
        P("5.1", f"  {arm} {rule} {seed}: R {fmt(fb['R'])} -> {fmt(f3['R'])}, A {fmt(fb['A'])} -> {fmt(f3['A'])}; R at the common end ({int(min(base['steps'][-1], r['steps'][-1])):,}) "
                 f"{np.mean(base['R'][n - 3:n]):.1f} -> {np.mean(r['R'][n - 3:n]):.1f}; fall from best at 0.003 {drop3:.1f}"
                 + (f"; at 0.01: R {fmt(full[hi['key']]['R'])}, end {np.mean(hi['R'][-3:]):.1f}, diverged {hi['diverged']}" if hi else ""))
    rat = [(x[4]["R"] / x[3]["R"]) for x in rows if x[3]["R"] and x[4]["R"]]
    ratA = [(x[4]["A"] / x[3]["A"]) for x in rows if x[3]["A"] and x[4]["A"]]
    for rule in "SB":
        rr = [(x[4]["R"] / x[3]["R"]) for x in rows if x[1] == rule and x[3]["R"] and x[4]["R"]]
        ra = [(x[4]["A"] / x[3]["A"]) for x in rows if x[1] == rule and x[3]["A"] and x[4]["A"]]
        only3 = sum(1 for x in rows if x[1] == rule and x[4]["R"] and not x[3]["R"])
        only1 = sum(1 for x in rows if x[1] == rule and x[3]["R"] and not x[4]["R"])
        lower = sum(1 for x in rows if x[1] == rule and x[6] < x[5] - 2)
        P("5.2", f"rule {rule}: {sum(1 for x in rows if x[1] == rule)} matched runs; updates to 80 at 0.003 over 0.001 on R: median {np.median(rr) if rr else float('nan'):.2f} "
                 f"(range {min(rr) if rr else 0:.2f}-{max(rr) if rr else 0:.2f}); on A median {np.median(ra) if ra else float('nan'):.2f}; reached only at 0.003: {only3}; only at 0.001: {only1}; "
                 f"ending more than 2 points lower at 0.003: {lower}")
    dv = [(r["key"], r["diverged"]) for r in s2["cur"] if r["group"] == "E52" and r["diverged"]]
    P("5.3", f"E52 runs marked diverged: {len(dv)} {dv[:4]}")
    hi = [r for r in s2["cur"] if r["group"] == "E52" and r["lr"] == 0.01]
    P("5.3", f"at 0.01: {len(hi)} runs; reached 80 on R in {sum(full[r['key']]['R'] is not None for r in hi)}; level of R at the end below 80 in {sum(np.mean(r['R'][-3:]) < 80 for r in hi)}")
    return rows


# ------------------------------------------------------------------------------------------------ section 6: screening
def section6(runs, s2, cm):
    full, cur = s2["full"], s2["cur"]
    pool = [r for r in cur if r["cells"] == 256 and r["dose"] >= 49152 and r["group"] in ("E50", "E54", "E57")]
    P("6.1", f"256-cell runs of 49,152 updates on record (E50, E54, E57): {len(pool)}; level of R and T at 8,192, 12,288 and 16,384 against the reading of the full run:")
    for r in sorted(pool, key=lambda r: (full[r["key"]]["R"] or 10**9)):
        P("6.1", f"  {r['key']} {r['rule']}: R {fmt(level(r['steps'], r['R'], 8192), 1)} / {fmt(level(r['steps'], r['R'], 12288), 1)} / {fmt(level(r['steps'], r['R'], 16384), 1)}; "
                 f"T {fmt(level(r['steps'], r['T'], 8192), 1)} / {fmt(level(r['steps'], r['T'], 16384), 1)}; to 80 on R {fmt(full[r['key']]['R'])}, on A {fmt(full[r['key']]['A'])}; R at end {np.mean(r['R'][-3:]):.1f}")
    # does the exact gradient bound the sweep? matched pairs where the sweep reaches and the exact gradient does not
    by = defaultdict(dict)
    for r in cur:
        by[(r["group"], r["arm"], r["seed"], r["lr"], r["dose"])][r["rule"]] = r
    sb = [(k, full[v["S"]["key"]]["R"], full[v["B"]["key"]]["R"]) for k, v in by.items() if "S" in v and "B" in v]
    P("6.2", f"matched pairs (same arm, seed, rate, dose) under both rules: {len(sb)}; sweep reaches 80 on R and exact gradient does not: {sum(1 for x in sb if x[1] and not x[2])}; "
             f"exact gradient first or alone: {sum(1 for x in sb if x[2] and (not x[1] or x[2] <= x[1]))}")
    e57 = [r for r in cur if r["key"] == "e57/e57_in16_row_S_u49152_73129"]
    e50 = [r for r in cur if r["key"] == "e50/e50_C16-in_B_73129"]
    e49 = [r for r in cur if r["key"] == "e49/e49_in16_B_73129"]
    if e57:
        P("6.2", f"for comparison (a learned input code with 16 values a cell, not E50's fixed code): E57 in16 row, sweep, 80 on R at {fmt(full[e57[0]['key']]['R'])}; "
                 f"E49 in16, exact gradient, {fmt(full[e49[0]['key']]['R']) if e49 else '-'} within 24,576; "
                 f"E50 C16-in (fixed code), exact gradient, {fmt(full[e50[0]['key']]['R']) if e50 else '-'} (R at end {np.mean(e50[0]['R'][-3:]):.1f})")
    # screening of items 2 to 4: a two-value first stage of the grid, and the stop rules; CPU-h a run from section 1's model and E47's receipts
    f, aa, bb, F, srat = cm["f"], cm["aa"], cm["bb"], cm["F"], cm["srat"]
    rowc = cm["rowc"]
    h = lambda fw, rule, upd=49152, rows=193: ((aa + bb * fw / 1e6) * (srat if rule == "S" else 1.0) * upd + rowc * fw / F * rows) / 3600
    r2 = {Rx: sum(h(F + (Rx - 1) * f["cells"], k) for k in "SB") for Rx in (2, 4, 8)}
    P("6.3", f"item 2, CPU-h for both rules: R = 2 {r2[2]:.1f}, R = 4 {r2[4]:.1f}, R = 8 {r2[8]:.1f}; R = 4 run only when R = 8 reaches and R = 2 does not: "
             f"saves {r2[4]:.1f} CPU-h ({r2[4] / (r2[2] + 2 * r2[4] + r2[8] + sum(h(F * 26 / 22, k) for k in 'SB')):.2f} of the job with its two controls) in the other outcomes")
    r3 = {"i": sum(h(F - f["l2"], k) for k in "SB"), "iv": sum(h(f["input"] + f["cells"] + 256 * 257, k) for k in "SB")}
    P("6.3", f"item 3, CPU-h for both rules: an arm without l2 {r3['i']:.1f} (arms i, ii, iii alike as now computed), arm iv {r3['iv']:.1f}; arm ii (32 cells) run only when arm i reaches "
             f"and arm iii does not: saves {r3['i']:.1f} CPU-h in the other outcomes")
    e47 = [r for r in runs if r["group"] == "E47" and r["arm"] == "one" and r["complete"]]
    if e47:
        cu = np.mean([r["cpu_up"] / r["steps"][-1] for r in e47]); cs = np.mean([r["cpu_sc"] / len(r["steps"]) for r in e47])
        bu = np.median([r["cpu_up"] / r["steps"][-1] for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
        bs = np.median([r["cpu_sc"] / len(r["steps"]) for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
        c4 = {Rx: 2 * ((bu + Rx * (cu - bu)) * 24576 + (bs + Rx * (cs - bs)) * 97) / 3600 for Rx in (4, 8)}
        P("6.3", f"item 4, CPU-h for both rules at 24,576 updates (if the cells' part grows with R): R = 4 {c4[4]:.1f}, R = 8 {c4[8]:.1f}; "
                 f"R = 4 first and R = 8 only if R = 4 leaves the outer rings unused: saves {c4[8]:.1f} CPU-h when R = 4 already uses them; "
                 f"R = 8 first and R = 4 only if R = 8 uses them: saves {c4[4]:.1f} when R = 8 does not")


def section8(runs, s2):
    full = s2["full"]
    for key in ("e50/e50_C46-one_S_73129", "e50/e50_C46-one_B_73129", "e50/e50_C16-in_S_73129", "e50/e50_C16-in_B_73129",
                "e42/e42_G_S_73129", "e42/e42_G_B_73129", "e47/e47_32_one_S_73129", "e47/e47_32_one_B_73129"):
        r = [x for x in s2["cur"] if x["key"] == key]
        if r:
            r = r[0]
            P("8.1", f"reference {key}: 80 on R at {fmt(full[key]['R'])}, on A at {fmt(full[key]['A'])}; R at 24,576 {fmt(level(r['steps'], r['R'], 24576), 1)}, "
                     f"at the end ({int(r['steps'][-1]):,}) {np.mean(r['R'][-3:]):.1f}; A at the end {np.mean(r['A'][-3:]):.1f}")


# ------------------------------------------------------------------------------------------------ section 9: the nine jobs
JOBS = [  # (order's item, name, runs, CPU-h as counted in the order, dose, typical forward a position, share of arms expected to learn: unknown)
    (2, "R = 2, 4, 8 exchanges on the input with 16 values a cell, two controls, both rules", 10, 26.0),
    (3, "the readout made local on the record's network, four arms, both rules", 8, 10.0),
    (4, "R = 4 and 8 on E47's network of 1,024 cells, both rules", 4, 26.0),
    (5, "the assembled network", 10, 25.0),
    (6, "requirement A, first measurement (4,096 updates of a second family)", 8, 8.0),
    (7, "a local rule, E = 1, 2, 4, 8, two forms, and two reference runs", 10, 20.0),
    (8, "items per change 64, 8, 1; state not reset; plain steps", 8, 25.0),
    (9, "requirement B: fewer pairs, three rates, local rule and dense reference", 15, 35.0),
    (10, "each cell sets its own number of connections", 4, 10.0),
]


def lpt(costs, lanes=3):
    t = [0.0] * lanes
    for c in sorted(costs, reverse=True):
        t[t.index(min(t))] += c
    return max(t)


def section9(cm, s2, runs):
    f, aa, bb, F, srat, rowc = cm["f"], cm["aa"], cm["bb"], cm["F"], cm["srat"], cm["rowc"]
    sc_now = sum(scoring_work(f).values())
    sc_new = sum(scoring_work(f, incremental=True, lookup=True, dedupe=True).values())
    sc_factor = sc_new / sc_now
    # adaptive rows: about this share of rows kept on the record's 49,152-update runs
    pool = [r for r in s2["cur"] if r["cells"] == 256 and r["dose"] == 49152]
    full = s2["full"]
    sav = []
    for r in pool:
        fu = full[r["key"]]
        if fu["R"] is not None and fu["A"] is not None:
            sav.append(max(0.0, 1 - max(max(fu["R"], fu["A"]) + 10 * ROW, 24576) / r["steps"][-1]))
    es = float(np.mean(sav))
    es_all = float(np.sum(sav) / len(pool))
    P("9.0", f"early stop (settled + 8 rows, not before 24,576) on the record's 256-cell runs of 49,152 updates: {len(sav)} of {len(pool)} reach both; "
             f"share of updates saved: {es:.3f} in a run that reaches, {es_all:.3f} over all {len(pool)}; scoring work after the scorer's changes {sc_factor:.3f} of now")
    # item-specific per-run costs (present code, from section 1's model) and the protocol's
    def run_h(fw, rule, updates=49152, rows=193, scf=1.0, rowfrac=1.0):
        t = (aa + bb * fw / 1e6) * (srat if rule == "S" else 1.0)
        return (t * updates + rowc * fw / F * scf * rows * rowfrac) / 3600
    item2 = []
    for Rx in (2, 4, 8, 4, 1):
        fw = F + (Rx - 1) * f["cells"]
        if Rx == 1:
            fw = F  # control 2 has four neutral bytes: about 26 positions in place of 22
            fw = F * 26 / 22
        for rule in "SB":
            item2.append((run_h(fw, rule), run_h(fw, rule, scf=sc_factor, rowfrac=ROWS_KEPT)))
    item3 = []
    for fw in (F - f["l2"], F - f["l2"], F - f["l2"], f["input"] + f["cells"] + 256 * 257):
        for rule in "SB":
            item3.append((run_h(fw, rule), run_h(fw, rule, scf=sc_factor, rowfrac=ROWS_KEPT)))
    P("9.1", f"item 2 by the model, present protocol: {sum(c[0] for c in item2):.1f} CPU-h (order: 26); with the scorer's changes and adaptive rows: {sum(c[1] for c in item2):.1f}")
    P("9.1", f"item 3 by the model, present protocol: {sum(c[0] for c in item3):.1f} CPU-h (order: 10); with the scorer's changes and adaptive rows: {sum(c[1] for c in item3):.1f}")
    e47 = [r for r in runs if r["group"] == "E47" and r["arm"] == "one" and r["complete"]]
    cu = np.mean([r["cpu_up"] / r["steps"][-1] for r in e47]); cs = np.mean([r["cpu_sc"] / len(r["steps"]) for r in e47])
    bu = np.median([r["cpu_up"] / r["steps"][-1] for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
    bs = np.median([r["cpu_sc"] / len(r["steps"]) for r in runs if r["group"] == "E42" and r["arm"] == "F10" and r["complete"]])
    item4 = []
    for Rx in (4, 8):
        for rule in "SB":
            u = bu + Rx * (cu - bu); s = bs + Rx * (cs - bs)
            now = (u * 24576 + s * 97) / 3600
            new = (u * 24576 + s * sc_factor * 97 * ROWS_KEPT) / 3600
            item4.append((now, new))
    P("9.1", f"item 4 by E47's receipts, 24,576 updates, if the cells' part grows with R: present protocol {sum(c[0] for c in item4):.1f} CPU-h (order: 26); "
             f"with the scorer's changes and adaptive rows {sum(c[1] for c in item4):.1f}; at 49,152 updates about twice")
    P("9.2", "job | runs | CPU-h as counted | protocol, no run reaching (scorer + rows) | protocol, every run reaching (and stopped) | wall hours on one line (3 processes), as counted -> protocol")
    for item, name, n, h in JOBS:
        if item == 2:
            per_now = [c[0] for c in item2]; per_new = [c[1] for c in item2]
        elif item == 3:
            per_now = [c[0] for c in item3]; per_new = [c[1] for c in item3]
        elif item == 4:
            per_now = [c[0] for c in item4]; per_new = [c[1] for c in item4]
        else:
            per_now = [h / n] * n
            sc_share = 0.16
            per_new = [h / n * (1 - sc_share + sc_share * sc_factor * ROWS_KEPT)] * n
        scale = h / sum(per_now)     # keep the order's count as the present figure; apply the protocol's ratios to it
        new_none = sum(per_new) * scale
        new_all = new_none * (1 - es) if item not in (4, 6) else new_none     # item 4 runs 24,576 updates and item 6 4,096: the stop never comes earlier
        P("9.2", f"item {item} ({name}) | {n} | {h:.0f} | {new_none:.1f} | {new_all:.1f} | {lpt([c * scale for c in per_now]):.1f} -> {lpt([c * scale for c in per_new]):.1f}")


def main():
    runs, other = load()
    cm = section1(runs, other)
    s2 = section2(runs, cm)
    section3(runs, s2)
    section4(runs, s2, cm)
    section5(runs, s2)
    section6(runs, s2, cm)
    section7(cm, runs)
    section8(runs, s2)
    section9(cm, s2, runs)


if __name__ == "__main__":
    main()
