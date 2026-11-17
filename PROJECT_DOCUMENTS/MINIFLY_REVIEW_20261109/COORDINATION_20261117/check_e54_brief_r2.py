import json, subprocess, sys
D, W, M = "origin/codex/minifly-consolidated-20261023", "origin/claude/gallant-dijkstra-i9zgsr", "MOD97_LOCAL_RULE_20261106"
SEEDS = (73129, 93683, 116089)
def raw(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{M}/{path}"], capture_output=True, check=True).stdout
show = lambda ref, path: json.loads(raw(ref, "exp45/" + path))
e41 = lambda rule, H, fp, s: show(D, f"e41/e41_{rule}_h{H}_f{fp}_{s}.json")
e42 = lambda cfg, rule, s: show(W, f"e42/e42_{cfg}_{rule}_{s}.json")
part = sys.argv[1] if len(sys.argv) > 1 else "all"
ROW = 256
def mean_rows(c, key, at, n):                      # mean of `key` over the n scored rows that end at update `at`
    rows = [r[key] for r in c if at - (n - 1) * ROW <= r["step"] <= at]; assert len(rows) == n, (at, len(rows))
    return sum(rows) / n
m3 = lambda c, at, key="R": mean_rows(c, key, at, 3)
m9 = lambda c, at: mean_rows(c, "R", at, 9)
rise9 = lambda c, at: m9(c, at) - m9(c, at - 8192)  # the brief's rule: flattened if this is below 3
def to80(c, key):
    v = [r[key] for r in c]
    return next((c[i]["step"] for i in range(len(v) - 2) if min(v[i:i + 3]) >= 80), None)

if part in ("all", "paths"):                       # every hash and path that the six checks cite
    r = show(W, "e42_qual/check8_73129_runs/e42_F10_S_dry2048_73129.json")
    print("check 1, F10 S dry 2,048:", r["tensors_sha256"][:8], "dose", r["dose"], "rows", len(r["curve"]), "complete", r["complete"])
    r = show(W, "e42_qual/check7_73129_runs/rep0/e42_P8_S_dry1024_73129.json")
    print("check 1, P8 S dry 1,024:", r["tensors_sha256"][:8], "theta", r["theta"], "replaced by row", [x["replaced_since_previous_row"] for x in r["curve"]],
          "has structure digests", sorted(r["structure_end_sha256"]))
    print("theta in e42_P8_S_73129.json:", e42("P8", "S", 73129)["theta"], "; chosen_theta in check10_73129.json:", show(W, "e42_qual/check10_73129.json")["chosen_theta"])
    st = show(W, "e42_qual/ckpt_selftest.json")
    print("ckpt_selftest.json: pass_", st["pass_"], "; cases", {k: (v["tensors_equal"], v["curve_equal"], v["digest_equal"], v["resumed_at"]) for k, v in st["selftest"].items()})
    for fp in (25, 10):
        for rule in "SB":
            for H in (512, 256):
                r = e41(rule, H, fp, 73129); d4 = next(x["data_digest"]["sha256"] for x in r["curve"] if x["step"] == 4096)
                print(f"e41_{rule}_h{H}_f{fp}_73129.json: dose {r['dose']}, rows {len(r['curve'])}, digest at 4,096 {d4[:16]}, tensors {r['tensors_sha256'][:8]}, row keys {sorted(r['curve'][1])}")
    src = raw(D, "exp42_cells.py").decode().splitlines()
    print("exp42_cells.py lines 414-415:", src[413].strip(), src[414].strip())
    for ref, p in ((W, "exp45/e42/interrupted"), (W, "exp45/e42/ckpt"), (W, "exp45/e42/selftest_case.sh"), (D, "exp45/e42"), (D, "exp45/e43/addendum2/e42_start/e42_P8_S_73129.json")):
        q = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{M}/{p}"], capture_output=True, text=True)
        q2 = subprocess.run(["git", "cat-file", "-t", f"{ref}:{M}/{p}"], capture_output=True, text=True)
        print(f"{ref.split('/')[-1][:12]}:{p}:", q2.stdout.strip() or "ABSENT", (q.stdout.split()[:6] if q.returncode == 0 else ""))
    wts = subprocess.run(["git", "ls-tree", "-r", "-l", f"{W}:{M}/exp45/e42"], capture_output=True, text=True).stdout.splitlines()
    pt = [l for l in wts if l.endswith(".pt")]
    print("E42's committed .pt files:", len(pt), "; total bytes", sum(int(l.split()[3]) for l in pt), "; examples", [l.split()[-1] for l in pt[:3]])
    n = 0
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                r = e42(cfg, rule, s); ck = r.get("checkpointing")
                if ck and ck["resumed"]:
                    n += 1; print(f"E42 {cfg} {rule} {s}: resumed at", [(x["at_update"], round(x["cpu_replay"], 1)) for x in ck["resumed"]])
    print("E42 receipts of G, F10, P8 with a resumed entry:", n, "of 18")

if part in ("all", "flat"):                        # the rule 'has flattened' applied to the curves on record
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            print(f"E42 {cfg} {rule}: nine-row rise at 24,576", [round(rise9(e42(cfg, rule, s)["curve"], 24576), 2) for s in SEEDS],
                  "; three-row", [round(m3(e42(cfg, rule, s)["curve"], 24576) - m3(e42(cfg, rule, s)["curve"], 16384), 2) for s in SEEDS])
    for H, fp in ((512, 25), (512, 10), (256, 25), (256, 10)):
        for rule in "SB":
            cs = [e41(rule, H, fp, s)["curve"] for s in SEEDS]; d = cs[0][-1]["step"]
            print(f"E41 H={H} f=0.{fp} {rule}: nine-row rise at {d}", [round(rise9(c, d), 2) for c in cs], "; three-row", [round(m3(c, d) - m3(c, d - 8192), 2) for c in cs],
                  "; level", [round(m3(c, d), 1) for c in cs])
    # the rule applied at earlier updates of the long curves, beside what the level did afterwards
    print("the rule applied before the end of a curve: (update, nine-row rise, level then, level at the end)")
    for H, fp, rule in ((256, 10, "B"), (256, 25, "S"), (256, 25, "B"), (256, 10, "S")):
        for s in SEEDS:
            c = e41(rule, H, fp, s)["curve"]; d = c[-1]["step"]
            print(f"  E41 H={H} f=0.{fp} {rule} {s}:", [(u, round(rise9(c, u), 1), round(m3(c, u), 1), round(m3(c, d), 1)) for u in range(12288, d + 1, 4096)])
    for cfg, rule in (("F10", "S"), ("F10", "B"), ("G", "S")):
        for s in SEEDS:
            c = e42(cfg, rule, s)["curve"]
            print(f"  E42 {cfg} {rule} {s}:", [(u, round(rise9(c, u), 1), round(m3(c, u), 1), round(m3(c, 24576), 1)) for u in range(12288, 24577, 4096)])

if part in ("all", "slow"):                        # a learner that follows the dense curve k times more slowly
    def slowed(c, k, dose):                        # R at update u is the dense curve's R at u / k (linear between rows)
        R = {r["step"]: r["R"] for r in c}
        def at(u):
            x = u / k; lo = int(x // ROW) * ROW; hi = min(lo + ROW, max(R))
            return R[lo] + (R[hi] - R[lo]) * (x - lo) / ROW if hi > lo else R[lo]
        return [dict(step=u, R=at(u)) for u in range(0, dose + 1, ROW)]
    for s in SEEDS:
        c = e41("B", 256, 10, s)["curve"]
        for k in (1.5, 2.0, 3.0):
            sc = slowed(c, k, 49152)
            print(f"dense, half width, f=0.10, B, seed {s}, a learner {k} times slower, at update 49,152: level {m3(sc, 49152):.1f}, dense level {m3(c, 40960):.1f} at 40,960, "
                  f"nine-row rise {rise9(sc, 49152):.1f} -> {'has flattened' if rise9(sc, 49152) < 3 else 'has not flattened'}")
    for s in SEEDS:
        c = e41("S", 256, 25, s)["curve"]
        for k in (1.5, 2.0):
            sc = slowed(c, k, 49152)
            print(f"dense, half width, f=0.25, S, seed {s}, a learner {k} times slower, at update 49,152: level {m3(sc, 49152):.1f}, dense level at the same update of its own run "
                  f"{m3(c, 40960):.1f} (40,960), nine-row rise {rise9(sc, 49152):.1f}")

if part in ("all", "cost"):
    tot = {}
    for s in SEEDS:
        h = {(cfg, rule): (lambda r: (r["cpu_updates"] + r["cpu_scoring"] + r["cpu_replacement"]) / 3600)(e42(cfg, rule, s)) for cfg in ("G", "F10", "P8") for rule in "SB"}
        # at 49,152 updates: updates and replacement doubled, 193 scored rows for 97
        h2 = {}
        for cfg in ("G", "F10", "P8"):
            for rule in "SB":
                r = e42(cfg, rule, s); h2[(cfg, rule)] = (2 * r["cpu_updates"] + r["cpu_scoring"] * 193 / 97 + 2 * r["cpu_replacement"]) / 3600
        p4 = 0.0
        for rule in "SB":
            r = e41(rule, 256, 10, s); pu, ps = r["cpu_updates"] / r["dose"], r["cpu_scoring"] / len(r["curve"])
            p4 += (122880 * pu + 481 * ps) / 3600
            if s == 73129: print(f"E41 h256 f10 {rule} 73129: {pu:.4f} CPU-s an update, {ps:.2f} a scored row; 122,880 updates and 481 rows: {(122880 * pu + 481 * ps) / 3600:.2f} CPU-h")
        tot[s] = (sum(h.values()), sum(h2.values()), p4)
        print(f"seed {s}: six runs at 24,576: {sum(h.values()):.2f} CPU-h {[round(v, 2) for v in h.values()]}; at 49,152: {sum(h2.values()):.2f}; twice the first: {2 * sum(h.values()):.2f}; part 4: {p4:.2f}; "
              f"three parts + part 4 + 0.6: {3 * sum(h2.values()) + p4 + 0.6:.1f} (with twice: {6 * sum(h.values()) + p4 + 0.6:.1f}; with part 4 at 3.3: {6 * sum(h.values()) + 3.3 + 0.6:.1f})")
    r = {(cfg, rule): e42(cfg, rule, 73129) for cfg in ("G", "F10", "P8") for rule in "SB"}
    per = lambda x, u: (x["cpu_updates"] + x["cpu_replacement"]) / 24576 * u + x["cpu_scoring"] / 97 * (u // 256 + 1)
    chk = 2 * per(r[("F10", "S")], 2048) + 2 * per(r[("P8", "S")], 1024) + per(r[("G", "S")], 2048) + per(r[("G", "B")], 2048) + per(r[("P8", "S")], 2048) + per(r[("P8", "S")], 2048)
    print(f"the checks (two F10 S of 2,048, two P8 S of 1,024, G S and G B of 2,048, two P8 S of 2,048): {chk / 3600:.2f} CPU-h")
    print("longest single run at 49,152 on seed 73129's rates:", round(max((2 * x["cpu_updates"] + x["cpu_scoring"] * 193 / 97 + 2 * x["cpu_replacement"]) / 3600 for x in r.values()), 2), "CPU-h")

if part in ("all", "record"):
    g = {rule: e42("G", rule, 73129)["curve"] for rule in "SB"}
    for cfg in ("F10", "P8"):
        for rule in "SB":
            c = e42(cfg, rule, 73129)["curve"]
            print(f"seed 73129, {cfg} {rule}: ratio to G on A {to80(c, 'A') / to80(g[rule], 'A'):.3f}, on R {to80(c, 'R') / to80(g[rule], 'R'):.3f}")
    for H, fp in ((512, 25), (256, 25), (512, 10), (256, 10)):
        S = [m3(e41("S", H, fp, s)["curve"], e41("S", H, fp, s)["dose"]) for s in SEEDS]; B = [m3(e41("B", H, fp, s)["curve"], e41("B", H, fp, s)["dose"]) for s in SEEDS]
        print(f"E41 H={H} f=0.{fp}: R under S {[round(x, 1) for x in S]}, under B {[round(x, 1) for x in B]}, B minus S {[round(b - a, 1) for a, b in zip(S, B)]}")
    import math
    print("scatter of a difference of two nine-row means for a row scatter of 2.0 and 4.0:", round(2.0 * math.sqrt(2) / 3, 2), round(4.0 * math.sqrt(2) / 3, 2),
          "; of a difference of two three-row levels:", round(2.0 * math.sqrt(2 / 3), 2), round(4.0 * math.sqrt(2 / 3), 2))
    Dd = 49152 - 512                                # describe's bounds at 49,152 updates (thr_desc: the dose less 512)
    print("describe with one run not reaching 80, both of 49,152 updates: 'more' if G's updates to 80 are at most", max(u for u in range(0, 49153, 256) if Dd / max(u, 1) >= 1.5),
          "; 'fewer' if the network's are at most", max(u for u in range(0, 49153, 256) if u / Dd <= 2 / 3), "; with the dose itself in place of the dose less 512:",
          max(u for u in range(256, 49153, 256) if 49152 / u >= 1.5), max(u for u in range(0, 49153, 256) if u / 49152 <= 2 / 3))
    c = e41("S", 512, 25, 73129)["curve"]
    print("E41 S h512 f25 73129: R of the last 12 rows", [x["R"] for x in c[-12:]], "; rows at or above 80:", sum(x["R"] >= 80 for x in c))
    for s in SEEDS:
        c = e41("S", 256, 25, s)["curve"]; u80 = to80(c, "R")
        after = [x["R"] for x in c if x["step"] >= u80]
        print(f"E41 S h256 f25 {s}: to 80 on R {u80}; afterwards {len(after)} rows, of which below 80: {sum(v < 80 for v in after)}; min {min(after)}, max {max(after)}; level at 40,960 {m3(c, 40960):.1f}")

if part in ("all", "rule2"):                       # the brief's rule and two stricter ones, applied before the end of E41's long curves
    rules = {"brief (one span of 8,192)": lambda c, u: rise9(c, u) < 3,
             "two spans, each under 3": lambda c, u: abs(rise9(c, u)) < 3 and abs(rise9(c, u - 8192)) < 3,
             "8,192 and 16,384, each under 3": lambda c, u: abs(m9(c, u) - m9(c, u - 8192)) < 3 and abs(m9(c, u) - m9(c, u - 16384)) < 3}
    for name, fn in rules.items():
        calls = wrong = 0; ex = []
        for H, fp in ((256, 10), (256, 25)):
            for rule in "SB":
                for s in SEEDS:
                    c = e41(rule, H, fp, s)["curve"]
                    for u in range(20480, 36865, 4096):
                        if fn(c, u):
                            calls += 1; gain = m3(c, 40960) - m3(c, u)
                            if gain >= 5:
                                wrong += 1; ex.append(f"{rule} f=0.{fp} {s} at {u}: {m3(c, u):.1f} then {m3(c, 40960):.1f}")
        print(f"{name}: 'has flattened' at {calls} of 60 points (12 half-width curves, updates 20,480 to 36,864); the level rose by 5 points or more afterwards in {wrong}: {ex}")
    for name, fn in rules.items():
        print(f"{name}: at the end of the curve:",
              {f"E41 h256 f{fp} {rule}": [fn(e41(rule, 256, fp, s)["curve"], 40960) for s in SEEDS] for fp in (25, 10) for rule in "SB"},
              {f"E42 {cfg} {rule}": [fn(e42(cfg, rule, s)["curve"], 24576) for s in SEEDS] for cfg in ("G", "F10", "P8") for rule in "SB"})

if part in ("all", "rule3"):                       # the first check's three-row form of the rule, for comparison
    calls = wrong = 0
    for H, fp in ((256, 10), (256, 25)):
        for rule in "SB":
            for s in SEEDS:
                c = e41(rule, H, fp, s)["curve"]
                for u in range(20480, 36865, 4096):
                    if m3(c, u) - m3(c, u - 8192) < 3:
                        calls += 1; wrong += m3(c, 40960) - m3(c, u) >= 5
    print(f"three-row form (one span of 8,192): 'has flattened' at {calls} of 60 points; the level rose by 5 points or more afterwards in {wrong}")
    # a fall: does any curve on record fall by 3 points or more between nine-row means 8,192 updates apart?
    falls = []
    for H, fp in ((256, 10), (256, 25)):
        for rule in "SB":
            for s in SEEDS:
                c = e41(rule, H, fp, s)["curve"]
                falls += [(rule, fp, s, u, round(rise9(c, u), 1)) for u in range(12288, 40961, 2048) if rise9(c, u) <= -3]
    for cfg in ("G", "F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                c = e42(cfg, rule, s)["curve"]
                falls += [(cfg, rule, s, u, round(rise9(c, u), 1)) for u in range(12288, 24577, 2048) if rise9(c, u) <= -3]
    print("nine-row falls of 3 points or more over 8,192 updates on record:", falls)
    # P8: share of connections replaced an event over the run, seed 73129
    for rule in "SB":
        r = e42("P8", rule, 73129)["curve"]; ev = [x["replaced_since_previous_row"] / 2560 for x in r[1:]]
        print(f"P8 {rule} 73129: share replaced an event, first 16 {sum(ev[:16]) / 16:.3f}, last 16 {sum(ev[-16:]) / 16:.3f}; by quarter of the run", [round(sum(ev[i:i + 24]) / 24, 3) for i in range(0, 96, 24)])

if part in ("all", "split"):                       # the places where a fraction is rounded (exp11_cell.py lines 238 and 240; exp42_cells.py line 219)
    DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
    for f in (0.50, 0.25, 0.15, 0.10):
        seeds = [73129 * 1000 + round(f * 100) + m * 7 + code[0] for _, m, code in DEV]
        print(f"f {f}: f * 100 = {f * 100!r}, round {round(f * 100)}, int {int(f * 100)}; taught pairs an operation {round(f * 7921)}; "
              f"the four shuffles of one run have distinct seeds: {len(set(seeds)) == 4}; reduction sample named e41|73129|{round(f * 100)}|add")
    allseeds = {(f, op): 73129 * 1000 + round(f * 100) + m * 7 + code[0] for f in (0.50, 0.25, 0.15, 0.10) for op, m, code in DEV}
    print("equal shuffle seeds across fractions:", [(a, b) for a in allseeds for b in allseeds if a < b and allseeds[a] == allseeds[b]])
    # the comparisons with G at a half under the brief's one-span rule and a margin of 10 points
    und = 0
    for cfg in ("F10", "P8"):
        for rule in "SB":
            for s in SEEDS:
                g, c = e42("G", rule, s)["curve"], e42(cfg, rule, s)["curve"]; d = m3(c, 24576) - m3(g, 24576)
                fg, fc = rise9(g, 24576) < 3, rise9(c, 24576) < 3
                und += not ((d >= 10 and fg) or (d <= -10 and fc) or (abs(d) < 10 and fg and fc))
    print("E42 at a half, the twelve comparisons with G at update 24,576 by the brief's rule: not decided", und, "of 12")
