"""E53: the tables and readings of the brief's "How the result is stated", from the receipts only.
`first_row`, `describe` and `use_form` are copied unchanged from e47_trial_read.py on origin/claude/e47-worker-stage-0-3d0ya6 at 578a877b
(which copies E42's e42_tables.describe). The trial's receipts and weight files are read with git show into exp45/e53_trial_files/,
which is not committed.
    python3 exp45/e53/e53_tables.py        (from MOD97_LOCAL_RULE_20261106; standard library and torch for the trial's ring sizes)"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE.parent.parent
TRIAL = HERE.parent / "e53_trial_files"
W7 = "origin/claude/e47-worker-stage-0-3d0ya6"
SEED = 73129
DOSE = 24576
D = DOSE - 512                                                       # 24,064: the last update at which three rows can begin
NEW = ("x3", "wd0", "x3-wd0", "xf3", "xf3-wd0")
GROUPS = ("interface", "ring1", "ring2", "ring3")


# ---------------------------------------------------------------- copied from e47_trial_read.py (578a877b), unchanged
def first_row(curve, key, th, version):                              # E42's e42_tables.first_row
    v = [(r["step"], r[key]) for r in curve]
    for i, (u, a) in enumerate(v):
        if version == "single" and a >= th:
            return i
        if version == "three" and i + 2 < len(v) and all(x >= th for _, x in v[i:i + 3]):
            return i
    return None


def describe(xr, xc, Dr, Dc):                                         # E42's e42_tables.describe (E41's table of four cases)
    if xr is not None and xc is not None:
        q = xc / xr
        return "fewer" if q < 2 / 3 else "more" if q > 1.5 else "about the same"
    if xr is not None:
        return "more" if Dc / xr >= 1.5 else "not decided"
    if xc is not None:
        return "fewer" if xc / Dr <= 2 / 3 else "not decided"
    return "neither reaches"


def use_form(fall_off_A, fall_off_R, fall_mean_A, fall_mean_R):
    """The brief's forms, the first that applies."""
    if fall_mean_A >= 15 or fall_mean_R >= 15:
        return "used"
    if fall_off_A < 5 and fall_off_R < 5:
        return "not used"
    if (fall_off_A >= 15 or fall_off_R >= 15) and fall_mean_A < 5 and fall_mean_R < 5:
        return "a constant is used"
    return "partly"
# ---------------------------------------------------------------- end of the copy


def trial(rel):
    p = TRIAL / Path(rel).name
    if not p.exists():
        TRIAL.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.run(["git", "show", f"{W7}:MOD97_LOCAL_RULE_20261106/{rel}"], cwd=M, capture_output=True, check=True).stdout)
    return p


def load(arm, rule):
    if arm == "one":
        return json.loads(trial(f"exp45/e47/e47_32_one_{rule}_{SEED}.json").read_text())
    p = HERE / f"e53_32_{arm}_{rule}_{SEED}.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return r if r.get("complete") else None


def to80(c, key):
    i = first_row(c, key, 80, "three")
    return None if i is None else c[i]["step"]


def summary(rec):
    c = rec["curve"]
    last3 = c[-3:]
    lv = {k: sum(r[k] for r in last3) / 3 for k in ("A", "R", "len40")}
    last = c[-1]
    fo = {k: last[k] - last["far_off"][k] for k in ("A", "R")}
    fm = {k: last[k] - last["far_mean"][k] for k in ("A", "R")}
    return dict(A80=to80(c, "A"), R80=to80(c, "R"), level=lv, read=lv["A"] >= 80 and lv["R"] >= 80, fall_off=fo, fall_mean=fm,
                far=use_form(fo["A"], fo["R"], fm["A"], fm["R"]), last=dict(A=last["A"], R=last["R"]),
                off=dict(A=last["far_off"]["A"], R=last["far_off"]["R"]), mean=dict(A=last["far_mean"]["A"], R=last["far_mean"]["R"]),
                rings={u: next((r.get("ring_magnitudes") for r in c if r["step"] == u), None) for u in (0, 4096, DOSE)},
                loss_finite=all(r["finite_losses"] for r in c), resumes=[x["at_update"] for x in rec.get("resumes", [])])


def trial_rings(rule):
    """The trial's ring sizes at its start and at its end, from its committed weight files (the same computation as ring_magnitudes)."""
    import torch as T
    sys_path = str(M)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import e47_local as L47
    ring = T.tensor([-(-L47.distance(i, 32) // L47.RADIUS) for i in range(1024)])
    out = {}
    for u, f in ((0, f"exp45/e47/e47_32_one_{rule}_{SEED}_start.pt"), (DOSE, f"exp45/e47/e47_32_one_{rule}_{SEED}.pt")):
        w = T.load(trial(f))["Wc"].abs()
        out[u] = {g: float(w[:, ring == k].mean()) for k, g in enumerate(GROUPS)}
    return out


def f1(v):
    return "–" if v is None else f"{v:.1f}"


def main():
    S = {}
    for rule in "SB":
        for arm in ("one",) + NEW:
            r = load(arm, rule)
            if r is not None:
                S[(arm, rule)] = summary(r)
    print("## Runs (updates to 80: first of three consecutive rows at or above 80; level: mean of the last three rows)\n")
    print("| Rule | Arm | Updates to 80 on A | Updates to 80 on R | Level A | Level R | Level length 40 | Read | Losses finite | Resumed at |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (arm, rule), s in S.items():
        print(f"| {rule} | {arm} | {s['A80'] or 'not reached'} | {s['R80'] or 'not reached'} | {f1(s['level']['A'])} | {f1(s['level']['R'])} | "
              f"{f1(s['level']['len40'])} | {'read' if s['read'] else 'not read'} | {s['loss_finite']} | {', '.join(map(str, s['resumes'])) or '–'} |")
    print("\n## Learning, in the words of `describe` (candidate against reference; 24,064 as the last possible start)\n")
    print("| Rule | Candidate | Reference | A: updates | A: words | R: updates | R: words |\n|---|---|---|---|---|---|---|")
    pairs = [(a, "one") for a in NEW] + [("x3-wd0", "x3"), ("x3-wd0", "wd0")]
    words = {}
    for rule in "SB":
        for cand, ref in pairs:
            if (cand, rule) in S and (ref, rule) in S:
                c, r = S[(cand, rule)], S[(ref, rule)]
                wa, wr = describe(r["A80"], c["A80"], D, D), describe(r["R80"], c["R80"], D, D)
                words[(rule, cand, ref)] = (wa, wr)
                print(f"| {rule} | {cand} | {ref} | {c['A80'] or '–'} / {r['A80'] or '–'} | {wa} | {c['R80'] or '–'} / {r['R80'] or '–'} | {wr} |")
    print("\n## The far cells as a whole (the two diagnostics at the last row; the forms of E47's brief, by `use_form`)\n")
    print("| Rule | Arm | A plain / far at zero / far at mean | R plain / far at zero / far at mean | Fall A zero, mean | Fall R zero, mean | Form |")
    print("|---|---|---|---|---|---|---|")
    for (arm, rule), s in S.items():
        print(f"| {rule} | {arm} | {f1(s['last']['A'])} / {f1(s['off']['A'])} / {f1(s['mean']['A'])} | {f1(s['last']['R'])} / {f1(s['off']['R'])} / {f1(s['mean']['R'])} | "
              f"{f1(s['fall_off']['A'])}, {f1(s['fall_mean']['A'])} | {f1(s['fall_off']['R'])}, {f1(s['fall_mean']['R'])} | {s['far']} |")
    print("\n## The size of the weights by ring (mean absolute recurrent weight into the cells of a group) at the first row, at update 4,096 and at the last row\n")
    print("| Rule | Arm | Update | Interface | Ring 1 | Ring 2 | Ring 3 |\n|---|---|---|---|---|---|---|")
    for rule in "SB":
        tr = trial_rings(rule)
        for u in (0, DOSE):
            print(f"| {rule} | one (from its weight files) | {u} | " + " | ".join(f"{tr[u][g]:.3f}" for g in GROUPS) + " |")
        print(f"| {rule} | one | 4096 | not recorded by the trial | | | |")
        for arm in NEW:
            s = S.get((arm, rule))
            if s:
                for u in (0, 4096, DOSE):
                    print(f"| {rule} | {arm} | {u} | " + " | ".join(f"{s['rings'][u][g]:.3f}" for g in GROUPS) + " |")
    # the condition of the conditional run
    print("\n## The condition of the conditional run (under back-propagation)\n")
    cond = {}
    for arm in ("x3", "x3-wd0"):
        s = S.get((arm, "B"))
        if s is None:
            cond[arm] = None; continue
        not_read = not s["read"]
        more = words.get(("B", arm, "one"), (None,))[0] == "more"
        cond[arm] = dict(not_read=not_read, more_updates_to_80_on_A=more, A80=s["A80"], trial_A80=S[("one", "B")]["A80"], qualifies=not_read or more)
        print(f"- {arm} B: not read {not_read}; updates to 80 on A {s['A80']} against the trial's {S[('one', 'B')]['A80']} ({words.get(('B', arm, 'one'), ('–',))[0]}); qualifies: {not_read or more}")
    q = [a for a, v in cond.items() if v and v["qualifies"]]
    print(f"- the conditional run: {'xf3 B (decay of x3)' if 'x3' in q else 'xf3-wd0 B' if q else 'not made'}")
    # predictions scored by the worker
    print("\n## Predictions P53.2 to P53.5, P53.8 to P53.10 and P53.12 (P53.1 from the checks; P53.6, P53.7, P53.11 are the reviewer's)\n")
    g = lambda a, r: S.get((a, r))
    out = []
    if g("wd0", "S") and g("wd0", "B"):
        out.append(("P53.2", "wd0 reaches 80 on A under both rules", g("wd0", "S")["A80"] is not None and g("wd0", "B")["A80"] is not None))
        out.append(("P53.3", "wd0 against one: about the same on A under both rules",
                    words.get(("S", "wd0", "one"), ("",))[0] == "about the same" and words.get(("B", "wd0", "one"), ("",))[0] == "about the same"))
    if g("x3-wd0", "B"):
        out.append(("P53.4", "x3-wd0 reaches 80 on A under back-propagation", g("x3-wd0", "B")["A80"] is not None))
    if g("x3-wd0", "S"):
        out.append(("P53.5", "x3-wd0 reaches 80 on A under the sweep", g("x3-wd0", "S")["A80"] is not None))
    news = [(rule, a) for rule in "SB" for a in NEW if (a, rule) in S]
    if len(news) >= 6:
        fewer = [f"{a} {rule}" for rule, a in news if words.get((rule, a, "one"), ("",))[0] == "fewer"]
        out.append(("P53.8", "no new run is fewer than the trial's on updates to 80 on A", not fewer, fewer))
    if g("x3", "B"):
        out.append(("P53.9", "x3 reaches 80 on A under back-propagation", g("x3", "B")["A80"] is not None))
    if g("x3", "S"):
        out.append(("P53.10", "x3 reaches 80 on A under the sweep", g("x3", "S")["A80"] is not None))
    if g("x3", "S") and g("x3", "B"):
        r3 = [g("x3", r)["rings"][4096]["ring3"] for r in "SB"]
        out.append(("P53.12", "in x3 the mean absolute weight into ring 3 at update 4,096 is at least 0.30 under both rules", all(v >= 0.30 for v in r3),
                    f"S {r3[0]:.3f}, B {r3[1]:.3f}"))
    for o in out:
        print(f"- {o[0]} ({o[1]}): {'true' if o[2] else 'false'}" + (f" ({o[3]})" if len(o) > 3 else ""))


if __name__ == "__main__":
    main()
