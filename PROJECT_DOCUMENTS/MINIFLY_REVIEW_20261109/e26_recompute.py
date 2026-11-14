#!/usr/bin/env python3
"""E26 (births, lives) and E26-X (crossed lives) recomputed from receipts and weight files. Read-only, via git show.

Usage:  python3 -I e26_recompute.py [REF]     default REF: origin/claude/adoring-cori-b07aog
"""
import hashlib, json, statistics as st, subprocess, sys
REF = sys.argv[1] if len(sys.argv) > 1 else "origin/claude/adoring-cori-b07aog"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/"
SEEDS = (73129, 93683, 116089)

def raw(path):
    r = subprocess.run(["git", "show", f"{REF}:{BASE}{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None
def load(path):
    b = raw(path); return json.loads(b) if b is not None else None
def pct(d): return 100 * d["exact"] / d["n"]
def fam(sc, mod, ops=("add", "sub", "max", "min")): return st.mean(pct(sc[f"{o}{mod}"]["12"]) for o in ops)
def pm(sc):
    c = sc["compose"]; return 100 * (c["+-"]["exact"] + c["-+"]["exact"]) / (c["+-"]["n"] + c["-+"]["n"])
def field(d, k):
    c = d.get("complete"); return c[k] if isinstance(c, dict) and k in c else d.get(k)
def file_hash_ok(d, folder):
    name = field(d, "weights"); blob = raw(folder + name) if name else None
    return None if blob is None else hashlib.sha256(blob).hexdigest() == field(d, "weights_sha256")

print("1. E26 births (arm S = sweep, arm B = backprop; both decay 0.25, same written-out step)")
births = {}
for s in SEEDS:
    row = {}
    for a in "SB":
        d = load(f"e26/e26_dev_{a}_{s}.json"); births[(a, s)] = d
        last = d["curve"][-1]["sealed"]
        row[a] = dict(hard=st.mean(pct(last[o]["12"]) for o in ("add89", "sub89")), steps=d["optimizer_steps"], rows=len(d["curve"]),
                      wd=d["optimizer_read_back"]["weight_decay"], local=d["local"], hash=file_hash_ok(d, "e26/"), digest=d["data_digest"]["sha256"],
                      finite=[x.get("finite_losses") for x in d["curve"]], complete=d.get("complete"))
    S, B = row["S"], row["B"]
    print(f"   {s}: hard mean S {S['hard']:.1f}, B {B['hard']:.1f}, S/B {S['hard']/B['hard']:.3f} | same data {S['digest'] == B['digest']} | "
          f"weights hash ok {S['hash']}/{B['hash']} | steps {S['steps']}/{B['steps']} rows {S['rows']}/{B['rows']} | decay {S['wd']}/{B['wd']} | local {S['local']}/{B['local']} | complete {S['complete']}/{B['complete']}")
    curve = [round(st.mean(pct(x['sealed'][o]['12']) for o in ('add89','sub89')) - st.mean(pct(y['sealed'][o]['12']) for o in ('add89','sub89')), 1)
             for x, y in zip(births[('S', s)]['curve'], births[('B', s)]['curve'])]
    print(f"        S minus B at the six checkpoints: {curve}")

def life_row(d):
    stg = d["stages"]; sc = stg[-1]["scores"]; c = sc["compose"]; se = stg[0]["settings"]
    return dict(within=pct(c["within"]), pm=pm(sc), pairs={k: pct(c[k]) for k in ("+-", "-+", "><", "<>")}, f1=fam(sc, 97), f3=fam(sc, 73), dev=fam(sc, 89),
                taught=(pct(sc["taught"]) if "exact" in sc["taught"] else None), stages=len(stg), steps=field(d, "optimizer_steps"),
                birth_sha=se.get("birth_sha256"), birth=se.get("birth", "").split("/")[-1], local=d["local"], panel=se.get("panel_sha256"),
                digests=[x.get("data_digest", {}).get("sha256") for x in stg[1:]], finite=all(x.get("finite_losses") is True for x in stg[1:]),
                wd=se["optimizer"]["weight_decay"], resumes=se.get("resumes"), cmd=se.get("provenance", {}).get("command"),
                pm_by_stage=[round(pm(x["scores"]), 1) for x in stg], within_by_stage=[round(pct(x["scores"]["compose"]["within"]), 1) for x in stg])

print("\n2. Lives: cell XY = birth X, later update Y. SS and BB are E26's lives; SB and BS are E26-X's")
cells = {}
for s in SEEDS:
    for cell, path, folder in (("SS", f"e26/e26_life_S_{s}.json", "e26/"), ("BB", f"e26/e26_life_B_{s}.json", "e26/"),
                               ("SB", f"e26x/e26x_life_SB_{s}.json", "e26x/"), ("BS", f"e26x/e26x_life_BS_{s}.json", "e26x/")):
        d = load(path); r = life_row(d); r["hash"] = file_hash_ok(d, folder); cells[(cell, s)] = r
        want_birth = births[(cell[0], s)]["weights_sha256"]; want_local = (cell[1] == "S")
        r["birth_ok"] = (r["birth_sha"] == want_birth and r["birth"] == f"e26_dev_{cell[0]}_{s}.pt"); r["update_ok"] = (r["local"] == want_local)
problems = []
for s in SEEDS:
    ref = cells[("SS", s)]
    for cell in ("SS", "SB", "BS", "BB"):
        r = cells[(cell, s)]
        ok = dict(stages=r["stages"] == 7, steps=r["steps"] == [36864], birth=r["birth_ok"], update=r["update_ok"], finite=r["finite"], decay=r["wd"] == 0.25,
                  same_data=r["digests"] == ref["digests"], same_panel=r["panel"] == ref["panel"])
        bad = [k for k, v in ok.items() if not v]
        if bad: problems.append((cell, s, bad))
        print(f"   {cell} {s}: within {r['within']:5.1f} | plus-minus {r['pm']:5.1f} | +- {r['pairs']['+-']:5.1f} -+ {r['pairs']['-+']:5.1f} >< {r['pairs']['><']:5.1f} <> {r['pairs']['<>']:5.1f} | "
              f"F1 {r['f1']:5.1f} F3 {r['f3']:5.1f} dev {r['dev']:5.1f} | weights file hash {r['hash']} | resumes {len(r['resumes']) if r['resumes'] else 0} | checks failed: {bad or 'none'}")
print("   contract problems:", problems or "none")

print("\n3. Joint passes (within >= 85 and plus-minus >= 40 in the same seed), and the effects on plus-minus and on within at the end of life")
for cell in ("SS", "SB", "BS", "BB"):
    ok = [s for s in SEEDS if cells[(cell, s)]["within"] >= 85 and cells[(cell, s)]["pm"] >= 40]
    print(f"   {cell}: {len(ok)} of 3 {ok}; within {[round(cells[(cell, s)]['within'], 1) for s in SEEDS]}; plus-minus {[round(cells[(cell, s)]['pm'], 1) for s in SEEDS]}")
for key in ("pm", "within"):
    eff = {"update effect, birth S (SB - SS)": [cells[("SB", s)][key] - cells[("SS", s)][key] for s in SEEDS],
           "update effect, birth B (BB - BS)": [cells[("BB", s)][key] - cells[("BS", s)][key] for s in SEEDS],
           "birth effect, update S (BS - SS)": [cells[("BS", s)][key] - cells[("SS", s)][key] for s in SEEDS],
           "birth effect, update B (BB - SB)": [cells[("BB", s)][key] - cells[("SB", s)][key] for s in SEEDS]}
    for name, v in eff.items():
        present = sum(x >= 5 for x in v) >= 2 and st.mean(v) >= 5
        print(f"   {key:6s} {name}: {[round(x, 1) for x in v]} mean {st.mean(v):+.1f} present {present}")

print("\n4. Plus-minus by stage (0..6), each cell")
for s in SEEDS:
    for cell in ("SS", "SB", "BS", "BB"):
        print(f"   {cell} {s}: pm {cells[(cell, s)]['pm_by_stage']}  within {cells[(cell, s)]['within_by_stage']}")

print("\n5. Interrupted crossed lives against the final receipts (scores and loss at every stage the interrupted run reached)")
for cell, s in (("BS", 116089), ("BS", 93683), ("SB", 116089)):
    a, b = load(f"e26x/interrupted/e26x_life_{cell}_{s}.json"), load(f"e26x/e26x_life_{cell}_{s}.json")
    if a is None: print(f"   {cell} {s}: interrupted receipt not found"); continue
    same = [a["stages"][i]["scores"] == b["stages"][i]["scores"] and a["stages"][i].get("loss") == b["stages"][i].get("loss") and a["stages"][i].get("data_digest") == b["stages"][i].get("data_digest") for i in range(len(a["stages"]))]
    print(f"   {cell} {s}: {len(same)} stages in the interrupted receipt; identical in the final receipt: {same}; resumes recorded in the final receipt: {cells[(cell, s)]['resumes']}")
print("\n6. Which final crossed lives were made by which runner")
for s in SEEDS:
    for cell in ("SB", "BS"):
        print(f"   {cell} {s}: {cells[(cell, s)]['cmd']} | resumes {cells[(cell, s)]['resumes']}")
