"""E26: the coordinator's scorer for the six lives, written on 12 November 2026 before any full-dose life existed (review 9.111).
Reads receipts only; trains nothing. Judges P26.3 to P26.5 of E26_PREDICTIONS.md and the candidate-pair reading of the brief.

  python3 e26_score_lives.py                     the full-dose lives in exp45/e26
  python3 e26_score_lives.py <dir> <devtag> <drytag>   for testing on reduced-dose receipts (e.g. exp45/e26_qual _dry2048 _dry256)

Writes <dir>/E26_LIVES_SCORE.json and prints the tables. Exits 2 if any contract check fails; the figures are then not to be read."""
import sys, json, hashlib, pathlib
SEEDS = (73129, 93683, 116089); ARMS = ("S", "B"); PM = ("+-", "-+"); PAIRS = ("+-", "-+", "><", "<>")
MARK_WITHIN, MARK_PM = 85.0, 40.0
D = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("exp45/e26")
DEVTAG = sys.argv[2] if len(sys.argv) > 2 else ""
DRYTAG = sys.argv[3] if len(sys.argv) > 3 else ""
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pct = lambda k, n: round(100 * k / n, 1) if n else None


def load(arm, seed):
    lp = D / f"e26_life_{arm}{DEVTAG}{DRYTAG}_{seed}.json"; bp = D / f"e26_dev_{arm}{DEVTAG}_{seed}.json"
    if not lp.exists():
        return None, [f"{lp.name}: absent (unmeasured)"]
    life, birth = json.loads(lp.read_text()), (json.loads(bp.read_text()) if bp.exists() else None)
    bad = []
    c = life.get("complete")
    if not isinstance(c, dict): bad.append(f"{lp.name}: no completion record")
    st = life.get("stages", [])
    if len(st) != 7: bad.append(f"{lp.name}: {len(st)} stages, not 7")
    if not all(s.get("finite_losses", False) for s in st[1:]): bad.append(f"{lp.name}: a stage without finite losses")
    if isinstance(c, dict):
        g = {grp["weight_decay"]: grp["params"] for grp in c["optimizer_read_back"]["groups"]}
        if sorted(g) != [0.0, 0.25] or g[0.0] != ["book.weight"]: bad.append(f"{lp.name}: optimizer groups are not decay 0.25 with the embedding table at 0")
        wp = D / c["weights"]
        if wp.exists() and sha(wp) != c["weights_sha256"]: bad.append(f"{lp.name}: weights file does not match its hash")
    se = st[0].get("settings", {}) if st else {}
    if birth is None:
        bad.append(f"{bp.name}: birth receipt absent")
    else:
        bw = (birth.get("complete") or {}).get("weights_sha256") if isinstance(birth.get("complete"), dict) else birth.get("weights_sha256")
        if se.get("birth_sha256") != bw: bad.append(f"{lp.name}: its birth hash is not the one its birth receipt records")
        if not str(se.get("birth", "")).endswith(f"e26_dev_{arm}{DEVTAG}_{seed}.pt"): bad.append(f"{lp.name}: its birth is not its own arm's and seed's")
    if life.get("arm") != arm or life.get("seed") != seed: bad.append(f"{lp.name}: arm or seed in the receipt differ from the file name")
    return dict(life=life, settings=se), bad


R = {}; problems = []; unmeasured = []
for s in SEEDS:
    for a in ARMS:
        x, bad = load(a, s)
        if x is None: unmeasured.append(f"{a} {s}")
        else: R[a, s] = x
        problems += [b for b in bad if "unmeasured" not in b]
    if ("S", s) in R and ("B", s) in R:      # one difference: the update. Same panels, same data at every stage
        ls, lb = R["S", s]["life"]["stages"], R["B", s]["life"]["stages"]
        if R["S", s]["settings"].get("panel_sha256") != R["B", s]["settings"].get("panel_sha256"): problems.append(f"{s}: the two arms' panels differ")
        for i, (x, y) in enumerate(zip(ls[1:], lb[1:]), 1):
            if x.get("data_digest") != y.get("data_digest"): problems.append(f"{s}: the two arms' data differ at stage {i}")

def endpoints(st):
    sc = st["scores"]; c = sc["compose"]
    return dict(within=pct(c["within"]["exact"], c["within"]["n"]), plus_minus=pct(sum(c[p]["exact"] for p in PM), sum(c[p]["n"] for p in PM)),
                **{f"pair {p}": pct(c[p]["exact"], c[p]["n"]) for p in PAIRS},
                taught=pct(sc["taught"]["exact"], sc["taught"]["n"]) if isinstance(sc.get("taught"), dict) else None,
                mod97=round(sum(pct(sc[o]["12"]["exact"], sc[o]["12"]["n"]) for o in ("add97", "sub97", "max97", "min97")) / 4, 1))

E = {k: [endpoints(st) for st in v["life"]["stages"]] for k, v in R.items()}
end = {k: v[-1] for k, v in E.items()}
meets = lambda a, s: (a, s) in end and end[a, s]["within"] >= MARK_WITHIN and end[a, s]["plus_minus"] >= MARK_PM
V = {}
V["P26.3 arm S ends at >= 85 on new instances of taught pairs, >= 2 of 3 seeds"] = [(a_ := ("S", s)) in end and end[a_]["within"] >= MARK_WITHIN for s in SEEDS]
V["P26.4 arm S ends at >= 40 on the untaught plus-minus pairs, >= 2 of 3 seeds"] = [("S", s) in end and end["S", s]["plus_minus"] >= MARK_PM for s in SEEDS]
V["P26.5 arm B meets both marks in the same >= 2 seeds"] = [meets("B", s) for s in SEEDS]
V["candidate pair: arm S meets both marks in the same >= 2 seeds"] = [meets("S", s) for s in SEEDS]
verdicts = {k: dict(per_seed=v, verdict=("holds" if sum(v) >= 2 else "misses")) for k, v in V.items()}
paired = {s: {key: [round(e_s[key] - e_b[key], 1) if None not in (e_s[key], e_b[key]) else None for e_s, e_b in zip(E["S", s], E["B", s])]
              for key in ("within", "plus_minus", "pair +-", "pair -+", "pair ><", "pair <>", "taught", "mod97")}
          for s in SEEDS if ("S", s) in E and ("B", s) in E}
out = dict(dir=str(D), problems=problems, unmeasured=unmeasured, denominator=len(SEEDS), verdicts=verdicts,
           end_of_life={f"{a} {s}": v for (a, s), v in end.items()}, by_stage={f"{a} {s}": v for (a, s), v in E.items()}, paired_S_minus_B_by_stage=paired,
           note="Unmeasured seeds count as not meeting a mark; the denominator stays three (brief, revision 2). Feasibility (the marks) and the gap (S - B) are read separately.")
(D / "E26_LIVES_SCORE.json").write_text(json.dumps(out, indent=1))
print("problems:", problems or "none"); print("unmeasured:", unmeasured or "none")
for k, v in verdicts.items(): print(f"  {k}: {v['verdict']} {v['per_seed']}")
print("\nend of life (within / plus-minus / +- -+ >< <> / taught / mod-97):")
for (a, s), v in sorted(end.items()): print(f"  {a} {s}: {v['within']} / {v['plus_minus']} / {v['pair +-']} {v['pair -+']} {v['pair ><']} {v['pair <>']} / {v['taught']} / {v['mod97']}")
for s, d in paired.items(): print(f"  S-B {s} within by stage {d['within']}  plus-minus {d['plus_minus']}")
sys.exit(2 if problems else 0)
