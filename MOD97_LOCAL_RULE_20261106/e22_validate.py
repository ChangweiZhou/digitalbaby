"""E22 version 2: the six-cell receipt validator and the verdicts, written before any E22 life ran (10 Nov 2026).
A cell that fails validation is not scored. Definitions are those of OPUS_TASK_E22_RULE_BORN_LIFE_20261110.md, version 2:
checkpoint 6 unless a block is named, k = 12, whole answers (exact), four operations weighted equally; hard mean = (add97 + sub97) / 2."""
import hashlib, json, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / "exp45"
SEEDS = (73129, 93683, 116089)
F1 = ("add97", "sub97", "max97", "min97"); F3 = ("add73", "sub73", "max73", "min73"); DV = ("add89", "sub89", "max89", "min89")
pc = lambda d: 100 * d["exact"] / d["n"]
man = json.loads((OUT / "BIRTH_MANIFEST.json").read_text())
codes = (OUT / "logs" / "e22_exit_codes.txt").read_text().split("\n") if (OUT / "logs" / "e22_exit_codes.txt").exists() else []
res, problems = {}, []
def cell(name, seed, born):
    p = OUT / f"{name}_{seed}.json"; bad = []
    if not p.exists():
        return None, ["receipt missing"]
    d = json.loads(p.read_text()); st = d["stages"]; se = [x["settings"] for x in st if "settings" in x][-1]
    birth = f"e17_dev_bank2_nbd{'_bT_tall' if born == 'rule' else ''}_{seed}.pt"
    if d.get("arm") != "T" or d.get("seed") != seed: bad.append("arm or seed")
    if [x["stage"] for x in st] != list(range(7)): bad.append("stage sequence")
    if se.get("stage_updates") != 6144: bad.append("dose")
    if se.get("birth") != birth or se.get("birth_sha256") != man[birth]["sha256"] or se.get("birth_admission") != "manifest": bad.append("birth")
    want = dict(arch="bank2", bookdecay="off", heads="top", feedback="sign", traceup="all", freeze="none", loss="answer", stream="blocks", admit="block", replay16="raw", cap=65536, rshare=0.5, wd=0.25, compgap=0, outtag="_e22", birthrule=("T" if born == "rule" else "D"))
    bad += [f"setting {k}={se.get(k)!r}" for k, v in want.items() if se.get(k) != v]
    c = d.get("complete") or {}
    w = OUT / c.get("weights", "missing")
    if c.get("stages") != 7 or not w.exists() or hashlib.sha256(w.read_bytes()).hexdigest() != c.get("weights_sha256"): bad.append("completion record or weight hash")
    return (d, se), bad
for s in SEEDS:
    (a, ba), (b, bb) = cell("e17_life_T_bank2_nbd_tall_bT_raw65536_e22", s, "rule"), cell("e17_life_T_bank2_nbd_tall_raw65536_e22", s, "backprop")
    if a and b:
        sa, sb = a[1], b[1]
        diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k) and k not in ("birthrule", "birth", "birth_sha256", "zero_grad_first_update"))
        if diff: ba.append(f"partners differ in {diff}"); bb.append(f"partners differ in {diff}")
        if sa.get("panel_sha256") != sb.get("panel_sha256"): ba.append("panel hash differs from partner"); bb.append("panel hash differs from partner")
    for lab, c, bad in (("rule-born", a, ba), ("backprop-born", b, bb)):
        if bad: problems.append(f"{s} {lab}: {bad}")
    if a and b and not ba and not bb:
        def fam(st, i, names): return sum(pc(st[i]["scores"][n]["12"]) for n in names) / len(names)
        A, Bp = a[0]["stages"], b[0]["stages"]
        res[s] = dict(rule=dict(m97_b1=fam(A, 1, F1), hard_b1=fam(A, 1, F1[:2]), m97=fam(A, 6, F1), hard=fam(A, 6, F1[:2]), m73=fam(A, 6, F3), dev=[fam(A, i, DV) for i in range(7)], start=fam(A, 0, DV)),
                      backprop=dict(m97_b1=fam(Bp, 1, F1), hard_b1=fam(Bp, 1, F1[:2]), m97=fam(Bp, 6, F1), hard=fam(Bp, 6, F1[:2]), m73=fam(Bp, 6, F3), dev=[fam(Bp, i, DV) for i in range(7)], start=fam(Bp, 0, DV)))
if any(c and not c.startswith("0 ") for c in codes): problems.append("a job exited nonzero")
v = {}
ok = lambda f: sum(f(res[s]) for s in res)
v["P22.1 acquisition (rule-born, after block 1: mod-97 >= 80 and hard mean >= 70; two seeds)"] = ok(lambda r: r["rule"]["m97_b1"] >= 80 and r["rule"]["hard_b1"] >= 70)
v["P22.2 final competence (rule-born: mod-97 >= 75, hard mean >= 65, mod-73 >= 75, same seed; two seeds)"] = ok(lambda r: r["rule"]["m97"] >= 75 and r["rule"]["hard"] >= 65 and r["rule"]["m73"] >= 75)
v["P22.3 parity, one-sided (rule-born no more than 5 below partner on mod-97 and on mod-73, same seed; two seeds)"] = ok(lambda r: r["backprop"]["m97"] - r["rule"]["m97"] <= 5 and r["backprop"]["m73"] - r["rule"]["m73"] <= 5)
common_loss = ok(lambda r: not (r["backprop"]["m97"] >= 75 and r["backprop"]["hard"] >= 65 and r["backprop"]["m73"] >= 75) and not (r["rule"]["m97"] >= 75 and r["rule"]["hard"] >= 65 and r["rule"]["m73"] >= 75))
out = dict(validated_pairs=len(res), problems=problems, verdict_counts=v, verdicts={k: ("holds" if n >= 2 else "misses") for k, n in v.items()} if len(res) == 3 else "not judged: fewer than three validated pairs",
           seeds_where_both_partners_are_under_the_marks=common_loss, per_seed={str(s): r for s, r in res.items()})
(OUT / "E22_VALIDATION.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("validated_pairs", "problems", "verdict_counts", "verdicts", "seeds_where_both_partners_are_under_the_marks")}, indent=1))
sys.exit(0 if not problems else 1)
