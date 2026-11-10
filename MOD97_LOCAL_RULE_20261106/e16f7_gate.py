"""E16-F7 copy of PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/gate_by_item_type.py: seed list extended to E16-F7's six seeds, REF
defaults to HEAD. Nothing else changed.

Re-score the E16 raw-reservoir lives from their committed receipts, by item type. Read-only, stdlib only.

Reads MOD97_LOCAL_RULE_20261106/exp45/*.json through `git show REF:...` (default REF origin/claude/vibrant-gauss-tops5f,
which holds all twelve drift seeds). Learner: e16_life_T_raw65536[_drift_online]_<seed>.json; reference ("twin"):
e16_life_D[_drift]_<seed>.json. For the three standing seeds on the staged stream the original E16-R3 run
(*_oldpanels.json) is used when present, since that is the run the gate was first declared on.
Run with:  python3 -I gate_by_item_type.py [REF]

Columns, all at the last checkpoint, as learner / reference = ratio:
  fam     the gate as pre-registered: the smallest of the three family ratios (mod-97 operations at k = 12, the four
          held operator pairs by final answer, mod-73 operations at k = 12)
  +-,-+   the two held pairs whose shortcut ceilings are about 1% (final answer)
  ><,<>   the two held pairs where a rule that ignores the first operator scores about 70% (final answer)
  whole   the four held pairs scored on the whole worked answer (intermediate ; final EOS), not the final answer alone
  a/s97   add and sub mod 97 at k = 12 (retention of the first family without max and min)
"""
import json
import statistics as st
import subprocess
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"   # E16-F7 copy: REF set to this job's HEAD
SEEDS = [73129, 93683, 116089, 200003, 300007, 400009, 500011, 600013, 700027, 800011, 900007, 1000003,
         1100009, 1200007, 1300021, 1400017, 1500007, 1600033]   # E16-F7 copy: six new seeds appended
HELD = ["+-", "-+", "><", "<>"]


def load(name):
    r = subprocess.run(["git", "show", f"{REF}:MOD97_LOCAL_RULE_20261106/exp45/{name}"], capture_output=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def ops(sc, names, k="12"):
    return 100 * sum(sc[n][k]["exact"] for n in names) / sum(sc[n][k]["n"] for n in names)


def comp(sc, pairs, key):
    return 100 * sum(sc["compose"][p][key] for p in pairs) / sum(sc["compose"][p]["n"] for p in pairs)


def fam_mean(sc, mod):
    return st.mean(100 * sc[f"{o}{mod}"]["12"]["exact"] / sc[f"{o}{mod}"]["12"]["n"] for o in ("add", "sub", "max", "min"))


def pair_mean(sc, key):
    return st.mean(100 * sc["compose"][p][key] / sc["compose"][p]["n"] for p in HELD)


total = {"fam": 0, "carry": 0, "whole": 0, "as97": 0, "n": 0}
for stream, tname, dname in (("staged", "e16_life_T_raw65536_{}{}.json", "e16_life_D_{}.json"),
                             ("drift", "e16_life_T_raw65536_drift_online_{}{}.json", "e16_life_D_drift_{}.json")):
    print(f"== {stream} stream")
    print("    seed |  fam (min ratio) |  +-,-+ learner/ref = ratio |  ><,<> learner/ref | whole answer ratio |  a/s97 learner/ref = ratio")
    cnt = {"fam": 0, "carry": 0, "whole": 0, "as97": 0, "n": 0}
    carry_l, carry_r, mm_l, mm_r, ceil = [], [], [], [], []
    for s in SEEDS:
        t = (load(tname.format(s, "_oldpanels")) if stream == "staged" else None) or load(tname.format(s, ""))
        d = load(dname.format(s))
        if not t or not d or len(t["stages"]) != 7 or len(d["stages"]) != 7:
            continue
        a, b = t["stages"][-1]["scores"], d["stages"][-1]["scores"]
        fam = min(fam_mean(a, 97) / fam_mean(b, 97), pair_mean(a, "final") / pair_mean(b, "final"), fam_mean(a, 73) / fam_mean(b, 73))
        cl, cr = comp(a, ["+-", "-+"], "final"), comp(b, ["+-", "-+"], "final")
        ml, mr = comp(a, ["><", "<>"], "final"), comp(b, ["><", "<>"], "final")
        whole = pair_mean(a, "exact") / pair_mean(b, "exact")
        al, ar = ops(a, ["add97", "sub97"]), ops(b, ["add97", "sub97"])
        sc0 = t["stages"][0].get("shortcut_ceilings", {})
        ceil += [max(sc0[p].values()) for p in ("><", "<>") if p in sc0]
        carry_l.append(cl); carry_r.append(cr); mm_l.append(ml); mm_r.append(mr)
        cnt["n"] += 1; cnt["fam"] += fam >= .8; cnt["carry"] += cl / cr >= .8; cnt["whole"] += whole >= .8; cnt["as97"] += al / ar >= .8
        print(f"{s:8d} |  {fam:.3f} {'pass' if fam >= .8 else 'FAIL'}     |  {cl:5.1f} / {cr:5.1f} = {cl / cr:.3f}     |  {ml:5.1f} / {mr:5.1f}     |  {whole:.3f}            |  {al:5.1f} / {ar:5.1f} = {al / ar:.3f}")
    print(f"   lives {cnt['n']}: at or above 0.80 -- as pre-registered {cnt['fam']}, +-/-+ only {cnt['carry']}, whole answer {cnt['whole']}, add/sub mod 97 {cnt['as97']}")
    print(f"   means: +-,-+ learner {st.mean(carry_l):.1f} vs reference {st.mean(carry_r):.1f}; ><,<> learner {st.mean(mm_l):.1f} vs reference {st.mean(mm_r):.1f}"
          + (f"; best single shortcut on ><,<> {100 * st.mean(ceil):.0f}%" if ceil else ""))
    for k in total:
        total[k] += cnt[k]
print(f"== all lives {total['n']}: as pre-registered {total['fam']}, +-/-+ only {total['carry']}, whole answer {total['whole']}, add/sub mod 97 {total['as97']}")
