"""E20-A: the coordinator's own recomputation from the twelve receipts (11 Nov 2026), with contract checks. Reads only. Writes exp45/E20A_COORDINATOR_CHECK.json."""
import json, hashlib, pathlib
SEEDS = (73129, 93683, 116089); PM = ("+-", "-+"); CMP = ("><", "<>")
ARMS = {"window reference (D)": "e16_life_D_cm6", "sign rule (T)": "e16_life_T_raw65536_cm6", "backprop on the pipeline (M)": "e16_life_M_raw65536_cm6", "two-bank reference (D bank2)": "e17_life_D_bank2_nbd_cm6"}
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out = dict(contract={}, scores={}, verdicts={}, killed={}); problems = []
R = {}
for arm, stem in ARMS.items():
    for s in SEEDS:
        p = pathlib.Path(f"exp45/{stem}_{s}.json"); r = json.loads(p.read_text()); R[arm, s] = r
        st = r["stages"]; c = r.get("complete"); set_ = st[0]["settings"]
        ok = dict(complete=isinstance(c, dict), stages=len(st), weights_match=isinstance(c, dict) and p.with_suffix(".pt").exists() and c["weights_sha256"] == sha(p.with_suffix(".pt")),
                  compgap=set_.get("compgap"), arch=set_.get("arch"), loss=set_.get("loss"), stream=set_.get("stream"), replay=set_.get("replay16"), cap=set_.get("cap"), rshare=set_.get("rshare"),
                  bookdecay=set_.get("bookdecay"), birth=set_.get("birth"), birth_admission=set_.get("birth_admission"), memory_panel=set_["memory_panels"]["sha256"][:16], panel=set_["panel_sha256"][:16],
                  stage_updates=set_.get("stage_updates"))
        out["contract"][f"{arm} {s}"] = ok
        if not (ok["complete"] and ok["stages"] == 7 and ok["weights_match"] and ok["compgap"] == 6 and ok["stage_updates"] == 6144): problems.append(f"{arm} {s}: {ok}")
for s in SEEDS:
    if len({out["contract"][f"{a} {s}"]["memory_panel"] for a in ARMS}) != 1: problems.append(f"seed {s}: memory panels differ between arms")
def pool(d, pairs, key):
    return sum(d[p][key] for p in pairs)
for arm in ARMS:
    for s in SEEDS:
        sc = R[arm, s]["stages"][-1]["scores"]; m, ms = sc["memory"], sc["memory_silenced"]
        n = pool(m["twins"], PM, "pairs")
        out["scores"][f"{arm} {s}"] = dict(pm_twins_both=pool(m["twins"], PM, "both_correct"), pm_twins=n, pm_twin_items=pool(m["twins"], PM, "items_correct"), pm_lead_items=pool(m["lead"], PM, "items_correct"),
                                           pm_lead_both=pool(m["lead"], PM, "both_correct"), pm_silenced_both=pool(ms["twins"], PM, "both_correct"),
                                           cmp_twins_both=pool(m["twins"], CMP, "both_correct"), cmp_twin_items=pool(m["twins"], CMP, "items_correct"), cmp_lead_items=pool(m["lead"], CMP, "items_correct"),
                                           taught_pairs_new_instances_hidden=f'{m["within_hidden"]["exact"]}/{m["within_hidden"]["n"]}', taught_triples_hidden=f'{m["taught_hidden"]["exact"]}/{m["taught_hidden"]["n"]}',
                                           cpu_updates=round(R[arm, s]["stages"][-1].get("cpu_updates_so_far", 0)), cpu_scoring=round(R[arm, s]["stages"][-1].get("cpu_scoring_so_far", 0)))
S = lambda arm, s: out["scores"][f"{arm} {s}"]
D, T_, M, B = list(ARMS)
V = out["verdicts"]
def v(name, hits, need, detail): V[name] = dict(per_seed=hits, verdict="holds" if sum(hits) >= need else "misses", detail=detail)
v("P20.4", [S(B, s)["pm_twins_both"] <= .05 * S(B, s)["pm_twins"] for s in SEEDS], 3, [f'{S(B, s)["pm_twins_both"]}/{S(B, s)["pm_twins"]}' for s in SEEDS])
r5 = [S(T_, s)["pm_twin_items"] / S(D, s)["pm_twin_items"] for s in SEEDS]; v("P20.5", [x <= .60 for x in r5], 2, [round(x, 3) for x in r5])
r6 = [S(M, s)["pm_twin_items"] / S(D, s)["pm_twin_items"] for s in SEEDS]; v("P20.6", [x >= .85 for x in r6], 2, [round(x, 3) for x in r6])
# the killed receipts: same settings as the completed life of the same name, and how far each got
for p in sorted(pathlib.Path("exp45").glob("*cm6_*_killed.json")):
    k = json.loads(p.read_text()); full = json.loads(pathlib.Path(str(p).replace("_killed", "")).read_text())
    ks, fs = k["stages"][0]["settings"], full["stages"][0]["settings"]
    diff = {key: (ks.get(key), fs.get(key)) for key in set(ks) | set(fs) if ks.get(key) != fs.get(key) and key not in ("zero_grad_first_update",)}
    same_scores = [json.dumps(a["scores"], sort_keys=True) == json.dumps(b["scores"], sort_keys=True) for a, b in zip(k["stages"], full["stages"])]
    out["killed"][p.name] = dict(stages_reached=len(k["stages"]), settings_that_differ=diff, stages_with_identical_scores_in_the_relaunch=sum(same_scores), has_completion_record=bool(k.get("complete")))
out["problems"] = problems
pathlib.Path("exp45/E20A_COORDINATOR_CHECK.json").write_text(json.dumps(out, indent=1, default=str))
print("problems:", problems)
for k_, x in V.items(): print(k_, x)
for arm in ARMS:
    print(arm); [print("  ", s, S(arm, s)) for s in SEEDS]
print("contract (first seed of each arm):"); [print("  ", a, out["contract"][f"{a} 73129"]) for a in ARMS]
print("killed:"); [print("  ", k_, x) for k_, x in out["killed"].items()]
