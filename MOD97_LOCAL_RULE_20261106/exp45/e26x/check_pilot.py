"""E26-X pilot checks (E26X_DESIGN.md): independence of birth and update, identity of the diagonal with exp26_candidate.life.
Reads only. Run from MOD97_LOCAL_RULE_20261106. Writes exp45/e26x/pilot/PILOT_CHECK.json."""
import json, sys
from pathlib import Path
P = Path("exp45/e26x/pilot"); B = Path("exp45/e26")
ld = lambda f: json.loads(Path(f).read_text())
orig = {a: ld(P / f"orig/e26_life_{a}_dry256_73129.json") for a in "SB"}
x = {c: ld(P / f"e26x_life_{c}_dry256_73129.json") for c in ("SS", "BB", "SB", "BS")}
bsha = {a: ld(B / f"e26_dev_{a}_73129.json")["weights_sha256"] for a in "SB"}
scores = lambda r: [s["scores"] for s in r["stages"]]
res = {}
for a in "SB":
    o, xx = orig[a], x[a + a]
    res[f"diag_{a}_identical"] = dict(tensors=o["complete"]["tensors_sha256"] == xx["complete"]["tensors_sha256"],
                                      data_digest=o["complete"]["data_digest"] == xx["complete"]["data_digest"],
                                      scores=scores(o) == scores(xx), losses=[s.get("loss") for s in o["stages"]] == [s.get("loss") for s in xx["stages"]])
for c, r in x.items():
    st = r["stages"][0]["settings"]
    res[f"cell_{c}"] = dict(birth_sha_is_its_birth=st["birth_sha256"] == bsha[c[0]], update_local=r["local"], update_local_expected=c[1] == "S",
                            local_ok=r["local"] == (c[1] == "S"), crossover=st["provenance"].get("crossover"),
                            data_digest_equals_orig_S=r["complete"]["data_digest"] == orig["S"]["complete"]["data_digest"],
                            complete=isinstance(r.get("complete"), dict), finite=all(s["finite_losses"] for s in r["stages"][1:]))
res["SB_differs_from_SS"] = x["SB"]["complete"]["tensors_sha256"] != x["SS"]["complete"]["tensors_sha256"]
res["BS_differs_from_BB"] = x["BS"]["complete"]["tensors_sha256"] != x["BB"]["complete"]["tensors_sha256"]
res["SB_differs_from_BB"] = x["SB"]["complete"]["tensors_sha256"] != x["BB"]["complete"]["tensors_sha256"]
res["BS_differs_from_SS"] = x["BS"]["complete"]["tensors_sha256"] != x["SS"]["complete"]["tensors_sha256"]
res["stage1_first_score_add97_k12"] = {c: r["stages"][1]["scores"]["add97"]["12"]["exact"] for c, r in x.items()}
ok = all(all(v.values()) for k, v in res.items() if k.startswith("diag_")) and \
     all(v["birth_sha_is_its_birth"] and v["local_ok"] and v["data_digest_equals_orig_S"] and v["complete"] and v["finite"] for k, v in res.items() if k.startswith("cell_")) and \
     all(res[k] for k in ("SB_differs_from_SS", "BS_differs_from_BB", "SB_differs_from_BB", "BS_differs_from_SS"))
res["pilot_passes"] = ok
(P / "PILOT_CHECK.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
