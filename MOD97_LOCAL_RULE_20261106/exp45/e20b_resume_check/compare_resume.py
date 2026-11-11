"""Compare runs of exp20_resume.py that were killed with SIGKILL and resumed against the qualified uninterrupted runs of
exp20_reference.py (exp45/e20b_qual, attempt 2). Writes RESUME_CHECK.json beside this file. Run from MOD97_LOCAL_RULE_20261106."""
import hashlib
import json
from pathlib import Path

import torch as T

C, Q = Path("exp45/e20b_resume_check"), Path("exp45/e20b_qual")


def tensors_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k].detach().contiguous().cpu()
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


out = {}
# birth G3: killed and resumed vs the qualified birth
r = json.loads((C / "e20_dev_G3_dry2048_73129.json").read_text()); q = json.loads((Q / "e20_dev_G3_dry2048_73129.json").read_text())
fields = ("step", "loss", "sealed", "book_shift_by_role", "silenced_k12")
out["birth_G3"] = dict(resumes=r["resumes"],
                       curve_equal=[all(a.get(k) == b.get(k) for k in fields) for a, b in zip(r["curve"], q["curve"])] if len(r["curve"]) == len(q["curve"]) else "lengths differ",
                       census_equal=r["credit_census_first_update"] == q["credit_census_first_update"], optimizer_steps=[r["optimizer_steps"], q["optimizer_steps"]],
                       weights_file_sha256_equal=r["weights_sha256"] == q["weights_sha256"],
                       tensors_sha256=[tensors_sha(T.load(C / "e20_dev_G3_dry2048_73129.pt")), q["tensors_sha256"]])
b = out["birth_G3"]
b["pass"] = all(b["curve_equal"]) is True and b["census_equal"] and b["weights_file_sha256_equal"] and b["tensors_sha256"][0] == b["tensors_sha256"][1] and len(b["resumes"]) > 0
# life G0: killed and resumed vs the qualified life from the same birth
r = json.loads((C / "e20_life_G0_dry2048_dry256_73129.json").read_text()); q = json.loads((Q / "e20_life_G0_dry2048_dry256_73129.json").read_text())
rows = []
for a, b_ in zip(r["stages"], q["stages"]):
    row = dict(stage=a["stage"], scores_equal=a["scores"] == b_["scores"])
    if a["stage"]:
        row.update(loss_equal=a["loss"] == b_["loss"], exposure_equal=a["exposure_rows_tokens"] == b_["exposure_rows_tokens"],
                   exposure_panels_equal=a["exposure_panels"] == b_["exposure_panels"], optimizer_steps_equal=a["optimizer_steps"] == b_["optimizer_steps"])
    rows.append(row)
s0, q0 = r["stages"][0]["settings"], q["stages"][0]["settings"]
out["life_G0"] = dict(resumes=s0["resumes"], stages=[len(r["stages"]), len(q["stages"])], per_stage=rows,
                      zero_grad_first_update_equal=s0.get("zero_grad_first_update") == q0.get("zero_grad_first_update"),
                      panel_sha256_equal=s0["panel_sha256"] == q0["panel_sha256"], birth_sha256_equal=s0["birth_sha256"] == q0["birth_sha256"],
                      weights_file_sha256_equal=r["complete"]["weights_sha256"] == q["complete"]["weights_sha256"],
                      tensors_sha256=[r["complete"]["tensors_sha256"], q["complete"]["tensors_sha256"]])
L = out["life_G0"]
L["pass"] = (L["stages"] == [7, 7] and all(all(v for k, v in x.items() if k != "stage") for x in rows) and L["zero_grad_first_update_equal"]
             and L["panel_sha256_equal"] and L["birth_sha256_equal"] and L["weights_file_sha256_equal"] and L["tensors_sha256"][0] == L["tensors_sha256"][1]
             and len(L["resumes"]) > 0)
out["pass"] = out["birth_G3"]["pass"] and L["pass"]
(C / "RESUME_CHECK.json").write_text(json.dumps(out, indent=1))
print(json.dumps(dict(birth_G3=out["birth_G3"]["pass"], life_G0=L["pass"], resumes_birth=len(out["birth_G3"]["resumes"]), resumes_life=len(L["resumes"]),
                      overall=out["pass"]), indent=1))
