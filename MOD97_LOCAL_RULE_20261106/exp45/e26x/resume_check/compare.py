"""Compare exp26_resume.py's runs (killed and resumed, or not) with exp26x_cross.py's uninterrupted pilot runs. Reads only."""
import json, sys
from pathlib import Path
D = Path("exp45/e26x/resume_check"); P = Path("exp45/e26x/pilot")
res = {}
for c in ("SB", "BS", "SS"):
    a = json.loads((D / f"e26x_life_{c}_dry256_73129.json").read_text()); r = json.loads((P / f"e26x_life_{c}_dry256_73129.json").read_text())
    strip = lambda st: {k: v for k, v in st.items() if k not in ("cpu", "cpu_scoring_so_far", "cpu_updates_so_far", "settings")}
    s0a, s0r = a["stages"][0]["settings"], r["stages"][0]["settings"]
    settings_same = {k: s0a.get(k) == s0r.get(k) for k in s0r if k != "provenance"}
    res[c] = dict(resumes=len(s0a["resumes"]), resumed_at=[x["at"] for x in s0a["resumes"]],
                  stages_equal_except_cpu=[strip(x) for x in a["stages"]] == [strip(x) for x in r["stages"]],
                  settings_equal_except_provenance=all(settings_same.values()), settings_differing=[k for k, v in settings_same.items() if not v],
                  tensors_equal=a["complete"]["tensors_sha256"] == r["complete"]["tensors_sha256"],
                  data_digest_equal=a["complete"]["data_digest"] == r["complete"]["data_digest"],
                  optimizer_steps_equal=a["complete"]["optimizer_steps"] == r["complete"]["optimizer_steps"],
                  no_checkpoint_left=not (D / f"e26x_life_{c}_dry256_73129.ckpt").exists())
res["qualified"] = all(v["stages_equal_except_cpu"] and v["settings_equal_except_provenance"] and v["tensors_equal"] and v["data_digest_equal"]
                       and v["optimizer_steps_equal"] and v["no_checkpoint_left"] for v in res.values()) and res["SB"]["resumes"] >= 1 and res["BS"]["resumes"] >= 1
(D / "RESUME_CHECK.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
