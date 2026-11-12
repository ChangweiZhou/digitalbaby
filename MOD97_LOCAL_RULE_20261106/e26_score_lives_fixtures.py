"""Fixtures for e26_score_lives.py: three corrupted copies of the reduced-dose receipts, each of which must be reported (exit 2). Writes exp45/e26_qual/E26_LIVES_SCORER_FIXTURES.json."""
import json, shutil, subprocess, sys, tempfile, pathlib
SRC = pathlib.Path("exp45/e26_qual"); STEMS = [f"e26_{k}_{a}_dry2048{'_dry256' if k == 'life' else ''}_73129.json" for k in ("dev", "life") for a in ("S", "B")]
def corrupt(name, fn):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for s in STEMS: shutil.copy(SRC / s, td / s)
        fn(td)
        r = subprocess.run([sys.executable, "e26_score_lives.py", str(td), "_dry2048", "_dry256"], capture_output=True, text=True)
        rep = json.loads((td / "E26_LIVES_SCORE.json").read_text())["problems"]
        return dict(exit=r.returncode, reported=rep, caught=r.returncode == 2 and bool(rep))
def edit(td, f, fn):
    p = td / f; d = json.loads(p.read_text()); fn(d); p.write_text(json.dumps(d))
L = "e26_life_S_dry2048_dry256_73129.json"
out = dict(
  clean=None,
  data_differs_at_stage_3=corrupt("digest", lambda td: edit(td, L, lambda d: d["stages"][3].__setitem__("data_digest", {"batches": 0, "sha256": "x"}))),
  birth_hash_not_the_birth_receipts=corrupt("birth", lambda td: edit(td, L, lambda d: d["stages"][0]["settings"].__setitem__("birth_sha256", "0" * 64))),
  a_stage_with_non_finite_losses=corrupt("finite", lambda td: edit(td, L, lambda d: d["stages"][5].__setitem__("finite_losses", False))),
  optimizer_decay_0_30=corrupt("decay", lambda td: edit(td, L, lambda d: d["complete"]["optimizer_read_back"]["groups"][0].__setitem__("weight_decay", 0.3))),
)
r = subprocess.run([sys.executable, "e26_score_lives.py", str(SRC), "_dry2048", "_dry256"], capture_output=True, text=True); out["clean"] = dict(exit=r.returncode)
out["pass"] = out["clean"]["exit"] == 0 and all(v["caught"] for k, v in out.items() if k not in ("clean", "pass"))
(SRC / "E26_LIVES_SCORER_FIXTURES.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: (v if k in ("clean", "pass") else v["caught"]) for k, v in out.items()}))
sys.exit(0 if out["pass"] else 1)
