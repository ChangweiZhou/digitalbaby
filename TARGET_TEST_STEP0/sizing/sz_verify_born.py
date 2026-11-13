"""Span brief (OPUS_TASK_SIZING_SPAN_20261113.md, revision 2), admission item 4: the one-off check of E26's backprop births.

The only file of the span job allowed to import the arithmetic harness. For each birth MOD97_LOCAL_RULE_20261106/exp45/e26/e26_dev_B_<seed>.pt
(seeds 73129, 93683, 116089):
  * its SHA-256 equals `weights_sha256` in its receipt e26_dev_B_<seed>.json;
  * it is a bare state dict that loads strictly into sz_learner.Net;
  * on one birth batch (the first batch of the birth's own stream, exp11_cell.batches over the development tasks, built as
    exp20_reference.dev builds it, with that birth's seed), the logits of sz_learner.Net and of arch26.forward (exact path) agree to 1e-4.

    python3 sz_verify_born.py      writes runs/span/verify_born.json; exit code nonzero on any failure
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
MOD = HERE.parents[1] / "MOD97_LOCAL_RULE_20261106"
sys.path.insert(0, str(HERE))
import sz_learner as L     # noqa: E402

sys.path.insert(0, str(MOD))
import arch20              # noqa: E402
import arch26              # noqa: E402

BIRTHS = (73129, 93683, 116089)
TOL = 1e-4


def main():
    T.set_num_threads(1)
    arch20.install("G1")
    import exp11_cell as E
    X = E.X
    out = dict(brief="OPUS_TASK_SIZING_SPAN_20261113.md rev. 2, admission item 4", python=platform.python_version(), torch=T.__version__,
               machine=platform.machine(), births={})
    ok = True
    for seed in BIRTHS:
        pt = MOD / "exp45" / "e26" / f"e26_dev_B_{seed}.pt"
        rec = json.loads(pt.with_suffix(".json").read_text())
        sha = hashlib.sha256(pt.read_bytes()).hexdigest()
        sd = T.load(pt, map_location="cpu", weights_only=True)
        net = L.Net(); missing = net.load_state_dict(sd, strict=True)
        ref = arch26.new_net(seed); ref.load_state_dict(sd, strict=True)
        tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
        items = next(E.batches(tp, seed, E.KMAX["M"]))
        x = X.batch_tensor(items)[0]
        with T.no_grad():
            a = net(x, T.zeros(1, x.shape[0], L.UNITS))[0]
            b = arch26.forward(ref, x, False, True)[0]
        diff = float((a - b).abs().max())
        good = sha == rec["weights_sha256"] and diff <= TOL and isinstance(sd, dict) and len(sd) == 13
        ok &= good
        out["births"][str(seed)] = dict(file=pt.name, sha256=sha, receipt_weights_sha256=rec["weights_sha256"], sha_equal=sha == rec["weights_sha256"],
                                        tensors=len(sd), loads_strictly=not missing.missing_keys and not missing.unexpected_keys,
                                        batch_shape=list(x.shape), max_abs_logit_difference=diff, tolerance=TOL, passed=good)
        print(seed, "sha", sha == rec["weights_sha256"], "max|dlogit|", f"{diff:.2e}", "PASS" if good else "FAIL", flush=True)
    out["passed"] = ok
    (HERE / "runs" / "span").mkdir(parents=True, exist_ok=True)
    (HERE / "runs" / "span" / "verify_born.json").write_text(json.dumps(out, indent=1))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
