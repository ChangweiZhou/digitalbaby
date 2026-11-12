"""E24 host check: compare a rerun birth with its pinned original (sealed scores at every checkpoint, weights tensor by tensor)."""
import hashlib, json, sys
import torch

orig_json, new_json, orig_pt, new_pt = sys.argv[1:5]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


a, b = json.load(open(orig_json)), json.load(open(new_json))
out = {"original": orig_json.split("/")[-1], "original_sha256": sha(orig_json), "rerun_sha256": sha(new_json),
       "original_weights_sha256": sha(orig_pt), "rerun_weights_sha256": sha(new_pt)}
ca, cb = a["curve"], b["curve"]
out["checkpoints"] = [len(ca), len(cb)]
rows = []
for x, y in zip(ca, cb):
    rows.append({"step": [x["step"], y["step"]], "sealed_equal": x["sealed"] == y["sealed"],
                 "loss": [x.get("loss"), y.get("loss")], "cpu_rerun": y.get("cpu")})
out["per_checkpoint"] = rows
out["sealed_equal_all"] = len(ca) == len(cb) and all(r["sealed_equal"] for r in rows)
out["other_keys_equal"] = {k: a.get(k) == b.get(k) for k in a if k != "curve"}


def tensors(obj, p=""):
    if isinstance(obj, torch.Tensor):
        yield p, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from tensors(v, f"{p}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{p}[{i}]")


ta = dict(tensors(torch.load(orig_pt, map_location="cpu", weights_only=False)))
tb = dict(tensors(torch.load(new_pt, map_location="cpu", weights_only=False)))
out["tensor_names_equal"] = sorted(ta) == sorted(tb)
diffs = {}
for k in ta:
    if k in tb:
        eq = ta[k].shape == tb[k].shape and torch.equal(ta[k], tb[k])
        mx = float((ta[k].double() - tb[k].double()).abs().max()) if ta[k].shape == tb[k].shape and ta[k].numel() else None
        diffs[k] = {"equal": eq, "max_abs_diff": mx, "shape": list(ta[k].shape)}
out["tensors"] = len(ta)
out["tensors_bit_equal"] = sum(d["equal"] for d in diffs.values())
out["largest_tensor_difference"] = max((d["max_abs_diff"] or 0.0) for d in diffs.values()) if diffs else None
out["tensor_detail"] = diffs
print(json.dumps(out, indent=1))
