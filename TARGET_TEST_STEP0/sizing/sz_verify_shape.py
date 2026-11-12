"""Sizing sweep, stage 1: one-off verification that sz_learner.Net is E20-B's G1 (arch20.NetRNN("gru")) in shape and initial weights.

This script imports E20-B's arch20 from MOD97_LOCAL_RULE_20261106 for the comparison only; the sweep's own code never imports it.
    python3 sz_verify_shape.py        prints one JSON line
"""
import json
import sys
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_learner as L     # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "MOD97_LOCAL_RULE_20261106"))
import arch20              # noqa: E402

out = dict(seeds={})
for seed in (1101, 1102, 1103, 1104):
    a, b = arch20.new_net(seed, "G1"), L.new_net(seed)
    na, nb = dict(a.named_parameters()), dict(b.named_parameters())
    out["seeds"][seed] = dict(same_names_in_order=list(na) == list(nb), same_shapes=all(na[k].shape == nb[k].shape for k in na),
                              initial_weights_bitwise_equal=all(T.equal(na[k], nb[k]) for k in na))
out.update(params_e20b=sum(p.numel() for p in a.parameters()), params_sizing=L.n_params(b), padding_idx=[a.book.padding_idx, b.book.padding_idx],
           outputs=[a.l3.out_features, b.l3.out_features], torch=T.__version__)
out["passed"] = all(all(v.values()) for v in out["seeds"].values()) and out["params_e20b"] == out["params_sizing"]
print(json.dumps(out))
