"""V167 C1 technical (design section 4). Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_t/bin/python v167_technical.py
Checks: LOCAL's updates equal the explicit formulas (float64, dropout mask included); with F_H = F_T = 0 no gradient reaches
H or T; the loop in mode 'dfa_h' reproduces V165's train_dfa exactly (25 lines, one epoch); the shuffle is a within-batch
permutation; LOCAL is deterministic (25 lines); learns (200 lines, one epoch: LOCAL >= 0.01 below LOCAL with F_H = 0).
Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v167_common as C  # noqa: E402
import v167_local as L  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

V, D = C.V, L.D
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def main():
    t0 = time.process_time()
    out, records = [], 0
    A = V.load_codes()["A"]                                  # Thursday (C2's text A)
    cls = V.R.Classes(A["train"]["y"])
    idx, c = A["train"]["idx"][:96], torch.from_numpy(cls.encode(A["train"]["y"][:96]))
    # 1. explicit formulas (float64, train mode, dropout mask regenerated from the same seed)
    net = L.make(cls).double()
    net.train()
    FH = D.feedback(cls.K, 512).double()
    FT = L.feedback(cls.K, 2048, L.S["F_T_seed"]).double()
    net.zero_grad()
    torch.manual_seed(5)
    L.local_step(net, FH, FT, idx, c)
    torch.manual_seed(5)
    with torch.no_grad():
        aH = F.embedding_bag(idx, net.W, mode="sum", padding_idx=V.PAD) + net.b
        h = torch.relu(aH)
        aT = net.top.first(h)
        msk = net.top.drop(torch.ones_like(aT))
        zd = torch.relu(aT) * msk
        p = torch.softmax(net.top.out(zd), 1)
        e = (p - F.one_hot(c, cls.K).double()) / len(c)
        dT = (e @ FT) * msk * (aT > 0).double()
        dH = (e @ FH) * (aH > 0).double()
        ref = {"top.out.weight": e.T @ zd, "top.out.bias": e.sum(0), "top.first.weight": dT.T @ h,
               "top.first.bias": dT.sum(0), "b": dH.sum(0)}
        dW = torch.zeros_like(net.W)
        for s in range(len(c)):
            ix = idx[s][idx[s] != V.PAD]
            dW[ix] += dH[s]
        ref["W"] = dW
    got = dict(net.named_parameters())
    err = {k: float((got[k].grad - v).abs().max()) for k, v in ref.items()}
    check(out, "LOCAL updates equal the explicit formulas (readout delta rule, DFA for T with dropout mask, DFA for H; <= 1e-10)",
          max(err.values()) <= 1e-10, err)
    # 2. no gradient crosses a layer
    net.zero_grad()
    L.local_step(net, torch.zeros_like(FH), torch.zeros_like(FT), idx, c)
    z = {k: float(got[k].grad.abs().max()) for k in ("W", "b", "top.first.weight", "top.first.bias")}
    check(out, "with F_H = F_T = 0, H and T receive no gradient (only the readout learns)",
          all(v == 0.0 for v in z.values()) and float(got["top.out.weight"].grad.abs().max()) > 0, z)
    # 3. parity of the loop with V165 (mode dfa_h)
    sub = C.first_lines(A["train"], C.SET["c1"]["det_lines"])
    w_mine = L.train(sub, A["calib"], "dfa_h", 1, 1)[0][0]
    w_v165 = D.train_dfa(sub, A["calib"], False, 1, 1)[0][0]
    records += 2 * C.SET["c1"]["det_lines"]
    check(out, "the loop in mode 'dfa_h' reproduces V165's train_dfa exactly (25 lines, one epoch)", torch.equal(w_mine, w_v165))
    # 4. shuffle
    g = torch.Generator().manual_seed(L.S["shuffle_seed"])
    perm = torch.randperm(len(c), generator=g)
    check(out, "the shuffle is a within-batch permutation", torch.equal(torch.sort(perm).values, torch.arange(len(c))))
    # 5. determinism
    w = [L.train(sub, A["calib"], "local", 1, 1)[0][0] for _ in range(2)]
    records += 2 * C.SET["c1"]["det_lines"]
    check(out, "LOCAL is deterministic (25 lines, one epoch)", torch.equal(w[0], w[1]))
    # 6. learns
    sub = C.first_lines(A["train"], C.SET["c1"]["learn_lines"])
    _, _, i1 = L.train(sub, A["calib"], "local", 1, 1)
    _, _, i0 = L.train(sub, A["calib"], "local", 1, 1, zero_FH=True)
    records += 2 * C.SET["c1"]["learn_lines"]
    check(out, "learns: after one epoch on 200 lines, LOCAL >= 0.01 below LOCAL with F_H = 0",
          i1["calib_bpb"] <= i0["calib_bpb"] - C.SET["c1"]["learn_margin"],
          {"local": i1["calib_bpb"], "local_FH0": i0["calib_bpb"], "cpu_s_per_200_lines": i1["cpu_s"]})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; records", records)
    C.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
