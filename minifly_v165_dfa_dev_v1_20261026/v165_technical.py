"""V165 C1 technical (design section 3). Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_t/bin/python v165_technical.py
Checks: the update equals the explicit formula (float64); with F = 0, W gets no gradient and the top's gradient equals
back-propagation within the top; the shuffle is a within-batch permutation; determinism (25 lines, one epoch);
learns (200 lines, one epoch: A calibration >= 0.01 below the frozen-H reference from C0). Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v165_common as C  # noqa: E402
import v165_dfa as D  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

V = C.V
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def model64(cls):
    torch.manual_seed(0)
    W, b = V.init_H()
    net = V.BPH(cls.K, W, b).double()
    net.eval()
    return net


def main():
    t0 = time.process_time()
    out, records = [], 0
    A = V.load_codes()["A"]
    cls = V.R.Classes(A["train"]["y"])
    idx, c = A["train"]["idx"][:96], torch.from_numpy(cls.encode(A["train"]["y"][:96]))
    # 1. explicit formula
    net = model64(cls)
    Fb = D.feedback(cls.K, net.W.shape[1]).double()
    net.zero_grad()
    e, delta = D.dfa_step(net, Fb, idx, c)
    with torch.no_grad():
        a = F.embedding_bag(idx, net.W, mode="sum", padding_idx=V.PAD) + net.b
        p = torch.softmax(net.top(torch.relu(a)), 1)
        e_ref = (p - F.one_hot(c, cls.K).double()) / len(c)
        d_ref = (e_ref @ Fb) * (a > 0).double()
        dW = torch.zeros_like(net.W)
        for s in range(len(c)):
            ix = idx[s][idx[s] != V.PAD]
            dW[ix] += d_ref[s]
    err = max(float((net.W.grad - dW).abs().max()), float((net.b.grad - d_ref.sum(0)).abs().max()))
    check(out, "DFA update equals delta = ReLU'(a) (e F), dW = sum delta x (float64, <= 1e-10)", err <= 1e-10,
          {"max_abs_diff": err, "max_abs_grad": float(net.W.grad.abs().max())})
    # 2. F = 0: no gradient to W; top gradient = back-propagation within the top
    net.zero_grad()
    D.dfa_step(net, torch.zeros_like(Fb), idx, c)
    top_dfa = {k: v.grad.clone() for k, v in net.top.named_parameters()}
    w0 = float(net.W.grad.abs().max())
    net.zero_grad()
    F.cross_entropy(net(idx), c).backward()
    top_bp = {k: v.grad for k, v in net.top.named_parameters()}
    terr = max(float((top_dfa[k] - top_bp[k]).abs().max()) for k in top_bp)
    check(out, "with F = 0, W gets no gradient and the top's gradient equals back-propagation within the top",
          w0 == 0.0 and terr <= 1e-12, {"max_abs_W_grad": w0, "top_grad_diff": terr})
    # 3. shuffle is a within-batch permutation
    g = torch.Generator().manual_seed(C.SET["dfa"]["shuffle_seed"])
    perm = torch.randperm(len(c), generator=g)
    net.zero_grad()
    e_s, _ = D.dfa_step(net, Fb, idx, c, perm)
    check(out, "the shuffled error is a permutation of the batch's errors",
          torch.equal(torch.sort(perm).values, torch.arange(len(c))) and torch.allclose(e_s.sum(0), e.sum(0), atol=1e-12))
    # 4. determinism, 25 lines one epoch
    sub = C.first_lines(A["train"], C.SET["c1"]["det_lines"])
    w = [D.train_dfa(sub, A["calib"], False, 1, 1)[0][0] for _ in range(2)]
    records += 2 * C.SET["c1"]["det_lines"]
    check(out, "one epoch on 25 lines is deterministic", torch.equal(w[0], w[1]))
    # 5. learns: 200 lines, one epoch, against C0's frozen-H reference
    sub = C.first_lines(A["train"], C.SET["c1"]["learn_lines"])
    _, _, info = D.train_dfa(sub, A["calib"], False, 1, 1)
    records += C.SET["c1"]["learn_lines"]
    ref = json.loads((HERE / "results" / "testbed" / "TESTBED.json").read_text())["frozen_h_reference_200_lines_1_epoch"]
    check(out, "learns: after one epoch on 200 lines, A calibration >= 0.01 below the frozen-H top",
          info["calib_bpb"] <= ref["calib_bpb"] - C.SET["c1"]["learn_margin"],
          {"dfa": info["calib_bpb"], "frozen_h": ref["calib_bpb"], "cpu_s": info["cpu_s"]})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; records", records)
    C.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
