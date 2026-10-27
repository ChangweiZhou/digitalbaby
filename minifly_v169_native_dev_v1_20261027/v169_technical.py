"""V169 C1 technical (design revision 2, section 4). Run through launch.py:
python launch.py run --stage technical -- /tmp/venv_t/bin/python v169_technical.py
Checks: replay batches are half B and half A positions (A from A's taught text only); continued learning starts from the
given network's exact parameters and leaves it untouched; continued learning is deterministic (10 lines); B symbols unseen
in A score as OTHER; cost per epoch. Writes results/technical/CHECKS.json.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v169_lib as X  # noqa: E402

import torch  # noqa: E402

V, C, L = X.V, X.C, X.L
STAGE = os.environ.get("MINIFLY_STAGE", "unlaunched")


def check(out, name, ok, detail=None):
    out.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, "" if detail is None else detail, flush=True)


def main():
    t0 = time.process_time()
    out, records = [], 0
    data = C.load_rehearsal()
    A, B = data["A"], data["B"]
    a10, b10 = C.first_lines(A["train"], 10), C.first_lines(B["train"], 10)
    (_, _), net0, i0 = L.train(a10, A["calib"], "local", 1, 1)
    records += i0["records"]
    cls = net0.cls
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(X.SET["persist"]["replay_seed"])
    jb, ja = next(X.batches(B["train"], A["train"], True, g1, g2))
    check(out, "a replay batch has equal B and A parts, with A indices inside A's taught text",
          len(jb) == len(ja) == 256 and int(ja.max()) < len(A["train"]["y"]) and int(ja.min()) >= 0)
    before = {k: v.clone() for k, v in net0.state_dict().items()}
    n1, c1 = X.cont(net0, cls, a10, b10, B["calib"], False, 1, 1)
    n2, c2 = X.cont(net0, cls, a10, b10, B["calib"], False, 1, 1)
    records += c1["records"] + c2["records"]
    check(out, "continued learning leaves the starting network untouched",
          all(torch.equal(before[k], v) for k, v in net0.state_dict().items()))
    check(out, "continued learning is deterministic (10 lines, one epoch)",
          all(torch.equal(a, b) for a, b in zip(n1.state_dict().values(), n2.state_dict().values())))
    n3, c3 = X.cont(net0, cls, a10, b10, B["calib"], True, 1, 1)
    records += c3["records"]
    unseen = sorted(set(int(v) for v in B["train"]["y"]) - set(int(v) for v in cls.seen))
    enc = cls.encode(unseen) if unseen else []
    check(out, "B symbols unseen in A are scored as OTHER", all(int(e) == cls.K - 1 for e in enc),
          {"unseen_symbols": unseen, "records_replay_run": c3["records"]})
    check(out, "cost per epoch measured (10-line runs)", True,
          {"cont_cpu_s": c1["cpu_s"], "replay_cpu_s": c3["cpu_s"], "local_cpu_s": i0["cpu_s"]})
    d = HERE / "results" / STAGE
    d.mkdir(parents=True, exist_ok=True)
    (d / "CHECKS.json").write_text(json.dumps({"checks": out, "learner_records": records,
                                              "cpu_s": round(time.process_time() - t0, 1)}, indent=1))
    print(sum(o["pass"] for o in out), "/", len(out), "pass; records", records)
    X.write_closure()
    sys.exit(0 if all(o["pass"] for o in out) else 1)


if __name__ == "__main__":
    main()
