"""E31 stage 1b agreement, worker A (addendum 2, ruling 1): where hashes differ, the two implementations agree if gradients on one common
batch are within 1e-6 relative when each worker's function is run.

Worker B's gate_term2, rows_norm and calibrate_p2 are taken verbatim from B's commit (git show <commit>:MOD97_LOCAL_RULE_20261106/exp31_switch.py),
exec'd here with the names they need; nothing of B's is paraphrased. B's arch31.forward(..., switch="P") is the sweep's forward, which is
arch26.forward(m, x, True, True) (stage 1: no-switch checks bit-identical on both branches); worker A's arch31.forward(m, x, None) is used for it.
Common batch: the first life batch of the admitted sweep birth, seed 73129.

    python3 e31_agree_1b_A.py <B commit>    -> exp45/e31/E31_AGREEMENT_1B_A.json
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp31_switch as A             # noqa: E402
import torch as T                    # noqa: E402
import arch26, arch31                # noqa: E402

X, E = A.X, A.E


def b_functions(commit):
    src = subprocess.run(["git", "show", f"{commit}:MOD97_LOCAL_RULE_20261106/exp31_switch.py"], cwd=HERE, capture_output=True, text=True, check=True).stdout
    tree = ast.parse(src); want = ("gate_term2", "rows_norm", "gate_rows", "calibrate_p2")
    code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want)

    class _Arch31B:          # B's calibrate_p2 calls arch31.forward(m, x, True, True, switch="P"): the sweep's forward
        @staticmethod
        def forward(m, x, local, through_time, switch=None):
            assert local and through_time and switch == "P"
            return arch31.forward(m, x, None)
    ns = dict(T=T, arch26=arch26, X=X, E=E, Z=slice(256, 512), arch31=_Arch31B)
    exec(code, ns)
    return ns, code


def grads_total(m, x, y, mask, lam, term_fn):
    m.zero_grad()
    logits, it = arch31.forward(m, x, None)
    loss = E.cross_entropy(logits[mask], y[mask])
    (loss + lam * term_fn(it)).backward()
    return {n: p.grad.detach().clone() for n, p in m.named_parameters()}


def main(commit):
    T.set_num_threads(1)
    ns, code = b_functions(commit)
    m, bsha, batch = A.life_birth_and_batch(73129)
    x, y, mask = X.batch_tensor(batch)
    out = dict(worker="A", b_commit=commit, b_functions_sha_of_text=__import__("hashlib").sha256(code.encode()).hexdigest(), birth_sha256=bsha, arms={})
    # the term's value
    logits, it = arch31.forward(m, x, None)
    va = float(A.penalty2(m, it, x)[0]); vb = float(ns["gate_term2"](m, it, x)[0])
    out["term_value"] = dict(A=va, B=vb, rel=abs(va - vb) / abs(vb))
    for arm, s in A.P2_STRENGTH.items():
        # lambda by each worker's own calibration
        swA = A.Switch(arm, 73129); swA.calibrate(m, x, y, mask); lamA = swA.lam
        lamB, recB = ns["calibrate_p2"](m, x, y, mask, s)
        r = dict(lam_A=lamA, lam_B=lamB, lam_rel=abs(lamA - lamB) / lamB)
        # gradients of task + lam * term, each worker's term function, the same lam (B's), and each with its own lam
        for tag, lam in (("same_lambda_B", lamB),):
            gA = grads_total(m, x, y, mask, lam, lambda it: A.penalty2(m, it, x)[0])
            gB = grads_total(m, x, y, mask, lam, lambda it: ns["gate_term2"](m, it, x)[0])
            r[f"grads_max_rel_{tag}"] = max(float((gA[n] - gB[n]).abs().max()) / max(float(gB[n].abs().max()), 1e-30) for n in gB)
        gA = grads_total(m, x, y, mask, lamA, lambda it: A.penalty2(m, it, x)[0])
        gB = grads_total(m, x, y, mask, lamB, lambda it: ns["gate_term2"](m, it, x)[0])
        r["grads_max_rel_each_own_lambda"] = max(float((gA[n] - gB[n]).abs().max()) / max(float(gB[n].abs().max()), 1e-30) for n in gB)
        r["agree"] = r["grads_max_rel_same_lambda_B"] <= 1e-6 and r["grads_max_rel_each_own_lambda"] <= 1e-6
        out["arms"][arm] = r
    out["agree"] = all(v["agree"] for v in out["arms"].values())
    p = HERE / "exp45" / "e31" / "E31_AGREEMENT_1B_A.json"; p.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
