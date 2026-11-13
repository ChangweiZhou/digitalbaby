"""E27, brief step 2: an independent check of the split, written without e27_measure.flood() and without e25_alignment.grads().

Own code: per-position cross-entropy with reduction='none', the four partial losses as means over their own masks, gradients by
torch.autograd.grad on one graph. For the GRU the forward is arch26.forward (the network E26's lives ran), not e25_alignment.forward_rnn.

Checked, under the exact path and under the sign path:
  1. every-byte gradient == w_answer * answer gradient + w_other * other gradient            (to 1e-5 of the largest entry)
  2. every-byte gradient == w_answer * answer + w_letters * letters + w_structural * structural
  3. able to fail: with the shares swapped, (1) does not hold
  4. flood(), amplification and usable as e27_measure.flood() reports them agree with the same figures computed here (relative 1e-4)
  5. for the GRU: e25_alignment.grads (e27's gradient source) agrees with the gradients through arch26.forward (relative 1e-4), both paths
On the two untrained networks and on ONE saved file (exp45/e26/e26_dev_B_73129.pt, checked against its receipt). Only differences and
pass/fail are printed and written, never a saved file's flood or amplification, so nothing is read from a predicted quantity before the run.

  E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 python3 e27_check_split.py"""
import sys, json, pathlib
import torch as T
from torch.nn.functional import cross_entropy
import e25_alignment as A, e27_measure as M27   # first: they set the harness up as e27_measure runs it
import arch17, arch26

SEED = 73129
SAVED = "exp45/e26/e26_dev_B_73129.pt"


def own_grads(m, x, y, masks, local):
    params = [(n, p) for n, p in m.named_parameters()]
    if hasattr(m, "kind"):
        logits, _ = arch26.forward(m, x, local, True)
    else:
        logits, _ = arch17.forward(m, x, local, True)
    ce = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
    out = {}
    for k, mk in masks.items():
        loss = ce[mk].sum() / mk.sum()
        gs = T.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
        out[k] = {n: (g if g is not None else T.zeros_like(p)).detach() for (n, p), g in zip(params, gs)}
    return out


def vec(g, keep=None):
    return T.cat([v.flatten() for n, v in g.items() if keep is None or keep(n)])


def check(m, items, label):
    x, y, amask = A.X.batch_tensor(items); every = y != A.E.BOS; other = every & ~amask
    letters = other & (y >= 65) & (y <= 90); struct = other & ~letters
    masks = dict(answer=amask, other=other, letters=letters, structural=struct, every=every)
    n = float(every.sum()); w = {k: float(v.sum()) / n for k, v in masks.items()}
    res = dict(label=label); g = {}
    for path, local in (("exact", False), ("sign", True)):
        g[path] = own_grads(m, x, y, masks, local); G_ = g[path]
        scale = max(float(v.abs().max()) for v in G_["every"].values())
        two = max(float((G_["every"][k] - (w["answer"] * G_["answer"][k] + w["other"] * G_["other"][k])).abs().max()) for k in G_["every"])
        three = max(float((G_["every"][k] - (w["answer"] * G_["answer"][k] + w["letters"] * G_["letters"][k] + w["structural"] * G_["structural"][k])).abs().max()) for k in G_["every"])
        swapped = max(float((G_["every"][k] - (w["other"] * G_["answer"][k] + w["answer"] * G_["other"][k])).abs().max()) for k in G_["every"])
        res[path] = dict(largest=scale, two_part_error=two, three_part_error=three, shares_swapped_error=swapped,
                         ok=two < 1e-5 * max(1.0, scale) and three < 1e-5 * max(1.0, scale) and swapped > 1e-3 * scale)
    # 4. against e27_measure.flood()
    f = M27.flood(m, items, null_draws=0); diffs = {}
    for sname, keep in M27.SETS.items():
        mine = {}
        for path in ("exact", "sign"):
            fl = w["other"] * float(vec(g[path]["other"], keep).norm()) / (w["answer"] * float(vec(g[path]["answer"], keep).norm()))
            a, b = vec(g[path]["every"], keep), vec(g["exact"]["answer"], keep)
            mine[path] = dict(flood=fl, usable=float((a * b).sum()) / (float(a.norm()) * float(b.norm())))
        mine_amp = mine["sign"]["flood"] / mine["exact"]["flood"]
        rel = lambda u, v: abs(u - v) / max(abs(v), 1e-12)
        diffs[sname] = dict(flood_exact=rel(f[sname]["exact"]["flood"], mine["exact"]["flood"]), flood_sign=rel(f[sname]["sign"]["flood"], mine["sign"]["flood"]),
                            usable_exact=rel(f[sname]["exact"]["usable"], mine["exact"]["usable"]), usable_sign=rel(f[sname]["sign"]["usable"], mine["sign"]["usable"]),
                            amplification=rel(f[sname]["amplification"], mine_amp))
    res["flood_agreement_relative"] = diffs; res["flood_agrees"] = all(v < 1e-4 for d in diffs.values() for v in d.values())
    # 5. the GRU: e25's written-out step against arch26's
    if hasattr(m, "kind"):
        r5 = {}
        for path, local in (("exact", False), ("sign", True)):
            _, ge, _, _ = A.grads(m, x, y, amask, local); go = g[path]["answer"]
            r5[path] = max(float((ge[k] - go[k]).abs().max()) / max(float(go[k].abs().max()), 1e-12) for k in go)
        res["e25_vs_arch26_gradients_max_relative"] = r5; res["e25_agrees_with_arch26"] = all(v < 1e-4 for v in r5.values())
    res["pass"] = res["exact"]["ok"] and res["sign"]["ok"] and res["flood_agrees"] and res.get("e25_agrees_with_arch26", True)
    return res


def main():
    T.set_num_threads(1)
    birth, _ = M27.panels(SEED); out = dict(seed=SEED, items=len(birth), cases=[])
    for kind in ("G0", "G1"):
        m, _ = M27.net(kind, None, SEED); out["cases"].append(check(m, birth, f"{kind} untrained"))
    m, rec = M27.net("G1", SAVED, SEED); r = check(m, birth, f"G1 saved {SAVED}"); r["receipt"] = rec; out["cases"].append(r)
    out["pass"] = all(c["pass"] for c in out["cases"])
    pathlib.Path("exp45/E27_SPLIT_CHECK.json").write_text(json.dumps(out, indent=1))
    for c in out["cases"]:
        print(c["label"], "pass" if c["pass"] else "FAIL", {p: (c[p]["two_part_error"], c[p]["three_part_error"], c[p]["shares_swapped_error"]) for p in ("exact", "sign")},
              "max flood rel diff", max(v for d in c["flood_agreement_relative"].values() for v in d.values()), c.get("e25_vs_arch26_gradients_max_relative"))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
