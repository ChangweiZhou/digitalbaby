"""E17 probe (read-only, 9 November 2026): state scale and gate saturation of bank2 weights, birth against life.

The bank's state is h = R h + w c with fixed R up to 0.999, so a unit's state is bounded only by w c / (1 - R): a thousand times the
input scale for the slowest group. The E11 cell's state is a convex combination (h = r h + (1 - r) c) and never leaves [-1, 1].
Under backprop the loss keeps the bank's slow units in check; the sign rule's constant-size steps need not. This probe measures, on
a fixed batch of development items, per rate group of each bank: the largest |h| over the sequence, the mean |h| at the last step,
the write gate's mean and its saturated fraction; the readout pre-activation's scale; and each parameter's relative move from birth.

    python3 e17_probe_state.py bank2 73129 e17_dev_bank2_73129.pt e17_life_T_bank2_raw65536_73129.pt e17_life_D_bank2_73129.pt ...

Writes exp45/PROBE_E17_<arch>_<seed>.json and prints one table per weight file.
"""
import json
import sys
from pathlib import Path
from random import Random

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arch17 as A
import exp11_cell as E
import exp45 as X

arch, seed, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
A.install(arch)
rng = Random(1)
items = [E.render(t, a, b, 12, rng) for t in E.DEV for a, b in E.split(t, .5, seed)[0][:16]]
x, y, mask = X.batch_tensor(items)
birth = T.load(E.OUT / files[0])
out = {}


def groups(m):
    vals = sorted(set(float(v) for v in m.R))
    return {f"{v:g}": (m.R == v) for v in vals}


@T.no_grad()
def probe(path):
    m = A.new_net(seed, arch); sd = T.load(E.OUT / path); m.load_state_dict(sd); m.eval()
    logits, it = A.forward(m, x, False, through_time=True)
    res = {"file": path}
    for tag, hs, w in (("bank1", it["hs"], it["w"]),) + ((("bank2", it["hs2"], it["w2"]),) if arch == "bank2" else ()):
        Hs = T.stack(hs, 1)          # (B, L, C)
        g = {}
        for name, msk in groups(m).items():
            h = Hs[:, :, msk]; wg = w[:, :, msk]
            g[name] = dict(max_abs_h=float(h.abs().max()), mean_abs_h_last=float(h[:, -1].abs().mean()),
                           mean_w=float(wg.mean()), w_above_099=float((wg > .99).float().mean()), w_below_001=float((wg < .01).float().mean()))
        res[tag] = g
    z1 = T.tanh(E.lin(m.book(x), m.l1, False)) if hasattr(E, "lin") else None
    Hs1 = T.stack(it["hs"], 1); pre = m.lm(Hs1)
    if arch == "bank2":
        pre = pre + m.lm2(T.stack(it["hs2"], 1))
    res["readout_from_state"] = dict(mean_abs=float(pre.abs().mean()), frac_abs_above_3=float((pre.abs() > 3).float().mean()), max_abs=float(pre.abs().max()))
    res["param_rel_move_from_birth"] = {n: float((sd[n] - birth[n]).norm() / (birth[n].norm() + 1e-12)) for n in sd if n.endswith("weight")}
    res["param_norm"] = {n: float(sd[n].norm()) for n in sd if n.endswith("weight")}
    pred = logits.argmax(-1); res["masked_acc_dev_batch"] = float((pred[mask] == y[mask]).float().mean())
    return res


for f in files:
    r = probe(f); out[f] = r
    print(f"\n== {f}   masked byte acc on dev batch {r['masked_acc_dev_batch']:.3f}   readout|pre| mean {r['readout_from_state']['mean_abs']:.2f} max {r['readout_from_state']['max_abs']:.1f} frac>3 {r['readout_from_state']['frac_abs_above_3']:.2f}")
    for tag in ("bank1", "bank2"):
        if tag in r:
            print(f"  {tag}: " + "  ".join(f"R={k}: max|h| {v['max_abs_h']:.2f} mean|h|_last {v['mean_abs_h_last']:.2f} w {v['mean_w']:.2f} sat↑{v['w_above_099']:.2f} sat↓{v['w_below_001']:.2f}" for k, v in r[tag].items()))
    print("  rel move from birth: " + "  ".join(f"{n.split('.')[0]} {v:.2f}" for n, v in r["param_rel_move_from_birth"].items()))
(E.OUT / f"PROBE_E17_{arch}_{seed}.json").write_text(json.dumps(out, indent=1))
