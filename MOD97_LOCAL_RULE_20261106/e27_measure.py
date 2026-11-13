"""E27: two measurements on saved weights, no training (13 Nov 2026, review 9.113). Suggested by the audit session's note of 13 November.

One hypothesis for two failures. Weight signs carry no magnitudes. (1) Under every-byte targets the error from bytes that cannot be predicted
might then flood back undamped and swamp the answer's error (E24's collapse). (2) Credit might not favour the pathways that already do the
work, which would show as worse alignment on composition items than on single operations (E26's weaker carry-over).

A. Flood. Under every-byte targets the loss is a weighted sum of an answer part and a part on all other bytes. Both parts' gradients are computed
   separately, under the exact path and under the sign path, on birth items (mod-89 operations at k = 12). Per parameter tensor:
     flood = |other part's gradient| / |answer part's gradient|   (each weighted by its share of positions)
     amplification = flood under signs / flood under the exact path
     usable = cosine(every-byte gradient under that path, the EXACT answer-only gradient)
B. Alignment by item kind. On end-of-life files of the GRU (E26 and its crossover), answer targets: cosine between the sweep's gradient and the
   true gradient, over all parameters and per tensor, on single operations (mod 97, k = 12), on new instances of taught operator pairs, and on
   untaught plus-minus pairs.

  E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0 python3 e27_measure.py fixtures <seed>   untrained nets only
  ... python3 e27_measure.py run <seed>     writes exp45/E27_MEASURE_<seed>.json; files that are absent are listed, not guessed"""
import sys, json, hashlib, pathlib
import torch as T
import e25_alignment as A
G, E, X, arch20 = A.G, A.E, A.X, A.arch20
N = 128      # items per panel half, as in E25; fixed before any saved file was read

FILES_A = {   # label: (network, path)
    "two-bank untrained": ("G0", None),
    "two-bank born by backprop, answer targets": ("G0", "exp45/e20b/e20_dev_G0_{s}.pt"),
    "two-bank born by the sweep, answer targets": ("G0", "exp45/e17_dev_bank2_nbd_bT_tall_{s}.pt"),
    "two-bank born by backprop, every byte": ("G0", "exp45/e24/e17_dev_bank2_nbd_lossall_{s}.pt"),
    "two-bank born by the sweep, every byte (collapsed)": ("G0", "exp45/e24/e17_dev_bank2_nbd_bT_tall_lossall_{s}.pt"),
    "two-bank born by the sweep, balanced": ("G0", "exp45/e24/e17_dev_bank2_nbd_bT_tall_lossall_bal_{s}.pt"),
    "GRU untrained": ("G1", None),
    "GRU born by backprop": ("G1", "exp45/e26/e26_dev_B_{s}.pt"),
    "GRU born by the sweep": ("G1", "exp45/e26/e26_dev_S_{s}.pt"),
}
FILES_B = {   # birth -> update during the life
    "sweep -> sweep": "exp45/e26/e26_life_S_{s}.pt", "backprop -> backprop": "exp45/e26/e26_life_B_{s}.pt",
    "sweep -> backprop": "exp45/e26x/e26x_life_SB_{s}.pt", "backprop -> sweep": "exp45/e26x/e26x_life_BS_{s}.pt",
}


def net(kind, path, seed):
    m = arch20.new_net(seed, kind)
    if path is not None:
        m.load_state_dict(T.load(path))
    return m


def norm(g):
    return float(T.cat([v.flatten() for v in g.values()]).norm())


def cos(a, b):
    va, vb = T.cat([v.flatten() for v in a.values()]), T.cat([v.flatten() for v in b.values()]); na, nb = float(va.norm()), float(vb.norm())
    return float((va * vb).sum()) / (na * nb) if na > 0 and nb > 0 else None


def flood(m, items):
    x, y, amask = X.batch_tensor(items); allmask = y != E.BOS; other = allmask & ~amask
    wa, wo = float(amask.sum()) / float(allmask.sum()), float(other.sum()) / float(allmask.sum())
    out = dict(positions=dict(answer=int(amask.sum()), other=int(other.sum())), loss={}, per_tensor={}, all_parameters={})
    g = {}
    for path, local in (("exact", False), ("sign", True)):
        la, ga, _, _ = A.grads(m, x, y, amask, local); lo, go, _, _ = A.grads(m, x, y, other, local); lall, gall, _, _ = A.grads(m, x, y, allmask, local)
        g[path] = dict(ans=ga, oth=go, all=gall); out["loss"] = dict(answer=la, other=lo, every_byte=lall)
        # the decomposition must be exact: every-byte gradient = share-weighted sum of the two parts
        out.setdefault("decomposition_error", {})[path] = max(float((gall[n] - (wa * ga[n] + wo * go[n])).abs().max()) for n in gall)
    for path in ("exact", "sign"):
        out["all_parameters"][path] = dict(flood=wo * norm(g[path]["oth"]) / max(wa * norm(g[path]["ans"]), 1e-30),
                                           usable=cos(g[path]["all"], g["exact"]["ans"]))
    out["all_parameters"]["amplification"] = out["all_parameters"]["sign"]["flood"] / max(out["all_parameters"]["exact"]["flood"], 1e-30)
    for n in g["exact"]["ans"]:
        if not n.endswith("weight"): continue
        f = {p: wo * float(g[p]["oth"][n].norm()) / max(wa * float(g[p]["ans"][n].norm()), 1e-30) for p in ("exact", "sign")}
        out["per_tensor"][n] = dict(flood_exact=f["exact"], flood_sign=f["sign"], amplification=f["sign"] / max(f["exact"], 1e-30),
                                    usable_exact=A.cosine(g["exact"]["all"][n], g["exact"]["ans"][n]), usable_sign=A.cosine(g["sign"]["all"][n], g["exact"]["ans"][n]))
    return out


def by_kind(m, panels):
    out = {}
    for name, items in panels.items():
        a = A.alignment(m, items, "answer")
        out[name] = dict(all_parameters=a["all_parameters"], loss=a["loss"], per_tensor={n: v["cosine"] for n, v in a["parameters"].items() if n.endswith("weight")})
    return out


def panels(seed):
    P, comp = G.panels(seed)
    birth = P["add89"][12][:N] + P["sub89"][12][:N]
    life = dict(single_operations=P["add97"][12][:N] + P["sub97"][12][:N], taught_pairs_new_instances=comp["within"][:2 * N], untaught_plus_minus=comp["+-"][:N] + comp["-+"][:N])
    return birth, life


def fixtures(seed):
    out = {}; ok = True; birth, life = panels(seed)
    for kind in ("G0", "G1"):
        m = net(kind, None, seed); f = flood(m, birth)
        # able to fail: with the two masks swapped in the weighting, the decomposition must not hold
        x, y, amask = X.batch_tensor(birth); allmask = y != E.BOS; other = allmask & ~amask
        wa = float(amask.sum()) / float(allmask.sum()); _, ga, _, _ = A.grads(m, x, y, amask, True); _, go, _, _ = A.grads(m, x, y, other, True); _, gall, _, _ = A.grads(m, x, y, allmask, True)
        wrong = max(float((gall[n] - ((1 - wa) * ga[n] + wa * go[n])).abs().max()) for n in gall)
        scale = max(float(v.abs().max()) for v in gall.values())
        k = by_kind(m, life)
        out[kind] = dict(decomposition_error=f["decomposition_error"], with_shares_swapped=wrong, largest_gradient=scale, positions=f["positions"],
                         untrained_flood=f["all_parameters"], untrained_alignment={n: v["all_parameters"] for n, v in k.items()})
        good = max(f["decomposition_error"].values()) < 1e-5 * max(1.0, scale) and wrong > 1e-3 * scale and all(v["all_parameters"] is not None for v in k.values())
        out[kind]["pass"] = good; ok = ok and good
    v = {"a": T.ones(4)}; out["cosine_self"] = cos(v, v); ok = ok and abs(out["cosine_self"] - 1) < 1e-6
    out["pass"] = ok
    pathlib.Path(f"exp45/E27_FIXTURES_{seed}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); sys.exit(0 if ok else 1)


def run(seed):
    birth, life = panels(seed); out = dict(seed=seed, items=2 * N, script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(), A={}, B={}, absent=[])
    for label, (kind, path) in FILES_A.items():
        p = None if path is None else path.format(s=seed)
        if p is not None and not pathlib.Path(p).exists(): out["absent"].append(p); continue
        out["A"][label] = dict(file=p, sha256=(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if p else None), **flood(net(kind, p, seed), birth))
    for label, path in FILES_B.items():
        p = path.format(s=seed)
        if not pathlib.Path(p).exists(): out["absent"].append(p); continue
        out["B"][label] = dict(file=p, sha256=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(), **by_kind(net("G1", p, seed), life))
    pathlib.Path(f"exp45/E27_MEASURE_{seed}.json").write_text(json.dumps(out, indent=1)); print("wrote", f"exp45/E27_MEASURE_{seed}.json", "absent:", out["absent"])


if __name__ == "__main__":
    T.set_num_threads(1)
    (fixtures if sys.argv[1] == "fixtures" else run)(int(sys.argv[2]))
