"""E16-RP same-weights measurement (amendment of 9 Nov 2026, 14:41 UTC; go-ahead in OPUS_TASK_E16F5_E16RP_CLOUD_JOBS_20261109.md @ 1cccb4d).

Read-only: loads each finished life's final weights and scores the OLD composition panels (built exactly as harness 7703d6c's
panels(): held = triples(pair, seed + 1, 600), within = triples(p, seed + 2, 50, exclude=taught[p])) and the NEW ones (the current
harness's panels()) on the same weights. No training, no store (the E8 store's block-6 contents were not saved); the receipt's
block-6 new-panel scores (made with the store) are shown beside the no-store new-panel scores to show what the store adds.

Usage: python3 e16rp_same_weights.py [weights_dir] [seeds...]
  defaults: exp45/ and 73129 93683 116089; writes exp45/E16RP_SAME_WEIGHTS.json only with the defaults.
"""
import json
import sys
from pathlib import Path

import torch as T

import exp16_integrated as H          # the current harness, imported (its life() is not called)

E = H.E
HERE = Path(__file__).resolve().parent
HELD = [f"{a}{b}" for a, b in H.HELD_PAIRS]


def old_panels(seed):
    taught = {p: set(H.triples(p, seed, 2000)) for p in H.TAUGHT_PAIRS}            # split="taught" is the original formula
    trip = {f"{o1}{o2}": H.triples((o1, o2), seed + 1, 600) for o1, o2 in H.HELD_PAIRS}
    trip["within"] = [tr for p in H.TAUGHT_PAIRS for tr in H.triples(p, seed + 2, 50, exclude=taught[p])]
    pairs_within = [p for p in H.TAUGHT_PAIRS for _ in range(50)]
    comp = {n: [H.render_pair(o1, o2, *tr) for (o1, o2), tr in zip([(n[0], n[1])] * 600, trip[n])] for n in HELD}
    comp["within"] = [H.render_pair(*p, *tr) for p, tr in zip(pairs_within, trip["within"])]
    return comp, trip, taught


def new_triples(seed):
    taught = {p: set(H.triples(p, seed, 2000)) for p in H.TAUGHT_PAIRS}
    allt = frozenset().union(*taught.values())
    trip = {f"{o1}{o2}": H.triples((o1, o2), seed, 600, exclude=allt, split="held") for o1, o2 in H.HELD_PAIRS}
    trip["within"] = [tr for p in H.TAUGHT_PAIRS for tr in H.triples(p, seed, 50, exclude=allt, split="within")]
    return trip, taught


def pc(d):
    return dict(n=d["n"], final=100 * d["final"] / d["n"], exact=100 * d["exact"] / d["n"])


def run(wdir, seed, receipt=None):
    m = E.new_net(seed)
    m.load_state_dict(T.load(wdir / f"e16_life_T_raw65536_{seed}.pt"))
    m.eval()
    oldc, oldt, taught = old_panels(seed)
    _, newc = H.panels(seed)
    newt, _ = new_triples(seed)
    assert all(newc[n] == [H.render_pair(n[0], n[1], *tr) for tr in newt[n]] for n in HELD), "new triples do not rebuild the harness's panels"
    allt = frozenset().union(*taught.values())
    disj = dict(old_pm_in_taught_gt=sum(tr in taught[("-", ">")] for tr in oldt["+-"]) if ("-", ">") in taught else None,
                old_in_any_taught={n: sum(tr in allt for tr in oldt[n]) for n in oldt},
                new_in_any_taught={n: sum(tr in allt for tr in newt[n]) for n in newt},
                old_vs_new_shared={n: len(set(oldt[n]) & set(newt[n])) for n in oldt})
    res = {"old": {n: pc(H.score(m, it)) for n, it in oldc.items()}, "new": {n: pc(H.score(m, it)) for n, it in newc.items()}}
    diff = {n: dict(final=res["new"][n]["final"] - res["old"][n]["final"], exact=res["new"][n]["exact"] - res["old"][n]["exact"]) for n in res["old"]}
    mad = dict(final=sum(abs(diff[n]["final"]) for n in HELD) / 4, exact=sum(abs(diff[n]["exact"]) for n in HELD) / 4)
    out = dict(seed=seed, weights=f"e16_life_T_raw65536_{seed}.pt", disjointness=disj, scores=res, diff_new_minus_old=diff, mean_abs_diff_held=mad,
               p16rp3=dict(pm_within_5=abs(diff["+-"]["final"]) <= 5, mad_under_3=mad["final"] < 3))
    if receipt and receipt.exists():
        st = json.loads(receipt.read_text())["stages"]
        if len(st) == 7:
            out["receipt_stage6_new_with_store"] = {n: pc(st[6]["scores"]["compose"][n]) for n in st[6]["scores"]["compose"]}
    return out


def main():
    wdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "exp45"
    seeds = [int(s) for s in sys.argv[2:]] or [73129, 93683, 116089]
    default = len(sys.argv) <= 2 and wdir == HERE / "exp45"
    print("taught pairs:", H.TAUGHT_PAIRS, "held:", H.HELD_PAIRS, flush=True)
    allres = []
    for s in seeds:
        r = run(wdir, s, wdir / f"e16_life_T_raw65536_{s}.json")
        allres.append(r)
        d = r["disjointness"]
        print(f"seed {s}: old +- triples in taught -> set: {d['old_pm_in_taught_gt']}/600; new panels in any taught set: {d['new_in_any_taught']}", flush=True)
        for n in HELD + ["within"]:
            o, w = r["scores"]["old"][n], r["scores"]["new"][n]
            print(f"  {n:7s} final old {o['final']:5.1f} new {w['final']:5.1f} diff {w['final'] - o['final']:+5.1f} | exact old {o['exact']:5.1f} new {w['exact']:5.1f}", flush=True)
        print(f"  mean |diff| held (final) {r['mean_abs_diff_held']['final']:.2f}; P16RP.3 {r['p16rp3']}", flush=True)
    if default:
        (HERE / "exp45" / "E16RP_SAME_WEIGHTS.json").write_text(json.dumps(dict(
            note="same final weights, old (7703d6c panels(), colliding) vs new (432e90fb panels(), operand-disjoint) composition panels; no store; read-only",
            results=allres), indent=1))


if __name__ == "__main__":
    main()
