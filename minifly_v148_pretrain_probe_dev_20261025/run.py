"""V148 protocol runner: one continuous MiniFly training history on a text, evaluated at nested prefixes, plus the
counting ladder on the same splits.

python run.py --text buster --prefixes 0,2.5,5,10 --calib 2 --held 4 --gen 10 --tag c2_dev
python run.py --text alice  --main          (settings from LOCK.json; refuses if any locked hash differs)
"""
import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

import corpus  # noqa: E402
import ladder  # noqa: E402
import probe  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_lock():
    lock = json.loads((HERE / "LOCK.json").read_text())
    bad = [rel for rel, d in lock["closure_sha256"].items() if sha(ROOT / rel) != d]
    if bad:
        raise SystemExit(f"LOCK: changed since lock: {bad}")
    return lock


def taught_slices(train_prefix, kb=5):
    first, acc = [], 0
    for r in train_prefix:
        if acc >= kb * 1024:
            break
        first.append(r)
        acc += corpus.symbols(r)
    last, acc = [], 0
    for r in reversed(train_prefix):
        if acc >= kb * 1024:
            break
        last.append(r)
        acc += corpus.symbols(r)
    return first, last[::-1]


def unseen_stats(L, T, seen):
    """Mean probability mass on symbols never seen in the taught text, and how often the argmax is such a symbol."""
    Z = L / T
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(1, keepdims=True)
    mask = np.ones(257, bool)
    mask[list(seen)] = False
    return float(P[:, mask].sum(1).mean()), float(np.mean(mask[L.argmax(1)]))


def mf_eval(mem, sp, n_train, gen_n):
    t0 = time.process_time()
    Lc, Yc = probe.collect(mem, sp["calib"])
    T, calib_bpb = probe.calibrate(Lc, Yc)
    Lh, Yh = probe.collect(mem, sp["held"])
    hb, n = probe.bits_at(Lh, Yh, T)
    h1, _ = probe.bits_at(Lh, Yh, 1.0)
    seen = {x for r in sp["train"][:n_train] for x in r} | {corpus.EOS}
    um, ua = unseen_stats(Lh, T, seen)
    # diagnostic only: renormalise over symbols seen in taught text (targets outside that set are excluded and counted)
    sl = sorted(seen)
    pos = {x: i for i, x in enumerate(sl)}
    keep = np.array([y in pos for y in Yh])
    if keep.any() and len(sl) > 1:
        Tc, _ = probe.calibrate(Lc[:, sl][np.array([y in pos for y in Yc])],
                                np.array([pos[y] for y in Yc if y in pos]))
        sb, sn = probe.bits_at(Lh[keep][:, sl], np.array([pos[y] for y in Yh[keep]]), Tc)
        seen_diag = {"held_bpb_renorm_seen": sb / sn, "T_renorm_seen": Tc, "held_fraction_target_seen": float(keep.mean())}
    else:
        seen_diag = {}
    out = {"T": T, "calib_bpb": calib_bpb, "held_bpb": hb / n, "held_bpb_T1": h1 / n,
           "held_acc": probe.accuracy(Lh, Yh), "held_symbols": n,
           "held_mass_on_unseen_symbols": um, "held_argmax_unseen_rate": ua, **seen_diag}
    if n_train > 0:
        first, last = taught_slices(sp["train"][:n_train])
        for nm, recs in (("taught_first5k", first), ("taught_last5k", last)):
            L, Y = probe.collect(mem, recs)
            b, m = probe.bits_at(L, Y, T)
            out[nm + "_bpb"] = b / m
            out[nm + "_acc"] = probe.accuracy(L, Y)
    if gen_n:
        gens = []
        for r in [r for r in sp["held"] if len(r) > 10][:gen_n]:
            g, why = probe.generate(mem, r[:10])
            gens.append({"prefix": r[:10].decode("latin-1"), "true": r[10:].decode("latin-1"),
                         "own": g.decode("latin-1"), "stop": why})
        out["generations"] = gens
    out["eval_cpu_s"] = round(time.process_time() - t0, 1)
    return out


def ladder_eval(sp, n_train):
    out = {}
    train = sp["train"][:n_train]
    first, last = taught_slices(train) if n_train else ([], [])
    for mod in ladder.ladder():
        mod.fit(train)
        b, n = ladder.bits(mod, sp["held"])
        row = {"held_bpb": b / n}
        if n_train:
            for nm, recs in (("taught_first5k", first), ("taught_last5k", last)):
                bb, nn = ladder.bits(mod, recs)
                row[nm + "_bpb"] = bb / nn
        out[mod.name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prefixes", default="0,5,10,20,40")
    ap.add_argument("--calib", type=float, default=5)
    ap.add_argument("--held", type=float, default=10)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--passes", default="1", help="total passes over the full train pool, evaluated after each")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.main:
        lock = check_lock()
        s = lock["settings"]
        if a.text != lock["main_text"]:
            raise SystemExit("LOCK: main text differs")
        a.prefixes, a.calib, a.held, a.gen, a.tag = s["prefixes"], s["calib_kb"], s["held_kb"], s["gen"], "main"
        a.passes = s["passes"]
    kbs = [float(x) for x in str(a.prefixes).split(",")]
    passes = sorted(int(x) for x in str(a.passes).split(","))
    tag = a.tag or a.text
    out_dir = HERE / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = corpus.records(corpus.load(a.text))
    sp = corpus.split(recs, max(kbs), a.calib, a.held)
    pf = corpus.prefixes(sp["train"], kbs)
    res_path = out_dir / "RESULTS.json"
    state_path = out_dir / "STATE.pkl"
    if a.resume and res_path.exists() and state_path.exists():
        res = json.loads(res_path.read_text())
        mem, done, passes_done = pickle.loads(state_path.read_bytes())
        print("resuming after", done, "records,", passes_done, "passes", flush=True)
    else:
        res = {"text": a.text, "tag": tag, "prefixes_kb": kbs, "calib_kb": a.calib, "held_kb": a.held,
               "offsets": sp["offsets"], "prefix_records": {str(k): v for k, v in pf.items()},
               "corpus_sha256": corpus.TEXTS[a.text][1], "passes": passes, "checkpoints": [], "train_cpu_s": 0.0}
        mem, done, passes_done = probe.new_memory(), 0, 1
    final = (max(kbs), max(passes))
    plan = [(kb, 1) for kb in kbs] + [(max(kbs), p) for p in passes if p > 1]
    for kb, p in plan:
        n = pf[kb]
        if any(c["kb"] == kb and c.get("pass", 1) == p for c in res["checkpoints"]):
            continue
        tt = time.process_time()
        if p == 1:
            for r in sp["train"][done:n]:
                mem.train(b"", r)
            done = n
        else:
            while passes_done < p:
                for r in sp["train"][:n]:
                    mem.train(b"", r)
                passes_done += 1
        res["train_cpu_s"] += time.process_time() - tt
        cp = {"kb": kb, "pass": p, "records": n, "symbols": sum(corpus.symbols(r) for r in sp["train"][:n]),
              "minifly": mf_eval(mem, sp, n, a.gen if (kb, p) == final else 0), "ladder": ladder_eval(sp, n)}
        res["checkpoints"].append(cp)
        res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        res_path.write_text(json.dumps(res, indent=1))
        state_path.write_bytes(pickle.dumps((mem, done, passes_done)))
        m = cp["minifly"]
        print(json.dumps({"kb": kb, "pass": p, "records": n, "minifly_held_bpb": round(m["held_bpb"], 3),
                          "T": round(m["T"], 3), "held_acc": round(m["held_acc"], 3),
                          "taught_last5k_bpb": round(m.get("taught_last5k_bpb", float("nan")), 3),
                          "unseen_mass": round(m["held_mass_on_unseen_symbols"], 3),
                          "ladder_held_bpb": {k: round(v["held_bpb"], 3) for k, v in cp["ladder"].items()},
                          "eval_cpu_s": m["eval_cpu_s"]}), flush=True)
    res["complete"] = True
    res_path.write_text(json.dumps(res, indent=1))
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
