"""Run LEARNED-BANK V1 histories. One history = one (task, arm, world). Results: one JSON per history.

python run.py --tag TAG --worlds D1,D2 --tasks MOD --arms LEARN,FROZEN --passes 20 --probe-every 5 [--jobs 4]
"""
import argparse
import json
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")

import numpy as np  # noqa: E402

import data  # noqa: E402
import lb1  # noqa: E402

HERE = Path(__file__).resolve().parent
ALL = [(a, b) for a in range(10) for b in range(10)]


def jaccard_matrix(X):
    X = np.asarray(X, float)
    inter = X @ X.T
    n = X.sum(1)
    union = n[:, None] + n[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 1.0)


def centered_alignment(K, labels):
    n = len(labels)
    Y = (np.asarray(labels)[:, None] == np.asarray(labels)[None, :]).astype(float)
    H = np.eye(n) - 1.0 / n
    Kc, Yc = H @ K @ H, H @ Y @ H
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / den) if den > 0 else 0.0


def meters(model, task, world, prev_codes):
    labels = [data.answer(task, a, b) for a, b in ALL]
    codes = [model.codes_for(data.prompt(a, b), lab) for (a, b), lab in zip(ALL, labels)]
    held = set(world["held_pairs"])
    hi = [i for i, p in enumerate(ALL) if p in held]
    ti = [i for i, p in enumerate(ALL) if p not in held]
    out, cur = {}, {}
    for name, k in (("S", 0), ("L", 1)):
        if codes[0][k] is None:
            continue
        X = np.array([c[k] for c in codes])
        cur[name] = X
        K = jaccard_matrix(X)
        lab = np.array(labels)
        gaps = []
        for h in hi:
            same = [K[h, t] for t in ti if lab[t] == lab[h]]
            diff = [K[h, t] for t in ti if lab[t] != lab[h]]
            if same and diff:
                gaps.append(np.mean(same) - np.mean(diff))
        offd = K[np.triu_indices(len(K), 1)]
        out[name] = {"kta_all": round(centered_alignment(K, labels), 5),
                     "held_transfer_gap": round(float(np.mean(gaps)), 5) if gaps else None,
                     "mean_offdiag_jaccard": round(float(offd.mean()), 5)}
        if prev_codes is not None and name in prev_codes:
            P = prev_codes[name]
            d = [float((X[i] * P[i]).sum() / max(1.0, ((X[i] + P[i]) > 0).sum())) for i in ti]
            out[name]["train_code_jaccard_vs_prev"] = round(float(np.mean(d)), 5)
    return out, cur


def accuracy(model, task, world, taught):
    held = set(world["held_pairs"])
    rows = {"held": {"sys": 0, "S": 0, "L": 0}, "taught": {"sys": 0, "S": 0, "L": 0},
            "taught_vs_truth": {"sys": 0, "S": 0, "L": 0}}
    n_h = n_t = 0
    outputs = {}
    for a, b in ALL:
        truth = data.answer(task, a, b)
        r = model.probe(data.prompt(a, b), truth)
        outputs[f"{a}{b}"] = [r["sys"], r["S"], r["L"]]
        if (a, b) in held:
            n_h += 1
            for k in ("sys", "S", "L"):
                rows["held"][k] += int(r[k] == truth)
        else:
            n_t += 1
            for k in ("sys", "S", "L"):
                rows["taught"][k] += int(r[k] == taught[(a, b)])
                rows["taught_vs_truth"][k] += int(r[k] == truth)
    res = {grp: {k: round(v / (n_h if grp == "held" else n_t), 4) for k, v in d.items()} for grp, d in rows.items()}
    if model.L is None:
        for grp in res:
            res[grp]["L"] = None
    return res, outputs


def history(job):
    task, arm, wname, passes, probe_every, settings, outdir = job
    t0 = time.process_time()
    world = data.make_world(wname)
    taught = {p: data.answer(task, *p) for p in world["train_pairs"]}
    if arm == "SHUF":
        taught = data.shuffled_labels(task, world)
    model = lb1.Model(arm, world["model_seed"], **settings)
    order = data.schedule(world, passes)
    per_pass = len(world["train_pairs"])
    checkpoints, prev = [], None

    def snapshot(label, n_pass):
        nonlocal prev
        tp = time.process_time()
        acc, outputs = accuracy(model, task, world, taught)
        met, prev = meters(model, task, world, prev)
        w = None if model.layer is None else {"W1_norm": round(float(np.linalg.norm(model.layer.W1)), 4),
                                              "b1_mean": round(float(model.layer.b1.mean()), 5)}
        checkpoints.append({"label": label, "pass": n_pass, "records": model.records, "acc": acc, "meters": met,
                            "layer": w, "log": {k: round(v, 4) for k, v in model.log.items()},
                            "probe_cpu_s": round(time.process_time() - tp, 2), "outputs": outputs})
        print(json.dumps({"task": task, "arm": arm, "world": wname, "label": label, "held": acc["held"],
                          "taught": acc["taught"], "L_meters": met.get("L")}), flush=True)

    snapshot("birth", 0)
    train_cpu = 0.0
    for i, (a, b) in enumerate(order):
        tt = time.process_time()
        model.train(data.prompt(a, b), taught[(a, b)], data.answer(task, a, b))
        train_cpu += time.process_time() - tt
        done = (i + 1) // per_pass
        if (i + 1) % per_pass == 0 and (done % probe_every == 0 or done == passes):
            snapshot(f"pass{done}", done)
    model.rest(3600.0)
    snapshot("final_rest1h", passes)
    res = {"task": task, "arm": arm, "world": wname, "passes": passes, "settings": settings,
           "held_orbits": world["held"], "train_records": model.records,
           "train_cpu_s": round(train_cpu, 2), "train_cpu_per_record_s": round(train_cpu / max(1, model.records), 4),
           "total_cpu_s": round(time.process_time() - t0, 2),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "checkpoints": checkpoints}
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{task}_{arm}_{wname}.json").write_text(json.dumps(res))
    return task, arm, wname, res["total_cpu_s"]


def check_lock(worlds, settings, passes, probe_every):
    """Main worlds (M*) run only under LOCK.json with unchanged code and the locked settings."""
    import hashlib
    if not any(w.startswith("M") for w in worlds):
        return
    lock = json.loads((HERE / "LOCK.json").read_text())
    for rel, digest in lock["code_sha256"].items():
        if hashlib.sha256((HERE / rel).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"LOCK: {rel} changed since the lock")
    s = lock["settings"]
    if settings != {k: s[k] for k in settings} or passes != s["passes"] or probe_every != s["probe_every"]:
        raise SystemExit("LOCK: settings differ from LOCK.json")
    if not set(worlds) <= set(lock["main_worlds"]):
        raise SystemExit("LOCK: unknown main world")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--tasks", default="MOD")
    ap.add_argument("--arms", default="LEARN,FROZEN")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--probe-every", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--eta0", type=float, default=0.003)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    settings = {"hidden": a.hidden, "eta0": a.eta0, "lam": a.lam}
    check_lock(a.worlds.split(","), settings, a.passes, a.probe_every)
    outdir = str(HERE / "results" / a.tag)
    jobs = [(t, arm, w, a.passes, a.probe_every, settings, outdir)
            for t in a.tasks.split(",") for w in a.worlds.split(",") for arm in a.arms.split(",")]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            for r in pool.imap_unordered(history, jobs):
                print("DONE", r, flush=True)
    else:
        for j in jobs:
            print("DONE", history(j), flush=True)


if __name__ == "__main__":
    sys.exit(main())
