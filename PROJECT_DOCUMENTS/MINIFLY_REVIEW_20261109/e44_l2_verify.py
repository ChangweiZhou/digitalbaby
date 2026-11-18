"""An independent verification of the final report of E44, job L2, release 1, "the first long lives" (E44_L2_REPORT.md), at the
report's commit 45e9ba16 (the commit named first, 4cd1d906, is its parent; the two differ only in the report and in the trial's report).
Behind E44_L2_VERIFICATION_20261118.md.

Read-only.  Nothing is trained and no file of another branch is executed: every file is read with `git show <commit>:<path>` into
memory, and trees are listed with `git ls-tree -r --name-only` (or plain `git ls-tree`, which reads only tree objects).  The world is
the reviewer's second implementation (e44_world_reader.py); D2 to D10 are computed with the reviewer's part J (e44_l2_checks.l2_quantities)
and the list keys with the reviewer's part K functions (rows_note, keys_of_list); D9, D14, D15, the intervals, the readings, the probe
and the rest are computed here.  Standard library and numpy only.  Run from the repository root, in one command:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e44_l2_verify.py

It writes only into the scratch folder named in SCR (each saved state read in part H is written there, hashed and deleted).

  A  sources; the 32 receipts against the roster note's settings; the roster's jobs
  B  for each world 1 to 8: the stream regenerated (1,024 visits for worlds 1 to 4, 256 for 5 to 8), against every visit's digest in every
     receipt of the world; the event files against the stream; the list keys against the reviewer's reference (part K); the logs and the
     records of starts (visits computed twice); every quantity D2 to D15 at 256 (and 1,024) visits against the 40 score files
  C  between arms (D11), D12, D13, the pooled D14 and D15, the expectations, the predictions, against E44_L2.json; the report's means
     over worlds; sensitivity analyses that are not the note's definitions (the shared births; the exact t; the age of the life in D15)
  D  the tables file, rebuilt row by row
  E  the report's text: every figure set beside the recomputed one; the four corrections of 45e9ba16; the trial report's corrections
  F  the checks under checks/ and the two 8-visit lives of the other births
  G  the learning probe: 24 branches, streams regenerated, first-visit levels recomputed
  H  the norms files against the receipts' states; the saved states (committed, and those resumed from) against the receipts
  I  form: placeholders, names, the committed-file list
"""
import calendar
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

T0 = time.time()
ROOT = Path.cwd()
assert (ROOT / ".git").exists(), "run from the repository root"
REV = ROOT / "PROJECT_DOCUMENTS" / "MINIFLY_REVIEW_20261109"
sys.path.insert(0, str(REV))
import e44_l2_checks as C                                                    # noqa: E402  (the reviewer's parts J and K)
from e44_world_reader import World                                           # noqa: E402

SCR = next(Path("/tmp").glob("*/-home-user-digitalbaby/42667b5c-b7ff-5d5d-9cfa-8990d6cdcbc6/scratchpad")) / "e44_l2_verify" / "run"
REP = "45e9ba16"                    # the report's commit verified (the branch's head when the verification was made)
REP0 = "4cd1d906"                   # the commit first named; its child REP carries four corrections
M = "MOD97_LOCAL_RULE_20261106/"
L2 = M + "exp45/e44/L2/"
LV = L2 + "lives/"
ARMS = ("naive", "replay", "naive-wd0", "replay-wd0")
SEED = {1: 73129, 2: 93683, 3: 116089, 4: 73129, 5: 93683, 6: 116089, 7: 73129, 8: 93683}
PER, VU = 16 * 256, 256
CONTROL_FAMILY = 55
T_EXACT = {2: 2.919985580355516, 3: 2.353363434801823, 7: 1.894578605219186}      # Student's t, 0.95 quantile
T_SCORER = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150, 6: 1.9432, 7: 1.8946}  # l2_score.py's tabled values
ABS = (("1 visit", 1, 1), ("2 to 4", 2, 4), ("5 to 32", 5, 32), ("33 or more", 33, 10 ** 9), ("5 or more", 5, 10 ** 9))
EARLIER = (("1", 1, 1), ("2 to 3", 2, 3), ("4 to 7", 4, 7), ("8 to 15", 8, 15), ("16 or more", 16, 10 ** 9))
ORDER = C.ORDER
WORLDS_B = tuple(int(x) for x in os.environ.get("E44V_WORLDS", "1,2,3,4,5,6,7,8").split(","))   # all eight unless set for a partial run
FREE_LIMIT = 0.9                    # GB that must remain free after a large read
N_CMP = [0]
DIFFS = []
TXT_N = [0]
TXT_DIFFS = []
E = {}                              # figures used later
REC = {}
R = {}                              # (arm, w, V) -> recomputed quantities
UNREAD = []


def stem(arm, w):
    base, wd0 = arm.split("-")[0], arm.endswith("wd0")
    V = 1024 if (w <= 4 and not wd0) else 256
    return f"lw_{base}_w{w}_v{V}" + ("_rho0.5" if base == "replay" else "") + ("_wd0" if wd0 else "")


def life_V(arm, w):
    return 1024 if (w <= 4 and not arm.endswith("wd0")) else 256


def worlds_of(V):
    return (1, 2, 3, 4) if V == 1024 else tuple(range(1, 9))


def git(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, cwd=str(ROOT))
    if check:
        assert r.returncode == 0, (a, r.stderr[:300])
    return r.stdout


def gb(path, ref=REP):
    return git("show", f"{ref}:{path}")


def gjson(path, ref=REP):
    return json.loads(gb(path, ref))


def free_gb():
    st = os.statvfs("/home/user")
    return st.f_bavail * st.f_frsize / 1e9


def gnpz(path, ref=REP, label=None):
    """A large file, read only if at least FREE_LIMIT GB stays free (25 MB allowed for the file itself); None otherwise."""
    if free_gb() < FREE_LIMIT + 0.025:
        UNREAD.append(label or path); print(f"   NOT READ (free disk {free_gb():.2f} GB): {path}")
        return None
    f = np.load(io.BytesIO(gb(path, ref)))
    return {k: f[k] for k in f.files}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def arrays_sha(arrs):                       # the receipts' formula for an arrays digest (the runner's text, reimplemented)
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}".encode()); h.update(a.tobytes())
    return h.hexdigest()


def cmp(label, mine, theirs, tol=1e-9):
    """One recomputed figure against the figure of a committed file; a difference is recorded and printed."""
    N_CMP[0] += 1
    if isinstance(mine, float) and math.isnan(mine):
        mine = None
    if isinstance(theirs, float) and math.isnan(theirs):
        theirs = None
    if mine is None or theirs is None:
        ok = mine is None and theirs is None
    elif isinstance(mine, str) or isinstance(theirs, str):
        ok = mine == theirs
    elif isinstance(mine, (bool, np.bool_)) or isinstance(theirs, (bool, np.bool_)):
        ok = bool(mine) == bool(theirs)
    elif isinstance(mine, (int, np.integer)) and isinstance(theirs, (int, np.integer)):
        ok = int(mine) == int(theirs)
    else:
        ok = (math.isnan(float(mine)) and math.isnan(float(theirs))) or abs(float(mine) - float(theirs)) <= tol
    if not ok:
        DIFFS.append((label, mine, theirs))
        print(f"   DIFFERS: {label}: recomputed {mine}, committed {theirs}")
    return ok


def txt(label, reported, mine, dec):
    """A figure of the report's text (as printed, with `dec` decimals) beside the recomputed value."""
    TXT_N[0] += 1
    d = float(mine) - float(reported)
    ok = abs(d) <= 0.5 * 10 ** -dec + 1e-9
    if not ok:
        TXT_DIFFS.append((label, reported, mine, d))
    return ok


def tmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def interval90(xs, scorer=False):
    """The roster note's D11: the mean over the worlds and its two-sided 90% interval, Student's t with n - 1 degrees of freedom
    (the exact quantile; with scorer=True the scorer's tabled value, for the comparison with the committed files)."""
    xs = [float(x) for x in xs if x is not None]
    n = len(xs); m = float(np.mean(xs)); sd = float(np.std(xs, ddof=1))
    t = (T_SCORER if scorer else T_EXACT)[n - 1]
    h = t * sd / math.sqrt(n)
    return dict(n=n, mean=m, sd=sd, low=m - h, high=m + h, values=xs)


T_GAP = []


def cmp_iv(label, xs, theirs):
    mine, tab = interval90(xs), interval90(xs, True)
    for k in ("n", "mean", "sd", "low", "high"):
        cmp(f"{label} {k}", tab[k], theirs[k])
    for k in ("low", "high"):
        T_GAP.append((label + " " + k, mine[k] - tab[k]))
    if verdict(mine) != verdict(tab):
        DIFFS.append((label + " verdict under the exact t", verdict(mine), verdict(tab)))
    return mine


def verdict(iv, up="higher", down="lower"):
    return up if iv["low"] > 0 else down if iv["high"] < 0 else "not distinguished"


def sg(x, d=4):
    return "—" if x is None else f"{x:+.{d}f}"


def f3(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ivs(iv, d=4):
    return f"{iv['mean']:+.{d}f} [{iv['low']:+.{d}f}, {iv['high']:+.{d}f}]"


# ================================================================ A. sources, receipts, roster
def part_a():
    print("A. Sources; the 32 receipts against the roster note's settings; the roster's jobs")
    print(f"   commits: report {git('rev-parse', REP).decode().strip()}; first named {git('rev-parse', REP0).decode().strip()}; "
          f"{REP0} is the parent of {REP}: {git('rev-parse', REP + '^').decode().strip().startswith(git('rev-parse', REP0).decode().strip()[:8])}; "
          f"head of the report's branch now {git('rev-parse', next(x for x in git('for-each-ref', '--format=%(refname)', 'refs/remotes/origin/').decode().split() if x.endswith('/e44-l2-first-long-lives'))).decode().strip()[:8]}")
    ck = gjson(L2 + "checks/E44_L2_CHECKS.json")
    ref_src = ck["sources_sha256"]
    now = {k: sha(gb(M + k)) for k in ref_src}
    E["src_equal"] = sum(now[k] == ref_src[k] for k in ref_src)
    print(f"   the twelve sources at {REP} against the check receipt's sources_sha256: {E['src_equal']} of {len(ref_src)} equal; "
          f"sw_run.py {now['smallworld/sw_run.py'][:8]}…; l2_score.py at {REP} {sha(gb(M + 'smallworld/l2_score.py'))[:8]}…")
    # files of smallworld/ changed after the checks' commit (tree objects only)
    def tree(ref):
        out = git("ls-tree", ref, M + "smallworld/").decode().splitlines()
        return {ln.split("\t")[1]: ln.split()[2] for ln in out}
    t0, t1 = tree(ck["git"]["commit"]), tree(REP)
    changed = sorted(p.split("/")[-1] for p in t0 if p in t1 and t0[p] != t1[p]); added = sorted(p.split("/")[-1] for p in t1 if p not in t0)
    print(f"   smallworld/ between the checks' commit {ck['git']['commit'][:8]} and {REP}: changed {changed or 'none'}; added {added or 'none'}")
    # the reviewer's runner change base: sw_run.py at 39e4e594
    jobs = gjson(L2 + "roster_jobs.json")
    lives_jobs = [j for j in jobs if "stem" in j]; probe_jobs = [j for j in jobs if "needs" in j]
    print(f"   roster_jobs.json: {len(jobs)} jobs: {len(lives_jobs)} lives and {len(probe_jobs)} probe branches")
    want_stems = [stem(a, w) for a in ("naive", "replay") for w in (1, 2, 3, 4)] + [stem(a, w) for a in ("naive", "replay") for w in (5, 6, 7, 8)] + \
                 [stem(a, w) for a in ("naive-wd0", "replay-wd0") for w in range(1, 9)]
    print(f"   the roster note's 32 lives (2 long arms x worlds 1-4 at 1,024; 2 arms x worlds 5-8 at 256; 2 decay-0 arms x worlds 1-8 at 256) "
          f"equal the jobs' lives as a set: {sorted(want_stems) == sorted(j['stem'] for j in lives_jobs)}")
    want_probe = sorted(f"{stem(a, w)}_br{g}_g52-53-54-55" for a in ("naive", "replay") for w in (1, 2, 3, 4) for g in (64, 256, 1024))
    print(f"   the 24 probe branches (states after 64, 256, 1,024 visits of the eight long lives, families 52-55) equal the jobs': "
          f"{want_probe == sorted(j['id'] for j in probe_jobs)}")
    tot = 0.0; bad = []
    E["loss_max"] = (0, None)
    for arm in ARMS:
        for w in range(1, 9):
            s = stem(arm, w); V = life_V(arm, w); r = gjson(LV + s + ".json"); REC[arm, w] = r
            st = r["settings"]; ln = r["learner"]; wd = 0.0 if arm.endswith("wd0") else 0.25
            rep = arm.startswith("replay")
            ss = {1024: [64, 128, 256, 384, 512, 640, 768, 896, 1024], 256: ([64, 128, 192, 256] if not arm.endswith("wd0") else list(range(8, 257, 8)))}[V]
            job = next(j for j in lives_jobs if j["stem"] == s)
            ok = dict(
                complete=r["complete"] is True and r["final"]["visits"] == V and r["final"]["updates"] == V * 256 and len(r["by_visit"]) == V
                and [b["visit"] for b in r["by_visit"]] == list(range(V)) and all(b["updates"] == 256 for b in r["by_visit"]),
                settings=st["arm"] == arm.split("-")[0] and st["world"] == w and st["visits"] == V and st["rho"] == (0.5 if rep else None)
                and st["lr"] == 0.001 and st["wd"] == wd and st["H"] == 256 and st["U"] == 128 and st["visit_updates"] == 256 and st["list_every"] == 8
                and st["decay_distances"] == [] and st["save_states"] == ss and st["world_kind"] == "lw" and st["in_turn"] is False and st["p_new"] == 0.05,
                learner=ln["rule"] == "B" and ln["local_in_birth"] is False and ln["seed"] == SEED[w] and ln["file"] == f"exp45/e41/e41_B_h256_f50_{SEED[w]}.pt"
                and r["local"] is False,
                optimizer=r["optimizer"]["name"] == "AdamW" and r["optimizer"]["lr"] == 0.001 and r["optimizer"]["weight_decay"] == wd
                and r["optimizer"]["groups"][0]["weight_decay"] == wd and r["optimizer"]["groups"][1]["params"] == ["book.weight"],
                store=(r["replay"] is rep and r["replay_items_per_update"] == (24 if rep else 0)
                       and (not rep or (r["ledger"]["buffer"]["capacity_items"] == 55209 and r["counters"]["max_buffer_items"] == 55209
                                        and r["ledger"]["buffer"]["items_at_end"] == 55209 and r["counters"]["first_replay_update"] == 1
                                        and r["counters"]["replayed_items"] == 24 * (V * 256 - 1)))),
                sources=r["provenance"]["sources_sha256"] == ref_src and r["provenance"]["git"]["sources_modified_vs_commit"] == [],
                host=r["provenance"]["host"]["python"] == "3.13.18" and r["provenance"]["host"]["torch"] == "2.15.0+cpu" and r["provenance"]["host"]["torch_threads"] == 1,
                chain=sha("".join(b["learned_sha256"] for b in r["by_visit"]).encode()) == r["learned_sha256"],
                finite=all(math.isfinite(b[k]) for b in r["by_visit"] for k in ("loss_mean", "loss_first", "loss_last")),
                job=job["args"][:4] == ["life", arm.split("-")[0], str(w), str(V)] and job["args"][job["args"].index("--learner") + 1].endswith(f"e41_B_h256_f50_{SEED[w]}.pt"))
            if not all(ok.values()):
                bad.append((s, [k for k, v in ok.items() if not v]))
            lm = max(b["loss_mean"] for b in r["by_visit"])
            if lm > E["loss_max"][0]:
                E["loss_max"] = (lm, s)
            tot += r["cpu"]["total"]
    print(f"   receipts checked: 32; complete to the last visit, settings, birth, optimizer, store, sources, host, learned-batch chain, finite losses, "
          f"job arguments as prescribed: {not bad} {bad if bad else ''}")
    E["cpu_total"] = tot / 3600
    print(f"   CPU-hours of the 32 lives (receipts' totals): {tot / 3600:.4f}; largest mean loss of a visit {E['loss_max'][0]:.4f} ({E['loss_max'][1]})")
    E["seeds"] = sorted({REC['naive', w]['learner']['seed'] for w in range(1, 9)})
    print(f"   birth seeds of the 8 worlds: " + ", ".join(f"w{w} {REC['naive', w]['learner']['seed']}" for w in range(1, 9)))


# ================================================================ B. per world
def regen(w, V):
    W = World(w); sched = W.schedule(V)
    n = V * PER
    task = np.zeros(n, np.uint16); A = np.zeros(n, np.uint8); B = np.zeros(n, np.uint8); tgt = np.zeros(n, np.int16)
    seen_x = np.zeros(n, np.uint32); blind_x = np.zeros(n, np.uint8); mirror_x = np.zeros(n, np.uint8); wrap_x = np.zeros(n, np.uint8)
    seen = [0] * len(W.task); last_t = [None] * len(W.task)
    life = hashlib.sha256(); digests = []; k = 0
    for v, f in enumerate(sched):
        hv = hashlib.sha256()
        for taught, asked in W.visit(v, f):
            for i, a, b, p, y in asked:
                task[k] = i; A[k] = a; B[k] = b; tgt[k] = (y[0] - 48) * 10 + (y[1] - 48)
                seen_x[k] = seen[i]; blind_x[k] = int(last_t[i] == y); mirror_x[k] = W.mirror(i, a, b); wrap_x[k] = int(W.wrap(i, a, b)); k += 1
            for i, a, b, p, y in taught:
                seen[i] += 1; last_t[i] = y
            for i, a, b, p, y in taught + asked:
                x = bytes([i // 256, i % 256, a, b, len(p)]) + bytes(p) + bytes(c % 256 for c in y)
                hv.update(x); life.update(x)
        digests.append(hv.hexdigest())
    assert k == n
    return W, sched, dict(task=task, a=A, b=B, target=tgt, seen=seen_x, blind=blind_x, mirror=mirror_x, wrap=wrap_x), digests, life.hexdigest()


def later_records(sched, q):
    times, out = {}, []
    lv = {r["visit"]: r for r in q["later_visits"]}
    for v, g in enumerate(sched[:len(q["kind"])]):
        if v in lv:
            r = dict(lv[v]); r["earlier_visits"] = times[g]; out.append(r)
        times[g] = times.get(g, 0) + 1
    return out


def kept_tables(later):
    out = {}
    for name, lo, hi in ABS:
        rs = [r for r in later if lo <= r["absence"] <= hi]
        n = sum(r["n"] for r in rs); k = sum(r["right"] for r in rs)
        out[name] = dict(visits=len(rs), items=n, right=k, share=k / n if n else None, first16=tmean([r["first16"] for r in rs]),
                         last_quarter=tmean([r["last_quarter"] for r in rs]), previous_last_quarter=tmean([r["previous_last_quarter"] for r in rs]))
    return out


def by_earlier(later):
    out = {}
    for name, lo, hi in ABS:
        out[name] = {}
        for en, elo, ehi in EARLIER:
            rs = [r for r in later if lo <= r["absence"] <= hi and elo <= r["earlier_visits"] <= ehi]
            n = sum(r["n"] for r in rs); k = sum(r["right"] for r in rs)
            out[name][en] = dict(visits=len(rs), items=n, right=k, share=k / n if n else None, first16=tmean([r["first16"] for r in rs]),
                                 last_quarter=tmean([r["last_quarter"] for r in rs]))
    return out


def list_shares(la):
    src0 = (la["source"] == 0) & (la["family"] >= 0)
    at, fam, c = la["at_visit"][src0].astype(np.int64), la["family"][src0].astype(np.int64), la["correct"][src0].astype(np.int64)
    key = at * 100 + fam
    order = np.argsort(key, kind="stable"); key, c = key[order], c[order]
    cut = np.nonzero(np.diff(key))[0] + 1
    st = np.concatenate([[0], cut]); en = np.concatenate([cut, [len(key)]])
    return {(int(key[s] // 100), int(key[s] % 100)): (float(c[s:e].mean()), e - s) for s, e in zip(st, en)}


def d9(sh, sched, V):
    acc = {n: dict(points=0, left=0, ratios=[], shares=[]) for n, _, _ in ABS}
    last, nrows = {}, 0
    for v, g in enumerate(sched[:V]):
        last[g] = v
        for f, l in last.items():
            if l == v or (v, f) not in sh:
                continue
            nrows += 1
            for name, lo, hi in ABS:
                if lo <= v - l <= hi:
                    o = acc[name]; o["shares"].append(sh[v, f][0])
                    if sh[l, f][0] == 0:
                        o["left"] += 1
                    else:
                        o["points"] += 1; o["ratios"].append(sh[v, f][0] / sh[l, f][0])
    return {n: dict(points=o["points"], left_out=o["left"], mean=tmean(o["ratios"]), median=float(np.median(o["ratios"])) if o["ratios"] else None,
                    mean_share_right=tmean(o["shares"])) for n, o in acc.items()}, nrows


def classify(op_add, p, a, b, g1, g2, g3, p_last, others_mask_fn, p_control, moduli):
    """Vectorised form of the reviewer's answer_class (e44_l2_checks.py), tried in its order; also the chance line (the never-met
    modulus tried in the place of the family taught last, after the two first classes)."""
    raw = np.where(op_add, a + b, a - b)
    valid = (g1 >= 48) & (g1 <= 57) & (g2 >= 48) & (g2 <= 57) & (g3 == 256)
    x = (g1 - 48) * 10 + (g2 - 48)
    cls = np.full(len(a), 7, np.int64)
    rest = valid.copy()

    def take(mask, k):
        nonlocal rest
        m = rest & mask; cls[m] = k; rest = rest & ~mask
    take(x == np.mod(raw, p), 0)
    take((x == np.abs(raw)) | (x == np.mod(raw, 100)), 1)
    after1 = rest.copy()
    take(x == np.mod(raw, p_last), 2)
    earlier = np.zeros(len(a), bool)
    for q, pq in enumerate(moduli):
        earlier |= others_mask_fn(q) & (x == np.mod(raw, pq))
    take(earlier, 3)
    take(x == np.mod(raw, 89), 4)
    take((x == a) | (x == b), 5)
    take(x == np.mod(raw, p_control), 6)
    cls[~valid] = 8
    chance = after1 & (x == np.mod(raw, p_control))
    return cls, chance


def d14_stream(W, sched, V, S, ev):
    mod = np.asarray(W.moduli, np.int64); ctrl = int(mod[CONTROL_FAMILY])
    rows, cls_name, fams, fam_last, plast = [], [], [], [], []
    first_at, lastv = {}, {}
    for v, g in enumerate(sched[:V]):
        if g in first_at:
            a_ = v - lastv[g]
            name = next(n for n, lo, hi in ABS[:4] if lo <= a_ <= hi)
            for k in range(v * PER, v * PER + 16):
                i = int(S["task"][k])
                if i % 4 > 1 or not S["wrap"][k]:
                    continue
                rows.append(k); cls_name.append(name); fams.append(g); fam_last.append(sched[v - 1]); plast.append(v)
        else:
            first_at[g] = v
        lastv[g] = v
    rows = np.asarray(rows, np.int64)
    a = S["a"][rows].astype(np.int64); b = S["b"][rows].astype(np.int64); task = S["task"][rows].astype(np.int64)
    fams = np.asarray(fams, np.int64); fl = np.asarray(fam_last, np.int64); vis = np.asarray(plast, np.int64)
    g1, g2, g3 = (ev[k][rows].astype(np.int64) for k in ("g1", "g2", "g3"))
    fv = np.full(len(mod), 10 ** 9, np.int64)
    for f_, v_ in first_at.items():
        fv[f_] = v_
    others = lambda q: (fv[q] < vis) & (fams != q) & (fl != q)
    # at an absence of 1 the family taught last is the item's own family: that class is empty (the scorer passes None)
    plast_mod = np.where(fl == fams, -1, mod[fl])
    cls, chance = classify(task % 4 == 0, mod[fams], a, b, g1, g2, g3, np.where(plast_mod < 0, 10 ** 6, plast_mod), others, ctrl, mod)
    assert ((cls == 0) == ev["correct"][rows].astype(bool)).all()
    out = {}
    cn = np.asarray(cls_name)
    for name, _, _ in ABS[:4]:
        m = cn == name; n = int(m.sum())
        out[name] = dict(answers=n, shares={c: (float((cls[m] == j).sum() / n) if n else None) for j, c in enumerate(ORDER)},
                         chance=(float(chance[m].sum() / n) if n else None))
    return out


def d14_list(W, sched, V, la):
    mod = np.asarray(W.moduli, np.int64); ctrl = int(mod[CONTROL_FAMILY])
    src0 = (la["source"] == 0) & (la["family"] >= 0) & (la["at_visit"] < V)
    at, fam = la["at_visit"][src0].astype(np.int64), la["family"][src0].astype(np.int64)
    tk, aa, bb = la["task"][src0].astype(np.int64), la["a"][src0].astype(np.int64), la["b"][src0].astype(np.int64)
    g1, g2, g3, cr = (la[k][src0].astype(np.int64) for k in ("g1", "g2", "g3", "correct"))
    sch = np.asarray(sched, np.int64)
    gv = sch[at]
    keep = fam != gv
    at, fam, tk, aa, bb, g1, g2, g3, cr, gv = (x[keep] for x in (at, fam, tk, aa, bb, g1, g2, g3, cr, gv))
    fv = np.full(len(mod), 10 ** 9, np.int64); lastv = np.full((V, len(mod)), -1, np.int64); cur = np.full(len(mod), -1, np.int64)
    for v, g in enumerate(sched[:V]):
        if fv[g] > v:
            fv[g] = v
        cur[g] = v; lastv[v] = cur
    ab = at - lastv[at, fam]
    others = lambda q: (fv[q] <= at) & (fam != q) & (gv != q)
    cls, chance = classify(tk % 4 == 0, mod[fam], aa, bb, g1, g2, g3, mod[gv], others, ctrl, mod)
    assert ((cls == 0) == (cr == 1)).all()
    out = {}
    for name, lo, hi in ABS[:4]:
        m = (ab >= lo) & (ab <= hi); n = int(m.sum())
        out[name] = dict(answers=n, shares={c: (float((cls[m] == j).sum() / n) if n else None) for j, c in enumerate(ORDER)},
                         chance=(float(chance[m].sum() / n) if n else None))
    return out


def quantities(W, sched, ev, la, V):
    q = C.l2_quantities(ev, V)
    fvs = q["first_visits"]
    lev = [r["level"] for r in fvs]
    tp = [len(W.taught[4 * r["family"]]) for r in fvs]
    X = np.stack([np.ones(len(lev)), [math.log(r["n"]) for r in fvs], np.log(tp)], 1)
    co = np.linalg.lstsq(X, np.asarray(lev), rcond=None)[0]
    q["joint"] = dict(intercept=float(co[0]), ln_n=float(co[1]), ln_taught_pairs=float(co[2]))
    q["taught_pairs"] = tp
    q["median_first"] = float(np.median(lev)); q["median_first_2_on"] = float(np.median(lev[1:]))
    q["later"] = later_records(sched, q)
    q["kinds"] = {k: q["kind"].count(k) for k in ("first", "continuation", "return")}
    qq = V - max(1, V // 4)
    q["late_visits_by_kind"] = {k: sum(1 for v in range(qq, V) if q["kind"][v] == k) for k in ("first", "continuation", "return")}
    q["score_trend_returns"] = sum(1 for v in range(V // 2, V) if q["kind"][v] == "return")
    q["kept"] = kept_tables(q["later"]); q["d15"] = by_earlier(q["later"])
    sh = list_shares(la)
    q["d9"], _ = d9(sh, sched, V)
    q["list_rows"] = sum(1 for (at, f) in sh if at < V)                       # the scorer's count: every (visit, family) row group read
    q["d14_list"] = d14_list(W, sched, V, la)
    q["families"] = len(fvs)
    q["reduction_items"] = int(((ev["wrap"][ev["visit"] < V] == 1) & (ev["task"][ev["visit"] < V] % 4 < 2)).sum())
    return q


def compare_score_file(lab, q, sf):
    n0 = len(DIFFS); c0 = N_CMP[0]; V = len(q["kind"])
    cmp(f"{lab} families", q["families"], sf["families_met"]); cmp(f"{lab} reduction items", q["reduction_items"], sf["reduction_items"])
    for k in ("first", "continuation", "return"):
        cmp(f"{lab} kinds {k}", q["kinds"][k], sf["kinds"][k]); cmp(f"{lab} late visits {k}", q["late_visits_by_kind"][k], sf["late_visits_by_kind"][k])
        cmp(f"{lab} late level {k}", q["late_level_by_kind"][k], sf["late_level_by_kind"][k])
    for v in range(V):
        cmp(f"{lab} D2 visit {v}", q["score_by_age"][v], sf["score_by_age"][v]); cmp(f"{lab} blind visit {v}", q["blind_by_age"][v], sf["blind_by_age"][v])
        cmp(f"{lab} kind visit {v}", q["kind"][v], sf["kind"][v])
    cmp(f"{lab} D4", q["late_level"], sf["late_level"]); cmp(f"{lab} blind late", q["blind_late"], sf["blind_late_level"])
    cmp(f"{lab} D5", q["score_trend"], sf["score_trend"]); cmp(f"{lab} D5 returns", q["score_trend_returns"], sf["score_trend_returns"])
    cmp(f"{lab} D5 all visits", q["score_trend_all"], sf["score_trend_all_visits"])
    cmp(f"{lab} D7", q["trend_of_learning"], sf["trend_of_learning"])
    for k in ("intercept", "ln_n", "ln_taught_pairs"):
        cmp(f"{lab} joint {k}", q["joint"][k], sf["trend_of_learning_joint_fit"][k], 1e-7)
    cmp(f"{lab} median D6", q["median_first"], sf["median_first_visit_level"]); cmp(f"{lab} median D6 2 on", q["median_first_2_on"], sf["median_first_visit_level_families_2_on"])
    for a_, b_ in zip(q["first_visits"], sf["first_visits"]):
        for k in ("n", "family", "visit", "level"):
            cmp(f"{lab} first visit n{a_['n']} {k}", a_[k], b_[k])
        cmp(f"{lab} first visit n{a_['n']} taught pairs", q["taught_pairs"][a_["n"] - 1], b_["taught_pairs"])
    mine_l = q["later"]; th_l = sf["later_visits"]
    cmp(f"{lab} later visits count", len(mine_l), len(th_l))
    for a_, b_ in zip(mine_l, th_l):
        for k in ("visit", "family", "absence", "earlier_visits", "right", "n", "first16", "last_quarter", "previous_last_quarter"):
            cmp(f"{lab} later visit {a_['visit']} {k}", a_[k], b_[k])
    for name, _, _ in ABS:
        for k in ("visits", "items", "right", "share", "first16", "last_quarter", "previous_last_quarter"):
            cmp(f"{lab} D8/D10 {name} {k}", q["kept"][name][k], sf["kept_at_return"][name][k])
        for en, _, _ in EARLIER:
            for k in ("visits", "items", "right", "share", "first16", "last_quarter"):
                cmp(f"{lab} D15 {name} {en} {k}", q["d15"][name][en][k], sf["kept_at_return_by_earlier_visits"][name][en][k])
        for k in ("points", "left_out", "mean", "median", "mean_share_right"):
            cmp(f"{lab} D9 {name} {k}", q["d9"][name][k], sf["kept_on_list"][name][k])
    cmp(f"{lab} list rows read", q["list_rows"], sf["list_rows_read"])
    for name, _, _ in ABS[:4]:
        for kind, key in (("stream", "wrong_answers_stream"), ("list", "wrong_answers_list")):
            t = sf[key][name]; mine = q[f"d14_{kind}"][name]
            cmp(f"{lab} D14 {kind} {name} answers", mine["answers"], t["answers"])
            for c in ORDER:
                cmp(f"{lab} D14 {kind} {name} {c}", mine["shares"][c], t["shares"][c])
            cmp(f"{lab} D14 {kind} {name} chance", mine["chance"], t["chance_never_met_modulus_in_place_of_the_last"])
    return len(DIFFS) - n0, N_CMP[0] - c0


def log_runs(s, log, sh):
    vis = []
    for ln in log:
        m = re.match(rf"{re.escape(s)} visit (\d+) .* loss ([0-9.]+) right ([0-9.]+) cpu ([0-9.]+)s", ln)
        if m:
            vis.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    runs, cur = [], [vis[0]]
    for a, b in zip(vis, vis[1:]):
        if b[0] == a[0] + 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    seenv, mism, dup, dup_diff = {}, 0, 0, 0
    for v, lo, x, c in vis:
        mism += abs(sh[v] - x) > 0.0005 + 1e-12
        if v in seenv:
            dup += 1; dup_diff += (seenv[v][0] != x) or (seenv[v][1] != lo)
        seenv[v] = (x, lo)
    return runs, mism, dup, dup_diff, len(seenv)


def part_b():
    print("B. Each world: the stream regenerated, the event files against it, list keys, logs, starts, and every quantity against the score files")
    tot_n, tot_d = 0, 0
    E["resumes"] = []
    for w in WORLDS_B:
        t = time.time()
        Vw = 1024 if w <= 4 else 256
        W, sched, S, digests, life = regen(w, Vw)
        lw = {256: hashlib.sha256()}
        print(f"   world {w} (birth seed {SEED[w]}): stream of {Vw} visits regenerated ({Vw * PER:,} asked items) in {time.time() - t:.0f} s; families met "
              f"in {Vw} visits {len(set(sched))}" + (f", in 256 visits {len(set(sched[:256]))}" if Vw == 1024 else ""))
        E[f"sched_{w}"] = sched
        for arm in ARMS:
            s = stem(arm, w); V = life_V(arm, w); r = REC[arm, w]
            bad = [v for v in range(V) if r["by_visit"][v]["stream_sha256"] != digests[v]]
            sch_ok = [x["group"] for x in r["schedule"]] == sched[:V]
            print(f"      {s}: visits whose stream digest differs from the receipt's: {len(bad)} of {V}; schedule equal to the reviewer's world: {sch_ok}"
                  + (f"; the life's digest equals the receipt's: {life == r['stream_sha256']}" if V == Vw else ""))
            E.setdefault("stream_bad", 0); E["stream_bad"] += len(bad); E.setdefault("stream_checked", 0); E["stream_checked"] += V
        for arm in ARMS:
            s = stem(arm, w); V = life_V(arm, w); r = REC[arm, w]
            ev = gnpz(LV + s + ".npz", label=f"{s} event file")
            la = gnpz(LV + s + "_list.npz", label=f"{s} list file") if ev is not None else None
            if ev is None or la is None:
                print(f"      {s}: event or list file NOT READ; its figures rest on the committed score files alone"); continue
            fsha = arrays_sha(ev) == r["events"]["arrays_sha256"]; lsha = arrays_sha(la) == r["list"]["arrays_sha256"]
            n = len(ev["task"]); St = {k: x[:n] for k, x in S.items()}
            upd_ok = bool((ev["update"] == np.repeat(np.arange(V * VU), 16)).all()); vis_ok = bool((ev["visit"] == np.repeat(np.arange(V), PER)).all())
            g = np.stack([ev["g1"], ev["g2"], ev["g3"]], 1).astype(np.int64); tg = St["target"].astype(np.int64)
            right = (g[:, 0] == 48 + tg // 10) & (g[:, 1] == 48 + tg % 10) & (g[:, 2] == 256)
            cols = {k: bool((ev[k] == St[k]).all()) for k in ("task", "wrap", "mirror", "blind", "seen")}
            cols["correct"] = bool((ev["correct"].astype(bool) == right).all())
            rp = ev["replayed"].astype(np.int64); tk = ev["task"].astype(np.int64)
            order = np.lexsort((np.arange(n), tk)); dd = np.diff(rp[order]); same_t = np.diff(tk[order]) == 0
            # the list keys against the reviewer's reference (part K's functions)
            want = C.rows_note(sched[:V], V, 8, (), vu=256, birth_live="K", born_decay="dist"); got = C.keys_of_list(la)
            keys_ok = sorted(got) == sorted(want)
            E[f"keys_{s}"] = keys_ok
            print(f"      {s}: events and list arrays sha256 equal to the receipt's {fsha and lsha}; rows {n:,}; updates and visits without gap or double "
                  f"{upd_ok and vis_ok}; columns equal to the regenerated stream {all(cols.values())} {'' if all(cols.values()) else cols}; `replayed` never "
                  f"falls {bool((dd[same_t] >= 0).all())} (largest {rp.max():,}); list keys equal the reviewer's reference {keys_ok} ({len(got)} keys, {sum(x[-1] for x in got):,} items)")
            sh_v = ev["correct"].reshape(V, PER).mean(1)
            log = gb(LV + s + ".log").decode().splitlines()
            runs, mism, dup, dup_diff, nlog = log_runs(s, log, sh_v)
            starts = [json.loads(x) for x in gb(LV + s + ".starts.jsonl").decode().splitlines() if x.strip()]
            rstr = "; ".join(f"{x[0][0]} to {x[-1][0]}" for x in runs)
            print(f"         starts {len(starts)}: " + "; ".join(f"{x['wall_utc'][5:16]} from {x['resumed_from_visits']}" for x in starts)
                  + f"; log runs {len(runs)}: {rstr}; visits logged {nlog}; share right against the event file, differences {mism}; visits logged twice {dup}, "
                  f"with a different share or loss {dup_diff}; receipt's resumes " + ("; ".join(f"from {x['done_visits']} ({x['file_sha256'][:8]}…)" for x in r["resumes"]) or "none"))
            E[f"runs_{s}"] = [(x[0][0], x[-1][0]) for x in runs]; E[f"dup_{s}"] = (dup, dup_diff); E[f"starts_{s}"] = starts
            for x in r["resumes"]:
                E["resumes"].append((s, x["done_visits"], x["file_sha256"], x["path"].split("/")[-1], x["wall_utc"]))
            for Vr in ((256, 1024) if V == 1024 else (256,)):
                q = quantities(W, sched, ev, la, Vr)
                q["d14_stream"] = d14_stream(W, sched, Vr, St, ev)
                sf = gjson(L2 + f"scores/{s}_V{Vr}.json")
                nd, nc = compare_score_file(f"{s} V{Vr}", q, sf)
                tot_n += nc; tot_d += nd
                R[arm, w, Vr] = {k: q[k] for k in ("late_level", "blind_late", "late_level_by_kind", "score_trend", "score_trend_all", "trend_of_learning",
                                                   "joint", "median_first", "kept", "d15", "d9", "d14_stream", "d14_list", "families", "kinds",
                                                   "score_by_age", "score_trend_returns", "late_visits_by_kind", "later", "first_visits", "kind")}
                print(f"         {Vr} visits: late level {q['late_level']:.4f}, D5 {q['score_trend']:+.4f}, D7 {q['trend_of_learning']:+.4f}, median D6 "
                      f"{q['median_first']:.4f}; figures compared with the score file {nc:,}, differing {nd}")
            del ev, la
        del S
        print(f"   world {w} done in {time.time() - t:.0f} s; free disk {free_gb():.2f} GB")
    # lives not read: their figures from the score files
    for arm in ARMS:
        for w in range(1, 9):
            for Vr in ((256, 1024) if life_V(arm, w) == 1024 else (256,)):
                if (arm, w, Vr) not in R:
                    sf = gjson(L2 + f"scores/{stem(arm, w)}_V{Vr}.json")
                    R[arm, w, Vr] = from_score(sf, None)
    E["score_file_figures"] = (tot_n, tot_d)
    print(f"   visits whose stream digest was compared: {E['stream_checked']:,}; differing {E['stream_bad']}")
    print(f"   figures of the 40 score files compared: {tot_n:,}; differing: {tot_d}")


def from_score(sf, sched):
    """The quantities of a life whose event file was not read, taken from its score file (marked)."""
    later = sf["later_visits"]
    return dict(late_level=sf["late_level"], blind_late=sf["blind_late_level"], late_level_by_kind=sf["late_level_by_kind"], score_trend=sf["score_trend"],
                score_trend_all=sf["score_trend_all_visits"], trend_of_learning=sf["trend_of_learning"], joint=sf["trend_of_learning_joint_fit"],
                median_first=sf["median_first_visit_level"], kept=sf["kept_at_return"], d15=sf["kept_at_return_by_earlier_visits"], d9=sf["kept_on_list"],
                d14_stream={k: dict(answers=v["answers"], shares=v["shares"], chance=v["chance_never_met_modulus_in_place_of_the_last"]) for k, v in sf["wrong_answers_stream"].items()},
                d14_list={k: dict(answers=v["answers"], shares=v["shares"], chance=v["chance_never_met_modulus_in_place_of_the_last"]) for k, v in sf["wrong_answers_list"].items()},
                families=sf["families_met"], kinds=sf["kinds"], score_by_age=sf["score_by_age"], score_trend_returns=sf["score_trend_returns"],
                late_visits_by_kind=sf["late_visits_by_kind"], later=later, first_visits=sf["first_visits"], from_score_file=True)


# ================================================================ C. between arms, readings, pooled tables
BT = {}
QTY = (("late level (D4)", lambda q: q["late_level"]), ("kept at return, 5 to 32 (D8)", lambda q: q["kept"]["5 to 32"]["share"]),
       ("kept at return, 33 or more (D8)", lambda q: q["kept"]["33 or more"]["share"]), ("kept at return, 5 or more (D8)", lambda q: q["kept"]["5 or more"]["share"]),
       ("score trend (D5)", lambda q: q["score_trend"]), ("trend of learning (D7)", lambda q: q["trend_of_learning"]))
COMPS = (("replay", "naive", 256), ("replay", "naive", 1024), ("naive-wd0", "naive", 256), ("replay-wd0", "replay", 256))


def pool_d15(arm, V):
    out = {}
    for name, lo, hi in ABS[1:4]:
        out[name] = {}
        for en, elo, ehi in EARLIER:
            rs = [r for w in worlds_of(V) for r in R[arm, w, V]["later"] if lo <= r["absence"] <= hi and elo <= r["earlier_visits"] <= ehi]
            n = sum(r["n"] for r in rs); k = sum(r["right"] for r in rs)
            f16 = [r["first16"] for r in rs if r["first16"] is not None and not math.isnan(r["first16"])]
            out[name][en] = dict(returns=len(rs), items=n, right=k, kept=k / n if n else None, first16=float(np.mean(f16)) if f16 else None,
                                 median_visit=float(np.median([r["visit"] for r in rs])) if rs else None)
    return out


def pool_d14(arm, V, kind):
    c = {o: 0.0 for o in ORDER}; n = 0; ch = 0.0
    for w in worlds_of(V):
        for name in ("2 to 4", "5 to 32", "33 or more"):
            t = R[arm, w, V][f"d14_{kind}"][name]
            if t["answers"]:
                for o in ORDER:
                    c[o] += t["shares"][o] * t["answers"]
                n += t["answers"]; ch += t["chance"] * t["answers"]
    return dict(answers=n, shares={o: c[o] / n for o in ORDER}, chance=ch / n)


def part_c():
    print("C. Between arms (D11), D12, D13, pooled D14 and D15, the expectations and predictions, against E44_L2.json")
    FJ = gjson(L2 + "E44_L2.json")
    E["FJ"] = FJ
    # per_life in the JSON against the recomputation
    for key, pl in FJ["per_life"].items():
        arm, w, V = key.split("|"); w, V = int(w), int(V); q = R[arm, w, V]
        cmp(f"per_life {key} D4", q["late_level"], pl["late_level"]); cmp(f"per_life {key} D5", q["score_trend"], pl["score_trend"])
        cmp(f"per_life {key} D7", q["trend_of_learning"], pl["trend_of_learning"]); cmp(f"per_life {key} median D6", q["median_first"], pl["median_first_visit_level"])
        for name, _, _ in ABS:
            cmp(f"per_life {key} kept {name}", q["kept"][name]["share"], pl["kept_at_return"][name]["share"])
            cmp(f"per_life {key} D9 {name}", q["d9"][name]["mean"], pl["kept_on_list"][name]["mean"])
    for a, b, V in COMPS:
        key = f"{a} - {b} at {V}"
        for qn, fn in QTY:
            per = {w: fn(R[a, w, V]) - fn(R[b, w, V]) for w in worlds_of(V) if fn(R[a, w, V]) is not None and fn(R[b, w, V]) is not None}
            t = FJ["between_arms"][key][qn]
            for w in per:
                cmp(f"{key} {qn} w{w}", per[w], t["per_world"][str(w)])
            iv = cmp_iv(f"{key} {qn}", list(per.values()), t["interval"])
            BT[a, b, V, qn] = dict(per=per, iv=iv, verdict=verdict(iv))
            cmp(f"{key} {qn} word", verdict(interval90(list(per.values()), True)), t["word"])
            print(f"   {key}, {qn}: per world " + " ".join(f"{per[w]:+.4f}" for w in per) + f"; mean {ivs(iv)} (n = {iv['n']}, t = {T_EXACT[iv['n'] - 1]:.4f}, "
                  f"half-width {iv['high'] - iv['mean']:.4f}) {verdict(iv)}; worlds above zero {sum(x > 0 for x in per.values())} of {len(per)}")
    l256, l1024 = BT["replay", "naive", 256, "late level (D4)"]["verdict"], BT["replay", "naive", 1024, "late level (D4)"]["verdict"]
    opp = {"higher": "lower", "lower": "higher"}.get(l256)
    closed = (l256 == l1024 and l256 != "not distinguished" and BT["replay", "naive", 1024, "score trend (D5)"]["verdict"] != opp
              and BT["replay", "naive", 1024, "trend of learning (D7)"]["verdict"] != opp)
    E["closed"] = closed
    cmp("D12 closed", closed, FJ["closed_D12"])
    print(f"   D12: replay against naive closed at 1,024 visits: {closed} (late level {l256} at 256, {l1024} at 1,024)")
    # D13
    for arm, V in (("naive", 1024), ("replay", 1024)) + tuple((a, 256) for a in ARMS):
        a5 = [R[arm, w, V]["score_trend"] for w in worlds_of(V)]; a7 = [R[arm, w, V]["trend_of_learning"] for w in worlds_of(V)]
        t = FJ["stuck_or_still_learning"][f"{arm} at {V}"]
        i5 = cmp_iv(f"D13 {arm} {V} D5", a5, t["score_trend"]); i7 = cmp_iv(f"D13 {arm} {V} D7", a7, t["trend_of_learning"])
        word = "still learning" if (i5["low"] > 0 or i7["low"] > 0) else "stuck"
        BT[arm, V, "stuck"] = (i5, i7, word)
        cmp(f"D13 {arm} {V} word", word, t["word"]); cmp(f"D13 {arm} {V} D5 declining", i5["high"] < 0, t["score_trend_declining"])
        cmp(f"D13 {arm} {V} D7 declining", i7["high"] < 0, t["trend_of_learning_declining"])
        print(f"   D13 {arm}, {V} visits ({i5['n']} worlds): D5 {ivs(i5)}; D7 {ivs(i7)}: {word}; declining: {i5['high'] < 0 or i7['high'] < 0}; "
              f"D7 per world " + " ".join(f"{x:+.4f}" for x in a7))
    # pooled D15 and D14
    for a in ARMS:
        for V in (256, 1024):
            if V == 1024 and a.endswith("wd0"):
                continue
            p = pool_d15(a, V); E[f"d15_{a}_{V}"] = p
            t = FJ["d15"][f"{a} at {V}"]
            for name in p:
                for en in p[name]:
                    for k in ("returns", "items", "right", "kept", "first16"):
                        cmp(f"D15 pooled {a} {V} {name} {en} {k}", p[name][en][k], t[name][en][k])
    for a in ("naive", "replay"):
        for V in (256, 1024):
            for kind, key in (("stream", "wrong_answers_stream"), ("list", "wrong_answers_list")):
                p = pool_d14(a, V, kind); E[f"d14_{a}_{V}_{kind}"] = p
                t = FJ["d14_pooled_returns_2_or_more"][f"{a} at {V}, {key}"]
                cmp(f"D14 pooled {a} {V} {kind} answers", p["answers"], t["answers"])
                for o in ORDER:
                    cmp(f"D14 pooled {a} {V} {kind} {o}", p["shares"][o], t["shares"][o])
                cmp(f"D14 pooled {a} {V} {kind} chance", p["chance"], t["chance"])
                print(f"   D14 pooled over worlds and absences of 2 or more, {a}, {V} visits, {kind}: answers {p['answers']:,}; right {p['shares'][ORDER[0]]:.4f}; "
                      f"no reduction {p['shares'][ORDER[1]]:.4f}; last family's modulus {p['shares'][ORDER[2]]:.4f}; an earlier family's {p['shares'][ORDER[3]]:.4f}; "
                      f"89 {p['shares'][ORDER[4]]:.4f}; operand {p['shares'][ORDER[5]]:.4f}; never-met {p['shares'][ORDER[6]]:.4f}; other two-digit {p['shares'][ORDER[7]]:.4f}; "
                      f"chance {p['chance']:.4f}")
    # D14 by class of absence (the note's D14), naive and replay, both lengths: the last family's modulus
    for a in ("naive", "replay"):
        for V in (256, 1024):
            for kind in ("stream", "list"):
                row = []
                for name, _, _ in ABS[1:4]:
                    n = sum(R[a, w, V][f"d14_{kind}"][name]["answers"] for w in worlds_of(V))
                    sl = sum(R[a, w, V][f"d14_{kind}"][name]["shares"][ORDER[2]] * R[a, w, V][f"d14_{kind}"][name]["answers"] for w in worlds_of(V)) / n
                    ch = sum(R[a, w, V][f"d14_{kind}"][name]["chance"] * R[a, w, V][f"d14_{kind}"][name]["answers"] for w in worlds_of(V)) / n
                    row.append(f"{name}: {sl:.3f} of {n:,} (chance {ch:.3f})")
                print(f"   D14 by class, {a}, {V}, {kind}, the last family's modulus: " + "; ".join(row))
    # the trial's stop condition on the score: the mean of D2 over visits 512 to 1,023, world 1, and the lowest visit
    for a in ("naive", "replay"):
        sb = R[a, 1, 1024]["score_by_age"][512:1024]
        print(f"   stop condition 3, {a}, world 1: mean D2 over visits 512 to 1,023 {np.mean(sb):.4f}; lowest visit {min(sb):.4f}; in every long life the lowest "
              f"mean over those visits is {min(np.mean(R[a, w, 1024]['score_by_age'][512:1024]) for w in (1, 2, 3, 4)):.4f}")
    # expectations
    ex = FJ["reviewer_expectations"]
    k1, k2, k3 = list(ex)
    nl = [R["naive", w, 1024]["late_level"] for w in (1, 2, 3, 4)]
    rm = [R["replay", w, 256]["median_first"] for w in range(1, 9)]
    p3 = E["d14_naive_1024_stream"]
    E["exp"] = (nl, rm, p3)
    cmp("(i) mean", float(np.mean(nl)), ex[k1]["mean"]); cmp("(i) holds", 0.40 <= np.mean(nl) <= 0.55, ex[k1]["holds"])
    cmp("(ii) below", sum(x < 0.20 for x in rm), ex[k2]["below"]); cmp("(ii) holds", sum(x < 0.20 for x in rm) >= 3, ex[k2]["holds"])
    cmp("(iii) share", p3["shares"][ORDER[2]], ex[k3]["share"]); cmp("(iii) holds", p3["shares"][ORDER[2]] > 0.10, ex[k3]["holds"])
    print(f"   (i) naive's late level at 1,024 per world {[round(x, 4) for x in nl]}, mean {np.mean(nl):.4f}: holds {0.40 <= np.mean(nl) <= 0.55}")
    print(f"   (ii) replay's median first-visit level at 256 per world {[round(x, 4) for x in rm]}; below 0.20: {sum(x < 0.20 for x in rm)}; lowest {min(rm):.4f}")
    p3b = E["d14_naive_256_stream"]
    print(f"   (iii) naive, returns after 2 or more visits, the last family's modulus: {p3['shares'][ORDER[2]]:.4f} of {p3['answers']:,} at 1,024; "
          f"{p3b['shares'][ORDER[2]]:.4f} of {p3b['answers']:,} at 256 (the note asks both lengths)")
    # predictions
    pr = {"PL2.1": BT["naive", 1024, "stuck"][2] == "stuck", "PL2.2": BT["replay", "naive", 1024, "late level (D4)"]["verdict"] == "higher",
          "PL2.3": BT["replay", "naive", 256, "kept at return, 33 or more (D8)"]["verdict"] == "higher",
          "PL2.4": BT["naive-wd0", "naive", 256, "late level (D4)"]["verdict"] == "not distinguished",
          "PL2.5": BT["naive-wd0", "naive", 256, "kept at return, 33 or more (D8)"]["verdict"] == "higher",
          "PL2.6": BT["replay", 1024, "stuck"][2] == "still learning", "PL2.7": BT["naive", 1024, "stuck"][1]["low"] > 0, "PL2.8": closed,
          "PL2.9": BT["replay-wd0", "replay", 256, "late level (D4)"]["verdict"] == "not distinguished"}
    for k, v in pr.items():
        cmp(f"prediction {k}", v, FJ["predictions"][k]["holds"])
    E["pred"] = pr
    print("   predictions, holds: " + ", ".join(f"{k} {v}" for k, v in pr.items()))
    # counts behind PL2.3 and PL2.5: returns and items after 33 or more visits, per world at 256
    for a in ("naive", "replay", "naive-wd0"):
        print(f"   {a}, 256 visits, absences of 33 or more: returns per world " + " ".join(str(R[a, w, 256]["kept"]["33 or more"]["visits"]) for w in range(1, 9))
              + "; items " + " ".join(str(R[a, w, 256]["kept"]["33 or more"]["items"]) for w in range(1, 9))
              + "; right " + " ".join(str(R[a, w, 256]["kept"]["33 or more"]["right"]) for w in range(1, 9)))
    for cls in ("5 or more", "33 or more"):
        print(f"   256 visits, returns after {cls}, summed over the 8 worlds (the same schedule under every arm): returns "
              f"{sum(R['naive', w, 256]['kept'][cls]['visits'] for w in range(1, 9))}, items {sum(R['naive', w, 256]['kept'][cls]['items'] for w in range(1, 9))}; "
              f"right under naive-wd0 {sum(R['naive-wd0', w, 256]['kept'][cls]['right'] for w in range(1, 9))}, under naive {sum(R['naive', w, 256]['kept'][cls]['right'] for w in range(1, 9))}")
    print("   per world kept after 33 or more, 256 visits, naive-wd0 and naive: " + " ".join(
        f"w{w} {f3(R['naive-wd0', w, 256]['kept']['33 or more']['share'])}/{f3(R['naive', w, 256]['kept']['33 or more']['share'])}" for w in range(1, 9)))
    # the report's means over worlds (section 4)
    for a, V in (("naive", 1024), ("replay", 1024)) + tuple((a, 256) for a in ARMS):
        ws = worlds_of(V); qs = [R[a, w, V] for w in ws]
        il = interval90([q["late_level"] for q in qs], True)
        E[f"sec4_{a}_{V}"] = dict(late=il, ret=np.mean([q["late_level_by_kind"]["return"] for q in qs]), med=np.mean([q["median_first"] for q in qs]),
                                  medmed=np.median([q["median_first"] for q in qs]),
                                  k1=np.mean([q["kept"]["1 visit"]["share"] for q in qs]), k5=np.mean([q["kept"]["5 or more"]["share"] for q in qs]),
                                  f16=np.mean([q["kept"]["5 or more"]["first16"] for q in qs]), d9=np.mean([q["d9"]["5 or more"]["mean"] for q in qs]),
                                  blind=[q["blind_late"] for q in qs])
        x = E[f"sec4_{a}_{V}"]
        print(f"   section 4, {a} at {V} ({len(ws)} worlds): late level mean {il['mean']:.4f} [{il['low']:.4f}, {il['high']:.4f}] (scorer's t); on returns {x['ret']:.4f}; "
              f"median D6, mean over worlds {x['med']:.5f} (median over worlds {x['medmed']:.4f}); kept absence 1 {x['k1']:.4f}; 5 or more {x['k5']:.4f}; first 16 "
              f"(5 or more) {x['f16']:.4f}; D9 (5 or more) {x['d9']:.4f}; blind late level {min(x['blind']):.4f} to {max(x['blind']):.4f}")
    # the median first-visit level between arms (not a reading of the report; recorded for section 6 of the document)
    for V in (256, 1024):
        per = [R["replay", w, V]["median_first"] - R["naive", w, V]["median_first"] for w in worlds_of(V)]
        iv = interval90(per); E[f"medD6_{V}"] = iv
        print(f"   replay - naive, median first-visit level (D6), {V} visits: per world " + " ".join(f"{x:+.3f}" for x in per) + f"; {ivs(iv)} {verdict(iv)}")
    # sensitivity, not the note's definition: worlds that share a birth averaged into one value per birth seed (t with 2 degrees of freedom)
    print("   Sensitivity (not the note's definition): the worlds that share a birth seed averaged into one value per seed; 3 values, t with 2 df")
    for a, b, V in COMPS:
        for qn in ("late level (D4)", "kept at return, 5 to 32 (D8)", "kept at return, 33 or more (D8)", "kept at return, 5 or more (D8)", "score trend (D5)", "trend of learning (D7)"):
            per = BT[a, b, V, qn]["per"]
            vals = [np.mean([per[w] for w in per if SEED[w] == sd]) for sd in (73129, 93683, 116089)]
            iv = interval90(vals); E[f"seed_{a}_{b}_{V}_{qn}"] = iv
            print(f"      {a} - {b}, {V}, {qn}: {ivs(iv)} {verdict(iv)} (the note's: {BT[a, b, V, qn]['verdict']})")
    for arm, V in (("naive", 1024), ("replay", 1024)) + tuple((a, 256) for a in ARMS):
        for nm, k in (("D5", "score_trend"), ("D7", "trend_of_learning")):
            vals = [np.mean([R[arm, w, V][k] for w in worlds_of(V) if SEED[w] == sd]) for sd in (73129, 93683, 116089)]
            iv = interval90(vals); E[f"seed_{arm}_{V}_{nm}"] = iv
            print(f"      D13 {arm}, {V}, {nm}: {ivs(iv)}")
    # the age of the life in D15: median visit of the returns of each cell (the report's section 7) and a split of the returns by age
    for V, arms in ((1024, ("naive", "replay")), (256, ARMS)):
        for a in arms:
            p = E[f"d15_{a}_{V}"]
            print(f"   D15 {a}, {V}: median visit of the returns, one earlier visit / 16 or more: " + "; ".join(
                f"{name} {p[name]['1']['median_visit']:.0f} / {p[name]['16 or more']['median_visit']:.0f} (returns {p[name]['1']['returns']} / {p[name]['16 or more']['returns']})" for name in p))
    for V, arms in ((1024, ("naive", "replay")), (256, ARMS)):
        for a in arms:
            for half, lo, hi in (("first half", 0, V // 2), ("second half", V // 2, V)):
                cells = []
                for en, elo, ehi in EARLIER:
                    rs = [r for w in worlds_of(V) for r in R[a, w, V]["later"] if r["absence"] >= 2 and elo <= r["earlier_visits"] <= ehi and lo <= r["visit"] < hi]
                    n = sum(r["n"] for r in rs); k = sum(r["right"] for r in rs)
                    f = [r["first16"] for r in rs if r["first16"] is not None and not math.isnan(r["first16"])]
                    cells.append(f"{en}: {f3(k / n if n else None)} / {f3(float(np.mean(f)) if f else None)} ({len(rs)})")
                E[f"agesplit_{a}_{V}_{half}"] = cells
                print(f"   D15 by age, {a}, {V}, returns after 2 or more visits in the {half}: kept / first 16 (returns): " + "; ".join(cells))
    # the probe between arms and with age (no reading in the note; recorded for section 6)
    probe = {(r["arm"], r["world"], r["age"]): r["mean"] for r in FJ["probe"]}
    E["probe_fj"] = probe


def C_kind(q, v):
    return q["kind"][v] if "kind" in q else None


# ================================================================ D. the tables file
def part_d():
    print("D. The tables file E44_L2_TABLES.md, rebuilt row by row in the author's formats")
    md = set(gb(L2 + "E44_L2_TABLES.md").decode().splitlines())
    FJ = E["FJ"]; rows = []
    for arm in ("naive", "replay"):
        pass
    order = [stem(a, w) for w in (1, 2, 3, 4) for a in ("naive", "replay")] + [stem(a, w) for w in (5, 6, 7, 8) for a in ("naive", "replay")] + \
            [stem(a, w) for w in range(1, 9) for a in ("naive-wd0", "replay-wd0")]
    inv = {stem(a, w): (a, w) for a in ARMS for w in range(1, 9)}
    for s in order:
        a, w = inv[s]; r = REC[a, w]
        rows.append(f"| `{s}` | {r['learner']['seed']} | {f3(r['cpu']['total'] / 3600, 2)} | {', '.join(str(x['done_visits']) for x in r['resumes']) or '—'} | "
                    f"{r['ledger']['buffer']['items_at_end']:,} | {r['optimizer']['weight_decay']} | {E.get(f'keys_{s}', True)} |")
    for arm, V in (("naive", 1024), ("replay", 1024)) + tuple((a, 256) for a in ARMS):
        i5, i7, word = BT[arm, V, "stuck"]
        i5s, i7s = interval90(i5["values"], True), interval90(i7["values"], True)
        dec = ", ".join(x for x, y in (("D5", i5s["high"] < 0), ("D7", i7s["high"] < 0)) if y) or "—"
        rows.append(f"| {arm} | {V} | {i5['n']} | {sg(i5s['mean'])} [{sg(i5s['low'])}, {sg(i5s['high'])}] | {sg(i7s['mean'])} [{sg(i7s['low'])}, {sg(i7s['high'])}] | {word} | {dec} |")
    for a, b, V in COMPS:
        for qn, _ in QTY:
            d = BT[a, b, V, qn]; iv = interval90(list(d["per"].values()), True)
            rows.append(f"| {a} - {b} at {V} | {qn} | {len(d['per'])} | " + " ".join(f"{x:+.3f}" for x in d["per"].values())
                        + f" | {sg(iv['mean'], 3)} [{sg(iv['low'], 3)}, {sg(iv['high'], 3)}] | {verdict(iv)} |")
    for arm in ARMS:
        for V in (256, 1024):
            for w in worlds_of(V):
                if V == 1024 and arm.endswith("wd0"):
                    continue
                q = R[arm, w, V]; lk = q["late_level_by_kind"]
                rows.append(f"| {arm} | {w} | {V} | {q['families']} | {f3(q['late_level'])} | {f3(lk['return'])} | {f3(lk['continuation'])} | {f3(lk['first'])} | "
                            f"{sg(q['score_trend'])} | {sg(q['trend_of_learning'])} | {f3(q['median_first'])} | {f3(q['kept']['5 to 32']['share'])} | "
                            f"{f3(q['kept']['33 or more']['share'])} | {f3(q['kept']['5 or more']['share'])} | {f3(q['d9']['5 or more']['mean'])} |")
    for arm in ARMS:
        for V in (256, 1024):
            if V == 1024 and arm.endswith("wd0"):
                continue
            p = E[f"d15_{arm}_{V}"]
            for name in p:
                rows.append(f"| {arm} | {V} | {name} | " + " | ".join("—" if not c["returns"] else f"{f3(c['kept'])} ({c['items']}) / {f3(c['first16'])} ({c['returns']})"
                                                                      for c in p[name].values()) + " |")
    for a in ("naive", "replay"):
        for V in (256, 1024):
            for kind, key in (("stream", "wrong_answers_stream"), ("list", "wrong_answers_list")):
                p = E[f"d14_{a}_{V}_{kind}"]
                rows.append(f"| {a} at {V}, {key} | {p['answers']} | " + " | ".join(f3(p["shares"][o]) for o in ORDER) + f" | {f3(p['chance'])} |")
    for r in sorted(FJ["probe"], key=lambda r: (r["arm"], r["age"], r["world"])):
        lv = E["probe"].get((r["arm"], r["world"], r["age"]))
        if lv is not None:
            rows.append(f"| {r['arm']} | {r['world']} | {r['age']} | {' '.join(f'{x:.3f}' for x in lv)} | {f3(float(np.mean(lv)))} |")
    for a in ("naive", "replay"):
        for g in (64, 256, 1024):
            ms = [float(np.mean(E["probe"][a, w, g])) for w in (1, 2, 3, 4) if (a, w, g) in E["probe"]]
            rows.append(f"| {a} | {g} | {len(ms)} | {f3(float(np.mean(ms)))} |")
    for arm, tag, vs in (("naive", "_v256_wd0", (0, 64, 128, 192, 256)), ("replay", "_v256_rho0.5_wd0", (0, 64, 128, 192, 256)),
                         ("naive", "_v1024", (0, 64, 256, 512, 1024)), ("replay", "_v1024_rho0.5", (0, 64, 256, 512, 1024)),
                         ("naive", "_v256", (0, 64, 128, 192, 256)), ("replay", "_v256_rho0.5", (0, 64, 128, 192, 256))):
        ls = [d for life, d in E["norms"].items() if life.startswith(f"lw_{arm}_w") and life.endswith(tag)]
        rows.append(f"| {arm + tag.replace('_rho0.5', '')} | {len(ls)} | " + " | ".join(f"{np.mean([d[v][0] for d in ls]):.1f} / {np.mean([d[v][1] for d in ls]):.1f}" for v in vs)
                    + f" | {', '.join(map(str, vs))} |")
    miss = [r for r in rows if r not in md]
    E["table_rows"] = (len(rows), len(miss))
    print(f"   rows rebuilt: {len(rows)}; found verbatim in E44_L2_TABLES.md: {len(rows) - len(miss)}")
    for r in miss:
        print(f"   NOT FOUND: {r}")
    # the curves file: blocks of 32 visits of D2 (all visits, and by kind) for every arm and world
    rows_c = [ln.split(",") for ln in gb(L2 + "E44_L2_score_by_age_blocks.csv").decode().splitlines()[1:]]
    nb = 0
    for arm, w, V, blk, al, ret, cont, fst, bl, nr, nc, nf in rows_c:
        q = R[arm, int(w), int(V)]; b0, b1 = (int(x) for x in blk.split(" to ")); vs = range(b0, b1 + 1)
        kinds = [C_kind(q, v) for v in vs]
        cmp(f"curves {arm} w{w} {blk} all", float(np.mean([q["score_by_age"][v] for v in vs])), float(al))
        for k, x in (("return", ret), ("continuation", cont), ("first", fst)):
            ys = [q["score_by_age"][v] for v in vs if C_kind(q, v) == k]
            cmp(f"curves {arm} w{w} {blk} {k}", float(np.mean(ys)) if ys else None, float(x) if x else None)
        for k, x in (("return", nr), ("continuation", nc), ("first", nf)):
            cmp(f"curves {arm} w{w} {blk} number of {k} visits", kinds.count(k), int(x))
        nb += 1
    print(f"   the curves file E44_L2_score_by_age_blocks.csv: {nb} rows (blocks of 32 visits) compared with D2 recomputed")
    d14h = [ln for ln in md if ln.startswith("## D14")]
    print(f"   the tables file's D14 section: {d14h}; rows by class of absence in the tables file: "
          f"{sum(1 for ln in md if 'wrong_answers' in ln and ('2 to 4' in ln or '33 or more' in ln))}")


# ================================================================ E. the report's text
def part_e():
    print("E. The report's text at the report's commit: every figure set beside the recomputed value (a difference larger than half a unit of the last")
    print("   printed decimal is listed); then what changed between the two commits, and the trial report's corrections")
    rep = gb(L2 + "E44_L2_REPORT.md").decode(); rep0 = gb(L2 + "E44_L2_REPORT.md", REP0).decode()
    a_, b_ = rep.splitlines(), rep0.splitlines()
    print(f"   report: {len(a_)} lines at {REP}, {len(b_)} at {REP0}; lines only in {REP}: {len(set(a_) - set(b_))} (lines {[i + 1 for i, x in enumerate(a_) if x not in set(b_)]}); "
          f"lines only in {REP0}: {len(set(b_) - set(a_))}")
    tr, tr0 = gb(L2 + "trial/E44_L2_TRIAL_REPORT.md").decode().splitlines(), gb(L2 + "trial/E44_L2_TRIAL_REPORT.md", REP0).decode().splitlines()
    print(f"   trial report: lines only in {REP}: {len(set(tr) - set(tr0))} (lines {[i + 1 for i, x in enumerate(tr) if x not in set(tr0)]}); only in {REP0}: {len(set(tr0) - set(tr))}")
    S4 = lambda a, V: E[f"sec4_{a}_{V}"]
    # section 1
    txt("1 CPU-hours of the 32 lives", 46.98, E["cpu_total"], 2)
    txt("1 probe CPU-hours", 0.26, E["probe_cpu"], 2)
    txt("1 checks and births CPU-hours", 0.58, E["check_cpu"], 2)
    txt("1 visits lost to interruptions (about 410)", 410, E["lost_visits"], -1)
    txt("1 CPU-hours lost (about 1.4)", 1.4, E["lost_cpu"], 1)
    txt("1 forecast before the trial", 48.2, E["forecast_before"], 1); txt("1 forecast after the trial", 48.0, E["forecast_after"], 1)
    for lab, rep_v, mine in E["interruption_visits"]:
        txt(f"1 interruptions: {lab}", rep_v, mine, 0)
    # section 2
    for lab, rep_v, mine in E["b_keys"]:
        txt(f"2 (b) {lab}", rep_v, mine, 0)
    txt("2 recommendation 3, world 2 median", 0.573, E["births"][2], 3); txt("2 recommendation 3, world 3 median", 0.313, E["births"][3], 3)
    txt("2 largest mean loss", 0.934, E["loss_max"][0], 3)
    txt("2 check 7 lowest", 11.7, E["c7"][0], 1); txt("2 check 7 highest", 12.1, E["c7"][1], 1)
    # section 3
    txt("3 replay median D6 (trial)", 0.595, R["replay", 1, 1024]["median_first"], 3); txt("3 naive median D6 (trial)", 0.757, R["naive", 1, 1024]["median_first"], 3)
    # section 4, first table
    T41 = {("naive", 1024): (0.724, 0.679, 0.769, 0.709, 0.801), ("replay", 1024): (0.771, 0.699, 0.843, 0.771, 0.672),
           ("naive", 256): (0.669, 0.640, 0.699, 0.642, 0.633), ("replay", 256): (0.684, 0.645, 0.722, 0.687, 0.416),
           ("naive-wd0", 256): (0.574, 0.504, 0.645, 0.568, 0.430), ("replay-wd0", 256): (0.541, 0.450, 0.631, 0.546, 0.273)}
    for (a, V), (m, lo, hi, ret, med) in T41.items():
        x = S4(a, V)
        txt(f"4 table 1 {a} {V} late mean", m, x["late"]["mean"], 3); txt(f"4 table 1 {a} {V} late low", lo, x["late"]["low"], 3)
        txt(f"4 table 1 {a} {V} late high", hi, x["late"]["high"], 3); txt(f"4 table 1 {a} {V} on returns", ret, x["ret"], 3)
        txt(f"4 table 1 {a} {V} median D6 (mean over worlds)", med, x["med"], 3)
    for a, V in T41:
        txt(f"4 blind late level 0.015, {a} at {V} (mean over worlds)", 0.015, float(np.mean(S4(a, V)["blind"])), 3)
    bl = [b for a, V in T41 for b in S4(a, V)["blind"]]
    print(f"   the blind predictor's late level, single lives: {min(bl):.4f} to {max(bl):.4f}")
    T42 = {("naive", 1024): (0.802, 0.011, 0.107, 0.011), ("replay", 1024): (0.750, 0.537, 0.611, 0.661), ("naive", 256): (0.696, 0.017, 0.110, 0.021),
           ("replay", 256): (0.668, 0.481, 0.523, 0.676), ("naive-wd0", 256): (0.660, 0.085, 0.210, 0.116), ("replay-wd0", 256): (0.615, 0.462, 0.486, 0.784)}
    for (a, V), vals in T42.items():
        x = S4(a, V)
        for nm, rv, mv in zip(("kept absence 1", "kept 5 or more", "first 16, 5 or more", "D9, 5 or more"), vals, (x["k1"], x["k5"], x["f16"], x["d9"])):
            txt(f"4 table 2 {a} {V} {nm}", rv, mv, 3)
    T43 = {("naive", 1024): ((0.015, -0.016, 0.046), (0.098, 0.083, 0.114)), ("replay", 1024): ((0.052, -0.006, 0.109), (0.108, 0.078, 0.137)),
           ("naive", 256): ((0.047, -0.009, 0.102), (0.070, 0.003, 0.136)), ("replay", 256): ((0.096, 0.025, 0.166), (0.041, -0.034, 0.115)),
           ("naive-wd0", 256): ((0.031, -0.042, 0.104), (-0.049, -0.119, 0.021)), ("replay-wd0", 256): ((-0.006, -0.081, 0.069), (-0.059, -0.131, 0.014))}
    for (a, V), (s5, s7) in T43.items():
        i5, i7, _ = BT[a, V, "stuck"]
        for nm, iv, vals in (("D5", i5, s5), ("D7", i7, s7)):
            for k, v in zip(("mean", "low", "high"), vals):
                txt(f"4 D13 {a} {V} {nm} {k}", v, iv[k], 3)
    T44 = {("replay", "naive", 256): ((0.015, -0.018, 0.047), (0.459, 0.413, 0.506), (0.511, 0.409, 0.613)),
           ("replay", "naive", 1024): ((0.047, -0.011, 0.105), (0.531, 0.440, 0.622), (0.519, 0.422, 0.616)),
           ("naive-wd0", "naive", 256): ((-0.095, -0.166, -0.024), (0.075, 0.045, 0.105), (0.031, 0.003, 0.058)),
           ("replay-wd0", "replay", 256): ((-0.143, -0.245, -0.041), None, None)}
    for (a, b, V), vals in T44.items():
        for qn, v3 in zip(("late level (D4)", "kept at return, 5 to 32 (D8)", "kept at return, 33 or more (D8)"), vals):
            if v3 is None:
                continue
            iv = BT[a, b, V, qn]["iv"]
            for k, v in zip(("mean", "low", "high"), v3):
                txt(f"4 between {a}-{b} {V} {qn} {k}", v, iv[k], 3)
    for (a, b, V), qn, v3 in ((("naive-wd0", "naive", 256), "trend of learning (D7)", (-0.119, -0.161, -0.076)),
                              (("replay-wd0", "replay", 256), "score trend (D5)", (-0.102, -0.181, -0.022)),
                              (("replay-wd0", "replay", 256), "trend of learning (D7)", (-0.099, -0.154, -0.045))):
        iv = BT[a, b, V, qn]["iv"]
        for k, v in zip(("mean", "low", "high"), v3):
            txt(f"4 between {a}-{b} {V} {qn} {k}", v, iv[k], 3)
    # D14 table
    pn, pr = E["d14_naive_1024_stream"], E["d14_replay_1024_stream"]
    txt("4 D14 answers", 12683, pn["answers"], 0); txt("4 D14 answers (replay)", 12683, pr["answers"], 0)
    for o, vn, vr in ((ORDER[0], 0.013, 0.549), (ORDER[2], 0.499, 0.013), (ORDER[3], 0.141, 0.183), (ORDER[1], 0.126, None), (ORDER[7], 0.196, 0.212)):
        txt(f"4 D14 naive {o}", vn, pn["shares"][o], 3)
        if vr is not None:
            txt(f"4 D14 replay {o}", vr, pr["shares"][o], 3)
    txt("4 D14 chance naive", 0.007, pn["chance"], 3); txt("4 D14 chance replay", 0.008, pr["chance"], 3)
    ln, lr = E["d14_naive_1024_list"], E["d14_replay_1024_list"]
    txt("4 D14 list answers", 2685000, ln["answers"], 0); txt("4 D14 list naive last", 0.543, ln["shares"][ORDER[2]], 3)
    txt("4 D14 list replay earlier", 0.258, lr["shares"][ORDER[3]], 3)
    print(f"   D14 table: replay's 'sum or difference without the reduction' is printed as '—'; recomputed {pr['shares'][ORDER[1]]:.4f}")
    # D15 table and sentences
    T15 = {("naive", 1024): {"2 to 4": ((0.014, 0.123), (0.010, 0.148), (0.021, 0.143), (0.019, 0.141), (0.014, 0.173)),
                             "5 to 32": ((0.005, 0.097), (0.027, 0.103), (0.018, 0.109), (0.008, 0.106), (0.010, 0.116)),
                             "33 or more": ((0.000, 0.086), (0.013, 0.117), (0.005, 0.089), (0.009, 0.093), (0.014, 0.111))},
           ("replay", 1024): {"2 to 4": ((0.093, 0.331), (0.360, 0.492), (0.480, 0.555), (0.566, 0.637), (0.714, 0.754)),
                              "5 to 32": ((0.094, 0.320), (0.279, 0.409), (0.395, 0.530), (0.564, 0.624), (0.693, 0.720)),
                              "33 or more": ((0.115, 0.337), (0.265, 0.452), (0.409, 0.542), (0.565, 0.612), (0.658, 0.696))},
           ("naive-wd0", 256): {"2 to 4": ((0.031, 0.119), (0.116, 0.223), (0.090, 0.247), (0.145, 0.332), (0.219, 0.414)),
                                "5 to 32": ((0.028, 0.105), (0.035, 0.105), (0.038, 0.151), (0.070, 0.190), (0.166, 0.334)),
                                "33 or more": ((0.000, 0.059), (0.000, 0.080), (0.025, 0.098), (0.018, 0.136), (0.060, 0.213))}}
    for (a, V), d in T15.items():
        p = E[f"d15_{a}_{V}"]
        for name, cells in d.items():
            for (en, _, _), (k, f) in zip(EARLIER, cells):
                txt(f"4 D15 {a} {V} {name} {en} kept", k, p[name][en]["kept"], 3); txt(f"4 D15 {a} {V} {name} {en} first16", f, p[name][en]["first16"], 3)
    # sentences
    n1, n2 = E["d15_naive_1024"], E["d15_naive_256"]
    k1 = [c["kept"] for nm in n1 for c in n1[nm].values()]; k2 = [c["kept"] for nm in n2 for c in n2[nm].values()]
    txt("4 D15 naive 1,024 kept lowest", 0.000, min(k1), 3); txt("4 D15 naive 1,024 kept highest", 0.027, max(k1), 3)
    txt("4 D15 naive 256 kept lowest", 0.000, min(k2), 3); txt("4 D15 naive 256 kept highest", 0.115, max(k2), 3)
    txt("4 D15 naive 256 the 0.115 rests on items", 26, n2["33 or more"]["2 to 3"]["items"], 0)
    for V, pairs in ((1024, ((0.173, 0.123), (0.116, 0.097), (0.111, 0.086))), (256, ((0.202, 0.099), (0.151, 0.102), (0.099, 0.076)))):
        p = E[f"d15_naive_{V}"]
        for name, (hi_, lo_) in zip(("2 to 4", "5 to 32", "33 or more"), pairs):
            txt(f"4 D15 naive {V} {name} first16 16+", hi_, p[name]["16 or more"]["first16"], 3); txt(f"4 D15 naive {V} {name} first16 1", lo_, p[name]["1"]["first16"], 3)
    for a, V, k_lo, k_hi, f_lo, f_hi in (("replay", 1024, (0.093, 0.094, 0.115), (0.714, 0.693, 0.658), (0.331, 0.320, 0.337), (0.754, 0.720, 0.696)),
                                         ("naive-wd0", 256, (0.031, 0.028, 0.000), (0.219, 0.166, 0.060), (0.119, 0.105, 0.059), (0.414, 0.334, 0.213)),
                                         ("replay-wd0", 256, (0.113, 0.156, 0.000), (0.630, 0.630, 0.592), (0.150, 0.244, 0.188), (0.626, 0.637, 0.618))):
        p = E[f"d15_{a}_{V}"]
        for i, name in enumerate(("2 to 4", "5 to 32", "33 or more")):
            txt(f"4 D15 sentence {a} {name} kept first", k_lo[i], p[name]["1"]["kept"], 3); txt(f"4 D15 sentence {a} {name} kept last", k_hi[i], p[name]["16 or more"]["kept"], 3)
            txt(f"4 D15 sentence {a} {name} f16 first", f_lo[i], p[name]["1"]["first16"], 3); txt(f"4 D15 sentence {a} {name} f16 last", f_hi[i], p[name]["16 or more"]["first16"], 3)
    for a, V in (("naive", 1024), ("naive", 256), ("replay", 1024), ("replay", 256), ("naive-wd0", 256), ("replay-wd0", 256)):
        p = E[f"d15_{a}_{V}"]
        mono = {name: (all(x <= y for x, y in zip([c["kept"] for c in p[name].values()], [c["kept"] for c in p[name].values()][1:])),
                       all(x <= y for x, y in zip([c["first16"] for c in p[name].values()], [c["first16"] for c in p[name].values()][1:]))) for name in p}
        lastfirst = {name: (p[name]["16 or more"]["kept"] > p[name]["1"]["kept"], p[name]["16 or more"]["first16"] > p[name]["1"]["first16"]) for name in p}
        print(f"   D15 {a}, {V}: rises at every step (kept, first 16) {mono}; last cell above the first {lastfirst}; returns per cell "
              f"{min(c['returns'] for nm in p for c in p[nm].values())} to {max(c['returns'] for nm in p for c in p[nm].values())}")
    # the probe
    for a, g, v in (("naive", 64, 0.427), ("naive", 256, 0.669), ("naive", 1024, 0.826), ("replay", 64, 0.267), ("replay", 256, 0.570), ("replay", 1024, 0.743)):
        txt(f"4 probe {a} {g}", v, float(np.mean([np.mean(E["probe"][a, w, g]) for w in (1, 2, 3, 4)])), 3)
    # the norms
    nm = E["norms"]
    def nmean(arm, tag, v, i):
        ls = [d for life, d in nm.items() if life.startswith(f"lw_{arm}_w") and life.endswith(tag)]
        return float(np.mean([d[v][i] for d in ls]))
    txt("4 norms born all", 52.8, nmean("naive", "_v256_wd0", 0, 0), 1); txt("4 norms born rnn", 18.5, nmean("naive", "_v256_wd0", 0, 1), 1)
    for arm, tag, cells in (("naive", "_v256_wd0", ((64, 88.8, 35.7), (256, 126.3, 52.8))), ("replay", "_v256_rho0.5_wd0", ((64, 89.1, 36.7), (256, 122.9, 54.8))),
                            ("naive", "_v1024", ((64, 67.6, 18.6), (256, 70.8, 17.6), (1024, 76.1, 17.2))),
                            ("replay", "_v1024_rho0.5", ((64, 64.9, 19.2), (256, 69.3, 19.2), (1024, 72.3, 18.6)))):
        for v, x, y in cells:
            txt(f"4 norms {arm}{tag} {v} all", x, nmean(arm, tag, v, 0), 1); txt(f"4 norms {arm}{tag} {v} rnn", y, nmean(arm, tag, v, 1), 1)
    long_rnn = [nmean(a, t, v, 1) for a, t in (("naive", "_v1024"), ("replay", "_v1024_rho0.5")) for v in (64, 256, 512, 1024)]
    txt("4 norms, with decay the recurrent matrix lowest (long lives' means)", 17.2, min(long_rnn), 1)
    txt("4 norms, with decay the recurrent matrix highest (long lives' means)", 19.2, max(long_rnn), 1)
    # section 5
    T5 = ((("replay", "naive", 1024), "late level (D4)", (0.047, -0.011, 0.105)), (("replay", "naive", 256), "kept at return, 33 or more (D8)", (0.511, 0.409, 0.613)),
          (("replay", "naive", 256), "kept at return, 5 or more (D8)", (0.464, 0.419, 0.508)), (("naive-wd0", "naive", 256), "late level (D4)", (-0.095, -0.166, -0.024)),
          (("naive-wd0", "naive", 256), "kept at return, 33 or more (D8)", (0.031, 0.003, 0.058)), (("naive-wd0", "naive", 256), "kept at return, 5 or more (D8)", (0.068, 0.038, 0.097)),
          (("replay-wd0", "replay", 256), "late level (D4)", (-0.143, -0.245, -0.041)))
    for (a, b, V), qn, v3 in T5:
        iv = BT[a, b, V, qn]["iv"]
        for k, v in zip(("mean", "low", "high"), v3):
            txt(f"5 {a}-{b} {V} {qn} {k}", v, iv[k], 3)
    for (a, V), (lo, hi) in ((("naive", 1024), (0.083, 0.114)), (("replay", 1024), (0.078, 0.137))):
        i7 = BT[a, V, "stuck"][1]; txt(f"5 D7 {a} low", lo, i7["low"], 3); txt(f"5 D7 {a} high", hi, i7["high"], 3)
    nl, rm, p3 = E["exp"]
    for w, v in zip((1, 2, 3, 4), (0.669, 0.756, 0.742, 0.732)):
        txt(f"5 (i) naive late level w{w}", v, nl[w - 1], 3)
    txt("5 (i) mean", 0.724, float(np.mean(nl)), 3); txt("5 (ii) lowest", 0.328, min(rm), 3); txt("5 (ii) below 0.20", 0, sum(x < 0.2 for x in rm), 0)
    txt("5 (iii) share", 0.499, p3["shares"][ORDER[2]], 3); txt("5 (iii) answers", 12683, p3["answers"], 0)
    # section 7: median visits (the correction of 45e9ba16)
    for V, vals in ((1024, (355, 451, 479, 563, 639, 650)), (256, (114, 122, 113, 152, 180, 211))):
        p = E[f"d15_{'naive' if V == 1024 else 'naive-wd0'}_{V}"]
        got = [p[nm][en]["median_visit"] for nm in ("2 to 4", "5 to 32", "33 or more") for en in ("1", "16 or more")]
        for i, (x, y) in enumerate(zip(vals, got)):
            txt(f"7 median visit of returns, {V} visits, value {i + 1}", x, y, 0)
    print(f"   section 7, median visit of the returns: long lives " + ", ".join(f"{x:.1f}" for x in [E['d15_naive_1024'][nm][en]['median_visit'] for nm in ('2 to 4', '5 to 32', '33 or more') for en in ('1', '16 or more')])
          + "; lives of 256 visits (worlds 1-8) " + ", ".join(f"{x:.1f}" for x in [E['d15_naive-wd0_256'][nm][en]['median_visit'] for nm in ('2 to 4', '5 to 32', '33 or more') for en in ('1', '16 or more')]))
    # the trial report's corrected figures (world 1, 1,024 visits)
    for a in ("naive", "replay"):
        later = R[a, 1, 1024]["later"]
        cells = {}
        for name, lo, hi in ABS[1:4]:
            for en, elo, ehi in (("1", 1, 1), ("16 or more", 16, 10 ** 9)):
                rs = [r for r in later if lo <= r["absence"] <= hi and elo <= r["earlier_visits"] <= ehi]
                cells[name, en] = (tmean([r["first16"] for r in rs]), float(np.median([r["visit"] for r in rs])))
        E[f"trial_{a}"] = cells
    tn, trp = E["trial_naive"], E["trial_replay"]
    for name, (hi_, lo_) in zip(("2 to 4", "5 to 32", "33 or more"), ((0.157, 0.132), (0.101, 0.068), (0.112, 0.055))):
        txt(f"trial naive first16 {name} 16+", hi_, tn[name, "16 or more"][0], 3); txt(f"trial naive first16 {name} 1", lo_, tn[name, "1"][0], 3)
    for name, (x, y) in zip(("2 to 4", "5 to 32", "33 or more"), ((264, 539), (434, 642), (572, 696))):
        txt(f"trial replay median visit {name} 1", x, trp[name, "1"][1], 0); txt(f"trial replay median visit {name} 16+", y, trp[name, "16 or more"][1], 0)
    print(f"   figures of the report's text compared: {TXT_N[0]}; differing: {len(TXT_DIFFS)}")
    for lab, rep_, mine, dd in TXT_DIFFS:
        print(f"   TEXT DIFFERS: {lab}: reported {rep_}, recomputed {mine:.6f} (difference {dd:+.6f})")


# ================================================================ F. the checks
def part_f():
    print("F. The checks under checks/, the two 8-visit lives of the other births, the logs of the driver, the costs")
    d = L2 + "checks/"
    CK = gjson(d + "E44_L2_CHECKS.json")
    print(f"   check receipt: all_passed {CK['all_passed']}; written at {CK['git']['commit'][:8]}; first committed at "
          f"{git('log', '--format=%h %ad', '--date=iso', REP, '--', d + 'E44_L2_CHECKS.json').decode().split(chr(10))[-2]}")
    def life(sub, name):
        return gjson(d + f"{sub}/{name}.json"), gnpz(d + f"{sub}/{name}.npz"), gnpz(d + f"{sub}/{name}_list.npz")
    # (a)
    rb, evb, lab = life("a_base", "lw_naive_w1_v8_turn"); rn, evn, lan = life("a_new", "lw_naive_w1_v8_turn")
    e44 = gjson(M + "exp45/e44/F1/lw_naive_w1_v8_turn.json")
    print(f"   (a): base runner {rb['provenance']['sources_sha256']['smallworld/sw_run.py'][:8]}… against changed {rn['provenance']['sources_sha256']['smallworld/sw_run.py'][:8]}…: "
          f"events arrays equal {arrays_sha(evb) == arrays_sha(evn)}, list arrays equal {arrays_sha(lab) == arrays_sha(lan)}, weights equal "
          f"{rb['final']['weights_tensors_sha256'] == rn['final']['weights_tensors_sha256']}, optimizer equal {rb['final']['optimizer_sha256'] == rn['final']['optimizer_sha256']}, "
          f"learned equal {rb['learned_sha256'] == rn['learned_sha256']}; both equal E44's committed receipt (weights, optimizer, learned, stream): "
          f"{rn['final']['weights_tensors_sha256'] == e44['final']['weights_tensors_sha256'] and rn['final']['optimizer_sha256'] == e44['final']['optimizer_sha256'] and rn['learned_sha256'] == e44['learned_sha256'] and rn['stream_sha256'] == e44['stream_sha256']}; "
          f"events arrays equal E44's receipt {arrays_sha(evn) == e44['events']['arrays_sha256']}")
    # (b), (c)
    runs = {}
    for sub, K, dist in (("b_default", 1, None), ("b_k4_d1248", 4, (1, 2, 4, 8)), ("b_k4_none", 4, ())):
        r, ev, la = life(sub, "lw_naive_w5_v32")
        got = C.keys_of_list(la); sched = World(5).schedule(32)
        want = C.rows_note(sched, 32, K, dist, vu=256, birth_live="K", born_decay="dist") if K != 1 else C.rows_note(sched, 32, 1, None, vu=256, birth_live="K", born_decay="dist")
        runs[sub] = (r, ev, la, got)
        E.setdefault("b_keys", [])
        E["b_keys"] += [(f"{sub} keys", {'b_default': 241, 'b_k4_d1248': 77, 'b_k4_none': 57}[sub], len(got)),
                        (f"{sub} items", {'b_default': 165200, 'b_k4_d1248': 38800, 'b_k4_none': 27600}[sub], sum(x[-1] for x in got))]
        print(f"   (b) {sub}: {len(got)} keys, {sum(x[-1] for x in got):,} items; equal to the reviewer's reference rows (part K's rows_note): {sorted(got) == sorted(want)}; "
              f"keys digest {C.keys_digest(got)[:12]}…")
    base = runs["b_default"]
    rowsets = {}
    for sub, (r, ev, la, got) in runs.items():
        src0 = la["source"] == 0
        key = list(zip(la["at_visit"][src0].tolist(), la["family"][src0].tolist(), la["task"][src0].tolist(), la["a"][src0].tolist(), la["b"][src0].tolist()))
        rowsets[sub] = dict(zip(key, zip(la["g1"][src0].tolist(), la["g2"][src0].tolist(), la["g3"][src0].tolist())))
    common_ok = True; ncommon = 0
    for x in ("b_k4_d1248", "b_k4_none"):
        for k, v in rowsets[x].items():
            if k in rowsets["b_default"]:
                ncommon += 1; common_ok &= rowsets["b_default"][k] == v
    print(f"   (b) network rows present in the default run and in a K = 4 run: {ncommon:,}; generated bytes equal in every such row: {common_ok}")
    print(f"   (c) the three runs: events arrays equal {len({arrays_sha(v[1]) for v in runs.values()}) == 1}; weights equal "
          f"{len({v[0]['final']['weights_tensors_sha256'] for v in runs.values()}) == 1}; optimizer equal {len({v[0]['final']['optimizer_sha256'] for v in runs.values()}) == 1}; "
          f"learned equal {len({v[0]['learned_sha256'] for v in runs.values()}) == 1}")
    # (d): the settings carried into the receipts (and states), kept on resume: read from c5_cut
    # check 3
    ra, eva, laa = life("c3_naive", "lw_naive_w1_v1"); rr, evr, lar = life("c3_replay_rho0", "lw_replay_w1_v1_rho0")
    print(f"   check 3: naive against replay at rho 0, one visit of world 1: weights equal {ra['final']['weights_tensors_sha256'] == rr['final']['weights_tensors_sha256']}, "
          f"optimizer equal {ra['final']['optimizer_sha256'] == rr['final']['optimizer_sha256']}, events arrays equal {arrays_sha(eva) == arrays_sha(evr)}, list arrays equal "
          f"{arrays_sha(laa) == arrays_sha(lar)}; replay's learner {rr['settings']['learner']['file'].split('/')[-1]}, rho {rr['settings']['rho']}, replayed {rr['counters']['replayed_items']}")
    # check 5
    rw, evw, law = life("c5_whole", "lw_replay_w1_v8_rho0.5_wd0"); rc, evc, lac = life("c5_cut", "lw_replay_w1_v8_rho0.5_wd0")
    st1 = [json.loads(x) for x in gb(d + "c5_cut/c5_cut.starts.jsonl").decode().splitlines() if x.strip()]
    st2 = [json.loads(x) for x in gb(d + "c5_cut/c5_resumed.starts.jsonl").decode().splitlines() if x.strip()]
    full_at = math.ceil(55209 / 48)
    E["c5"] = (rw["settings"]["list_every"], rw["settings"]["decay_distances"], rc["resumes"][0]["done_visits"] if rc["resumes"] else None, full_at - 1)
    print(f"   check 5: whole against cut and resumed (replay-wd0, world 1, 8 visits): weights equal {rw['final']['weights_tensors_sha256'] == rc['final']['weights_tensors_sha256']}, "
          f"optimizer equal {rw['final']['optimizer_sha256'] == rc['final']['optimizer_sha256']}, events arrays equal {arrays_sha(evw) == arrays_sha(evc)}, list arrays equal "
          f"{arrays_sha(law) == arrays_sha(lac)}, learned equal {rw['learned_sha256'] == rc['learned_sha256']}, counters equal {rw['counters'] == rc['counters']}; "
          f"resumed from visit {E['c5'][2]}; starts {[x['resumed_from_visits'] for x in st1]} then {[x['resumed_from_visits'] for x in st2]}; settings of the check: "
          f"list every {rw['settings']['list_every']}, decay distances {rw['settings']['decay_distances']} (the lives: 8, none); the settings recorded in the resumed "
          f"receipt equal the whole run's: {rw['settings'] == rc['settings']}; the store (55,209 items, 48 offered an update) is full from update {full_at - 1}, "
          f"in visit {(full_at - 1) // 256}, after the cut at visit 4")
    # check 6 on the roster's own receipts
    wd_ok = all(REC[a, w]["optimizer"]["groups"][0]["weight_decay"] == 0.0 and REC[a, w]["settings"]["wd"] == 0.0 for a in ("naive-wd0", "replay-wd0") for w in range(1, 9))
    print(f"   check 6: the 16 decay-0 receipts read back weight decay 0.0 in the first group at the end and in settings: {wd_ok}; check receipt's arms "
          + ", ".join(f"{k} {v['first_group_at_start']}/{v['first_group_at_end']}" for k, v in CK["checks"]["6"]["arms"].items()))
    # check 7
    c7 = CK["checks"]["7"]["arms"]
    E["c7"] = (min(v["mean"] for v in c7.values()), max(v["mean"] for v in c7.values()))
    E["forecast_before"] = CK["checks"]["7"]["forecast_release_1_cpu_hours"]
    print(f"   check 7: CPU-seconds a visit " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in c7.items()) + f"; forecast {E['forecast_before']:.2f} CPU-hours")
    # the other births
    meds = {}
    for w in (2, 3):
        ev = gnpz(L2 + f"births/lw_naive_w{w}_v8_turn.npz")
        lev = []
        for v in range(8):
            m = ev["visit"] == v; t = ev["task"][m] % 4; red = (ev["wrap"][m] == 1) & (t < 2); u = ev["update"][m]; c = ev["correct"][m]
            late = u >= u.min() + 192
            lev.append(float(np.mean([c[red & late & (t == j)].mean() for j in (0, 1)])))
        meds[w] = float(np.median(lev[1:])); E.setdefault("births", {})[w] = meds[w]
        cmp(f"birth life w{w} first-visit levels", True, all(abs(a - b) < 1e-12 for a, b in zip(lev, CK["other_births"]["lives"][f"world {w}"]["first_visit_levels"])))
        print(f"   other birth, world {w}: first-visit levels recomputed equal the receipt's: {all(abs(a - b) < 1e-12 for a, b in zip(lev, CK['other_births']['lives'][f'world {w}']['first_visit_levels']))}; "
              f"median of families 2 to 8 {meds[w]:.4f}; median of all 8 {np.median(lev):.4f}")
    # costs: the probe branches, checks and births
    pc = 0.0
    for a in ("naive", "replay"):
        for w in (1, 2, 3, 4):
            for g in (64, 256, 1024):
                pc += gjson(L2 + f"probe/{stem(a, w)}_br{g}_g52-53-54-55.json")["cpu"]["total"]
    E["probe_cpu"] = pc / 3600
    cc = 0.0; names = []
    for p in git("ls-tree", "-r", "--name-only", REP, "--", d, L2 + "births/").decode().splitlines():
        if p.endswith(".json") and "E44_L2_CHECKS" not in p and "check_jobs" not in p:
            j = gjson(p)
            if "cpu" in j:
                cc += j["cpu"]["total"]; names.append(p.split("/")[-2])
    E["check_cpu"] = cc / 3600
    print(f"   CPU-hours: probe branches {E['probe_cpu']:.4f}; the {len(names)} receipts of check lives and other births {E['check_cpu']:.4f}")
    # interruptions: visits computed and then lost
    lost = 0; il = []
    for arm in ARMS:
        for w in range(1, 9):
            s = stem(arm, w); runs_ = E.get(f"runs_{s}")
            if not runs_ or len(runs_) < 2:
                continue
            for (a0, a1), (b0, b1) in zip(runs_, runs_[1:]):
                lost += a1 - b0 + 1; il.append((s, a1, b0))
    E["lost_visits"] = lost
    per_visit = E["cpu_total"] * 3600 / (8 * 1024 + 24 * 256)
    E["lost_cpu"] = lost * per_visit / 3600
    print(f"   interruptions, from the logs: " + "; ".join(f"{s} reached visit {a1}, continued from {b0}" for s, a1, b0 in il))
    print(f"   visits computed and then computed again: {lost}; at the lives' mean {per_visit:.2f} CPU-seconds a visit, {E['lost_cpu']:.2f} CPU-hours")
    rr_ = {x[0]: x for x in il}
    E["interruption_visits"] = []
    first = [x for x in il if x[2] == 0 and x[0] in (stem("naive", 1), stem("replay", 1), stem("naive", 2))]
    E["interruption_visits"] += [(f"first stop at visit 57 ({x[0]})", 57, x[1]) for x in first]
    w4 = sorted(x[1] for x in il if x[0] in (stem("naive", 4), stem("replay", 4)))          # the report names the two visits without saying which is which
    E["interruption_visits"] += [("world 4, the lower of the two visits reached", 309, w4[0]), ("world 4, the higher", 315, w4[1])]
    E["interruption_visits"] += [(f"{x[0]} reached (started again from the beginning)", 37, x[1]) for x in il if x[0] == stem("replay", 5)]
    print(f"   lives started again from the beginning after a stop: {sorted({x[0] for x in il if x[2] == 0})}; resumed from a state: {sorted({x[0] for x in il if x[2] > 0})}")
    sec = [x[1] for x in il if x[2] == 768]
    if sec:
        E["interruption_visits"] += [("second restart, lowest visit reached", 790, min(sec)), ("second restart, highest visit reached", 797, max(sec))]
    tj = gjson(L2 + "trial/E44_L2_TRIAL.json")
    E["forecast_after"] = tj["stop_conditions"]["forecast_above_75_cpu_hours"]["forecast_cpu_hours"]
    print(f"   the trial's forecast {E['forecast_after']:.2f}; before the trial {E['forecast_before']:.2f}")
    drv = gb(L2 + "logs/roster_driver.log").decode().splitlines()
    done = {ln.split()[2]: ln.split()[0] for ln in drv if " done " in ln}
    for part, lives in (("trial", [stem("naive", 1), stem("replay", 1)]), ("long", [stem(a, w) for a in ("naive", "replay") for w in (2, 3, 4)]),
                        ("short", [stem(a, w) for a in ("naive", "replay") for w in (5, 6, 7, 8)]), ("decay", [stem(a, w) for a in ("naive-wd0", "replay-wd0") for w in range(1, 9)])):
        print(f"   driver log: the {part} lives' last completion at {max(done[s] for s in lives)} UTC")


# ================================================================ G. the probe
def part_g():
    print("G. The learning probe: 24 branches; streams regenerated by the reviewer's world; first-visit levels (D6) recomputed from the event files")
    E["probe"] = {}
    FJ = E["FJ"]; fj = {(r["arm"], r["world"], r["age"]): r for r in FJ["probe"]}
    for a in ("naive", "replay"):
        for w in (1, 2, 3, 4):
            Wd = World(w)
            for g in (64, 256, 1024):
                s = f"{stem(a, w)}_br{g}_g52-53-54-55"; r = gjson(L2 + f"probe/{s}.json")
                ok_set = r["complete"] is True and r["final"]["visits"] == g + 4 and [b["group"] for b in r["by_visit"]] == [52, 53, 54, 55] \
                    and [b["visit"] for b in r["by_visit"]] == [g, g + 1, g + 2, g + 3] and r["settings"]["arm"] == a and r["settings"]["wd"] == 0.25 \
                    and r["settings"]["rho"] == (0.5 if a == "replay" else None)
                lr = REC[a, w]
                rec = {x["visits"]: x["file_sha256"] for x in lr["states"]}; rec.update({x["done_visits"]: x["file_sha256"] for x in lr["resumes"]})
                st_ok = r["branch"]["from_state"]["done_visits"] == g and r["branch"]["from_state"]["file_sha256"] == rec.get(g)
                dig = _branch_digests(Wd, g)
                dg_ok = [b["stream_sha256"] for b in r["by_visit"]] == dig
                ev = gnpz(L2 + f"probe/{s}.npz", label=f"probe {s}")
                if ev is None:
                    print(f"   {s}: event file NOT READ"); continue
                lev = []
                for v in range(g, g + 4):
                    m = ev["visit"] == v; t = ev["task"][m] % 4; red = (ev["wrap"][m] == 1) & (t < 2); u = ev["update"][m]; c = ev["correct"][m]
                    late = u >= u.min() + 192
                    lev.append(float(np.mean([c[red & late & (t == j)].mean() for j in (0, 1)])))
                E["probe"][a, w, g] = lev
                for i, x in enumerate(lev):
                    cmp(f"probe {s} family {52 + i}", x, fj[a, w, g]["levels"][i])
                print(f"   {s}: complete, 4 visits of families 52-55 from visit {g}, settings of its life: {ok_set}; branched from the state the life's receipt records "
                      f"(sha256 {r['branch']['from_state']['file_sha256'][:8]}…): {st_ok}; visit digests equal the regenerated branch stream: {dg_ok}; "
                      f"levels {' '.join(f'{x:.4f}' for x in lev)} (mean {np.mean(lev):.4f})")
    met = {w: set(E[f"sched_{w}"]) for w in range(1, 9)}
    print(f"   families 52 to 55 met by any life's schedule: {any(f in met[w] for w in met for f in (52, 53, 54, 55))}")
    for a in ("naive", "replay"):
        for g in (64, 256, 1024):
            ms = [float(np.mean(E["probe"][a, w, g])) for w in (1, 2, 3, 4)]
            print(f"   probe {a}, after {g} visits: per world " + " ".join(f"{x:.3f}" for x in ms) + f"; mean {np.mean(ms):.4f}")
    for a in ("naive", "replay"):
        for g0, g1 in ((64, 256), (256, 1024), (64, 1024)):
            per = [np.mean(E["probe"][a, w, g1]) - np.mean(E["probe"][a, w, g0]) for w in (1, 2, 3, 4)]
            iv = interval90(per); E[f"probe_age_{a}_{g0}_{g1}"] = iv
            print(f"   probe {a}, after {g1} minus after {g0} visits: per world " + " ".join(f"{x:+.3f}" for x in per) + f"; {ivs(iv)} {verdict(iv)}")
    for g in (64, 256, 1024):
        per = [np.mean(E["probe"]["replay", w, g]) - np.mean(E["probe"]["naive", w, g]) for w in (1, 2, 3, 4)]
        iv = interval90(per); E[f"probe_arm_{g}"] = iv
        print(f"   probe replay - naive after {g} visits: per world " + " ".join(f"{x:+.3f}" for x in per) + f"; {ivs(iv)} {verdict(iv)}")
    # the first family of the branch only (no other probe family before it)
    for a in ("naive", "replay"):
        print(f"   probe {a}, family 52 alone (the branch's first visit), by age 64 / 256 / 1,024: " + "; ".join(
            f"w{w} " + "/".join(f"{E['probe'][a, w, g][0]:.3f}" for g in (64, 256, 1024)) for w in (1, 2, 3, 4)))


def _branch(Wd, g, fams):
    return Wd.branch(g, fams)


def _branch_digests(Wd, g):
    out = []
    cur = None; h = None
    for v, f, taught, asked in Wd.branch(g, [52, 53, 54, 55]):
        if v != cur:
            if h is not None:
                out.append(h.hexdigest())
            cur = v; h = hashlib.sha256()
        for i, a, b, p, y in taught + asked:
            h.update(bytes([i // 256, i % 256, a, b, len(p)]) + bytes(p) + bytes(c % 256 for c in y))
    out.append(h.hexdigest())
    return out


# ================================================================ H. norms and saved states
def part_h():
    print("H. The norms files against the receipts' states, and the saved states against the receipts")
    E["norms"] = {}
    csv_rows = [ln.split(",") for ln in gb(L2 + "E44_L2_norms.csv").decode().splitlines()[1:]]
    csv_d = {(r[0], int(r[1])): (float(r[2]), float(r[3])) for r in csv_rows}
    bad_state, n_state, n_pts = 0, 0, 0
    mono = {}
    for arm in ARMS:
        for w in range(1, 9):
            s = stem(arm, w); nf = gjson(L2 + f"norms/{s}_norms.json"); r = REC[arm, w]
            rec = {x["visits"]: (x["file_sha256"], x["weights_tensors_sha256"]) for x in r["states"]}
            res = {x["done_visits"]: x["file_sha256"] for x in r["resumes"]}
            d = {}
            for v, e in nf["visits"].items():
                v = int(v); n = e["norms"]
                tot = math.sqrt(sum(x * x for k, x in n.items() if not k.startswith("book.")))
                d[v] = (tot, n["rnn.weight_hh_l0"])
                cmp(f"norms csv {s} {v} total", tot, csv_d[s, v][0]); cmp(f"norms csv {s} {v} rnn", n["rnn.weight_hh_l0"], csv_d[s, v][1])
                n_pts += 1
                if v > 0:
                    n_state += 1
                    fs = e.get("state_file_sha256")
                    ok = (v in rec and fs == rec[v][0] and e["weights_tensors_sha256"] == rec[v][1]) or (v in res and fs == res[v])
                    bad_state += not ok
            E["norms"][s] = d
            vs = sorted(d)
            mono[s] = (all(d[a][0] < d[b][0] for a, b in zip(vs, vs[1:])), all(d[a][1] < d[b][1] for a, b in zip(vs, vs[1:])),
                       min(d[v][1] for v in vs if v > 0), max(d[v][1] for v in vs if v > 0))
    print(f"   norms points read: {n_pts}; from states: {n_state}; whose state file and weights digests are not those the life's receipt records (states or resumes): {bad_state}")
    for arm in ("naive-wd0", "replay-wd0"):
        ls = [stem(arm, w) for w in range(1, 9)]
        print(f"   {arm}: lives whose norm of all weights rises at every eighth visit {sum(mono[s][0] for s in ls)} of 8; whose recurrent matrix rises at every eighth "
              f"visit {sum(mono[s][1] for s in ls)} of 8; at 64, 128, 192, 256 (all weights / recurrent): " + "; ".join(
                  f"w{w} " + ", ".join(f"{E['norms'][stem(arm, w)][v][0]:.1f}/{E['norms'][stem(arm, w)][v][1]:.1f}" for v in (64, 128, 192, 256)) for w in range(1, 9)))
    for arm in ("naive", "replay"):
        ls = [stem(arm, w) for w in range(1, 9)]
        print(f"   {arm} (decay 0.25): recurrent matrix over the saved states of the 8 lives from {min(mono[s][2] for s in ls):.2f} to {max(mono[s][3] for s in ls):.2f}; "
              f"born " + ", ".join(f"w{w} {E['norms'][stem(arm, w)][0][1]:.2f}" for w in range(1, 9)))
    # the saved states
    out = SCR / "state.tmp"
    todo = [(stem(a, 1), v, REP, "committed") for a in ("naive", "replay") for v in (256, 1024)]
    for s, v, sh, fn, wall in E["resumes"]:
        p = LV + fn
        cs = git("log", "--format=%H", REP, "--", p).decode().split()
        todo.append((s, v, cs[-1] if cs else None, "resumed from"))
    E["states_read"] = []
    for s, v, ref, kind in todo:
        fg = free_gb()
        p = LV + f"{s}_state_v{v}.pt"
        if fg < FREE_LIMIT + 0.03:
            UNREAD.append(f"state {s} after {v}"); print(f"   {s} state after {v}: NOT READ (free disk {fg:.2f} GB)"); continue
        if ref is None:
            print(f"   {s} state after {v}: no commit holds it"); continue
        b = gb(p, ref); out.write_bytes(b); h = sha(out.read_bytes()); out.unlink()
        arm = [a for a in ARMS if stem(a, int(s.split('_w')[1][0])) == s][0]; r = REC[arm, int(s.split("_w")[1][0])]
        rs = {x["visits"]: x["file_sha256"] for x in r["states"]}; rr = {x["done_visits"]: x["file_sha256"] for x in r["resumes"]}
        where = "states" if v in rs else ("resumes" if v in rr else "neither")
        want = rs.get(v) or rr.get(v)
        E["states_read"].append((s, v, h == want, where, ref[:8]))
        rep_named = h in gb(L2 + "E44_L2_REPORT.md").decode()
        print(f"   {s} state after {v} ({kind}, commit {ref[:8]}, {len(b) / 1e6:.1f} MB): sha256 {h[:12]}…; in the receipt under {where}; equal {h == want}; "
              f"named in the report with this sha256: {rep_named}; free disk {fg:.2f} GB")
        del b


# ================================================================ I. form
def part_i():
    print("I. Form: placeholders, names, the committed-file list, terms")
    files = [p for p in git("ls-tree", "-r", "--name-only", REP, "--", L2).decode().splitlines()
             if not p.startswith(L2 + "sweep/") and not p.endswith((".npz", ".pt"))]
    texts = {p: gb(p).decode("utf-8", "replace") for p in files}
    for mark in ("@" * 2, "TO" + "DO", "TB" + "D", "X" * 2):                    # the four placeholder marks, written so that this file holds none
        hits = [(p.split("/")[-1], t.count(mark)) for p, t in texts.items() if mark in t]
        print(f"   '{mark}': {sum(n for _, n in hits)} occurrences in {len(hits)} files {hits[:6]}")
    author = git("log", "-1", "--format=%an", REP).decode().strip()
    roster = [p for p in git("ls-tree", "--name-only", REP).decode().splitlines() if p.endswith("_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md")]
    terms = sorted({author.lower()} | ({roster[0].split("_")[0].lower()} if roster else set()))
    tot, kinds, nf = 0, {"in a branch name (followed by '/')": 0, "in the roster note's file name": 0, "elsewhere": 0}, 0
    where_else = []
    for p, t in texts.items():
        low = t.lower(); hit = False
        for term in terms:
            for m in re.finditer(re.escape(term), low):
                tot += 1; hit = True
                nxt = low[m.end():m.end() + 5]
                k = "in a branch name (followed by '/')" if nxt.startswith("/") else "in the roster note's file name" if nxt.startswith("_task") else "elsewhere"
                kinds[k] += 1
                if k == "elsewhere":
                    where_else.append(p.split("/")[-1])
        nf += hit
    E["names"] = (tot, nf, kinds, len(texts))
    print(f"   names searched: {len(terms)} (from the repository's records, not printed); occurrences in the {len(texts)} text files under L2/ outside sweep/: "
          f"{tot}, in {nf} files; by place: {kinds}; files with other places: {sorted(set(where_else))[:8]}")
    rep = texts[L2 + "E44_L2_REPORT.md"]
    listed = [ln.strip() for ln in rep.splitlines() if ln.startswith(M) or ln.endswith("_20261117.md")]
    at = set(git("ls-tree", "-r", "--name-only", "1e4eceae", "--", L2).decode().splitlines()) | set(
        x for x in git("ls-tree", "-r", "--name-only", "1e4eceae", "--", M + "smallworld/").decode().splitlines() if x in listed)
    at_rep = set(git("ls-tree", "-r", "--name-only", REP, "--", L2).decode().splitlines())
    lm = set(x for x in listed if x.startswith(M))
    print(f"   paths of the list outside L2/: {sorted(x.split('/')[-1] for x in lm if not x.startswith(L2))}")
    print(f"   the report's file list: {len(lm)} paths (L2/ and smallworld/); equal to `git ls-tree` at 1e4eceae for L2/ outside sweep/: "
          f"{lm == {p for p in at if not p.startswith(L2 + 'sweep/')}} ({len(lm)} listed, {len(at)} in the tree); files under L2/ outside sweep/ at {REP} not in the list: "
          f"{sorted(p.split('/')[-1] for p in at_rep - lm if not p.startswith(L2 + 'sweep/'))[:10]}")
    for term in ("blind predictor", "continuation", "declining", "stuck", "still learning", "level", "not distinguished", "rise", "the same", "interim"):
        print(f"   the report uses '{term}' {len(re.findall(re.escape(term), rep))} times")


PARTS = dict(A=part_a, B=part_b, C=part_c, G=part_g, H=part_h, D=part_d, E=part_e, F=part_f, I=part_i)

if __name__ == "__main__":
    if SCR.exists():
        shutil.rmtree(SCR)
    SCR.mkdir(parents=True)
    print(f"free disk at the start: {free_gb():.2f} GB; numpy {np.__version__}; python {sys.version.split()[0]}; started {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
    for p in (sys.argv[1:] or ("A", "B", "C", "G", "H", "F", "D", "E", "I")):
        t = time.time()
        PARTS[p]()
        print(f"   part {p} done in {time.time() - t:.0f} s")
    g = max(abs(x) for _, x in T_GAP)
    print(f"intervals compared with committed ones: {len(T_GAP) // 2}; the committed bounds use the scorer's tabled t (2.3534 for 3 and 1.8946 for 7 degrees of "
          f"freedom); the exact quantiles are {T_EXACT[3]:.6f} and {T_EXACT[7]:.6f}; largest gap of a bound {g:.2e}; readings that change under the exact t: "
          f"{sum(1 for d in DIFFS if 'exact t' in d[0])}")
    print(f"large files left unread for lack of disk: {len(UNREAD)} {UNREAD[:10]}")
    print(f"SUMMARY: figures of committed files compared {N_CMP[0]:,}, differing {len(DIFFS)}; figures of the report's text compared {TXT_N[0]}, "
          f"differing {len(TXT_DIFFS)}; free disk at the end {free_gb():.2f} GB")
    print(f"running time {time.time() - T0:.0f} s; ended {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
