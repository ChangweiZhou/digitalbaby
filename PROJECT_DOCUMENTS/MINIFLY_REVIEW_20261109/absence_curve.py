"""What is answered at a return against the length of the absence, in the long lives (read-only; the outside memo's proposal 5.4, first step).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/absence_curve.py

Reads the committed score files of the lives with `git show` (no file of a worker's branch is executed; nothing is written):
the first long lives at 45e9ba16 (verified: E44_L2_VERIFICATION_20261118.md, every figure of these files recomputed from the event
files) and the sweep's long lives at 738e2d61 (verified: E44_L2_SWEEP_VERIFICATION_20261118.md). A *return* is a visit to a family met
before; its *absence* is the number of visits since that family's previous visit (1: the visit directly before). *Kept* is the share of
the family's items answered right at the first update of the return, before anything of the visit is learned (the note's D8).
*First 16* is the mean share right over the first 16 updates of the return (D10). *Before* is the share right over the last quarter of
the family's previous visit. The classes of absence are powers of two."""
import json, subprocess

M = "MOD97_LOCAL_RULE_20261106/exp45/e44/L2/"
SRC = {"naive": ("45e9ba16", M + "scores/lw_naive_w{w}_v1024_V1024.json", (1, 2, 3, 4)),
       "replay": ("45e9ba16", M + "scores/lw_replay_w{w}_v1024_rho0.5_V1024.json", (1, 2, 3, 4)),
       "sweep": ("738e2d61", M + "sweep/scores/lw_sweep_w{w}_v1024_V1024.json", (1, 2, 3, 4)),
       "sweep+replay": ("738e2d61", M + "sweep/scores/lw_sweep+replay_w{w}_v1024_rho0.5_V1024.json", (1, 2, 3, 4)),
       "naive-wd0 (256 visits)": ("45e9ba16", M + "scores/lw_naive_w{w}_v256_wd0_V256.json", tuple(range(1, 9))),
       "replay-wd0 (256 visits)": ("45e9ba16", M + "scores/lw_replay_w{w}_v256_rho0.5_wd0_V256.json", tuple(range(1, 9))),
       "naive, first 256 visits": ("45e9ba16", M + "scores/lw_naive_w{w}_v1024_V256.json", (1, 2, 3, 4)),
       "replay, first 256 visits": ("45e9ba16", M + "scores/lw_replay_w{w}_v1024_rho0.5_V256.json", (1, 2, 3, 4))}
BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 10 ** 9))
name = lambda lo, hi: str(lo) if lo == hi else (f"{lo} or more" if hi > 10 ** 6 else f"{lo} to {hi}")


def show(ref, path):
    return json.loads(subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True).stdout)


def main():
    for arm, (ref, pat, worlds) in SRC.items():
        later = {w: show(ref, pat.format(w=w))["later_visits"] for w in worlds}
        print(f"\n{arm}: {len(worlds)} worlds, {sum(len(v) for v in later.values()):,} returns (score files at {ref})")
        print("   absence | returns | items | kept, pooled | kept by world, smallest to largest | first 16 | before")
        for lo, hi in BINS:
            rs = {w: [r for r in later[w] if lo <= r["absence"] <= hi] for w in worlds}
            allr = [r for w in worlds for r in rs[w]]
            if not allr:
                continue
            n = sum(r["n"] for r in allr); k = sum(r["right"] for r in allr)
            byw = sorted(sum(r["right"] for r in rs[w]) / sum(r["n"] for r in rs[w]) for w in worlds if rs[w])
            f16 = sum(r["first16"] for r in allr) / len(allr); bef = sum(r["previous_last_quarter"] for r in allr) / len(allr)
            print(f"   {name(lo, hi):>11} | {len(allr):5d} | {n:6d} | {k / n:.3f} | {byw[0]:.3f} to {byw[-1]:.3f} ({len(byw)} worlds) | {f16:.3f} | {bef:.3f}")
        # the least-squares slope of kept (by return) on log2(absence), absences of 2 or more, pooled over worlds with items as weights
        import math
        pts = [(math.log2(r["absence"]), r["right"] / r["n"], r["n"]) for w in worlds for r in later[w] if r["absence"] >= 2 and r["n"]]
        sw = sum(p[2] for p in pts); mx = sum(p[0] * p[2] for p in pts) / sw; my = sum(p[1] * p[2] for p in pts) / sw
        b = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts) / sum(p[2] * (p[0] - mx) ** 2 for p in pts)
        print(f"   slope of kept on log2(absence), absences of 2 or more, weighted by items: {b:+.4f} a doubling (mean kept {my:.3f}, {len(pts)} returns)")


if __name__ == "__main__":
    main()
