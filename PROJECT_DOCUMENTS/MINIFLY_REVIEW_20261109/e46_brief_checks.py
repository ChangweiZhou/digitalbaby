"""Checks behind the audit of the E46 brief at 3126dead (16 November 2026): does learning the first family need weight decay?

Read-only.  Nothing is trained, torch is not used, and no code of the default branch or of a worker's branch is run: files are
read with `git show`, and AdamW's first update is rebuilt in numpy from its published form.

  A  the reference the brief reads: E41's two present-width cells of seed 73129 on the default branch (receipts, weights,
     runner hash, sources), and what their `optimizer` field holds
  B  the length test at 40 letters under the standing decay around update 12,288 (E42's `G` runs, which are the same
     network and data): how far single rows and three-row means move; and that `G` equals E41's cell row by row
  C  AdamW's first update at decay d and at decay 0 from the same start and batch: the relation the brief's check 2 asserts,
     and the relation that holds
  D  the sentence the job licenses: every combination of the three cells, and where two readings of the brief's words differ
  E  what each decay does to a weight that nothing pushes on, over 12,288 and 24,576 updates
  F  cost, from the CPU figures in E41's and E42's receipts
  G  the full dose: E26's six birth hashes, and whether the standing network run again with the added scores (E42's `G`)
     reproduces them
  H  the brief's figures from E38 (a second family with and without decay), from the receipts

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e46_brief_checks.py [A B C D E F G H]
"""
import hashlib, json, subprocess, sys
from itertools import product
import numpy as np

REF = '3126dead'                                    # the default branch at the brief
E42 = 'origin/claude/gallant-dijkstra-i9zgsr'       # E42's worker (seed 73129 complete at 7fbdbdc2)
M = 'MOD97_LOCAL_RULE_20261106/'
want = set(sys.argv[1:]) or set("ABCDEFGH")
show = lambda p, ref=REF: subprocess.check_output(['git', 'show', f'{ref}:{p}'])
sha = lambda b: hashlib.sha256(b).hexdigest()


def three(c, key, thr, upto=None):
    v = [(r['step'], r[key]) for r in c if r['step'] > 0 and (upto is None or r['step'] <= upto)]
    for i in range(len(v) - 2):
        if all(x[1] >= thr for x in v[i:i + 3]):
            return v[i][0]
    return None


if "A" in want:
    print("A. the reference on the default branch at", REF)
    print(f"   exp41_scale.py at {REF}: sha256 {sha(show(M + 'exp41_scale.py'))[:16]}")
    for rule in "SB":
        d = json.loads(show(M + f'exp45/e41/e41_{rule}_h512_f50_73129.json'))
        pv = d['provenance']
        differ = [s for s, h in pv['sources_sha256'].items() if sha(show(M + s)) != h]
        wok = sha(show(M + f'exp45/e41/e41_{rule}_h512_f50_73129.pt')) == d['weights_sha256']
        c = d['curve']
        print(f"   {rule}: complete {d['complete']}, dose {d['dose']}, rows {len(c)}, tensors_sha256 {d['tensors_sha256'][:16]}, runner {pv['runner_sha256'][:16]}; "
              f"recorded sources differing from the default branch: {differ}; weights file equals its recorded hash: {wok}")
        print(f"      80 on A at {three(c, 'A', 80)}, 80 on R at {three(c, 'R', 80)} (three rows in a row); "
              f"`optimizer` field: weight_decay {d['optimizer']['weight_decay']} beside groups {[g['weight_decay'] for g in d['optimizer']['groups']]}")
        print(f"      CPU: {d['cpu_updates'] / d['dose']:.4f} s per update, {d['cpu_scoring'] / len(c):.2f} s per scored row")
    src = show(M + 'exp26_candidate.py').decode()
    line = [ln.strip() for ln in src.splitlines() if 'def opt_record' in ln or ('weight_decay=WD' in ln and 'name="AdamW"' in ln)]
    print("   exp26_candidate.opt_record:", ' | '.join(line))
    print("   so a receipt written through opt_record names the constant 0.25 as `weight_decay` whatever the group holds")

if "B" in want:
    print("B. the length test under the standing decay near update 12,288 (E42's `G`, seed 73129)")
    for rule in "SB":
        g = json.loads(show(M + f'exp45/e42/e42_G_{rule}_73129.json', E42))['curve']
        e = json.loads(show(M + f'exp45/e41/e41_{rule}_h512_f50_73129.json'))['curve']
        ge = {r['step']: r for r in g}
        same = all(ge[r['step']][k] == r[k] for r in e for k in ('A_counts', 'R_counts', 'taught_counts'))
        rows = list(range(9216, 12289, 256))
        l40 = [ge[s]['len40'] for s in rows]
        m3 = [sum(l40[i:i + 3]) / 3 for i in range(len(l40) - 2)]
        d1 = np.diff(l40)
        print(f"   {rule}: `G` equals E41's cell at every one of its {len(e)} rows (A, R and taught counts): {same}")
        print(f"      40 letters at rows 9,216 to 12,288: {[round(v, 1) for v in l40]}")
        print(f"      single rows from {min(l40):.1f} to {max(l40):.1f}; three-row means from {min(m3):.1f} to {max(m3):.1f}; "
              f"final row {l40[-1]:.1f}, mean of the last three {m3[-1]:.1f}; standard deviation of a row-to-row change {d1.std():.1f}")
        end = [ge[s]['len40'] for s in (24064, 24320, 24576)]
        print(f"      for scale, the same test at the end of the full birth (rows 24,064 to 24,576): {[round(v, 1) for v in end]}")
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', E42, '--', M + 'exp45/e42/'], text=True).split()
    print("   `G` receipts of the other seeds on that branch:", [n.split('/')[-1] for n in names if '/e42_G_' in n and '73129' not in n and n.endswith('.json') and 'structure' not in n] or "none yet")

if "C" in want:
    print("C. AdamW's first update from one start and one batch, at decay d and at decay 0 (float32, the order of torch's AdamW: shrink, then step)")
    lr, b1, b2, eps = np.float32(1e-3), 0.9, 0.98, np.float32(1e-8)
    rng = np.random.default_rng(46)

    def first_update(p0, g, d):
        p = p0 * np.float32(1 - float(lr) * d)                       # param.mul_(1 - lr * weight_decay)
        m = np.float32(1 - b1) * g                                   # exp_avg after one step
        v = np.float32(1 - b2) * g * g                               # exp_avg_sq after one step
        denom = np.sqrt(v) / np.float32(np.sqrt(1 - b2)) + eps
        return (p - np.float32(float(lr) / (1 - b1)) * (m / denom)).astype(np.float32)

    shapes = {"l1.weight": ((512, 8), 8 ** -0.5), "rnn.weight_ih_l0": ((768, 512), 1 / 16), "rnn.weight_hh_l0": ((768, 256), 1 / 16),
              "rnn.bias_hh_l0": ((768,), 1 / 16), "lm.weight": ((512, 256), 1 / 16), "l2.weight": ((512, 512), 512 ** -0.5), "l3.weight": ((257, 512), 512 ** -0.5)}
    for d in (0.25, 0.05):
        print(f"   decay {d}:")
        for name, (shape, bound) in shapes.items():
            p0 = rng.uniform(-bound, bound, shape).astype(np.float32)
            g = (rng.standard_normal(shape) * 1e-3).astype(np.float32)
            g[rng.random(shape) < 0.02] = 0                           # a few entries with no gradient at all
            pd, pz = first_update(p0, g, d), first_update(p0, g, 0.0)
            brief = np.abs(pz.astype(np.float64) - pd.astype(np.float64) / (1 - float(lr) * d)) / np.maximum(np.abs(pz.astype(np.float64)), 1e-30)
            delta = pz.astype(np.float64) - pd.astype(np.float64)
            dev = np.abs(delta - float(lr) * d * p0.astype(np.float64))
            dhat = float((delta * p0).sum() / (float(lr) * (p0.astype(np.float64) ** 2).sum()))
            print(f"      {name:18s} brief's relation within 1e-6 relative: {100 * (brief <= 1e-6).mean():5.1f}% of entries "
                  f"(of those with a gradient {100 * (brief[g != 0] <= 1e-6).mean():5.1f}%, without one {100 * (brief[g == 0] <= 1e-6).mean():5.1f}%), median deviation {np.median(brief):.1e}"
                  f" | p(0) - p(d) = lr d p0: largest miss {dev.max():.1e} = {dev.max() / np.abs(p0).max():.1e} of the largest weight; decay estimated from the difference {dhat:.5f}")
    print("   the step lr * g / (|g| + eps) is the same in both runs, so the two differ by lr * d * p0 added, not by a factor on the result")

if "D" in want:
    print("D. the sentence, over every combination of the three cells")
    r80 = ["falls", "stays", "rises", "not decided", "neither reaches", "mixed"]
    lev = ["lower", "about the same", "higher", "mixed"]

    def sentence(a, b, c, not_reached_is_bad):
        bad_r = a == "rises" or (not_reached_is_bad and a in ("not decided", "neither reaches"))
        if bad_r or b == "lower" or c == "lower":
            return "needs"
        if a in ("falls", "stays") and b in ("about the same", "higher") and c in ("about the same", "higher"):
            return "does not need"
        return "mixed"
    count = {}
    differ = []
    for a, b, c in product(r80, lev, lev):
        s1, s2 = sentence(a, b, c, True), sentence(a, b, c, False)
        count[s2] = count.get(s2, 0) + 1
        if s1 != s2:
            differ.append((a, b, c))
    print(f"   {len(r80) * len(lev) ** 2} combinations; read on the cells' descriptions alone: {count}")
    print(f"   combinations where 'not reached' read as the run's own failure gives another sentence: {len(differ)}, all with the threshold cell in "
          f"{sorted({a for a, _, _ in differ})} and neither level lower")
    print("   'needs' is given by one cell alone in", sum(1 for a, b, c in product(r80, lev, lev) if sentence(a, b, c, False) == 'needs' and [a == 'rises', b == 'lower', c == 'lower'].count(True) == 1), "of them, so the sentence must name the cell")
    D_ = 12288 - 512
    print(f"   a run that does not reach the threshold can be described as *rises* only if the reference reached it by {int(D_ / 1.5):,} "
          f"(the last recordable row {D_:,} over 1.5); at 24,576 updates, by {int((24576 - 512) / 1.5):,}")

if "E" in want:
    print("E. a weight that nothing pushes on, multiplied by (1 - 0.001 d) at every update")
    for d in (0.25, 0.05, 0.0):
        print(f"   decay {d}: time constant {'none' if d == 0 else format(int(1 / (1e-3 * d)), ',') + ' updates'}; left after 2,048 updates {(1 - 1e-3 * d) ** 2048:.3f}, "
              f"after 12,288 {(1 - 1e-3 * d) ** 12288:.3f}, after 24,576 {(1 - 1e-3 * d) ** 24576:.4f}, after a life of 36,864 {(1 - 1e-3 * d) ** 36864:.5f}")

if "F" in want:
    print("F. cost")
    upd = {r: json.loads(show(M + f'exp45/e41/e41_{r}_h512_f50_73129.json')) for r in "SB"}
    per_update = {r: upd[r]['cpu_updates'] / upd[r]['dose'] for r in "SB"}
    g = {r: json.loads(show(M + f'exp45/e42/e42_G_{r}_73129.json', E42))['curve'] for r in "SB"}
    per_row = {r: g[r][-1]['cpu_scoring'] / len(g[r]) for r in "SB"}                 # A, R, taught and the length test, as E46 scores
    print(f"   per update (E41's runs): S {per_update['S']:.4f}, B {per_update['B']:.4f} CPU-s; per scored row with the length test (E42's `G`): S {per_row['S']:.2f}, B {per_row['B']:.2f} CPU-s")
    for dose in (12288, 24576):
        rows = dose // 256 + 1
        one = {r: per_update[r] * dose + per_row[r] * rows for r in "SB"}
        for n, label in ((12, "the twelve new runs"), (16, "with four reference cells for the two missing seeds"), (18, "all three decays through one runner")):
            print(f"   dose {dose:,}: {label}: {(n / 2) * (one['S'] + one['B']) / 3600:.1f} CPU-hours")

if "G" in want:
    print("G. the full dose: E26's births, and E42's `G` runs of the same network with A, R, the taught panel and the length test scored every 256 updates")
    for rule in "SB":
        hs = []
        for seed in (73129, 93683, 116089):
            e = json.loads(show(M + f'exp45/e26/e26_dev_{rule}_{seed}.json'))
            hs.append(e['tensors_sha256'][:8])
            if seed == 73129:
                g = json.loads(show(M + f'exp45/e42/e42_G_{rule}_73129.json', E42))
                print(f"   {rule} 73129: E26's birth (dose {e['dev_updates']:,}) {e['tensors_sha256'][:16]}; E42's `G` {g['tensors_sha256'][:16]}; equal {g['tensors_sha256'] == e['tensors_sha256']}")
        print(f"   {rule}: E26's `tensors_sha256` for seeds 73129, 93683, 116089 begin {hs}")

if "H" in want:
    print("H. E38, seed 73129: the new family (add and sub mod 97, held-out at 12 letters, 400 each) at the end; `none` is the arm without decay, `zero` the arm with it")
    for arm in ("none", "zero"):
        out = []
        for rule in "SB":
            d = json.loads(show(M + f'exp45/e38/e38_{arm}_{rule}_73129.json'))
            r = d['rows'][-1]['new_heldout_k12']
            out.append(f"{rule} {100 * (r['add97']['exact'] + r['sub97']['exact']) / 800:.2f}")
        print(f"   {arm}: {', '.join(out)}")
