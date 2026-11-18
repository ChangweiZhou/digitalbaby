"""Checks behind the audit of the E36, E37 and E38 briefs at 4dd0b9f7 (15 November 2026).

Read-only: receipts, saved tensors and a pure-Python rebuild of the streams.  Nothing trains and no harness code is run.

A. The E36 brief's table: held-out add and sub mod 89 at 12 letters, mean of the two, by stage, recomputed from the nine life
   receipts; with it, each reminder's starting level (the last stage), which the marks have to allow for.
B. The saved feedback matrices of the three K lives: the hash the receipt records for the end of the life, the worker that made
   them, and the matrix beside the life-end recurrent weights (the receipt's own last-stage measures).
C. The life's first two blocks rebuilt from X16.entry_stream(family_entries("F1", seed), seed + 101) in batches of 64 and hashed as
   exp26_candidate.Digest does: equal to stages[1] and stages[2] of the receipts or not.  Then the family drawn for every item of
   blocks 3 to 6 by X16.joint_stream, counted and set beside the receipts' exposure counts.
D. The rehearsal stream of E38: its entries, its random-generator seed beside every other stream's, the letters it renders, the
   share of each block it takes under joint_stream, and that none of its pairs is a held-out pair.
E. The number of weights outside the embedding table (what an anchor stores), and the reminder stream of E36 beside the birth's.
F. What the proposed marks do on figures already on file.
G. Scoring cost per item from the life receipts, and what scoring all 7,921 pairs at every row adds.
H. Score 2 of the briefs as a second implementation: whole answers (two digits and the end symbol) on every pair of add and sub
   mod 89 without letters, split by whether the pair needs the wrap.  Usage: ... H <file under exp45/> [<file> ...] [--ref <git ref>]

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/e36_e38_brief_checks.py [A B C D E F G]
"""
import hashlib, json, os, sys
from random import Random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from np_gru import git_bytes, load_weights, split, render, DEV, LIFE, BOS, EOS, P
from compare_pt_tensors import tensors
from e32_r3_checks import batch_tensor

REF = '4dd0b9f7'; SEEDS = (73129, 93683, 116089); BATCH = 64; KMAX = 12; BLOCK = 6144
F1 = LIFE; F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
want = {a for a in sys.argv[1:] if len(a) == 1} or set("ABCDEFG")
rec = lambda arm, s: json.loads(git_bytes(P + (f'e31/e31_life_K_{s}.json' if arm == 'K' else f'e26/e26_life_{arm}_{s}.json'), REF))
pc = lambda d: 100 * d['exact'] / d['n']
hard = lambda sc, m: (pc(sc[f'add{m}']['12']) + pc(sc[f'sub{m}']['12'])) / 2

def entry_stream(entries, seed):                      # exp16_integrated.entry_stream / render for ("op", task, a, b)
    rng = Random(seed + 77)
    while True:
        order = list(range(len(entries))); rng.shuffle(order)
        for i in order:
            _, t, a, b = entries[i]
            yield render(t, a, b, rng.randint(0, KMAX), rng)

def family(fam, seed, frac=.25):
    return [("op", t, a, b) for t in (F1 if fam == "F1" else F3) for a, b in split(t, frac, seed)[0]]

if "A" in want:
    print("A. birth family, held-out add/sub at 12 letters (mean), stages 0 to 6")
    for arm in "SBK":
        for s in SEEDS:
            st = rec(arm, s)['stages']
            print(f"   {arm} {s}: " + ", ".join(f"{hard(x['scores'], 89):.1f}" for x in st))

if "B" in want:
    print("B. the K lives' saved feedback matrices")
    for s in SEEDS:
        r = rec('K', s); blob = git_bytes(P + f'e31/e31_life_K_{s}_Bhh.pt', REF); t = tensors(blob)
        (name, (stype, shape, raw)), = t.items()
        h = hashlib.sha256(); h.update(b"Bhh"); h.update(b"torch.float32"); h.update(str(tuple(shape)).encode()); h.update(raw)
        B = np.frombuffer(raw, '<f4').reshape(shape).astype(np.float64)
        W = load_weights(P + f'e31/e31_life_K_{s}.pt', REF)[0]['rnn.weight_hh_l0'].astype(np.float64)
        cos = float((B * W).sum() / np.sqrt((B * B).sum() * (W * W).sum())); rel = float(np.linalg.norm(B - W) / np.linalg.norm(W))
        last = r['stages'][-1]['e31']
        print(f"   {s}: worker {r['e31']['worker']}, runner files {sorted(r['e31']['runner_sources_sha256'])}; tensor {name.strip('/')} {stype} {tuple(shape)}; "
              f"hash equals the receipt's Bhh_final_tensors_sha256: {h.hexdigest() == r['e31']['Bhh_final_tensors_sha256']}; "
              f"cos(Bhh, W) {cos:.10f} (receipt {last['cos_Bhh_W']:.10f}), |Bhh - W| / |W| {rel:.3e} (receipt {last['rel_diff_Bhh_W']:.3e}), updates {last['feedback_updates']}")

if "C" in want:
    print("C. the life's first two blocks rebuilt from the F1 entry stream alone (stream seed = seed + 101), batches of 64")
    for s in SEEDS:
        st = entry_stream(family("F1", s), s + 101); h = hashlib.sha256(); out = []
        for blk in (1, 2):
            for _ in range(BLOCK):
                for t in batch_tensor([next(st) for _ in range(BATCH)]):
                    h.update(str(tuple(t.shape)).encode()); h.update(t.tobytes())
            got = h.copy().hexdigest()
            out.append(" ".join(f"{arm}:{rec(arm, s)['stages'][blk]['data_digest']['sha256'] == got}" for arm in "SBK"))
        print(f"   {s}: digest after block 1 equals stages[1] ({out[0]}); after block 2 equals stages[2] ({out[1]})")
    print("   family drawn per item by joint_stream (Random(seed + si + 78), uniform over the families seen), beside the receipts")
    for s in SEEDS:
        r = rec('S', s)['stages']; line = []
        for si, nf in ((3, 2), (4, 2), (5, 3), (6, 3)):
            rng = Random(s + si + 78); c = [0] * nf
            for _ in range(BLOCK * BATCH): c[rng.randrange(nf)] += 1
            got = dict(zip(("F1", "F2", "F3"), c)); rcp = {k.split('/')[1]: v[0] for k, v in r[si]['exposure_rows_tokens'].items()}
            line.append(f"block {si}: {got} {'=' if got == rcp else '!='} receipt")
        print(f"   {s}: " + "; ".join(line))
        f3 = [r[si]['exposure_rows_tokens']['joint/F3'][0] for si in (5, 6)]
        print(f"      F3 items: block 5 {f3[0]} (row 2,048 of a fresh run: 131,072); blocks 5 and 6 together {sum(f3)} (row 4,096: 262,144)")

if "D" in want:
    print("D. E38's rehearsal stream: X16.entry_stream(birth_entries, seed + 100)")
    for s in SEEDS:
        ent = [("op", t, a, b) for t in DEV for a, b in split(t, .5, s)[0]]
        held = {(t[0], a, b) for t in DEV for a, b in split(t, .5, s)[1]}
        leak = sum((e[1][0], e[2], e[3]) in held for e in ent)
        st = entry_stream(ent, s + 100); ks = []
        for _ in range(20000):
            p, _t = next(st); ks.append(sum(65 <= b <= 90 for b in p))
        seeds = {"birth stream (exp11_cell.stream)": s + 77, "E36 reminder (seed + 36)": s + 36 + 77, "rehearsal (seed + 100)": s + 100 + 77,
                 **{f"life family stream si={si}": s + 100 + si + 77 for si in (1, 3, 5)}, **{f"joint stream si={si}": s + si + 78 for si in (1, 2)}}
        share = []
        for si in (1, 2):
            rng = Random(s + si + 78); share.append(sum(rng.randrange(2) == 0 for _ in range(BLOCK * BATCH)))
        print(f"   {s}: {len(ent)} entries ({len(ent) // 4} taught pairs per operation); held-out pairs among them {leak}; letters {min(ks)}..{max(ks)}, mean {np.mean(ks):.2f}; "
              f"birth-family items in block 1 / 2: {share[0]} / {share[1]} of {BLOCK * BATCH}; generator seeds all distinct: {len(set(seeds.values())) == len(seeds)}")
        codes = sorted({chr(t[2][0]) for t in DEV}); f1c = sorted({chr(t[2][0]) for t in F1}); f3c = sorted({chr(t[2][0]) for t in F3})
    print(f"   code bytes: birth {codes}, F1 {f1c}, F3 {f3c}: life()'s fam_of files a birth item under 'other', so its count appears as exposure 'joint/other'")

if "E" in want:
    W = load_weights(P + 'e26/e26_dev_S_73129.pt', REF)[0]
    n = sum(v.size for k, v in W.items() if not k.startswith('book.'))
    print(f"E. weights outside the embedding table (group 0 of make_opt; what an anchor stores): {n:,} in {sum(not k.startswith('book.') for k in W)} tensors; embedding table {W['book.weight'].size:,}")
    for s in SEEDS[:1]:
        tp = [(t, split(t, .5, s)[0]) for t in DEV]; flat = [(t, pr) for t, pairs in tp for pr in pairs]
        def stream(seed):
            rng = Random(seed + 77)
            while True:
                order = list(range(len(flat))); rng.shuffle(order)
                for i in order:
                    t, (a, b) = flat[i]; yield render(t, a, b, rng.randint(0, KMAX), rng)
        a, b = stream(s), stream(s + 36)
        same = sum(next(a) == next(b) for _ in range(BATCH * 16))
        print(f"   E36 reminder stream (seed + 36) beside the birth's, seed {s}: identical items among the first 1,024: {same}; taught pairs per operation {len(flat) // 4}; items per operation in 256 updates {256 * BATCH // 4}")

if "F" in want:
    print("F. the marks on figures already on file")
    start = {arm: [hard(rec(arm, s)['stages'][-1]['scores'], 89) for s in SEEDS] for arm in "SBK"}
    for arm in "SBK":
        print(f"   E36, reminder {arm} at row 0: {', '.join(f'{v:.1f}' for v in start[arm])}  -> seeds already 30 or more above a reference that scores 0: {sum(v >= 30 for v in start[arm])} of 3")
    for arm in "SB":
        st = rec(arm, 73129)['stages']
        print(f"   E38, E26 life {arm} 73129: birth family after block 1 / 2: {hard(st[1]['scores'], 89):.1f} / {hard(st[2]['scores'], 89):.1f} (the 'loses' mark is 50); F1 after block 1 / 2: {hard(st[1]['scores'], 97):.1f} / {hard(st[2]['scores'], 97):.1f} (the 'acquires' mark is 85)")
    from np_gru import panel
    for s in SEEDS:                                       # what "everything kept except the wrap" scores on the receipts' held-out panels
        share = []
        for t in DEV[:2]:
            pk = panel(t, split(t, .5, s)[1], s, (12,), 300)[12]
            ab = [(int(bytes(p[1:3])), int(bytes(p[-3:-1]))) for p, _ in pk]
            share.append(100 * np.mean([(x + y < 89) if t[0] == "add" else (x - y >= 0) for x, y in ab]))
        st = {arm: rec(arm, s)['stages'] for arm in "SB"}
        print(f"   held-out panel of seed {s}: pairs that need no wrap, add {share[0]:.1f}%, sub {share[1]:.1f}%, mean {np.mean(share):.1f}%; E26 after block 1 and 2 (add, sub): "
              + "; ".join(f"{arm} " + " ".join(f"({pc(st[arm][i]['scores']['add89']['12']):.1f}, {pc(st[arm][i]['scores']['sub89']['12']):.1f})" for i in (1, 2)) for arm in "SB"))
    for arm in "SBK":
        for fam, m, stages in (("F3", 73, (5, 6)), ("F1", 97, (1,))):
            print(f"   E37, life {arm}, {fam}: " + "; ".join(f"after block {si}: " + ", ".join(f"{hard(rec(arm, s)['stages'][si]['scores'], m):.1f}" for s in SEEDS) for si in stages))

if "G" in want:
    print("G. scoring cost")
    for arm in "SB":
        r = rec(arm, 73129); st = r['stages']
        per = [(st[i]['cpu_scoring_so_far'] - st[i - 1]['cpu_scoring_so_far']) for i in range(2, 7)]
        sc = st[1]['scores']; items = 0
        for k, v in sc.items():
            if k in ('compose', 'silenced_compose'): items += sum(x['n'] for x in v.values())
            elif k == 'taught': items += v['n']
            else: items += sum(x['n'] for kk in v.values() for x in ([kk] if 'n' in kk else kk.values())) if isinstance(v, dict) else 0
        upd = (st[2]['cpu_updates_so_far'] - st[1]['cpu_updates_so_far']) / BLOCK
        print(f"   life {arm} 73129: one evaluate() scores {items:,} items in {np.mean(per):.0f} CPU-s ({1000 * np.mean(per) / items:.2f} ms per item, lengths mixed up to 40 letters); one update {upd:.3f} CPU-s")
    print("   scoring 2 x 7,921 pairs at one row costs about as much as", f"{2 * 7921 * 0.003 / 0.13:.0f}", "updates at 3 ms per item and 0.13 s per update (short items cost less than the mixed average)")

def wrap_split(W, task):
    """Teacher-forced exactness of all three target bytes equals greedy whole-answer exactness (the brief's exp16_integrated.score)."""
    from np_gru import sigmoid, apply, SYM
    op, m, code = task
    ab = np.array([(a, b) for a in range(m) for b in range(m)]); a, b = ab[:, 0], ab[:, 1]
    c = np.array([apply(op, m, int(u), int(v)) for u, v in ab])
    x = np.array([[BOS] + list(code) + list(f"{u:02d}".encode()) + [ord(SYM[op])] + list(f"{v:02d}".encode()) + [ord("=")] + list(f"{r:02d}".encode()) for (u, v), r in zip(ab, c)], dtype=np.int64)
    tgt = np.array([list(f"{r:02d}".encode()) + [EOS] for r in c], dtype=np.int64)
    z1 = np.tanh(W['book.weight'] @ W['l1.weight'].T + W['l1.bias'])
    gi_tok = z1 @ W['rnn.weight_ih_l0'].T + W['rnn.bias_ih_l0']; l2_tok = z1 @ W['l2.weight'].T + W['l2.bias']
    Whh, bhh = W['rnn.weight_hh_l0'], W['rnn.bias_hh_l0']; U = Whh.shape[1]
    h = np.zeros((len(x), U), np.float32); ok = np.ones(len(x), bool); L = x.shape[1]
    for t in range(L):
        gi = gi_tok[x[:, t]]; gh = h @ Whh.T + bhh
        r = sigmoid(gi[:, :U] + gh[:, :U]); z = sigmoid(gi[:, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, 2 * U:] + r * gh[:, 2 * U:]); h = (1 - z) * n + z * h
        if t >= L - 3:
            ok &= (np.tanh(l2_tok[x[:, t]] + h @ W['lm.weight'].T + W['lm.bias']) @ W['l3.weight'].T + W['l3.bias']).argmax(1) == tgt[:, t - (L - 3)]
    wrap = (a + b >= m) if op == "add" else (a - b < 0)
    return 100 * ok[~wrap].mean(), 100 * ok[wrap].mean(), int(wrap.sum())

if "H" in want:
    args = sys.argv[1:]; ref = args[args.index("--ref") + 1] if "--ref" in args else REF
    files = [a for a in args if a.endswith(".pt")]
    print("H. score 2 (whole answers, all pairs, no letters): no wrap / wrap, add then sub")
    for f in files:
        W = load_weights(P + f, ref)[0]
        (an, aw, na), (sn, sw, ns) = wrap_split(W, DEV[0]), wrap_split(W, DEV[1])
        print(f"   {f}: add {an:.1f} / {aw:.1f} ({na} wrap pairs); sub {sn:.1f} / {sw:.1f} ({ns} wrap pairs); wrap mean {(aw + sw) / 2:.1f}")
