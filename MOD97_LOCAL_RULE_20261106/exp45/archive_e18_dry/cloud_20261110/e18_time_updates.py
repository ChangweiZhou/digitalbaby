# Times training updates and one checkpoint's scoring for the E18 birth setup (copies develop17's setup; writes nothing).
import sys, time
sys.path.insert(0, "/home/user/digitalbaby/.worktrees/codex/MOD97_LOCAL_RULE_20261106")
import exp16_integrated as I
E, X, T = I.E, I.X, I.T
seed = 73129
m = E.new_net(seed)
plain = I.HEADS16 == "top" and I.BIRTHRULE16 == "D"
opt = I.adamw16(m, .001, .3 if I.BIRTHRULE16 == "D" else E.WD)
step = (lambda items: E.update(m, opt, items, "D")) if plain else (lambda items: I.update16(m, opt, items, I.BIRTHRULE16, None))
tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
bt = E.batches(tp, seed, E.KMAX["M"])
for _ in range(16): step(next(bt))
t0 = time.process_time()
for _ in range(256): step(next(bt))
u = (time.process_time() - t0) / 256
t0 = time.process_time(); {n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}; ev = time.process_time() - t0
print(f"per_update {u:.4f}s  eval {ev:.1f}s  forecast_curve {24576*u + 6*ev:.0f}s")
