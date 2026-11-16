"""Coordinator's rerun of the E43 audit's main finding, 16 Nov 2026. Read-only, no training. A saved E26 birth with every parameter but the
byte table multiplied by (1 - 0.00025)^t, scored on the first family's reduction pairs (all of them, no letters, whole answers).
    python3 -I decay_alone_check.py <S|B> <seed>"""
import sys
from pathlib import Path
from random import Random
HERE = Path(__file__).resolve(); M = HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"; sys.path.insert(0, str(M))
import exp26_candidate as C26
import arch26
import torch as T
T.set_num_threads(1); arch26.install(); E = C26.E
arm, seed = sys.argv[1], int(sys.argv[2])
sd = T.load(M / "exp45" / "e26" / f"e26_dev_{arm}_{seed}.pt")
rng = Random(0)
items = {}
for t in E.DEV[:2]:
    op, m_, code = t
    pairs = [(a, b) for a in range(m_) for b in range(m_) if (a + b >= m_ if op == "add" else a - b < 0)]
    items[op] = [E.render(t, a, b, 0, rng) for a, b in pairs]
base = None
for upd in (0, 64, 256, 1024, 2048, 4096):
    m = arch26.new_net(seed); f = (1 - 0.00025) ** upd
    m.load_state_dict({k: (v if k.startswith("book.") else v * f) for k, v in sd.items()})
    sc = {op: E.score(m, it)["exact"] / len(it) * 100 for op, it in items.items()}
    red = (sc["add"] + sc["sub"]) / 2
    base = base or red
    print(f"{arm} {seed} t={upd:5d} factor {f:.3f}: add {sc['add']:.1f} sub {sc['sub']:.1f} reduction score {red:.1f} = {100 * red / base:.1f}% of its start")
