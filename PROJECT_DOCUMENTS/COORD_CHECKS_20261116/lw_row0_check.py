"""Coordinator's check for the long-life world brief, 16 Nov 2026. Read-only, no training. What a born network answers on a new family of
the world before any update: asked pairs of add and sub, no letters, split by reduction; and max and min.
    python3 -I lw_row0_check.py <S|B> <world_seed> [family ...]      the born network is E41's half-width birth of seed 73129"""
import sys
from pathlib import Path
from random import Random
HERE = Path(__file__).resolve(); M = HERE.parents[2] / "MOD97_LOCAL_RULE_20261106"
sys.path.insert(0, str(M)); sys.path.insert(0, str(HERE.parent))
import exp26_candidate as C26
import arch26, exp41_scale as X41
import torch as T
import lw_world_spec as LW
T.set_num_threads(1); arch26.install(); E = C26.E
rule, wseed = sys.argv[1], int(sys.argv[2]); fams = [int(v) for v in sys.argv[3:]] or [0, 1, 2]
m = X41.NetW(256, 128); m.load_state_dict(T.load(M / "exp45" / "e41" / f"e41_{rule}_h256_f50_73129.pt"))
w = LW.World(wseed); rng = Random(0)
def items(t, pairs):
    out = []
    for a, b in pairs:
        prompt = t["code"] + f"{a:02d}".encode() + bytes([t["symbol"]]) + f"{b:02d}".encode() + b"="
        out.append((list(prompt), list(f"{LW.apply(t['op'], t['p'], a, b):02d}".encode()) + [LW.EOS]))
    return out
for g in fams:
    row = []
    for t in w.groups[g]:
        asked = rng.sample(t["asked"], min(600, len(t["asked"])))
        if t["op"] in ("add", "sub"):
            red = [pr for pr in asked if LW.needs_wrap(t["op"], t["p"], *pr)]; non = [pr for pr in asked if not LW.needs_wrap(t["op"], t["p"], *pr)]
            row.append(f"{t['op']}: reduction {100 * E.score(m, items(t, red))['exact'] / len(red):.1f}, none {100 * E.score(m, items(t, non))['exact'] / len(non):.1f}")
        else:
            row.append(f"{t['op']}: {100 * E.score(m, items(t, asked))['exact'] / len(asked):.1f}")
    print(f"born {rule} 73129 (half width), world {wseed}, family {g} (mod {w.groups[g][0]['p']}), before any update: " + "; ".join(row))
