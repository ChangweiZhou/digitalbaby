"""Sizing sweep, stage 1: the frozen taught panels (brief rev. 3, "What is read").

For each world and modulus: 96 taught pairs per family that the revision does not change (family 1: a cue that is not among the changed
cues; family 2: any), each rendered twice: once with filler of 10 to 14 bytes (so the context is out of the last ten bytes) and once with
filler of 0 to 2 bytes. Prompts end with '='. The truth is the pair's outcome, which is the same before and after the change point.

The taught set depends on the world and the modulus (the twin redraws depend on the values), not on the presentations: the panel is built
once per (world, modulus), and the build checks that the taught sets at 4, 32 and 256 presentations are identical.

    python3 sz_panels.py      writes TAUGHT_PANELS.json for worlds 101, 102, 103 (the roster) and 104 (profiling only), moduli 97, 31, 11
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_config as CF     # noqa: E402
import sz_stream as G      # noqa: E402

N_PER_FAMILY = 96
LONG, SHORT = (10, 14), (0, 2)
OUT = HERE / "TAUGHT_PANELS.json"


def build_panel(cfg, seed):
    world = G.make_world(cfg, seed); split = G.make_split(cfg, world, seed)
    r = G.rng(seed, "sizing-taught-panel")
    items = []
    for f in (0, 1):
        cand = [p for p in split["families"][f]["taught"] if not (f == 0 and p[1] in world["changed"])]
        chosen = sorted(r.sample(cand, N_PER_FAMILY))
        for (x, c) in chosen:
            truth = G.true_outcome(cfg, world, f, x, c, 0)
            assert truth == G.true_outcome(cfg, world, f, x, c, 1)
            for kind, (lo, hi) in (("long", LONG), ("short", SHORT)):
                fl = bytes(r.choice(G.FILL) for _ in range(r.randint(lo, hi)))
                prompt = bytes([G.ctx_byte(x)]) + fl + bytes([G.cue_byte(cfg, f, c)]) + b"="
                items.append(dict(f=f, x=x, c=c, filler=kind, prompt=prompt.decode(), truth=truth))
    return items, sorted(map(tuple, split["families"][0]["taught"])), sorted(map(tuple, split["families"][1]["taught"]))


def checks(items, cfg):
    W = cfg["window"]; bad = []
    for it in items:
        p = it["prompt"].encode()
        if it["filler"] == "long" and any(65 <= b < 65 + cfg["contexts"] for b in p[-W:]):
            bad.append(("context inside the last ten bytes", it))
        if it["filler"] == "short" and len(p) > 5:
            bad.append(("short filler too long", it))
    per = {(f, k): sum(1 for it in items if it["f"] == f and it["filler"] == k) for f in (0, 1) for k in ("long", "short")}
    return dict(problems=bad[:3], n_problems=len(bad), counts={f"family{f + 1}/{k}": v for (f, k), v in per.items()})


def load(seed, modulus):
    d = json.loads(OUT.read_text())
    e = d["panels"][f"{seed}/{modulus}"]
    return [dict(it, prompt=it["prompt"].encode()) for it in e["items"]], e["sha256"]


def main():
    out = dict(brief="OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3", n_per_family=N_PER_FAMILY, long_filler=LONG, short_filler=SHORT,
               note="Frozen before any fit. World 104 is for stage-1 profiling only; it is in no roster.", panels={})
    ok = True
    for seed in CF.ROSTER_WORLDS + (CF.PROFILE_WORLD,):
        for M in CF.MODULI:
            sets = {}
            for P in CF.PRESENTATIONS:
                items, t0, t1 = build_panel(CF.make(M, P), seed)
                sets[P] = (G.sha([[it["prompt"], it["truth"]] for it in items]), G.sha([t0, t1]))
            same = len(set(sets.values())) == 1
            items, _, _ = build_panel(CF.make(M, 4), seed)
            ck = checks(items, CF.make(M, 4))
            ok &= same and ck["n_problems"] == 0
            out["panels"][f"{seed}/{M}"] = dict(world=seed, modulus=M, sha256=sets[4][0], taught_sets_sha256=sets[4][1],
                                                 identical_at_4_32_256_presentations=same, checks=ck, items=items)
            print(seed, M, sets[4][0][:12], "same across presentations:", same, "problems:", ck["n_problems"])
    out["passed"] = ok
    OUT.write_text(json.dumps(out, indent=0))
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["passed"] else 1)
