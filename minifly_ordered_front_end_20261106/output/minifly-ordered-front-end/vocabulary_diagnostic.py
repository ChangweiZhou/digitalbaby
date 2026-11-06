"""Read-only supplementary audit of topology versus actor vocabulary."""
import argparse
from pathlib import Path
from random import Random

from campaign import ROOT, RUN, builder, load, read, sources, write
from qualify import ALPHABET, ARMS, Backend, Record, TEACH, score, seed, state_hash
from revised_panel import BLOCKS, FILL, held, material, render
from summarize_campaign import quartet, summary


def translate(pairs, source, target):
    table = bytes.maketrans(bytes(range(source, source + 8)), bytes(range(target, target + 8)))
    return [tuple(Record(r.prefix.translate(table), r.target) for r in pair) for pair in pairs]


def teaching_render(block, law):
    rng = Random(seed(block, "fresh-render-of-taught-topologies"))
    feedback = {p for phase in ("A","B","A_prime") for p,*_ in material(block,phase)}
    rows = []
    for i in range(64):
        graph, paths = TEACH[i % len(TEACH)]
        graph = graph + FILL[:8-len(graph)]
        mask = rng.randrange(256)
        actors = rng.sample(range(128,136),8)
        places = rng.sample(range(48,58),10)
        order = rng.sample(range(8),8)
        prefix = render(graph, mask, ALPHABET, actors, places, order)
        assert prefix not in feedback
        moves = any(all((mask >> j & 1) == (law == 0) for j in path) for path in paths)
        rows.append(Record(prefix, places[8 if moves else 3]))
    assert len({r.prefix for r in rows}) == 64
    return list(zip(rows[::2], rows[1::2]))


def run(archive, arm_filter=None):
    lock = read(RUN / "LOCK.json")
    assert lock["sources"] == sources(archive)
    backend = Backend(archive)
    result = {}
    for block in BLOCKS:
        result[block] = {}
        for arm in ARMS if arm_filter is None else (arm_filter,):
            directory = RUN / f"{block}-{arm}"
            path = directory / "A_100k.pt"
            receipt = directory / "A_100k.json"
            if not receipt.exists():
                continue
            _, make = builder(backend, arm, block)
            lives = [make(), make()]
            load(backend, lives, path)
            assert [state_hash(x) for x in lives] == read(receipt)["state_hashes"]
            tests = []
            for i, life in enumerate(lives):
                novel = held(block, "A", i)
                familiar = teaching_render(block, i)
                cases = {
                    "taught_topology_familiar_actor_bytes": familiar,
                    "taught_topology_fresh_actor_bytes": translate(familiar, 128, 192),
                    "held_topology_familiar_actor_bytes": translate(novel, 192, 128),
                    "held_topology_fresh_actor_bytes": novel,
                }
                tests.append({name: score(life,pairs) for name,pairs in cases.items()})
                assert tests[-1]["held_topology_fresh_actor_bytes"] == read(receipt)["panels"][i][f"held_A_law{i}"]
            joined = {name:quartet(tests[0][name]["outputs"],tests[1][name]["outputs"])
                      for name in tests[0] if name.startswith("held_")}
            result[block][arm] = {"joint_by_held_panel":joined,
                                  "metrics":[{name:summary(rows) for name,rows in t.items()} for t in tests],
                                  "raw":tests}
    name = "VOCABULARY_DIAGNOSTIC" + (f"_{arm_filter}" if arm_filter else "") + ".json"
    write(RUN / name, {"status":"SUPPLEMENTARY_READ_ONLY","by_block":result})
    for block, arms in result.items():
        for arm, info in arms.items():
            print(block, arm, info["joint_by_held_panel"],info["metrics"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    ap.add_argument("--arm",choices=ARMS)
    args = ap.parse_args()
    run(args.archive,args.arm)
