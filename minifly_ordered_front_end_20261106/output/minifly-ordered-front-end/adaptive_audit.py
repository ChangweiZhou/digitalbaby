"""Supplementary frozen-checkpoint audit using adaptive link queries and all masks."""
import argparse
from functools import lru_cache
from pathlib import Path
from random import Random

import numpy as np

from campaign import RUN, builder, load, read, sources, write
from qualify import ALPHABET, ARMS, Backend, Record, score, seed, state_hash
from revised_panel import BLOCKS, GRAPHS, MASK_PAIRS, render, truth


def adaptive_quartet_ceiling(graph, pairs):
    gold = truth(graph)
    families = []
    for law in range(2):
        vals = [m ^ (255 * law) for p in pairs for m in p]
        actual = sum(gold[m] << i for i, m in enumerate(vals))
        edges = tuple(sum((m >> e & 1) << i for i, m in enumerate(vals)) for e in range(8))

        @lru_cache(None)
        def visit(rows, depth):
            fits = {rows & actual, rows & ~actual}
            if depth:
                for col in edges:
                    left, right = rows & col, rows & ~col
                    if left and right:
                        fits.update(a | b for a in visit(left, depth - 1)
                                    for b in visit(right, depth - 1))
            return fits

        feasible = {sum((m >> i & 3 == 3) << (i // 2) for i in range(0, len(vals), 2))
                    for m in visit((1 << len(vals)) - 1, 3)}
        covered = np.zeros(1 << len(pairs), dtype=bool)
        covered[list(feasible)] = True
        for i in range(len(pairs)):
            width = 1 << i
            view = covered.reshape(-1, 2, width)
            view[:, 0, :] |= view[:, 1, :]
        families.append(covered)
    return max(bin(int(m)).count("1") for m in np.flatnonzero(families[0] & families[1]))


def balanced_accuracy_ceiling(graph):
    actual = sum(y << i for i, y in enumerate(truth(graph)))
    allrows = (1 << 256) - 1
    edges = [sum((m >> e & 1) << m for m in range(256)) for e in range(8)]
    counts = lambda bits: bin(bits).count("1")
    n1 = counts(actual)
    n0 = 256 - n1

    @lru_cache(None)
    def visit(rows, depth):
        best = max(counts(rows & actual) * n0, counts(rows & ~actual) * n1)
        if depth:
            for col in edges:
                a, b = rows & col, rows & ~col
                if a and b:
                    best = max(best, visit(a, depth-1) + visit(b, depth-1))
        return best

    return {str(d):visit(allrows, d) / (2 * n1 * n0) for d in range(4)}


def all_masks(block, family, law):
    graph = GRAPHS[family]
    gold = truth(graph)
    rng = Random(seed(block, family, "all-mask-held"))
    rows = []
    labels = []
    for mask in range(256):
        actors = rng.sample(range(192,200),8)
        places = rng.sample(range(48,58),10)
        order = rng.sample(range(8),8)
        prefix = render(graph,mask,ALPHABET,actors,places,order)
        moved = gold[mask ^ (255 * law)]
        rows.append(Record(prefix,places[8 if moved else 3]))
        labels.append(moved)
    return list(zip(rows[::2], rows[1::2])), labels


def accuracy(panel, labels):
    rows = panel["outputs"]
    correct = [[0,0],[0,0]]
    for y,row in zip(labels, rows):
        correct[y][0] += row["output"] == row["target"]
        correct[y][1] += row["output"][0] == row["target"][0]
    denom = [labels.count(i) for i in range(2)]
    return {"stay_correct_exact_and_first":correct[0], "move_correct_exact_and_first":correct[1],
            "stay_move_denominators":denom,
            "balanced_exact":sum(correct[i][0]/denom[i] for i in range(2))/2,
            "balanced_first":sum(correct[i][1]/denom[i] for i in range(2))/2}


def audit():
    caps = [adaptive_quartet_ceiling(g,p) for g,p in zip(GRAPHS,MASK_PAIRS)]
    result = {"status":"SUPPLEMENTAL_AUDIT", "three_adaptive_queries_quartets_by_family":caps,
              "three_adaptive_queries_quartets_total":sum(caps),
              "all_mask_adaptive_balanced_accuracy_ceiling_by_family":[balanced_accuracy_ceiling(g) for g in GRAPHS]}
    write(RUN / "ADAPTIVE_AUDIT.json",result)
    print(result, flush=True)


def model_scores(archive):
    lock = read(RUN / "LOCK.json")
    assert lock["sources"] == sources(archive)
    backend = Backend(archive)
    result = {}
    for block in BLOCKS:
        result[block] = {}
        for arm in ARMS:
            directory = RUN / f"{block}-{arm}"
            life_file = directory / "A_100k.pt"
            receipt_file = directory / "A_100k.json"
            assert receipt_file.exists()
            _, make = builder(backend, arm, block)
            lives = [make(), make()]
            load(backend, lives, life_file)
            assert [state_hash(x) for x in lives] == read(receipt_file)["state_hashes"]
            result[block][arm] = {}
            for family in range(len(GRAPHS)):
                records = []
                for law, life in enumerate(lives):
                    pairs, labels = all_masks(block,family,law)
                    panel = score(life,pairs)
                    records.append({"metrics":accuracy(panel,labels), "outputs":panel["outputs"]})
                result[block][arm][str(family)] = records
            print(block, arm, [[round(x["metrics"]["balanced_exact"],4) for x in rows]
                               for rows in result[block][arm].values()], flush=True)
    write(RUN / "ALL_MASK_MODEL_SCORES.json", {"status":"SUPPLEMENTARY_READ_ONLY","blocks":result})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path)
    a = ap.parse_args()
    audit() if a.archive is None else model_scores(a.archive)
