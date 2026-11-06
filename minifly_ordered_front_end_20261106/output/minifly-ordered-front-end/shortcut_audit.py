"""Adversarial evaluator-only checks for two held graph panels."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random

from qualify import ALPHABET, graph_fixture, parse, relabel, reachable

CANDIDATE = ((0, 1), (0, 2), (1, 4), (1, 5), (2, 4), (2, 5), (4, 3), (5, 3))
MASK_PAIRS = tuple((a, b) for a in (90, 101, 153, 166) for b in (165, 154, 102, 89) if a ^ b != 255)


def route_edges(query):
    def visit(node, seen, path):
        if node == query.target:
            yield path
            return
        for op, src, dst in query.edges:
            if src == node and dst not in seen:
                yield from visit(dst, seen | {dst}, (*path, (op, src, dst)))
    return tuple(visit(query.mover, {query.mover}, ()))


def one_link_output(row, active):
    query = parse(row.prefix)
    paths = sorted(route_edges(query), key=len)
    assert len(paths[0]) < len(paths[1])
    shared = set(paths[0]).intersection(e for path in paths[1:] for e in path)
    assert len(shared) == 1
    op, _, _ = shared.pop()
    return query.destination if op == active else query.stay


def candidate_checks():
    def prompt(mask, actors, locs, order):
        state = [v for who, where in zip(actors, locs) for v in (ord("O"), who, where)]
        links = [v for i in order for v in (
            ALPHABET[0] if mask >> i & 1 else ALPHABET[1],
            actors[CANDIDATE[i][0]], actors[CANDIDATE[i][1]])]
        return bytes((*state, *links, ord("M"), actors[0], locs[8], ord("Q"), actors[3]))

    def solve(prefix, active):
        reached = {prefix[-4]}
        links = tuple(zip(prefix[24:-5:3], prefix[25:-5:3], prefix[26:-5:3]))
        for _ in range(8):
            reached |= {b for op, a, b in links if op == active and a in reached}
        return prefix[-1] in reached

    rng = Random(20261106)
    for a, b in MASK_PAIRS:
        order = list(range(len(CANDIDATE)))
        rng.shuffle(order)
        edited = a ^ b
        unchanged = next(i for i in order if not edited >> i & 1)
        order.remove(unchanged)
        order.append(unchanged)
        actors = rng.sample(range(192, 200), 8)
        locs = rng.sample(range(48, 58), 10)
        left, right = (prompt(m, actors, locs, order) for m in (a, b))
        assert len(left) == len(right) == 53
        assert Counter(left) == Counter(right) and left[-8:] == right[-8:]
        for end, node in ((0, 0), (1, 3)):
            incident = [i for i, e in enumerate(CANDIDATE) if e[end] == node]
            assert sum(a >> i & 1 for i in incident) == sum(b >> i & 1 for i in incident)
        assert [solve(p, op) for op in ALPHABET for p in (left, right)] == [True, False, False, True]

    def strongest(k):
        best = (-1, (), ())
        for edges in combinations(range(len(CANDIDATE)), k):
            def sig(mask):
                return sum((mask >> i & 1) << j for j, i in enumerate(edges))
            patterns = [(sig(a), sig(b), sig(a ^ 255), sig(b ^ 255)) for a, b in MASK_PAIRS]
            for table in range(1 << (1 << k)):
                hits = sum(all((table >> x & 1) == y for x, y in zip(p, (1, 0, 0, 1)))
                           for p in patterns)
                if hits > best[0]:
                    best = (hits, edges, tuple(i for i in range(1 << k) if table >> i & 1))
        return {"quartets": best[0], "edge_indices": best[1], "move_signatures": best[2]}

    # An all-mask diagnostic exposes shortcuts induced by selecting only flip pairs.
    actors, locs, order = list(range(192, 200)), list(range(48, 58)), list(range(8))
    truth = [int(solve(prompt(m, actors, locs, order), ALPHABET[0])) for m in range(256)]
    best_accuracy = 0
    for edges in combinations(range(8), 3):
        counts = [[0, 0] for _ in range(8)]
        for mask, label in enumerate(truth):
            sig = sum((mask >> i & 1) << j for j, i in enumerate(edges))
            counts[sig][label] += 1
        best_accuracy = max(best_accuracy, sum(max(c) for c in counts))
    three = strongest(3)
    parity_accuracy = sum(sum(m >> i & 1 for i in three["edge_indices"]) % 2 == label
                          for m, label in enumerate(truth))
    assert (three["quartets"], best_accuracy) == (12, 199)
    return {"scope": "single new motif, evaluator only", "pairs": len(MASK_PAIRS),
            "rooted_edges": CANDIDATE, "one_link": strongest(1),
            "two_link": strongest(2), "three_link": three,
            "all_mask_cases": len(truth), "all_mask_move_cases": sum(truth),
            "all_mask_best_three_link_accuracy_count": best_accuracy,
            "all_mask_pair_perfect_parity_accuracy_count": parity_accuracy}


def main():
    hits = []
    for block in range(8):
        held = graph_fixture(block)
        for law in range(2):
            for pair in relabel(held, law):
                for row in pair:
                    assert (parse(row.prefix).destination if reachable(parse(row.prefix), ALPHABET[law])
                            else parse(row.prefix).stay) == row.target
        hits.append(sum(all(one_link_output(row, ALPHABET[law]) == row.target
                            for law in range(2) for row in relabel([pair], law)[0])
                        for pair in held))
    assert hits == [32] * 8
    result = {"status": "REASONING_GATE_NOT_QUALIFIED",
              "original_one_link_quartets_by_block": hits, "original_quartets_per_block": 32,
              "alternate_panel": candidate_checks()}
    result["sources"] = {p.name: sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), Path(__file__).with_name("qualify.py"))}
    path = Path(__file__).with_name("SHORTCUT_AUDIT.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
