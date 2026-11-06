"""Raw byte rendering, fixed teaching schedules and evaluator-only qualification."""
from collections import Counter
from itertools import combinations
from random import Random

import numpy as np

from qualify import ALPHABET, TEACH, Encoder, Record, seed
from benchmark import Query, expected, reachable, routes, structure_key

BLOCKS = ("ordered-dev-68071", "ordered-dev-80413")
FILL = ((4, 5), (5, 6), (6, 7), (4, 7), (4, 6), (5, 7))
GRAPHS = (
    ((0, 1), (0, 2), (1, 4), (1, 5), (2, 4), (2, 5), (4, 3), (5, 3)),
    ((0, 1), (0, 2), (1, 2), (1, 4), (2, 5), (4, 5), (4, 3), (5, 3)),
)
MASK_PAIRS = (
    ((90, 102), (89, 101), (101, 102), (89, 153), (73, 97), (93, 117), (154, 170), (141, 165),
     (148, 176), (82, 133), (90, 149), (102, 106), (137, 146), (197, 202), (109, 186), (54, 57)),
    ((101, 149), (101, 169), (89, 101), (86, 149), (89, 105), (206, 218), (146, 162), (28, 44),
     (180, 184), (173, 182), (53, 54), (141, 154), (77, 166), (78, 154), (91, 167), (90, 101)),
)
DESIGNATED = (69, 73)
LENGTHS = (10000, 100000, 10000, 10000)
TARGET_SYMBOLS = 55


def render(graph, mask, alphabet, actors, places, order):
    state = [v for a, p in zip(actors, places) for v in (ord("O"), a, p)]
    links = [v for i in order for v in (
        alphabet[0 if mask >> i & 1 else 1], actors[graph[i][0]], actors[graph[i][1]])]
    return bytes((*state, *links, ord("M"), actors[0], places[8], ord("Q"), actors[3]))


def parse(prefix):
    assert len(prefix) == 53 and prefix[:24:3] == b"O" * 8 and prefix[-5] == 77 and prefix[-2] == 81
    locs = dict(zip(prefix[1:24:3], prefix[2:24:3]))
    links = tuple(zip(prefix[24:-5:3], prefix[25:-5:3], prefix[26:-5:3]))
    mover, destination, target = prefix[-4], prefix[-3], prefix[-1]
    assert len(locs) == 8 and mover != target and locs[target] != destination
    assert len({(a, b) for _, a, b in links}) == 8 and all(a in locs and b in locs for _, a, b in links)
    return Query(locs, links, mover, destination, target)


def truth(graph):
    def paths(node, mask):
        if node == 3:
            yield mask
        else:
            for i, (a, b) in enumerate(graph):
                if a == node:
                    yield from paths(b, mask | 1 << i)
    path_masks = tuple(paths(0, 0))
    return tuple(int(any(m & path == path for path in path_masks)) for m in range(256))


def alphabet_for(phase):
    return b"UV" if phase == "B" else ALPHABET


def material(block, phase):
    """32 fixed environments, two byte relabelings, all 256 edge-opcode assignments."""
    namespace = 160 if phase == "B" else 128
    alphabet = alphabet_for(phase)
    rng = Random(seed(block, phase, "32-schedule"))
    bank = []
    for i in range(32):
        graph, paths = TEACH[i % len(TEACH)]
        full = graph + FILL[:8 - len(graph)]
        actors = rng.sample(range(namespace, namespace + 8), 8)
        places = rng.sample(range(48, 58), 10)
        order = rng.sample(range(8), 8)
        # Vary distractors and all relevant assignments without expanding the fixed
        # 32-instance semantic roster or handing phase information to the learner.
        masks = tuple(m for m in range(256) if ((m >> 6) ^ (m >> 4)) & 3 == i % 4)
        for delta in (0, 3):
            locs = [(p - 48 + delta) % 10 + 48 for p in places]
            for m in masks:
                prefix = render(full, m, alphabet, actors, locs, order)
                ys = tuple(locs[8] if any(all((m >> j & 1) == (law == 0) for j in path)
                                          for path in paths) else locs[3] for law in range(2))
                bank.append((prefix, ys, locs[3], locs[8]))
    return bank


def stream(block, phase, bank):
    # Every epoch uses every registered rendering once in a reproducibly shuffled order.
    order = list(range(len(bank)))
    rng = Random(seed(block, phase, "permutation"))
    while True:
        rng.shuffle(order)
        yield from order


def held(block, phase, law):
    rng = Random(seed(block, phase, "held-v2"))
    alphabet = alphabet_for(phase)
    namespace = 208 if phase == "B" else 192
    pairs = []
    for graph, masks in zip(GRAPHS, MASK_PAIRS):
        moves = truth(graph)
        for a, b in masks:
            edited = a ^ b
            actors = rng.sample(range(namespace, namespace + 8), 8)
            places = rng.sample(range(48, 58), 10)
            order = rng.sample(range(8), 8)
            final = rng.choice([i for i in order if not edited >> i & 1])
            order.remove(final)
            order.append(final)
            pair = tuple(Record(render(graph, m, alphabet, actors, places, order),
                                places[8 if moves[m ^ (255 * law)] else 3]) for m in (a, b))
            pairs.append(pair[::-1] if rng.randrange(2) else pair)
    return pairs


def masked_baseline(graph, masks, k):
    """Oracle roles. Each law may choose its own role subset and Boolean table."""
    moves = truth(graph)
    masks = np.array([(a, b, a ^ 255, b ^ 255) for a, b in masks])
    labels = np.array(moves)[masks]
    by_law = [set(), set()]
    shared = 0
    weights = 1 << np.arange(len(masks))
    for ids in combinations(range(8), k):
        role_codes = sum((masks >> i & 1) << j for j, i in enumerate(ids))
        table = np.arange(1 << (1 << k))[:, None, None]
        answers = (table >> role_codes & 1) == labels
        shared = max(shared, int(answers.all(axis=2).sum(axis=1).max()))
        for law in range(2):
            fits = answers[:, :, law*2:law*2+2].all(axis=2)
            by_law[law].update((fits * weights).sum(axis=1).tolist())
    independent = max(bin(a & b).count("1") for a in by_law[0] for b in by_law[1])
    return {"one_table_both_laws": shared, "separate_edge_subsets_and_tables_by_law": independent}


def certify():
    baselines = {}
    for family, (graph, masks, main) in enumerate(zip(GRAPHS, MASK_PAIRS, DESIGNATED)):
        gold = truth(graph)
        outcomes = [tuple(gold[m] for m in (a,b,a^255,b^255)) for a,b in masks]
        assert all(p in ((1,0,0,1),(0,1,1,0)) for p in outcomes[:4])
        assert all(a & main == b & main for a, b in masks[4:8])
        assert all(a & main != b & main for a, b in masks[8:])
        assert all((p[0]==p[1]) != (p[2]==p[3]) for p in outcomes[4:])
        def parity(mask):
            return bin(mask & main).count("1") % 2
        parity_hits = sum(tuple(parity(m) for m in (a,b,a^255,b^255)) == p
                          for (a,b),p in zip(masks,outcomes))
        baselines[str(family)] = {"parity": parity_hits, "oracle_roles": {
            str(k): masked_baseline(graph, masks, k) for k in range(1,4)}}

    encoder = Encoder("bag", None)
    ordered = Encoder("ordered", None)
    rows = {}
    for block in BLOCKS:
        block_audits = {}
        for phase in ("A", "B", "A_prime"):
            bank = material(block, phase)
            assert len({p for p, *_ in bank}) == len(bank) == 4096
            parsed = [(parse(p), ys) for p, ys, *_ in bank]
            assert all(ys[law] == expected(q, alphabet_for(phase)[law])
                       for q, ys in parsed for law in range(2))
            teach_keys = {structure_key(q) for q, _ in parsed}
            assert teach_keys == {(2,), (3,), (1,2)}
            for law in range(2):
                assert [op for op in alphabet_for(phase) if all(expected(q, op) == ys[law]
                                                               for q, ys in parsed)] == [alphabet_for(phase)[law]]
                for cutoff in (1, 2):
                    assert any((q.destination if any(len(r)<=cutoff and
                               all(op == alphabet_for(phase)[law] for op in r) for r in routes(q))
                               else q.stay) != ys[law] for q, ys in parsed)
            block_audits[phase] = {"feedback_renderings": len(bank), "path_lengths": sorted(teach_keys)}
        for phase in ("A", "B"):
            x, y = (held(block, phase, law) for law in range(2))
            assert len({r.prefix for p in x for r in p}) == 64
            for i, (p, q) in enumerate(zip(x,y)):
                a,b = (r.prefix for r in p)
                assert (a,b) == tuple(r.prefix for r in q)
                assert Counter(a) == Counter(b) and a[-8:] == b[-8:]
                assert np.array_equal(encoder.codes(a)[-1], encoder.codes(b)[-1])
                assert not np.array_equal(ordered.codes(a)[-1], ordered.codes(b)[-1])
                def endpoint_counts(prefix):
                    z = parse(prefix)
                    return (Counter(op for op,s,t in z.edges if s==z.mover),
                            Counter(op for op,s,t in z.edges if t==z.target))
                assert endpoint_counts(a) == endpoint_counts(b)
                assert p[0].target != p[1].target or q[0].target != q[1].target
                assert any(a.target != b.target for a,b in zip(p,q))
                for law, pair in enumerate((p,q)):
                    for row in pair:
                        z = parse(row.prefix)
                        assert structure_key(z) == ((3,3,3,3) if i<16 else (3,3,4,4))
                        assert row.target == expected(z, alphabet_for(phase)[law])
        rows[block] = block_audits
    maximum = sum(r["oracle_roles"]["3"]["separate_edge_subsets_and_tables_by_law"]
                  for r in baselines.values())
    assert maximum <= 24 and maximum + 4 <= 32
    return {"status":"PANEL_QUALIFIED","qualification":rows,"family_shortcuts":baselines,
            "graph_oracle_quartets":32, "registered_shortcut_ceiling":maximum,
            "exact_and_canonical_lookup_quartets":0, "same_counts_or_recent_only_ceiling":0,
            "required_ordered_joint_at_least":max(8,maximum+4)}
