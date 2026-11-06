"""Evaluator-side graph checks and shortcut controls. Keep out of learner inputs."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from task import ALPHABETS, active_opcode, probes, training


@dataclass(frozen=True)
class Query:
    locations: dict
    edges: tuple
    mover: int
    destination: int
    target: int

    @property
    def stay(self):
        return self.locations[self.target]


def parse(prefix):
    assert len(prefix) == 47 and prefix[:24:3] == b"O" * 8
    assert prefix[42] == ord("M") and prefix[45] == ord("Q")
    locations = dict(zip(prefix[1:24:3], prefix[2:24:3]))
    edges = tuple(tuple(prefix[i:i + 3]) for i in range(24, 42, 3))
    mover, destination, target = prefix[43], prefix[44], prefix[46]
    assert len(locations) == 8 and mover != target and destination != locations[target]
    assert len({(a, b) for _, a, b in edges}) == 6
    assert all(a in locations and b in locations for _, a, b in edges)
    return Query(locations, edges, mover, destination, target)


def routes(query):
    def visit(node, seen, opcodes):
        if node == query.target:
            yield opcodes
            return
        for op, source, target in query.edges:
            if node == source and target not in seen:
                yield from visit(target, seen | {target}, (*opcodes, op))
    return tuple(visit(query.mover, {query.mover}, ()))


def reachable(query, active):
    reached = {query.mover}
    while True:
        expanded = reached | {b for op, a, b in query.edges if op == active and a in reached}
        if expanded == reached:
            return query.target in reached
        reached = expanded


def expected(query, active):
    return query.destination if reachable(query, active) else query.stay


def infer(records, alphabet):
    witnesses = [(parse(row.prefix), row.target) for row in records]
    choices = [op for op in alphabet if all(expected(q, op) == y for q, y in witnesses)]
    assert len(choices) == 1
    active, = choices
    alternatives = {
        "direct_only": lambda rs: (active,) in rs,
        "all_routes_required": lambda rs: all(all(op == active for op in r) for r in rs),
        "inactive_carries_move": lambda rs: bool(rs),
    }
    for rule in alternatives.values():
        assert any((q.destination if rule(routes(q)) else q.stay) != y for q, y in witnesses)
    return active


def structure_key(query):
    # A coarser exclusion than canonical isomorphism. No taught query has two length-two routes.
    return tuple(sorted(map(len, routes(query))))


def majority(counter):
    return max(counter, key=lambda label: (counter[label], -label)) if counter else None


def simple_features(prefix):
    q = parse(prefix)
    away = [op for op, a, _ in q.edges if a == q.mover]
    into = [op for op, _, b in q.edges if b == q.target]
    return {
        "frequency": (),
        "normalized": (q.locations[q.mover], q.stay, q.destination, q.edges[0][0]),
        "byte_bag": tuple(sorted(Counter(prefix).items())),
        "opcode_counts": tuple(sorted(Counter(op for op, _, _ in q.edges).items())),
        "root_and_target_opcode_counts": (tuple(sorted(away)), tuple(sorted(into))),
        "first_outgoing": away[0],
        "last_outgoing": away[-1],
        "first_incoming": into[0],
        "last_incoming": into[-1],
        **{f"position_{i}": prefix[i] for i in range(24, 42, 3)},
    }


def shortcut_scores(records, pairs, active):
    exact = {row.prefix: row.target for row in records}
    tables = defaultdict(lambda: defaultdict(Counter))
    for row in records:
        q = parse(row.prefix)
        outcome = row.target == q.destination
        for name, key in simple_features(row.prefix).items():
            tables[name][key][int(outcome)] += 1
    totals = Counter()
    for pair in pairs:
        decisions = []
        for row in pair:
            q = parse(row.prefix)
            predicted = {
                "exact": exact.get(row.prefix),
                "recent_move": q.destination,
                "stay": q.stay,
            }
            for name, key in simple_features(row.prefix).items():
                choice = majority(tables[name].get(key, Counter()))
                predicted[name] = None if choice is None else q.destination if choice else q.stay
            for name, key in (("first_out", 0), ("last_out", -1)):
                code = [op for op, a, _ in q.edges if a == q.mover][key]
                predicted["oracle_opcode_" + name] = q.destination if code == active else q.stay
            decisions.append({name: output == row.target for name, output in predicted.items()})
        for name, hit in decisions[0].items():
            totals[name] += hit and decisions[1][name]
    return dict(totals)


def benchmark_family(world):
    roster = {diversity: list(training(world, "A", diversity, 512)) for diversity in (4, 32)}
    a, b = roster.values()
    assert a[:64] == list(training(world, "A", 4, 64))
    assert b[:64] == list(training(world, "A", 32, 64))
    assert a != b and all(len(row.observed) + 1 == 49 for row in (*a, *b))
    assert [r.target == parse(r.prefix).destination for r in a] == [
        r.target == parse(r.prefix).destination for r in b
    ]
    for records in roster.values():
        counts = [Counter(r.prefix[i] for r in records) for i in range(1, 24, 3)]
        assert all(count == counts[0] for count in counts)
    held = {kind: probes(world, "A", kind) for kind in ("structure", "invariant", "polarity")}
    active = active_opcode(world, "A")
    reverse_active = active_opcode(world, "A_prime")
    assert all(infer(rows, ALPHABETS["A"]) == active for rows in roster.values())
    exposed = {row.prefix for rows in roster.values() for row in rows}
    for kind, pairs in held.items():
        assert len(pairs) == len({frozenset(row.prefix for row in pair) for pair in pairs}) == 32
        for x, y in pairs:
            assert x.prefix not in exposed and y.prefix not in exposed
            assert all(structure_key(parse(row.prefix)) == (2, 2) for row in (x, y))
            assert all(expected(parse(row.prefix), active) == row.target for row in (x, y))
            assert (x.target == y.target) == (kind == "invariant")
            if kind != "polarity":
                assert Counter(x.prefix) == Counter(y.prefix)
    for rows in roster.values():
        assert all(structure_key(parse(row.prefix)) in ((2,), (1, 2)) for row in rows)
    assert all(expected(parse(row.prefix), reverse_active) == row.target
               for pair in held["structure"] for row in pair)
    reversal = probes(world, "A_prime", "polarity")
    for old, revised in zip(held["polarity"], reversal):
        assert all(x.prefix == y.prefix and x.target != y.target for x, y in zip(old, revised))
        assert all(expected(parse(row.prefix), reverse_active) == row.target for row in revised)
    unedited = list(training(world, "A_prime", 32, 32, edit=False))
    edited = list(training(world, "A_prime", 32, 32))
    assert [row.prefix for row in unedited] == [row.prefix for row in edited]
    assert infer(edited, ALPHABETS["A"]) == reverse_active
    assert infer(unedited, ALPHABETS["A"]) == active
    shortcuts = {str(n): shortcut_scores(rows, held["structure"], active) for n, rows in roster.items()}
    assert all(max(scores.values()) <= 8 for scores in shortcuts.values()), shortcuts
    return {
        "world": world, "structure": 32, "invariant": 32, "polarity": 32,
        "structural_pairs_unchanged_after_rule_reversal": 32,
        "polarity_answers_changed_at_identical_prefix": 64,
        "strongest_surface_shortcuts": shortcuts,
        "unique_training_records": {str(n): len({r.observed for r in rows}) for n, rows in roster.items()},
    }
