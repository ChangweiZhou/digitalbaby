"""Three bounded engineering cycles for the ordered-byte proposal."""
import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations, permutations
import json
from pathlib import Path
from random import Random
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "minifly-reasoning-scaling"))
from backend import Backend, parameter_count, state_hash
from benchmark import parse, reachable, structure_key
from cycles import source_receipt, score
from task import Record, EOS, frame, seed

ARMS = ("stock", "ordered", "bag")
ALPHABET = b"XY"
EDGE = ((0, 1), (0, 2), (1, 2), (1, 4), (2, 3), (4, 3))
TRANSPOSE = tuple((3 if b == 0 else 0 if b == 3 else b,
                   3 if a == 0 else 0 if a == 3 else a) for a, b in EDGE)
TEACH = (
    (((0, 1), (1, 3)), ((0, 1),)),
    (((0, 1), (1, 2), (2, 3)), ((0, 1, 2),)),
    (((0, 1), (1, 3), (0, 3)), ((0, 1), (2,))),
)
FILL = ((4, 5), (5, 6), (6, 7), (4, 7))
D = 5177
WORLD = "front-end-engineering"


def training(phase, count, law, edit=False):
    rng = Random(seed(WORLD, phase))
    namespace = 160 if phase == "B" else 128
    alphabet = b"UV" if phase == "B" else ALPHABET
    active = alphabet[law ^ int(edit)]
    # Stratify patterns and paths so every short witness block covers both law assignments.
    for i in range(count):
        edges, paths = TEACH[i % len(TEACH)]
        full = edges + FILL[:6 - len(edges)]
        mask = (i // len(TEACH)) % (1 << len(edges))
        x_edges = {e for j, e in enumerate(edges) if (mask >> j) & 1}
        x_edges.update(e for e in full[len(edges):] if rng.randrange(2))
        actors = rng.sample(range(namespace, namespace + 8), 8)
        locations = rng.sample(range(48, 58), 10)
        order = rng.sample(range(6), 6)
        moves = any(all((edges[j] in x_edges) == (active == alphabet[0]) for j in p) for p in paths)
        yield frame(full, actors, locations, alphabet, alphabet[0], x_edges, order, moves)


def graph_fixture(block, phase="A"):
    rng = Random(seed(block, phase, "held"))
    alphabet = b"UV" if phase == "B" else ALPHABET
    namespace = 208 if phase == "B" else 192
    pairs = []
    for graph in (EDGE, TRANSPOSE):
        # Indices 4/5 are the swapped edges in both rooted motifs.
        orders = [o for o in permutations(range(6)) if o[-1] not in (4, 5)]
        for _ in range(16):
            order = rng.choice(orders)
            actors = rng.sample(range(namespace, namespace + 8), 8)
            locations = rng.sample(range(48, 58), 10)
            a = frame(graph, actors, locations, alphabet, alphabet[0],
                      {graph[j] for j in (0, 2, 4)}, order, True)
            b = frame(graph, actors, locations, alphabet, alphabet[0],
                      {graph[j] for j in (0, 2, 5)}, order, False)
            pairs.append((a, b) if rng.randrange(2) else (b, a))
    return pairs


def relabel(pairs, law):
    if law == 0:
        return pairs
    return [tuple(Record(r.prefix, parse(r.prefix).stay if r.target == parse(r.prefix).destination
                         else parse(r.prefix).destination) for r in pair) for pair in pairs]


def codebook():
    rng = Random(20261106)
    rows = []
    seen = set()
    while len(rows) < 256:
        row = tuple(sorted(rng.sample(range(80), 4)))
        if row not in seen:
            rows.append(row)
            seen.add(row)
    return np.asarray(rows, dtype=np.int64)


class Encoder:
    def __init__(self, kind, stock):
        self.kind = kind
        self.stock = stock
        self.book = codebook()

    def codes(self, record):
        if self.kind == "stock":
            return self.stock.codes(record)
        output = np.zeros((len(record) + 1, D), dtype=np.float32)
        for i in range(1, len(record) + 1):
            recent = record[max(i - 64, 0):i]
            if self.kind == "bag" and len(recent) > 8:
                recent = bytes(sorted(recent[:-8])) + recent[-8:]
            symbols = np.frombuffer(recent[::-1], dtype=np.uint8)
            cells = np.arange(len(symbols))[:, None] * 80 + self.book[symbols]
            output[i, cells] = 1
        return output


def make_life(backend, arm):
    class Life(backend.Life):
        def generate(self, prefix):
            out = []
            for _ in range(32):
                code = self.codes(prefix)[-1:]
                token = int(self.model.forward(code)[0][0].argmax())
                out.append(token)
                if token == EOS:
                    break
                prefix += bytes((token,))
            return out
    return Life(WORLD, Encoder(arm, backend.sensor))


def cycle1():
    assert all({(m[a], m[b]) for a, b in EDGE} != set(TRANSPOSE)
               for p in permutations((1, 2, 4))
               for m in [{0: 0, 3: 3, **dict(zip((1, 2, 4), p))}])
    rows = []
    for block in range(8):
        held = graph_fixture(block)
        assert len({r.prefix for p in held for r in p}) == 64
        for law in range(2):
            records = list(training("A", 48, law))
            assert all(structure_key(parse(r.prefix)) in ((2,), (3,), (1, 2)) for r in records)
            def matches(active, limit):
                from benchmark import routes
                return all((q.destination if any(len(p) <= limit and all(x == active for x in p)
                                                  for p in routes(q)) else q.stay) == r.target
                           for r in records for q in (parse(r.prefix),))
            assert [matches(op, 3) for op in ALPHABET] == [law == 0, law == 1]
            assert not any(matches(op, 2) for op in ALPHABET)
            for a, b in relabel(held, law):
                assert Counter(a.prefix) == Counter(b.prefix) and a.prefix[-8:] == b.prefix[-8:]
                assert a.target != b.target
                for row in (a, b):
                    q = parse(row.prefix)
                    assert structure_key(q) == (2, 3, 3)
                    assert row.target == (q.destination if reachable(q, ALPHABET[law]) else q.stay)
        # A single fixed edge slot, even with oracle opcode meaning, is a harder shortcut
        # than raw lookup/counts. Its per-family scores are disclosed.
        hits = {}
        for slot in range(24, 42, 3):
            hits[slot] = sum(
                all((parse(r.prefix).destination if r.prefix[slot] == ALPHABET[law]
                     else parse(r.prefix).stay) == r.target
                    for law in range(2) for r in relabel([pair], law)[0])
                for pair in held
            )
        rows.append({"block": block, "verified_quartets": len(held), "oracle_single_slot_quartets": hits})
    return {"status": "ENGINEERING_PASS", "blocks": rows, "lookup_orbit_overlap": 0,
            "scope": "Structural and literal pair qualification on engineering fixtures"}


def cycle2(backend):
    T = backend.T
    stock = backend.birth(WORLD, 64)
    candidate = make_life(backend, "stock")
    records = list(training("A", 24, 0))
    for row in records:
        stock.train(row.observed)
        candidate.train(row.observed)
        assert state_hash(stock) == state_hash(candidate)
    held = graph_fixture("sensor")
    ordered = Encoder("ordered", backend.sensor)
    bag = Encoder("bag", backend.sensor)
    for pair in held:
        a, b = (r.prefix for r in pair)
        assert stock.generate(a) == candidate.generate(a)
        assert not np.array_equal(ordered.codes(a)[-1], ordered.codes(b)[-1])
        assert np.array_equal(bag.codes(a)[-1], bag.codes(b)[-1])
        assert np.array_equal(ordered.codes(a).sum(axis=1), bag.codes(a).sum(axis=1))
    learned = {}
    for arm in ARMS[1:]:
        life = make_life(backend, arm)
        before = state_hash(life)
        for row in records:
            changed = row.prefix + bytes((48 if row.target != 48 else 49,))
            assert T.equal(life.codes(row.observed)[:-1], life.codes(changed)[:-1])
            life.train(row.observed)
        assert before != state_hash(life) and parameter_count(life) == 37313
        learned[arm] = life.events
    return {"status": "ENGINEERING_PASS", "stock_update_parity": 24, "stock_native_output_parity": 32,
            "both_codebooks_match_active_counts": True, "paired_ordered_distinct_bag_identical": 32,
            "local_updates": learned, "trainable_scalars_per_arm": 37313}


def train(life, records):
    for row in records:
        life.train(row.observed)


def cycle3(backend, checkpoint):
    begin = time.process_time()
    held = graph_fixture("rehearsal")
    results = {}
    for arm in ARMS:
        results[arm] = []
        for law in range(2):
            life = make_life(backend, arm)
            a_rows = list(training("A", 1024, law))
            held_law = relabel(held, law)
            early = score(life, held_law)
            train(life, a_rows[:64])
            early64 = score(life, held_law)
            train(life, a_rows[64:])
            a = score(life, held_law)
            taught = score(life, list(zip(a_rows[-32::2], a_rows[-31::2])))
            train(life, training("B", 64, law))
            b_panel = relabel(graph_fixture("rehearsal", "B"), law)
            b = score(life, b_panel)
            noedit = make_life(backend, arm)
            noedit.model.restore(life.model.state())
            noedit.events = life.events
            assert state_hash(noedit) == state_hash(life)
            edited = list(training("A_prime", 64, law, edit=True))
            unchanged = list(training("A_prime", 64, law))
            assert [r.prefix for r in edited] == [r.prefix for r in unchanged]
            assert any(x.target != y.target for x, y in zip(edited, unchanged))
            if arm == "ordered" and law == 1:
                backend.checkpoint(life, checkpoint, WORLD, 64)
                train(life, edited[:1])
                restart = {"expected": state_hash(life), "checkpoint_sha256": digest(checkpoint),
                           "record_hex": edited[0].observed.hex(), "checkpoint_bytes": checkpoint.stat().st_size}
                train(life, edited[1:])
            else:
                train(life, edited)
            train(noedit, unchanged)
            assert state_hash(life) != state_hash(noedit)
            results[arm].append({"law": law, "cold": early, "A_64": early64, "A": a, "taught_A": taught,
                "B": b, "B_after_edit": score(life, b_panel),
                "A_prime": score(life, relabel(held, 1-law)), "noedit": score(noedit, held_law)})
    quartet = {}
    for arm, (x, y) in results.items():
        quartet[arm] = sum(all(r["output"] == r["target"] for r in
                          (*x["A"]["outputs"][i:i+2], *y["A"]["outputs"][i:i+2]))
                          for i in range(0, 64, 2))
    assert quartet["bag"] == 0
    return {"status": "RESTART_PENDING", "one_paired_engineering_block": results, "joint_A": quartet,
            "training_events": 6*(1024+3*64), "training_target_symbols": 49*6*(1024+3*64),
            "worker_cpu_seconds": time.process_time()-begin,
            "peak_rss_macos_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "restart": restart}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path)
    ap.add_argument("cycle", choices=("1", "2", "3", "resume"))
    ap.add_argument("checkpoint", type=Path, nargs="?")
    args = ap.parse_args()
    sources = source_receipt(args.archive)
    sources["ordered_harness_sha256"] = digest(Path(__file__))
    number = "3" if args.cycle == "resume" else args.cycle
    path = ROOT / f"CYCLE{number}_TEST.json"
    if args.cycle == "1":
        receipt = cycle1()
    else:
        backend = Backend(args.archive)
        if args.cycle == "2":
            receipt = cycle2(backend)
        elif args.cycle == "3":
            assert args.checkpoint is not None and not args.checkpoint.exists()
            receipt = cycle3(backend, args.checkpoint)
        else:
            receipt = json.loads(path.read_text())
            assert receipt["status"] == "RESTART_PENDING" and receipt["sources"] == sources
            check = receipt["restart"]
            assert digest(args.checkpoint) == check["checkpoint_sha256"]
            old = backend.restore(args.checkpoint)
            life = make_life(backend, "ordered")
            life.model.restore(old.model.state())
            life.events = old.events
            life.train(bytes.fromhex(check["record_hex"]))
            check["actual"] = state_hash(life)
            assert check["actual"] == check["expected"]
            receipt["status"] = "ENGINEERING_PASS"
            receipt["additional_restart_update_events"] = 2
    receipt["sources"] = sources
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2) + chr(10))
    temp.replace(path)
    print(json.dumps({"cycle": args.cycle, "status": receipt["status"]}))


if __name__ == "__main__":
    main()
