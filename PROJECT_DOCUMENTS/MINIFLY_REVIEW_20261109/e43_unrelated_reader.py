"""A second reader of E43's unrelated task, written from the words of the brief (revision 1, 75613729), standard library only.

The brief's words: 64 symbols (bytes 128 to 191), 32 fillers (192 to 223), a code byte (224), a query byte (225).
An item is `code s1 s2 f_1 .. f_k query`, answer `s1 s2` and the end symbol (256), k from 0 to 12.
The 4,096 ordered pairs of symbols are shuffled by Random(f"e43|{seed}|split"); the first 2,048 are taught.
Stream: Random(f"e43|{seed}|stream"); the taught pairs in a fresh shuffled order each pass; for each item randint(0, 12) for k,
then k fillers by choice.  Batches of 64 consecutive items.
Panels: Random(f"e43|{seed}|panel|held") samples 300 held-out pairs and renders each at k = 12; the same with `taught`.

Where the words leave a choice, the reading taken is a keyword argument, so that the other reading can be run beside it:
  pairs_order   "s1 outer" (s1 runs slowest, the default) or "s2 outer"
  pass_order    "fresh" (each pass shuffles the positions 0..2047 afresh, the default) or "running" (each pass shuffles the
                order the previous pass left)
  population    "as split" (the sample is drawn from the pairs in the order the split left them, the default) or "sorted"

The two digests are not defined in the brief's words.  They are the specification file's: over the first n updates, for every
item sha256 takes one byte for the prompt's length, the prompt's bytes, and the answer's bytes modulo 256; a panel's digest is
the sha256 of Python's repr of the list of (prompt, answer) pairs.

    python3 -I e43_unrelated_reader.py        prints the digests of the three seeds
"""
import hashlib
from random import Random

SYMBOLS = list(range(128, 192))
FILLERS = list(range(192, 224))
CODE, QUERY, END = 224, 225, 256
SEEDS = (73129, 93683, 116089)


def split(seed, pairs_order="s1 outer"):
    if pairs_order == "s1 outer":
        pairs = [(s1, s2) for s1 in SYMBOLS for s2 in SYMBOLS]
    else:
        pairs = [(s1, s2) for s2 in SYMBOLS for s1 in SYMBOLS]
    Random(f"e43|{seed}|split").shuffle(pairs)
    return pairs[:2048], pairs[2048:]


def item(s1, s2, k, rng):
    fillers = [rng.choice(FILLERS) for _ in range(k)]
    return [CODE, s1, s2] + fillers + [QUERY], [s1, s2, END]


def stream(seed, pairs_order="s1 outer", pass_order="fresh"):
    taught, _ = split(seed, pairs_order)
    rng = Random(f"e43|{seed}|stream")
    order = list(range(len(taught)))
    while True:
        if pass_order == "fresh":
            order = list(range(len(taught)))
        rng.shuffle(order)
        for i in order:
            k = rng.randint(0, 12)
            yield item(taught[i][0], taught[i][1], k, rng)


def batches(seed, n, **kw):
    st = stream(seed, **kw)
    return [[next(st) for _ in range(64)] for _ in range(n)]


def panel(seed, which, pairs_order="s1 outer", population="as split"):
    taught, held = split(seed, pairs_order)
    pop = taught if which == "taught" else held
    if population == "sorted":
        pop = sorted(pop)
    rng = Random(f"e43|{seed}|panel|{which}")
    chosen = rng.sample(pop, 300)
    return [item(s1, s2, 12, rng) for s1, s2 in chosen]


def stream_digest(seed, updates, **kw):
    h = hashlib.sha256()
    for batch in batches(seed, updates, **kw):
        for prompt, answer in batch:
            h.update(bytes([len(prompt)]) + bytes(prompt) + bytes(v % 256 for v in answer))
    return h.hexdigest()


def panel_digest(seed, which, **kw):
    return hashlib.sha256(repr(panel(seed, which, **kw)).encode()).hexdigest()


if __name__ == "__main__":
    for s in SEEDS:
        t, h = split(s)
        print(f"seed {s}: taught {len(t)}, held out {len(h)}, disjoint {set(t).isdisjoint(h)}")
        print(f"   stream, first 64 updates   {stream_digest(s, 64)}")
        print(f"   stream, first 4,096 updates {stream_digest(s, 4096)}")
        print(f"   held-out panel             {panel_digest(s, 'held')}")
        print(f"   taught panel               {panel_digest(s, 'taught')}")
        first = batches(s, 1)[0][0]
        print(f"   first item of the stream   {first}")
