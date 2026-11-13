"""Reference ladder: simple counting predictors over the 257-symbol alphabet (256 bytes + EOS), evaluator-side only.

All are trained on the same records as MiniFly and frozen at test. Contexts never cross a record (line) boundary:
at a line start the context is empty.
"""
import math
from collections import defaultdict

from corpus import EOS

V = 257


def stream(rec):
    return list(rec) + [EOS]


class Uniform:
    name = "uniform"

    def fit(self, recs):
        return self

    def prob(self, ctx, x):
        return 1.0 / V


class Unigram:
    name = "unigram"

    def fit(self, recs):
        self.c = defaultdict(int)
        self.n = 0
        for r in recs:
            for x in stream(r):
                self.c[x] += 1
                self.n += 1
        return self

    def prob(self, ctx, x):
        return (self.c[x] + 1.0) / (self.n + V)          # add-one


class Bigram:
    """Witten-Bell interpolation of the previous-symbol model with the add-one unigram. Line start = context ()."""
    name = "bigram"

    def fit(self, recs):
        self.uni = Unigram().fit(recs)
        self.c = defaultdict(lambda: defaultdict(int))
        for r in recs:
            prev = ()
            for x in stream(r):
                self.c[prev][x] += 1
                prev = (x,)
        return self

    def prob(self, ctx, x):
        key = tuple(ctx[-1:])
        d = self.c.get(key)
        pu = self.uni.prob(ctx, x)
        if not d:
            return pu
        n, t = sum(d.values()), len(d)
        lam = n / (n + t)
        return lam * d.get(x, 0) / n + (1 - lam) * pu


class PPMC:
    """Static PPM with escape method C and full exclusion, maximum order k, falling back to order 0 and then to a
    uniform distribution over the symbols not yet excluded."""

    def __init__(self, k):
        self.k = k
        self.name = f"ppmc{k}"

    def fit(self, recs):
        self.c = defaultdict(lambda: defaultdict(int))     # context tuple -> symbol -> count
        for r in recs:
            s = stream(r)
            for i, x in enumerate(s):
                for order in range(0, self.k + 1):
                    if order > i:
                        break
                    self.c[tuple(s[i - order:i])][x] += 1
        return self

    def prob(self, ctx, x):
        ctx = list(ctx)
        excluded = set()
        p_escape = 1.0
        top = min(self.k, len(ctx))
        for order in range(top, -1, -1):
            d = self.c.get(tuple(ctx[len(ctx) - order:]) if order else ())
            if not d:
                continue
            avail = {s: n for s, n in d.items() if s not in excluded}
            if not avail:
                continue
            n, t = sum(avail.values()), len(avail)
            if x in avail:
                return p_escape * avail[x] / (n + t)
            p_escape *= t / (n + t)
            excluded |= set(avail)
        return p_escape / (V - len(excluded))


def bits(model, recs):
    """Total bits and symbol count of recs under model (contexts reset at each record)."""
    tot, n = 0.0, 0
    for r in recs:
        s = stream(r)
        for i, x in enumerate(s):
            p = model.prob(s[:i], x)
            tot -= math.log2(p)
            n += 1
    return tot, n


def ladder():
    return [Uniform(), Unigram(), Bigram(), PPMC(2), PPMC(3), PPMC(4), PPMC(5)]
