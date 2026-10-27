"""V169 counting baselines (design section 2), evaluator-side, MiniFly-free (numpy only).
PPM-C: V148's ladder, unchanged. Interpolated Kneser-Ney: absolute discount D at every order, raw counts for the maximal
context, continuation counts for its proper suffixes, uniform base over 257 symbols; contexts never cross a line boundary."""
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(1, str(ROOT / "minifly_v148_pretrain_probe_dev_20261025"))
import ladder  # noqa: E402  (V148; its 'corpus' import resolves to V148's, which only supplies EOS)

_s = importlib.util.spec_from_file_location("v155_corpus", ROOT / "minifly_v155_keys_dev_v1_20261026" / "corpus.py")
corpus = importlib.util.module_from_spec(_s)
sys.modules["v155_corpus"] = corpus
_s.loader.exec_module(corpus)
EOS, V = 256, 257
GUT = b"End of Project Gutenberg"


def splits(text, role):
    """V155's split rules; texts are cut at their first 'End of Project Gutenberg' record (V167 amendment 1)."""
    recs = corpus.records(corpus.load(text))
    cut = next((i for i, r in enumerate(recs) if r.startswith(GUT)), len(recs))
    return corpus.split(recs[:cut], *corpus.SIZES[role])


def stream(r):
    return list(r) + [EOS]


class IKN:
    """Interpolated Kneser-Ney with context length k (k previous symbols, as PPMC(k)). A context shorter than k occurs
    only at a line start and is treated as maximal (raw counts); proper suffixes use continuation counts."""

    def __init__(self, k, D):
        self.k, self.D, self.name = k, D, f"ikn{k}_D{D}"

    def fit(self, recs):
        raw = defaultdict(lambda: defaultdict(int))
        ext = defaultdict(set)                                  # (suffix, x) -> distinct left extensions
        for r in recs:
            s = stream(r)
            for i, x in enumerate(s):
                ctx = tuple(s[max(0, i - self.k):i])
                raw[ctx][x] += 1
                for j in range(1, len(ctx) + 1):
                    ext[(ctx[j:], x)].add(ctx[j - 1])
        cont = defaultdict(lambda: defaultdict(int))
        for (suf, x), vs in ext.items():
            cont[suf][x] = len(vs)
        pack = lambda t: {c: (dict(d), sum(d.values()), len(d)) for c, d in t.items()}  # noqa: E731
        self.raw, self.cont = pack(raw), pack(cont)
        return self

    def _p(self, ctx, x, top):
        if ctx is None:
            return 1.0 / V
        lower = self._p(ctx[1:] if ctx else None, x, False)
        e = (self.raw if top else self.cont).get(ctx)
        if e is None:
            return lower
        d, tot, types = e
        return max(d.get(x, 0) - self.D, 0.0) / tot + self.D * types / tot * lower

    def prob(self, ctx, x):
        return self._p(tuple(ctx[-self.k:]) if self.k else (), x, True)


def line_bits(model, recs):
    """Per-line bit sums with the ladder's context convention (context resets at each line)."""
    out = []
    for r in recs:
        s, b = stream(r), 0.0
        for i, x in enumerate(s):
            b -= math.log2(model.prob(s[:i], x))
        out.append(b)
    return np.array(out)


def models():
    for k in (2, 3, 4, 5):
        yield ("PPM", k, None, lambda k=k: ladder.PPMC(k))
    for k in (2, 3, 4, 5):
        for D in (0.5, 0.75, 0.9):
            yield ("IKN", k, D, lambda k=k, D=D: IKN(k, D))
