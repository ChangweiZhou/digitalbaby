"""V148 post-hoc descriptive figures (written during the main run; not in the lock, not a prediction, never touches
MiniFly). python posthoc.py results/<tag>

1. Held-out top-1 accuracy of the counting rungs at the largest prefix, as context for MiniFly's held accuracy.
2. The generation statistics named in design section 5 item 4 that the locked summary omits: fraction of real words
   (alphabetic tokens that occur in the taught text) and repetition (1 - distinct/total byte 3-grams), for the
   memory's own continuations and, as a reference, for the true continuations.
"""
import json
import re
import sys
from pathlib import Path

import corpus
import ladder

TOK = re.compile(rb"[A-Za-z]+")


def acc(model, recs):
    hit = n = 0
    for r in recs:
        s = ladder.stream(r)
        for i, x in enumerate(s):
            hit += max(range(ladder.V), key=lambda y: model.prob(s[:i], y)) == x
            n += 1
    return hit / n


def gen_stats(texts, vocab):
    toks = [t.lower() for s in texts for t in TOK.findall(s)]
    rep = [1 - len({s[i:i + 3] for i in range(len(s) - 2)}) / (len(s) - 2) for s in texts if len(s) > 3]
    return {"tokens": len(toks), "real_word_fraction": sum(t in vocab for t in toks) / max(1, len(toks)),
            "repetition_3gram": sum(rep) / max(1, len(rep)), "mean_bytes": sum(map(len, texts)) / len(texts)}


def main():
    d = Path(sys.argv[1])
    r = json.loads((d / "RESULTS.json").read_text())
    sp = corpus.split(corpus.records(corpus.load(r["text"])), max(r["prefixes_kb"]), r["calib_kb"], r["held_kb"])
    train = sp["train"][:r["prefix_records"][str(max(r["prefixes_kb"]))]]
    out = {"note": "post hoc, descriptive, evaluator-only", "held_top1_accuracy_at_max_prefix": {}}
    for m in (ladder.Unigram(), ladder.Bigram(), ladder.PPMC(3)):
        out["held_top1_accuracy_at_max_prefix"][m.name] = acc(m.fit(train), sp["held"])
    vocab = {t.lower() for rec in train for t in TOK.findall(rec)}
    gens = next(c["minifly"]["generations"] for c in r["checkpoints"] if "generations" in c["minifly"])
    out["generations"] = {k: gen_stats([g[k].encode("latin-1") for g in gens], vocab) for k in ("own", "true")}
    out["minifly_held_accuracy"] = {f"{c['kb']:g}KB_pass{c.get('pass', 1)}": c["minifly"]["held_acc"]
                                    for c in r["checkpoints"]}
    (d / "POSTHOC.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
