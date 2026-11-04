# MiniFly V209 — Contextual Answer Selection

All 2,016 evaluations completed across 12 story batches and three hash seeds. The primary lexical-transfer gate passed. Contextual selection scored 100% on jointly new names and locations in familiar wording, compared with 0.8% for word identity. The unseen-template gate failed at 0%; two-location distractor accuracy was 44.6%.

On jointly new words, shuffling word–context bindings reduced accuracy to 18.8%. The selector tied on 82.4% of questions; randomizing candidate order reduced accuracy to 54.5%. Random-location feedback matched true feedback at 100% under the original ordering. Scores measure selected whole words.

V209 reimplements Round 14's described mechanics. The original nobuilt.py was unavailable. The identity arm is a matched V209 control.

## Reproduction

Use Python 3 and SciPy. In this directory, run `python -B qualify.py` followed by `python -B official.py`. The runner generates synthetic stories from the recorded seeds and requires no external data. Results include per-run emitted words and batch-paired contrasts.
