# Memo: round 24 — state-law learner transfer, done carefully (2026-11-04)

Evidence: dev; 10 world seeds; token-level scaffold (as in round 23). Code and logs: `minifly_review_experiments/round24_statelaw_transfer_20261104/`.

## Protocol

**Stream:** a learner reads either the fresh world alone (**k0**) or 24 prior worlds followed by the fresh world (**k24**).

**Fresh world:**
- new names, places and objects;
- movement sentences only from the **held-out templates** ("strolled into the", "At the L one finds N", "has reached the");
- 80 questions;
- scored online, before each answer is seen.

Get and drop templates are shared with history.

**Metrics:** answer bits (chance log2 6 = 2.58), top-1.

## Results: three rounds of checking, because the first numbers were misleading

| run | what was different | k0 → k24, object top-1 | k0 → k24, person top-1 | bits saved per answer (obj / per) |
|---|---|---|---|---|
| A, 5 seeds (`transfer.txt`) | original generator | mixed: seeds 0, 1 large gains; seeds 2, 4 large losses | – | not computed (inconsistent) |
| B, 10 seeds (`transfer_u.txt`) | **removed name/place homographs** (seed 2 had "Tete" the person and "tete" the place) | 0.535 → 0.553 | 0.79 → 0.72 | **−1.98 / −1.85 (negative transfer)** |
| B2 (`transfer_u2.txt`) | case-folded function-word test, "long-lived word" criterion | 0.535 → 0.495 | 0.79 → 0.64 | −2.43 / −2.44 |
| **C, 10 seeds (`transfer_v.txt`)** | **vocabulary strictly disjoint across worlds and from function words** | **0.535 → 0.980** | **0.80 → 0.96** | **+1.70 ± 0.29 / +0.66 ± 0.25** |

In C the answer bits are:
- **object: 2.31 → 0.61;**
- **person: 1.56 → 0.90.**

**Robustness:** the function-word "span" setting of 800 / 1,500 / 3,000 lines gives the same C result (object top-1 0.98, object bits 0.61–0.62).

## What the debugging showed

**The failure in A and B was the learner's function-word detector, not the state law.** The syllable generator reuses names across worlds (Codex V211B flagged this). A place like "bito", met in an earlier world, looks long-lived, so it is classed as a function word. That corrupts the event signature (`at the bito _ _ _`) and its place slot. The experienced learner is *more* exposed to this than the fresh one, which never has any function words in a 400-line life. That explains the negative transfer.

So the claim is narrow and conditional.

**With clean, non-recurring content words:**
- 24 worlds of experience let the learner apply learned event laws to a new world with new templates almost immediately: object answers go from 54% to 98% right, at about 0.6 bits.
- What transfers is the verb → operation mapping (follow holder / freeze at drop) plus how to type slots.

**With realistic vocabulary reuse, the content/function split collapses and transfer turns negative.** This is now the main weakness. Telling *roles* (function words) from *fillers* (content words) by surface statistics alone is fragile. A better test would use distributional role: fillers are interchangeable within a slot, function words are not.

## Discipline notes

- I fixed the generator (homographs, reuse) instead of tuning the learner on test numbers, and I report all versions.
- **Scaffolds remain:** words, line = event, the 4-operation repertoire, the entity → place store.
- This is still not in the byte-level compressor.

## Next

1. **Robust role detection.** Within a signature slot, measure how *substitutable* a word is: many different words fill that slot → filler. Use this in place of lifetime span, so transfer survives vocabulary reuse. Test on condition B.
2. Report the write effect (does experience change *state laws*, or only typing?) by freezing op reliabilities after history versus letting them adapt.
3. Then remove the operation repertoire, and embed the learner as an expert in the GLN compressor.
