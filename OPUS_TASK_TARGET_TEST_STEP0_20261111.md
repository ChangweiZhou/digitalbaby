# Draft for audit: step 0 of the target test, the assay with no learner (11 November 2026)

> **Status.** Draft for Codex's audit. **No worker starts from this file.** It turns step 0 of `PROJECT_DOCUMENTS/TARGET_TEST_DESIGN_DRAFT_20261110.md` (rows 0a to 0d of its section 5, with the conditions of its section 7a) into something that can be built and checked. Every size below is a proposal, marked so; the audit is asked to change them. When audited it becomes a job for one worker: new files only, no training, no model, no use of the arithmetic harness.

## What step 0 has to establish

That the stream and its panels can tell a learner that uses earlier history from one that does not, before any learner exists. If step 0 cannot show that, nothing built on it means anything. This is the lesson of E20-A, whose first design could not have told the two apart (review 9.79).

## The stream, made concrete (all sizes proposed)

- **Symbols.** Contexts: 24, written as capital letters `A` to `X`. Cues: two families of 12, lower-case `a` to `l` and `m` to `x`. Filler: 8 punctuation bytes, drawn uniformly. Outcome: two decimal digits. `=` before the outcome; an end byte after it.
- **Episode.** `X` filler `c` `=` `dd` end. One context, 0 to 14 filler bytes, one cue.
- **Hidden rule.** Each context has a value v(X) and each cue a value u(c), in 0 to 96, drawn once per world seed. Family 1: outcome = (v + u) mod 97. Family 2: outcome = (v − u) mod 97, with its own cue values. Twin contexts must have different values.
- **Declared window: 10 bytes.** At `=` the last ten bytes are the end of the filler, the cue and `=`. A panel episode has at least 10 filler bytes, so the context is outside them. Teaching episodes use the whole range of 0 to 14.
- **Taught and held pairs.** Per family 24 × 12 = 288 pairs; 216 taught, 72 held. The held 72 are split before anything is generated into three sets of 24: acquisition, retention and revision. A pair is scored the first time it occurs in its phase and never again as held.
- **Phases.** Acquire: both families, taught pairs, with that phase's held pairs arriving once each. Interfere: family 1 absent for a declared stretch; its retention pairs arrive when it returns. Revise: at a declared byte position the values of two family-1 cues change; taught pairs with those cues go on being revealed; the revision pairs include eight with a changed cue.
- **Panels that are never revealed.** For each phase, 150 twins: two episodes with the same cue and the same filler bytes, at least 10 of them, and two contexts with different values. Sealed: their pairs never occur in the stream.
- **The wider-view arm and its two comparison streams** are generated here but not used until step 4 of the design: after an episode's end byte, a line `X:vv c:uu`; the same lines shuffled within strata of delay and of how often each symbol has occurred; and a stream with the same number of extra ordinary bytes.

## Checks. Each must pass, and each must be able to fail

`target_step0_check.py` exits nonzero on any failure and writes `TARGET_STEP0_CHECK.json`, for three world seeds.

| # | Check | Passes when | Shown able to fail by |
|---|---|---|---|
| 1 | **Identification.** For the taught pairs of each phase, every held pair's outcome is the same under every assignment of values consistent with the taught outcomes | the taught graph of contexts and cues is connected per family, **and** a solver over Z_97 confirms that each held pair's sum (or difference) is a fixed combination of taught outcomes | a fixture split with one context taught with no cue shared with the rest: the check must report its held pairs as undetermined |
| 2 | **Oracle.** A responder given the hidden values | scores every twin (both members), every first-encounter pair, in every phase, including after the revision | a fixture oracle with one value wrong must lose exactly the twins and pairs that use it |
| 3 | **Recent bytes only.** A responder that answers from the last ten bytes at `=`, memorising one answer per ten-byte suffix | scores no twin; scores at chance on first-encounter pairs whose filler is at least 10 | the same responder given a 40-byte window must score every twin |
| 4 | **Recency table.** A responder that answers with the last outcome it saw for the cue | scores no twin; after the revision, answers the old outcome on revision pairs and is counted as following the obsolete rule | a fixture stream with one context only, where recency is correct, must give it full marks |
| 5 | **Exposure.** A ledger over the whole stream | shows no held pair, and no sealed twin pair, occurring before its scoring point; counts every occurrence of every pair | a fixture stream with one held pair inserted early must be reported |
| 6 | **Twins.** | members share their last ten bytes at `=`, differ in context, differ in both outcome digits' value as a number, and have the same length | a fixture twin whose contexts share a value must be rejected |
| 7 | **The shuffled stream.** | each revealed symbol keeps its frequency and each line its delay stratum; no shuffled line describes its own episode more often than chance | a fixture "shuffle" that leaves lines in place must be reported as aligned |
| 8 | **Determinism and hashes.** | the same world seed gives the same bytes on two runs; the stream, each panel and each phase's held set have recorded hashes | changing the seed changes every hash |

## What a learner will be asked later, fixed now so the assay matches it

One output per scored episode: the bytes after `=`, written by the learner itself before the stream reveals them, compared whole, both digits and the end byte. A twin counts only when both members are right. No answer mask, no task code, no boundary signal: the learner sees bytes and predicts the next one, and the scoring points are known to the scorer, not to the learner.

## Deliverables, when this becomes a job

`target_stream.py` (generator, panels, the three streams, no dependencies beyond the standard library), `target_step0_check.py`, `TARGET_STEP0_CHECK.json` for three seeds, and a report of at most two pages: the counts, every check with its fixture, the hashes, and whatever the worker found unclear in this text. No learner is trained. Nothing is concluded about any learner.

## Questions for the audit

1. Are 24 contexts and 12 cues per family enough for an ordinary recurrent reference to learn the values from 216 taught pairs, or does the reference need the count of revealed episodes fixed first?
2. Is sum and difference modulo 97 the right hidden structure? It makes held outcomes inferable and checkable. It also brings back two-digit arithmetic, whose shortcuts this project knows. A table with no algebra would make held pairs uninferable.
3. Family 2 as the unrelated skill shares the contexts with family 1. Should the unrelated skill share nothing?
4. The filler is uniform noise. After the every-byte result of 10 November (review 9.88), in which unpredictable bytes are the first suspect for the working rule's collapse, should a predictable-filler version of the same stream be generated alongside, so that step 1 can separate the two?
5. Does first-encounter scoring need a minimum gap between a pair's last taught neighbour and its scoring point?
6. Check 3 says "at chance". The chance level of a two-digit outcome is about 1%; is a bound of 5% the right pass mark?
