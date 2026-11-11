# Draft for audit: step 0 of the target test, the assay with no learner (11 November 2026)

> **Status.** Draft for Codex's audit. **No worker starts from this file.** It turns step 0 of `PROJECT_DOCUMENTS/TARGET_TEST_DESIGN_DRAFT_20261110.md` (rows 0a to 0d of its section 5, with the conditions of its section 7a) into something that can be built and checked. Every size below is a proposal, marked so; the audit is asked to change them. When audited it becomes a job for one worker: new files only, no training, no model, no use of the arithmetic harness.

> **Version 2, 10 November 2026, 22:09 EST (clock), after Codex's survey memo** (`PROJECT_DOCUMENTS/MINIFLY_PROGRESS_SURVEY_AND_TARGET_ASSAY_20261110.md`). Both of its corrections were simulated and hold (`PROJECT_DOCUMENTS/target_step0_draft_checks.py`, output `TARGET_STEP0_DRAFT_CHECKS.json`). What changed from version 1:
> 1. Version 1 taught 216 pairs and revealed the other 72, which used all 288 pairs of a family and left none for twins that are "never revealed". The split now reserves a sealed pool before anything is generated.
> 2. Version 1 showed check 3 "able to fail" by giving the memorising responder a 40-byte window and requiring every twin. That cannot happen: a table that has never seen a sealed pair scores none of them at any window (0 of 300 in the simulation). Visibility is now qualified with the rule-aware evaluator under two inputs, and the tables are separate memorisation controls.
> 3. The two families shared context values. Then the control that removes a context's family-1 teaching still leaves its answer fixed through family 2 (confirmed by a solver over Z_97). Each family now has its own context values.
> 4. The two members of a twin are scored from copies of the same learner state.
> Still a draft for audit. No worker starts from this file.

## What step 0 has to establish

That the stream and its panels can tell a learner that uses earlier history from one that does not, before any learner exists. If step 0 cannot show that, nothing built on it means anything. This is the lesson of E20-A, whose first design could not have told the two apart (review 9.79).

## The stream, made concrete (all sizes proposed)

- **Symbols.** Contexts: 24, written as capital letters `A` to `X`. Cues: two families of 12, lower-case `a` to `l` and `m` to `x`. Filler: 8 punctuation bytes, drawn uniformly. Outcome: two decimal digits. `=` before the outcome; an end byte after it.
- **Episode.** `X` filler `c` `=` `dd` end. One context, 0 to 14 filler bytes, one cue.
- **Hidden rule.** Each cue has a value u(c) and each context has one value per family, v1(X) and v2(X), all in 0 to 96, drawn once per world seed. Family 1: outcome = (v1 + u) mod 97. Family 2: outcome = (v2 − u) mod 97. The families share the context symbols and nothing else. Twin contexts must have different values in the twin's family.
- **Declared window: 10 bytes.** At `=` the last ten bytes are the end of the filler, the cue and `=`. A panel episode has at least 10 filler bytes, so the context is outside them. Teaching episodes use the whole range of 0 to 14.
- **Taught, first-encounter and sealed pairs.** Per family 24 × 12 = 288 pairs, split before anything is generated: 216 taught; 48 first-encounter, in three sets of 16 for acquisition, retention and revision; 24 sealed for good. A first-encounter pair is scored the first time it occurs in its phase and never again as held. The sealed 24 are 12 twin pairs, one per cue: the same cue with two contexts. These numbers follow Codex's illustration and are proposals; the worker qualifies the actual split with check 1.
- **Phases.** Acquire: both families, taught pairs, with that phase's held pairs arriving once each. Interfere: family 1 absent for a declared stretch; its retention pairs arrive when it returns. Revise: at a declared byte position the values of two family-1 cues change; taught pairs with those cues go on being revealed; the revision pairs include six with a changed cue.
- **Panels that are never revealed.** For each phase and family, 150 twins drawn from the 12 sealed twin pairs: two episodes with the same cue and the same filler bytes, at least 10 of them, and the pair's two contexts. The sealed pairs never occur in the stream. Renderings differ only in filler, so results are reported twice: over the 150 renderings, and over the 12 distinct twin pairs per family (a pair counts when a declared share of its renderings is right).
- **The wider-view arm and its two comparison streams** are generated here but not used until step 4 of the design: after an episode's end byte, a line `X:vv c:uu`; the same lines shuffled within strata of delay and of how often each symbol has occurred; and a stream with the same number of extra ordinary bytes.

## Checks. Each must pass, and each must be able to fail

`target_step0_check.py` exits nonzero on any failure and writes `TARGET_STEP0_CHECK.json`, for three world seeds.

| # | Check | Passes when | Shown able to fail by |
|---|---|---|---|
| 1 | **Identification.** For the taught pairs of each phase, every held pair's outcome is the same under every assignment of values consistent with the taught outcomes | the taught graph of contexts and cues is connected per family, **and** a solver over Z_97 confirms that each held pair's sum (or difference) is a fixed combination of taught outcomes | a fixture split with one context taught with no cue shared with the rest: the check must report its held pairs as undetermined |
| 2 | **Oracle.** A responder given the hidden values | scores every twin (both members), every first-encounter pair, in every phase, including after the revision | a fixture oracle with one value wrong must lose exactly the twins and pairs that use it |
| 3 | **Visibility.** The rule-aware evaluator of check 2 (it solves the values from what the stream has revealed) under two inputs | given the complete episode it scores every twin; given only the shared last ten bytes it scores no twin, and at most one member of any twin, because it must give one answer per suffix | the evaluator given the suffix plus the context byte must score every twin again |
| 3b | **Memorisation controls.** Three tables trained on the stream: one answer per ten-byte suffix; one per forty-byte suffix; one per exact context and cue | each scores no sealed twin and no first-encounter pair before its reveal. Their failure shows the mapping is absent, not that it is out of view | a fixture stream into which the sealed pairs are inserted: the exact-key and forty-byte tables must then score them |
| 4 | **Recency table.** A responder that answers with the last outcome it saw for the cue | scores no twin; after the revision, answers the old outcome on revision pairs and is counted as following the obsolete rule | a fixture stream with one context only, where recency is correct, must give it full marks |
| 5 | **Exposure.** A ledger over the whole stream | shows no held pair, and no sealed twin pair, occurring before its scoring point; counts every occurrence of every pair | a fixture stream with one held pair inserted early must be reported |
| 6 | **Twins.** | members share their last ten bytes at `=`, differ in context, differ in both outcome digits' value as a number, and have the same length | a fixture twin whose contexts share a value must be rejected |
| 7 | **The shuffled stream.** | each revealed symbol keeps its frequency and each line its delay stratum; no shuffled line describes its own episode more often than chance | a fixture "shuffle" that leaves lines in place must be reported as aligned |
| 9 | **The prior-teaching control.** A second stream with every family-1 teaching episode of the control contexts removed | the solver reports the family-1 answers of those contexts as undetermined, with all of family 2 still present | a fixture world in which the families share context values: the solver must report those answers as still determined |
| 10 | **Probes leave no trace.** A stateful fixture responder scored on a twin | its state after the twin is byte-for-byte its state before; each member was answered from the same state | a fixture scorer that presents the members one after the other without restoring must be reported |
| 8 | **Determinism and hashes.** | the same world seed gives the same bytes on two runs; the stream, each panel and each phase's held set have recorded hashes | changing the seed changes every hash |

## What a learner will be asked later, fixed now so the assay matches it

One output per scored episode: the bytes after `=`, written by the learner itself before the stream reveals them, compared whole, both digits and the end byte. A twin counts only when both members are right. The scorer saves the learner's state before a twin, presents one member, restores the state, presents the other, and restores again; the stream continues from the saved state. No answer mask, no task code, no boundary signal: the learner sees bytes and predicts the next one, and the scoring points are known to the scorer, not to the learner.

## Deliverables, when this becomes a job

`target_stream.py` (generator, panels, the three streams, no dependencies beyond the standard library), `target_step0_check.py`, `TARGET_STEP0_CHECK.json` for three seeds, and a report of at most two pages: the counts, every check with its fixture, the hashes, and whatever the worker found unclear in this text. No learner is trained. Nothing is concluded about any learner.

## Questions for the audit

1. Are 24 contexts and 12 cues per family enough for an ordinary recurrent reference to learn the values from 216 taught pairs, or does the reference need the count of revealed episodes fixed first?
2. Is sum and difference modulo 97 the right hidden structure? It makes held outcomes inferable and checkable. It also brings back two-digit arithmetic, whose shortcuts this project knows. A table with no algebra would make held pairs uninferable.
3. (Answered in version 2: the families share context symbols and no values. Should they also use different context symbols?)
4. The filler is uniform noise. After the every-byte result of 10 November (review 9.88), in which unpredictable bytes are the first suspect for the working rule's collapse, should a predictable-filler version of the same stream be generated alongside, so that step 1 can separate the two?
5. Does first-encounter scoring need a minimum gap between a pair's last taught neighbour and its scoring point?
6. Check 3 says "at chance". The chance level of a two-digit outcome is about 1%; is a bound of 5% the right pass mark?
7. Twelve distinct twin pairs per family is a small number to rest a claim on, however many renderings are scored. Should the world be larger (for example 32 contexts), or the sealed pool be taken from the taught 216?
