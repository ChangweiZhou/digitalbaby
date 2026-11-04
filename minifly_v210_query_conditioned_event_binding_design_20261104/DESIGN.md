# MiniFly V210 · Query-conditioned selection from retained events

**Status** DESIGN_ONLY  
**Date** November 4, 2026  
**Question** Given two people, two events and several possible values of the same type, can a small local readout learn to choose the value that answers the current question, even when the candidate list is reordered and the person has left the recent byte window?

## Why this test

V206 showed that retaining early raw bytes can restore person–destination binding under neutral padding, with write timing supplied. V209B reached 100% on renamed words in familiar syntax, 58.3% after candidate reordering, and 0% on a new template despite 100% target availability. Candidate order can legitimately reflect retrieval evidence. The ambiguity is whether the **trained selector** contributes question-dependent choice beyond that evidence. V209's random-location feedback still teaches the answer category, so it is an incomplete control for the event and role distinction.

V210 fixes the candidate set in its main diagnostic, exposes each candidate's retrieval vote as a number, and tests two questions about different roles on exactly the same observed story. V207 prose pretraining and V208 action credit remain separate. V208's transitive parent source still needs publication for reproducibility.

[V206 report](https://github.com/ChangweiZhou/digitalbaby/blob/c449ba61cca8569afbc839de21d31ac9dade56e1/minifly_v206_causal_entity_binding_design_20261104/official/REPORT.md) · [V209 report](https://github.com/ChangweiZhou/digitalbaby/blob/aea279f04afccb2fbe105fec2499a5ab8cee0362/minifly_v209_contextual_answer_selection/README.md) · [V209B matched comparison](https://github.com/ChangweiZhou/digitalbaby/blob/52843ed0e637363b7a9b10966e6e87fde93db763/minifly_v209b_matched_round14/README.md)

## Task, exposure and candidate interface

A story contains two completed event statements about different people. Each statement describes going to one place and avoiding another. All four places are distinct. For example:

> Lucy avoided the porch and went to the cellar.  
> Paul went to the studio and avoided the garage.

After both statements, ask both questions about one person, for example “Where did Lucy go?” and “Which place did Lucy avoid?” Each question is answered from an identical read-only snapshot. No answer, question, or score can enter the event memory. Balance queried person, event order, clause order, and which question comes first. Place identity is independent of role. Use two event wrappers (plain and an initial “Then ”) and two question frames per role, using the same role words in both frames. Lock exact byte forms before opening held data. The primary test uses **unseen pairings of familiar event and question frames**. With two frames on each side, teach pairs (0,0) and (1,1) and hold out (0,1) and (1,0). Teach both clause orders in every trained frame. Use disjoint generated person and place tokens for training and held panels. Hold out entire stories, never individual questions from a training story.

Each world has 128 training stories, each with two answer observations, one chronological pass. Train with both unpadded and padded statements in equal numbers. The evaluation has 64 fresh paired story identities, each rendered at zero and 40 extra bytes. Insert opaque neutral bytes between subject and the clause phrases. The recent context must lose the subject at both padded candidate sites, while the retained prefix must include it. Preserve the local role cues near each candidate and validate these properties on the actual bytes before training. Freeze the answer lengths, output bounds and disjoint vocabulary. No probe feedback, including gold answers after a question.

**Main pool diagnostic.** A fixed event boundary and a supplied candidate-span interface retain the four observed location spans. Each candidate contains its observed payload bytes and the two bounded context views defined below. The interface supplies all four spans to both questions, without roles, truth values, canonical entity keys or a relevant-event flag. All four candidates are valid locations. The candidate interface is an experimental provision of availability and type. The experiment does not learn span detection or write timing. Its results are conditional on this access.

**Source retrieval diagnostic.** In a separate read-only panel, run the pinned V209 retriever over the same stories and report top-16 target coverage, including wrong types, and selector accuracy both unconditionally and given coverage. Preserve V209's own tokenization, retrieval cap and frozen question-frequency statistics. Do not substitute the supplied pool when the target is absent. Verify that freezing or swapping the selector does not change the candidate pool. A result on the supplied pool qualifies only the readout. This extra panel localizes the retrieval gap and does not gate the primary contrast.

## Operative readout

Take the **first 16 observed bytes of each statement** as its retained anchor. Snapshot at the candidate's completion, after its payload has arrived. Store the trailing 32 bytes ending at the beginning of the candidate span as the recent view, and use the candidate's immediately preceding observed bytes as its local role context. Never use following bytes in that candidate's context. The statement boundary is observable and fixed. An event memory contains at most two statements and four candidate spans in the supplied-pool assay. All candidates enter with the same admission rule. Parent V206 supplies the retention principle and byte limits, not a claim of source-level integration with its episodic writer.

For each query q and candidate c use the score:

    s(c,q) = r(c,q) + θ · φ(q, anchor(c), recent(c), local(c))

Extract raw byte 2–4-grams from the bounded query and context views. Use fixed domain-separated hashes, separate slots for view identity, and two interaction families: shared query–anchor grams and query grams crossed with local candidate-context grams. Shared grams can signal raw continuity across observations without parsing a person name. Crossed grams allow the same question feature to score candidates differently depending on preceding context. Include local-only and recent-overlap features in all learned arms. Do not provide answer type, role, clause index, entity ID, template ID, queried-person flag or ground-truth position as a feature. Do not use the candidate word as a learned answer-class ID. Preserve the payload for actual output and the audit.

Expose V209's pre-sort nonnegative retrieval votes; normalize over the supplied candidates, using zero if no vote reaches that span. Match votes to candidate occurrences by exact observed payload and source context, with an auditable occurrence map. If V209 merges indistinguishable occurrences, give those occurrences the same reported vote and record the collision. Use coefficient one on the normalized vote in every learned arm. No fallback to implicit list order. For the retriever-only control, give every arm's candidate the same vote. Test an additional all-zero-votes panel on the same frozen checkpoint, without retraining. This panel is diagnostic for dependence on retrieval evidence.

Use a fixed 65,536-entry float32 readout, initialized to zero. Bound feature generation to 16 query grams, 16 grams per context view and 96 selected sparse features per candidate, using deterministic hash ranking and no global table scan. For the shared-gram family, intersect raw gram sets before quota selection so independently truncated lists do not erase the overlap by construction. Reserve fixed family quotas during qualification before opening the held roster. L2-normalize each sparse feature vector. With the same correct answer feedback for all true-learning arms, find the highest-scoring incorrect candidate, and when the correct candidate's lead is below margin 1 apply:

    θ ← θ + η [φ(correct) − φ(highest-scoring incorrect)]

This is a local readout correction, with no gradient or feedback to the event memory. On exact ties between incorrect candidates use a frozen occurrence digest independent of enumeration. Use step sizes 0.1 and 0.3 for training story blocks 1 and 2 (16 stories each), then 0.6. Record applied updates. A missing target is a coverage failure with no update. Main-pool qualification requires 100% target coverage for both questions.

The selected payload is emitted verbatim as the output of this finite-choice port. This assay scores **selected whole byte spans**, separately from generated answers with own EOS. Report strict unique-correct top score as primary and actual emitted accuracy as secondary. For output only, exact top ties use the same frozen enumeration-independent occurrence digest. Never break ties with answer labels, candidate insertion order, or role. Keep score arithmetic and feature summation order fixed. Audit original, reversed and preseeded permuted orders while carrying the full record and vote along. Require identical scores by occurrence, tie sets and emitted payloads.

## Contrasts

| Arm | Information and update |
|---|---|
| VOTE | Explicit retrieval score alone, with the same candidates and tie rule |
| POSITION | Retrieval plus a fixed position prior fitted from training answers without the question. Report both position alone and the sum using unit coefficients. |
| QUERY-ANCHOR | Full readout with true answer feedback |
| QUERY-RECENT | Same readout and feature quotas, using a second recent-context view in the anchor lane |
| QUESTION-BLIND | Anchor and candidate features, with no question in the learned feature map. Keeps the same question-derived retrieval score. |
| SAME-TYPE-SHUFFLE | Identical to QUERY-ANCHOR. Swap the two answer targets for half the training stories in each prelocked frame/order/padding stratum, leaving them unchanged for the other half. |

Every training story supplies a pair of answers about the same person. Swapping within a story preserves the exact two target words, their event, answer category, exposures, and available candidates, while eliminating a consistent question-to-role association in expectation. It preserves information about the queried event. It tests the added role association, with that preserved information explicitly acknowledged. Randomization is frozen before learning. In the question-blind arm, remove all learned features derived from q, including overlap and crossing. Its explicit votes may still differ across the paired questions. Compare realized write counts, coverage and sparse feature visits, rather than presuming equal learning motion.

As evaluator-only witnesses, score a canonical person-and-role table of observed events and a fit-on-training surface role rule. An exact training question/answer lookup has no taught answer on the new story and vocabulary. The surface rule can solve this grammar and defines the scope of a positive result. Include candidate-order-only and last-location references. The shipped scorer must show paired role switching within a story, not only aggregate correct answers.

## Endpoints and admission

Use eight independently generated worlds. Each world has its own training data, held data and selector state. Fix seeds from the first 64 bits of SHA-256 of MINIFLY|V210|WORLD|i for i = 0 through 7. Report world-level paired differences and two-sided 95% Student-t intervals. The primary endpoint is the fraction of **long-padding, held-frame stories on which both questions give their respective unique correct answer**. A tie or a missing answer fails the pair. Short-padding, single-question, same-event wrong-role and wrong-entity errors are secondary.

Advance to a later retrieval-integrated comparison only if:

- QUERY-ANCHOR reaches at least 75% on the primary endpoint and beats each of VOTE, POSITION's two prespecified variants, QUESTION-BLIND and SAME-TYPE-SHUFFLE by at least 10 points in the paired mean. Both the individual paired intervals and a Bonferroni-adjusted one-sided lower bound for each of these five comparisons must exclude zero. Treat the world as the only independent unit.
- QUERY-ANCHOR beats QUERY-RECENT at long padding with paired lower 95% bound above zero, and loses at most five points on unpadded pairs. Report long-padding differences by clause order and queried person. Neither order may have less than 65% paired success.
- The primary candidate and exposure audits pass. All arms are invariant to enumeration, and QUERY-ANCHOR still answers both roles uniquely on at least 60% of long pairs when votes are removed. Report vote removal without changing the trained checkpoint.

Thresholds are prospective development criteria. A gain over VOTE without a gain over SAME-TYPE-SHUFFLE leaves the correct-feedback mechanism unresolved. Success on supplied candidates with poor source-retrieval coverage localizes an integration gap. Success only at zero padding leaves retained binding unresolved. If training fails on unpadded engineering stories, audit feature access and ranker reachability before opening the held roster. Do not adjust salts, endpoints, arms, or data after seeing held outcomes.

This is a bounded readout diagnostic for recombining familiar role cues and frames. The supplied boundaries and spans, and the ability of a surface role rule to solve the task, limit the claim. Report held-combination performance explicitly. Do not treat it as autonomous write selection, semantic interpretation of an unseen construction, or generative QA.

## Budget and three-cycle exercise

Eight worlds × 128 stories × two questions gives **2,048 distinct training question events**. Four learned arms give **8,192 charged training question applications**. The six main arms, eight worlds, 64 held story identities, two padding lengths and two questions give **12,288 scored main-pool question applications**. Reuse those predictions' underlying sparse scores for permutation audits only after a smaller full recomputation establishes parity. Account for actual additional reads, the zero-votes and source-retrieval panels, and fixed-position variants as separate compute and evaluation charges. Lock exact byte and row census before science. Proposed cumulative caps are 10,000 supervised question applications, 64,000 read-only question applications, 600 CPU seconds, 1,200 active wall seconds, 1 GiB peak RSS and 128 MiB retained outputs.

1. **Refine → audit → test the V209 bridge.** Pin its source and reproduce one engineering lexical-order contrast. Expose pre-sort votes and verify equivalence before introducing the new scorer.
2. **Refine → audit → test the paired-role assay.** Use small hand-audited stories to check held-frame exposure, balance, actual subject visibility, candidate occurrence provenance, timing, shuffled answer counts, tie invariance and the surface-rule witness. Include a case where removing the question leaves the pair unsolvable.
3. **Refine → audit → test the complete entry point.** Run separate engineering worlds through training, all arms, read-only panels, reducer pass/fail cases and archive readback. Dry-generate and hash the full roster. Lock cost admission and require separate official authorization.

**Delivery** Design only.
