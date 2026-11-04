# MiniFly V214 · Prequential recall and transfer across a lifetime

**Status** DESIGN_ONLY  
**Date** November 4, 2026  
**Decision** Does the Round 20 episodic recall path improve online byte prediction on fresh MiniWorld streams, and does the benefit from prior worlds decrease as the learner accumulates more experience?

## Why this test

Round 20 reports a promising single-stream contrast. Adding one- and two-hop association experts to the byte predictor reduced MiniWorld loss from 0.655 to 0.571 bits/byte. Its shuffled-cue control scored 0.652. The reported gain over a fresh learner on one target world fell from 775 bits after two prior worlds to 563 bits after 24. These are reasons to replicate both contrasts. One stream per comparison does not establish an effect across worlds or identify a loss of plasticity.

V213 remains a separate state-transition experiment. Its local mechanism solved 0/32 held engineering twins while a pooled count reference solved 29/32. A gain in compression would show that the recall path helps a predictor. It would leave the ability to update a relational state, choose an action, or generate an answer autonomously unmeasured.

[Round 20 memo](/Users/pencilbard/Downloads/MINIFLY_MEMO_ROUND20_20261104.md) · [V213 technical result](https://github.com/ChangweiZhou/digitalbaby/blob/6159448b8b84a91da1cd434456e6577928edec8b/minifly_v213_observed_transition_design_20261104/CYCLE3.md) · [V211B forward-transfer result and exposure audit](https://github.com/ChangweiZhou/digitalbaby/blob/20a00e7d061d43eebde06452221a38862d7dc0dc/minifly_v211b_recovered_miniworld_20261104/REPORT.md)

## Source and causal interface

Recover the exact Round 20 `cmr.py`, world generator, configurations, and a saved development receipt before learner qualification. The memo was available. The executable was absent from the checked work folders and parallel checkout. A reconstruction requires explicit provenance and behavioral parity on a bounded historical fixture before it can count as a replication. Pin the GLN mixer, its ordinary experts, H1/H2, association update, vocabulary processing, hashing, numerical floors, and output alphabet. Document any source difference. No new recall mechanism enters Stage A.

Every arm runs the same chronological interface. At site t, it emits a normalized distribution for the next byte using only the bytes already delivered. The observer records −log2 of the probability assigned to the actual byte. The byte then arrives and may update the learner. Preserve a source-defined start context and a common declared distribution floor with proper renormalization. Charge all bytes, including questions, answers, markers and separators. The world log includes answer text as ordinary subsequent observation. It never passes answer-span indices, valid answer lists, story state, a hidden writer cue, or future text to the learner. The evaluator alone uses those fields to compute diagnostic scores. The output under test is a probability distribution, so this assay does not grade autonomous answer generation or own EOS.

H1 and H2 may look up previously observed associates using completed cues and the currently visible word prefix. Qualification must establish the source's write timing. No future character of the current word or unfinished line can enter its association update, IDF, cache, confidence bin, or prediction. A full line may update the store once it has arrived. Verify with suffix perturbation tests that change only bytes beyond the prediction boundary. Check that saved checkpoints contain all operative mutable state, including counts, mixer, association store, byte/word buffers and RNG. Hash and lock the source. Track dictionary and association growth, including any source-specified eviction. If the recovered store is unbounded, characterize its observed size under the campaign cap and leave bounded-lifetime scalability unclaimed.

## Stage A · Paired replication and lifetime curve

| Arm | Information and compute path |
|---|---|
| BASE | Original GLN without informative recall. Preserve the source baseline. Report its actual lower lookup cost separately. |
| RECALL | Original H1 and H2 with the observed causal cues. |
| CUE-SHAM | Same mixer and recall paths as RECALL with cue associations randomized. Keep visible context, teacher bytes and update cadence the same. |

Audit the historical shuffle and first reproduce its behavior on the archived fixture. For the prospective contrast, require the sham to match the number of H1/H2 calls, prefix restrictions, update opportunities, and budget. Pin any causally available shuffle pools and seeds before science. If the source shuffle systematically changes which experts can fire or uses later tokens, it is unsuitable for the prospective contrast. An adapted shuffle must be documented and tested before admission. Log recall hits, availability, confidence, fan-out, and actual row visits in both arms. This is a test of the combined recall path. H1 versus H2 attribution remains open.

Use **16 independent paired families**. Fix seeds from the first 64 bits of SHA-256 of MINIFLY|V214|FAMILY|i for i = 0 through 15, with disjoint engineering seeds. Each family contains one chronological 24-world history D and one held target world T. Generate both before training. Use the same exact D and T bytes across arms. There is no hidden reset, namespace switch, or world-ID input at a boundary. Exclude target names, places and object strings from every prior world in that family under the source tokenizer and normalization. Hold out whole target template families while retaining a prespecified inventory of familiar constituent cues. Report exact template and n-gram overlap. If the recovered generator fails these checks, revise the generator prospectively, disclose the difference from the original run, and verify parity on the separate historical fixture. Never silently treat independently drawn vocabulary as disjoint.

For each arm, train once through D. At birth and after 2, 8, and 24 worlds, copy the checkpoint and expose the fork to **the same** T. All forks predict then update on T through the normal mechanism. No target fork writes back into the ongoing history. Select the first target byte by the same causal interface in every fork. Do not align internal state by injecting a reset. Score the full target and its first quarter separately. Fresh is the birth checkpoint for the **same arm**, so fresh RECALL can learn to use recall during T. Comparing RECALL with BASE also measures effects of the extra mechanism and compute. Comparing RECALL with CUE-SHAM more directly tests whether the observed cue-to-memory association matters.

### Metric contract

For arm a and checkpoint k, let C(a,k) be the prequential code length of T in bits. Define:

- **Forward transfer** G(a,k) = C(a,0) − C(a,k). Positive means prior exposure saves bits on an identical target stream. Report bits, target bits/byte, and the paired interval across families.
- **Common-budget efficiency** G(a,k) divided by the physical number of history bytes, reported as bits saved per KiB of prior exposure. Use exactly the same denominator across arms in a family. Report actual training CPU time and peak state separately. There is no efficiency ratio at k = 0.
- **Round 20 diagnostic** G(a,k) / I(a,k), where I is that arm's own prequential history code length. Report it separately. A learner that predicts D better changes this denominator, so this ratio does not gate cross-arm comparison.
- **Marginal transfer** [G(a,k₂) − G(a,k₁)] divided by the added history bytes over each adjacent interval. Show the numerator as well. A falling ratio can follow from diminishing returns even if G grows. A negative increment on matched T shows worse transfer under this protocol. It cannot, by itself, distinguish a worse starting predictor from impaired ability to learn on T.

Do not pool bytes or worlds within a life as independent observations. Use family means and paired Student-t intervals across the 16 families, with raw per-family values. For the trajectory report, use paired changes at each checkpoint. Ratios may be summarized after computing them within family. Show aggregate numerator and denominator to make their scale visible.

### Main outcomes and interpretation

Primary contrasts on the held target are the savings at k = 24 from RECALL against **both** BASE and CUE-SHAM. Require positive paired lower bounds using one-sided tests with Bonferroni correction at family alpha 0.05. Require a mean saving of at least **0.02 bits/byte** versus each, and report both unrounded differences. Separately require positive RECALL transfer versus its own fresh fork after 24 worlds, with paired two-sided 95% lower bound above zero. A system can improve byte prediction on T while learning nothing reusable from D. Report the cross-arm transfer contrast G(RECALL,24) − G(BASE,24) to distinguish an immediate architecture advantage from an advantage acquired through prior life.

Report target loss on person-answer and object-answer spans, plus nonanswer bytes. Score total bits per complete answer span using a locked boundary and report counts and byte lengths. The evaluator's uniform reference is a distribution over the six current valid place strings, with the same span boundary. It costs log2(6) bits per complete answer when those six possibilities are distinct. That reference has a supplied candidate vocabulary. It is different from chance over bytes and from the true conditional entropy of the world. No answer reference supplies probabilities to MiniFly. Also report an ordinary causal byte-count predictor with the same observations and a position/recency shortcut on answer spans. Treat these as references, not extra mechanisms to tune.

For a prose guard, use eight prelocked, previously unused held-author excerpts, at most 16 KiB each. Replay every arm from its birth on the same excerpts under predict-then-update. Compare paired loss by author. Require the upper two-sided 95% bound on RECALL minus BASE below +0.02 bits/byte to claim the prose guard passed. The prose path is separate. It does not test whether prose pretraining transfers to MiniWorld. If held prose provenance cannot be established, the guard remains unqualified.

Only test the planned late-life contrast G(RECALL,24) − G(RECALL,2) as the entry condition for Stage B. Require its paired two-sided 95% upper bound below zero. Check answer and nonanswer contributions, early versus late target bands, the analogous BASE and SHAM curves, and the prose guard. An inconclusive contrast is a reportable outcome. Do not pick a different checkpoint because it happened to be the maximum.

This assay tests byte prediction and forward learning transfer across jointly new vocabulary and templates. Every score is pre-feedback at that byte, but T itself supplies data for later learning. Report the first-quarter and frozen-start limitations plainly. It is not a held-out relational state-update result or general E3/E4 evidence. Better compression on familiar syntax, or on nonanswer bytes alone, should be reported at that scope.

## Stage B · Conditional diagnosis, separate authorization

Run only after Stage A confirms the recall contrast and detects a late decline. Preserve Stage A and lock a new roster and cost budget before intervention.

First compare frozen and online target forks at k = 2 and k = 24. Use two target segments. Learn on the first segment in the online fork and score both states on the same later read-only segment. Report predictions before any new feedback and improvement after the same number of new bytes. Include a small predeclared novel-pattern acquisition panel for both experienced checkpoints and a fresh learner. Check for comparable starting headroom and show error relative to a fixed reference. A smaller improvement can reflect an already better starting score. Call it loss of plasticity only when the older state learns the matched new material more slowly under comparable conditions.

If the loss remains, use a **diagnostic state intervention**, not a new policy. From the same late checkpoint compare original state, recall-store reset only, mixer reset only, and both reset. Preserve ordinary byte experts and observations across these branches. Use source-defined birth values for reset components. If the source couples a component with an inseparable tokenizer, IDF or cache, specify and audit that joint reset explicitly. Measure scores immediately after intervention, then learning after equal new bytes. Immediate recovery suggests a stale state or bad weighting; a steeper acquisition curve suggests restored ability to learn. Report interaction and old-world retention on unchanged observations. A reset is an evaluator intervention and does not qualify as autonomous forgetting.

Test at most one deployable intervention afterward, chosen by the diagnosed component. A decay or bounded counter rule is a candidate for a supported accumulation bottleneck. A learned replacement expert needs its own causal state-tracking test. No world-specific hashing, replacement, larger predictor, or prose-in-history arm is part of this replication.

## Counts, resources and three-cycle qualification

Each generated history world and target is capped at 8 KiB of delivered bytes. Establish valid full streams at this cap in qualification. No truncation of a story or substitution of an over-budget world after lock. The upper bounds before technical work and any comparator replays are:

| Component | Charged byte applications |
|---|---:|
| 16 families × 3 arms × 24 history worlds × 8 KiB | 9,437,184 |
| 16 families × 3 arms × 4 target forks × 8 KiB | 1,572,864 |
| 8 held-author excerpts × 3 arms × 16 KiB | 393,216 |
| Total | 11,403,264 |

Pin actual bytes, shared physical input, reference work, archive work and qualification counts before launch. An online byte-count reference can reuse the same external event delivery while its compute and its own byte applications are charged separately. The proposed cumulative cap is **15.5 million charged byte applications**, with **1,800 CPU seconds**, **3,600 active wall seconds**, **1 GiB peak RSS**, and **128 MiB retained outputs**. Profile the exact parent path and stop at admission if any ceiling cannot fit. Save compact per-family, per-arm receipts with raw bit sums, byte census, state hashes and diagnostics. Full per-byte probability arrays need not be retained.

1. **Refine → audit → test the source bridge.** Recover and pin Round 20. Reproduce a bounded historical contrast. Audit causal input, exact bit reductions, vocabulary and template exposure, prefix-invariance, and the sham's information and cost. No qualified parent means no official learner run.
2. **Refine → audit → test the metric and forks.** Use technical streams to verify matched target bytes, checkpoint resume, no writeback to D, and ordinary feedback after every scored byte. Exercise cases with positive G and falling ratio, negative late marginal, tied sham, and a failed prose guard. Test the answer-span reference on a known six-word distribution.
3. **Refine → audit → test the production entry point.** Run a reduced engineering roster through all arms, four checkpoints, and prose scoring. Verify exposure, exact reductions, resource projection, receipt hashes and archive readback. Dry-generate the full roster and lock source and cost. An official launch needs separate authorization.

**Delivery** Design only. Stage B, prose-to-world transfer, and state replacement remain outside the authorized work.
