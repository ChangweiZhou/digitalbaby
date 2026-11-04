# MiniFly V198 · Multiscale memory addresses at matched row traffic

**Status** DESIGN_ONLY. **Date** November 3, 2026, New York time. **Anchors** V195-R1 at ad619f45c96c2be1c3409c310e4e406b6b6ecf41, V196 at 965990387e471ca8714211c19b11bcb5aed56900, and parallel V197 at f932a2361e86ad3b306768ec297c8b9c14b658c4. V198 is the proposed identifier. The branch registry and existing results remain historical records.

## Decision and evidence

Does replacing some short n-gram addresses with four-gram addresses help MiniFly retain corrected facts across intervening learning, at the same storage, teaching dose and number of active rows? A matched rehash control tests whether four-gram context helps beyond simply changing which rows collide.

V195-R1 established a working own-EOS continuation for its one-byte fact task. Its BASE retained 88.21% of all current facts at the end of the life. Scoring the same 1,792 correction probes at the end of their own wave and again at the end of life gives 1,735 → 1,501 exact answers and 50 → 156 obsolete first bytes. These are net changes within fixed cohorts. They do not identify individual losses or the responsible rows. The combined reopening and decay package did not pass, so retain ordinary importance-protected NLMS.

Parallel V197 used twelve presentations and a float32, 256-class fact head with first-byte scoring. In its 1,024-fact importance arm, adding four-grams raised oldest-wave recall from 84.1% to 94.5% and corrected recall from 93.1% to 98.3%, and reduced obsolete first bytes from 3.24% to 0.56%. Similar gains appeared with Alice filler. Its four-gram addresses were added to existing short addresses, so the row table stayed the same size but active-row traffic could grow. V198 tests substitution under a matched budget, using V195-R1's four-presentation schedule and own-EOS scoring. A negative result applies to that lower-dose substitution package.

[Source V195-R1 report](https://github.com/ChangweiZhou/digitalbaby/blob/ad619f45c96c2be1c3409c310e4e406b6b6ecf41/minifly_v195_selective_revision_design_20261103/r1/REPORT.md) · [V197 factorial results](https://github.com/ChangweiZhou/digitalbaby/blob/f932a2361e86ad3b306768ec297c8b9c14b658c4/minifly_v197_long_life_factual_memory/RESULTS.json) · [V197 address source](https://github.com/ChangweiZhou/digitalbaby/blob/f932a2361e86ad3b306768ec297c8b9c14b658c4/minifly_v197_long_life_factual_memory/round7_recovered.py)

## One changed component

Pin V195-R1's float16 65,536 × 257 fact matrix, float32 update scratch, protected NLMS step, row importance rule, initial state, and learned continuation address indexed by the model's own emitted byte. Preserve the inherited fact-query cue, marked teaching and all-byte predictor. The cue indicates when to answer. It carries no key, value, correction or stopping information. Use the same actual boundary targets and greedy 257-class decoder, capped at eight outputs. No candidate uses a canonical key table, future bytes, attention, backpropagation, ELM or arithmetic features.

On the raw causal 16-byte suffix let S be the sorted unique short-address rows from the recovered Round 6 hash. The feature coefficient is 1/sqrt(K) on every active row, where K = |S|. For an empty S, reuse the source's explicit empty-case convention and leave it identical across arms. For every nonempty S, compute a common replacement count q. Set q to the minimum of floor(K/4) and the number of unique four-gram rows outside S. Four-gram proposals use the V197 hash and fan-out. Select rows to drop by a fixed independent keyed ordering of the addresses, with salts frozen before training. Select q four-gram rows outside S by a second fixed ordering. These choices are deterministic from the observed suffix and never use learned weights or outcome labels.

| Arm | Address set |
|---|---|
| SHORT | The unmodified S |
| MULTISCALE | Replace the q selected short rows with q four-gram rows |
| REHASH | Drop the same q short rows. Rehash the observed short grams with independent fixed salts and choose q unique rows outside S, using a third fixed ordering. |

If REHASH cannot supply q distinct replacement rows, reduce q for **both** changed arms to the common feasible count. Record zero-replacement events, and never backfill to only one arm. Lock the salts and the 25% replacement rule as design choices. No search over replacement fractions. All arms read and write K fact rows per eligible training prefix. The continuation path is identical and independent of this address substitution. The scientific claim is about this fixed address allocation, which may change code geometry, row occupancy and realized importance. Equal row traffic does not imply equal weight movement or identical hashing cost. Measure all three.

The extra four-gram enumeration and selection must be bounded by the suffix and fan-out. Count address computations as well as row reads and writes. Precomputation may use raw prefixes and fixed hashes only, with the same cache memory budget and accounted misses for all arms. The learner makes no global search or sort over the full store.

## Fresh paired lifetime and scoring

Use four paired worlds from the first 64 bits of SHA-256 of MINIFLY|V198|WORLD|i for i = 0, 1, 2, 3. Use independent salted RNG streams for keys, values, order, correction identities, wrappers and filler. Preserve the V195-R1 schedule and generator conventions other than the fresh seeds. Lock source, exact inputs and scoring before training.

Each life sees eight waves of 128 distinct facts. Present each fact four times across the same two teaching wrappers. In waves two through eight, interleave four presentations of revised values for 32 designated keys from the preceding wave. Each revised value differs from the original. Use one unmarked prose record per four marked fact records, including mentions of known keys. Preserve valid facts and keep every held wrapper out of training. Verify actual held prefixes and code collision/overlap separately. No held response is written before or after its probe.

Score all learned keys at each wave boundary. Probe both trained wrappers and both held wrapper families with the same provided query cue. Freeze the identity and prompt of each correction cohort so its end-of-own-wave score is directly comparable to its end-of-life score. Track per-probe outcomes and paired flips, not only net differences. Score correct first byte, correct byte followed by own EOS, obsolete first byte, obsolete complete output, premature stop, extra payload and cap separately. Report wave zero untouched, all untouched, each correction cohort, latest corrections and all current labels. The predictor uses the same observed bytes and can be computed once per world because it neither receives nor supplies fact writes. Report actual predictor traffic separately from per-arm scheduled exposures.

For source-localization, record on fixed read-only panels the active-row overlap between different facts and between wrappers of the same fact. At a small prospectively selected set of correction events, record pre/post old and new score margins for the corrected query and still-valid queries that share rows. These probes are evaluator-only and leave state unchanged. Their purpose is to distinguish better separation from merely changing acquisition, not to choose rows or tune the gate.

A trained-prefix table, canonical-key table, and frequency/wrapper-only baseline establish the shortcut envelope. Canonical lookup with the supplied cue can solve the held-wrapper factual task in principle. The strongest claim is bounded E2 changed-presentation recall of already taught facts, including retention and correction. No new relation, arithmetic, autonomous query detection or arbitrary-length output is graded.

## Prespecified decision

Primary endpoint: complete current answer on held-wrapper correction cohorts at the final checkpoint. Use the paired world as the unit. Combine wrappers within world only for the primary mean, and show each wrapper separately. Require MULTISCALE to improve on SHORT by at least five percentage points in the paired mean, consistent with V197's roughly five-point corrected-fact gain. Require positive paired mean effects versus both SHORT and REHASH, with both lower 95% paired t bounds above zero. Report individual paired worlds and intervals. Four worlds support a bounded development decision.

For advancement, also require the following:

- At each cohort's own correction-wave endpoint, MULTISCALE's mean complete recall is within two points of SHORT. The final improvement must be accompanied by a smaller same-cohort decline. This prevents a higher starting score from being mistaken for better persistence.
- The final all-current and oldest-untouched complete recall are each within two points of SHORT. Neither held wrapper may show a negative paired-mean effect on final corrected recall. Obsolete first-byte rate must decrease on the combined corrected panel, with no increase in premature or capped responses.
- Equal active rows on every marked teacher-prefix site, with all full-size resource limits met. Measure per-query free-generation costs as actual, since trajectories may diverge.

If MULTISCALE improves final scores without reducing the same-cohort decline, record an acquisition/addressing benefit and leave the persistence claim open. If it beats SHORT but ties REHASH, a change in collision pattern may suffice and the four-gram claim remains unproven. If it loses only on held wrappers, record the specificity versus invariance trade-off. No tuning after held results. An additive V197-like arm is an engineering reference only if needed for source checks, and its extra row traffic is recorded. It does not enter the matched contrast.

## Budget and three-cycle admission

Each life has 4,096 new-fact presentations, 896 revised-fact presentations and 1,248 filler records. Four worlds × three arms × 6,240 = **74,880 scheduled presentations**, including **59,904 marked fact records**. A limit of 64 payload bytes plus EOS gives at most **4,867,200 scheduled byte/EOS sites**. Charge probes, per-module updates, qualification, cache, serialization and archive work separately. Shared predictor execution is 24,960 physical records if the design reuses that identical computation. Retain the conservative full-arm scheduled census for admission.

Proposed ceilings including technical work: 82,000 scheduled records, 5.4 million target sites, 900 CPU seconds, 1,800 active wall seconds, 1 GiB peak RSS, 48 MiB deployed state per learner, and 256 MiB retained outputs. Preserve compact receipts and per-item outcomes, with state hashes at each probe. If keeping any full states, lock the count and include them in admission. Use the exact roster and measured sparse-read/update cost before admission. A resource failure stops the run rather than reducing exposures or dropping arms.

1. **Refine → audit → test the source bridge.** Reproduce a bounded V195-R1 fixture, check that short-code reads and writes agree, qualify held and trained first bytes and learned EOS. Validate causal suffix and marker timing. Count exact technical presentations first.
2. **Refine → audit → test matching.** Check raw addresses and same-K invariants, including collisions, empty/short histories and unsupported replacement rows. Verify sham-like rehash has no four-gram input. Audit that countermodels cannot access held labels. Test mixed fact/wrapper collision canaries and measure actual row traffic and CPU.
3. **Refine → audit → test the launch path.** Rehearse the production runner through all arms, all eight wave boundaries on a reduced roster, per-item same-cohort reduction, hash-verified receipts and ZIP readback. Bind the rehearsal to launch hashes. Profile and lock the exact four-world roster before any authorized scientific test.

V196's observation-only relation reuse is a separate track. Keep its raw representation and provenance/sham controls as the reference for a later learned routing comparison. V194's biased-expansion hypothesis remains a clean mechanistic test rather than a dependency of this memory experiment.

**Work performed** Design and arithmetic audit only. No implementation, training, qualification, repository publication or adoption.
