# MiniFly V206 · Causal context retention for entity–value binding

**Status** DESIGN_ONLY · **Date** November 4, 2026  
**Lineage** V204 Delayed Write Selection. V201 action–consequence confirmation is a separate track.

## Decision

When neutral text pushes the subject out of the recent address window, can a bounded state carrying earlier bytes recover the correct entity–value association? Hold write selection fixed throughout. A result should tell us whether to work on context retention before returning to learned write selection.

In V204 the correct-position control reached 99.6–99.9% on the missing- and misleading-cue conditions, but only 16.5% when another location preceded the final destination. The source uses a 32-byte address window. The lengthened construction can move the subject outside it. This motivates a causal intervention on the address. The multi-event construction itself also changes the meaning and placement of the writes, so its result alone cannot quantify the effect of subject loss. V204's original verdict stands.

[V204 report](https://github.com/ChangweiZhou/digitalbaby/blob/64d99849b80b4c572b24532682d35f1f65c1a811/minifly_v204_delayed_write_selection/README.md) · [V204 source](https://github.com/ChangweiZhou/digitalbaby/blob/64d99849b80b4c572b24532682d35f1f65c1a811/minifly_v204_delayed_write_selection/qualify.py)

## Fix the write opportunity

Use the frozen V204 correct-position path as the starting reference. Before any new training or evaluation, pin its exact transitive source, checkpoint, answer protocol, row budget, and commit behavior. Reproduce one published engineering batch and verify the original input-to-write trace. Build the prospective arms from the same checkpoint. Hold all trainable components fixed and reset episodic state to the same starting state for each story. Only the episodic store writes during evaluation.

A diagnostic harness calls the existing episodic write port at the designated destination position. Every arm gets the same calls, at the same points in the raw stream. The flag is privileged information about **when** to write. It carries no entity, address, destination bytes, or future continuation. Confirm that each call is delivered after the required observed bytes are available and before any later clause or query. If the source port commits multiple bytes, each written value must already have arrived. Lock this chronology in qualification. Score model-emitted complete payload and its own newline with the inherited cap. The output port remains read-only. The experiment makes no learned-gate claim.

## Three contrasts

Use the same episodic row count, dtype, payload writer, overwrite order, readout and output cap. Preserve the original step index used for multi-byte answers. Route through each variant during both writes and reads. Read-side context ends at the query boundary, before any emitted answer byte.

| Arm | Available bytes for the episodic address | Role |
|---|---|---|
| RECENT32 | Original trailing 32 bytes and source address recipe | Historical baseline |
| RECENT96 | Trailing 96 bytes through the same recipe, with the same active-row ceiling | Diagnostic for recoverable earlier input |
| SPLIT-RECENT | Two disjoint row namespaces, both encoded from trailing 32 bytes | Matched control for splitting the address and reallocating rows |
| SPLIT-ANCHOR | Same namespaces, with one encoding trailing 32 bytes and the other encoding the first 16 observed bytes of the current record | Tests retention of earlier raw bytes |

A record starts at the observable stream start or after an actual newline. The anchor fills causally and then stays fixed through the write or query. It resets at the next observable boundary. Use the same rule for all records, including questions. The anchor receives no extracted name, role flag, or parsed fact. Retain the source's feature recipes within each namespace. Use a frozen hash salt and deterministic row ordering where the split or long-window arms require remapping or truncation. Split arms reserve one third of the inherited active-row ceiling, rounded down, for the second namespace and the balance for the recent namespace. Apply each namespace's quota after deduplication, with no backfill across namespaces. Freeze the exact source-derived ceiling, salts, cast order and row allocator before exposing outcomes. Report realized unique rows, collisions and read/write traffic separately. This comparison matches the ceiling and storage. The long-window diagnostic can still spend more on constructing features.

The split contrast tests the availability of earlier bytes with the same two-lane layout. RECENT32 versus RECENT96 also changes which local features compete for the row budget. A negative RECENT96 result therefore cannot by itself rule out an information-loss mechanism. On a read-only panel, trace bytes entering each lane, the support overlap between each subject's write and query, and cross-subject overlap. Verify that the anchor actually sees both subjects' identity bytes in the declared source renderings at their respective write and query boundaries. This audit uses entity metadata only to analyze traces.

## Paired stories and exposure

Use eight independent generated worlds and 32 stories per world. Each story draws two distinct people and four distinct destinations from the frozen source's supported vocabulary. Randomize person, destination, order and padding streams independently. Teach the two initial locations, then change each person's location once. Keep both subjects and all current values distinct within a story. Interleave the two subjects and balance which is written last. Ask both questions before and after the changes. Probe from the same episodic state without letting questions or generated answers enter the write stream.

Render each world in four paired conditions. Insert **0, 16, 40 or 64 raw bytes** of neutral material between the subject and the original destination phrase. Preserve original actor, verb, destination, their order and the answer protocol. Padding contains no person, location, negation, extra event, or newline. Use predeclared bytes from a frozen generator and counterbalance the padding choices across subjects. Use the same raw story in all four arms at a given length. Log the exact position of the subject relative to the write and report whether it is absent, partly visible, or fully visible in each address view. Qualification must verify a nonempty near/far contrast where 32 bytes lose the subject and 96 retain it. If the source tokenizer or byte history invalidates that contrast, revise before the lock.

The four lengths are matched counterfactual renderings of already specified events. The primary comparison uses the 40- and 64-byte renderings after both changes. Initial queries, the shorter conditions, previous-value errors, cross-person errors, and failure to emit a complete answer are secondary. The **design intentionally excludes sequential clauses and negation from the primary roster**. They confound retention with interpretation. In a subsequent sequential-events assay, score the current state after each observable completed event and allow a provisional true value to be replaced. Any single-final-commit assay should give an observable boundary first.

## Scoring and decision

Primary score is exact own-output destination and own newline at the final checkpoint, averaged over both subjects, both long lengths and stories within each world. The independent unit is the world. Report the paired two-sided 95% Student-t interval for SPLIT-ANCHOR minus SPLIT-RECENT, the corresponding contrast against RECENT32, and individual world differences. Report RECENT96 as a diagnostic.

Advance the retained-state architecture to a later gate-selection study only if:

- SPLIT-ANCHOR improves on SPLIT-RECENT by at least 10 percentage points on the primary endpoint, and the paired lower confidence bound exceeds zero.
- SPLIT-ANCHOR also improves on RECENT32 on that endpoint, with paired lower bound above zero. At each long length, its mean is at least 75%. At zero padding, it loses at most five points against RECENT32, which must itself reach at least 90% on the final questions.
- Both subjects benefit in the paired mean, cross-person errors decrease versus SPLIT-RECENT, and exposure, write timing, source, and resource audits pass.

The thresholds are prospective development criteria. An improvement over RECENT32 alone does not isolate the anchor. If RECENT96 recovers the correct binding and SPLIT-ANCHOR does not, the tested state allocation is insufficient. If the anchor is byte-accessible but neither intervention rescues behavior, inspect write–query address compatibility before changing the gate. A failed zero-padding reference makes the fresh comparison unqualified. Preserve and report all outcomes.

Run evaluator-only comparators. An exact raw query table has no current locations without observing and associating the new event values. A canonical per-person latest-location table receives only the causally completed event ledger and should solve the assay. A latest-location-without-subject baseline exposes one-entity and recency shortcuts. Report all three. This is bounded **E2 changed-presentation binding of observed facts**, under supplied write timing. It does not establish semantic interpretation, unseen relations, or autonomous write selection. Final accuracy and verified write behavior are distinct outputs.

## Budget and launch qualification

Eight worlds × 32 stories × four renderings × four arms gives **4,096 charged story-arm applications**, **16,384 marked destination write opportunities**, and **16,384 read-only answer generations**. A rendered story is generated once per world, story and length, then replayed through all four arms. Count raw bytes, actual bytes committed, row touches, emitted bytes and any reused computation. Charge any historical reproduction or checkpoint preparation separately from these counts. Before launch, lock the source-derived output cap, exact byte census, checkpoint provenance, and CPU/RSS projection on the measured path. Proposed cumulative limits are 5,000 story-arm applications including qualification, 20,000 write opportunities and answer generations each, 1,200 CPU seconds, 2,400 active wall seconds, 1 GiB peak RSS, and 128 MiB retained output. Do not bypass a failed resource gate by dropping arms or renderings.

1. **Refine → audit → test the source bridge.** Reproduce the V204 position control on a bounded historical fixture. Verify exact output, causal byte availability, frozen-state parity and the exposed write port.
2. **Refine → audit → test address retention.** Use two-person canaries at every length, matched write times, causal anchor reset, and a no-entity overlap probe. Audit namespace capacity and compute.
3. **Refine → audit → test the full runner.** Use separate engineering worlds through all arms and checkpoints. Verify read-only scoring, positive and negative reducer cases, source and roster hashes, receipt commits, and bundle readback. Dry-generate the science roster and lock it before an authorized run.

**Delivery** Design only. No implementation or experiment has run.
