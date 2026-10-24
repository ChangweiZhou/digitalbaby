# V53 — Cue formation and write interference

## Complete the interaction panel, then test one new operation

**Namespace:** `V53-CUE-FORMATION`
**Date:** 12 September 2026
**Frozen substrate:** V50/V51 core, unchanged
**Predecessors:** V52 (two independent implementations), V51 frozen-core scope

**V53 answers one question: can the existing bounded associative core learn from the
complete byte stream once its sensory cue is stable and identity-preserving? It does
not search for another memory mechanism, and it does not adopt neurotransmitter signs.**

---

## 1. Corrections carried into this design

Four apply to my own V52 implementation and are accepted without qualification.

**1.1 My shuffled-phase control was confounded.** `PHASE-Z-SHUFFLED` assigned a
pseudo-random phase per byte index, so the same context received different phases at
different times: recurring-context self-overlap ≈ 0.819, not 1.0. It therefore varied
alignment *and* stationarity together and cannot isolate either. V53 replaces it with
`CONTEXT-BIN-FROZEN`: statistics conditioned on `hash(context) mod P`. Same machinery,
same bin count, stationary by construction (a context always lands in one bin), and no
correspondence to record position. That is the control that isolates alignment.

**1.2 My fixed-nontarget control proves less than I claimed.** It writes at a constant
prefix position and never teaches the association, so its failure shows only that
regularly writing a framing symbol does not create the association. It does **not**
establish that a learned relevance detector is needed. Retained with that scope.

**1.3 My identity retrieval used a 32-identity gallery, not 128.** The ~95% figure is a
32-way result from cohort A only and must not be read as full-load identity decoding.
V53 scores retrieval against the complete gallery at each load.

**1.4 My `INVARIANT-TAIL4` arm is wrong under all-byte writing.** The mask retains fixed
context positions at *every* byte event, but those positions are the invariant key tail
only when the context is target-aligned. A naive `TAIL4 × ALL` would conflate harmful
writing with an artificially damaged non-target representation. V53 uses the corrected
form in §4.

Two further scope corrections apply to the anatomy:

**1.5 "Compartment-resolved" means morphological parts** — axon, dendrite and related
categories — **not** the named mushroom-body learning compartments. The split table
carries no synapse coordinates and cannot reconstruct a dopamine-controlled compartment.

**1.6 Transmitter predictions are evidence, not sign assignment.** The curated
annotation for right APL is GABA; the raw classifier returns acetylcholine at 0.5825.
Taking the argmax would overwrite a curated inhibitory annotation with a low-confidence
prediction. Disagreements also exist for some PNs, KCs, MBONs and DANs. These are
annotation disagreements, not a measured error rate.

---

## 2. The methodological gap V53 must close

Both V52 implementations used target-only performance as a gate on whether to run the
all-byte condition. That was a compute heuristic, **not a logical entailment**.

> An interface can produce a modest target-only gain and still materially reduce
> interference between byte positions under all-byte learning. The converse is also
> possible.

So the existing results do **not** establish that every stationary phase interface fails
under all-byte learning. Those cells were never run. V53 runs the full panel.

---

## 3. New evidence: pair discovery is easy at the byte level, hard in KC space

The leading V53 idea — learn which sensory features stay reliable across alternate views
— depends on proposing same-identity pairs without labels. That was identified as the
difficult part. It was measured before designing around it.

All causal target contexts at training wrappers 0/1, seed 52501, load 128: 2,016 pairs,
32 same-identity (1.6% base rate).

| proposal signal | AUC | precision @ 100% recall |
|---|---:|---:|
| **raw 12-byte position match** | **0.9960** | **0.821** |
| KC Jaccard, `V51-LAG` | 0.9842 | 0.092 |
| KC Jaccard, `PHASE-Z-FROZEN` | 0.9594 | 0.125 |

A trivial position-match comparator captures every same-identity pair while admitting
seven false ones. The KC expansion destroys the information a proposer needs.

Two consequences for the design:

1. **The reliability learner operates on the raw rolling context, upstream of the PN
   adaptor** — not on KC codes.
2. **Proposal is by sensory similarity, never by shared outcome.** This sidesteps the
   merging failure directly: in this fixture many identities share one of 24 target
   letters, and a learner that merged everything currently predicting `a` would look
   healthy until one of them is revised. A similarity proposer never consults outcomes,
   so that failure mode cannot arise by construction. It is still gated in §6.

---

## 4. Block A — the missing interaction panel

Load 128, fresh seeds `53501…53516`. Both V52 runs consumed `52501…52516`; reusing them
would not be an independent replication.

Every interface runs in **both** write conditions. Target-only is a diagnostic, not a gate.

| interface | deployable | target-only | all-byte |
|---|:-:|:-:|:-:|
| `V51-LAG` | yes | ✓ | ✓ |
| `PHASE-MASK-FROZEN` | yes | ✓ | ✓ |
| `PHASE-CENTER-FROZEN` | yes | ✓ | ✓ |
| `PHASE-Z-FROZEN` | yes | ✓ | ✓ |
| `CONTEXT-BIN-FROZEN` (§1.1 control) | no | ✓ | ✓ |
| `XOR-OFF` | no | ✓ | ✓ |
| `PHASE-Z + XOR-OFF` | no | ✓ | ✓ |
| `INVARIANT-CUE` (below) | no | ✓ | ✓ |

### 4.1 The corrected invariant-cue oracle

> **At target events, substitute the invariant-tail code. At every other event, keep the
> original sensory code exactly unchanged.**

Explicitly evaluator-assisted and nondeployable. It asks whether improving the
associative cue alone suffices while leaving the rest of the stream intact.

- **Works under ALL** → cue quality explains much of the apparent write-allocation
  problem in this fixture.
- **Still fails under ALL** → improving the target cue alone is insufficient and
  interference from other writes matters independently.

Neither outcome generalises beyond this fixture.

### 4.2 Endpoints

Reported separately, never collapsed into a single headline: old-A held-out (true
untouched cohort — see §8.3), new-D held-out, acquisition loss and exposures-to-criterion,
one-off noise, revision at 1/2/4/8, revised-answer survival, framing/key/target byte-class
loss and accuracy, whole-stream bpb against the Markov2 baseline, synapse reset and
shuffle localization, state and sample budget. The joint score
`U = min(old-A held-out, new-D held-out)` is reported **alongside** its components.

---

## 5. Block B — measured write damage

Jaccard overlap ignores activity signs, amplitudes, which MBON receives the connection,
and the current prediction error. Two equally overlapping KC codes can produce very
different interference. V53 measures damage instead of inferring it.

Sampled diagnostic: clone the learned state, apply **one** actual write, measure the
change in selected old-memory predictions. Stratify by:

- **write class** — framing, key, target;
- **relationship to the probed memory** — same identity, different identity with the
  same outcome, different identity with a conflicting outcome.

**Required reference arm (addition).** A write of matched L2 norm applied to a random KC
code of matched sparsity. Without it, "framing writes damage memories" is trivially true
of any write; the reference is what makes class-specific damage a claim.

This is read-only diagnostic accounting: the cloned state is discarded and never
influences the learner.

---

## 6. Block C — feature reliability upstream of the PN adaptor

One new operation only. It runs **after** Block A, and only on the interface that Block A
selects as the best deployable reference.

### 6.1 Mechanism

A bounded buffer of recent raw contexts. A pair is proposed when raw position-match
exceeds a fixed threshold, using only previously observed sensory information, before the
current outcome arrives. No identity labels, no wrapper labels, no byte-role labels, no
outcomes enter proposal.

Per context coordinate, maintain two bounded statistics:

- variation **across** observations (is this coordinate informative, or constant boilerplate?);
- variation **within** proposed pairs (does it change when only the surface presentation changes?).

A coordinate is down-weighted when it varies within pairs relative to how much it varies
across observations. A constant prefix is stable but uninformative; a wrapper-dependent
character is variable but unreliable; a coordinate that survives alternate views and still
distinguishes observations is the useful case.

The interface is developed on **separate development identities**, then **frozen** before
the persistence experiment. Development examples and transient state are charged in the
sample budget. This establishes *reusable learned cue formation*, not continual
representation adaptation.

### 6.2 Controls

- `SHUFFLED-PAIR` — matched pair count, partners drawn at random. Destroys the
  correspondence while preserving the amount of adaptation.
- `MATCHED-SUPPRESSION` — suppress the same number of coordinates, chosen at random.
- `RAW` — no suppression.
- `PAIR-C` — the historical Candidate-C / V45 construction as a comparator, not as the
  novelty. That work produced a real representation effect with increased harmful
  sharing; V47's attempt to protect distinguishing directions did not beat its shuffled
  control. V53 does not relabel it as new.

### 6.3 The decisive gate

> Train several distinct identities that share one outcome. Revise one. The unrevised
> identities must retain their original outcomes.

Predeclared: if revising one member moves the unrevised same-outcome identities by more
than **10 percentage points** relative to the `RAW` interface, the reliability learner is
merging distinct objects and is rejected regardless of its headline accuracy. This is a
sharper test than showing that nearby representations predict the same byte.

---

## 7. Explicitly deferred or excluded

**Two-state sensory/teaching cell → V54, contingent.** The new split table does support a
distinction between sensory input and teaching/output modulation: PN→KC is 89.7% on KC
dendrites while DAN→KC is 60.7% on KC axons, and MBON→DAN is 73.3% on DAN dendrites,
weighted by recorded counts. That makes the abstraction more defensible than before. But
it is motivated by anatomy, whereas the V52 oracle ladder localized the missing operation
to invariance discovery. Two new operations in one round would confound them. If V54 runs
it, it must retain the one-state core as reference, compare location-specific against
whole-cell modulation, include a control that scrambles location assignments while
preserving the amount of modulation, and must not claim that dendrites are fast or axons
slow — the table records where contacts lie, not dynamics.

**Spatial APL model — excluded.** 94.7% of the APL side of its outgoing KC contacts is
labelled unknown. The table cannot resolve APL's internal branches, so a local-APL
implementation is premature. The V48 scalar result remains negative for that abstraction
and neither proves nor disproves richer local physiology.

**Automatic transmitter signs — excluded.** Transmitter predictions enter as evidence and
uncertainty only. The APL GABA/acetylcholine disagreement is recorded, not resolved by
argmax. Joins use snapshot identifiers, with the 1,203 redundant annotation rows
deduplicated rather than summed — otherwise duplicate inflation would appear as a finding.

**Contextual-information write gating — conditional.** Only if Block A shows that a good
cue still fails under all-byte learning. If run, the signal is whether the longer context
carries predictive information beyond a short-context predictor — both predictions made
before the byte is seen — rather than raw surprise, since a random character is highly
surprising and worth nothing, while a learned association is unsurprising and may still
need reinforcement. Such a mechanism needs a bootstrap (durable learning must be able to
start before the long-context predictor is good), matched rate and update-norm controls,
and must preserve revision. V47's one-byte surprise filter was a filter on negative
representation teaching, not a write-allocation test, and should not be reported as one.

---

## 8. Validity and implementation

**8.1 Graph reconciliation.** After deduplication the split table reproduces the existing
22,417 PN→KC and 15,523 KC→MBON pairs. V53 enriches the existing core; it does not
silently substitute a different graph. The runner embeds the small relevant arrays with
provenance hashes; the large table is processed once.

**8.2 Fresh seeds.** `53501…53516` for screen, `53601…53616` reserved for any science
stage. Engineering seeds are disjoint and never enter summaries.

**8.3 Probe-cohort correction.** V51's byte world sets `revised` to the first quarter and
`noisy` to the second, and its A-retention probe used the first *half* — exactly the
identities it had damaged. All V53 old-A probes use the true untouched complement.
Legacy first-half probes may be emitted separately for comparison, clearly labelled.
V51's CUE path uses disjoint slices and is unaffected.

**8.4 Statistics.** Seeds are the inferential unit. Identities, byte events, probes and
exposures are not independent replicates. Primary contrasts are paired by seed with
Student and bootstrap intervals and raw seed values. Right-censored
exposures-to-criterion stay censored.

**8.5 Compute.** Sparse, load 128, two workers for full-stream jobs, bounded diagnostic
sampling rather than retained pairwise matrices. Load 256 is a robustness test after a
meaningful result, not an axis to sweep first.

---

## 9. Decision structure

| outcome | reading | next |
|---|---|---|
| A phase interface improves all-byte `U` materially with lower bound above zero | The target-only gate was hiding a real interface gain | Freeze it; run load 256 |
| `INVARIANT-CUE × ALL` works, deployable interfaces do not | Cue quality is the dominant term; the missing operation is invariance discovery | Block C is the right target |
| `INVARIANT-CUE × ALL` fails | A good target cue is insufficient; other writes interfere independently | Write allocation becomes the target, with §7's conditional design |
| Block C beats `SHUFFLED-PAIR` and `MATCHED-SUPPRESSION` and passes §6.3 | Reusable learned cue formation established for this fixture | V54: sensory/teaching separation |
| Block C fails §6.3 | The learner merges distinct objects that share an outcome | Reject; do not tune the threshold after seeing this |

---

## 10. Non-claims

V53 cannot establish that phase is a biological variable, that the serialization is
natural, that the invariant positions are known to the learner, that transmitter
predictions supply a learning rule, that the split table locates learning compartments,
that success at load 128 establishes scaling, or that byte serialization is the only
remaining obstacle. It tests whether a bounded core with a stable, identity-preserving
cue can learn from a complete byte stream, and measures which writes damage which
memories while doing so.
