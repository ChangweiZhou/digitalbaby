# MiniFly V196 · Frozen-readout reuse in a mixed stream

**Status** DESIGN_ONLY. **Date** November 3, 2026, New York time. **Source anchor** Parallel V193 at ca74be1fd03158fd78e7ff0569e1f5c21b42498f. Consolidated evidence through V195-R1 at ad619f45c96c2be1c3409c310e4e406b6b6ecf41. The user requested a design proposal. Implementation and execution are pending.

## Decision

Can MiniFly reuse a previously trained relation readout after a structural change when its successor maps also see conflicting observations? If reuse degrades, can source separation restore it at the same update count as a random selection control?

V193 provides the positive anchor. After 128 clean post-change blocks, its frozen readout and updating maps answered 99.5% of changed held pairs and 100% of still-valid pairs, with no new relation labels. Freezing the maps and training the readout on new labels reached only 4.7% on changed held pairs at that checkpoint. The V193 clean and continued-noise suites used different seeds, so their gap is not a paired mixed-stream effect. V194's normalization failed its promotion gate. V192 tested a different question with revised labels available and stopped before science. V196 keeps the readout frozen and tests whether competing observations interfere with reuse before adding a learned context gate.

[Source and four-arm results for V193](https://github.com/ChangweiZhou/digitalbaby/tree/ca74be1fd03158fd78e7ff0569e1f5c21b42498f/minifly_v193_reusable_relational_operator) · [V194 report](https://github.com/ChangweiZhou/digitalbaby/blob/3eea7ca115b55429a8abddd08054c300c4392858/minifly_v194_contrast_normalized_successor/README.md)

## Source and intervention

Pin V193's direct online stride-one and stride-two TD updates, source precision, 0.65/0.85 discounts, 0.12 learning rate, 2,000-row fixed projection and 100 active rows. Preserve its structured parity query parser, readout and predicted-newline table. Train the readout only before the common fork. After the fork, freeze both readout and newline-table bytes, and supply no relation targets to any arm. Only observed transitions may change the candidate's maps. Check frozen-state hashes at every declared checkpoint. Query probes are read-only and never become observations.

| Arm | Observations after the fork | Deployed map writes |
|---|---|---|
| CLEAN | Same structural record and a silent slot in each block | Structural record only |
| MIX | Structural and competing record | Both records, in chronological order |
| SOURCE | Same mixed stream | Only the structural record, using evaluator-supplied provenance |
| SHAM | Same mixed stream | One of the two records per block, using a prelocked balanced selector |
| FROZEN | Same mixed stream | None |

CLEAN and SOURCE have the same structural records and chronological writes. The source model has no autonomous map decay or byte history across these observation records. Therefore their deployed maps should match exactly. Verify this invariant in the production rehearsal, then represent CLEAN by the same saved trajectory in the full run. They are one reference and do not count as independent results. A CLEAN sentinel audit scores identical queries at identical logical times. If the adapter causes the trajectories to diverge, stop and audit the state boundary.

SOURCE is an evaluator-assisted positive control for separation. It does not demonstrate a learned routing policy. SHAM accepts exactly one 16-byte record per block, giving the same 14 updates per stride as SOURCE. Balance SHAM to select 16 structural and 16 competing records within every 32-block stratum by a fixed independent seeded permutation. It never selects using byte content, held outcomes or target labels. Record both row occupancy and update movement. Equal update count need not imply equal displacement on diverged states. MIX writes twice as often. SOURCE versus SHAM separates placement from simple exposure reduction. All mixed arms see identical bytes and slot times.

## Paired data and lifetime

Use eight fresh synthetic worlds. Derive seeds from the first 64 bits of SHA-256 of MINIFLY|V196|WORLD|i, i = 0 to 7. Use independent substreams for symbol order, label polarity, training pairs, structural records, competing records and SHAM assignments. Preserve V193's 12-symbol setup and one adjacent opposite-parity swap. The swap changes the relation for some pairs and leaves others valid.

Before training, freeze 16 changed and 16 still-valid held ordered pairs per world. Each panel must be balanced between the two answer labels both before and after the swap. Keep each reverse pair in the same split. Neither a held pair, its reverse, nor its re-rendering may receive relation feedback. Lock the exact query schedule and prefixes. Verify label polarity is randomized independently of byte identities. Competing records use the same byte alphabet and the same per-record symbol multiset as the original-cycle record at the paired start index, with order changed by a fixed random permutation. Reject intact cycle windows by a prospective generator rule. Reuse the exact competing bytes in EDIT and NOEDIT. A swapped 16-byte window can change the marginal counts in EDIT. Record that difference instead of resampling the competing record. This is a synthetic interference manipulation. It does not model natural prose.

1. **Common A and B.** Follow the V193 source schedule: 1,200 clean 16-byte observations, 2,048 original relation feedback records, then 512 blocks with one original-cycle observation and one original relation feedback. Retain V193's 10% common B corruption and identical exposure for every arm. Keep the resulting state as one verified fork per world. Report A and B held results. Fresh world generation uses separate random substreams, so V193's historical seeds are engineering references, not exact official V196 trajectories.
2. **EDIT.** From the common fork, present 512 paired observation blocks. Structural records follow the swapped cycle. Freeze the readout and supply zero changed relation labels. Measure held pairs at 0, 8, 32, 128 and 512 blocks.
3. **NOEDIT.** From the same fork, retain the original cycle and original labels for scoring. Keep the same competing records, structural start indices, query times and selector schedule as EDIT. Give no relation feedback. Measure the same panels and checkpoints. This distinguishes harm to previously valid relations from failure to revise after the swap.

## Outcomes and decisions

Score exact first byte on the fixed held query interface. Also score correct byte followed by the source's own bigram-predicted newline, separately. The newline is the historical reference convention. V196 does not add a native EOS head, because changing the readout would obscure the V193 reuse contrast. This run can support bounded frozen-readout reuse within the supplied parity interface. It cannot establish native byte-plus-EOS generation, learned task parsing, arithmetic or core adoption.

Primary comparison uses changed pairs in EDIT at 128 blocks. Use the eight paired worlds for signed effects and two-sided 95% paired t intervals. Present every world and both panels rather than treating queries or forks as independent replications. Use 85% changed accuracy and 90% still-valid accuracy as the historical V193 floors. Use five percentage points as the practical difference, consistent with the prior continual-structure design. Predeclare and retain both byte and byte-plus-newline endpoints, with the latter used to qualify the clean source bridge.

- **Reference.** CLEAN/SOURCE must meet the V193 floors and exceed FROZEN by at least ten points on changed pairs, with the interval above zero. If it fails, report a source/assay mismatch or lack of headroom. Do not infer a failure of context separation.
- **Interference.** Require CLEAN minus MIX of at least five points on mean changed accuracy and an interval above zero. Compare still-valid EDIT and both NOEDIT panels. If the tested mixture does not show a material loss, this assay gives no reason to add a context gate.
- **Recoverability.** Require SOURCE to exceed SHAM by at least five points on mean changed accuracy with the interval above zero. Keep SOURCE within the five-point preservation margin of MIX performance on still-valid EDIT and both NOEDIT panels, and above the 90% floor. Because CLEAN and SOURCE share a trajectory, recovery to the clean level follows by construction. The meaningful empirical contrast is SOURCE versus SHAM and MIX, with the clean invariant checked separately.
- **Next decision.** If the clean reference qualifies, mixed exposure hurts, and provenance selection beats the dose-matched sham, proceed to a separate learned-context experiment. If SOURCE and SHAM tie despite a clear MIX loss, test whether thinning updates is enough before introducing new routing. If the clean reference does not qualify, investigate representation, source parity or the assay.

Include constant/frequency, operand marginals, trained exact/reverse lookup and an observed-transition graph baseline. Fit the graph's label polarity from pre-fork feedback and freeze it after B. Show graph estimates from all observed transitions and the provenance-filtered subset. Keep these evaluator-only. For the primary causal claim FROZEN supplies identical mixed observations and identical query timing without map writes. No candidate sees evaluator ground-truth graph edges, order or query labels.

## Physical budget and launch checks

Common A/B costs 4,272 presentations per world, including observation and query feedback records. Four actual continuation trajectories per world, two forks each, and 1,024 observation records per fork give **99,712 physical records** across eight worlds. The CLEAN/SOURCE equivalence checks in the technical rehearsal are charged as technical work, not scientific duplication. Common exposure is counted once per world. With 17 sites per observation record and nine per teaching record the scientific ceiling is **1,531,264 byte/boundary sites**. Held probes and evaluator work are separate costs. Report eligible and accepted transitions by stride, actual write counts and measured runtime, memory and output sizes.

Proposed limits include technical work: 115,000 physical records, 1.8 million byte/boundary sites, 7,200 CPU seconds, 10,800 active wall seconds, 1 GiB RSS, 8 MiB deployed state and full checkpoint, and 256 MiB retained outputs. Store one complete common fork per world, including RNG and fixed arrays or a verified regeneration contract. Record C scores, state hashes and ledgers. C states are not resume points. Profile both strides, forks, all retained files and packaging before admission. Do not treat the prior V192 allowance as transferable.

If authorized, use three successive refine → audit → test cycles:

1. Qualify the pinned clean reference and its newline scorer on an engineering roster separate from science. Use a bounded source prefix or saved checkpoint where possible. Count the exact technical exposure before invoking any full historical runner.
2. Audit same-byte pairing, held reversal closure, matched SHAM write counts, frozen readout and EOS checks, no post-fork labels, and CLEAN/SOURCE state equality. Test intentionally invalid masks and checkpoint corruption.
3. Rehearse the actual production entry point through complete forks, paired reduction, receipts and ZIP readback. Bind its source hashes and compute the exact full-roster census before launch. If admission fails, preserve the technical receipts and leave the science unrun.

**Work performed** Design only. No model implementation, training, promotion or version-registry update.
