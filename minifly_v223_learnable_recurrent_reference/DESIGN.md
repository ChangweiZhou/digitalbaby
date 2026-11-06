# MiniFly design — establish acquisition before transferring credit

**2026-11-05 · Proposed V223 · Design only**

Build a connected recurrent reference on MiniFly's existing byte stream. First establish that it can learn ordinary associations and produce complete answers. Then test whether forward credit can train the same computation from birth and whether the resulting learner can acquire new facts, correct obsolete ones, and retain still-valid information.

V222 remains the comparison record for the isolated re-entry learner. R8 and its longer-credit diagnostic each saw 65,536 additions and 967,507 symbols on one paired development world. Both ended at 0/128 taught and 0/512 held exact answers. Causal matrix updates improved true-prefix EOS prediction, while later digits remained weak. The fourfold response-weight pilot raised causal recall from 0/8 to 1/8 on the repeated fixture. These results justify shifting the main effort from the V222 factorial to an acquisition comparison. They leave connectivity, sensing, and optimization as distinct explanations. [V222 report](https://github.com/ChangweiZhou/digitalbaby/blob/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/minifly_v222_reentry_continuation/REPORT.md)

## 1. Start from an existing recurrent reference

Reuse the October 17 diagnostic's GRU cell and 257-class byte/EOS head. Its width-64 reference acquired 99% of taught single-digit records in the online condition and answered 6% of the held records. That harness used BEGIN and REPLY controls and scored response targets on individual records. Use a short reproduction of its original configuration as a harness check. Establish acquisition again under the continuous protocol below before using the reference to interpret credit failures. [Historical report](https://github.com/ChangweiZhou/digitalbaby/blob/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/addition_difficulty_diagnosis_dev_20261017/REPORT.md) · [Reference implementation](https://github.com/ChangweiZhou/digitalbaby/blob/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/addition_difficulty_diagnosis_dev_20261017/diag.py)

### Continuous byte protocol

Use the V222 generator, record rendering and true EOS. The learner consumes each revealed symbol once. Its state carries over EOS, throughout each phase, and across training blocks. Start independent lives with zero state. At birth predict from the zero-state head, then ingest the revealed symbol. Thereafter predict from the state containing the observed prefix. Use one native softmax over all 257 symbols. Generate answers greedily, feeding each emitted symbol back to the model until its own EOS or the 12-symbol cap.

Use the generic GRU reset, update and candidate gates, with reset applied after the recurrent affine map as in the archived PyTorch cell. Replace the learned embedding with a frozen, birth-seeded 257 × 24 projection for this bridge. Each direct-input symbol has a distinct row. The inherited route updates the existing Full151-derived FE/KC sensor from the same prefix and selects 257 fixed KC coordinates without replacement. Its feature vector is the selected binary activity divided by the square root of its active count, using zero when the selection is empty. Multiply that vector by the **same** projection. Log collisions and empty KC vectors. This comparison tests the declared KC projection against direct identity under a common downstream width, parameter count, and birth. The fixed projection supplies no numerical magnitudes, arithmetic features, addresses, or task identifiers.

Train every arm with the same observable response weighting used in the V222 pilot. Weight symbols after an observed `=` through true EOS by four and the other symbols by one. Rescale each completed 32-symbol block's weights to total 32. All observations advance the forward state and contribute to the loss. The model receives no response flag. Update with Adam at 0.003, betas 0.9 and 0.999, epsilon `1e-8`, and separate gradient norm caps of 1 for the recurrent cell and head. Hold parameters fixed over each block. The response weighting is an explicit protocol difference from V222's uniform main objective. Keep it fixed across the comparisons in this design. [V222 objective pilot](https://github.com/ChangweiZhou/digitalbaby/blob/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/minifly_v222_reentry_continuation/response_weight.py)

## 2. Find a computation that acquires the records

Run three paired development worlds on the existing repeated-record diagnostic. Each contains four arithmetic records and four facts with distinct random keys and values. Use the V222 fixture as the first world and fix two new world/birth seeds before training. Train on seeded reshuffles of the same eight records, with checks after 1, 4, 16, 64, and 256 presentations per record. Match the chronological stream and birth across arms at the same width. Keep exact answers plus EOS as the score and report facts and arithmetic separately.

| Initial BPTT32 reference | Recurrent graph | Input | Purpose |
|---|---|---|---|
| Dense 64 | Dense, 64 units | Direct | Continuously trained port of the historical reference family |
| Connected 32 | Eight incoming recurrent edges per unit, 32 units | Direct | Lower-cost forward-credit candidate |
| Connected 64 | Same in-degree, 64 units | Direct | Tests more recurrent state at the same degree |

For a connected mask, use the self-edge, one incoming edge from a seeded Hamiltonian ring, and six other distinct birth-seeded predecessors per unit. Use the same mask in all gates. Keep the mask fixed and count only active parameters. For an isolated control, partition units into blocks of eight and give each unit the eight predecessors in its block. Both masks have equal in-degree, identical state size, and the same input and output dimensions.

At 256 presentations, require at least **3/4 exact taught facts in each of the three worlds** for a computation to advance. Report arithmetic acquisition at the same checkpoints. Favor the smaller connected width that clears the fact screen. If only Dense 64 clears it, retain that as the candidate forward graph and record the recurrent capacity difference. If all fail, examine the interface and shared optimization before committing resources to credit transfer. A failure here concerns acquisition on this protocol.

Use at most two additional matched comparisons on the selected graph and width.

- **Connectivity.** For a selected sparse connected graph, run the isolated mask with the same degree and parameter count under direct input. Compare exact fact and arithmetic scores, learning curves, and actual CPU. With a dense-only success, the dense/sparse gap leaves parameter count and graph structure unresolved.
- **Input.** Run the selected connected graph with inherited KC input. Compare it against the direct run on the same seeds. The KC route is eligible for the next stage if it clears the same fact floor, matches or exceeds the direct exact fact count in each world, and profiles within the same measured resource envelope. Use direct input otherwise. This small screen selects a route. A formal equivalence claim would need independent data and a prespecified margin.

Select using the fixed endpoint and count all arms. Save results for every dose. The repeated fixture tests whether the learner can acquire and emit the taught sequences on this interface. The next stage measures acquisition under interference and genuinely held behavior.

## 3. Train the selected graph with forward credit

For each symbol transition `h[t+1] = F_theta(h[t], x[t])`, propagate

\[
S_{t+1}=J_tS_t+P_t,\qquad
J_t=\partial F_\theta/\partial h_t,\quad
P_t=\partial F_\theta/\partial\theta.
\]

Use the ordinary head gradient and multiply the head error by the current outgoing weights and the sensitivity of the current state for the recurrent gradient. Treat the revealed symbol and frozen input projection as exogenous. Reset sensitivities at 32-symbol boundaries while keeping the actual recurrent and sensory states. At each site score the target from the current state, then ingest that target and advance its sensitivity. After the 32nd reveal update the parameters once and detach the incoming state for the next block. Preserve the state computed with the preceding block’s weights. The candidate starts from the paired random birth, with no imported BPTT-trained weights.

**Exact forward credit is the first option.** On a matched 32-symbol block, its gradient is mathematically the same as BPTT32. Test that identity on the assembled graph with directional finite differences and independent reverse mode. Include nonzero incoming states, KC or direct input as selected, and mask gradients. Compare complete updates and continued state across multiple blocks. At the same horizon, the question is whether the exact forward implementation delivers the reference computation within a usable resource envelope. A gap between the exact matched methods calls for checking numerical precision, optimizer updates, and state replay.

**A single prespecified approximation is available if exact forward credit exceeds the measured envelope.** Keep the connected GRU's forward graph and project the sensitivity to eight-unit groups. A group tracks derivatives of its states with respect to parameters entering that group's gates. For its sensitivity recurrence use the corresponding diagonal block of the state Jacobian. Each group's forward states continue to depend on the full connected graph. Report the discarded cross-group derivative and alignment against exact gradients on development blocks. This is circuit-level forward credit. The approximation's trainability becomes the comparison against BPTT32 on the same forward graph.

Profile the exact implementation before selecting the credit path. Record a resource-based decision before opening the new worlds. A numerical mismatch halts the affected code path for repair. A cost failure is a resource result. Keep the source, selected graph, sensor, objective and approximation fixed after that decision.

| Width and graph | Trainable cell | Trainable head | Exact sensitivity entries |
|---|---:|---:|---:|
| 32, degree eight | 3,264 | 8,481 | 104,448 |
| 64, degree eight | 6,528 | 16,705 | 417,792 |
| 64, dense | 17,280 | 16,705 | 1,105,920 |

These counts use two GRU bias vectors per gate, an output bias, and a frozen projection. Count that projection's 6,168 scalars and all sensory assets separately. Include optimizer state, sensitivities, recurrent state, BPTT tapes, copied checkpoints, and process memory in measured costs. Report throughput and total CPU by arm. Exact gradient storage alone is a poor cost estimate.

## 4. Measure acquisition and persistence on the same life

Lock three fresh paired development worlds after selection. Run the selected forward-credit learner and its same-graph BPTT32 reference on the existing V222 A, B and A′ generator. Fix the A-dose checkpoints at **4,096, 8,192, 16,384, 32,768, and 65,536 addition records**. Interleave the existing 64 initial facts. Preserve the original order, proportions, commutative-orbit exclusions and checkpoints. All arms receive the same observed records. Score held answers from saved states with no optimizer updates and discard each generation branch.

| Phase | Existing experience | Primary measurement |
|---|---|---|
| A | Addition stream plus 64 recurring facts | Taught fact acquisition, with arithmetic taught/held scores and first-byte, continuation and EOS breakdowns |
| B | 8,192 new additions plus 64 new facts | New fact acquisition and still-valid original fact recall |
| A′ | 4,096 new additions and revised values for 32 old keys | Corrected and obsolete outputs after 1, 4, 16, and 32 correction exposures per key, plus unchanged original facts, retained B facts, and arithmetic retention |

Run all locked A doses. Use the fixed final A checkpoint for decisions. If the reference reaches at least 80% exact initial facts in at least two of the three worlds, continue both arms through B and A′. Otherwise record the full A curve and treat interference and revision as untested under adequate acquisition. Arithmetic has its own development endpoint and does not govern admission to the factual lifetime.

For the candidate's development screen, require the same initial fact floor, at least 70% exact B facts, at least 70% exact corrected values, and at most 20% obsolete outputs at the final endpoint. The B and A′ rate thresholds use equal weight across worlds. Limit mean drops on unchanged A facts and B facts to ten points relative to their last acquired checkpoint. Report both unconditional counts and retention among previously correct items. Show the individual paired worlds. Compare fact acquisition and revision against the BPTT reference at equal observed doses and report CPU alongside scores. Independent confirmation remains a separate step.

Keep held arithmetic separated into same-length unseen orbits, longer operands, and both-correct carry counterfactuals. Use exact answer plus EOS with strong frequency, nearest taught record, and digit-without-carry comparators. Report digit loss, first byte, and EOS under a true prefix as diagnostics. Evaluate the latest-value table as a factual ceiling. Acquisition, continued learning, and held rule reuse each receive their own result.

The main six-life budget is at most **7,360,512 delivered symbols** through A′. The 15 short screening histories use roughly **0.4 million** symbols at the existing fixture length. Add the historical harness check, engineering, checkpoint replays, and generation separately. Set a CPU and memory envelope from measured complete-path throughput with 25% reserve before locking the three fresh worlds. Keep exposure counts, informative fact and consequence targets, random operand bytes, and repeated presentations distinct. This lets a flat learning curve be separated from a short or interrupted run.

## 5. Decisions after the study

| Observation | Decision |
|---|---|
| Connected sparse learns and equally sized isolated sparse does not | Prioritize recurrent communication in this tested graph family. |
| Dense learns and both sparse configurations remain weak | Investigate the tested density and capacity difference before attributing the gain to a single connection pattern. |
| Direct input learns and matched KC input falls short | Keep direct input for the bridge. Preserve the measured sensory gap as a separate result. |
| BPTT32 acquires, and projected forward credit falls short at the same dose | Investigate lost credit through the measured off-group paths. Keep the forward graph fixed for that diagnosis. |
| Forward credit acquires facts and revises them while preserving valid knowledge | Carry this implementation forward as a candidate for independent lifetime confirmation. |
| Both methods acquire, and held arithmetic remains weak | Study reusable computation on the existing held panels and report factual persistence separately. |
| Both methods acquire, and later learning erases valid facts | Test an acquired memory path as an incremental intervention on this learner. Compare continued learning and retention against its own recurrent-only control. |

The first study varies dose and recurrent capacity. Experience diversity remains a subsequent same-learner comparison. Use existing environment families with genuinely different consequence laws and match delivered symbols and informative events. Include held laws and exposure accounting to separate new dynamics from vocabulary changes. Scale pretraining only on a family with measured acquisition and a useful learning curve.

Preserve the V222 source and negative receipts alongside the next study. Keep source locks, splits, raw native predictions and resource receipts sufficient to reproduce the decisions. The reported numerical thresholds are proposed development screens and should be fixed before training.

Sources include the supplied post-V222 review, the original GRU diagnostic, and the published V222 continuation. [V221 generator and interface](https://github.com/ChangweiZhou/digitalbaby/tree/366377ee9d0422abc6ee17e886b7b2c0b4e937a1/minifly_reentry_credit_engineering) · [V222 version and results](https://github.com/ChangweiZhou/digitalbaby/tree/9dfd12e2f7dfedce679e5a0732e9c1ac95478374/minifly_v222_reentry_continuation)
